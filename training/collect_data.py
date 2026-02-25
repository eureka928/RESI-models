"""
Fetch recently sold US residential properties from Zillow via RapidAPI.

Uses the "Real-Time Zillow Data" API by OpenWeb Ninja on RapidAPI.
This is the most popular and actively maintained Zillow API (replaces deprecated zillow-com1).

Two endpoints are used:
  1. GET /propertyExtendedSearch  — search recently sold properties by location
     Params: location, status_type=RecentlySold, home_type, page
     Returns: list of property summaries with zpid
  2. GET /property                — full property details by zpid
     Params: zpid
     Returns: 259+ fields including resoFacts, nearbySchools, priceHistory

Setup:
  1. Go to https://rapidapi.com/letscrape-6bRBa3QguO5/api/real-time-zillow-data
  2. Subscribe (Free=100 req/mo, Pro=$25/10K, Ultra=$75/50K, Mega=$150/200K)
  3. Copy your API key from the RapidAPI dashboard
  4. export RAPIDAPI_KEY=your_key_here

Usage:
    python collect_data.py --rapidapi-key YOUR_KEY --num-properties 50000
    python collect_data.py  # uses RAPIDAPI_KEY env var

Saves raw JSON responses to training/raw_data/{zpid}.json for reprocessing.
Supports resume — already-fetched zpids are skipped.
"""

import argparse
import asyncio
import json
import logging
import time
from pathlib import Path

import httpx

from config import RAPIDAPI_HOST, RAPIDAPI_KEY, TARGET_MARKETS

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
)
logger = logging.getLogger(__name__)

RAW_DATA_DIR = Path(__file__).parent / "raw_data"

# Rate limiting defaults
DEFAULT_DELAY = 0.5  # seconds between requests
MAX_CONCURRENT = 3   # concurrent requests to API


def get_headers(api_key: str, host: str = RAPIDAPI_HOST) -> dict:
    return {
        "x-rapidapi-key": api_key,
        "x-rapidapi-host": host,
    }


async def search_sold_properties(
    client: httpx.AsyncClient,
    headers: dict,
    location: str,
    page: int = 1,
    delay: float = DEFAULT_DELAY,
) -> list[dict]:
    """
    Search for recently sold properties in a location.

    Uses GET /propertyExtendedSearch with status_type=RecentlySold.
    Returns list of property summary dicts containing zpid.
    """
    host = headers["x-rapidapi-host"]
    url = f"https://{host}/propertyExtendedSearch"
    params = {
        "location": location,
        "status_type": "RecentlySold",
        "page": str(page),
    }

    try:
        resp = await client.get(url, headers=headers, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        await asyncio.sleep(delay)

        # The zillow-com1 API returns {"props": [...]} for search results
        results = []
        if isinstance(data, dict):
            # Primary response format: data["props"]
            props = data.get("props", [])
            if isinstance(props, list) and props:
                results = props
            # Fallback: some versions use data["results"]
            elif "results" in data:
                fallback = data["results"]
                if isinstance(fallback, list):
                    results = fallback
            # Fallback: searchResults.listResults
            elif "searchResults" in data:
                sr = data.get("searchResults", {})
                if isinstance(sr, dict):
                    list_results = sr.get("listResults", [])
                    if list_results:
                        results = list_results

        logger.debug(f"  Search returned {len(results)} results")
        return results

    except httpx.HTTPStatusError as e:
        logger.warning(f"Search failed for {location} page {page}: HTTP {e.response.status_code}")
        if e.response.status_code == 429:
            logger.warning("Rate limited! Waiting 60s...")
            await asyncio.sleep(60)
        elif e.response.status_code == 403:
            logger.error("403 Forbidden — check your API key and subscription")
        return []
    except Exception as e:
        logger.warning(f"Search error for {location}: {e}")
        return []


async def fetch_property_detail(
    client: httpx.AsyncClient,
    headers: dict,
    zpid: str,
    semaphore: asyncio.Semaphore,
    delay: float,
) -> dict | None:
    """
    Fetch full property details for a zpid.

    Uses GET /property with zpid parameter.
    Returns the full property detail JSON (259+ fields) including
    resoFacts, nearbySchools, priceHistory, homeType, etc.
    """
    async with semaphore:
        host = headers["x-rapidapi-host"]
        url = f"https://{host}/property"
        params = {"zpid": zpid}

        try:
            resp = await client.get(
                url, headers=headers, params=params, timeout=30
            )
            resp.raise_for_status()
            data = resp.json()
            await asyncio.sleep(delay)
            return data
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:
                logger.warning(f"Rate limited on zpid {zpid}, waiting 30s...")
                await asyncio.sleep(30)
            else:
                logger.warning(f"Detail fetch failed for zpid {zpid}: HTTP {e.response.status_code}")
            await asyncio.sleep(delay)
            return None
        except Exception as e:
            logger.warning(f"Detail error for zpid {zpid}: {e}")
            await asyncio.sleep(delay)
            return None


def get_existing_zpids(data_dir: Path) -> set[str]:
    """Get set of already-fetched zpids for resume support."""
    existing = set()
    if data_dir.exists():
        for f in data_dir.glob("*.json"):
            existing.add(f.stem)
    return existing


def extract_zpid(result: dict) -> str | None:
    """Extract zpid from a search result (handles different response formats)."""
    zpid = result.get("zpid")
    if zpid:
        return str(zpid)
    # Fallback: extract from detailUrl (e.g., "/homedetails/.../12345_zpid/")
    detail_url = result.get("detailUrl", "")
    if "_zpid" in detail_url:
        try:
            # URL format: /homedetails/address/12345_zpid/
            segment = detail_url.split("_zpid")[0]
            return segment.split("/")[-1]
        except (IndexError, ValueError):
            pass
    return None


async def collect_for_market(
    client: httpx.AsyncClient,
    headers: dict,
    location: str,
    existing_zpids: set[str],
    semaphore: asyncio.Semaphore,
    delay: float,
    data_dir: Path,
    max_pages: int = 10,
) -> int:
    """Collect sold properties for a single market."""
    collected = 0

    for page in range(1, max_pages + 1):
        logger.info(f"  Searching {location} page {page}...")
        results = await search_sold_properties(
            client, headers, location, page, delay
        )

        if not results:
            logger.info(f"  No more results for {location} at page {page}")
            break

        zpids_to_fetch = []
        for result in results:
            zpid = extract_zpid(result)
            if zpid and zpid not in existing_zpids:
                zpids_to_fetch.append(zpid)
                existing_zpids.add(zpid)

        if not zpids_to_fetch:
            logger.info(f"  All zpids already fetched for {location} page {page}")
            continue

        logger.info(
            f"  Fetching {len(zpids_to_fetch)} property details for {location}..."
        )

        tasks = [
            fetch_property_detail(client, headers, zpid, semaphore, delay)
            for zpid in zpids_to_fetch
        ]
        details = await asyncio.gather(*tasks)

        for zpid, detail in zip(zpids_to_fetch, details):
            if detail:
                out_path = data_dir / f"{zpid}.json"
                with open(out_path, "w") as f:
                    json.dump(detail, f)
                collected += 1

        logger.info(
            f"  Saved {collected} properties from {location} so far"
        )
        await asyncio.sleep(delay)

    return collected


async def collect_all(
    api_key: str,
    num_properties: int = 50000,
    delay: float = DEFAULT_DELAY,
    max_concurrent: int = MAX_CONCURRENT,
    max_pages_per_market: int = 20,
) -> int:
    """
    Collect properties across all target markets.

    Returns total number of newly collected properties.
    """
    data_dir = RAW_DATA_DIR
    data_dir.mkdir(parents=True, exist_ok=True)

    existing_zpids = get_existing_zpids(data_dir)
    logger.info(f"Found {len(existing_zpids)} existing properties (will skip)")

    if len(existing_zpids) >= num_properties:
        logger.info(
            f"Already have {len(existing_zpids)} >= {num_properties} target. Done."
        )
        return 0

    headers = get_headers(api_key)
    semaphore = asyncio.Semaphore(max_concurrent)
    total_collected = 0

    async with httpx.AsyncClient() as client:
        for location in TARGET_MARKETS:
            if len(existing_zpids) >= num_properties:
                logger.info(f"Reached target of {num_properties} properties")
                break

            logger.info(
                f"Collecting from {location} "
                f"({len(existing_zpids)}/{num_properties})..."
            )
            count = await collect_for_market(
                client,
                headers,
                location,
                existing_zpids,
                semaphore,
                delay,
                data_dir,
                max_pages=max_pages_per_market,
            )
            total_collected += count

    logger.info(
        f"Collection complete: {total_collected} new properties, "
        f"{len(existing_zpids)} total"
    )
    return total_collected


def main():
    parser = argparse.ArgumentParser(
        description="Collect recently sold property data from Zillow via RapidAPI"
    )
    parser.add_argument(
        "--rapidapi-key",
        default=RAPIDAPI_KEY,
        help="RapidAPI key (or set RAPIDAPI_KEY env var)",
    )
    parser.add_argument(
        "--num-properties",
        type=int,
        default=50000,
        help="Target number of properties to collect",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=DEFAULT_DELAY,
        help="Delay between API requests in seconds",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=MAX_CONCURRENT,
        help="Maximum concurrent API requests",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=20,
        help="Maximum search result pages per market",
    )
    args = parser.parse_args()

    if not args.rapidapi_key:
        logger.error(
            "No API key provided. Set RAPIDAPI_KEY env var or use --rapidapi-key"
        )
        return

    asyncio.run(
        collect_all(
            api_key=args.rapidapi_key,
            num_properties=args.num_properties,
            delay=args.delay,
            max_concurrent=args.max_concurrent,
            max_pages_per_market=args.max_pages,
        )
    )


if __name__ == "__main__":
    main()
