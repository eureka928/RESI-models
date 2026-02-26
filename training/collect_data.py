"""
Fetch recently sold US residential properties from Zillow via RapidAPI.

Uses the "Real-Time Real Estate Data" API by OpenWeb Ninja on RapidAPI.
Host: real-time-real-estate-data.p.rapidapi.com

Two endpoints are used:
  1. GET /search — search recently sold properties by location
     Params: location, home_status=RECENTLY_SOLD
     Returns: {"data": [{"zpid", "homeType", "bedrooms", "bathrooms", "latitude", ...}]}
  2. GET /property-details — full property details by zpid
     Params: zpid
     Returns: {"data": {...}} with resoFacts(187 fields), schools, priceHistory, homeType

Modes:
  --search-only   Save search results directly (40 props/call, ~22 features each)
  --detail-zpids  Fetch details only for specific zpids from a file
  --select-details N  Select N zpids from search results using price-stratified sampling

Setup:
  1. Go to https://rapidapi.com/letscrape-6bRBa3QguO5/api/real-time-real-estate-data
  2. Subscribe (Free=100 req/mo, Pro=$25/10K, Ultra=$75/50K, Mega=$150/200K)
  3. Copy your API key from the RapidAPI dashboard
  4. Set in training/.env: RAPIDAPI_KEY=your_key_here

Usage:
    # Original mode: search + detail for every property
    python collect_data.py --rapidapi-key YOUR_KEY --num-properties 50000

    # Search-only mode: save search summaries (40x more cost efficient)
    python collect_data.py --search-only --rapidapi-key YOUR_KEY --max-pages 50

    # Select zpids for detail calls from search results (price-stratified)
    python collect_data.py --select-details 7500

    # Fetch details for selected zpids only
    python collect_data.py --detail-zpids training/detail_zpids.txt --rapidapi-key YOUR_KEY

Saves raw JSON responses to training/raw_data/{zpid}.json (details) or
training/search_data/{zpid}.json (search summaries). Supports resume.
"""

import argparse
import asyncio
import json
import logging
import random
from pathlib import Path

import httpx
from config import RAPIDAPI_HOST, RAPIDAPI_KEY, TARGET_MARKETS

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

RAW_DATA_DIR = Path(__file__).parent / "raw_data"
SEARCH_DATA_DIR = Path(__file__).parent / "search_data"

# Rate limiting defaults
DEFAULT_DELAY = 0.5  # seconds between requests
MAX_CONCURRENT = 3  # concurrent requests to API


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

    Uses GET /search with home_status=RECENTLY_SOLD.
    Returns list of property summary dicts containing zpid.
    """
    host = headers["x-rapidapi-host"]
    url = f"https://{host}/search"
    params = {
        "location": location,
        "home_status": "RECENTLY_SOLD",
    }
    if page > 1:
        params["page"] = str(page)

    try:
        resp = await client.get(url, headers=headers, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        await asyncio.sleep(delay)

        results = []
        if isinstance(data, dict):
            # Primary format: {"data": [...]}
            results = data.get("data", [])
            if not isinstance(results, list):
                results = []

        logger.debug(f"  Search returned {len(results)} results")
        return results

    except httpx.HTTPStatusError as e:
        logger.warning(
            f"Search failed for {location} page {page}: HTTP {e.response.status_code}"
        )
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

    Uses GET /property-details with zpid parameter.
    Returns the property detail dict with resoFacts(187 fields),
    schools, priceHistory, homeType, etc.
    """
    async with semaphore:
        host = headers["x-rapidapi-host"]
        url = f"https://{host}/property-details"
        params = {"zpid": zpid}

        try:
            resp = await client.get(url, headers=headers, params=params, timeout=30)
            resp.raise_for_status()
            raw = resp.json()
            await asyncio.sleep(delay)
            # API wraps detail in {"data": {...}}
            if isinstance(raw, dict) and "data" in raw:
                return raw["data"]
            return raw
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:
                logger.warning(f"Rate limited on zpid {zpid}, waiting 30s...")
                await asyncio.sleep(30)
            else:
                logger.warning(
                    f"Detail fetch failed for zpid {zpid}: HTTP {e.response.status_code}"
                )
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


def save_search_result(result: dict, zpid: str, data_dir: Path) -> bool:
    """
    Save a search result summary as JSON.

    Search results contain basic property data: beds, baths, sqft,
    lat/lon, yearBuilt, homeType, price — enough for ~22 features.
    """
    out_path = data_dir / f"{zpid}.json"
    try:
        with open(out_path, "w") as f:
            json.dump(result, f)
        return True
    except OSError as e:
        logger.warning(f"Failed to save search result for {zpid}: {e}")
        return False


async def collect_search_only(
    client: httpx.AsyncClient,
    headers: dict,
    location: str,
    existing_zpids: set[str],
    delay: float,
    data_dir: Path,
    max_pages: int = 50,
) -> int:
    """
    Collect search results only (no detail calls) for a single market.

    Each search call returns ~40 properties with basic data.
    This is 40x more cost-efficient than detail calls.
    """
    collected = 0
    consecutive_dupes = 0

    for page in range(1, max_pages + 1):
        logger.info(f"  Searching {location} page {page}...")
        results = await search_sold_properties(client, headers, location, page, delay)

        if not results:
            logger.info(f"  No more results for {location} at page {page}")
            break

        new_count = 0
        for result in results:
            zpid = extract_zpid(result)
            if (
                zpid
                and zpid not in existing_zpids
                and save_search_result(result, zpid, data_dir)
            ):
                existing_zpids.add(zpid)
                collected += 1
                new_count += 1

        if new_count == 0:
            consecutive_dupes += 1
            if consecutive_dupes >= 3:
                logger.info(
                    f"  3 consecutive all-duplicate pages for {location}, moving on"
                )
                break
            continue
        else:
            consecutive_dupes = 0

        logger.debug(f"  Saved {new_count} new search results from page {page}")

    return collected


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
    """Collect sold properties for a single market (search + detail for each)."""
    collected = 0

    for page in range(1, max_pages + 1):
        logger.info(f"  Searching {location} page {page}...")
        results = await search_sold_properties(client, headers, location, page, delay)

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

        logger.info(f"  Saved {collected} properties from {location} so far")
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
    Collect properties across all target markets (original mode: search + detail).

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


async def collect_all_search_only(
    api_key: str,
    delay: float = DEFAULT_DELAY,
    max_pages_per_market: int = 50,
) -> int:
    """
    Collect search results only (no detail calls) across all target markets.

    Each search call returns ~40 properties. With 50 pages per market
    and 50 markets, that's ~2,500 search calls = ~100K properties.

    Returns total number of newly collected search results.
    """
    data_dir = SEARCH_DATA_DIR
    data_dir.mkdir(parents=True, exist_ok=True)

    # Also skip zpids already in raw_data (no need to re-search detailed ones)
    existing_search = get_existing_zpids(data_dir)
    existing_detail = get_existing_zpids(RAW_DATA_DIR)
    existing_zpids = existing_search | existing_detail
    logger.info(
        f"Found {len(existing_search)} existing search + "
        f"{len(existing_detail)} existing detail results (will skip)"
    )

    headers = get_headers(api_key)
    total_collected = 0

    async with httpx.AsyncClient() as client:
        for location in TARGET_MARKETS:
            logger.info(
                f"Search-only collecting from {location} "
                f"({len(existing_zpids)} total so far)..."
            )
            count = await collect_search_only(
                client,
                headers,
                location,
                existing_zpids,
                delay,
                data_dir,
                max_pages=max_pages_per_market,
            )
            total_collected += count

    logger.info(
        f"Search-only collection complete: {total_collected} new results, "
        f"{len(existing_zpids)} total"
    )
    return total_collected


async def collect_detail_zpids(
    api_key: str,
    zpids_file: Path,
    delay: float = DEFAULT_DELAY,
    max_concurrent: int = MAX_CONCURRENT,
) -> int:
    """
    Fetch property details for a specific list of zpids.

    Reads zpids from a text file (one per line) and fetches details
    only for those not already in raw_data/.

    Returns total number of newly collected properties.
    """
    data_dir = RAW_DATA_DIR
    data_dir.mkdir(parents=True, exist_ok=True)

    # Read zpid list
    with open(zpids_file) as f:
        all_zpids = [line.strip() for line in f if line.strip()]
    logger.info(f"Loaded {len(all_zpids)} zpids from {zpids_file}")

    existing_zpids = get_existing_zpids(data_dir)
    zpids_to_fetch = [z for z in all_zpids if z not in existing_zpids]
    logger.info(
        f"Need to fetch {len(zpids_to_fetch)} details "
        f"({len(existing_zpids)} already exist)"
    )

    if not zpids_to_fetch:
        logger.info("All zpids already fetched!")
        return 0

    headers = get_headers(api_key)
    semaphore = asyncio.Semaphore(max_concurrent)
    total_collected = 0

    async with httpx.AsyncClient() as client:
        # Process in batches to show progress
        batch_size = 50
        for i in range(0, len(zpids_to_fetch), batch_size):
            batch = zpids_to_fetch[i : i + batch_size]
            logger.info(
                f"Fetching details batch {i // batch_size + 1} "
                f"({i}/{len(zpids_to_fetch)})..."
            )

            tasks = [
                fetch_property_detail(client, headers, zpid, semaphore, delay)
                for zpid in batch
            ]
            details = await asyncio.gather(*tasks)

            for zpid, detail in zip(batch, details):
                if detail:
                    out_path = data_dir / f"{zpid}.json"
                    with open(out_path, "w") as f:
                        json.dump(detail, f)
                    total_collected += 1

    logger.info(f"Detail collection complete: {total_collected} new properties")
    return total_collected


def select_detail_zpids(
    num_details: int = 7500,
    search_dir: Path | None = None,
    output_file: Path | None = None,
) -> Path:
    """
    Select zpids for detail collection using price-stratified sampling.

    Reads search results from search_data/, stratifies by price, and
    selects num_details zpids proportionally:
      - Under $200K: 25% (cheap properties have highest MAPE impact)
      - $200K-$500K: 30% (largest volume)
      - $500K-$1M: 25% (mid-range)
      - Over $1M: 20% (luxury)

    Excludes zpids already in raw_data/ (already have details).

    Returns path to the output file containing selected zpids.
    """
    if search_dir is None:
        search_dir = SEARCH_DATA_DIR
    if output_file is None:
        output_file = Path(__file__).parent / "detail_zpids.txt"

    # Price strata and their budget allocation
    strata = [
        (0, 200_000, 0.25, "Under $200K"),
        (200_000, 500_000, 0.30, "$200K-$500K"),
        (500_000, 1_000_000, 0.25, "$500K-$1M"),
        (1_000_000, float("inf"), 0.20, "Over $1M"),
    ]

    # Read all search results and extract prices
    existing_detail_zpids = get_existing_zpids(RAW_DATA_DIR)
    zpid_prices: dict[str, float] = {}

    search_files = list(search_dir.glob("*.json"))
    logger.info(f"Reading {len(search_files)} search results for stratification...")

    for json_path in search_files:
        zpid = json_path.stem
        if zpid in existing_detail_zpids:
            continue  # Already have details

        try:
            with open(json_path) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue

        # Extract price from search result
        price = None
        for field in ["lastSoldPrice", "soldPrice", "price"]:
            val = data.get(field)
            if val:
                try:
                    p = float(val)
                    if p > 0:
                        price = p
                        break
                except (ValueError, TypeError):
                    pass

        if price is not None:
            zpid_prices[zpid] = price

    logger.info(f"Found {len(zpid_prices)} search results with valid prices")

    # Stratified sampling
    selected = []
    for low, high, fraction, label in strata:
        budget = int(num_details * fraction)
        candidates = [z for z, p in zpid_prices.items() if low <= p < high]
        n_select = min(budget, len(candidates))
        chosen = random.sample(candidates, n_select) if candidates else []
        selected.extend(chosen)
        logger.info(
            f"  {label}: {len(candidates)} candidates, "
            f"selected {n_select}/{budget} target"
        )

    # If we have leftover budget (some strata had fewer candidates than target),
    # fill from remaining candidates across all strata
    remaining_budget = num_details - len(selected)
    if remaining_budget > 0:
        selected_set = set(selected)
        remaining_candidates = [z for z in zpid_prices if z not in selected_set]
        fill = random.sample(
            remaining_candidates,
            min(remaining_budget, len(remaining_candidates)),
        )
        selected.extend(fill)
        if fill:
            logger.info(f"  Filled {len(fill)} extra zpids from remaining pool")

    # Write to file
    with open(output_file, "w") as f:
        for zpid in selected:
            f.write(f"{zpid}\n")

    logger.info(
        f"Selected {len(selected)} zpids for detail collection -> {output_file}"
    )
    return output_file


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
        help="Target number of properties to collect (original mode)",
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
    # New modes
    parser.add_argument(
        "--search-only",
        action="store_true",
        help="Save search results directly without detail calls (40x more efficient)",
    )
    parser.add_argument(
        "--detail-zpids",
        type=Path,
        default=None,
        help="File containing zpids to fetch details for (one per line)",
    )
    parser.add_argument(
        "--select-details",
        type=int,
        default=None,
        help="Select N zpids from search results using price-stratified sampling",
    )
    args = parser.parse_args()

    # Mode: select zpids for detail calls (no API key needed)
    if args.select_details is not None:
        select_detail_zpids(num_details=args.select_details)
        return

    # Mode: search-only
    if args.search_only:
        if not args.rapidapi_key:
            logger.error(
                "No API key provided. Set RAPIDAPI_KEY env var or use --rapidapi-key"
            )
            return
        asyncio.run(
            collect_all_search_only(
                api_key=args.rapidapi_key,
                delay=args.delay,
                max_pages_per_market=args.max_pages,
            )
        )
        return

    # Mode: detail zpids from file
    if args.detail_zpids is not None:
        if not args.rapidapi_key:
            logger.error(
                "No API key provided. Set RAPIDAPI_KEY env var or use --rapidapi-key"
            )
            return
        asyncio.run(
            collect_detail_zpids(
                api_key=args.rapidapi_key,
                zpids_file=args.detail_zpids,
                delay=args.delay,
                max_concurrent=args.max_concurrent,
            )
        )
        return

    # Default mode: original search + detail
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
