"""
Download geographic data for building price surfaces.

Data sources (all free, no API key needed):
1. Zillow Home Value Index (ZHVI) - CSV from Zillow Research
2. ZIP code centroids - Census ZCTA gazetteer
3. Census ACS 5-Year - median income, education, population by ZIP
4. Redfin market data - median sale price, days on market by ZIP

Builds two-resolution grids of geographic features and saves as .npy files.

Usage:
    python collect_geo_data.py
    python collect_geo_data.py --output-dir training/geo_surfaces
"""

import argparse
import io
import logging
from pathlib import Path

import httpx
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from tqdm import tqdm

from config import (
    G0_NLAT,
    G0_NLON,
    G0_PRECISION,
    G1_NLAT,
    G1_NLON,
    G1_PRECISION,
    LAT_MAX,
    LAT_MIN,
    LON_MAX,
    LON_MIN,
    NUM_RP_FEATURES,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
)
logger = logging.getLogger(__name__)

# Zillow Research data URLs
ZHVI_URL = (
    "https://files.zillowstatic.com/research/public_csvs/zhvi/"
    "Zip_zhvi_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv"
)

# Census ZCTA gazetteer (ZIP code centroids)
ZCTA_URL = (
    "https://www2.census.gov/geo/docs/maps-data/data/gazetteer/"
    "2024_Gazetteer/2024_Gaz_zcta_national.zip"
)
ZCTA_URL_FALLBACK = (
    "https://www2.census.gov/geo/docs/maps-data/data/gazetteer/"
    "2023_Gazetteer/2023_Gaz_zcta_national.txt"
)

# Redfin market data (ZIP-level, monthly, free TSV download)
REDFIN_ZIP_URL = (
    "https://redfin-public-data.s3.us-west-2.amazonaws.com/"
    "redfin_market_tracker/zip_code_market_tracker.tsv000.gz"
)

# Census ACS 5-Year data via API (no key needed for small requests)
CENSUS_ACS_BASE = "https://api.census.gov/data/2022/acs/acs5"

GEO_DATA_DIR = Path(__file__).parent / "geo_data"
GEO_SURFACES_DIR = Path(__file__).parent / "geo_surfaces"


def download_zhvi(output_dir: Path) -> pd.DataFrame:
    """Download ZHVI data from Zillow Research."""
    cache_path = output_dir / "zhvi_raw.csv"

    if cache_path.exists():
        logger.info(f"Loading cached ZHVI data from {cache_path}")
        return pd.read_csv(cache_path, dtype={"RegionName": str})

    logger.info("Downloading ZHVI data from Zillow Research...")
    resp = httpx.get(ZHVI_URL, timeout=120, follow_redirects=True)
    resp.raise_for_status()

    output_dir.mkdir(parents=True, exist_ok=True)
    cache_path.write_bytes(resp.content)
    logger.info(f"Saved ZHVI data to {cache_path}")

    return pd.read_csv(io.BytesIO(resp.content), dtype={"RegionName": str})


def download_zcta_centroids(output_dir: Path) -> pd.DataFrame:
    """Download ZIP code centroid coordinates from Census."""
    import zipfile

    cache_path = output_dir / "zcta_centroids.tsv"

    if cache_path.exists():
        logger.info(f"Loading cached ZCTA centroids from {cache_path}")
        return pd.read_csv(cache_path, sep="\t", dtype={"GEOID": str})

    output_dir.mkdir(parents=True, exist_ok=True)

    for url in [ZCTA_URL, ZCTA_URL_FALLBACK]:
        try:
            logger.info(f"Downloading ZCTA centroids from {url}...")
            resp = httpx.get(url, timeout=120, follow_redirects=True)
            resp.raise_for_status()

            if url.endswith(".zip"):
                with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
                    txt_names = [n for n in zf.namelist() if n.endswith(".txt")]
                    if not txt_names:
                        raise ValueError("No .txt found in ZCTA zip")
                    with zf.open(txt_names[0]) as f:
                        content = f.read()
            else:
                content = resp.content

            cache_path.write_bytes(content)
            logger.info(f"Saved ZCTA centroids to {cache_path}")
            return pd.read_csv(io.BytesIO(content), sep="\t", dtype={"GEOID": str})
        except Exception as e:
            logger.warning(f"Failed to download from {url}: {e}")
            continue

    raise RuntimeError("All ZCTA download URLs failed")


def download_redfin_zip_data(output_dir: Path) -> pd.DataFrame:
    """Download Redfin ZIP-code level market data (free TSV)."""
    cache_path = output_dir / "redfin_zip_market.tsv.gz"

    if cache_path.exists():
        logger.info(f"Loading cached Redfin data from {cache_path}")
        try:
            return pd.read_csv(
                cache_path, sep="\t", compression="gzip",
                dtype={"region": str}, low_memory=False,
            )
        except Exception as e:
            logger.warning(f"Failed to read cached Redfin data: {e}")

    logger.info("Downloading Redfin ZIP market data (~400MB compressed)...")
    try:
        resp = httpx.get(REDFIN_ZIP_URL, timeout=600, follow_redirects=True)
        resp.raise_for_status()
        output_dir.mkdir(parents=True, exist_ok=True)
        cache_path.write_bytes(resp.content)
        logger.info(f"Saved Redfin data to {cache_path}")
        return pd.read_csv(
            io.BytesIO(resp.content), sep="\t", compression="gzip",
            dtype={"region": str}, low_memory=False,
        )
    except Exception as e:
        logger.warning(f"Failed to download Redfin data: {e}")
        logger.warning("Continuing without Redfin data (surfaces will be zero-filled)")
        return pd.DataFrame()


def download_census_acs(output_dir: Path) -> pd.DataFrame:
    """
    Download Census ACS 5-Year data by ZCTA (ZIP Code Tabulation Area).

    Fetches: median household income (B19013_001E),
             % bachelor's degree+ (B15003_022E..025E / B15003_001E),
             total population (B01003_001E)
    """
    cache_path = output_dir / "census_acs_zcta.csv"

    if cache_path.exists():
        logger.info(f"Loading cached Census ACS data from {cache_path}")
        return pd.read_csv(cache_path, dtype={"zcta": str})

    logger.info("Downloading Census ACS 5-Year data by ZCTA...")

    # Variables: median income, total population, total education denominator,
    # bachelor's, master's, professional, doctorate
    variables = "B19013_001E,B01003_001E,B15003_001E,B15003_022E,B15003_023E,B15003_024E,B15003_025E"

    try:
        url = f"{CENSUS_ACS_BASE}?get=NAME,{variables}&for=zip%20code%20tabulation%20area:*"
        resp = httpx.get(url, timeout=120, follow_redirects=True)
        resp.raise_for_status()
        data = resp.json()

        # First row is header
        header = data[0]
        rows = data[1:]
        df = pd.DataFrame(rows, columns=header)

        # Rename and compute
        df = df.rename(columns={
            "zip code tabulation area": "zcta",
            "B19013_001E": "median_income",
            "B01003_001E": "total_pop",
            "B15003_001E": "edu_total",
            "B15003_022E": "bachelors",
            "B15003_023E": "masters",
            "B15003_024E": "professional",
            "B15003_025E": "doctorate",
        })

        for col in ["median_income", "total_pop", "edu_total", "bachelors",
                     "masters", "professional", "doctorate"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # Compute % with bachelor's degree or higher
        df["pct_bachelors"] = (
            (df["bachelors"].fillna(0) + df["masters"].fillna(0) +
             df["professional"].fillna(0) + df["doctorate"].fillna(0))
            / df["edu_total"].clip(lower=1)
        )

        df["zcta"] = df["zcta"].str.zfill(5)

        output_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(cache_path, index=False)
        logger.info(f"Saved Census ACS data to {cache_path} ({len(df)} ZCTAs)")
        return df

    except Exception as e:
        logger.warning(f"Failed to download Census ACS data: {e}")
        logger.warning("Continuing without Census data (surfaces will be zero-filled)")
        return pd.DataFrame()


def build_zip_zhvi_lookup(
    zhvi_df: pd.DataFrame, zcta_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Join ZHVI values with ZIP centroid coordinates.

    Returns DataFrame with columns: zip, lat, lon, zhvi (log-scale), zhvi_raw
    """
    # Get the most recent ZHVI value (last date column)
    date_cols = [c for c in zhvi_df.columns if c.startswith("20")]
    if not date_cols:
        raise ValueError("No date columns found in ZHVI data")

    latest_col = sorted(date_cols)[-1]
    logger.info(f"Using ZHVI data from {latest_col}")

    zhvi_subset = zhvi_df[["RegionName", latest_col]].copy()
    zhvi_subset.columns = ["zip", "zhvi_raw"]
    zhvi_subset = zhvi_subset.dropna(subset=["zhvi_raw"])
    zhvi_subset["zip"] = zhvi_subset["zip"].str.zfill(5)

    # Parse ZCTA centroids (column name varies by year: INTPTLONG vs INTPTLON)
    lon_col = "INTPTLONG" if "INTPTLONG" in zcta_df.columns else "INTPTLON"
    zcta_subset = zcta_df[["GEOID", "INTPTLAT", lon_col]].copy()
    zcta_subset.columns = ["zip", "lat", "lon"]
    zcta_subset["zip"] = zcta_subset["zip"].str.strip().str.zfill(5)
    zcta_subset["lat"] = pd.to_numeric(zcta_subset["lat"], errors="coerce")
    zcta_subset["lon"] = pd.to_numeric(zcta_subset["lon"], errors="coerce")
    zcta_subset = zcta_subset.dropna()

    # Join
    merged = zhvi_subset.merge(zcta_subset, on="zip", how="inner")
    merged["zhvi"] = np.log(merged["zhvi_raw"].clip(lower=1))

    logger.info(
        f"Built ZIP-ZHVI lookup: {len(merged)} ZIPs with ZHVI data "
        f"(ZHVI range: ${merged['zhvi_raw'].min():,.0f} - ${merged['zhvi_raw'].max():,.0f})"
    )

    return merged


def enrich_zip_data(
    zip_data: pd.DataFrame,
    census_df: pd.DataFrame,
    redfin_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Enrich ZIP-level data with Census ACS and Redfin market features.

    Adds columns: median_income, pct_bachelors, population_density,
                  redfin_median_price, redfin_dom
    """
    # Census ACS enrichment
    if not census_df.empty and "zcta" in census_df.columns:
        census_sub = census_df[["zcta", "median_income", "pct_bachelors", "total_pop"]].copy()
        census_sub = census_sub.rename(columns={"zcta": "zip"})
        census_sub["zip"] = census_sub["zip"].str.zfill(5)
        zip_data = zip_data.merge(census_sub, on="zip", how="left")
        logger.info(f"  Census enrichment: {zip_data['median_income'].notna().sum()} ZIPs matched")
    else:
        zip_data["median_income"] = np.nan
        zip_data["pct_bachelors"] = np.nan
        zip_data["total_pop"] = np.nan

    # Redfin enrichment — get the most recent month's data per ZIP
    if not redfin_df.empty and "region" in redfin_df.columns:
        # Redfin uses region_type = "zip_code", period_duration = "monthly"
        rf = redfin_df.copy()
        if "region_type" in rf.columns:
            rf = rf[rf["region_type"] == "zip_code"]
        if "period_duration" in rf.columns:
            rf = rf[rf["period_duration"] == "monthly"]

        # Get most recent period per ZIP
        if "period_end" in rf.columns:
            rf["period_end"] = pd.to_datetime(rf["period_end"], errors="coerce")
            rf = rf.sort_values("period_end", ascending=False).drop_duplicates(
                subset=["region"], keep="first"
            )

        rf_sub = pd.DataFrame()
        rf_sub["zip"] = rf["region"].str.zfill(5)

        if "median_sale_price" in rf.columns:
            rf_sub["redfin_median_price"] = pd.to_numeric(
                rf["median_sale_price"].values, errors="coerce"
            )
        if "median_dom" in rf.columns:
            rf_sub["redfin_dom"] = pd.to_numeric(
                rf["median_dom"].values, errors="coerce"
            )

        zip_data = zip_data.merge(rf_sub, on="zip", how="left")
        logger.info(
            f"  Redfin enrichment: {zip_data.get('redfin_median_price', pd.Series()).notna().sum()} ZIPs matched"
        )

    # Fill missing and compute derived
    zip_data["median_income"] = zip_data.get("median_income", pd.Series(dtype=float)).fillna(0)
    zip_data["pct_bachelors"] = zip_data.get("pct_bachelors", pd.Series(dtype=float)).fillna(0)
    zip_data["total_pop"] = zip_data.get("total_pop", pd.Series(dtype=float)).fillna(0)
    zip_data["redfin_median_price"] = zip_data.get("redfin_median_price", pd.Series(dtype=float)).fillna(0)
    zip_data["redfin_dom"] = zip_data.get("redfin_dom", pd.Series(dtype=float)).fillna(0)

    # Log-scale median income (same scale as ZHVI)
    zip_data["log_median_income"] = np.log(zip_data["median_income"].clip(lower=1))
    # Log-scale Redfin median price
    zip_data["log_redfin_price"] = np.log(zip_data["redfin_median_price"].clip(lower=1))
    # Approximate population density (pop / rough ZCTA area ~25 sq km average)
    zip_data["pop_density_log"] = np.log(zip_data["total_pop"].clip(lower=1) / 25.0 + 1)

    return zip_data


def build_surfaces_for_grid(
    zip_data: pd.DataFrame,
    precision: int,
    nlat: int,
    nlon: int,
    label: str,
) -> dict[str, np.ndarray]:
    """
    Build geographic feature surfaces for a single grid resolution.

    For each grid cell computes 17 features:
    - zhvi: log(median home value) from nearest ZIP
    - is_zip: whether the cell has ZHVI data nearby
    - rp0-rp9: regional price percentiles at 2 radii
    - median_income: log(median household income)
    - pct_bachelors: fraction with bachelor's+
    - pop_density: log(pop/area)
    - redfin_price: log(Redfin median sale price)
    - redfin_dom: median days on market

    Returns dict of surface_name -> 1D array of shape (nlat * nlon,).
    """
    logger.info(f"Building {label} surfaces ({nlat}x{nlon} = {nlat * nlon} cells)...")

    # Build KD-tree from ZIP centroids
    zip_coords = zip_data[["lat", "lon"]].values
    zip_zhvi = zip_data["zhvi"].values
    tree = cKDTree(zip_coords)

    # Extra feature arrays indexed by ZIP
    zip_log_income = zip_data["log_median_income"].values
    zip_pct_bach = zip_data["pct_bachelors"].values
    zip_pop_dens = zip_data["pop_density_log"].values
    zip_rf_price = zip_data["log_redfin_price"].values
    zip_rf_dom = zip_data["redfin_dom"].values

    # Generate grid cell centers
    lat_edges = np.linspace(LAT_MIN, LAT_MAX, nlat + 1)
    lon_edges = np.linspace(LON_MIN, LON_MAX, nlon + 1)
    lat_centers = (lat_edges[:-1] + lat_edges[1:]) / 2
    lon_centers = (lon_edges[:-1] + lon_edges[1:]) / 2

    # Create meshgrid of all cell centers
    lon_grid, lat_grid = np.meshgrid(lon_centers, lat_centers)
    grid_points = np.column_stack([lat_grid.ravel(), lon_grid.ravel()])
    n_cells = len(grid_points)

    # Initialize surfaces
    zhvi_surface = np.zeros(n_cells, dtype=np.float32)
    is_zip_surface = np.zeros(n_cells, dtype=np.float32)
    rp_surfaces = [np.zeros(n_cells, dtype=np.float32) for _ in range(NUM_RP_FEATURES)]
    income_surface = np.zeros(n_cells, dtype=np.float32)
    bachelors_surface = np.zeros(n_cells, dtype=np.float32)
    pop_density_surface = np.zeros(n_cells, dtype=np.float32)
    rf_price_surface = np.zeros(n_cells, dtype=np.float32)
    rf_dom_surface = np.zeros(n_cells, dtype=np.float32)

    # Radii for regional price features (in degrees, ~111km per degree)
    radii = [0.5] * 5 + [1.0] * 5
    all_percentiles = [10, 25, 50, 75, 90, 10, 25, 50, 75, 90]

    # Process in chunks
    chunk_size = 10000
    for start in tqdm(range(0, n_cells, chunk_size), desc=f"  {label}"):
        end = min(start + chunk_size, n_cells)
        chunk_points = grid_points[start:end]

        # Nearest ZIP for point features
        dists_nearest, idxs_nearest = tree.query(chunk_points, k=1)
        zhvi_surface[start:end] = zip_zhvi[idxs_nearest]
        is_zip_surface[start:end] = (dists_nearest < 0.2).astype(np.float32)
        income_surface[start:end] = zip_log_income[idxs_nearest]
        bachelors_surface[start:end] = zip_pct_bach[idxs_nearest]
        pop_density_surface[start:end] = zip_pop_dens[idxs_nearest]
        rf_price_surface[start:end] = zip_rf_price[idxs_nearest]
        rf_dom_surface[start:end] = zip_rf_dom[idxs_nearest]

        # Regional price features: query neighbors within radius
        for i, (radius, pctls) in enumerate(zip(radii, all_percentiles)):
            neighbors = tree.query_ball_point(chunk_points, r=radius)
            for j, (cell_idx, nbrs) in enumerate(
                zip(range(start, end), neighbors)
            ):
                if nbrs:
                    nbr_zhvi = zip_zhvi[nbrs]
                    rp_surfaces[i][cell_idx] = np.percentile(nbr_zhvi, pctls)
                else:
                    rp_surfaces[i][cell_idx] = zhvi_surface[cell_idx]

    surfaces = {
        "zhvi": zhvi_surface,
        "is_zip": is_zip_surface,
    }
    for i in range(NUM_RP_FEATURES):
        surfaces[f"rp{i}"] = rp_surfaces[i]

    # Extra surfaces (Census + Redfin)
    surfaces["median_income"] = income_surface
    surfaces["pct_bachelors"] = bachelors_surface
    surfaces["pop_density"] = pop_density_surface
    surfaces["redfin_price"] = rf_price_surface
    surfaces["redfin_dom"] = rf_dom_surface

    return surfaces


def build_and_save_all_surfaces(
    zip_data: pd.DataFrame, output_dir: Path
) -> None:
    """Build surfaces at both resolutions and save as .npy files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Coarse grid (g0)
    g0_surfaces = build_surfaces_for_grid(
        zip_data, G0_PRECISION, G0_NLAT, G0_NLON, label="g0"
    )
    for name, arr in g0_surfaces.items():
        path = output_dir / f"g0_{name}.npy"
        np.save(path, arr)
        logger.info(f"Saved {path} shape={arr.shape}")

    # Fine grid (g1)
    g1_surfaces = build_surfaces_for_grid(
        zip_data, G1_PRECISION, G1_NLAT, G1_NLON, label="g1"
    )
    for name, arr in g1_surfaces.items():
        path = output_dir / f"g1_{name}.npy"
        np.save(path, arr)
        logger.info(f"Saved {path} shape={arr.shape}")

    # Save grid metadata for ONNX export
    metadata = {
        "g0_precision": G0_PRECISION,
        "g0_nlat": G0_NLAT,
        "g0_nlon": G0_NLON,
        "g1_precision": G1_PRECISION,
        "g1_nlat": G1_NLAT,
        "g1_nlon": G1_NLON,
        "lat_min": LAT_MIN,
        "lat_max": LAT_MAX,
        "lon_min": LON_MIN,
        "lon_max": LON_MAX,
    }
    np.save(output_dir / "grid_metadata.npy", metadata)

    # Estimate total size
    total_bytes = sum(arr.nbytes for arr in g0_surfaces.values())
    total_bytes += sum(arr.nbytes for arr in g1_surfaces.values())
    logger.info(
        f"All geographic surfaces saved ({total_bytes / 1024 / 1024:.1f} MB total)"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Download geographic data and build price surfaces"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=GEO_SURFACES_DIR,
        help="Directory to save surface .npy files",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=GEO_DATA_DIR,
        help="Directory to cache downloaded raw data",
    )
    parser.add_argument(
        "--skip-redfin",
        action="store_true",
        help="Skip Redfin data download (large file)",
    )
    args = parser.parse_args()

    # Download data
    zhvi_df = download_zhvi(args.data_dir)
    zcta_df = download_zcta_centroids(args.data_dir)
    census_df = download_census_acs(args.data_dir)
    redfin_df = pd.DataFrame() if args.skip_redfin else download_redfin_zip_data(args.data_dir)

    # Build ZIP-ZHVI lookup
    zip_data = build_zip_zhvi_lookup(zhvi_df, zcta_df)

    # Enrich with Census + Redfin
    zip_data = enrich_zip_data(zip_data, census_df, redfin_df)

    # Build and save surfaces
    build_and_save_all_surfaces(zip_data, args.output_dir)


if __name__ == "__main__":
    main()
