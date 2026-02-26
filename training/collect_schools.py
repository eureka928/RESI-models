"""
Download NCES Public School Universe data and build a KD-tree for spatial school lookup.

NCES (National Center for Education Statistics) provides free data on ~100K US public schools
including latitude, longitude, school level, and operational status.

This replaces Zillow's nearbySchools data for search-only properties, providing:
  - school_count: number of schools within radius
  - min_school_distance: distance to nearest school (miles)
  - avg_school_rating, max_school_rating: defaulted to 5.5 (NCES has no ratings)
  - elementary_rating, middle_rating, high_rating: defaulted to 5.5

Usage:
    python collect_schools.py
    python collect_schools.py --output-dir training/school_data
"""

import argparse
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from config import MIN_SCHOOL_DISTANCE_DEFAULT, SCHOOL_RATING_DEFAULT
from scipy.spatial import cKDTree

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SCHOOL_DATA_DIR = Path(__file__).parent / "school_data"

# NCES Public School Universe Survey data
# Using the most recent available year
NCES_URL = "https://nces.ed.gov/ccd/Data/zip/ccd_sch_029_2223_w_1a_080923.zip"

# Relevant NCES columns
NCES_COLUMNS = {
    "NCESSCH": "nces_id",  # unique school ID
    "SCH_NAME": "name",  # school name
    "LSTATE": "state",  # state abbreviation
    "LAT": "lat",  # latitude
    "LON": "lon",  # longitude
    "LEVEL": "level",  # school level code
    "SY_STATUS": "status",  # operational status (1=open)
    "LOCALE": "locale",  # locale code (urban/suburban/rural)
}

# School level codes from NCES
LEVEL_MAP = {
    "1": "elementary",  # Primary (low grade = PK through 03; high grade = PK through 08)
    "2": "middle",  # Middle (low grade = 04 through 07; high grade = 04 through 09)
    "3": "high",  # High (low grade = 07 through 12; high grade = 12 only)
    "4": "other",  # Other (any other grade span)
}

# Earth radius in miles for distance calculation
EARTH_RADIUS_MILES = 3958.8
# Degrees per mile (approximate, for KD-tree radius)
DEG_PER_MILE = 1.0 / 69.0


def download_nces_data(output_dir: Path) -> pd.DataFrame:
    """
    Download and parse NCES Public School Universe data.

    Falls back to a direct CSV download if the zip approach fails.
    Returns DataFrame with columns: nces_id, name, state, lat, lon, level, status, locale.
    """
    import io
    import zipfile

    import httpx

    output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = output_dir / "nces_schools_raw.parquet"

    if cache_path.exists():
        logger.info(f"Loading cached NCES data from {cache_path}")
        return pd.read_parquet(cache_path)

    logger.info("Downloading NCES Public School Universe data...")
    logger.info(f"URL: {NCES_URL}")

    try:
        with httpx.Client(timeout=120, follow_redirects=True) as client:
            resp = client.get(NCES_URL)
            resp.raise_for_status()

        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            csv_names = [n for n in zf.namelist() if n.endswith(".csv")]
            if not csv_names:
                raise ValueError("No CSV found in NCES zip file")
            logger.info(f"Extracting {csv_names[0]}...")
            with zf.open(csv_names[0]) as csv_file:
                df = pd.read_csv(
                    csv_file,
                    dtype=str,
                    encoding="latin-1",
                    low_memory=False,
                )
    except Exception as e:
        logger.warning(f"Zip download failed ({e}), trying alternate URL...")
        # Fallback: try a different year or format
        alt_url = "https://nces.ed.gov/ccd/Data/zip/ccd_sch_029_2122_w_1a_091222.zip"
        try:
            with httpx.Client(timeout=120, follow_redirects=True) as client:
                resp = client.get(alt_url)
                resp.raise_for_status()

            with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
                csv_names = [n for n in zf.namelist() if n.endswith(".csv")]
                if not csv_names:
                    raise ValueError("No CSV in fallback zip")
                with zf.open(csv_names[0]) as csv_file:
                    df = pd.read_csv(
                        csv_file,
                        dtype=str,
                        encoding="latin-1",
                        low_memory=False,
                    )
        except Exception as e2:
            logger.error(f"Both NCES downloads failed: {e2}")
            raise

    logger.info(f"Raw NCES data: {len(df)} rows, {len(df.columns)} columns")

    # Find matching columns (NCES column names vary slightly by year)
    col_map = {}
    for nces_col, our_col in NCES_COLUMNS.items():
        matches = [c for c in df.columns if nces_col in c.upper()]
        if matches:
            col_map[matches[0]] = our_col
        else:
            logger.warning(f"NCES column {nces_col} not found in data")

    df = df.rename(columns=col_map)

    # Keep only columns we need (that were found)
    keep_cols = [c for c in NCES_COLUMNS.values() if c in df.columns]
    df = df[keep_cols].copy()

    # Parse coordinates
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")

    # Filter: valid coordinates in continental US
    df = df.dropna(subset=["lat", "lon"])
    df = df[
        (df["lat"] >= 24.0)
        & (df["lat"] <= 50.0)
        & (df["lon"] >= -125.0)
        & (df["lon"] <= -66.0)
    ]

    # Filter: open schools only (status code "1" = open)
    if "status" in df.columns:
        # Keep open schools; if status column is messy, keep all
        open_mask = df["status"].astype(str).str.strip().isin(["1", "Open", "OPEN"])
        if open_mask.sum() > 0:
            df = df[open_mask]
            logger.info(f"Filtered to {len(df)} open schools")

    # Map school level
    if "level" in df.columns:
        df["level"] = df["level"].astype(str).str.strip().map(LEVEL_MAP).fillna("other")

    # Cache
    df.to_parquet(cache_path, index=False)
    logger.info(f"Cached {len(df)} schools to {cache_path}")

    return df


def build_school_tree(df: pd.DataFrame) -> cKDTree:
    """Build a KD-tree from school lat/lon coordinates."""
    coords = df[["lat", "lon"]].values
    tree = cKDTree(coords)
    logger.info(f"Built KD-tree with {len(coords)} schools")
    return tree


def lookup_schools(
    tree: cKDTree,
    school_df: pd.DataFrame,
    lat: float,
    lon: float,
    radius_miles: float = 5.0,
) -> dict:
    """
    Look up school features for a given lat/lon.

    Uses KD-tree for efficient spatial query.

    Returns dict with 7 school features matching the validator's format:
      - school_count: number of schools within radius
      - min_school_distance: distance to nearest school (miles)
      - avg_school_rating: defaulted to SCHOOL_RATING_DEFAULT (5.5)
      - max_school_rating: defaulted to SCHOOL_RATING_DEFAULT
      - elementary_rating: defaulted to SCHOOL_RATING_DEFAULT
      - middle_rating: defaulted to SCHOOL_RATING_DEFAULT
      - high_rating: defaulted to SCHOOL_RATING_DEFAULT
    """
    if lat == 0 or lon == 0:
        return _default_school_features()

    # Query KD-tree for schools within radius
    radius_deg = radius_miles * DEG_PER_MILE
    point = np.array([lat, lon])

    # Find all schools within radius
    indices = tree.query_ball_point(point, radius_deg)

    if not indices:
        return _default_school_features()

    # Compute actual distances for found schools (Haversine approximation)
    school_coords = school_df.iloc[indices][["lat", "lon"]].values
    distances = _haversine_miles(lat, lon, school_coords[:, 0], school_coords[:, 1])

    # Filter by actual distance (KD-tree uses Euclidean on degrees)
    within = distances <= radius_miles
    if not within.any():
        return _default_school_features()

    distances = distances[within]

    school_count = float(len(distances))
    min_distance = float(distances.min())

    # NCES doesn't have GreatSchools ratings — default to sentinel value
    # This matches the validator's default and is consistent across all rows
    return {
        "school_count": school_count,
        "min_school_distance": min_distance,
        "avg_school_rating": SCHOOL_RATING_DEFAULT,
        "max_school_rating": SCHOOL_RATING_DEFAULT,
        "elementary_rating": SCHOOL_RATING_DEFAULT,
        "middle_rating": SCHOOL_RATING_DEFAULT,
        "high_rating": SCHOOL_RATING_DEFAULT,
    }


def _default_school_features() -> dict:
    """Return default school features when no schools found."""
    return {
        "school_count": 0.0,
        "min_school_distance": MIN_SCHOOL_DISTANCE_DEFAULT,
        "avg_school_rating": SCHOOL_RATING_DEFAULT,
        "max_school_rating": SCHOOL_RATING_DEFAULT,
        "elementary_rating": SCHOOL_RATING_DEFAULT,
        "middle_rating": SCHOOL_RATING_DEFAULT,
        "high_rating": SCHOOL_RATING_DEFAULT,
    }


def _haversine_miles(
    lat1: float,
    lon1: float,
    lat2: np.ndarray,
    lon2: np.ndarray,
) -> np.ndarray:
    """Compute Haversine distance in miles from a point to an array of points."""
    lat1_r = np.radians(lat1)
    lon1_r = np.radians(lon1)
    lat2_r = np.radians(lat2)
    lon2_r = np.radians(lon2)

    dlat = lat2_r - lat1_r
    dlon = lon2_r - lon1_r

    a = np.sin(dlat / 2) ** 2 + np.cos(lat1_r) * np.cos(lat2_r) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))

    return EARTH_RADIUS_MILES * c


def save_school_data(output_dir: Path | None = None) -> Path:
    """
    Download NCES data, build KD-tree, and save as pickle.

    Returns path to the saved pickle file containing (school_df, tree).
    """
    if output_dir is None:
        output_dir = SCHOOL_DATA_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    pkl_path = output_dir / "schools.pkl"

    df = download_nces_data(output_dir)
    tree = build_school_tree(df)

    # Save both the DataFrame and tree
    with open(pkl_path, "wb") as f:
        pickle.dump({"school_df": df, "tree": tree}, f)

    logger.info(f"Saved school data + KD-tree to {pkl_path}")

    # Print summary
    if "level" in df.columns:
        level_counts = df["level"].value_counts()
        logger.info(f"School levels: {level_counts.to_dict()}")
    if "state" in df.columns:
        logger.info(f"States covered: {df['state'].nunique()}")

    return pkl_path


def load_school_data(school_dir: Path | None = None) -> tuple[pd.DataFrame, cKDTree]:
    """
    Load saved school data and KD-tree from pickle.

    Returns (school_df, tree) tuple.
    """
    if school_dir is None:
        school_dir = SCHOOL_DATA_DIR
    pkl_path = school_dir / "schools.pkl"

    if not pkl_path.exists():
        logger.info("School data not found, downloading...")
        save_school_data(school_dir)

    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    return data["school_df"], data["tree"]


def main():
    parser = argparse.ArgumentParser(
        description="Download NCES school data and build spatial lookup"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=SCHOOL_DATA_DIR,
        help="Output directory for school data",
    )
    args = parser.parse_args()

    save_school_data(args.output_dir)

    # Verify by doing a sample lookup
    school_df, tree = load_school_data(args.output_dir)
    # Test: Times Square, NYC
    test_result = lookup_schools(tree, school_df, 40.7580, -73.9855)
    logger.info(f"Test lookup (Times Square): {test_result}")


if __name__ == "__main__":
    main()
