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

# NCES EDGE Geocode data (school locations with lat/lon)
NCES_URLS = [
    "https://nces.ed.gov/programs/edge/data/EDGE_GEOCODE_PUBLICSCH_2425.zip",
    "https://nces.ed.gov/programs/edge/data/EDGE_GEOCODE_PUBLICSCH_2324.zip",
    "https://nces.ed.gov/programs/edge/data/EDGE_GEOCODE_PUBLICSCH_2223.zip",
]

# Relevant NCES columns (EDGE geocode format)
NCES_COLUMNS = {
    "NCESSCH": "nces_id",  # unique school ID
    "NAME": "name",  # school name
    "STATE": "state",  # state abbreviation
    "LAT": "lat",  # latitude
    "LON": "lon",  # longitude
    "LOCALE": "locale",  # locale code (urban/suburban/rural)
}

# Note: EDGE geocode data doesn't include school level or status.
# All schools with valid lat/lon in the geocode file are included.

# Earth radius in miles for distance calculation
EARTH_RADIUS_MILES = 3958.8
# Degrees per mile (approximate, for KD-tree radius)
DEG_PER_MILE = 1.0 / 69.0


def download_nces_data(output_dir: Path) -> pd.DataFrame:
    """
    Download and parse NCES EDGE school location data.

    Tries multiple years of EDGE Geocode files (shapefiles with .dbf).
    Returns DataFrame with columns: nces_id, name, state, lat, lon, locale.
    """
    import io
    import zipfile

    import httpx

    output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = output_dir / "nces_schools_raw.parquet"

    if cache_path.exists():
        logger.info(f"Loading cached NCES data from {cache_path}")
        return pd.read_parquet(cache_path)

    df = None
    for url in NCES_URLS:
        try:
            logger.info(f"Downloading NCES school data from {url}...")
            with httpx.Client(timeout=180, follow_redirects=True) as client:
                resp = client.get(url)
                resp.raise_for_status()

            with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
                # Look for .dbf file (shapefile attribute table)
                dbf_names = [n for n in zf.namelist() if n.lower().endswith(".dbf")]
                csv_names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
                xlsx_names = [n for n in zf.namelist() if n.lower().endswith(".xlsx")]

                if csv_names:
                    logger.info(f"Extracting CSV: {csv_names[0]}")
                    with zf.open(csv_names[0]) as f:
                        df = pd.read_csv(f, dtype=str, encoding="latin-1", low_memory=False)
                elif dbf_names:
                    logger.info(f"Extracting DBF: {dbf_names[0]}")
                    with zf.open(dbf_names[0]) as f:
                        df = _read_dbf(f)
                elif xlsx_names:
                    logger.info(f"Extracting XLSX: {xlsx_names[0]}")
                    with zf.open(xlsx_names[0]) as f:
                        df = pd.read_excel(io.BytesIO(f.read()), dtype=str)
                else:
                    raise ValueError(f"No CSV/DBF/XLSX found in {url}")

            logger.info(f"Downloaded {len(df)} school records")
            break
        except Exception as e:
            logger.warning(f"Failed: {url} — {e}")
            df = None
            continue

    if df is None:
        raise RuntimeError("All NCES download URLs failed")

    logger.info(f"Raw NCES data: {len(df)} rows, columns: {list(df.columns)}")

    # Find matching columns (names vary slightly by year)
    col_map = {}
    for nces_col, our_col in NCES_COLUMNS.items():
        matches = [c for c in df.columns if c.upper().strip() == nces_col]
        if not matches:
            matches = [c for c in df.columns if nces_col in c.upper().strip()]
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

    logger.info(f"Filtered to {len(df)} schools in continental US")

    # Cache
    df.to_parquet(cache_path, index=False)
    logger.info(f"Cached {len(df)} schools to {cache_path}")

    return df


def _read_dbf(fileobj) -> pd.DataFrame:
    """Read a DBF file into a pandas DataFrame (no external dependency)."""
    import struct

    data = fileobj.read()
    numrec = struct.unpack_from("<I", data, 4)[0]
    header_size = struct.unpack_from("<H", data, 8)[0]
    record_size = struct.unpack_from("<H", data, 10)[0]

    # Parse field descriptors (32 bytes each, starting at offset 32)
    fields = []
    offset = 32
    while offset < header_size - 1 and data[offset] != 0x0D:
        name = data[offset : offset + 11].split(b"\x00")[0].decode("ascii").strip()
        ftype = chr(data[offset + 11])
        fsize = data[offset + 16]
        fields.append((name, ftype, fsize))
        offset += 32

    # Parse records
    records = []
    for i in range(numrec):
        rec_offset = header_size + i * record_size
        if data[rec_offset] == 0x2A:  # deleted record
            continue
        rec = {}
        field_offset = rec_offset + 1  # skip deletion flag
        for name, _ftype, fsize in fields:
            raw = data[field_offset : field_offset + fsize].decode("latin-1").strip()
            rec[name] = raw
            field_offset += fsize
        records.append(rec)

    return pd.DataFrame(records)


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
    if "state" in df.columns:
        logger.info(f"States covered: {df['state'].nunique()}")
    logger.info(f"Total schools: {len(df)}")

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
