"""
Generate diverse + hard edge case test samples for miner-cli local evaluation.

Three layers:
  1. Basic (50 stratified samples): diverse coverage across price ranges
  2. Edge cases (200 hard samples): model errors + statistical outliers
  3. Validator failure cases (250 hard samples): patterns that cause top models
     to fail during daily evaluation — sentinel defaults, unknown home types,
     search-only zero-feature properties, price boundaries, zero-area ratios

Edge cases are identified by:
  - Model prediction errors (if model.onnx exists) — worst predictions
  - Statistical outliers — extreme values in key features
  - Boundary properties — near price clip bounds, unusual price/sqft
  - Rare property types — manufactured, multi-family, lots
  - Validator failure patterns — sentinel defaults, zero features, encoding gaps

Usage:
    # Generate 50 basic + 200 edge + 250 validator-failure = 500 total
    cd training && python generate_test_samples.py

    # Basic only (50 samples)
    python generate_test_samples.py --basic-only

    # Custom counts
    python generate_test_samples.py --num-basic 50 --num-edge 200 --num-validator 250
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from config import (
    FEATURE_ORDER, MIN_PRICE, MAX_PRICE,
    SCHOOL_RATING_DEFAULT, MIN_SCHOOL_DISTANCE_DEFAULT,
    YEARS_SINCE_LAST_SALE_DEFAULT, MONTHS_SINCE_LAST_SALE_DEFAULT,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_DATASET = Path(__file__).parent / "dataset.parquet"
DEFAULT_MODEL = Path(__file__).parent / "model.onnx"
OUTPUT_PATH = Path(__file__).parent.parent / "real_estate" / "miner_cli" / "test_samples.json"


def load_dataset(dataset_path: Path) -> pd.DataFrame:
    """Load and basic-filter dataset."""
    logger.info(f"Loading dataset from {dataset_path}...")
    df = pd.read_parquet(dataset_path)
    logger.info(f"Loaded {len(df)} samples")

    # Keep valid price range
    df = df[(df["price"] >= MIN_PRICE) & (df["price"] <= MAX_PRICE)].copy()
    logger.info(f"After price filtering: {len(df)} samples")
    return df


def select_stratified_basic(df: pd.DataFrame, num_samples: int, rng: np.random.RandomState) -> list[int]:
    """Select stratified basic samples across price ranges."""
    buckets = [
        ("Under $100K", df["price"] < 100_000),
        ("$100K-$200K", (df["price"] >= 100_000) & (df["price"] < 200_000)),
        ("$200K-$300K", (df["price"] >= 200_000) & (df["price"] < 300_000)),
        ("$300K-$500K", (df["price"] >= 300_000) & (df["price"] < 500_000)),
        ("$500K-$750K", (df["price"] >= 500_000) & (df["price"] < 750_000)),
        ("$750K-$1M", (df["price"] >= 750_000) & (df["price"] < 1_000_000)),
        ("$1M-$2M", (df["price"] >= 1_000_000) & (df["price"] < 2_000_000)),
        ("Over $2M", df["price"] >= 2_000_000),
    ]

    per_bucket = max(1, num_samples // len(buckets))
    selected = []

    for name, mask in buckets:
        bucket_idx = df[mask].index.tolist()
        if not bucket_idx:
            logger.info(f"  Basic {name}: 0 available")
            continue
        n = min(per_bucket, len(bucket_idx))
        chosen = rng.choice(bucket_idx, size=n, replace=False).tolist()
        selected.extend(chosen)
        logger.info(f"  Basic {name}: {n} samples")

    # Fill remainder
    if len(selected) < num_samples:
        remaining = list(set(df.index) - set(selected))
        extra = rng.choice(remaining, size=min(num_samples - len(selected), len(remaining)), replace=False).tolist()
        selected.extend(extra)

    return selected[:num_samples]


def find_model_error_cases(
    df: pd.DataFrame, model_path: Path, num_cases: int, rng: np.random.RandomState,
    exclude_idx: set[int],
) -> list[int]:
    """Find properties where the ONNX model makes the worst predictions."""
    if not model_path.exists():
        logger.info("  No model.onnx found — skipping model-based edge cases")
        return []

    try:
        import onnxruntime as ort
    except ImportError:
        logger.info("  onnxruntime not installed — skipping model-based edge cases")
        return []

    logger.info(f"  Running model inference on {len(df)} samples to find worst predictions...")

    session = ort.InferenceSession(str(model_path))
    input_name = session.get_inputs()[0].name

    # Prepare features
    features = df[FEATURE_ORDER].fillna(0).values.astype(np.float32)
    actual_prices = df["price"].values

    # Run inference in batches
    batch_size = 1000
    predictions = np.zeros(len(df))
    for i in range(0, len(df), batch_size):
        batch = features[i:i + batch_size]
        pred = session.run(None, {input_name: batch})[0].flatten()
        predictions[i:i + len(pred)] = pred

    # Clip predictions
    predictions = np.clip(predictions, MIN_PRICE, MAX_PRICE)

    # Compute per-sample MAPE
    pct_errors = np.abs(actual_prices - predictions) / actual_prices

    # Get indices sorted by error (worst first), excluding already selected
    available_mask = np.array([i not in exclude_idx for i in df.index])
    error_with_idx = list(zip(pct_errors, df.index))
    error_with_idx = [(e, idx) for e, idx in error_with_idx if idx not in exclude_idx]
    error_with_idx.sort(key=lambda x: -x[0])

    # Take top worst predictions
    worst = [idx for _, idx in error_with_idx[:num_cases]]

    if worst:
        worst_errors = [pct_errors[df.index.get_loc(idx)] for idx in worst[:10]]
        logger.info(f"  Model error cases: top 10 errors = {[f'{e:.1%}' for e in worst_errors]}")

    return worst


def find_statistical_edge_cases(
    df: pd.DataFrame, num_cases: int, rng: np.random.RandomState,
    exclude_idx: set[int],
) -> list[int]:
    """Find statistical outliers across multiple feature dimensions."""
    selected = set()

    def _add_extremes(series: pd.Series, label: str, n: int = 5):
        """Add top and bottom n from a series."""
        available = series.drop(index=list(exclude_idx | selected), errors="ignore")
        if len(available) == 0:
            return
        # Bottom n
        bottom = available.nsmallest(n).index.tolist()
        selected.update(bottom)
        # Top n
        top = available.nlargest(n).index.tolist()
        selected.update(top)
        logger.info(f"  {label}: +{len(bottom) + len(top)} samples")

    # --- Price extremes ---
    _add_extremes(df["price"], "Extreme prices (cheapest/most expensive)", n=8)

    # --- Price per sqft extremes ---
    price_per_sqft = df["price"] / df["living_area_sqft"].replace(0, np.nan)
    price_per_sqft = price_per_sqft.dropna()
    _add_extremes(price_per_sqft, "Extreme price/sqft", n=8)

    # --- Living area extremes ---
    _add_extremes(df["living_area_sqft"], "Extreme living area", n=5)

    # --- Lot size extremes ---
    _add_extremes(df["lot_size_sqft"], "Extreme lot size", n=5)

    # --- Age extremes (very old / brand new) ---
    _add_extremes(df["property_age"], "Extreme property age", n=5)

    # --- Bedroom/bathroom ratio extremes ---
    bath_ratio = df["bedrooms"] / df["bathrooms"].replace(0, np.nan)
    bath_ratio = bath_ratio.dropna()
    _add_extremes(bath_ratio, "Extreme beds/bath ratio", n=5)

    # --- Lot to living ratio extremes ---
    if "lot_to_living_ratio" in df.columns:
        _add_extremes(df["lot_to_living_ratio"], "Extreme lot/living ratio", n=5)

    # --- Near price clip boundaries ---
    near_min = df[(df["price"] >= MIN_PRICE) & (df["price"] < MIN_PRICE * 1.2)]
    near_max = df[(df["price"] > MAX_PRICE * 0.8) & (df["price"] <= MAX_PRICE)]
    boundary = list(set(near_min.index) | set(near_max.index))
    boundary = [i for i in boundary if i not in exclude_idx and i not in selected]
    if boundary:
        n_take = min(10, len(boundary))
        chosen = rng.choice(boundary, size=n_take, replace=False).tolist()
        selected.update(chosen)
        logger.info(f"  Near price clip boundaries: +{n_take} samples")

    # --- Rare property types ---
    for home_type_col in ["home_type_MULTI_FAMILY", "home_type_MANUFACTURED", "home_type_LOT"]:
        if home_type_col in df.columns:
            rare = df[df[home_type_col] == 1.0]
            rare = rare[~rare.index.isin(exclude_idx | selected)]
            if len(rare) > 0:
                n_take = min(5, len(rare))
                chosen = rng.choice(rare.index.tolist(), size=n_take, replace=False).tolist()
                selected.update(chosen)
                logger.info(f"  {home_type_col}: +{n_take} samples")

    # --- Properties with previous sale data (flips, appreciation) ---
    if "has_previous_sale_data" in df.columns:
        has_sale = df[df["has_previous_sale_data"] == 1.0]
        has_sale = has_sale[~has_sale.index.isin(exclude_idx | selected)]
        if len(has_sale) > 0:
            n_take = min(10, len(has_sale))
            chosen = rng.choice(has_sale.index.tolist(), size=n_take, replace=False).tolist()
            selected.update(chosen)
            logger.info(f"  Has previous sale data: +{n_take} samples")

    # --- Recent flips ---
    if "is_recent_flip" in df.columns:
        flips = df[df["is_recent_flip"] == 1.0]
        flips = flips[~flips.index.isin(exclude_idx | selected)]
        if len(flips) > 0:
            n_take = min(5, len(flips))
            chosen = rng.choice(flips.index.tolist(), size=n_take, replace=False).tolist()
            selected.update(chosen)
            logger.info(f"  Recent flips: +{n_take} samples")

    # --- High amenity count vs zero amenity ---
    if "total_amenity_count" in df.columns:
        _add_extremes(df["total_amenity_count"], "Extreme amenity count", n=5)

    # --- Geographic outliers (extreme lat/lon) ---
    _add_extremes(df["latitude"], "Extreme latitude", n=5)
    _add_extremes(df["longitude"], "Extreme longitude", n=5)

    # --- Zero living area (data quality edge case) ---
    zero_area = df[df["living_area_sqft"] == 0]
    zero_area = zero_area[~zero_area.index.isin(exclude_idx | selected)]
    if len(zero_area) > 0:
        n_take = min(5, len(zero_area))
        chosen = rng.choice(zero_area.index.tolist(), size=n_take, replace=False).tolist()
        selected.update(chosen)
        logger.info(f"  Zero living area: +{n_take} samples")

    # --- High school rating variation ---
    if "avg_school_rating" in df.columns:
        _add_extremes(df["avg_school_rating"], "Extreme school ratings", n=3)

    result = list(selected - exclude_idx)
    logger.info(f"  Total statistical edge cases: {len(result)}")

    # Trim to requested size
    if len(result) > num_cases:
        result = rng.choice(result, size=num_cases, replace=False).tolist()

    return result


def find_validator_failure_cases(
    df: pd.DataFrame, num_cases: int, rng: np.random.RandomState,
    exclude_idx: set[int],
) -> list[int]:
    """Find properties matching patterns that cause top models to fail during
    validator evaluation.

    These target specific encoding/feature gaps in the validator pipeline:
    sentinel defaults, unknown home types, zero-feature search-only properties,
    zero-area ratio issues, and properties in tricky price ranges.
    """
    selected = set()
    budget = num_cases

    def _pick(mask_or_idx, label: str, n: int = 8):
        """Pick up to n samples from mask, respecting exclusions."""
        nonlocal selected
        if isinstance(mask_or_idx, (pd.Series, np.ndarray)):
            candidates = df[mask_or_idx].index.tolist()
        else:
            candidates = list(mask_or_idx)
        candidates = [i for i in candidates if i not in exclude_idx and i not in selected]
        if not candidates:
            logger.info(f"    {label}: 0 available")
            return
        n_take = min(n, len(candidates))
        chosen = rng.choice(candidates, size=n_take, replace=False).tolist()
        selected.update(chosen)
        logger.info(f"    {label}: +{n_take} samples")

    # === CATEGORY 1: Unknown / unusual home types ===
    # When home type doesn't match any known category, all one-hot = 0 and
    # home_type_nan = 1. Models trained mostly on SINGLE_FAMILY struggle here.
    logger.info("  [Cat 1] Unknown / unusual home types")
    if "home_type_nan" in df.columns:
        _pick(df["home_type_nan"] == 1.0, "home_type_nan=1 (unknown type)", n=15)
    if "home_type_HOME_TYPE_UNKNOWN" in df.columns:
        _pick(df["home_type_HOME_TYPE_UNKNOWN"] == 1.0, "HOME_TYPE_UNKNOWN", n=10)
    # Condos/townhouses often mapped to nan since they're not in the 5 categories
    if "home_type_MULTI_FAMILY" in df.columns:
        _pick(df["home_type_MULTI_FAMILY"] == 1.0, "MULTI_FAMILY", n=8)
    if "home_type_LOT" in df.columns:
        _pick(df["home_type_LOT"] == 1.0, "LOT (land-only)", n=8)
    if "home_type_MANUFACTURED" in df.columns:
        _pick(df["home_type_MANUFACTURED"] == 1.0, "MANUFACTURED", n=8)

    # === CATEGORY 2: Sentinel school defaults (no nearby schools) ===
    # Rural properties with no NCES schools within search radius get sentinel
    # defaults: school_count=0, all ratings=5.5, min_distance=1.2
    logger.info("  [Cat 2] Sentinel school defaults (rural/no schools)")
    if "school_count" in df.columns and "avg_school_rating" in df.columns:
        # No schools at all
        _pick(df["school_count"] == 0, "school_count=0", n=15)
        # Sentinel rating (5.5 default)
        sentinel_rating = np.isclose(df["avg_school_rating"], SCHOOL_RATING_DEFAULT, atol=0.01)
        _pick(sentinel_rating, "avg_school_rating=5.5 (sentinel)", n=10)
    if "min_school_distance" in df.columns:
        # Sentinel distance (1.2 default — means no school found)
        sentinel_dist = np.isclose(df["min_school_distance"], MIN_SCHOOL_DISTANCE_DEFAULT, atol=0.01)
        _pick(sentinel_dist, "min_school_distance=1.2 (sentinel)", n=8)

    # === CATEGORY 3: No previous sale data (sentinel sale defaults) ===
    # New construction or properties with no price history get sentinels:
    # years_since_last_sale=12, months=144, previous_sale_price=0, all change=0
    logger.info("  [Cat 3] No previous sale data (new construction / first sale)")
    if "has_previous_sale_data" in df.columns:
        _pick(df["has_previous_sale_data"] == 0.0, "no previous sale data", n=15)
    if "is_new_construction" in df.columns:
        _pick(df["is_new_construction"] == 1.0, "new construction", n=10)
    if "years_since_last_sale" in df.columns:
        sentinel_sale = np.isclose(df["years_since_last_sale"], YEARS_SINCE_LAST_SALE_DEFAULT, atol=0.1)
        _pick(sentinel_sale, "years_since_last_sale=12 (sentinel)", n=8)

    # === CATEGORY 4: Zero/near-zero living area (ratio division issues) ===
    # When living_area=0, lot_to_living_ratio = inf or 0, beds_per_bath = nan
    # These cause unpredictable model behavior
    logger.info("  [Cat 4] Zero/near-zero key features (ratio edge cases)")
    _pick(df["living_area_sqft"] == 0, "living_area=0", n=8)
    _pick(df["living_area_sqft"] < 200, "living_area<200 (tiny)", n=5)
    _pick(df["bathrooms"] == 0, "bathrooms=0 (ratio issues)", n=5)
    _pick(df["bedrooms"] == 0, "bedrooms=0", n=5)
    if "lot_to_living_ratio" in df.columns:
        _pick(df["lot_to_living_ratio"] > 100, "extreme lot/living ratio >100", n=5)
        _pick(df["lot_to_living_ratio"] == 0, "lot/living ratio = 0", n=3)

    # === CATEGORY 5: Search-only properties (many zero features) ===
    # Properties from search-only collection have ~57 features as 0.0
    # (only ~22 features available from search results)
    logger.info("  [Cat 5] Search-only properties (many zero features)")
    # Count how many features are exactly 0 per row
    feature_cols = [f for f in FEATURE_ORDER if f in df.columns]
    zero_count = (df[feature_cols] == 0.0).sum(axis=1)
    _pick(zero_count >= 50, "50+ zero features (search-only)", n=15)
    _pick(zero_count >= 40, "40+ zero features (sparse data)", n=10)

    # === CATEGORY 6: Price boundary stress test ===
    # Properties near the MIN_PRICE/MAX_PRICE clip bounds are tricky because
    # any prediction bias gets amplified at the boundary
    logger.info("  [Cat 6] Price boundary stress tests")
    _pick(
        (df["price"] >= MIN_PRICE) & (df["price"] < MIN_PRICE * 1.1),
        f"price near floor ($50K-$55K)", n=10,
    )
    _pick(
        (df["price"] >= MIN_PRICE * 1.1) & (df["price"] < MIN_PRICE * 1.5),
        f"price $55K-$75K (ultra-cheap)", n=8,
    )
    _pick(
        (df["price"] >= 75_000) & (df["price"] < 100_000),
        f"price $75K-$100K (cheap rural)", n=8,
    )
    _pick(
        (df["price"] > 5_000_000) & (df["price"] <= MAX_PRICE),
        "price $5M+ (ultra-luxury)", n=8,
    )
    _pick(
        (df["price"] > 3_000_000) & (df["price"] <= 5_000_000),
        "price $3M-$5M (luxury)", n=5,
    )

    # === CATEGORY 7: Extreme price/sqft (valuation anomalies) ===
    # Very high or low $/sqft properties confuse models — likely condos in
    # expensive cities vs rural land
    logger.info("  [Cat 7] Extreme price/sqft (valuation anomalies)")
    price_per_sqft = df["price"] / df["living_area_sqft"].replace(0, np.nan)
    _pick(price_per_sqft > 1000, "price/sqft > $1000", n=8)
    _pick(price_per_sqft < 30, "price/sqft < $30 (very cheap)", n=5)
    _pick(
        (price_per_sqft > 500) & (price_per_sqft <= 1000),
        "price/sqft $500-$1000 (expensive metro)", n=5,
    )

    # === CATEGORY 8: Geographic edge cases ===
    # Properties at extreme latitudes/longitudes fall outside the dense
    # geographic surface data, getting poor geo feature values
    logger.info("  [Cat 8] Geographic edge cases")
    _pick(df["latitude"] > 47, "high latitude (northern states)", n=5)
    _pick(df["latitude"] < 26, "low latitude (southern FL/HI)", n=5)
    _pick(df["longitude"] > -70, "eastern longitude (New England coast)", n=5)
    _pick(df["longitude"] < -122, "western longitude (Pacific coast)", n=5)

    # === CATEGORY 9: Feature combination stress ===
    # Unusual combinations that models rarely see together
    logger.info("  [Cat 9] Unusual feature combinations")
    if "has_pool" in df.columns and "has_fireplace" in df.columns:
        _pick(
            (df["has_pool"] == 1.0) & (df["has_fireplace"] == 1.0) & (df["price"] < 200_000),
            "pool+fireplace but cheap (<$200K)", n=5,
        )
    if "stories" in df.columns:
        _pick(df["stories"] >= 4, "4+ stories", n=3)
    _pick(df["bedrooms"] >= 7, "7+ bedrooms", n=5)
    _pick(df["bathrooms"] >= 6, "6+ bathrooms", n=3)
    if "garage_capacity" in df.columns:
        _pick(df["garage_capacity"] >= 4, "4+ car garage", n=3)
    if "fireplaces_count" in df.columns:
        _pick(df["fireplaces_count"] >= 3, "3+ fireplaces", n=3)

    # === CATEGORY 10: All amenity flags off ===
    # Properties with zero amenities — no pool, no garage, no fireplace,
    # no cooling, no heating — these get minimal feature signal
    logger.info("  [Cat 10] Minimal amenity properties")
    bool_features = [
        "has_garage", "has_cooling", "has_heating", "has_fireplace",
        "has_pool", "has_spa", "has_view",
    ]
    available_bools = [f for f in bool_features if f in df.columns]
    if available_bools:
        amenity_sum = df[available_bools].sum(axis=1)
        _pick(amenity_sum == 0, "zero amenity flags (no pool/garage/cooling/etc)", n=10)
        # Also max amenities
        _pick(amenity_sum == len(available_bools), "all amenity flags on", n=5)

    result = list(selected - exclude_idx)
    logger.info(f"  Total validator failure cases: {len(result)}")

    # Trim to requested size or fill remainder
    if len(result) > num_cases:
        result = rng.choice(result, size=num_cases, replace=False).tolist()

    return result


def build_sample_json(df: pd.DataFrame, indices: list[int]) -> list[dict]:
    """Convert DataFrame rows to test sample JSON format."""
    samples = []
    for idx in indices:
        row = df.loc[idx]
        features = {}
        for feat in FEATURE_ORDER:
            val = float(row[feat]) if feat in row.index and pd.notna(row[feat]) else 0.0
            if val == int(val) and abs(val) < 1e10:
                val = float(int(val))
            features[feat] = val

        zpid = str(int(row["zpid"])) if "zpid" in row.index and pd.notna(row.get("zpid")) else str(idx)
        samples.append({
            "external_id": zpid,
            "actual_price": float(row["price"]),
            "features": features,
        })

    samples.sort(key=lambda s: s["actual_price"])
    return samples


def generate_test_samples(
    dataset_path: Path = DEFAULT_DATASET,
    model_path: Path = DEFAULT_MODEL,
    num_basic: int = 50,
    num_edge: int = 200,
    num_validator: int = 250,
    output_path: Path = OUTPUT_PATH,
    basic_only: bool = False,
    seed: int = 123,
):
    """Generate test samples: basic stratified + edge cases + validator failure cases."""
    df = load_dataset(dataset_path)
    rng = np.random.RandomState(seed)

    # Step 1: Basic stratified samples
    logger.info(f"\n=== Selecting {num_basic} basic stratified samples ===")
    basic_idx = select_stratified_basic(df, num_basic, rng)
    all_selected = set(basic_idx)
    logger.info(f"Selected {len(basic_idx)} basic samples")

    edge_idx = []
    validator_idx = []
    if not basic_only:
        # Step 2: Model-based error cases (worst predictions)
        logger.info(f"\n=== Finding model prediction error cases ===")
        model_cases = find_model_error_cases(df, model_path, num_edge // 2, rng, all_selected)
        edge_idx.extend(model_cases)
        all_selected.update(model_cases)
        logger.info(f"Found {len(model_cases)} model error cases")

        # Step 3: Statistical edge cases
        remaining_edge = num_edge - len(model_cases)
        logger.info(f"\n=== Finding {remaining_edge} statistical edge cases ===")
        stat_cases = find_statistical_edge_cases(df, remaining_edge, rng, all_selected)
        edge_idx.extend(stat_cases)
        all_selected.update(stat_cases)
        logger.info(f"Found {len(stat_cases)} statistical edge cases")

        # Fill any remaining edge slots randomly
        total_edge = len(edge_idx)
        if total_edge < num_edge:
            remaining = list(set(df.index) - all_selected)
            if remaining:
                extra = min(num_edge - total_edge, len(remaining))
                fill = rng.choice(remaining, size=extra, replace=False).tolist()
                edge_idx.extend(fill)
                all_selected.update(fill)
                logger.info(f"Filled {extra} remaining edge case slots randomly")

        # Step 4: Validator failure pattern cases
        logger.info(f"\n=== Finding {num_validator} validator failure cases ===")
        val_cases = find_validator_failure_cases(df, num_validator, rng, all_selected)
        validator_idx.extend(val_cases)
        all_selected.update(val_cases)
        logger.info(f"Found {len(val_cases)} validator failure cases")

        # Fill any remaining validator slots randomly
        if len(validator_idx) < num_validator:
            remaining = list(set(df.index) - all_selected)
            if remaining:
                extra = min(num_validator - len(validator_idx), len(remaining))
                fill = rng.choice(remaining, size=extra, replace=False).tolist()
                validator_idx.extend(fill)
                logger.info(f"Filled {extra} remaining validator failure slots randomly")

    # Build output
    all_indices = basic_idx + edge_idx + validator_idx
    samples = build_sample_json(df, all_indices)

    # Count by category
    basic_set = set(basic_idx)
    edge_set = set(edge_idx)
    validator_set = set(validator_idx)

    n_basic = sum(1 for idx in all_indices if idx in basic_set)
    n_edge = sum(1 for idx in all_indices if idx in edge_set)
    n_validator = sum(1 for idx in all_indices if idx in validator_set)

    output = {
        "description": (
            f"Test samples for miner CLI local evaluation. "
            f"{n_basic} basic + {n_edge} edge + {n_validator} validator-failure = {len(samples)} total. "
            f"Features match feature_config.yaml order (79 features)."
        ),
        "samples": samples,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    # Summary
    prices = [s["actual_price"] for s in samples]
    logger.info(f"\n{'=' * 60}")
    logger.info(f"Saved {len(samples)} test samples to {output_path}")
    logger.info(f"File size: {output_path.stat().st_size / 1024:.1f} KB")
    logger.info(f"  Basic samples:          {n_basic}")
    logger.info(f"  Edge cases:             {n_edge}")
    logger.info(f"  Validator failure cases: {n_validator}")
    logger.info(f"  Price range:   ${min(prices):,.0f} - ${max(prices):,.0f}")
    logger.info(f"  Median price:  ${sorted(prices)[len(prices)//2]:,.0f}")

    logger.info(f"\nPrice distribution:")
    for name, lo, hi in [
        ("Under $100K", 0, 100_000),
        ("$100K-$200K", 100_000, 200_000),
        ("$200K-$300K", 200_000, 300_000),
        ("$300K-$500K", 300_000, 500_000),
        ("$500K-$1M", 500_000, 1_000_000),
        ("$1M-$2M", 1_000_000, 2_000_000),
        ("Over $2M", 2_000_000, float("inf")),
    ]:
        count = sum(1 for p in prices if lo <= p < hi)
        if count > 0:
            logger.info(f"  {name}: {count}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate test samples (basic + edge cases) for miner CLI"
    )
    parser.add_argument(
        "--dataset", type=Path, default=DEFAULT_DATASET,
        help="Path to training Parquet file",
    )
    parser.add_argument(
        "--model", type=Path, default=DEFAULT_MODEL,
        help="Path to ONNX model for finding worst predictions",
    )
    parser.add_argument(
        "--num-basic", type=int, default=50,
        help="Number of basic stratified samples",
    )
    parser.add_argument(
        "--num-edge", type=int, default=200,
        help="Number of edge case samples",
    )
    parser.add_argument(
        "--num-validator", type=int, default=250,
        help="Number of validator failure pattern samples",
    )
    parser.add_argument(
        "--basic-only", action="store_true",
        help="Only generate basic stratified samples (no edge/validator cases)",
    )
    parser.add_argument(
        "--output", type=Path, default=OUTPUT_PATH,
        help="Output path for test_samples.json",
    )
    parser.add_argument(
        "--seed", type=int, default=123,
        help="Random seed for reproducibility",
    )
    args = parser.parse_args()

    generate_test_samples(
        dataset_path=args.dataset,
        model_path=args.model,
        num_basic=args.num_basic,
        num_edge=args.num_edge,
        num_validator=args.num_validator,
        output_path=args.output,
        basic_only=args.basic_only,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
