"""
Generate diverse + hard edge case test samples for miner-cli local evaluation.

Two modes:
  1. Basic (50 stratified samples): diverse coverage across price ranges
  2. Edge cases (200 hard samples): properties most likely to cause high MAPE

Edge cases are identified by:
  - Model prediction errors (if model.onnx exists) — worst predictions
  - Statistical outliers — extreme values in key features
  - Boundary properties — near price clip bounds, unusual price/sqft
  - Rare property types — manufactured, multi-family, lots

Usage:
    # Generate 50 basic + 200 edge cases (250 total)
    cd training && python generate_test_samples.py

    # Basic only (50 samples)
    python generate_test_samples.py --basic-only

    # Custom counts
    python generate_test_samples.py --num-basic 50 --num-edge 200
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from config import FEATURE_ORDER, MIN_PRICE, MAX_PRICE

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
    output_path: Path = OUTPUT_PATH,
    basic_only: bool = False,
    seed: int = 123,
):
    """Generate test samples combining basic stratified + edge cases."""
    df = load_dataset(dataset_path)
    rng = np.random.RandomState(seed)

    # Step 1: Basic stratified samples
    logger.info(f"\n=== Selecting {num_basic} basic stratified samples ===")
    basic_idx = select_stratified_basic(df, num_basic, rng)
    all_selected = set(basic_idx)
    logger.info(f"Selected {len(basic_idx)} basic samples")

    edge_idx = []
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

        # Fill any remaining slots randomly from unselected data
        total_edge = len(edge_idx)
        if total_edge < num_edge:
            remaining = list(set(df.index) - all_selected)
            if remaining:
                extra = min(num_edge - total_edge, len(remaining))
                fill = rng.choice(remaining, size=extra, replace=False).tolist()
                edge_idx.extend(fill)
                logger.info(f"Filled {extra} remaining edge case slots randomly")

    # Build output
    all_indices = basic_idx + edge_idx
    samples = build_sample_json(df, all_indices)

    # Tag samples with their category
    basic_set = set(basic_idx)
    edge_set = set(edge_idx)

    n_basic = sum(1 for idx in all_indices if idx in basic_set)
    n_edge = sum(1 for idx in all_indices if idx in edge_set)

    output = {
        "description": (
            f"Test samples for miner CLI local evaluation. "
            f"{n_basic} basic stratified + {n_edge} edge cases = {len(samples)} total. "
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
    logger.info(f"  Basic samples: {n_basic}")
    logger.info(f"  Edge cases:    {n_edge}")
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
        "--basic-only", action="store_true",
        help="Only generate basic stratified samples (no edge cases)",
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
        output_path=args.output,
        basic_only=args.basic_only,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
