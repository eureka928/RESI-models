"""
Generate test samples for miner-cli local evaluation, closely matching the
validator's actual daily evaluation data.

Layers:
  1. Validator-realistic (300 samples): mirrors the validator's price/geography
     distribution — complete feature data, real-sale-like properties
  2. Model errors (100 samples): properties where our model predicts worst
  3. Challenging realistic (100 samples): realistic but harder — unusual home
     types, new construction, extreme $/sqft, geographic edges

The validator sends real recently-sold properties with ALL 80 features complete.
No price filters, no type filters. MAPE scoring with no error cap.

Usage:
    # Generate 500 validator-realistic test samples
    cd training && python generate_test_samples.py

    # Quick sanity check (100 samples)
    python generate_test_samples.py --num-realistic 100 --num-error 0 --num-challenge 0

    # Custom counts
    python generate_test_samples.py --num-realistic 300 --num-error 100 --num-challenge 100
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
    """Load dataset and apply validator-realistic filters.

    The validator sends real recently-sold properties with complete data.
    We filter to properties that look like what the validator would actually send:
    - Valid price range
    - Non-zero living area (real properties always have this)
    - Non-zero bathrooms (real listings always have this)
    """
    logger.info(f"Loading dataset from {dataset_path}...")
    df = pd.read_parquet(dataset_path)
    logger.info(f"Loaded {len(df)} samples")

    # Keep valid price range
    df = df[(df["price"] >= MIN_PRICE) & (df["price"] <= MAX_PRICE)].copy()
    logger.info(f"After price filtering: {len(df)} samples")
    return df


def _filter_complete_properties(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to properties with complete, realistic data (like validator sends).

    The validator requires ALL 80 features present with no None values.
    Properties from search-only collection or with data gaps are filtered out.
    """
    feature_cols = [f for f in FEATURE_ORDER if f in df.columns]
    zero_count = (df[feature_cols] == 0.0).sum(axis=1)

    mask = (
        (df["living_area_sqft"] > 0) &        # Real properties have area
        (df["bathrooms"] > 0) &                 # Real listings have bathrooms
        (zero_count < 30)                       # Not a search-only sparse record
    )
    filtered = df[mask]
    logger.info(f"  Complete-data filter: {len(filtered)}/{len(df)} properties pass")
    return filtered


def select_validator_realistic(
    df: pd.DataFrame, num_samples: int, rng: np.random.RandomState,
) -> list[int]:
    """Select samples matching the validator's actual daily evaluation distribution.

    The validator evaluates ~100 recently-sold properties per day across all US states.
    The distribution is weighted toward the $150K-$800K range where most US sales happen.
    We mimic this with realistic price buckets.
    """
    complete = _filter_complete_properties(df)

    # Price distribution matching real US home sales (NAR data)
    # ~60% in $150K-$500K, ~20% in $500K-$1M, ~10% below $150K, ~10% above $1M
    buckets = [
        ("$50K-$150K",   (complete["price"] >= 50_000)    & (complete["price"] < 150_000),  0.08),
        ("$150K-$250K",  (complete["price"] >= 150_000)   & (complete["price"] < 250_000),  0.18),
        ("$250K-$350K",  (complete["price"] >= 250_000)   & (complete["price"] < 350_000),  0.18),
        ("$350K-$500K",  (complete["price"] >= 350_000)   & (complete["price"] < 500_000),  0.18),
        ("$500K-$750K",  (complete["price"] >= 500_000)   & (complete["price"] < 750_000),  0.15),
        ("$750K-$1M",    (complete["price"] >= 750_000)   & (complete["price"] < 1_000_000), 0.10),
        ("$1M-$2M",      (complete["price"] >= 1_000_000) & (complete["price"] < 2_000_000), 0.08),
        ("$2M+",         (complete["price"] >= 2_000_000), 0.05),
    ]

    selected = []
    for name, mask, fraction in buckets:
        target = max(1, int(num_samples * fraction))
        bucket_idx = complete[mask].index.tolist()
        if not bucket_idx:
            logger.info(f"  Realistic {name}: 0 available")
            continue
        n = min(target, len(bucket_idx))
        chosen = rng.choice(bucket_idx, size=n, replace=False).tolist()
        selected.extend(chosen)
        logger.info(f"  Realistic {name}: {n} samples (target {target})")

    # Fill remainder from complete data
    if len(selected) < num_samples:
        remaining = list(set(complete.index) - set(selected))
        if remaining:
            extra = min(num_samples - len(selected), len(remaining))
            fill = rng.choice(remaining, size=extra, replace=False).tolist()
            selected.extend(fill)
            logger.info(f"  Fill: +{len(fill)} random complete samples")

    return selected[:num_samples]


def find_model_error_cases(
    df: pd.DataFrame, model_path: Path, num_cases: int, rng: np.random.RandomState,
    exclude_idx: set[int],
) -> list[int]:
    """Find complete-data properties where the ONNX model makes the worst predictions.

    These are the most valuable test cases — real-looking properties that our model
    gets wrong, which is exactly what loses MAPE points during validator evaluation.
    """
    if num_cases <= 0:
        return []

    complete = _filter_complete_properties(df)

    if not model_path.exists():
        logger.info("  No model.onnx found — selecting random complete samples instead")
        available = list(set(complete.index) - exclude_idx)
        n = min(num_cases, len(available))
        return rng.choice(available, size=n, replace=False).tolist()

    try:
        import onnxruntime as ort
    except ImportError:
        logger.info("  onnxruntime not installed — selecting random complete samples instead")
        available = list(set(complete.index) - exclude_idx)
        n = min(num_cases, len(available))
        return rng.choice(available, size=n, replace=False).tolist()

    logger.info(f"  Running model inference on {len(complete)} complete samples...")

    session = ort.InferenceSession(str(model_path))
    input_name = session.get_inputs()[0].name

    # Prepare features
    features = complete[FEATURE_ORDER].fillna(0).values.astype(np.float32)
    actual_prices = complete["price"].values

    # Run inference in batches
    batch_size = 1000
    predictions = np.zeros(len(complete))
    for i in range(0, len(complete), batch_size):
        batch = features[i:i + batch_size]
        pred = session.run(None, {input_name: batch})[0].flatten()
        predictions[i:i + len(pred)] = pred

    predictions = np.clip(predictions, MIN_PRICE, MAX_PRICE)

    # Compute per-sample MAPE (same as validator scoring)
    pct_errors = np.abs(actual_prices - predictions) / actual_prices

    # Get worst predictions, excluding already selected
    error_with_idx = [
        (e, idx) for e, idx in zip(pct_errors, complete.index)
        if idx not in exclude_idx
    ]
    error_with_idx.sort(key=lambda x: -x[0])

    worst = [idx for _, idx in error_with_idx[:num_cases]]

    if worst:
        top_errors = [pct_errors[complete.index.get_loc(idx)] for idx in worst[:10]]
        logger.info(f"  Top 10 model errors: {[f'{e:.1%}' for e in top_errors]}")

    return worst


def find_challenging_realistic(
    df: pd.DataFrame, num_cases: int, rng: np.random.RandomState,
    exclude_idx: set[int],
) -> list[int]:
    """Find realistic but challenging properties — things the validator DOES send
    that are harder to predict accurately.

    Unlike the old edge cases, these are all complete-data properties that could
    appear in any daily evaluation. No zero-living-area or sparse-data cases.
    """
    complete = _filter_complete_properties(df)
    selected = set()

    def _pick(mask_or_idx, label: str, n: int = 8):
        """Pick up to n samples from mask, respecting exclusions."""
        nonlocal selected
        if isinstance(mask_or_idx, (pd.Series, np.ndarray)):
            candidates = complete[mask_or_idx].index.tolist()
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

    # --- Non-single-family homes (validator sends all types) ---
    logger.info("  [1] Non-single-family home types")
    if "home_type_nan" in complete.columns:
        _pick(complete["home_type_nan"] == 1.0, "unknown/condo/townhouse type", n=10)
    if "home_type_MULTI_FAMILY" in complete.columns:
        _pick(complete["home_type_MULTI_FAMILY"] == 1.0, "multi-family", n=8)
    if "home_type_MANUFACTURED" in complete.columns:
        _pick(complete["home_type_MANUFACTURED"] == 1.0, "manufactured home", n=5)

    # --- New construction (no price history, different valuation) ---
    logger.info("  [2] New construction / no sale history")
    if "is_new_construction" in complete.columns:
        _pick(complete["is_new_construction"] == 1.0, "new construction", n=10)
    if "has_previous_sale_data" in complete.columns:
        _pick(complete["has_previous_sale_data"] == 0.0, "no prior sale data", n=8)

    # --- Extreme price/sqft (same structure, very different valuations) ---
    logger.info("  [3] Extreme price per sqft")
    price_per_sqft = complete["price"] / complete["living_area_sqft"]
    _pick(price_per_sqft > 600, "high $/sqft > $600 (urban/luxury)", n=8)
    _pick(price_per_sqft < 80, "low $/sqft < $80 (rural/value)", n=8)

    # --- Cheap properties (high MAPE impact — 10% off on $100K = $10K) ---
    logger.info("  [4] Low-price properties (high MAPE sensitivity)")
    _pick(
        (complete["price"] >= 50_000) & (complete["price"] < 120_000),
        "$50K-$120K (MAPE-sensitive)", n=10,
    )

    # --- Luxury properties ---
    logger.info("  [5] Luxury properties")
    _pick(complete["price"] > 2_000_000, "$2M+ luxury", n=5)
    _pick(
        (complete["price"] > 1_000_000) & (complete["price"] <= 2_000_000),
        "$1M-$2M", n=5,
    )

    # --- Geographic diversity (validator sends from all states) ---
    logger.info("  [6] Geographic edge areas")
    _pick(complete["latitude"] > 45, "northern US (>45 lat)", n=5)
    _pick(complete["latitude"] < 27, "southern US (<27 lat)", n=5)
    _pick(complete["longitude"] > -72, "northeast coast", n=5)
    _pick(complete["longitude"] < -120, "pacific coast", n=5)

    # --- Recent flips (price changed rapidly) ---
    logger.info("  [7] Recent flips / rapid appreciation")
    if "is_recent_flip" in complete.columns:
        _pick(complete["is_recent_flip"] == 1.0, "recent flip", n=5)
    if "annual_appreciation_rate" in complete.columns:
        _pick(complete["annual_appreciation_rate"] > 0.15, "high appreciation >15%/yr", n=5)
        _pick(complete["annual_appreciation_rate"] < -0.05, "depreciation <-5%/yr", n=5)

    # --- Large/unusual structures ---
    logger.info("  [8] Unusual structures")
    _pick(complete["bedrooms"] >= 6, "6+ bedrooms", n=3)
    _pick(complete["living_area_sqft"] > 5000, "5000+ sqft", n=3)
    _pick(
        (complete["living_area_sqft"] > 0) & (complete["living_area_sqft"] < 600),
        "tiny homes <600 sqft", n=5,
    )
    if "property_age" in complete.columns:
        _pick(complete["property_age"] > 100, "100+ years old", n=5)
        _pick(complete["property_age"] <= 1, "brand new (age <=1)", n=3)

    result = list(selected - exclude_idx)
    logger.info(f"  Total challenging realistic: {len(result)}")

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
    num_realistic: int = 600,
    num_error: int = 100,
    num_challenge: int = 300,
    output_path: Path = OUTPUT_PATH,
    seed: int = 42,
):
    """Generate test samples closely matching validator evaluation data.

    Three layers:
      1. Validator-realistic: price-weighted random from complete-data properties
      2. Model errors: complete-data properties where our model predicts worst
      3. Challenging realistic: harder but still realistic (unusual types, prices, geography)
    """
    df = load_dataset(dataset_path)
    rng = np.random.RandomState(seed)

    # Step 1: Validator-realistic (main bulk — mirrors real evaluation)
    logger.info(f"\n=== Selecting {num_realistic} validator-realistic samples ===")
    realistic_idx = select_validator_realistic(df, num_realistic, rng)
    all_selected = set(realistic_idx)
    logger.info(f"Selected {len(realistic_idx)} realistic samples")

    # Step 2: Model error cases (worst predictions on complete data)
    error_idx = []
    if num_error > 0:
        logger.info(f"\n=== Finding {num_error} model error cases ===")
        error_idx = find_model_error_cases(df, model_path, num_error, rng, all_selected)
        all_selected.update(error_idx)
        logger.info(f"Found {len(error_idx)} model error cases")

    # Step 3: Challenging but realistic properties
    challenge_idx = []
    if num_challenge > 0:
        logger.info(f"\n=== Finding {num_challenge} challenging realistic cases ===")
        challenge_idx = find_challenging_realistic(df, num_challenge, rng, all_selected)
        all_selected.update(challenge_idx)
        logger.info(f"Found {len(challenge_idx)} challenging cases")

        # Fill remaining challenge slots from complete data
        if len(challenge_idx) < num_challenge:
            complete = _filter_complete_properties(df)
            remaining = list(set(complete.index) - all_selected)
            if remaining:
                extra = min(num_challenge - len(challenge_idx), len(remaining))
                fill = rng.choice(remaining, size=extra, replace=False).tolist()
                challenge_idx.extend(fill)
                logger.info(f"  Filled {extra} remaining challenge slots")

    # Build output
    all_indices = realistic_idx + error_idx + challenge_idx
    samples = build_sample_json(df, all_indices)

    n_realistic = len(realistic_idx)
    n_error = len(error_idx)
    n_challenge = len(challenge_idx)

    output = {
        "description": (
            f"Validator-realistic test samples for miner CLI local evaluation. "
            f"{n_realistic} realistic + {n_error} model-errors + {n_challenge} challenging "
            f"= {len(samples)} total. "
            f"All samples have complete feature data (79 features, no sparse/zero-area)."
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
    logger.info(f"  Realistic samples:  {n_realistic}")
    logger.info(f"  Model error cases:  {n_error}")
    logger.info(f"  Challenging cases:  {n_challenge}")
    logger.info(f"  Price range:  ${min(prices):,.0f} - ${max(prices):,.0f}")
    logger.info(f"  Median price: ${sorted(prices)[len(prices)//2]:,.0f}")

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
        description="Generate validator-realistic test samples for miner CLI"
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
        "--num-realistic", type=int, default=600,
        help="Number of validator-realistic samples (price-weighted)",
    )
    parser.add_argument(
        "--num-error", type=int, default=100,
        help="Number of model error cases (worst predictions)",
    )
    parser.add_argument(
        "--num-challenge", type=int, default=300,
        help="Number of challenging but realistic cases",
    )
    parser.add_argument(
        "--output", type=Path, default=OUTPUT_PATH,
        help="Output path for test_samples.json",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility",
    )
    args = parser.parse_args()

    generate_test_samples(
        dataset_path=args.dataset,
        model_path=args.model,
        num_realistic=args.num_realistic,
        num_error=args.num_error,
        num_challenge=args.num_challenge,
        output_path=args.output,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
