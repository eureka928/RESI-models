"""
Generate diverse test samples for miner-cli local evaluation.

Extracts 50 real properties from dataset.parquet, stratified by price range
to ensure coverage across cheap, mid-range, and expensive properties.

Usage:
    python generate_test_samples.py
    python generate_test_samples.py --dataset dataset.parquet --num-samples 50
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from config import FEATURE_ORDER, MIN_PRICE, MAX_PRICE

DEFAULT_DATASET = Path(__file__).parent / "dataset.parquet"
OUTPUT_PATH = Path(__file__).parent.parent / "real_estate" / "miner_cli" / "test_samples.json"


def generate_test_samples(
    dataset_path: Path = DEFAULT_DATASET,
    num_samples: int = 50,
    output_path: Path = OUTPUT_PATH,
    seed: int = 123,
):
    """Extract stratified test samples from training data."""
    print(f"Loading dataset from {dataset_path}...")
    df = pd.read_parquet(dataset_path)
    print(f"Loaded {len(df)} samples")

    # Filter to valid price range
    df = df[(df["price"] >= MIN_PRICE) & (df["price"] <= MAX_PRICE)].copy()
    print(f"After price filtering: {len(df)} samples")

    # Stratify by price buckets for diverse coverage
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

    rng = np.random.RandomState(seed)
    samples_per_bucket = max(1, num_samples // len(buckets))
    remainder = num_samples - samples_per_bucket * len(buckets)

    selected_indices = []
    for name, mask in buckets:
        bucket_df = df[mask]
        n_take = samples_per_bucket
        # Give extra samples to mid-range buckets (most common in validator data)
        if remainder > 0 and name in ("$200K-$300K", "$300K-$500K", "$500K-$750K"):
            n_take += 1
            remainder -= 1

        if len(bucket_df) == 0:
            print(f"  {name}: 0 available, skipping")
            continue

        n_take = min(n_take, len(bucket_df))
        idx = rng.choice(bucket_df.index, size=n_take, replace=False)
        selected_indices.extend(idx)
        print(f"  {name}: selected {n_take} samples (from {len(bucket_df)} available)")

    # If we still need more (some buckets were empty), fill from all data
    if len(selected_indices) < num_samples:
        remaining = num_samples - len(selected_indices)
        available = df.index.difference(selected_indices)
        extra = rng.choice(available, size=min(remaining, len(available)), replace=False)
        selected_indices.extend(extra)
        print(f"  Filled {len(extra)} extra samples from full dataset")

    selected = df.loc[selected_indices]
    print(f"\nTotal selected: {len(selected)} samples")
    print(f"Price range: ${selected['price'].min():,.0f} - ${selected['price'].max():,.0f}")
    print(f"Median price: ${selected['price'].median():,.0f}")

    # Build JSON output
    samples = []
    for _, row in selected.iterrows():
        features = {}
        for feat in FEATURE_ORDER:
            val = float(row[feat]) if feat in row and pd.notna(row[feat]) else 0.0
            # Clean up float representation
            if val == int(val) and abs(val) < 1e10:
                val = float(int(val))
            features[feat] = val

        sample = {
            "external_id": str(int(row.get("zpid", row.name))),
            "actual_price": float(row["price"]),
            "features": features,
        }
        samples.append(sample)

    # Sort by price for readability
    samples.sort(key=lambda s: s["actual_price"])

    output = {
        "description": f"Test samples for miner CLI local evaluation. {len(samples)} properties stratified by price range. Features match feature_config.yaml order (79 features).",
        "samples": samples,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nSaved {len(samples)} test samples to {output_path}")
    print(f"File size: {output_path.stat().st_size / 1024:.1f} KB")

    # Summary by price bucket
    prices = [s["actual_price"] for s in samples]
    print(f"\nPrice distribution of test samples:")
    for name, lo, hi in [
        ("Under $100K", 0, 100_000),
        ("$100K-$300K", 100_000, 300_000),
        ("$300K-$500K", 300_000, 500_000),
        ("$500K-$1M", 500_000, 1_000_000),
        ("Over $1M", 1_000_000, float("inf")),
    ]:
        count = sum(1 for p in prices if lo <= p < hi)
        print(f"  {name}: {count}")


def main():
    parser = argparse.ArgumentParser(description="Generate test samples for miner CLI")
    parser.add_argument(
        "--dataset", type=Path, default=DEFAULT_DATASET,
        help="Path to training Parquet file",
    )
    parser.add_argument(
        "--num-samples", type=int, default=50,
        help="Number of test samples to generate",
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
        num_samples=args.num_samples,
        output_path=args.output,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
