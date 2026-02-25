"""
End-to-end training pipeline orchestrator for RESI miner models.

Runs the full sequence:
1. Collect property data from Zillow (via RapidAPI - Real-Time Zillow Data)
2. Collect geographic data (ZHVI + Census ACS + Redfin, build surfaces)
3. Engineer features (raw JSON -> 79-feature Parquet)
4. Train 4-model stacking ensemble (2 LightGBM + XGBoost + CatBoost + Ridge meta)
5. Export to ONNX with baked-in geographic lookups (34 geo features)
6. Validate the exported model

Usage:
    # Full pipeline
    python run_pipeline.py --rapidapi-key YOUR_KEY

    # Skip data collection (use existing data)
    python run_pipeline.py --skip-collect --skip-geo

    # Just collect data
    python run_pipeline.py --collect-only --rapidapi-key YOUR_KEY

    # Quick test with minimal data
    python run_pipeline.py --skip-collect --skip-geo --output-model quick_model.onnx
"""

import argparse
import logging
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("pipeline")


def step_collect_data(api_key: str, num_properties: int) -> None:
    """Step 1: Collect property data from Zillow API."""
    logger.info("=" * 60)
    logger.info("STEP 1: Collecting property data from Zillow API")
    logger.info("=" * 60)

    import asyncio

    from collect_data import collect_all

    asyncio.run(
        collect_all(api_key=api_key, num_properties=num_properties)
    )


def step_collect_geo() -> None:
    """Step 2: Download ZHVI + Census ACS + Redfin data and build geographic surfaces."""
    logger.info("=" * 60)
    logger.info("STEP 2: Collecting geographic data and building surfaces")
    logger.info("=" * 60)

    from collect_geo_data import (
        GEO_DATA_DIR,
        GEO_SURFACES_DIR,
        build_and_save_all_surfaces,
        build_zip_zhvi_lookup,
        download_census_acs,
        download_redfin_zip_data,
        download_zcta_centroids,
        download_zhvi,
        enrich_zip_data,
    )

    zhvi_df = download_zhvi(GEO_DATA_DIR)
    zcta_df = download_zcta_centroids(GEO_DATA_DIR)
    census_df = download_census_acs(GEO_DATA_DIR)
    redfin_df = download_redfin_zip_data(GEO_DATA_DIR)
    zip_data = build_zip_zhvi_lookup(zhvi_df, zcta_df)
    zip_data = enrich_zip_data(zip_data, census_df, redfin_df)
    build_and_save_all_surfaces(zip_data, GEO_SURFACES_DIR)


def step_feature_engineer(min_price: float = 10000.0) -> Path:
    """Step 3: Transform raw JSON to 79-feature Parquet."""
    logger.info("=" * 60)
    logger.info("STEP 3: Engineering features from raw property data")
    logger.info("=" * 60)

    from feature_engineer import DEFAULT_OUTPUT, RAW_DATA_DIR, process_raw_data

    df = process_raw_data(RAW_DATA_DIR, DEFAULT_OUTPUT, min_price=min_price)
    if df.empty:
        logger.error("Feature engineering produced no data!")
        sys.exit(1)
    return DEFAULT_OUTPUT


def step_train(
    dataset_path: Path, skip_geo: bool = False
) -> Path:
    """Step 4: Train 4-model stacking ensemble (2 LightGBM + XGBoost + CatBoost + Ridge meta)."""
    logger.info("=" * 60)
    logger.info("STEP 4: Training 4-model stacking ensemble")
    logger.info("=" * 60)

    from train_model import DEFAULT_GEO_DIR, DEFAULT_MODEL_DIR, train_ensemble

    geo_dir = None if skip_geo else DEFAULT_GEO_DIR

    models, meta, feature_names, geo_lookup = train_ensemble(
        dataset_path=dataset_path,
        geo_dir=geo_dir,
        output_dir=DEFAULT_MODEL_DIR,
    )
    logger.info(f"Trained {len(models)} base models + Ridge meta-learner")
    logger.info(f"Meta-learner weights: {meta.coef_}")
    return DEFAULT_MODEL_DIR


def step_export(model_dir: Path, output_path: Path, skip_geo: bool = False) -> Path:
    """Step 5: Export 4-model stacking ensemble to ONNX with baked-in geo lookups."""
    logger.info("=" * 60)
    logger.info("STEP 5: Exporting stacking ensemble to ONNX")
    logger.info("=" * 60)

    from export_onnx import DEFAULT_GEO_DIR, export_onnx

    geo_dir = None if skip_geo else DEFAULT_GEO_DIR

    ok = export_onnx(
        model_dir=model_dir,
        geo_dir=geo_dir,
        output_path=output_path,
    )

    if not ok:
        logger.warning("ONNX validation had warnings — check output above")

    return output_path


def step_validate(model_path: Path) -> None:
    """Step 6: Final validation with miner-cli style checks."""
    logger.info("=" * 60)
    logger.info("STEP 6: Final validation")
    logger.info("=" * 60)

    import numpy as np
    import onnx
    import onnxruntime as ort

    # Check file exists and size
    if not model_path.exists():
        logger.error(f"Model file not found: {model_path}")
        return

    size_mb = model_path.stat().st_size / (1024 * 1024)
    logger.info(f"Model size: {size_mb:.1f} MB")

    if size_mb > 200:
        logger.error(f"Model exceeds 200 MB limit!")
        return

    # Check ONNX validity
    try:
        onnx.checker.check_model(str(model_path))
        logger.info("ONNX format: VALID")
    except Exception as e:
        logger.error(f"ONNX format: INVALID - {e}")
        return

    # Load and check interface
    session = ort.InferenceSession(
        str(model_path), providers=["CPUExecutionProvider"]
    )
    inputs = session.get_inputs()
    outputs = session.get_outputs()

    logger.info(f"Input: name={inputs[0].name}, shape={inputs[0].shape}, type={inputs[0].type}")
    logger.info(f"Output: name={outputs[0].name}, shape={outputs[0].shape}, type={outputs[0].type}")

    if inputs[0].shape[1] != 79:
        logger.error(f"Expected 79 input features, got {inputs[0].shape[1]}")
        return

    # Run inference on test samples
    try:
        # Load test samples from the subnet
        import json

        test_path = Path(__file__).parent.parent / "real_estate" / "miner_cli" / "test_samples.json"
        if test_path.exists():
            with open(test_path) as f:
                test_data = json.load(f)

            from config import FEATURE_ORDER

            features_list = []
            prices = []
            for sample in test_data["samples"]:
                feats = [float(sample["features"].get(f, 0.0)) for f in FEATURE_ORDER]
                features_list.append(feats)
                prices.append(sample["actual_price"])

            features_arr = np.array(features_list, dtype=np.float32)
            actual_prices = np.array(prices)

            preds = session.run(None, {inputs[0].name: features_arr})[0].flatten()

            logger.info("\nTest sample predictions:")
            for i, sample in enumerate(test_data["samples"]):
                zpid = sample["zpid"]
                actual = actual_prices[i]
                pred = preds[i]
                error = abs(pred - actual) / actual
                logger.info(
                    f"  zpid {zpid}: actual=${actual:,.0f}, pred=${pred:,.0f}, "
                    f"error={error:.2%}"
                )

            mape = np.mean(np.abs(actual_prices - preds) / actual_prices)
            score = max(0.0, 1.0 - mape)
            logger.info(f"\nTest MAPE: {mape:.4f}")
            logger.info(f"Test Score: {score:.4f}")
        else:
            logger.info("Test samples not found, skipping test evaluation")

    except Exception as e:
        logger.warning(f"Test evaluation failed: {e}")

    logger.info("\nValidation complete!")
    logger.info(f"Model ready at: {model_path.absolute()}")
    logger.info(
        "\nNext steps:"
        "\n  1. Test locally: uv run miner-cli evaluate --model.path " + str(model_path)
        + "\n  2. Submit: uv run miner-cli submit --model.path " + str(model_path)
        + " --hf.repo_id YOUR/REPO --wallet.name miner"
    )


def main():
    parser = argparse.ArgumentParser(
        description="RESI training pipeline - from raw data to competitive ONNX model"
    )
    parser.add_argument(
        "--rapidapi-key",
        default="",
        help="RapidAPI key for Zillow data collection",
    )
    parser.add_argument(
        "--num-properties",
        type=int,
        default=50000,
        help="Target number of properties to collect",
    )
    parser.add_argument(
        "--collect-only",
        action="store_true",
        help="Only run data collection steps",
    )
    parser.add_argument(
        "--skip-collect",
        action="store_true",
        help="Skip property data collection (use existing raw_data/)",
    )
    parser.add_argument(
        "--skip-geo",
        action="store_true",
        help="Skip geographic data collection and surface building",
    )
    parser.add_argument(
        "--output-model",
        type=Path,
        default=Path(__file__).parent / "model.onnx",
        help="Output ONNX model path",
    )
    parser.add_argument(
        "--min-price",
        type=float,
        default=10000.0,
        help="Minimum sale price filter for training data",
    )
    args = parser.parse_args()

    start_time = time.time()

    # Step 1: Collect property data
    if not args.skip_collect:
        if not args.rapidapi_key:
            import os
            args.rapidapi_key = os.environ.get("RAPIDAPI_KEY", "")
        if not args.rapidapi_key:
            logger.error(
                "No RapidAPI key provided. Use --rapidapi-key or set RAPIDAPI_KEY env var.\n"
                "Use --skip-collect to use existing data in training/raw_data/"
            )
            sys.exit(1)
        step_collect_data(args.rapidapi_key, args.num_properties)

    # Step 2: Collect geographic data
    if not args.skip_geo:
        step_collect_geo()

    if args.collect_only:
        logger.info("Collection complete (--collect-only specified)")
        return

    # Step 3: Feature engineering
    dataset_path = step_feature_engineer(min_price=args.min_price)

    # Step 4: Train models
    model_dir = step_train(dataset_path, skip_geo=args.skip_geo)

    # Step 5: Export to ONNX
    onnx_path = step_export(model_dir, args.output_model, skip_geo=args.skip_geo)

    # Step 6: Validate
    step_validate(onnx_path)

    elapsed = time.time() - start_time
    logger.info(f"\nTotal pipeline time: {elapsed / 60:.1f} minutes")


if __name__ == "__main__":
    main()
