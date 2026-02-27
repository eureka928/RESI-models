"""
End-to-end training pipeline orchestrator for RESI miner models.

Runs the full sequence:
1. Collect geographic data (ZHVI + Census ACS + Redfin, build surfaces)
2. Collect NCES school data (free, ~100K US public schools)
3. Search-only collection (2,500 API calls → ~100K property summaries)
4. Select zpids for detail collection (price-stratified sampling)
5. Detail collection (7,500 API calls → 7,500 fully detailed properties)
6. Engineer features (both sources → 79-feature Parquet)
7. Train 4-model stacking ensemble (2 LightGBM + XGBoost + CatBoost + Ridge meta)
8. Export to ONNX with baked-in geographic lookups (34 geo features)
9. Validate the exported model

API Budget (Pro plan $25/month = 10,000 calls):
  - Search: 2,500 calls × ~40 results = ~100K properties (basic features)
  - Detail: 7,500 calls × 1 result = 7,500 properties (full 79 features)
  - Total:  10,000 calls → ~107,500 properties

Usage:
    # Full pipeline
    python run_pipeline.py --rapidapi-key YOUR_KEY

    # Skip data collection (use existing data)
    python run_pipeline.py --skip-collect --skip-geo

    # Just collect data
    python run_pipeline.py --collect-only --rapidapi-key YOUR_KEY

    # Quick test with minimal data
    python run_pipeline.py --skip-collect --skip-geo --output-model quick_model.onnx

    # Custom API budget split
    python run_pipeline.py --rapidapi-key YOUR_KEY --search-pages 50 --detail-budget 7500
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


def step_collect_geo() -> None:
    """Step 1: Download ZHVI + Census ACS + Redfin data and build geographic surfaces."""
    logger.info("=" * 60)
    logger.info("STEP 1: Collecting geographic data and building surfaces")
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


def step_collect_schools() -> None:
    """Step 2: Download NCES public school data and build spatial lookup."""
    logger.info("=" * 60)
    logger.info("STEP 2: Collecting NCES school data")
    logger.info("=" * 60)

    from collect_schools import save_school_data

    save_school_data()


def step_collect_search(api_key: str, max_pages: int = 50) -> None:
    """Step 3: Search-only collection across all markets (~2,500 API calls)."""
    logger.info("=" * 60)
    logger.info("STEP 3: Search-only collection (~2,500 API calls)")
    logger.info("=" * 60)

    import asyncio

    from collect_data import collect_all_search_only

    asyncio.run(
        collect_all_search_only(
            api_key=api_key,
            max_pages_per_market=max_pages,
        )
    )


def step_select_details(num_details: int = 7500) -> Path:
    """Step 4: Select zpids for detail collection using price-stratified sampling."""
    logger.info("=" * 60)
    logger.info(f"STEP 4: Selecting {num_details} zpids for detail collection")
    logger.info("=" * 60)

    from collect_data import select_detail_zpids

    return select_detail_zpids(num_details=num_details)


def step_collect_details(api_key: str, zpids_file: Path) -> None:
    """Step 5: Fetch property details for selected zpids (~7,500 API calls)."""
    logger.info("=" * 60)
    logger.info("STEP 5: Detail collection for selected zpids")
    logger.info("=" * 60)

    import asyncio

    from collect_data import collect_detail_zpids

    asyncio.run(
        collect_detail_zpids(
            api_key=api_key,
            zpids_file=zpids_file,
        )
    )


def step_collect_data(api_key: str, num_properties: int) -> None:
    """Legacy step: Collect property data (search + detail for every property)."""
    logger.info("=" * 60)
    logger.info("LEGACY: Collecting property data (search + detail per property)")
    logger.info("=" * 60)

    import asyncio

    from collect_data import collect_all

    asyncio.run(collect_all(api_key=api_key, num_properties=num_properties))


def step_feature_engineer(
    min_price: float = 75000.0, include_search: bool = True
) -> Path:
    """Step 6: Transform raw JSON to 79-feature Parquet."""
    logger.info("=" * 60)
    logger.info("STEP 6: Engineering features from property data")
    logger.info("=" * 60)

    from feature_engineer import (
        DEFAULT_OUTPUT,
        RAW_DATA_DIR,
        SEARCH_DATA_DIR,
        process_raw_data,
    )

    search_dir = SEARCH_DATA_DIR if include_search else None
    df = process_raw_data(
        RAW_DATA_DIR,
        DEFAULT_OUTPUT,
        min_price=min_price,
        search_dir=search_dir,
    )
    if df.empty:
        logger.error("Feature engineering produced no data!")
        sys.exit(1)
    return DEFAULT_OUTPUT


def step_train(dataset_path: Path, skip_geo: bool = False) -> Path:
    """Step 7: Train 5-model stacking ensemble (3 LightGBM + XGBoost + CatBoost + Ridge meta)."""
    logger.info("=" * 60)
    logger.info("STEP 7: Training 5-model stacking ensemble")
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
    """Step 8: Export 4-model stacking ensemble to ONNX with baked-in geo lookups."""
    logger.info("=" * 60)
    logger.info("STEP 8: Exporting stacking ensemble to ONNX")
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
    """Step 9: Final validation with miner-cli style checks."""
    logger.info("=" * 60)
    logger.info("STEP 9: Final validation")
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
        logger.error("Model exceeds 200 MB limit!")
        return

    # Check ONNX validity
    try:
        onnx.checker.check_model(str(model_path))
        logger.info("ONNX format: VALID")
    except Exception as e:
        logger.error(f"ONNX format: INVALID - {e}")
        return

    # Load and check interface
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    inputs = session.get_inputs()
    outputs = session.get_outputs()

    logger.info(
        f"Input: name={inputs[0].name}, shape={inputs[0].shape}, type={inputs[0].type}"
    )
    logger.info(
        f"Output: name={outputs[0].name}, shape={outputs[0].shape}, type={outputs[0].type}"
    )

    if inputs[0].shape[1] != 79:
        logger.error(f"Expected 79 input features, got {inputs[0].shape[1]}")
        return

    # Run inference on test samples
    try:
        # Load test samples from the subnet
        import json

        test_path = (
            Path(__file__).parent.parent
            / "real_estate"
            / "miner_cli"
            / "test_samples.json"
        )
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
                sample_id = sample.get("external_id") or sample.get("zpid", "?")
                actual = actual_prices[i]
                pred = preds[i]
                error = abs(pred - actual) / actual
                logger.info(
                    f"  {sample_id}: actual=${actual:,.0f}, pred=${pred:,.0f}, "
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
        "\n  1. Test locally: uv run miner-cli evaluate --model.path "
        + str(model_path)
        + "\n  2. Submit: uv run miner-cli submit --model.path "
        + str(model_path)
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
        help="Target number of properties (legacy mode only)",
    )
    parser.add_argument(
        "--collect-only",
        action="store_true",
        help="Only run data collection steps",
    )
    parser.add_argument(
        "--skip-collect",
        action="store_true",
        help="Skip property data collection (use existing data)",
    )
    parser.add_argument(
        "--skip-geo",
        action="store_true",
        help="Skip geographic data collection and surface building",
    )
    parser.add_argument(
        "--skip-schools",
        action="store_true",
        help="Skip NCES school data collection",
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
        default=75000.0,
        help="Minimum sale price filter for training data",
    )
    # Hybrid collection budget controls
    parser.add_argument(
        "--search-pages",
        type=int,
        default=50,
        help="Max search result pages per market (50 markets × N pages = API calls)",
    )
    parser.add_argument(
        "--detail-budget",
        type=int,
        default=7500,
        help="Number of zpids to select for detail collection",
    )
    parser.add_argument(
        "--legacy-collect",
        action="store_true",
        help="Use legacy collection mode (search + detail per property)",
    )
    args = parser.parse_args()

    start_time = time.time()

    # Step 1: Collect geographic data
    if not args.skip_geo:
        step_collect_geo()

    # Step 2: Collect school data
    if not args.skip_schools:
        step_collect_schools()

    # Steps 3-5: Property data collection
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

        if args.legacy_collect:
            # Legacy mode: search + detail for every property
            step_collect_data(args.rapidapi_key, args.num_properties)
        else:
            # New hybrid mode: search-heavy + strategic details
            # Step 3: Search-only collection
            step_collect_search(args.rapidapi_key, max_pages=args.search_pages)

            # Step 4: Select zpids for detail collection
            zpids_file = step_select_details(num_details=args.detail_budget)

            # Step 5: Detail collection for selected zpids
            step_collect_details(args.rapidapi_key, zpids_file)

    if args.collect_only:
        logger.info("Collection complete (--collect-only specified)")
        return

    # Step 6: Feature engineering (processes both raw_data/ and search_data/)
    dataset_path = step_feature_engineer(min_price=args.min_price)

    # Step 7: Train models
    model_dir = step_train(dataset_path, skip_geo=args.skip_geo)

    # Step 8: Export to ONNX
    onnx_path = step_export(model_dir, args.output_model, skip_geo=args.skip_geo)

    # Step 9: Validate
    step_validate(onnx_path)

    elapsed = time.time() - start_time
    logger.info(f"\nTotal pipeline time: {elapsed / 60:.1f} minutes")


if __name__ == "__main__":
    main()
