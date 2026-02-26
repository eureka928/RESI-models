"""
Export trained 4-model stacking ensemble to a single ONNX file with baked-in geographic lookups.

The ONNX graph:
1. Accepts (batch, 79) float32 input (standard RESI interface)
2. Extracts lat/lon (indices 4,5) -> computes grid cell index -> looks up geo features
3. Concatenates 79 original features + 34 geo features = 113 total
4. Runs 4 tree ensembles (2 LightGBM + 1 XGBoost + 1 CatBoost)
5. Meta-learner: Ridge linear combination of 4 base predictions
6. expm1 + clip to [$50K, $20M]

Validator constraints:
  - Input: (N, 79) float32, output: (N, 1) float32 in USD
  - NaN/Inf in output → model fails evaluation entirely
  - Max file size: 200 MB
  - Execution: CPUExecutionProvider only, 2GB RAM, 1 CPU, 5 min timeout

Usage:
    python export_onnx.py --model-dir training/models --geo-dir training/geo_surfaces
    python export_onnx.py --model-dir training/models --skip-geo --output model.onnx
"""

import argparse
import logging
import pickle
from pathlib import Path

import lightgbm as lgb
import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper
from onnxmltools.convert import convert_lightgbm
from onnxmltools.convert.common.data_types import FloatTensorType

from config import (
    FEATURE_ORDER,
    G0_NLAT,
    G0_NLON,
    G0_PRECISION,
    G1_NLAT,
    G1_NLON,
    G1_PRECISION,
    GEO_FEATURES_PER_GRID,
    LAT_INDEX,
    LAT_MIN,
    LON_INDEX,
    LON_MIN,
    MAX_PRICE,
    MIN_PRICE,
    NUM_EXTRA_GEO_SURFACES,
    NUM_RP_FEATURES,
    TOTAL_GEO_FEATURES,
)
from geo_features import GeoFeatureLookup

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
)
logger = logging.getLogger(__name__)

DEFAULT_MODEL_DIR = Path(__file__).parent / "models"
DEFAULT_GEO_DIR = Path(__file__).parent / "geo_surfaces"
DEFAULT_OUTPUT = Path(__file__).parent / "model.onnx"

# Surface names in order (must match geo_features.py)
SURFACE_NAMES = (
    ["zhvi", "is_zip"]
    + [f"rp{i}" for i in range(NUM_RP_FEATURES)]
    + ["median_income", "pct_bachelors", "pop_density", "redfin_price", "redfin_dom"]
)


def convert_lgbm_to_onnx(model, n_features: int, name: str) -> onnx.ModelProto:
    """Convert a LightGBM model to ONNX format."""
    # onnxmltools doesn't support 'mape' objective — patch to 'regression'
    # The tree structure is identical; only the loss function differs during training
    booster = model.booster_
    model_str = booster.model_to_string()
    for obj_name in ["mape", "regression_l1"]:
        model_str = model_str.replace(
            f"objective={obj_name}", "objective=regression"
        )
    patched_booster = lgb.Booster(model_str=model_str)
    model._Booster = patched_booster

    initial_type = [("input", FloatTensorType([None, n_features]))]
    return convert_lightgbm(model, initial_types=initial_type, name=name)


def convert_xgb_to_onnx(model, n_features: int) -> onnx.ModelProto:
    """Convert an XGBoost model to ONNX via onnxmltools."""
    from onnxmltools.convert import convert_xgboost
    initial_type = [("input", FloatTensorType([None, n_features]))]
    return convert_xgboost(model, initial_types=initial_type, name="m3")


def convert_catboost_to_onnx(model, n_features: int, output_path: str) -> onnx.ModelProto:
    """Convert a CatBoost model to ONNX via its native export."""
    model.save_model(
        output_path,
        format="onnx",
        export_parameters={"onnx_domain": "ai.onnx.ml", "onnx_model_version": 1},
    )
    return onnx.load(output_path)


def build_geo_lookup_subgraph(
    surfaces: dict[str, dict[str, np.ndarray]],
    input_name: str,
) -> tuple[list, list, list, str]:
    """
    Build ONNX nodes for geographic feature lookups.

    Returns (nodes, initializers, value_infos, output_name)
    """
    nodes = []
    initializers = []
    value_infos = []

    # Constants for lat/lon extraction
    initializers.append(
        numpy_helper.from_array(np.array([LAT_INDEX], dtype=np.int64), name="lat_idx_const")
    )
    initializers.append(
        numpy_helper.from_array(np.array([LON_INDEX], dtype=np.int64), name="lon_idx_const")
    )

    nodes.append(helper.make_node("Gather", [input_name, "lat_idx_const"], ["lat_raw"], axis=1))
    nodes.append(helper.make_node("Gather", [input_name, "lon_idx_const"], ["lon_raw"], axis=1))

    geo_outputs = []

    for grid_prefix, precision, nlat, nlon in [
        ("g0", G0_PRECISION, G0_NLAT, G0_NLON),
        ("g1", G1_PRECISION, G1_NLAT, G1_NLON),
    ]:
        p = grid_prefix
        initializers.extend([
            numpy_helper.from_array(np.array(LAT_MIN, dtype=np.float32), name=f"{p}_lat_min"),
            numpy_helper.from_array(np.array(LON_MIN, dtype=np.float32), name=f"{p}_lon_min"),
            numpy_helper.from_array(np.array(float(precision), dtype=np.float32), name=f"{p}_precision"),
            numpy_helper.from_array(np.array(nlon, dtype=np.int64), name=f"{p}_nlon"),
            numpy_helper.from_array(np.array(0, dtype=np.int64), name=f"{p}_zero"),
            numpy_helper.from_array(np.array(nlat * nlon - 1, dtype=np.int64), name=f"{p}_max_idx"),
        ])

        # lat_idx = floor((lat - lat_min) * precision)
        nodes.append(helper.make_node("Sub", ["lat_raw", f"{p}_lat_min"], [f"{p}_lat_shifted"]))
        nodes.append(helper.make_node("Mul", [f"{p}_lat_shifted", f"{p}_precision"], [f"{p}_lat_scaled"]))
        nodes.append(helper.make_node("Floor", [f"{p}_lat_scaled"], [f"{p}_lat_floored"]))
        nodes.append(helper.make_node("Cast", [f"{p}_lat_floored"], [f"{p}_lat_int"], to=TensorProto.INT64))

        # lon_idx = floor((lon - lon_min) * precision)
        nodes.append(helper.make_node("Sub", ["lon_raw", f"{p}_lon_min"], [f"{p}_lon_shifted"]))
        nodes.append(helper.make_node("Mul", [f"{p}_lon_shifted", f"{p}_precision"], [f"{p}_lon_scaled"]))
        nodes.append(helper.make_node("Floor", [f"{p}_lon_scaled"], [f"{p}_lon_floored"]))
        nodes.append(helper.make_node("Cast", [f"{p}_lon_floored"], [f"{p}_lon_int"], to=TensorProto.INT64))

        # cell_idx = lat_idx * nlon + lon_idx, clamped
        nodes.append(helper.make_node("Mul", [f"{p}_lat_int", f"{p}_nlon"], [f"{p}_lat_x_nlon"]))
        nodes.append(helper.make_node("Add", [f"{p}_lat_x_nlon", f"{p}_lon_int"], [f"{p}_cell_raw"]))
        nodes.append(helper.make_node("Clip", [f"{p}_cell_raw", f"{p}_zero", f"{p}_max_idx"], [f"{p}_cell_idx"]))

        # Lookup each surface
        for sname in SURFACE_NAMES:
            surface_data = surfaces[grid_prefix].get(sname)
            if surface_data is None:
                # Missing surface: use zeros
                expected_size = nlat * nlon
                surface_data = np.zeros(expected_size, dtype=np.float32)

            tensor_name = f"{p}_{sname}_surface"
            output_name = f"{p}_{sname}_val"
            unsqueeze_name = f"{output_name}_unsq"

            initializers.append(numpy_helper.from_array(surface_data.astype(np.float32), name=tensor_name))
            nodes.append(helper.make_node("Gather", [tensor_name, f"{p}_cell_idx"], [output_name], axis=0))

            initializers.append(numpy_helper.from_array(np.array([1], dtype=np.int64), name=f"{output_name}_axis"))
            nodes.append(helper.make_node("Unsqueeze", [output_name, f"{output_name}_axis"], [unsqueeze_name]))
            geo_outputs.append(unsqueeze_name)

    concat_output = "geo_features_concat"
    nodes.append(helper.make_node("Concat", geo_outputs, [concat_output], axis=1))

    return nodes, initializers, value_infos, concat_output


def extract_tree_node(onnx_model: onnx.ModelProto, input_name: str, output_name: str):
    """Extract TreeEnsembleRegressor node from an ONNX model and rewire it."""
    for node in onnx_model.graph.node:
        if "TreeEnsemble" in node.op_type:
            new_node = helper.make_node(
                node.op_type,
                inputs=[input_name],
                outputs=[output_name],
                domain=node.domain,
            )
            new_node.attribute.extend(node.attribute)
            return new_node
    raise ValueError(f"No TreeEnsembleRegressor found in ONNX model for {output_name}")


def build_ensemble_graph(
    m1_onnx: onnx.ModelProto,
    m2_onnx: onnx.ModelProto,
    m3_onnx: onnx.ModelProto,
    m4_onnx: onnx.ModelProto,
    meta_coef: np.ndarray,
    meta_intercept: float,
    geo_surfaces: dict | None,
    n_base_features: int = 79,
) -> onnx.ModelProto:
    """Build the complete ONNX graph."""
    nodes = []
    initializers = []
    value_infos = []

    input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, [None, n_base_features])

    has_geo = geo_surfaces is not None

    if has_geo:
        geo_nodes, geo_inits, geo_vis, geo_output = build_geo_lookup_subgraph(geo_surfaces, "input")
        nodes.extend(geo_nodes)
        initializers.extend(geo_inits)
        value_infos.extend(geo_vis)
        nodes.append(helper.make_node("Concat", ["input", geo_output], ["augmented_input"], axis=1))
        tree_input = "augmented_input"
    else:
        tree_input = "input"

    # Extract tree ensemble nodes from each model
    nodes.append(extract_tree_node(m1_onnx, tree_input, "m1_raw"))
    nodes.append(extract_tree_node(m2_onnx, tree_input, "m2_raw"))
    nodes.append(extract_tree_node(m3_onnx, tree_input, "m3_raw"))
    nodes.append(extract_tree_node(m4_onnx, tree_input, "m4_raw"))

    # Reshape all to (batch, 1)
    initializers.append(numpy_helper.from_array(np.array([-1, 1], dtype=np.int64), name="reshape_shape"))
    for name in ["m1", "m2", "m3", "m4"]:
        nodes.append(helper.make_node("Reshape", [f"{name}_raw", "reshape_shape"], [f"{name}_pred"]))

    # Stack to (batch, 4)
    nodes.append(helper.make_node("Concat", ["m1_pred", "m2_pred", "m3_pred", "m4_pred"], ["base_preds"], axis=1))

    # Meta-learner: output = base_preds @ coef + intercept
    initializers.append(
        numpy_helper.from_array(meta_coef.astype(np.float32).reshape(4, 1), name="meta_coef")
    )
    initializers.append(
        numpy_helper.from_array(np.array([[meta_intercept]], dtype=np.float32), name="meta_intercept")
    )

    # MatMul: (batch, 4) @ (4, 1) -> (batch, 1)
    nodes.append(helper.make_node("MatMul", ["base_preds", "meta_coef"], ["meta_prod"]))
    nodes.append(helper.make_node("Add", ["meta_prod", "meta_intercept"], ["log_pred"]))

    # expm1: exp(x) - 1
    nodes.append(helper.make_node("Exp", ["log_pred"], ["exp_pred"]))
    initializers.append(numpy_helper.from_array(np.array(1.0, dtype=np.float32), name="one_const"))
    nodes.append(helper.make_node("Sub", ["exp_pred", "one_const"], ["expm1_pred"]))

    # Clip to [MIN_PRICE, MAX_PRICE]
    initializers.append(numpy_helper.from_array(np.array(MIN_PRICE, dtype=np.float32), name="min_price"))
    initializers.append(numpy_helper.from_array(np.array(MAX_PRICE, dtype=np.float32), name="max_price"))
    nodes.append(helper.make_node("Clip", ["expm1_pred", "min_price", "max_price"], ["output"]))

    output_tensor = helper.make_tensor_value_info("output", TensorProto.FLOAT, [None, 1])

    graph = helper.make_graph(
        nodes, "resi_stacking_ensemble",
        [input_tensor], [output_tensor],
        initializer=initializers, value_info=value_infos,
    )

    opset_imports = [
        helper.make_opsetid("", 17),
        helper.make_opsetid("ai.onnx.ml", 3),
    ]

    model = helper.make_model(graph, opset_imports=opset_imports)
    model.ir_version = 8
    return model


def validate_onnx_model(
    model_path: Path,
    models: dict,
    meta,
    geo_lookup: GeoFeatureLookup | None,
    n_samples: int = 10,
) -> bool:
    """Validate the exported ONNX model against Python predictions."""
    logger.info("Validating ONNX model...")
    all_ok = True

    # 1. Check model validity
    try:
        onnx.checker.check_model(str(model_path))
        logger.info("  [PASS] onnx.checker.check_model()")
    except Exception as e:
        logger.error(f"  [FAIL] onnx.checker: {e}")
        all_ok = False

    # 2. Load and check interface
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    input_info = session.get_inputs()[0]
    output_info = session.get_outputs()[0]

    if input_info.shape[1] == 79:
        logger.info(f"  [PASS] Input shape: {input_info.shape}")
    else:
        logger.error(f"  [FAIL] Input shape: {input_info.shape}, expected (None, 79)")
        all_ok = False

    logger.info(f"  [INFO] Output shape: {output_info.shape}")

    # 3. Compare predictions
    np.random.seed(42)
    test_input = np.random.randn(n_samples, 79).astype(np.float32)
    test_input[:, LAT_INDEX] = np.random.uniform(25, 48, n_samples)
    test_input[:, LON_INDEX] = np.random.uniform(-124, -67, n_samples)
    test_input[:, 0] = np.random.uniform(500, 5000, n_samples)
    test_input[:, 2] = np.random.uniform(1, 6, n_samples)
    test_input[:, 3] = np.random.uniform(1, 4, n_samples)
    test_input[:, 6] = np.random.uniform(1950, 2024, n_samples)

    onnx_preds = session.run(None, {input_info.name: test_input})[0].flatten()

    # Python prediction
    if geo_lookup is not None:
        geo_feats = geo_lookup.lookup(test_input[:, LAT_INDEX], test_input[:, LON_INDEX])
        augmented = np.hstack([test_input, geo_feats])
    else:
        augmented = test_input

    base_preds = np.column_stack([
        models["m1"].predict(augmented),
        models["m2"].predict(augmented),
        models["m3"].predict(augmented),
        models["m4"].predict(augmented),
    ])
    meta_pred = meta.predict(base_preds)
    python_preds = np.clip(np.expm1(meta_pred), MIN_PRICE, MAX_PRICE)

    max_diff = np.max(np.abs(onnx_preds - python_preds))
    rel_diff = np.max(np.abs(onnx_preds - python_preds) / np.maximum(python_preds, 1))

    if rel_diff < 0.01:
        logger.info(f"  [PASS] Max relative diff: {rel_diff:.6f}, absolute: ${max_diff:.2f}")
    else:
        logger.warning(f"  [WARN] Max relative diff: {rel_diff:.4f}, absolute: ${max_diff:.2f}")

    # 4. NaN/Inf check
    if np.any(np.isnan(onnx_preds)) or np.any(np.isinf(onnx_preds)):
        logger.error("  [FAIL] ONNX output contains NaN/Inf")
        all_ok = False
    else:
        logger.info("  [PASS] No NaN/Inf in predictions")

    # 5. File size
    size_mb = model_path.stat().st_size / (1024 * 1024)
    if size_mb <= 200:
        logger.info(f"  [PASS] File size: {size_mb:.1f} MB")
    else:
        logger.error(f"  [FAIL] File size: {size_mb:.1f} MB (exceeds 200 MB)")
        all_ok = False

    logger.info(f"\n  Sample predictions (first 5):")
    for i in range(min(5, n_samples)):
        logger.info(f"    ONNX: ${onnx_preds[i]:,.0f}  Python: ${python_preds[i]:,.0f}")

    return all_ok


def export_onnx(
    model_dir: Path = DEFAULT_MODEL_DIR,
    geo_dir: Path | None = DEFAULT_GEO_DIR,
    output_path: Path = DEFAULT_OUTPUT,
) -> bool:
    """Export the trained stacking ensemble to a single ONNX file."""
    # Load trained models
    logger.info("Loading trained models...")
    models = {}
    for name in ["m1", "m2", "m3", "m4"]:
        with open(model_dir / f"{name}.pkl", "rb") as f:
            models[name] = pickle.load(f)

    with open(model_dir / "meta.pkl", "rb") as f:
        meta = pickle.load(f)

    with open(model_dir / "metadata.pkl", "rb") as f:
        metadata = pickle.load(f)

    n_features = metadata["n_features"]
    has_geo = metadata["has_geo"]
    meta_coef = np.array(metadata["meta_coef"])
    meta_intercept = metadata["meta_intercept"]

    logger.info(f"Models loaded: {n_features} features, has_geo={has_geo}")
    logger.info(f"Meta-learner: coef={meta_coef}, intercept={meta_intercept:.6f}")

    # Load geo surfaces
    geo_surfaces = None
    geo_lookup = None
    if has_geo and geo_dir and Path(geo_dir).exists():
        geo_lookup = GeoFeatureLookup(geo_dir)
        geo_surfaces = geo_lookup.get_surface_arrays()
        logger.info("Loaded geographic surfaces for ONNX baking")

    # Convert base models to ONNX
    logger.info("Converting m1 (LightGBM) to ONNX...")
    m1_onnx = convert_lgbm_to_onnx(models["m1"], n_features, "m1")

    logger.info("Converting m2 (LightGBM) to ONNX...")
    m2_onnx = convert_lgbm_to_onnx(models["m2"], n_features, "m2")

    logger.info("Converting m3 (XGBoost) to ONNX...")
    m3_onnx = convert_xgb_to_onnx(models["m3"], n_features)

    logger.info("Converting m4 (CatBoost) to ONNX...")
    tmp_cb_path = str(model_dir / "m4_temp.onnx")
    m4_onnx = convert_catboost_to_onnx(models["m4"], n_features, tmp_cb_path)

    # Build combined graph
    logger.info("Building stacking ensemble ONNX graph...")
    combined_model = build_ensemble_graph(
        m1_onnx, m2_onnx, m3_onnx, m4_onnx,
        meta_coef, meta_intercept,
        geo_surfaces, n_base_features=79,
    )

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save_model(combined_model, str(output_path))
    size_mb = output_path.stat().st_size / (1024 * 1024)
    logger.info(f"Saved ONNX model to {output_path} ({size_mb:.1f} MB)")

    # Clean up temp file
    tmp_path = Path(tmp_cb_path)
    if tmp_path.exists():
        tmp_path.unlink()

    # Validate
    all_ok = validate_onnx_model(output_path, models, meta, geo_lookup)

    if all_ok:
        logger.info("\nAll validation checks passed!")
    else:
        logger.warning("\nSome validation checks failed — review warnings above")

    return all_ok


def main():
    parser = argparse.ArgumentParser(
        description="Export stacking ensemble to ONNX with geo lookups"
    )
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--geo-dir", type=Path, default=DEFAULT_GEO_DIR)
    parser.add_argument("--skip-geo", action="store_true")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    geo_dir = None if args.skip_geo else args.geo_dir
    export_onnx(model_dir=args.model_dir, geo_dir=geo_dir, output_path=args.output)


if __name__ == "__main__":
    main()
