"""
Train a 5-model stacking ensemble for RESI price prediction.

Architecture (beats top model's 2-LightGBM approach):
  Level 0 (base models):
    - m1: LightGBM (127 leaves, MAE obj) — high capacity, correct MAPE proxy
    - m2: LightGBM (63 leaves, MAPE obj) — regularized, diversity
    - m3: XGBoost  (depth 6, MAE obj)    — different inductive bias
    - m4: CatBoost (depth 6, MAPE obj)   — ordered boosting, diversity
    - m5: LightGBM (127 leaves, Huber)   — outlier-robust, reduces tail risk
  Level 1 (meta-learner):
    - RidgeCV on out-of-fold predictions → optimal blending weights

All predict in log1p(price) space. Final: expm1(meta_pred), clipped to [$50K, $20M].

Validator alignment:
  - Score = max(0, 1 - MAPE) where MAPE = mean(|actual - pred| / actual)
  - NO error capping (max_pct_error=None) — one bad prediction destroys score
  - Winner set = within 0.01 of best score, then earliest commit wins
  - MAE on log-space ≈ price-space MAPE (see config.py for math)

Usage:
    python train_model.py --dataset training/dataset.parquet --geo-dir training/geo_surfaces
    python train_model.py --dataset training/dataset.parquet --skip-geo
"""

import argparse
import logging
import pickle
from pathlib import Path

import catboost as cb
import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold

from config import (
    FEATURE_ORDER,
    MAX_PRICE,
    MIN_PRICE,
    NUM_BASE_MODELS,
    PARAMS_M1,
    PARAMS_M2,
    PARAMS_M3,
    PARAMS_M4,
    PARAMS_M5,
)
from geo_features import GeoFeatureLookup

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
)
logger = logging.getLogger(__name__)

DEFAULT_DATASET = Path(__file__).parent / "dataset.parquet"
DEFAULT_GEO_DIR = Path(__file__).parent / "geo_surfaces"
DEFAULT_MODEL_DIR = Path(__file__).parent / "models"


def compute_mape(y_true_log: np.ndarray, y_pred_log: np.ndarray) -> float:
    """MAPE in price space from log-space predictions."""
    actual = np.expm1(y_true_log)
    pred = np.clip(np.expm1(y_pred_log), MIN_PRICE, MAX_PRICE)
    return float(np.mean(np.abs(actual - pred) / actual))


def load_and_prepare_data(
    dataset_path: Path,
    geo_dir: Path | None = None,
    val_fraction: float = 0.2,
) -> tuple:
    """
    Load dataset and split into train/validation.

    Returns (X_train, y_train, X_val, y_val, feature_names, geo_lookup)
    """
    logger.info(f"Loading dataset from {dataset_path}...")
    df = pd.read_parquet(dataset_path)
    logger.info(f"Loaded {len(df)} samples with {len(df.columns)} columns")

    # Outlier filtering: remove obviously bad data points
    n_before = len(df)
    living_area = df["living_area_sqft"]
    price = df["price"]
    price_per_sqft = price / living_area.replace(0, np.nan)

    bad_mask = (
        (price_per_sqft.notna() & ((price_per_sqft < 25) | (price_per_sqft > 1500)))
        | ((living_area > 0) & ((living_area < 300) | (living_area > 15000)))
        | (
            (df["bedrooms"] == 0)
            & (df["bathrooms"] == 0)
            & (living_area == 0)
        )
    )
    df = df[~bad_mask].reset_index(drop=True)
    n_removed = n_before - len(df)
    if n_removed > 0:
        logger.info(f"Outlier filtering: removed {n_removed} bad samples ({n_removed/n_before:.1%})")

    # Target: log1p(price)
    y = np.log1p(df["price"].values)

    # Features: 79 base features
    X = df[FEATURE_ORDER].copy()
    feature_names = list(FEATURE_ORDER)

    # Add geographic features if available
    geo_lookup = None
    if geo_dir and Path(geo_dir).exists():
        logger.info("Augmenting with geographic features...")
        geo_lookup = GeoFeatureLookup(geo_dir)
        X = geo_lookup.augment_dataframe(X)
        feature_names = list(X.columns)
        logger.info(f"Total features after geo augmentation: {len(feature_names)}")

    X = X.values.astype(np.float32)

    # Random split (data may be ordered by market/source)
    rng = np.random.RandomState(42)
    indices = rng.permutation(len(X))
    split_idx = int(len(X) * (1 - val_fraction))

    train_idx, val_idx = indices[:split_idx], indices[split_idx:]
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    # Compute sample weights: gentle inverse-price weighting (~2:1 cheap vs expensive)
    # Power=0.25 gives gentler ratio than sqrt (power=0.5) since validator scores
    # all properties equally by MAPE
    train_prices = np.expm1(y_train)
    sample_weight = 1.0 / np.power(np.maximum(train_prices, 1.0), 0.25)
    sample_weight = sample_weight / sample_weight.mean()  # normalize to mean=1

    logger.info(f"Train: {len(X_train)} samples, Val: {len(X_val)} samples")
    logger.info(
        f"Train price range: ${np.expm1(y_train.min()):,.0f} - ${np.expm1(y_train.max()):,.0f}"
    )
    logger.info(
        f"Val price range: ${np.expm1(y_val.min()):,.0f} - ${np.expm1(y_val.max()):,.0f}"
    )
    logger.info(
        f"Sample weight range: {sample_weight.min():.3f} - {sample_weight.max():.3f}"
    )

    return X_train, y_train, X_val, y_val, feature_names, geo_lookup, sample_weight


def train_lgbm(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    params: dict, feature_names: list[str], name: str,
    sample_weight: np.ndarray | None = None,
) -> lgb.LGBMRegressor:
    """Train a single LightGBM model."""
    logger.info(f"Training {name} (LightGBM)...")
    fit_params = dict(params)
    n_estimators = fit_params.pop("n_estimators", 5000)

    model = lgb.LGBMRegressor(n_estimators=n_estimators, **fit_params)
    model.fit(
        X_train, y_train,
        sample_weight=sample_weight,
        eval_set=[(X_val, y_val)],
        eval_metric="mape",
        callbacks=[
            lgb.early_stopping(stopping_rounds=100),
            lgb.log_evaluation(period=200),
        ],
        feature_name=feature_names,
    )

    val_mape = compute_mape(y_val, model.predict(X_val))
    logger.info(f"  {name} Val MAPE: {val_mape:.4f} (Score: {1-val_mape:.4f}), best_iter={model.best_iteration_}")
    return model


def train_xgboost(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    params: dict, feature_names: list[str], name: str,
    sample_weight: np.ndarray | None = None,
) -> xgb.XGBRegressor:
    """Train a single XGBoost model."""
    logger.info(f"Training {name} (XGBoost)...")
    fit_params = dict(params)
    n_estimators = fit_params.pop("n_estimators", 4000)

    model = xgb.XGBRegressor(
        n_estimators=n_estimators,
        early_stopping_rounds=100,
        feature_names=feature_names,
        **fit_params,
    )
    model.fit(
        X_train, y_train,
        sample_weight=sample_weight,
        eval_set=[(X_val, y_val)],
        verbose=200,
    )

    val_mape = compute_mape(y_val, model.predict(X_val))
    logger.info(f"  {name} Val MAPE: {val_mape:.4f} (Score: {1-val_mape:.4f}), best_iter={model.best_iteration}")
    return model


def train_catboost(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    params: dict, feature_names: list[str], name: str,
    sample_weight: np.ndarray | None = None,
) -> cb.CatBoostRegressor:
    """Train a single CatBoost model."""
    logger.info(f"Training {name} (CatBoost)...")
    fit_params = dict(params)

    model = cb.CatBoostRegressor(
        random_seed=42,
        **fit_params,
    )
    pool_train = cb.Pool(X_train, y_train, feature_names=feature_names, weight=sample_weight)
    pool_val = cb.Pool(X_val, y_val, feature_names=feature_names)
    model.fit(
        pool_train,
        eval_set=pool_val,
        early_stopping_rounds=100,
    )

    val_mape = compute_mape(y_val, model.predict(X_val))
    logger.info(f"  {name} Val MAPE: {val_mape:.4f} (Score: {1-val_mape:.4f}), best_iter={model.best_iteration_}")
    return model


def generate_oof_predictions(
    X_train: np.ndarray, y_train: np.ndarray,
    feature_names: list[str], sample_weight: np.ndarray | None = None,
    n_folds: int = 5,
) -> tuple[np.ndarray, list]:
    """
    Generate out-of-fold predictions for stacking.

    Trains each base model on k-1 folds and predicts the held-out fold.
    Returns (oof_preds of shape (n_train, NUM_BASE_MODELS), list of trained models per fold).
    """
    logger.info(f"\nGenerating out-of-fold predictions ({n_folds} folds)...")
    n = len(X_train)
    oof_preds = np.zeros((n, NUM_BASE_MODELS), dtype=np.float64)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    fold_models = []

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_train)):
        logger.info(f"\n--- Fold {fold_idx + 1}/{n_folds} ---")
        X_tr, X_vl = X_train[train_idx], X_train[val_idx]
        y_tr, y_vl = y_train[train_idx], y_train[val_idx]
        sw_tr = sample_weight[train_idx] if sample_weight is not None else None

        models = {}

        # m1: LightGBM high-capacity
        m1 = train_lgbm(X_tr, y_tr, X_vl, y_vl, PARAMS_M1, feature_names, f"m1_fold{fold_idx}", sample_weight=sw_tr)
        oof_preds[val_idx, 0] = m1.predict(X_vl)
        models["m1"] = m1

        # m2: LightGBM regularized
        m2 = train_lgbm(X_tr, y_tr, X_vl, y_vl, PARAMS_M2, feature_names, f"m2_fold{fold_idx}", sample_weight=sw_tr)
        oof_preds[val_idx, 1] = m2.predict(X_vl)
        models["m2"] = m2

        # m3: XGBoost
        m3 = train_xgboost(X_tr, y_tr, X_vl, y_vl, PARAMS_M3, feature_names, f"m3_fold{fold_idx}", sample_weight=sw_tr)
        oof_preds[val_idx, 2] = m3.predict(X_vl)
        models["m3"] = m3

        # m4: CatBoost
        m4 = train_catboost(X_tr, y_tr, X_vl, y_vl, PARAMS_M4, feature_names, f"m4_fold{fold_idx}", sample_weight=sw_tr)
        oof_preds[val_idx, 3] = m4.predict(X_vl)
        models["m4"] = m4

        # m5: LightGBM with Huber loss (outlier-robust)
        m5 = train_lgbm(X_tr, y_tr, X_vl, y_vl, PARAMS_M5, feature_names, f"m5_fold{fold_idx}", sample_weight=sw_tr)
        oof_preds[val_idx, 4] = m5.predict(X_vl)
        models["m5"] = m5

        fold_models.append(models)

    # Report OOF performance per model
    for i, name in enumerate(["m1(lgbm)", "m2(lgbm)", "m3(xgb)", "m4(cat)", "m5(huber)"]):
        mape = compute_mape(y_train, oof_preds[:, i])
        logger.info(f"  OOF {name}: MAPE={mape:.4f}, Score={1-mape:.4f}")

    return oof_preds, fold_models


def train_meta_learner(
    oof_preds: np.ndarray, y_train: np.ndarray
) -> RidgeCV:
    """Train RidgeCV meta-learner on out-of-fold predictions."""
    logger.info("\nTraining meta-learner (RidgeCV) on OOF predictions...")
    meta = RidgeCV(
        alphas=np.logspace(-1, 1.5, 20),  # 0.1 to ~31.6, 20 points
        fit_intercept=True,
        cv=5,
    )
    meta.fit(oof_preds, y_train)

    meta_pred = meta.predict(oof_preds)
    meta_mape = compute_mape(y_train, meta_pred)

    logger.info(f"  Meta-learner OOF MAPE: {meta_mape:.4f} (Score: {1-meta_mape:.4f})")
    logger.info(f"  Meta-learner best alpha: {meta.alpha_}")
    logger.info(f"  Meta-learner weights: {meta.coef_}")
    logger.info(f"  Meta-learner intercept: {meta.intercept_:.6f}")

    return meta


def train_final_models(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    feature_names: list[str],
    sample_weight: np.ndarray | None = None,
) -> dict:
    """Train final base models on full training set."""
    logger.info("\n=== Training final models on full training set ===")

    models = {}
    models["m1"] = train_lgbm(X_train, y_train, X_val, y_val, PARAMS_M1, feature_names, "m1_final", sample_weight=sample_weight)
    models["m2"] = train_lgbm(X_train, y_train, X_val, y_val, PARAMS_M2, feature_names, "m2_final", sample_weight=sample_weight)
    models["m3"] = train_xgboost(X_train, y_train, X_val, y_val, PARAMS_M3, feature_names, "m3_final", sample_weight=sample_weight)
    models["m4"] = train_catboost(X_train, y_train, X_val, y_val, PARAMS_M4, feature_names, "m4_final", sample_weight=sample_weight)
    models["m5"] = train_lgbm(X_train, y_train, X_val, y_val, PARAMS_M5, feature_names, "m5_final", sample_weight=sample_weight)

    return models


def evaluate_stacking_ensemble(
    models: dict, meta: RidgeCV,
    X: np.ndarray, y: np.ndarray, label: str,
) -> float:
    """
    Evaluate the full stacking ensemble using validator-exact MAPE formula.

    Validator computes: MAPE = mean(|y_true - y_pred| / y_true), no error capping.
    One outlier prediction can dominate MAPE, so we report tail risk metrics.
    """
    base_preds = np.column_stack([
        models["m1"].predict(X),
        models["m2"].predict(X),
        models["m3"].predict(X),
        models["m4"].predict(X),
        models["m5"].predict(X),
    ])
    meta_pred = meta.predict(base_preds)

    actual = np.expm1(y)
    pred = np.clip(np.expm1(meta_pred), MIN_PRICE, MAX_PRICE)

    # Per-property percentage errors (validator-exact formula)
    pct_errors = np.abs(actual - pred) / actual

    # Core metrics (what the validator uses)
    mape = float(np.mean(pct_errors))
    mdape = float(np.median(pct_errors))
    within_5 = float(np.mean(pct_errors < 0.05))
    within_10 = float(np.mean(pct_errors < 0.10))
    within_15 = float(np.mean(pct_errors < 0.15))

    # Tail risk (validator has NO error capping — these kill your score)
    p95_error = float(np.percentile(pct_errors, 95))
    p99_error = float(np.percentile(pct_errors, 99))
    max_error = float(np.max(pct_errors))
    n_over_50pct = int(np.sum(pct_errors > 0.50))
    n_over_100pct = int(np.sum(pct_errors > 1.00))

    logger.info(f"\n{label} Stacking Ensemble (validator-exact MAPE):")
    logger.info(f"  MAPE:  {mape:.4f}  (Score: {1-mape:.4f})")
    logger.info(f"  MdAPE: {mdape:.4f}")
    logger.info(f"  Accuracy@5%: {within_5:.2%}, @10%: {within_10:.2%}, @15%: {within_15:.2%}")
    logger.info(f"  --- Tail Risk (no error capping in validator) ---")
    logger.info(f"  P95 error: {p95_error:.2%}, P99 error: {p99_error:.2%}, Max: {max_error:.2%}")
    logger.info(f"  Properties >50% error: {n_over_50pct}, >100% error: {n_over_100pct}")

    # Worst 5 predictions (these dominate MAPE)
    if len(pct_errors) > 0:
        worst_idx = np.argsort(pct_errors)[-5:][::-1]
        logger.info(f"  --- Worst 5 predictions ---")
        for idx in worst_idx:
            logger.info(
                f"    actual=${actual[idx]:,.0f}, pred=${pred[idx]:,.0f}, "
                f"error={pct_errors[idx]:.2%}"
            )

    # MAPE by price bucket (find vulnerable segments)
    buckets = [
        ("Under $100K", actual < 100_000),
        ("$100K-$300K", (actual >= 100_000) & (actual < 300_000)),
        ("$300K-$500K", (actual >= 300_000) & (actual < 500_000)),
        ("$500K-$1M", (actual >= 500_000) & (actual < 1_000_000)),
        ("Over $1M", actual >= 1_000_000),
    ]
    bucket_strs = []
    for bname, mask in buckets:
        if mask.sum() > 0:
            bucket_mape = float(np.mean(pct_errors[mask]))
            bucket_strs.append(f"{bname}: {bucket_mape:.2%} (n={mask.sum()})")
    if bucket_strs:
        logger.info(f"  --- MAPE by price range ---")
        for s in bucket_strs:
            logger.info(f"    {s}")

    return mape


def train_ensemble(
    dataset_path: Path,
    geo_dir: Path | None = None,
    output_dir: Path = DEFAULT_MODEL_DIR,
    val_fraction: float = 0.2,
) -> tuple[dict, RidgeCV, list[str], GeoFeatureLookup | None]:
    """
    Train the full 5-model stacking ensemble.

    Returns (models_dict, meta_learner, feature_names, geo_lookup)
    """
    X_train, y_train, X_val, y_val, feature_names, geo_lookup, sample_weight = load_and_prepare_data(
        dataset_path, geo_dir, val_fraction
    )

    # Step 1: Generate out-of-fold predictions for meta-learner training
    oof_preds, _ = generate_oof_predictions(X_train, y_train, feature_names, sample_weight=sample_weight)

    # Step 2: Train meta-learner on OOF predictions
    meta = train_meta_learner(oof_preds, y_train)

    # Step 3: Train final base models on full training set
    models = train_final_models(X_train, y_train, X_val, y_val, feature_names, sample_weight=sample_weight)

    # Step 4: Evaluate
    logger.info("\n" + "=" * 60)
    evaluate_stacking_ensemble(models, meta, X_train, y_train, "Train")
    val_mape = evaluate_stacking_ensemble(models, meta, X_val, y_val, "Val")

    # Feature importance (from m1)
    logger.info("\n=== Top 20 Feature Importances (m1) ===")
    importance = models["m1"].feature_importances_
    sorted_idx = np.argsort(importance)[::-1][:20]
    for i, idx in enumerate(sorted_idx):
        logger.info(f"  {i+1}. {feature_names[idx]}: {importance[idx]}")

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)

    for name, model in models.items():
        with open(output_dir / f"{name}.pkl", "wb") as f:
            pickle.dump(model, f)

    with open(output_dir / "meta.pkl", "wb") as f:
        pickle.dump(meta, f)

    with open(output_dir / "metadata.pkl", "wb") as f:
        pickle.dump({
            "feature_names": feature_names,
            "val_mape": val_mape,
            "has_geo": geo_lookup is not None,
            "n_features": len(feature_names),
            "meta_coef": meta.coef_.tolist(),
            "meta_intercept": float(meta.intercept_),
            "num_base_models": NUM_BASE_MODELS,
        }, f)

    logger.info(f"\nModels saved to {output_dir}")

    return models, meta, feature_names, geo_lookup


def main():
    parser = argparse.ArgumentParser(
        description="Train 4-model stacking ensemble for RESI"
    )
    parser.add_argument(
        "--dataset", type=Path, default=DEFAULT_DATASET,
        help="Path to training Parquet file",
    )
    parser.add_argument(
        "--geo-dir", type=Path, default=DEFAULT_GEO_DIR,
        help="Directory containing geographic surface .npy files",
    )
    parser.add_argument(
        "--skip-geo", action="store_true",
        help="Skip geographic feature augmentation",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_MODEL_DIR,
        help="Directory to save trained models",
    )
    parser.add_argument(
        "--val-fraction", type=float, default=0.2,
        help="Fraction of data for validation",
    )
    args = parser.parse_args()

    geo_dir = None if args.skip_geo else args.geo_dir

    train_ensemble(
        dataset_path=args.dataset,
        geo_dir=geo_dir,
        output_dir=args.output_dir,
        val_fraction=args.val_fraction,
    )


if __name__ == "__main__":
    main()
