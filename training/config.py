"""
Constants, feature definitions, and configuration for the RESI training pipeline.

The FEATURE_ORDER list defines the exact 79-feature interface that validators expect.
All models must accept a (batch, 79) float32 input in this exact order.
"""

import os
from pathlib import Path

# --- Load .env file if present ---
_env_path = Path(__file__).parent / ".env"
if _env_path.exists():
    with open(_env_path) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _key, _val = _line.split("=", 1)
                os.environ.setdefault(_key.strip(), _val.strip())

# --- API Configuration ---

RAPIDAPI_KEY = os.environ.get("RAPIDAPI_KEY", "")
RAPIDAPI_HOST = os.environ.get(
    "RAPIDAPI_HOST", "real-time-real-estate-data.p.rapidapi.com"
)

# --- Feature Order (79 features, matches feature_config.yaml exactly) ---

FEATURE_ORDER = [
    # Numeric features (51)
    "living_area_sqft",
    "lot_size_sqft",
    "bedrooms",
    "bathrooms",
    "latitude",           # index 4
    "longitude",          # index 5
    "year_built",
    "property_age",
    "stories",
    "bathrooms_full",
    "bathrooms_half",
    "bathrooms_three_quarter",
    "garage_capacity",
    "covered_parking_capacity",
    "open_parking_capacity",
    "parking_capacity_total",
    "fireplaces_count",
    "cooling_count",
    "heating_count",
    "appliances_count",
    "flooring_count",
    "construction_materials_count",
    "interior_features_count",
    "exterior_features_count",
    "community_features_count",
    "parking_features_count",
    "pool_features_count",
    "laundry_features_count",
    "lot_features_count",
    "view_features_count",
    "sewer_count",
    "water_source_count",
    "electric_count",
    "school_count",
    "min_school_distance",
    "avg_school_rating",
    "max_school_rating",
    "elementary_rating",
    "middle_rating",
    "high_rating",
    "total_parking",
    "total_bathrooms",
    "lot_to_living_ratio",
    "beds_per_bath",
    "total_amenity_count",
    "previous_sale_price",
    "years_since_last_sale",
    "months_since_last_sale",
    "price_change_since_last_sale",
    "price_appreciation_rate",
    "annual_appreciation_rate",
    # Boolean features (22)
    "has_basement",
    "has_garage",
    "has_attached_garage",
    "has_cooling",
    "has_heating",
    "has_fireplace",
    "has_spa",
    "has_view",
    "has_pool",
    "has_open_parking",
    "has_home_warranty",
    "is_new_construction",
    "is_senior_community",
    "has_waterfront_view",
    "has_central_air",
    "has_forced_air_heating",
    "has_natural_gas",
    "has_hardwood_floors",
    "has_tile_floors",
    "has_any_pool_or_spa",
    "has_previous_sale_data",
    "is_recent_flip",
    # Home type one-hot (6)
    "home_type_SINGLE_FAMILY",
    "home_type_MULTI_FAMILY",
    "home_type_MANUFACTURED",
    "home_type_LOT",
    "home_type_HOME_TYPE_UNKNOWN",
    "home_type_nan",
]

assert len(FEATURE_ORDER) == 79, f"Expected 79 features, got {len(FEATURE_ORDER)}"

LAT_INDEX = FEATURE_ORDER.index("latitude")   # 4
LON_INDEX = FEATURE_ORDER.index("longitude")  # 5

# --- Sentinel Defaults (must match validator encoding exactly) ---

SCHOOL_RATING_DEFAULT = 5.5
MIN_SCHOOL_DISTANCE_DEFAULT = 1.2000000476837158  # float32 precision
YEARS_SINCE_LAST_SALE_DEFAULT = 12.0
MONTHS_SINCE_LAST_SALE_DEFAULT = 144.0

# --- Price Clipping ---
# CRITICAL: Validator has NO price floor (max_pct_error=None, no clipping).
# If validation data includes a $70K property and we clip to $100K, that's
# a 43% error on ONE sample. With ~100 daily samples, +0.43% MAPE — enough
# to lose when the winner threshold is only 1%.
# $50K floor: catches cheap rural properties/lots while still protecting
# against absurd negative predictions on normal homes.
MIN_PRICE = 50_000.0
MAX_PRICE = 20_000_000.0

# --- Geographic Grid Configuration ---

# Grid covering continental US
LAT_MIN = 24.0
LAT_MAX = 50.0
LON_MIN = -125.0
LON_MAX = -66.0

# Coarse grid (g0): ~0.1 degree resolution (~11km)
G0_PRECISION = 10  # cells per degree
G0_NLAT = int((LAT_MAX - LAT_MIN) * G0_PRECISION)   # 260
G0_NLON = int((LON_MAX - LON_MIN) * G0_PRECISION)   # 590

# Fine grid (g1): ~0.033 degree resolution (~3.7km)
G1_PRECISION = 30  # cells per degree
G1_NLAT = int((LAT_MAX - LAT_MIN) * G1_PRECISION)   # 780
G1_NLON = int((LON_MAX - LON_MIN) * G1_PRECISION)   # 1770

# Number of regional price features per grid resolution
NUM_RP_FEATURES = 10

# Extra geo surfaces beyond base (zhvi + is_zip + rp0-rp9):
# median_income, pct_bachelors, population_density, redfin_median_price,
# redfin_dom (days on market)
NUM_EXTRA_GEO_SURFACES = 5

# Total geo features per resolution: zhvi + is_zip + rp0-rp9 + 5 extra = 17
GEO_FEATURES_PER_GRID = 1 + 1 + NUM_RP_FEATURES + NUM_EXTRA_GEO_SURFACES  # 17
# Total geo features: 17 * 2 grids = 34
TOTAL_GEO_FEATURES = GEO_FEATURES_PER_GRID * 2  # 34

# --- Target Markets for Data Collection ---

TARGET_MARKETS = [
    # Top 50 US metros by population for geographic diversity
    "New York, NY", "Los Angeles, CA", "Chicago, IL", "Houston, TX",
    "Phoenix, AZ", "Philadelphia, PA", "San Antonio, TX", "San Diego, CA",
    "Dallas, TX", "Austin, TX", "Jacksonville, FL", "San Jose, CA",
    "Fort Worth, TX", "Columbus, OH", "Charlotte, NC", "Indianapolis, IN",
    "San Francisco, CA", "Seattle, WA", "Denver, CO", "Nashville, TN",
    "Oklahoma City, OK", "Washington, DC", "El Paso, TX", "Las Vegas, NV",
    "Boston, MA", "Portland, OR", "Memphis, TN", "Louisville, KY",
    "Baltimore, MD", "Milwaukee, WI", "Albuquerque, NM", "Tucson, AZ",
    "Fresno, CA", "Mesa, AZ", "Sacramento, CA", "Atlanta, GA",
    "Kansas City, MO", "Omaha, NE", "Colorado Springs, CO", "Raleigh, NC",
    "Miami, FL", "Tampa, FL", "Minneapolis, MN", "Cleveland, OH",
    "Detroit, MI", "Pittsburgh, PA", "Cincinnati, OH", "Orlando, FL",
    "St. Louis, MO", "Richmond, VA",
]

# --- LightGBM Hyperparameters ---
#
# OBJECTIVE RATIONALE (aligned with validator MAPE = mean(|actual-pred|/actual)):
#
# We train on log1p(price) targets. The key insight:
#   MAE on log(price) = mean(|log(pred) - log(actual)|)
#                     ≈ mean(|pred - actual| / actual)  for small errors
#                     = price-space MAPE
#
# So MAE ("regression_l1") on log-space is the CORRECT proxy for price-space MAPE.
#
# MAPE on log-space = mean(|log(pred)-log(actual)| / log(actual)) — this
# over-weights cheap properties by 1/log(price) factor (~28% more weight on
# $50K vs $1M properties), which DIFFERS from the validator's equal weighting.
#
# Strategy: m1+m3 use MAE (correct proxy), m2+m4 use MAPE (diversity, slight
# cheap-property bias that may help). Ridge meta-learner blends optimally.

PARAMS_M1 = {
    "objective": "regression_l1",  # MAE on log-space ≈ price-space MAPE
    "metric": "mape",              # early-stop on what validator actually scores
    "boosting_type": "gbdt",
    "num_leaves": 255,
    "learning_rate": 0.03,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "min_child_samples": 20,
    "n_estimators": 5000,
    "verbose": -1,
}

PARAMS_M2 = {
    "objective": "mape",           # diversity: log-space MAPE (cheap-property bias)
    "metric": "mape",
    "boosting_type": "gbdt",
    "num_leaves": 127,
    "learning_rate": 0.05,
    "feature_fraction": 0.7,
    "bagging_fraction": 0.7,
    "bagging_freq": 3,
    "min_child_samples": 30,
    "n_estimators": 4000,
    "verbose": -1,
}

# --- XGBoost Hyperparameters ---

PARAMS_M3 = {
    "objective": "reg:absoluteerror",  # MAE on log-space ≈ price-space MAPE
    "eval_metric": "mape",
    "tree_method": "hist",
    "max_depth": 8,
    "learning_rate": 0.03,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 20,
    "n_estimators": 4000,
    "verbosity": 0,
}

# --- CatBoost Hyperparameters ---

PARAMS_M4 = {
    "loss_function": "MAPE",       # diversity: log-space MAPE
    "eval_metric": "MAPE",
    "depth": 8,
    "learning_rate": 0.03,
    "iterations": 3000,
    "l2_leaf_reg": 3.0,
    "subsample": 0.8,
    "random_strength": 1.0,
    "verbose": 0,
}

# --- Ensemble Configuration ---

# Number of base models in the stacking ensemble
NUM_BASE_MODELS = 4  # m1 (lgbm), m2 (lgbm), m3 (xgb), m4 (catboost)

# --- Home Type Mapping ---

HOME_TYPE_CATEGORIES = [
    "SINGLE_FAMILY",
    "MULTI_FAMILY",
    "MANUFACTURED",
    "LOT",
    "HOME_TYPE_UNKNOWN",
]
