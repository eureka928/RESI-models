"""
Transform raw Zillow property JSON into the exact 79-feature format validators expect.

Reads raw JSON files from training/raw_data/ and produces a Parquet file
with 79 feature columns (in FEATURE_ORDER) + price column.

Usage:
    python feature_engineer.py
    python feature_engineer.py --input-dir training/raw_data --output training/dataset.parquet
"""

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from config import (
    FEATURE_ORDER,
    HOME_TYPE_CATEGORIES,
    MIN_SCHOOL_DISTANCE_DEFAULT,
    MONTHS_SINCE_LAST_SALE_DEFAULT,
    SCHOOL_RATING_DEFAULT,
    YEARS_SINCE_LAST_SALE_DEFAULT,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
)
logger = logging.getLogger(__name__)

RAW_DATA_DIR = Path(__file__).parent / "raw_data"
DEFAULT_OUTPUT = Path(__file__).parent / "dataset.parquet"

# Current year for property_age calculation
CURRENT_YEAR = datetime.now().year


def safe_float(value, default: float = 0.0) -> float:
    """Safely convert a value to float."""
    if value is None:
        return default
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


def safe_int(value, default: int = 0) -> int:
    """Safely convert a value to int."""
    if value is None:
        return default
    try:
        return int(value)
    except (ValueError, TypeError):
        return default


def count_list(data: dict, key: str) -> float:
    """Count items in a list field, returning 0.0 if missing."""
    val = data.get(key)
    if isinstance(val, list):
        return float(len(val))
    return 0.0


def has_keyword_in_list(data: dict, key: str, keyword: str) -> float:
    """Check if any item in a list field contains a keyword (case-insensitive)."""
    val = data.get(key)
    if isinstance(val, list):
        keyword_lower = keyword.lower()
        for item in val:
            if isinstance(item, str) and keyword_lower in item.lower():
                return 1.0
    return 0.0


def extract_school_features(prop: dict) -> dict:
    """
    Extract school features from nearbySchools array.

    Returns dict with school_count, min_school_distance, avg/max/elementary/middle/high ratings.
    Uses sentinel defaults when no school data is available.
    """
    schools = prop.get("nearbySchools", prop.get("schools", []))
    if not schools or not isinstance(schools, list):
        return {
            "school_count": 0.0,
            "min_school_distance": MIN_SCHOOL_DISTANCE_DEFAULT,
            "avg_school_rating": SCHOOL_RATING_DEFAULT,
            "max_school_rating": SCHOOL_RATING_DEFAULT,
            "elementary_rating": SCHOOL_RATING_DEFAULT,
            "middle_rating": SCHOOL_RATING_DEFAULT,
            "high_rating": SCHOOL_RATING_DEFAULT,
        }

    ratings = []
    distances = []
    elementary_ratings = []
    middle_ratings = []
    high_ratings = []

    for school in schools:
        rating = school.get("rating")
        if rating is not None:
            try:
                r = float(rating)
                ratings.append(r)

                # Categorize by level
                level = str(school.get("level", school.get("grades", ""))).lower()
                link = str(school.get("link", "")).lower()
                name = str(school.get("name", "")).lower()

                is_elementary = any(
                    kw in level or kw in link or kw in name
                    for kw in ["elementary", "primary", "pk-", "k-5", "k-4"]
                )
                is_middle = any(
                    kw in level or kw in link or kw in name
                    for kw in ["middle", "junior", "6-8", "5-8"]
                )
                is_high = any(
                    kw in level or kw in link or kw in name
                    for kw in ["high", "senior", "9-12"]
                )

                if is_elementary:
                    elementary_ratings.append(r)
                if is_middle:
                    middle_ratings.append(r)
                if is_high:
                    high_ratings.append(r)
            except (ValueError, TypeError):
                pass

        distance = school.get("distance")
        if distance is not None:
            try:
                distances.append(float(distance))
            except (ValueError, TypeError):
                pass

    school_count = float(len(schools))
    min_distance = min(distances) if distances else MIN_SCHOOL_DISTANCE_DEFAULT
    avg_rating = np.mean(ratings) if ratings else SCHOOL_RATING_DEFAULT
    max_rating = max(ratings) if ratings else SCHOOL_RATING_DEFAULT
    elem_rating = np.mean(elementary_ratings) if elementary_ratings else SCHOOL_RATING_DEFAULT
    mid_rating = np.mean(middle_ratings) if middle_ratings else SCHOOL_RATING_DEFAULT
    hi_rating = np.mean(high_ratings) if high_ratings else SCHOOL_RATING_DEFAULT

    return {
        "school_count": school_count,
        "min_school_distance": min_distance,
        "avg_school_rating": float(avg_rating),
        "max_school_rating": float(max_rating),
        "elementary_rating": float(elem_rating),
        "middle_rating": float(mid_rating),
        "high_rating": float(hi_rating),
    }


def extract_sale_history(prop: dict) -> dict:
    """
    Extract previous sale data from priceHistory array.

    Returns dict with has_previous_sale_data, previous_sale_price,
    years/months_since_last_sale, price_change_since_last_sale,
    price_appreciation_rate, annual_appreciation_rate, is_recent_flip.
    """
    defaults = {
        "has_previous_sale_data": 0.0,
        "previous_sale_price": 0.0,
        "years_since_last_sale": YEARS_SINCE_LAST_SALE_DEFAULT,
        "months_since_last_sale": MONTHS_SINCE_LAST_SALE_DEFAULT,
        "price_change_since_last_sale": 0.0,
        "price_appreciation_rate": 0.0,
        "annual_appreciation_rate": 0.0,
        "is_recent_flip": 0.0,
    }

    price_history = prop.get("priceHistory", [])
    if not price_history or not isinstance(price_history, list):
        return defaults

    # Find sold events (not listings)
    sold_events = []
    for event in price_history:
        event_type = str(event.get("event", "")).lower()
        if "sold" in event_type:
            price = event.get("price")
            date_str = event.get("date")
            if price and date_str:
                try:
                    sold_events.append({
                        "price": float(price),
                        "date": date_str,
                    })
                except (ValueError, TypeError):
                    pass

    if len(sold_events) < 2:
        # Need at least 2 sold events: current sale + previous sale
        return defaults

    # Sort by date descending
    sold_events.sort(key=lambda x: x["date"], reverse=True)

    current_sale = sold_events[0]
    previous_sale = sold_events[1]

    current_price = current_sale["price"]
    previous_price = previous_sale["price"]

    # Parse dates
    try:
        current_date = datetime.strptime(current_sale["date"][:10], "%Y-%m-%d")
        previous_date = datetime.strptime(previous_sale["date"][:10], "%Y-%m-%d")
    except (ValueError, IndexError):
        return defaults

    delta = current_date - previous_date
    years = delta.days / 365.25
    months = delta.days / 30.44

    if years <= 0 or previous_price <= 0:
        return defaults

    price_change = current_price - previous_price
    appreciation_rate = price_change / previous_price if previous_price > 0 else 0.0
    annual_rate = appreciation_rate / years if years > 0 else 0.0

    # Flip detection: sold within 2 years
    is_flip = 1.0 if years <= 2.0 else 0.0

    return {
        "has_previous_sale_data": 1.0,
        "previous_sale_price": previous_price,
        "years_since_last_sale": years,
        "months_since_last_sale": months,
        "price_change_since_last_sale": price_change,
        "price_appreciation_rate": appreciation_rate,
        "annual_appreciation_rate": annual_rate,
        "is_recent_flip": is_flip,
    }


def extract_features(prop: dict) -> dict | None:
    """
    Extract all 79 features from a raw Zillow property JSON.

    Returns dict mapping feature_name -> float value, or None if
    the property lacks critical fields (price, sqft, lat/lon).
    """
    # Extract the nested property data (API may wrap it)
    if "zpid" not in prop and isinstance(prop.get("data"), dict):
        prop = prop["data"]

    # Get sale price (target variable)
    price = prop.get("price") or prop.get("zestimate")
    if price is None:
        # Try priceHistory for most recent sold price
        ph = prop.get("priceHistory", [])
        for event in (ph if isinstance(ph, list) else []):
            if "sold" in str(event.get("event", "")).lower():
                price = event.get("price")
                if price:
                    break
    if price is None:
        return None

    price = safe_float(price)
    if price <= 0:
        return None

    # Get resoFacts (detailed property attributes)
    reso = prop.get("resoFacts", {}) or {}

    # --- Core numeric features ---
    living_area = safe_float(
        prop.get("livingArea") or reso.get("livingArea")
    )
    lot_size = safe_float(
        prop.get("lotSize") or prop.get("lotAreaValue") or reso.get("lotSize")
    )
    bedrooms = safe_float(
        prop.get("bedrooms") or reso.get("bedrooms")
    )
    bathrooms = safe_float(
        prop.get("bathrooms") or reso.get("bathrooms")
    )
    latitude = safe_float(prop.get("latitude"))
    longitude = safe_float(prop.get("longitude"))
    year_built = safe_float(
        prop.get("yearBuilt") or reso.get("yearBuilt")
    )

    # Skip if missing critical data
    if living_area <= 0 or latitude == 0 or longitude == 0:
        return None

    property_age = float(CURRENT_YEAR - year_built) if year_built > 0 else 0.0
    stories = safe_float(reso.get("stories", prop.get("stories")))

    # --- Bathroom detail ---
    bathrooms_full = safe_float(reso.get("bathroomsFull"))
    bathrooms_half = safe_float(reso.get("bathroomsHalf"))
    bathrooms_three_quarter = safe_float(reso.get("bathroomsThreeQuarter"))

    # --- Parking ---
    garage_capacity = safe_float(reso.get("garageSpaces", reso.get("garageParkingCapacity")))
    covered_parking = safe_float(reso.get("coveredSpaces", reso.get("coveredParkingCapacity")))
    open_parking = safe_float(reso.get("openParkingSpaces", reso.get("openParkingCapacity")))
    parking_total = safe_float(
        reso.get("parkingCapacity")
    ) or (garage_capacity + covered_parking + open_parking)

    # --- Amenity counts (count list items) ---
    fireplaces_count = safe_float(reso.get("fireplaces", 0))

    cooling_count = count_list(reso, "cooling")
    heating_count = count_list(reso, "heating")
    appliances_count = count_list(reso, "appliances")
    flooring_count = count_list(reso, "flooring")
    construction_materials_count = count_list(reso, "constructionMaterials")
    interior_features_count = count_list(reso, "interiorFeatures")
    exterior_features_count = count_list(reso, "exteriorFeatures")
    community_features_count = count_list(reso, "communityFeatures")
    parking_features_count = count_list(reso, "parkingFeatures")
    pool_features_count = count_list(reso, "poolFeatures")
    laundry_features_count = count_list(reso, "laundryFeatures")
    lot_features_count = count_list(reso, "lotFeatures")
    view_features_count = count_list(reso, "viewDescription") if isinstance(
        reso.get("viewDescription"), list
    ) else (1.0 if reso.get("viewDescription") else 0.0)
    sewer_count = count_list(reso, "sewer")
    water_source_count = count_list(reso, "waterSource")
    electric_count = count_list(reso, "electric")

    # --- School features ---
    school_feats = extract_school_features(prop)

    # --- Derived aggregates ---
    total_parking = safe_float(reso.get("parkingCapacity")) or (
        garage_capacity + covered_parking + open_parking
    )
    total_bathrooms = bathrooms_full + 0.5 * bathrooms_half + 0.75 * bathrooms_three_quarter
    if total_bathrooms == 0:
        total_bathrooms = bathrooms  # fallback

    lot_to_living_ratio = lot_size / living_area if living_area > 0 else 0.0
    beds_per_bath = bedrooms / bathrooms if bathrooms > 0 else 0.0

    total_amenity_count = sum([
        cooling_count, heating_count, appliances_count, flooring_count,
        construction_materials_count, interior_features_count,
        exterior_features_count, community_features_count,
        parking_features_count, pool_features_count,
        laundry_features_count, lot_features_count,
    ])

    # --- Sale history ---
    sale_feats = extract_sale_history(prop)

    # --- Boolean flags ---
    has_basement = 1.0 if reso.get("hasBasement") or (
        isinstance(reso.get("basement"), list) and len(reso["basement"]) > 0
    ) else 0.0

    has_garage = 1.0 if garage_capacity > 0 or reso.get("hasGarage") else 0.0
    has_attached_garage = 1.0 if reso.get("hasAttachedGarage") or has_keyword_in_list(
        reso, "parkingFeatures", "attached"
    ) else 0.0

    has_cooling = 1.0 if cooling_count > 0 or reso.get("hasCooling") else 0.0
    has_heating = 1.0 if heating_count > 0 or reso.get("hasHeating") else 0.0
    has_fireplace = 1.0 if fireplaces_count > 0 or reso.get("hasFireplace") else 0.0
    has_spa = 1.0 if reso.get("hasSpa") or has_keyword_in_list(
        reso, "poolFeatures", "spa"
    ) else 0.0
    has_view = 1.0 if reso.get("hasView") or view_features_count > 0 else 0.0
    has_pool = 1.0 if reso.get("hasPool") or pool_features_count > 0 else 0.0
    has_open_parking = 1.0 if open_parking > 0 or reso.get("hasOpenParking") else 0.0
    has_home_warranty = 1.0 if reso.get("hasHomeWarranty") else 0.0
    is_new_construction = 1.0 if prop.get("isNewConstruction") or reso.get(
        "isNewConstruction"
    ) else 0.0
    is_senior_community = 1.0 if reso.get("isSeniorCommunity") else 0.0
    has_waterfront_view = 1.0 if reso.get("hasWaterfront") or has_keyword_in_list(
        reso, "viewDescription", "water"
    ) else 0.0

    has_central_air = has_keyword_in_list(reso, "cooling", "central")
    has_forced_air_heating = has_keyword_in_list(reso, "heating", "forced")
    has_natural_gas = 1.0 if has_keyword_in_list(
        reso, "heating", "natural gas"
    ) or has_keyword_in_list(reso, "electric", "natural gas") else 0.0
    has_hardwood_floors = has_keyword_in_list(reso, "flooring", "hardwood")
    has_tile_floors = has_keyword_in_list(reso, "flooring", "tile")
    has_any_pool_or_spa = 1.0 if has_pool or has_spa else 0.0

    # --- Home type one-hot ---
    home_type = prop.get("homeType", "")
    home_type_flags = {}
    matched = False
    for cat in HOME_TYPE_CATEGORIES:
        if home_type == cat:
            home_type_flags[f"home_type_{cat}"] = 1.0
            matched = True
        else:
            home_type_flags[f"home_type_{cat}"] = 0.0
    home_type_flags["home_type_nan"] = 0.0 if matched or home_type else 1.0

    # --- Build feature dict ---
    features = {
        "living_area_sqft": living_area,
        "lot_size_sqft": lot_size,
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "latitude": latitude,
        "longitude": longitude,
        "year_built": year_built,
        "property_age": property_age,
        "stories": stories,
        "bathrooms_full": bathrooms_full,
        "bathrooms_half": bathrooms_half,
        "bathrooms_three_quarter": bathrooms_three_quarter,
        "garage_capacity": garage_capacity,
        "covered_parking_capacity": covered_parking,
        "open_parking_capacity": open_parking,
        "parking_capacity_total": parking_total,
        "fireplaces_count": fireplaces_count,
        "cooling_count": cooling_count,
        "heating_count": heating_count,
        "appliances_count": appliances_count,
        "flooring_count": flooring_count,
        "construction_materials_count": construction_materials_count,
        "interior_features_count": interior_features_count,
        "exterior_features_count": exterior_features_count,
        "community_features_count": community_features_count,
        "parking_features_count": parking_features_count,
        "pool_features_count": pool_features_count,
        "laundry_features_count": laundry_features_count,
        "lot_features_count": lot_features_count,
        "view_features_count": view_features_count,
        "sewer_count": sewer_count,
        "water_source_count": water_source_count,
        "electric_count": electric_count,
        **school_feats,
        "total_parking": total_parking,
        "total_bathrooms": total_bathrooms,
        "lot_to_living_ratio": lot_to_living_ratio,
        "beds_per_bath": beds_per_bath,
        "total_amenity_count": total_amenity_count,
        "previous_sale_price": sale_feats["previous_sale_price"],
        "years_since_last_sale": sale_feats["years_since_last_sale"],
        "months_since_last_sale": sale_feats["months_since_last_sale"],
        "price_change_since_last_sale": sale_feats["price_change_since_last_sale"],
        "price_appreciation_rate": sale_feats["price_appreciation_rate"],
        "annual_appreciation_rate": sale_feats["annual_appreciation_rate"],
        "has_basement": has_basement,
        "has_garage": has_garage,
        "has_attached_garage": has_attached_garage,
        "has_cooling": has_cooling,
        "has_heating": has_heating,
        "has_fireplace": has_fireplace,
        "has_spa": has_spa,
        "has_view": has_view,
        "has_pool": has_pool,
        "has_open_parking": has_open_parking,
        "has_home_warranty": has_home_warranty,
        "is_new_construction": is_new_construction,
        "is_senior_community": is_senior_community,
        "has_waterfront_view": has_waterfront_view,
        "has_central_air": has_central_air,
        "has_forced_air_heating": has_forced_air_heating,
        "has_natural_gas": has_natural_gas,
        "has_hardwood_floors": has_hardwood_floors,
        "has_tile_floors": has_tile_floors,
        "has_any_pool_or_spa": has_any_pool_or_spa,
        "has_previous_sale_data": sale_feats["has_previous_sale_data"],
        "is_recent_flip": sale_feats["is_recent_flip"],
        **home_type_flags,
    }

    # Validate all 79 features are present
    missing = set(FEATURE_ORDER) - set(features.keys())
    if missing:
        logger.warning(f"Missing features: {missing}")
        return None

    # Return features + price
    features["price"] = price
    return features


def process_raw_data(
    input_dir: Path, output_path: Path, min_price: float = 10000.0
) -> pd.DataFrame:
    """
    Process all raw JSON files into a feature DataFrame.

    Args:
        input_dir: Directory containing raw {zpid}.json files
        output_path: Path to save the output Parquet file
        min_price: Minimum sale price filter

    Returns:
        DataFrame with 79 feature columns + price column
    """
    json_files = list(input_dir.glob("*.json"))
    logger.info(f"Processing {len(json_files)} raw property files...")

    records = []
    errors = 0
    skipped = 0

    for json_path in tqdm(json_files, desc="Extracting features"):
        try:
            with open(json_path) as f:
                raw = json.load(f)
        except (json.JSONDecodeError, OSError):
            errors += 1
            continue

        features = extract_features(raw)
        if features is None:
            skipped += 1
            continue

        if features["price"] < min_price:
            skipped += 1
            continue

        records.append(features)

    logger.info(
        f"Extracted {len(records)} properties "
        f"(skipped {skipped}, errors {errors})"
    )

    if not records:
        logger.error("No valid records extracted!")
        return pd.DataFrame()

    df = pd.DataFrame(records)

    # Ensure column order matches FEATURE_ORDER + price
    columns = FEATURE_ORDER + ["price"]
    df = df[columns]

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    logger.info(f"Saved dataset to {output_path} ({len(df)} rows, {len(df.columns)} columns)")

    # Print summary stats
    logger.info(f"Price range: ${df['price'].min():,.0f} - ${df['price'].max():,.0f}")
    logger.info(f"Median price: ${df['price'].median():,.0f}")
    logger.info(f"Mean living area: {df['living_area_sqft'].mean():,.0f} sqft")

    return df


def main():
    parser = argparse.ArgumentParser(
        description="Transform raw Zillow JSON to 79-feature format"
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=RAW_DATA_DIR,
        help="Directory containing raw {zpid}.json files",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output Parquet file path",
    )
    parser.add_argument(
        "--min-price",
        type=float,
        default=10000.0,
        help="Minimum sale price filter",
    )
    args = parser.parse_args()

    process_raw_data(args.input_dir, args.output, args.min_price)


if __name__ == "__main__":
    main()
