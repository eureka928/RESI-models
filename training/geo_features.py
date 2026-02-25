"""
Geographic feature lookup for training and ONNX export.

Loads pre-built geographic surfaces (.npy files from collect_geo_data.py)
and provides functions to:
1. Augment training data with geographic features (lat/lon -> lookup values)
2. Load surface arrays for baking into ONNX graphs

Surfaces per grid resolution (17 each, 34 total):
  zhvi, is_zip, rp0-rp9, median_income, pct_bachelors,
  pop_density, redfin_price, redfin_dom

Usage in training:
    from geo_features import GeoFeatureLookup
    geo = GeoFeatureLookup("training/geo_surfaces")
    augmented_df = geo.augment_dataframe(df)  # adds 34 geo columns
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from config import (
    G0_NLAT,
    G0_NLON,
    G0_PRECISION,
    G1_NLAT,
    G1_NLON,
    G1_PRECISION,
    GEO_FEATURES_PER_GRID,
    LAT_MIN,
    LON_MIN,
    NUM_RP_FEATURES,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
)
logger = logging.getLogger(__name__)

# Surface names in canonical order (must match collect_geo_data.py output)
SURFACE_NAMES = (
    ["zhvi", "is_zip"]
    + [f"rp{i}" for i in range(NUM_RP_FEATURES)]
    + ["median_income", "pct_bachelors", "pop_density", "redfin_price", "redfin_dom"]
)


class GeoFeatureLookup:
    """Loads geographic surfaces and provides lat/lon -> feature lookups."""

    def __init__(self, surfaces_dir: str | Path):
        self.surfaces_dir = Path(surfaces_dir)
        self._load_surfaces()

    def _load_surfaces(self) -> None:
        """Load all surface .npy files."""
        self.g0_surfaces: dict[str, np.ndarray] = {}
        self.g1_surfaces: dict[str, np.ndarray] = {}

        for name in SURFACE_NAMES:
            g0_path = self.surfaces_dir / f"g0_{name}.npy"
            g1_path = self.surfaces_dir / f"g1_{name}.npy"

            if g0_path.exists():
                self.g0_surfaces[name] = np.load(g0_path)
            else:
                logger.warning(f"Missing g0 surface: {g0_path}, using zeros")
                self.g0_surfaces[name] = np.zeros(G0_NLAT * G0_NLON, dtype=np.float32)

            if g1_path.exists():
                self.g1_surfaces[name] = np.load(g1_path)
            else:
                logger.warning(f"Missing g1 surface: {g1_path}, using zeros")
                self.g1_surfaces[name] = np.zeros(G1_NLAT * G1_NLON, dtype=np.float32)

        logger.info(
            f"Loaded {len(self.g0_surfaces)} g0 surfaces and "
            f"{len(self.g1_surfaces)} g1 surfaces "
            f"({GEO_FEATURES_PER_GRID} features per grid, {GEO_FEATURES_PER_GRID * 2} total)"
        )

    @staticmethod
    def _lat_lon_to_cell_index(
        lat: np.ndarray, lon: np.ndarray,
        precision: int, nlat: int, nlon: int,
    ) -> np.ndarray:
        """Convert lat/lon arrays to flat cell indices."""
        lat_idx = np.floor((lat - LAT_MIN) * precision).astype(np.int64)
        lon_idx = np.floor((lon - LON_MIN) * precision).astype(np.int64)
        lat_idx = np.clip(lat_idx, 0, nlat - 1)
        lon_idx = np.clip(lon_idx, 0, nlon - 1)
        return lat_idx * nlon + lon_idx

    def lookup(self, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
        """
        Look up geographic features for arrays of lat/lon coordinates.

        Returns (N, 34) array: [g0 surfaces (17), g1 surfaces (17)]
        """
        n = len(lat)
        result = np.zeros((n, GEO_FEATURES_PER_GRID * 2), dtype=np.float32)

        # Coarse grid
        g0_idx = self._lat_lon_to_cell_index(lat, lon, G0_PRECISION, G0_NLAT, G0_NLON)
        col = 0
        for name in SURFACE_NAMES:
            result[:, col] = self.g0_surfaces[name][g0_idx]
            col += 1

        # Fine grid
        g1_idx = self._lat_lon_to_cell_index(lat, lon, G1_PRECISION, G1_NLAT, G1_NLON)
        for name in SURFACE_NAMES:
            result[:, col] = self.g1_surfaces[name][g1_idx]
            col += 1

        return result

    def augment_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add geographic feature columns to a DataFrame with latitude/longitude."""
        lat = df["latitude"].values
        lon = df["longitude"].values
        geo_feats = self.lookup(lat, lon)

        for i, name in enumerate(self.get_feature_names()):
            df[name] = geo_feats[:, i]

        logger.info(f"Added {len(self.get_feature_names())} geographic features")
        return df

    @staticmethod
    def get_feature_names() -> list[str]:
        """Return ordered list of geographic feature column names."""
        names = []
        for prefix in ["g0", "g1"]:
            for sname in SURFACE_NAMES:
                names.append(f"{prefix}_{sname}")
        return names

    def get_surface_arrays(self) -> dict[str, dict[str, np.ndarray]]:
        """Return raw surface arrays for ONNX export."""
        return {
            "g0": dict(self.g0_surfaces),
            "g1": dict(self.g1_surfaces),
        }
