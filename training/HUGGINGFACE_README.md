---
license: mit
tags:
  - real-estate
  - onnx
  - bittensor
  - resi
---

# RESI Real Estate Price Prediction Model

ONNX model for US residential real estate price prediction, built for [RESI Subnet 46](https://github.com/resilabs-subnet/resi) on Bittensor.

## Model Details

- **Input**: 79 float32 features (property attributes, location, school data, census data)
- **Output**: Predicted price in USD
- **Format**: ONNX (compatible with onnxruntime 1.20.1)
- **License**: MIT

## Architecture

5-model stacking ensemble with geographic features baked into the ONNX graph:

| Component | Description |
|-----------|-------------|
| **m1** | LightGBM (127 leaves, MAE objective) |
| **m2** | LightGBM (63 leaves, MAPE objective) |
| **m3** | XGBoost (depth 6, MAE objective) |
| **m4** | CatBoost (depth 6, MAPE objective) |
| **m5** | LightGBM (95 leaves, Huber objective) |
| **Meta-learner** | RidgeCV on out-of-fold predictions |

All base models predict in log1p(price) space. The meta-learner blends predictions, then the ONNX graph applies expm1 conversion and clips to [$50K, $20M].

## Geographic Features

34 geographic features are baked into the ONNX model as constant lookup tensors at 2 grid resolutions (~11km coarse, ~3.7km fine):

- Zillow Home Value Index (ZHVI)
- Regional price statistics (percentiles, mean, std)
- Census ACS (median income, education, population density)
- Redfin market data (median sale price, days on market)

## Training Data

- US residential property sales across 50+ metro areas
- Non-disclosure states excluded (TX, UT, etc.)
- Outlier filtering on price/sqft and living area
- Sample weighting to balance across price ranges

## Evaluation

```bash
uv run miner-cli evaluate --model.path model.onnx
```
