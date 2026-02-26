# RESI Training Pipeline

Train competitive ONNX models for RESI Subnet 46 (Bittensor real estate price prediction).

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Set up your API key
cp .env.example .env
# Edit .env and add your RapidAPI key

# Option A: Full pipeline (search-heavy hybrid, $25 for ~107K properties)
python run_pipeline.py

# Option B: Skip data collection (use existing data)
python run_pipeline.py --skip-collect --skip-geo --skip-schools

# Option C: Custom API budget split
python run_pipeline.py --search-pages 50 --detail-budget 7500

# Option D: Legacy mode (detail every property)
python run_pipeline.py --legacy-collect --rapidapi-key YOUR_KEY
```

## Architecture

The pipeline produces a model that surpasses the current top competitor's 2-model LightGBM approach:

1. **4-model stacking ensemble** — 2 LightGBM + XGBoost + CatBoost base models, with Ridge regression meta-learner for optimal blending
2. **Enhanced geographic surfaces** baked into ONNX — ZHVI, regional prices, Census ACS (income, education, population density), and Redfin market data (median price, days on market) at 2 grid resolutions (17 features per grid, 34 total)
3. **Log-space prediction** with expm1 conversion and price clipping to [$100K, $20M]

The final ONNX model accepts the standard **(batch, 79)** float32 input and outputs **(batch, 1)** price predictions.

### Why This Beats the Top Model

| Aspect | Top Model (Jordun01) | Our Model |
|--------|---------------------|-----------|
| Base models | 2 LightGBM (0.6/0.4 fixed) | 2 LightGBM + XGBoost + CatBoost |
| Blending | Fixed weights | Ridge regression (learned optimal) |
| Geo features | 12 per grid (ZHVI + regional) | 17 per grid (+income, education, population, Redfin) |
| Total geo features | 24 | 34 |
| Diversity | Same algorithm, different params | 3 different algorithms with different inductive biases |

## Pipeline Steps

| Step | Script | Description |
|------|--------|-------------|
| 1 | `collect_geo_data.py` | Download ZHVI + Census ACS + Redfin data, build geo surfaces |
| 2 | `collect_schools.py` | Download NCES public school data (~100K schools), build KD-tree |
| 3 | `collect_data.py` (search) | Search-only collection: 2,500 API calls → ~100K property summaries |
| 4 | `collect_data.py` (select) | Price-stratified sampling: pick 7,500 zpids for detail collection |
| 5 | `collect_data.py` (detail) | Detail collection: 7,500 API calls → 7,500 fully detailed properties |
| 6 | `feature_engineer.py` | Transform both sources → 79-feature Parquet |
| 7 | `train_model.py` | Train 4-model stacking ensemble with Ridge meta-learner |
| 8 | `export_onnx.py` | Export to single ONNX with baked-in geo lookups |
| 9 | `run_pipeline.py` | Orchestrate all steps end-to-end |

### API Budget ($25/month Pro plan = 10,000 calls)

| Phase | API Calls | Properties | Features |
|-------|-----------|-----------|----------|
| Search | 2,500 | ~100,000 | ~22 of 79 |
| Detail | 7,500 | 7,500 | 79 of 79 |
| NCES schools | 0 (free) | All | +7 school features |
| Geo surfaces | 0 (free) | All | +34 ONNX geo features |

## RapidAPI Setup (Step-by-Step)

The pipeline uses the **"Real-Time Zillow Data" API by OpenWeb Ninja** on RapidAPI. This is the best active replacement for the deprecated zillow-com1 API, with the same endpoints and full Zillow data payload.

### 1. Create a RapidAPI Account

Go to https://rapidapi.com and sign up (free).

### 2. Subscribe to the Real-Time Zillow Data API

Go to: **https://rapidapi.com/letscrape-6bRBa3QguO5/api/real-time-zillow-data**

Click **"Subscribe to Test"** or **"Pricing"** and choose a plan:

| Plan | Monthly Cost | Requests/Month | Best For |
|------|-------------|----------------|----------|
| **Free** | $0 | ~500 | Testing the pipeline, ~50 properties |
| **Basic** | ~$25 | ~10,000 | Small dataset, ~5K properties |
| **Pro** | ~$100-150 | ~100,000+ | Full dataset, ~50K+ properties |

> **Cost to collect 50K properties:** You need ~50K search requests + ~50K detail requests = ~100K total. The Pro plan covers this. Start with the Free plan to test the pipeline on a handful of properties first.

### 3. Get Your API Key

After subscribing:
1. Go to https://rapidapi.com/developer/dashboard
2. Click **"Security"** in the left sidebar (or the **"Apps"** tab)
3. Your **default application** will have an API key -- copy it

### 4. Set the Key

```bash
cp .env.example .env
# Edit .env and paste your key
```

Or export directly:
```bash
export RAPIDAPI_KEY=your_key_here
```

### 5. Test It

```bash
# Quick test: fetch ~20 properties from one market
python collect_data.py --num-properties 20 --max-pages 1
```

Check `training/raw_data/` -- you should see JSON files like `12345678.json`.

### API Endpoints Used

The script calls two endpoints on `real-time-zillow-data.p.rapidapi.com`:

**1. Search Recently Sold Properties**
```
GET /propertyExtendedSearch
  ?location=Phoenix, AZ
  &status_type=RecentlySold
  &page=1
```
Returns property summaries with zpids.

**2. Get Full Property Details**
```
GET /property
  ?zpid=12345678
```
Returns 259+ fields including `resoFacts`, `nearbySchools`, `priceHistory`, `homeType`, `latitude`, `longitude`, `yearBuilt`, etc.

Both endpoints require:
```
X-RapidAPI-Key: your_key_here
X-RapidAPI-Host: real-time-zillow-data.p.rapidapi.com
```

## Geographic Data (Free -- No API Key)

Step 2 (`collect_geo_data.py`) downloads free public data from 4 sources:

| Source | Data | URL |
|--------|------|-----|
| **Zillow Research** | Home Value Index (ZHVI) by ZIP | `files.zillowstatic.com/research/...` |
| **Census ZCTA** | ZIP code lat/lon centroids | `www2.census.gov/geo/docs/...` |
| **Census ACS 5-Year** | Median income, education, population | `api.census.gov/data/2022/acs/acs5` |
| **Redfin Public Data** | Median sale price, days on market by ZIP | `redfin-public-data.s3.us-west-2.amazonaws.com/...` |

These are combined to build 17 geographic surfaces at 2 resolutions (coarse ~11km, fine ~3.7km), which get baked into the ONNX model as constant lookup tensors.

```bash
python collect_geo_data.py
# Downloads ~100MB of data, builds surfaces in training/geo_surfaces/
```

### Surfaces Built (17 per grid, 34 total)

| Surface | Source | Description |
|---------|--------|-------------|
| `zhvi` | Zillow | Log median home value |
| `is_zip` | Zillow | Whether cell has ZHVI data |
| `rp0`-`rp9` | Zillow | Regional price statistics (percentiles, mean, std) |
| `median_income` | Census ACS | Log median household income |
| `pct_bachelors` | Census ACS | % population with bachelor's degree |
| `pop_density` | Census ACS | Log population density |
| `redfin_price` | Redfin | Log median sale price |
| `redfin_dom` | Redfin | Days on market (normalized) |

## Validation

```bash
# Local evaluation with miner CLI (from project root)
uv run miner-cli evaluate --model.path training/model.onnx

# Check model size (must be < 200 MB)
ls -lh training/model.onnx
```

## File Structure

```
training/
├── README.md
├── requirements.txt          # Training dependencies (LightGBM, XGBoost, CatBoost, etc.)
├── .env.example              # Template for API keys
├── .env                      # (gitignored) Your actual API keys
├── config.py                 # Constants, 79-feature order, hyperparameters for all 4 models
├── collect_data.py           # Zillow API data collection (search + detail modes)
├── collect_geo_data.py       # ZHVI + Census ACS + Redfin download + surface building
├── collect_schools.py        # NCES public school data + KD-tree spatial lookup
├── feature_engineer.py       # Raw JSON + search JSON -> 79-feature Parquet
├── geo_features.py           # Lat/lon -> 34 geographic feature lookups
├── train_model.py            # 4-model stacking ensemble + Ridge meta-learner
├── export_onnx.py            # ONNX export with baked-in geo lookups + meta-learner
├── run_pipeline.py           # End-to-end 9-step orchestrator
├── raw_data/                 # (generated) Detailed property JSON files
├── search_data/              # (generated) Search-only property JSON files
├── school_data/              # (generated) NCES school data + KD-tree pickle
├── geo_data/                 # (generated) Cached ZHVI/ZCTA/Census/Redfin downloads
├── geo_surfaces/             # (generated) Geographic surface .npy files (17 per grid)
├── models/                   # (generated) Trained model .pkl files (m1-m4 + meta + metadata)
├── dataset.parquet           # (generated) Training dataset
└── model.onnx                # (generated) Final ONNX model
```

## Troubleshooting

**"403 Forbidden" from API**: Your API key is invalid or you haven't subscribed to the Real-Time Zillow Data API. Go to https://rapidapi.com/letscrape-6bRBa3QguO5/api/real-time-zillow-data/pricing and subscribe.

**"429 Too Many Requests"**: You've hit the rate limit. The script auto-waits 60s. Reduce concurrency: `--max-concurrent 1 --delay 1.0`

**Empty search results**: Some smaller cities return few recently sold properties. The pipeline covers 50 major metros to maximize geographic diversity.

**ONNX model > 200 MB**: The geographic surfaces are large (~127MB for 17 surfaces at 2 resolutions). If model exceeds 200MB, try reducing grid resolution in `config.py` (lower `G1_PRECISION` from 30 to 20).

**Census ACS download fails**: The Census API occasionally has downtime. The script caches downloads in `geo_data/`, so re-running will skip already-fetched data. Census data is optional -- the pipeline still works without it (surfaces will be zeros for income/education/population features).
