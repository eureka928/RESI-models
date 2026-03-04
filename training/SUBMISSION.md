# Submission Guide

UID: **61** | HF Repo: **mihai-777/hallelujah777**

## Step 1: Upload model to HuggingFace

```bash
pip install huggingface_hub
huggingface-cli login
huggingface-cli upload mihai-777/hallelujah777 training/model.onnx model.onnx
```

## Step 2: Submit commitment on-chain

```bash
uv run miner-cli submit \
  --model.path ./training/model.onnx \
  --hf.repo_id mihai-777/hallelujah777 \
  --wallet.name miner
```

## Step 3: Create & upload extrinsic_record.json

**Critical** — without this file, validators skip your model entirely.

```bash
cd training

# Auto-detect from chain (recommended)
python create_extrinsic_record.py --uid 61 --upload mihai-777/hallelujah777

# OR manual mode if you know the extrinsic ID
python create_extrinsic_record.py --uid 61 \
  --extrinsic BLOCK-INDEX \
  --hotkey YOUR_SS58_HOTKEY \
  --upload mihai-777/hallelujah777
```

## Step 4: Verify

```bash
huggingface-cli repo info mihai-777/hallelujah777
```

HF repo must contain:
- `model.onnx` — at root, < 200 MB
- `README.md` — with MIT license in metadata
- `extrinsic_record.json` — `{"extrinsic": "BLOCK-INDEX", "hotkey": "SS58_ADDRESS"}`

## Important Notes

- Steps must be done **in order** — submit on-chain (step 2) before creating extrinsic record (step 3), since the script queries the chain for your commitment.
- Model must be committed **28+ hours** before the next 18:00 UTC evaluation window.
- Evaluation happens daily at **18:00 UTC** when validator fetches fresh sales data from `dashboard.resilabs.ai`.
- Winner-takes-all: best score wins 99% of emissions. Within winner set (threshold 0.01), earliest commit wins.
