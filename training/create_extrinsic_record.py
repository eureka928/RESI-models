"""
Create extrinsic_record.json required for validator model verification.

Queries the chain for your commitment, finds the extrinsic, and writes
the file to the same directory as model.onnx. Optionally uploads to HuggingFace.

Usage:
    python create_extrinsic_record.py --uid 61 --wallet.name miner
    python create_extrinsic_record.py --uid 61 --wallet.name miner --upload mihai-777/hallelujah777
"""

import argparse
import json
import sys
from pathlib import Path

NETUID = 46
OUTPUT_DIR = Path(__file__).parent


def get_hotkey_and_commitment(uid: int, network: str = "finney"):
    """Get hotkey and commitment info from chain."""
    try:
        import bittensor as bt
    except ImportError:
        print("ERROR: bittensor not installed. Run: pip install bittensor")
        sys.exit(1)

    subtensor_cls = getattr(bt, "Subtensor", None) or getattr(bt, "subtensor")
    sub = subtensor_cls(network=network)
    meta = sub.metagraph(netuid=NETUID)

    hotkey = meta.hotkeys[uid]
    print(f"UID {uid} hotkey: {hotkey}")

    # Query commitment
    substrate = getattr(sub, "substrate", None)
    if substrate is None:
        print("ERROR: Cannot access substrate interface")
        sys.exit(1)

    result = substrate.query(
        module="Commitments",
        storage_function="CommitmentOf",
        params=[NETUID, hotkey],
    )

    if result is None or not result.value:
        print(f"ERROR: No commitment found for UID {uid}")
        sys.exit(1)

    raw = result.value
    block = raw.get("block", 0)
    print(f"Commitment found at block: {block}")

    return sub, hotkey, block


def find_extrinsic(sub, hotkey: str, commit_block: int, scan_range: int = 50):
    """Scan blocks around the commitment to find the extrinsic ID."""
    substrate = sub.substrate
    print(f"Scanning blocks {commit_block - 5} to {commit_block + scan_range} for extrinsic...")

    for block_num in range(max(0, commit_block - 5), commit_block + scan_range):
        try:
            block_hash = substrate.get_block_hash(block_num)
            if not block_hash:
                continue
            block = substrate.get_block(block_hash)
            if not block or "extrinsics" not in block:
                continue

            extrinsics = block["extrinsics"]
            for idx, ext in enumerate(extrinsics):
                try:
                    call = ext.value.get("call", {})
                    call_module = call.get("call_module", "")
                    call_function = call.get("call_function", "")

                    if call_module == "Commitments" and call_function == "set_commitment":
                        # Check if signer matches
                        signer = ext.value.get("address", "")
                        if signer == hotkey:
                            extrinsic_id = f"{block_num}-{idx}"
                            print(f"Found extrinsic: {extrinsic_id}")
                            return extrinsic_id
                except (AttributeError, KeyError):
                    continue
        except Exception as e:
            if block_num % 10 == 0:
                print(f"  ...scanning block {block_num}")
            continue

    print(f"WARNING: Extrinsic not found in range. Trying commit block directly...")
    # Fallback: use the commitment block itself
    return None


def main():
    parser = argparse.ArgumentParser(description="Create extrinsic_record.json")
    parser.add_argument("--uid", type=int, required=True, help="Your miner UID")
    parser.add_argument("--network", default="finney", help="Network (finney/test)")
    parser.add_argument("--scan-range", type=int, default=50, help="Blocks to scan")
    parser.add_argument("--upload", type=str, default=None,
                        help="HuggingFace repo ID to upload to (e.g. mihai-777/hallelujah777)")
    parser.add_argument("--extrinsic", type=str, default=None,
                        help="Manually provide extrinsic ID (skip scanning)")
    parser.add_argument("--hotkey", type=str, default=None,
                        help="Manually provide hotkey (skip chain query)")
    args = parser.parse_args()

    if args.hotkey and args.extrinsic:
        # Manual mode — no chain query needed
        hotkey = args.hotkey
        extrinsic_id = args.extrinsic
    else:
        sub, hotkey, commit_block = get_hotkey_and_commitment(args.uid, args.network)

        if args.extrinsic:
            extrinsic_id = args.extrinsic
        else:
            extrinsic_id = find_extrinsic(sub, hotkey, commit_block, args.scan_range)

    if not extrinsic_id:
        print("\nERROR: Could not find extrinsic automatically.")
        print("You can provide it manually:")
        print(f"  python create_extrinsic_record.py --uid {args.uid} --extrinsic BLOCK-INDEX")
        print("\nCheck a block explorer for your hotkey's Commitments::set_commitment call.")
        sys.exit(1)

    # Write extrinsic_record.json
    record = {
        "extrinsic": extrinsic_id,
        "hotkey": hotkey,
    }

    output_path = OUTPUT_DIR / "extrinsic_record.json"
    with open(output_path, "w") as f:
        json.dump(record, f, indent=2)

    print(f"\nCreated {output_path}")
    print(f"  extrinsic: {extrinsic_id}")
    print(f"  hotkey:    {hotkey}")

    # Upload to HuggingFace if requested
    if args.upload:
        try:
            from huggingface_hub import HfApi
            api = HfApi()
            api.upload_file(
                path_or_fileobj=str(output_path),
                path_in_repo="extrinsic_record.json",
                repo_id=args.upload,
            )
            print(f"\nUploaded to https://huggingface.co/{args.upload}")
        except ImportError:
            print("\nhuggingface_hub not installed. Upload manually:")
            print(f"  pip install huggingface_hub")
            print(f"  huggingface-cli upload {args.upload} {output_path} extrinsic_record.json")
        except Exception as e:
            print(f"\nUpload failed: {e}")
            print(f"Upload manually: huggingface-cli upload {args.upload} {output_path} extrinsic_record.json")


if __name__ == "__main__":
    main()
