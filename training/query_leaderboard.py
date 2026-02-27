"""
Query RESI Subnet 46 leaderboard: metagraph rankings and miner submissions.

Usage:
    python query_leaderboard.py                    # Full leaderboard + submissions
    python query_leaderboard.py --top 20           # Top 20 only
    python query_leaderboard.py --miners-only      # Only miners (skip validators)
    python query_leaderboard.py --network test      # Use testnet
"""

import argparse
import json
import sys
from datetime import datetime, timezone

NETUID = 46


def query_metagraph(network: str = "finney"):
    """Query metagraph using bittensor SDK."""
    try:
        import bittensor as bt
    except ImportError:
        print("ERROR: bittensor not installed. Run: pip install bittensor")
        sys.exit(1)

    print(f"Connecting to {network} network...")

    # Handle both old (bt.subtensor) and new (bt.Subtensor) SDK versions
    subtensor_cls = getattr(bt, "Subtensor", None) or getattr(bt, "subtensor")
    metagraph_cls = getattr(bt, "Metagraph", None) or getattr(bt, "metagraph")

    sub = subtensor_cls(network=network)
    print(f"Fetching metagraph for subnet {NETUID}...")

    # Try new SDK signature first, fall back to old
    try:
        meta = metagraph_cls(netuid=NETUID, network=network, sync=True)
    except TypeError:
        meta = metagraph_cls(netuid=NETUID)
        meta.sync(subtensor=sub)

    return sub, meta


def query_commitments(sub, meta):
    """Query on-chain commitments for all neurons."""
    commitments = {}
    hotkeys = meta.hotkeys

    print(f"Fetching commitments for {len(hotkeys)} neurons...")
    for i, hotkey in enumerate(hotkeys):
        if i % 50 == 0 and i > 0:
            print(f"  ...checked {i}/{len(hotkeys)} neurons")
        try:
            # Try different SDK method names for querying chain storage
            query_fn = getattr(sub, "query_module", None) or getattr(sub, "query", None)
            if query_fn is None:
                print("  WARNING: Cannot query commitments — SDK missing query_module/query method")
                break
            result = query_fn(
                module="Commitments",
                name="CommitmentOf",
                params=[NETUID, hotkey],
            )
            if result and hasattr(result, "value") and result.value:
                raw = result.value
                # Commitment is stored as {"info": {"fields": [{"Raw<N>": hex_data}]}, "block": N}
                info = raw.get("info", {})
                block = raw.get("block", 0)
                fields = info.get("fields", [])

                hex_data = None
                for field in fields:
                    if isinstance(field, dict):
                        for key, val in field.items():
                            if key.startswith("Raw") and val:
                                hex_data = val
                                break
                    if hex_data:
                        break

                if hex_data:
                    try:
                        hex_str = hex_data[2:] if hex_data.startswith("0x") else hex_data
                        decoded = bytes.fromhex(hex_str).decode("utf-8")
                        data = json.loads(decoded)
                        commitments[hotkey] = {
                            "hf_repo_id": data.get("r", ""),
                            "model_hash": data.get("h", ""),
                            "block": block,
                        }
                    except (ValueError, json.JSONDecodeError, UnicodeDecodeError):
                        commitments[hotkey] = {
                            "hf_repo_id": "?",
                            "model_hash": "?",
                            "block": block,
                            "raw": hex_data[:40] + "...",
                        }
        except Exception:
            continue

    print(f"Found {len(commitments)} commitments\n")
    return commitments


def display_leaderboard(meta, commitments, top_n=None, miners_only=False):
    """Display formatted leaderboard."""
    n = meta.n.item() if hasattr(meta.n, "item") else int(meta.n)

    # Build rows
    rows = []
    for i in range(n):
        uid = int(meta.uids[i])
        hotkey = meta.hotkeys[i]
        stake = float(meta.stake[i])
        trust = float(meta.trust[i])
        consensus = float(meta.consensus[i])
        incentive = float(meta.incentive[i])
        dividends = float(meta.dividends[i])
        emission = float(meta.emission[i])
        is_validator = dividends > 0 or stake > 1000

        if miners_only and is_validator:
            continue

        commitment = commitments.get(hotkey, None)

        rows.append({
            "uid": uid,
            "hotkey": hotkey,
            "stake": stake,
            "trust": trust,
            "consensus": consensus,
            "incentive": incentive,
            "dividends": dividends,
            "emission": emission,
            "is_validator": is_validator,
            "commitment": commitment,
        })

    # Sort by incentive (descending) — this is the primary ranking metric
    rows.sort(key=lambda r: r["incentive"], reverse=True)

    if top_n:
        rows = rows[:top_n]

    # Print header
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    print("=" * 120)
    print(f"  RESI Subnet {NETUID} Leaderboard — {now}")
    print("=" * 120)

    # Metagraph rankings
    print(f"\n{'Rank':<5} {'UID':<5} {'Incentive':>10} {'Emission':>10} "
          f"{'Trust':>7} {'Consensus':>10} {'Stake':>10} {'Role':<5} "
          f"{'HF Repo':<35} {'Hotkey'}")
    print("-" * 120)

    for rank, row in enumerate(rows, 1):
        role = "VAL" if row["is_validator"] else "MIN"
        hf_repo = ""
        if row["commitment"]:
            hf_repo = row["commitment"].get("hf_repo_id", "")
        hotkey_short = row["hotkey"][:8] + "..." + row["hotkey"][-6:]

        print(
            f"{rank:<5} {row['uid']:<5} {row['incentive']:>10.6f} {row['emission']:>10.6f} "
            f"{row['trust']:>7.4f} {row['consensus']:>10.6f} {row['stake']:>10.1f} "
            f"{role:<5} {hf_repo:<35} {hotkey_short}"
        )

    # Submissions detail
    print(f"\n{'=' * 120}")
    print(f"  Miner Submissions Detail")
    print(f"{'=' * 120}\n")

    # Sort submissions by block (earliest first)
    submissions = []
    for row in rows:
        if row["commitment"] and not row["is_validator"]:
            submissions.append(row)

    submissions.sort(key=lambda r: r["commitment"].get("block", 0))

    if not submissions:
        print("  No miner submissions found.\n")
    else:
        print(f"{'UID':<5} {'Incentive':>10} {'Block':>10} "
              f"{'HF Repo':<40} {'Model Hash':<20} {'Hotkey'}")
        print("-" * 120)

        for row in submissions:
            c = row["commitment"]
            hf_repo = c.get("hf_repo_id", "?")
            model_hash = c.get("model_hash", "?")
            if len(model_hash) > 16:
                model_hash = model_hash[:16] + "..."
            block = c.get("block", 0)
            hotkey_short = row["hotkey"][:8] + "..." + row["hotkey"][-6:]

            print(
                f"{row['uid']:<5} {row['incentive']:>10.6f} {block:>10} "
                f"{hf_repo:<40} {model_hash:<20} {hotkey_short}"
            )

    # Summary stats
    miners = [r for r in rows if not r["is_validator"]]
    validators = [r for r in rows if r["is_validator"]]
    active_miners = [r for r in miners if r["incentive"] > 0]
    with_submissions = [r for r in miners if r["commitment"]]

    print(f"\n{'=' * 120}")
    print(f"  Summary")
    print(f"{'=' * 120}")
    print(f"  Total neurons:          {len(rows)}")
    print(f"  Validators:             {len(validators)}")
    print(f"  Miners:                 {len(miners)}")
    print(f"  Miners with incentive:  {len(active_miners)}")
    print(f"  Miners with submission: {len(with_submissions)}")
    if active_miners:
        top = active_miners[0] if miners else None
        if top:
            hf = top["commitment"].get("hf_repo_id", "?") if top["commitment"] else "?"
            print(f"  Current winner:         UID {top['uid']} — {hf} (incentive: {top['incentive']:.6f})")
    print()


def main():
    parser = argparse.ArgumentParser(description="Query RESI Subnet 46 leaderboard")
    parser.add_argument(
        "--top", type=int, default=None,
        help="Show only top N neurons",
    )
    parser.add_argument(
        "--miners-only", action="store_true",
        help="Show only miners (exclude validators)",
    )
    parser.add_argument(
        "--network", type=str, default="finney",
        help="Bittensor network (finney, test, local)",
    )
    parser.add_argument(
        "--skip-commitments", action="store_true",
        help="Skip fetching commitments (faster, no HF repo info)",
    )
    args = parser.parse_args()

    sub, meta = query_metagraph(args.network)

    commitments = {}
    if not args.skip_commitments:
        commitments = query_commitments(sub, meta)

    display_leaderboard(
        meta, commitments,
        top_n=args.top,
        miners_only=args.miners_only,
    )


if __name__ == "__main__":
    main()
