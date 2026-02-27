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


def _get_meta_array(meta, *names):
    """Get a metagraph attribute by trying multiple names (handles SDK version diffs).

    New SDK uses short names (S, I, D, E, C, T), old uses long names
    (stake, incentive, dividends, emission, consensus, trust).
    """
    for name in names:
        val = getattr(meta, name, None)
        if val is not None:
            return val
    # Return zeros if nothing found
    import numpy as np
    n = meta.n.item() if hasattr(meta.n, "item") else int(meta.n)
    return np.zeros(n)


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
    sub = subtensor_cls(network=network)
    print(f"Fetching metagraph for subnet {NETUID}...")

    # Try multiple ways to get metagraph
    meta = None

    # Method 1: sub.metagraph() (newer SDK)
    if meta is None:
        try:
            meta = sub.metagraph(netuid=NETUID)
        except (AttributeError, TypeError):
            pass

    # Method 2: bt.Metagraph / bt.metagraph class
    if meta is None:
        metagraph_cls = getattr(bt, "Metagraph", None) or getattr(bt, "metagraph", None)
        if metagraph_cls:
            try:
                meta = metagraph_cls(netuid=NETUID, network=network, sync=True)
            except TypeError:
                try:
                    meta = metagraph_cls(netuid=NETUID)
                    meta.sync(subtensor=sub)
                except Exception:
                    pass

    if meta is None:
        print("ERROR: Could not fetch metagraph. Check bittensor SDK version.")
        sys.exit(1)

    return sub, meta


def query_commitments(sub, meta):
    """Query on-chain commitments for all neurons."""
    commitments = {}
    hotkeys = list(meta.hotkeys)

    print(f"Fetching commitments for {len(hotkeys)} neurons...")

    # Get substrate interface for direct RPC queries
    substrate = getattr(sub, "substrate", None)
    if substrate is None:
        print("  WARNING: Cannot access substrate interface — skipping commitments")
        return commitments

    for i, hotkey in enumerate(hotkeys):
        if i % 50 == 0 and i > 0:
            print(f"  ...checked {i}/{len(hotkeys)} neurons")
        try:
            # Direct substrate query for commitment storage
            result = substrate.query(
                module="Commitments",
                storage_function="CommitmentOf",
                params=[NETUID, hotkey],
            )
            if result is None:
                continue

            raw = result.value if hasattr(result, "value") else result
            if not raw or raw == {"info": {"fields": [{"None": None}]}, "block": 0}:
                continue

            # Parse the commitment structure
            hex_data = None
            block = 0

            if isinstance(raw, dict):
                block = raw.get("block", 0)
                info = raw.get("info", {})
                fields = info.get("fields", [])
                for field in fields:
                    if isinstance(field, dict):
                        for key, val in field.items():
                            if val and key != "None":
                                hex_data = str(val)
                                break
                    elif isinstance(field, str) and len(field) > 10:
                        hex_data = field
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
                    }
        except Exception:
            continue

    print(f"Found {len(commitments)} commitments\n")
    return commitments


def display_leaderboard(meta, commitments, top_n=None, miners_only=False):
    """Display formatted leaderboard."""
    n = meta.n.item() if hasattr(meta.n, "item") else int(meta.n)

    # Get arrays — handle both old (long names) and new (short names) SDK
    stakes = _get_meta_array(meta, "stake", "S", "total_stake")
    trusts = _get_meta_array(meta, "trust", "T")
    consensuses = _get_meta_array(meta, "consensus", "C")
    incentives = _get_meta_array(meta, "incentive", "I")
    dividendses = _get_meta_array(meta, "dividends", "D")
    emissions = _get_meta_array(meta, "emission", "E")

    # Build rows
    rows = []
    for i in range(n):
        uid = int(meta.uids[i])
        hotkey = meta.hotkeys[i]
        stake = float(stakes[i])
        trust = float(trusts[i])
        consensus = float(consensuses[i])
        incentive = float(incentives[i])
        dividends = float(dividendses[i])
        emission = float(emissions[i])
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
    all_rows = rows  # already filtered by miners_only if needed
    miners = [r for r in all_rows if not r["is_validator"]]
    validators = [r for r in all_rows if r["is_validator"]]
    active_miners = [r for r in miners if r["incentive"] > 0]
    with_submissions = [r for r in miners if r["commitment"]]

    print(f"\n{'=' * 120}")
    print(f"  Summary")
    print(f"{'=' * 120}")
    print(f"  Total neurons:          {n}")
    print(f"  Validators:             {len(validators)}")
    print(f"  Miners:                 {len(miners)}")
    print(f"  Miners with incentive:  {len(active_miners)}")
    print(f"  Miners with submission: {len(with_submissions)}")
    if active_miners:
        top = active_miners[0]
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
    parser.add_argument(
        "--debug", action="store_true",
        help="Print debug info about SDK attributes",
    )
    args = parser.parse_args()

    sub, meta = query_metagraph(args.network)

    if args.debug:
        print(f"\n[DEBUG] Metagraph attributes: {[a for a in dir(meta) if not a.startswith('_')]}")
        print(f"[DEBUG] Subtensor attributes: {[a for a in dir(sub) if not a.startswith('_')]}")
        print(f"[DEBUG] Has substrate: {hasattr(sub, 'substrate')}\n")

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
