"""
Direct, standalone test of the Combat Edge relay -- bypasses the normal
combat_edge_checked/wikipedia_checked eligibility gating entirely, so it
gives an immediate, unambiguous answer instead of waiting for the right
fighter to naturally come up in rotation (which is what made the last
real production run inconclusive -- every fighter processed that run had
already exhausted Combat Edge before the relay existed, so it was never
even attempted).

Run with the relay env vars set, exactly like the real GitHub Actions job
does -- either locally:
    COMBAT_EDGE_RELAY_URL="https://your-relay.workers.dev" \\
    COMBAT_EDGE_RELAY_TOKEN="your-token" \\
    python3 scripts/test_combat_edge_relay.py

...or as a one-off manual GitHub Actions run if you'd rather test it there
directly (same env, no local Python setup needed).

Tests against a known, real, well-established fighter (Islam Makhachev)
who's virtually guaranteed to be on Combat Edge's A-Z directory, so a
failure here means the relay/block, not "this fighter isn't listed."
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.fighter_backfill import _fetch_method_breakdown_from_combat_edge, RATE_LIMITED

TEST_FIGHTER = "Islam Makhachev"


def main():
    relay_url = os.environ.get("COMBAT_EDGE_RELAY_URL")
    relay_token = os.environ.get("COMBAT_EDGE_RELAY_TOKEN")

    print(f"Relay configured: {'YES' if (relay_url and relay_token) else 'NO'}")
    if relay_url:
        print(f"  COMBAT_EDGE_RELAY_URL = {relay_url}")
    if not (relay_url and relay_token):
        print("  Both COMBAT_EDGE_RELAY_URL and COMBAT_EDGE_RELAY_TOKEN must be set as")
        print("  environment variables for this test to actually exercise the relay --")
        print("  without them, this just tests the old direct (blocked) path.")
    print()
    print(f"Fetching method breakdown for {TEST_FIGHTER!r} via combat-edge.com...")
    print("-" * 70)

    result = _fetch_method_breakdown_from_combat_edge(TEST_FIGHTER)

    print("-" * 70)
    if result is RATE_LIMITED:
        print("RESULT: Rate-limited (429) -- the block is NOT bypassed.")
        print("Either the relay isn't configured, or Combat Edge blocks Cloudflare's")
        print("IP range too (in which case this approach is a dead end).")
    elif result is None:
        print("RESULT: None -- request completed without a 429, but no data came back.")
        print("This could mean the relay WORKED (request went through, fighter just")
        print("wasn't found/parsed) -- check the printed log lines above this for the")
        print("specific reason. If you see 'not found on the directory page' or a")
        print("parse-related message rather than a 429, that's a good sign.")
    else:
        print(f"RESULT: SUCCESS -- got real data back: {result}")
        print("The relay worked. Combat Edge is reachable again.")


if __name__ == "__main__":
    main()
