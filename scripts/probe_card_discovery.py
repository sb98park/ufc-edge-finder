"""
Why did card_discovery skip an event UFC.com clearly has?

THE SILENCE THIS EXISTS TO BREAK. card_discovery walks ESPN's calendar, and
for each in-window event it calls _fetch_espn_full_card(). Until now, an event
that came back with no fights fell through with no output whatsoever -- so an
event ESPN has not published, an event whose label does not match, and a
broken fetch all looked identical: nothing. UFC.com listed six confirmed cards
while the site tracked three, and every run printed a clean result.

This prints the whole decision for every calendar entry, so the actual reason
is visible:
  - is the event in ESPN's calendar at all?
  - is it inside the days_ahead window?
  - is its label already tracked?
  - how many fights does the full-card fetch return?

Usage:
    python3 scripts/probe_card_discovery.py
    python3 scripts/probe_card_discovery.py --days 120
"""

import argparse
import datetime as dt
import os
import sys

import pandas as pd
import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.results_fetcher import BASE_HEADERS, REQUEST_TIMEOUT, ESPN_SCOREBOARD_URL  # noqa: E402
from src.card_discovery import _fetch_espn_full_card  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=120)
    args = ap.parse_args()

    tracked = set()
    for path in ("data/fight_cards.csv", "data/future_cards.csv"):
        try:
            df = pd.read_csv(path)
        except (FileNotFoundError, pd.errors.EmptyDataError):
            continue
        if "event_name" in df.columns:
            tracked |= set(df["event_name"].dropna().unique())
    print(f"tracked events: {len(tracked)}")
    for t in sorted(tracked):
        print(f"   {t}")

    try:
        r = requests.get(ESPN_SCOREBOARD_URL, headers=BASE_HEADERS, timeout=REQUEST_TIMEOUT)
        r.raise_for_status()
        cal = (r.json().get("leagues") or [{}])[0].get("calendar") or []
    except (requests.RequestException, ValueError, IndexError) as e:
        print(f"calendar fetch failed: {e}")
        sys.exit(1)

    today = dt.datetime.now(dt.timezone.utc).date()
    cutoff = today + dt.timedelta(days=args.days)
    print(f"\nESPN calendar has {len(cal)} entries; window {today} .. {cutoff}\n")
    print(f"  {'date':<12}{'days':>5}  {'in win':<7}{'tracked':<9}{'fights':>7}  event")
    print("  " + "-" * 86)

    for entry in cal:
        label, start = entry.get("label"), entry.get("startDate")
        if not label or not start:
            continue
        try:
            d = dt.datetime.fromisoformat(start.replace("Z", "+00:00")).date()
        except (ValueError, TypeError):
            continue
        days = (d - today).days
        if days < -7 or days > args.days + 60:
            continue
        in_window = today <= d <= cutoff
        is_tracked = label in tracked
        # Only pay for the fetch where it actually decides something.
        fights = ""
        if in_window and not is_tracked:
            rows = _fetch_espn_full_card(label, d.isoformat())
            fights = str(len(rows)) if rows else "0"
        print(f"  {d.isoformat():<12}{days:>5}  {str(in_window):<7}{str(is_tracked):<9}{fights:>7}  {label}")

    print("\nREADING THIS")
    print("  in win=False           -> outside days_ahead; raise --days")
    print("  tracked=True           -> already have it, nothing to do")
    print("  fights=0               -> ESPN has the EVENT but no card yet. Nothing to")
    print("                            add; it will appear once ESPN publishes bouts.")
    print("  absent from this table -> ESPN's calendar does not list it at all, even")
    print("                            though UFC.com may. That is a source gap, and")
    print("                            the fight would need adding by hand via")
    print("                            scripts/add_fight_manually.py.")


if __name__ == "__main__":
    main()
