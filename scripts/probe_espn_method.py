"""
Diagnostic probe: does ESPN publish method-of-victory ANYWHERE we can reach?

Background. results_fetcher.py reads the SITE api scoreboard
(site.api.espn.com/.../mma/ufc/scoreboard). On that response the only
method-ish text available is status.type.{description,detail,shortDetail},
and on real completed UFC cards all three say "Final" -- confirmed 12/12
on a full card. Because the pipeline (correctly) refuses to log a winner
with no confident method, results never auto-confirm from that source.

Two leads have never actually been looked at:

  1. competition["details"] -- present in the key list on every competition
     object we've logged, contents never inspected. On other ESPN sports
     this is where scoring/finish detail lives.

  2. The CORE api (sports.core.api.espn.com/v2/...), a different, richer
     API than the site one, whose competition object exposes $ref children
     (status / details / linescores / situation). fighter_backfill.py
     already uses this API successfully for last-fight dates, so we know
     it's reachable and parseable -- it has simply never been asked for
     method.

This script fetches both and dumps what it finds to a JSON file. It only
READS -- it writes nothing except the output file, and touches no
pipeline data.

Usage (from the repo root):
    python3 scripts/probe_espn_method.py --date 2026-07-25 --fighter "Magomed Ankalaev"

Then send the generated espn_method_probe.json back.
"""

import argparse
import json

import requests

SITE_SCOREBOARD = "https://site.api.espn.com/apis/site/v2/sports/mma/ufc/scoreboard"
CORE_EVENT = "https://sports.core.api.espn.com/v2/sports/mma/leagues/ufc/events/{event_id}"
BASE_HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"}
TIMEOUT = 20
OUT_PATH = "espn_method_probe.json"

# $ref children of the core-api competition worth resolving. Deliberately
# a short list: each one costs a request, and these are the only names
# that plausibly carry finish information.
INTERESTING_REFS = ["status", "details", "linescores", "situation", "odds", "predictor"]


def _get(url, params=None):
    try:
        r = requests.get(url, params=params, headers=BASE_HEADERS, timeout=TIMEOUT)
        r.raise_for_status()
        return r.json()
    except (requests.RequestException, ValueError) as e:
        return {"__error__": str(e)}


def _shrink(obj, depth=0):
    """Keep dumps readable: trim long lists and deep nesting, never values we care about."""
    if depth > 6:
        return "<...depth cut...>"
    if isinstance(obj, dict):
        return {k: _shrink(v, depth + 1) for k, v in obj.items()}
    if isinstance(obj, list):
        shown = [_shrink(v, depth + 1) for v in obj[:8]]
        if len(obj) > 8:
            shown.append(f"<...{len(obj) - 8} more items...>")
        return shown
    return obj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True, help="event date, YYYY-MM-DD (e.g. 2026-07-25)")
    ap.add_argument("--fighter", required=True, help="any fighter on that card, to identify the event")
    args = ap.parse_args()

    out = {"query": {"date": args.date, "fighter": args.fighter}}

    # ---- 1. SITE api scoreboard: inspect competition["details"] ----
    board = _get(SITE_SCOREBOARD, {"dates": args.date.replace("-", "")})
    if "__error__" in board:
        out["site_api"] = {"error": board["__error__"]}
        print(f"site scoreboard failed: {board['__error__']}")
    else:
        target = args.fighter.strip().lower()
        matched_event, matched_comp = None, None
        for ev in board.get("events", []):
            for comp in ev.get("competitions", []):
                names = [
                    c.get("athlete", {}).get("fullName", "").strip().lower()
                    for c in comp.get("competitors", [])
                ]
                if target in names:
                    matched_event, matched_comp = ev, comp
                    break
            if matched_comp:
                break

        if not matched_comp:
            all_names = sorted({
                c.get("athlete", {}).get("fullName", "")
                for ev in board.get("events", [])
                for comp in ev.get("competitions", [])
                for c in comp.get("competitors", [])
            })
            out["site_api"] = {"error": "fighter not found on that date", "names_seen": all_names}
            print("Fighter not found. Names ESPN lists for that date:")
            for n in all_names:
                print("   ", n)
            with open(OUT_PATH, "w") as f:
                json.dump(_shrink(out), f, indent=2)
            return

        out["site_api"] = {
            "event_id": matched_event.get("id"),
            "event_name": matched_event.get("name"),
            "competition_id": matched_comp.get("id"),
            "competition_keys": sorted(matched_comp.keys()),
            # THE never-inspected field:
            "details": matched_comp.get("details"),
            "status": matched_comp.get("status"),
            "type": matched_comp.get("type"),
            "notes": matched_comp.get("notes"),
            "has_linescores": [
                bool(c.get("linescores")) for c in matched_comp.get("competitors", [])
            ],
            "playByPlayAvailable": matched_comp.get("playByPlayAvailable"),
        }
        print(f"site api: matched {matched_event.get('name')} (event {matched_event.get('id')})")
        print(f"  competition['details'] -> {str(matched_comp.get('details'))[:400]}")

        # ---- 2. CORE api: every competition's status.result ----
        # This is where the method of victory actually lives. Dumping ALL
        # competitions (not just one) captures the full taxonomy in a
        # single run: a real card has submissions, KO/TKOs and decisions,
        # so one pass confirms every string the parser needs to map.
        event_id = matched_event.get("id")
        if event_id:
            core_event = _get(CORE_EVENT.format(event_id=event_id))
            out["core_api"] = {"event_keys": sorted(core_event.keys()) if isinstance(core_event, dict) else None}
            comps = core_event.get("competitions") if isinstance(core_event, dict) else None

            found = []
            if isinstance(comps, list):
                for comp in comps:
                    if not isinstance(comp, dict):
                        continue
                    status_node = comp.get("status")
                    ref = status_node.get("$ref") if isinstance(status_node, dict) else None
                    if not ref:
                        continue
                    status = _get(ref)
                    if not isinstance(status, dict):
                        continue
                    found.append({
                        "competition_id": comp.get("id"),
                        "weight_class": (comp.get("type") or {}).get("text"),
                        "competitor_ids": [c.get("id") for c in comp.get("competitors", []) if isinstance(c, dict)],
                        "winner_id": next((c.get("id") for c in comp.get("competitors", [])
                                           if isinstance(c, dict) and c.get("winner")), None),
                        "period": status.get("period"),
                        "displayClock": status.get("displayClock"),
                        "result": status.get("result"),
                    })
            out["core_api"]["all_competition_results"] = found
            print(f"core api: pulled result objects for {len(found)} competition(s)")
            for f in found:
                res = f.get("result") or {}
                print(f"   {str(f.get('weight_class') or '?'):18} "
                      f"name={res.get('name')!r} display={res.get('displayName')!r} "
                      f"desc={res.get('description')!r} R{f.get('period')} {f.get('displayClock')}")

    with open(OUT_PATH, "w") as f:
        json.dump(_shrink(out), f, indent=2)
    print(f"\nWrote {OUT_PATH} -- send that file back.")


if __name__ == "__main__":
    main()
