"""
Probe: does ESPN's athlete-detail endpoint carry the KO/SUB/DEC breakdown?

WHY THIS ISN'T A REPEAT. A previous attempt parsed the SCOREBOARD's
`records` array and found only a single 'overall' entry across ~80 fighters,
which got recorded as "a confirmed absence in the source itself." That
conclusion was too broad: it was true of ONE endpoint. ESPN's athlete page
plainly displays a method breakdown -- e.g. a 14-3-0 fighter shown with
KO 10-1 and SUB 4-1 -- so the data exists somewhere in their API.

src/fighter_backfill.py already calls the athlete-detail endpoint via
_fetch_espn_athlete_detail() and mines it only for height, reach and stance.
If the breakdown lives in that response, we are already paying for the
request and discarding the answer.

This is the same move that cracked method-of-victory: the site scoreboard had
nothing, the core api had everything. Probe first, then wire.

Prints EVERY records-like structure it finds rather than guessing at field
names, because guessing is what produced the over-broad conclusion last time.

Usage:
    python3 scripts/probe_espn_method_records.py "Vlasto Cepo"
    python3 scripts/probe_espn_method_records.py 4691323        # athlete id
"""

import json
import sys

import requests

HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"}
SEARCH = "https://site.web.api.espn.com/apis/common/v3/search"
ATHLETE = "https://sports.core.api.espn.com/v2/sports/mma/athletes/{}"
SITE_ATHLETE = "https://site.web.api.espn.com/apis/common/v3/sports/mma/ufc/athletes/{}"


def find_records(obj, path="", out=None):
    """Walk the whole payload for anything that looks like a record breakdown."""
    if out is None:
        out = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            p = f"{path}.{k}" if path else k
            kl = k.lower()
            if kl in ("records", "record", "stats", "statistics", "categories", "splits"):
                out.append((p, v))
            find_records(v, p, out)
    elif isinstance(obj, list):
        for i, v in enumerate(obj[:12]):
            find_records(v, f"{path}[{i}]", out)
    return out


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return
    arg = sys.argv[1]
    athlete_id = arg if arg.isdigit() else None

    if not athlete_id:
        # ESPN's search nests hits under results[].contents[]; the previous
        # string-splitting approach picked up unrelated "id" keys and returned
        # nothing. Walk the structure properly, and DUMP it when that fails so
        # the shape is visible rather than guessed at.
        try:
            r = requests.get(SEARCH, headers=HEADERS, timeout=20,
                             params={"query": arg, "limit": 10, "sport": "mma"})
            data = r.json()
            cands = []

            def walk(o):
                if isinstance(o, dict):
                    i, n = o.get("id"), o.get("displayName") or o.get("name")
                    if i and n and str(i).isdigit():
                        cands.append((str(i), str(n), o.get("type") or o.get("sport") or ""))
                    for v in o.values():
                        walk(v)
                elif isinstance(o, list):
                    for v in o:
                        walk(v)

            walk(data)
            if cands:
                print(f"search hits for {arg!r}:")
                for i, n, t in cands[:8]:
                    print(f"   id={i:<10} {n}  {t}")
                athlete_id = cands[0][0]
                print(f"\nusing id {athlete_id}")
            else:
                print(f"HTTP {r.status_code} -- no athlete ids in the response.")
                print("raw response (first 700 chars) so the shape is visible:")
                print(json.dumps(data)[:700])
        except Exception as e:
            print(f"search failed ({e})")

    if not athlete_id:
        print("\nFASTEST PATH: open the fighter on espn.com. The URL contains the id:")
        print("   espn.com/mma/fighter/_/id/<ID>/vlasto-cepo")
        print("Then re-run:  python3 scripts/probe_espn_method_records.py <ID>")
        return

    for label, url in (("core athlete", ATHLETE.format(athlete_id)),
                       ("site athlete", SITE_ATHLETE.format(athlete_id))):
        print(f"\n{'='*68}\n{label}: {url}\n{'='*68}")
        try:
            resp = requests.get(url, headers=HEADERS, timeout=20)
            print(f"HTTP {resp.status_code}")
            if resp.status_code != 200:
                continue
            data = resp.json()
            print("top-level keys:", sorted(data.keys())[:24])
            hits = find_records(data)
            if not hits:
                print("  no records/stats structures found")
            for p, v in hits[:10]:
                # Print raw content -- the point is to see what's actually
                # there, not to confirm a field name we already assumed.
                s = json.dumps(v)[:420]
                print(f"\n  {p}:\n    {s}")
        except Exception as e:
            print(f"  failed: {e}")

    print(f"\n{'='*68}")
    print("WHAT TO LOOK FOR: any entry whose summary/displayValue looks like a")
    print("method split -- '10-1' next to 'KO', '4-1' next to 'Submission'. If")
    print("one appears, send the path and I'll wire it into the backfill. Note")
    print("KO 10-1 + SUB 4-1 against a 14-3 record leaves DEC 0-1 by")
    print("subtraction, so two of the three splits are enough.")


if __name__ == "__main__":
    main()
