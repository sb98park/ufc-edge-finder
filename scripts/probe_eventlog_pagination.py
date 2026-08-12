"""
Is ESPN's athlete eventlog paginated?

WHY THIS MATTERS BEFORE ANYTHING ELSE. Sumudaerji's eventlog returned 24
fights; his ESPN history PAGE lists 27, and the three missing ones are the
oldest. ESPN's core API paginates with count / pageIndex / pageSize /
pageCount, and a default page size of 25 would produce exactly that symptom.

If the eventlog is paginated, the incompleteness is ours -- we read page one
and stop -- and the fix is a loop over an API we already use and already
cache. That is far better than scraping the history page, which is HTML that
may be client-rendered and can change layout without notice.

If it is NOT paginated and the eventlog genuinely holds fewer fights than the
history page, then the history page really is the better source and the
scraping tradeoff is worth taking knowingly.

Usage:
    python3 scripts/probe_eventlog_pagination.py "Sumudaerji"
"""

import os
import re
import sys
import unicodedata

import pandas as pd
import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.results_fetcher import BASE_HEADERS, REQUEST_TIMEOUT  # noqa: E402

EVENTLOG = "https://sports.core.api.espn.com/v2/sports/mma/leagues/ufc/athletes/{id}/eventlog"


def fold(v):
    s = unicodedata.normalize("NFKD", str(v)).encode("ascii", "ignore").decode()
    return " ".join(re.sub(r"[^a-z0-9 ]", " ", s.lower()).split())


def get(url, **params):
    try:
        r = requests.get(url, params=params or None, headers=BASE_HEADERS, timeout=REQUEST_TIMEOUT)
        return r.json() if r.status_code == 200 else None
    except (requests.RequestException, ValueError):
        return None


def main():
    target = sys.argv[1] if len(sys.argv) > 1 else "Sumudaerji"
    ids = {fold(r["name"]): str(r["espn_id"])
           for _, r in pd.read_csv("data/espn_athlete_ids.csv").iterrows()}
    aid = ids.get(fold(target))
    if not aid:
        print(f"No ESPN id for {target!r}")
        sys.exit(1)

    url = EVENTLOG.format(id=aid)
    data = get(url)
    if not data:
        print("eventlog fetch failed")
        sys.exit(1)

    ev = data.get("events") or {}
    print(f"{target} (id {aid})\n")
    for k in ("count", "pageIndex", "pageSize", "pageCount"):
        print(f"  {k:<12}{ev.get(k)}")
    items = ev.get("items") or []
    print(f"  items       {len(items)}")
    played = sum(1 for e in items if e.get("played"))
    print(f"  played      {played}")

    total = ev.get("count")
    pages = ev.get("pageCount") or 1
    if total and len(items) < total:
        print(f"\n  PAGINATED: page 1 holds {len(items)} of {total} across {pages} page(s).")
        print("  Everything we have ever derived from this endpoint has been")
        print("  page one only -- method splits, per-fight stats, durations.")
        allitems = list(items)
        for p in range(2, pages + 1):
            more = get(url, page=p)
            got = ((more or {}).get("events") or {}).get("items") or []
            print(f"     page {p}: {len(got)} item(s)")
            allitems += got
        pl = sum(1 for e in allitems if e.get("played"))
        print(f"\n  ALL PAGES: {len(allitems)} item(s), {pl} played")
        print("  -> fix is a page loop in whatever reads this endpoint, NOT scraping HTML.")
    else:
        print(f"\n  NOT paginated -- page one holds all {total} entries.")
        print("  If the history page still lists more fights than this, the eventlog")
        print("  genuinely omits them and the history page is the better source.")


if __name__ == "__main__":
    main()
