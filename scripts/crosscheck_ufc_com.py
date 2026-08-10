"""
Cross-check the tracked card against UFC.com, the promotion's own listing.

WHY. ESPN drives the pipeline and it lags. On UFC Fight Night: Gamrot vs
Salkilld it missed a bout added four days out (the card OPENER), missed a
one-day replacement, and carried an outright wrong opponent -- three hand
corrections on one event, each found by the user noticing rather than by
anything in the system. UFC.com had all three right, is server-rendered Drupal
with no bot challenge, and is authoritative rather than crowd-edited.

REPORTS ONLY. NEVER WRITES. That is deliberate and worth keeping: a markup
change on UFC.com must not be able to silently rewrite a card, and the hand
corrections in fight_cards.csv (a corrected opponent, a manually added bout)
are the user's judgement and outrank a scraper's. This prints what disagrees;
a human decides what to do about it. scripts/add_fight_manually.py and
scripts/mark_title_fight.py are how changes actually get made.

WHAT IT CHECKS
  - bouts on UFC.com missing from our card       (a late addition we lack)
  - bouts on our card missing from UFC.com       (a cancellation, or our error)
  - card_position disagreements                  (segment moved)
  - title-fight disagreements                    (is_title_fight wrong)

Title status matters beyond the badge: it drives is_five_round, so a title
fight flagged wrongly gets a three-round round-distribution and every
Over/Under built on it is wrong.

Events are matched by DATE, not by name -- "UFC Fight Night: Gamrot vs
Salkilld" and UFC.com's "Gamrot vs Salkilld" are the same card under
different strings, and dates are unambiguous.

Usage:
    python3 scripts/crosscheck_ufc_com.py
    python3 scripts/crosscheck_ufc_com.py --event ufc-330
"""

import argparse
import os
import re
import sys
import time
import unicodedata

import pandas as pd
import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.results_fetcher import BASE_HEADERS, REQUEST_TIMEOUT  # noqa: E402

try:
    from bs4 import BeautifulSoup
except ImportError:
    print("BeautifulSoup needed:  pip3 install beautifulsoup4")
    sys.exit(1)

EVENTS_URL = "https://www.ufc.com/events"
EVENT_URL = "https://www.ufc.com/event/{}"
CARD_FILES = ["data/fight_cards.csv", "data/future_cards.csv"]
ACK_FILE = "data/crosscheck_ack.csv"
REQUEST_DELAY = 0.5

# Which container a bout sits in -> our card_position vocabulary.
SEGMENT_IDS = {"main-card": "Main Card", "prelims-card": "Prelims", "early-prelims": "Early Prelims"}

# OUR VOCABULARY IS FINER THAN THEIRS. UFC.com groups every bout into just
# Main Card / Prelims / Early Prelims; we additionally distinguish Main Event
# and Co-Main Event, which ARE main-card bouts. Comparing the raw strings
# flagged both headliners on every single card -- eight false positives that
# buried the three real discrepancies. A refinement is not a disagreement.
POSITION_EQUIVALENT = {
    "Main Event": "Main Card",
    "Co-Main Event": "Main Card",
}


def _fold(v) -> str:
    s = unicodedata.normalize("NFKD", str(v)).encode("ascii", "ignore").decode()
    return " ".join(re.sub(r"[^a-z0-9 ]", " ", s.lower()).split())


def _loose(name: str) -> tuple:
    """
    First and last token, SORTED, as a fallback identity for one fighter.

    Two real mismatches this handles, both seen on live cards:
      - nicknames:      UFC.com "Michael Venom Page"  vs ours "Michael Page"
      - reversed order: UFC.com "Xiong Jingnan"       vs ours "Jingnan Xiong"
    Sorting the pair makes the second case symmetric. Deliberately NOT fuzzy
    string distance -- two different fighters sharing an exact first AND last
    token on one card is far rarer than either variation above, and a fuzzy
    matcher would quietly pair the wrong people.
    """
    parts = _fold(name).split()
    if not parts:
        return ()
    return tuple(sorted({parts[0], parts[-1]}))


def _pair(a, b) -> frozenset:
    return frozenset({_fold(a), _fold(b)})


def _loose_pair(a, b) -> frozenset:
    return frozenset({_loose(a), _loose(b)})


def _get(url: str):
    try:
        r = requests.get(url, headers=BASE_HEADERS, timeout=REQUEST_TIMEOUT)
        time.sleep(REQUEST_DELAY)
    except requests.RequestException as e:
        print(f"  fetch failed {url}: {e}")
        return None
    if r.status_code != 200:
        print(f"  HTTP {r.status_code} for {url}")
        return None
    return r.text


def event_slugs() -> list[str]:
    html = _get(EVENTS_URL)
    if not html:
        return []
    slugs = set(re.findall(r'/event/([a-z0-9\-]+)', html))
    # The index also yields ticketing ids and stray numbers; a real event slug
    # always carries letters.
    return sorted(s for s in slugs if re.search(r"[a-z]", s))


def parse_event(slug: str) -> dict | None:
    """{'date': 'YYYY-MM-DD', 'bouts': [ {...} ]} or None."""
    html = _get(EVENT_URL.format(slug))
    if not html:
        return None
    soup = BeautifulSoup(html, "html.parser")

    date = None
    t = soup.find("time", attrs={"datetime": True})
    if t:
        date = t["datetime"][:10]

    bouts = []
    for seg_id, position in SEGMENT_IDS.items():
        container = soup.find(id=seg_id)
        if not container:
            continue
        for fight in container.select(".c-listing-fight"):
            # SEPARATOR MATTERS. The corner-name element wraps given and family
            # names in separate child spans, so get_text(strip=True) returns
            # "IslamMakhachev" -- which matches nothing on our side, and made
            # every single fight report as both missing AND extra.
            names = [n.get_text(" ", strip=True) for n in fight.select(".c-listing-fight__corner-name")]
            # Each corner's name appears more than once in the markup (mobile
            # and desktop rows), so dedupe while preserving order rather than
            # taking the first two blindly.
            seen, corners = set(), []
            for n in names:
                if n and _fold(n) not in seen:
                    seen.add(_fold(n))
                    corners.append(n)
            if len(corners) < 2:
                continue
            label_el = fight.select_one(".c-listing-fight__class-text")
            label = label_el.get_text(strip=True) if label_el else ""
            bouts.append({
                "fighter_a": corners[0],
                "fighter_b": corners[1],
                "card_position": position,
                "is_title_fight": "title bout" in label.lower(),
                "weight_class": re.sub(r"\s*(Title\s*)?Bout\s*$", "", label, flags=re.I).strip(),
            })
    return {"date": date, "bouts": bouts} if bouts else None


def load_acknowledged() -> dict:
    """
    {pair key: note} for discrepancies a human has already reviewed and judged.

    NEEDED BECAUSE NEITHER SOURCE DOMINATES. UFC.com beat ESPN on three late
    changes on one card -- an added opener, a one-day replacement, a wrong
    opponent -- which is why this cross-check exists. But the reverse happens
    too: Charles Johnson vs Jose Ochoa was on ESPN and priced by DraftKings
    while UFC.com never listed it. A book does not post a line on a bout that
    does not exist, so ESPN was right and UFC.com was incomplete.
    Without this list that fight reports every single run until the card
    passes, and a checker that always prints the same known non-problem is one
    people stop reading -- which is how the next REAL discrepancy gets missed.
    """
    try:
        df = pd.read_csv(ACK_FILE)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        return {}
    return {_pair(r["fighter_a"], r["fighter_b"]): str(r.get("note", "")).strip()
            for _, r in df.iterrows()}


def load_ours() -> dict:
    """{date: [rows]} from both card files."""
    by_date = {}
    for path in CARD_FILES:
        try:
            df = pd.read_csv(path)
        except (FileNotFoundError, pd.errors.EmptyDataError):
            continue
        if "event_date" not in df.columns:
            continue
        for _, r in df.iterrows():
            d = str(r["event_date"])[:10]
            by_date.setdefault(d, []).append(r)
    return by_date


def apply_title_flags(writes: list[tuple]) -> int:
    """
    Write is_title_fight for the listed bouts, across both card files.

    THE ONE EXCEPTION to this script's report-only rule, and it earns it.
    Everywhere else the two sources genuinely disagree and neither dominates.
    Title status is different: UFC.com states it outright ("Welterweight Title
    Bout") on the promotion's own page, and it is the one field whose being
    wrong corrupts the MODEL rather than the display -- is_title_fight drives
    is_five_round, so a title co-main flagged as three rounds gets a wrong
    round distribution and wrong Over/Unders with no visible error anywhere.
    Still opt-in via --apply-titles; the default run writes nothing.
    """
    written = 0
    for path in CARD_FILES:
        try:
            df = pd.read_csv(path)
        except (FileNotFoundError, pd.errors.EmptyDataError):
            continue
        if "is_title_fight" not in df.columns:
            df["is_title_fight"] = None
        df["is_title_fight"] = df["is_title_fight"].astype("object")
        changed = False
        for fa, fb, want in writes:
            mask = df.apply(lambda r: _pair(r["fighter_a"], r["fighter_b"]) == _pair(fa, fb)
                            or _loose_pair(r["fighter_a"], r["fighter_b"]) == _loose_pair(fa, fb), axis=1)
            if mask.any():
                df.loc[mask, "is_title_fight"] = want
                written += int(mask.sum())
                changed = True
        if changed:
            df.to_csv(path, index=False)
            print(f"  wrote {path}")
    return written


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--event", help="single UFC.com slug, e.g. ufc-330")
    ap.add_argument("--apply-titles", action="store_true",
                    help="WRITE is_title_fight from UFC.com for matched bouts. "
                         "The only field this script will ever write.")
    args = ap.parse_args()

    ours = load_ours()
    acked = load_acknowledged()
    if not ours:
        print("No tracked cards found.")
        sys.exit(1)
    print(f"tracked event dates: {sorted(ours)}\n")

    slugs = [args.event] if args.event else event_slugs()
    print(f"checking {len(slugs)} UFC.com event page(s)\n")

    total_issues = 0
    title_writes = []
    for slug in slugs:
        parsed = parse_event(slug)
        if not parsed or not parsed["date"]:
            continue
        date = parsed["date"]
        if date not in ours:
            continue        # not a card we track; silence is correct

        print(f"=== {slug}  ({date}) ===")
        theirs = parsed["bouts"]
        mine = ours[date]

        their_keys = {_pair(b["fighter_a"], b["fighter_b"]): b for b in theirs}
        my_keys = {_pair(r["fighter_a"], r["fighter_b"]): r for r in mine}
        # Loose index, consulted only when the exact key misses.
        my_loose = {_loose_pair(r["fighter_a"], r["fighter_b"]): r for r in mine}
        their_loose = {_loose_pair(b["fighter_a"], b["fighter_b"]): b for b in theirs}

        matched_pairs = []
        for k, b in their_keys.items():
            if k in my_keys:
                matched_pairs.append((b, my_keys[k]))
                continue
            lk = _loose_pair(b["fighter_a"], b["fighter_b"])
            if lk in my_loose:
                r = my_loose[lk]
                matched_pairs.append((b, r))
                print(f"  name variant: UFC.com {b['fighter_a']!r}/{b['fighter_b']!r} "
                      f"= ours {r['fighter_a']!r}/{r['fighter_b']!r}")
                continue
            # Neither exact nor loose matched: UFC.com has a bout we don't.
            total_issues += 1
            print(f"  MISSING FROM OUR CARD: {b['fighter_a']} vs {b['fighter_b']} "
                  f"({b['card_position']}, {b['weight_class']}"
                  f"{', TITLE' if b['is_title_fight'] else ''})")
            print(f"     -> scripts/add_fight_manually.py --event \"<name>\" "
                  f"--fighter-a \"{b['fighter_a']}\" --fighter-b \"{b['fighter_b']}\" "
                  f"--weight-class \"{b['weight_class']}\" --position \"{b['card_position']}\"")

        for k, r in my_keys.items():
            if k in their_keys or _loose_pair(r["fighter_a"], r["fighter_b"]) in their_loose:
                continue
            # A cancelled bout SHOULD be absent from UFC.com -- that is the
            # expected state, not a discrepancy.
            if str(r.get("cancelled", "")).strip().lower() == "true":
                continue
            if k in acked:
                print(f"  (acknowledged) {r['fighter_a']} vs {r['fighter_b']} -- {acked[k]}")
                continue
            total_issues += 1
            print(f"  NOT ON UFC.COM: {r['fighter_a']} vs {r['fighter_b']} "
                  f"({r.get('card_position')}) -- cancelled, or wrong on our side")

        for b, r in matched_pairs:
            ours_pos = str(r.get("card_position", "")).strip()
            if POSITION_EQUIVALENT.get(ours_pos, ours_pos) != b["card_position"]:
                total_issues += 1
                print(f"  POSITION: {b['fighter_a']} vs {b['fighter_b']} -- "
                      f"ours {ours_pos!r}, UFC.com {b['card_position']!r}"
                      f"  (a real segment change, not the Main Event/Main Card wording)")
            ours_title = str(r.get("is_title_fight", "")).strip().lower() == "true"
            if ours_title != b["is_title_fight"]:
                print(f"  TITLE FLAG: {b['fighter_a']} vs {b['fighter_b']} -- "
                      f"ours {ours_title}, UFC.com {b['is_title_fight']}")
                if args.apply_titles:
                    title_writes.append((r["fighter_a"], r["fighter_b"], b["is_title_fight"]))
                    print(f"     -> WILL WRITE is_title_fight={b['is_title_fight']}")
                else:
                    total_issues += 1
                print(f"     -> affects is_five_round, so the round/total grid is wrong too")
                print(f"     -> scripts/mark_title_fight.py \"{b['fighter_a']}\" \"{b['fighter_b']}\""
                      f"{'' if b['is_title_fight'] else ' --unset'}")
        print()

    if title_writes:
        n = apply_title_flags(title_writes)
        print(f"\nWrote is_title_fight on {n} row(s). Re-run generate_site.py -- this "
              f"changes is_five_round, so the round and total grids are rebuilt.")
    print(f"\n{total_issues} discrepancy(ies) needing a human. "
          f"{'Title flags were written.' if title_writes else 'Nothing was written.'}")


if __name__ == "__main__":
    main()
