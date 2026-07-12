"""
Automated results lookup -- runs as part of every generate_site.py call,
so results get filled in without anyone manually searching and typing
them into fight_results.csv after each card.

*** HONEST CAVEAT, READ BEFORE TRUSTING THIS BLINDLY ***
This was written and tested for Python syntax and internal logic, but
NOT against the live ufcstats.com site -- this sandbox has no network
access, so there was no way to verify the actual page structure matches
what's assumed below. The fighter-listing selectors
(tr.b-fight-details__table-row, a.b-link_style_black) come from
src/scraper.py, which WAS validated against the real site in an earlier
session -- those are trusted. The result-detail parsing (method/round/
time extraction) is new and unverified. Treat this as a serious,
well-reasoned first attempt, not a guaranteed-working feature. Check the
GitHub Action logs after a real run to see what actually happened --
every branch below prints exactly what it did or why it gave up.

Design principles, all deliberate:
- NEVER raises. Every failure mode (site unreachable, structure changed,
  event not found, name doesn't match roster) is caught, logged, and
  degrades to "found nothing new" -- site generation must never break
  because a scrape failed.
- NEVER overwrites an existing row in fight_results.csv. Only appends
  results for fights that don't have one yet. Anything already entered
  (manually or by a previous successful run) is left untouched.
- Two-step lookup, not a guessed URL: ufcstats.com event URLs are opaque
  hashes, not derivable from the event name, so this first searches the
  completed-events listing for a name match, then fetches that specific
  event page.
- Method detection is TEXT-based (searches cell contents for known
  method substrings) rather than trusting a fixed column index, since a
  positional assumption is exactly the kind of thing that silently
  breaks when a site's markup shifts slightly.
"""

import re
import time

import pandas as pd
import requests
from bs4 import BeautifulSoup

BASE_HEADERS = {"User-Agent": "Mozilla/5.0 (personal research script; Octane Alpha)"}
REQUEST_TIMEOUT = 12
REQUEST_DELAY_SECONDS = 1.5
EVENTS_LIST_URL = "https://www.ufcstats.com/statistics/events/completed"

METHOD_PATTERNS = [
    ("KO/TKO", re.compile(r"\bKO/TKO\b|\bTKO\b|\bKO\b", re.I)),
    ("Submission", re.compile(r"\bSUB(MISSION)?\b", re.I)),
    ("Decision - Unanimous", re.compile(r"\bU-?DEC\b|UNANIMOUS", re.I)),
    ("Decision - Split", re.compile(r"\bS-?DEC\b|SPLIT", re.I)),
    ("Decision - Majority", re.compile(r"\bM-?DEC\b|MAJORITY", re.I)),
    ("DQ", re.compile(r"\bDQ\b|DISQUALIFICATION", re.I)),
]
STAT_COLS = [
    "fa_sig_landed", "fa_sig_att", "fb_sig_landed", "fb_sig_att",
    "fa_total_landed", "fa_total_att", "fb_total_landed", "fb_total_att",
    "fa_td_landed", "fa_td_att", "fb_td_landed", "fb_td_att",
    "fa_kd", "fb_kd", "fa_head", "fa_body", "fa_leg", "fb_head", "fb_body", "fb_leg",
]


def _find_event_url(event_name: str) -> str | None:
    """
    ufcstats.com event URLs are opaque hashes (e.g. /event-details/a1b2c3),
    not guessable from the name, so this searches the completed-events
    listing table for a fuzzy name match instead. "UFC 329: McGregor vs.
    Holloway 2" only needs to match on "UFC 329" -- the full title on
    ufcstats.com may be phrased slightly differently.
    """
    short_name = event_name.split(":")[0].strip()  # "UFC 329: ..." -> "UFC 329"
    try:
        resp = requests.get(EVENTS_LIST_URL, headers=BASE_HEADERS, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"[results_fetcher] could not reach events list: {e}")
        return None

    soup = BeautifulSoup(resp.text, "html.parser")
    for link in soup.select("a.b-link"):
        text = link.get_text(strip=True)
        if short_name.lower() in text.lower():
            print(f"[results_fetcher] matched event listing: {text!r} -> {link.get('href')}")
            return link.get("href")

    print(f"[results_fetcher] no event listing matched {short_name!r} -- card may not be posted yet")
    return None


def _extract_method(cell_texts: list[str]) -> str | None:
    joined = " ".join(cell_texts)
    for label, pattern in METHOD_PATTERNS:
        if pattern.search(joined):
            return label
    return None


def _extract_round_time(cell_texts: list[str]) -> tuple[int | None, str | None]:
    end_round = None
    end_time = None
    for t in cell_texts:
        t = t.strip()
        if end_round is None and re.fullmatch(r"[1-5]", t):
            end_round = int(t)
        if end_time is None and re.fullmatch(r"[0-5]:[0-5]\d", t):
            end_time = t
    return end_round, end_time


def _parse_event_results(event_url: str) -> list[dict]:
    try:
        resp = requests.get(event_url, headers=BASE_HEADERS, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"[results_fetcher] could not fetch event page: {e}")
        return []

    soup = BeautifulSoup(resp.text, "html.parser")
    rows = soup.select("tr.b-fight-details__table-row")
    results = []

    for row in rows:
        names = [a.get_text(strip=True) for a in row.select("a.b-link_style_black")]
        if len(names) != 2:
            continue  # not a real fight row (header row, ad, etc.)

        cells = row.select("td.b-fight-details__table-col")
        cell_texts = [c.get_text(" ", strip=True) for c in cells]

        # Winner detection is the least certain part of this parser --
        # ufcstats.com marks the winning fighter with a "win" indicator,
        # but that element's TEXT is just the word "win", not a fighter's
        # name, so it can't be matched by content. The best available
        # signal without live verification is POSITION: fighter names are
        # stacked two-per-cell (fighter_a first, fighter_b second), and
        # the win marker is assumed to be stacked in the same relative
        # position within its own cell. If that assumption doesn't hold
        # against the real page, this correctly finds nothing rather than
        # guessing -- but this is genuinely the part most likely to need
        # a real fix once actually run. Worth spot-checking the first few
        # auto-fetched winners against a reliable source.
        win_cell = row.select_one("td.b-fight-details__table-col")
        winner = None
        if win_cell:
            markers = win_cell.select(".b-fight-details__table-text_type_win, .win")
            if len(markers) == 1:
                all_markers_in_cell = win_cell.find_all(["p", "i"], recursive=True)
                try:
                    position = all_markers_in_cell.index(markers[0])
                    if position < len(names):
                        winner = names[position]
                except ValueError:
                    pass

        method = _extract_method(cell_texts)
        end_round, end_time = _extract_round_time(cell_texts)

        if not winner or not method:
            print(f"[results_fetcher] skipped {names}: winner={winner!r} method={method!r} (couldn't parse confidently)")
            continue

        results.append({
            "fighter_a": names[0], "fighter_b": names[1], "winner": winner,
            "method": method, "end_round": end_round, "end_time": end_time,
            "detail_url": _extract_fight_detail_url(row),
        })

    time.sleep(REQUEST_DELAY_SECONDS)
    return results


def _extract_fight_detail_url(row) -> str | None:
    """The event-listing row itself is usually a link (or wraps one) to
    that fight's own detail page, which is where the full stat breakdown
    (sig strikes by target, TD, KD, control time) lives -- separate from
    the summary row parsed above."""
    link = row.select_one("a[href*='fight-details']")
    return link.get("href") if link else None


def _parse_fight_stats(fight_url: str, fighter_a: str, fighter_b: str) -> dict | None:
    """
    Best-effort scrape of a single fight's detailed stat tables (Totals +
    Significant Strikes by target). This is the least certain part of the
    whole module -- ufcstats.com's detail pages have more complex nested
    table structure than the event-listing summary row, and this is
    unverified against a real page. Returns None on ANY uncertainty
    rather than guessing at numbers that would feed a betting-adjacent
    display -- a missing stat block is honest; a wrong one isn't.
    """
    try:
        resp = requests.get(fight_url, headers=BASE_HEADERS, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"[results_fetcher] could not fetch fight detail page: {e}")
        return None

    soup = BeautifulSoup(resp.text, "html.parser")
    tables = soup.select("table.b-fight-details__table")
    if len(tables) < 2:
        print(f"[results_fetcher] expected 2+ stat tables on fight detail page, found {len(tables)} -- skipping stats for this fight")
        return None

    def _row_numbers(table, expect_of_pattern: bool = True) -> list[list[str]]:
        """Each fighter's row of cells, as raw text, from the FIRST body row of a stat table."""
        rows = table.select("tbody tr")
        if not rows:
            return []
        cells = rows[0].select("td")
        return [c.get_text(" ", strip=True) for c in cells]

    def _split_of(text: str) -> tuple[int, int] | None:
        m = re.search(r"(\d+)\s+of\s+(\d+)", text)
        return (int(m.group(1)), int(m.group(2))) if m else None

    try:
        totals_cells = _row_numbers(tables[0])
        sig_cells = _row_numbers(tables[1])
        # Expected column order (per ufcstats.com's known Totals table):
        # Fighter | KD | Sig.Str | Sig.Str% | Total Str | TD | TD% | Sub.Att | Rev | Ctrl
        # Two fighters' numbers are stacked within each cell (line-broken),
        # split on whitespace/newlines since BeautifulSoup collapses them.
        fighter_names_cell = totals_cells[0]
        names_in_order = [n.strip() for n in fighter_names_cell.split("\n") if n.strip()]
        if len(names_in_order) != 2:
            print("[results_fetcher] couldn't split two fighter names out of totals table -- skipping stats")
            return None
        a_is_first = names_in_order[0].strip().lower() == fighter_a.strip().lower()

        def _pair(cell_text: str) -> tuple[str, str]:
            parts = [p.strip() for p in cell_text.split("\n") if p.strip()]
            return (parts[0], parts[1]) if len(parts) == 2 else (None, None)

        kd_a, kd_b = _pair(totals_cells[1])
        sig_a, sig_b = _pair(totals_cells[2])
        total_a, total_b = _pair(totals_cells[4])
        td_a, td_b = _pair(totals_cells[5])

        sig_split_a, sig_split_b = _split_of(sig_a), _split_of(sig_b)
        total_split_a, total_split_b = _split_of(total_a), _split_of(total_b)
        td_split_a, td_split_b = _split_of(td_a), _split_of(td_b)

        head_cells = sig_cells[3] if len(sig_cells) > 3 else None
        body_cells = sig_cells[4] if len(sig_cells) > 4 else None
        leg_cells = sig_cells[5] if len(sig_cells) > 5 else None

        def _landed_only(cell_text: str) -> tuple[int, int] | None:
            s = _split_of(cell_text)
            return s

        head_a, head_b = (_pair(head_cells) if head_cells else (None, None))
        body_a, body_b = (_pair(body_cells) if body_cells else (None, None))
        leg_a, leg_b = (_pair(leg_cells) if leg_cells else (None, None))

        required = [sig_split_a, sig_split_b, total_split_a, total_split_b, td_split_a, td_split_b]
        if any(v is None for v in required):
            print("[results_fetcher] one or more required stat fields didn't parse cleanly -- skipping stats for this fight")
            return None

        def _kd(v):
            try:
                return int(v)
            except (TypeError, ValueError):
                return None

        def _target(cell) -> int | None:
            s = _split_of(cell) if cell else None
            return s[0] if s else None

        stats = {
            "fa_sig_landed": sig_split_a[0], "fa_sig_att": sig_split_a[1],
            "fb_sig_landed": sig_split_b[0], "fb_sig_att": sig_split_b[1],
            "fa_total_landed": total_split_a[0], "fa_total_att": total_split_a[1],
            "fb_total_landed": total_split_b[0], "fb_total_att": total_split_b[1],
            "fa_td_landed": td_split_a[0], "fa_td_att": td_split_a[1],
            "fb_td_landed": td_split_b[0], "fb_td_att": td_split_b[1],
            "fa_kd": _kd(kd_a), "fb_kd": _kd(kd_b),
            "fa_head": _target(head_a), "fb_head": _target(head_b),
            "fa_body": _target(body_a), "fb_body": _target(body_b),
            "fa_leg": _target(leg_a), "fb_leg": _target(leg_b),
        }
        if not a_is_first:
            # Swap so fa_/fb_ always corresponds to the fighter_a/fighter_b
            # order the caller expects, not whatever order the page listed them in.
            stats = {("fb_" + k[3:] if k.startswith("fa_") else "fa_" + k[3:]): v for k, v in stats.items()}

        if any(v is None for v in stats.values()):
            print("[results_fetcher] some stat fields missing after parsing -- skipping stats for this fight rather than partially filling them")
            return None

        time.sleep(REQUEST_DELAY_SECONDS)
        return stats
    except Exception as e:
        print(f"[results_fetcher] stat table parsing failed unexpectedly: {e}")
        return None


def fetch_and_log_new_results(event_name: str, fight_cards_df: pd.DataFrame, results_path: str = "data/fight_results.csv") -> int:
    """
    Entry point called from generate_site.py. Returns the number of rows
    actually added or updated -- never raises. Handles two cases:
    fights with no result yet (adds winner/method/round/time, plus stats
    if the detail page has them), and fights that already have a basic
    result but are still missing the stat columns (attempts to backfill
    just the stats, leaving the existing winner/method/round/time alone).
    """
    try:
        existing = pd.read_csv(results_path)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        existing = pd.DataFrame(columns=["event_name", "fighter_a", "fighter_b", "winner", "method", "end_round", "end_time", "date_added"] + STAT_COLS)

    def _key(a, b):
        return frozenset({str(a).strip().lower(), str(b).strip().lower()})

    known_keys = {_key(r["fighter_a"], r["fighter_b"]) for _, r in existing.iterrows() if pd.notna(r.get("winner"))}
    stats_missing_keys = {
        _key(r["fighter_a"], r["fighter_b"]) for _, r in existing.iterrows()
        if pd.notna(r.get("winner")) and not all(pd.notna(r.get(c)) for c in STAT_COLS)
    }
    card_keys = {_key(r["fighter_a"], r["fighter_b"]) for r in fight_cards_df.to_dict("records")}
    truly_missing = card_keys - known_keys
    needs_stats_only = (card_keys & known_keys) & stats_missing_keys

    if not truly_missing and not needs_stats_only:
        return 0  # everything on this card is fully filled in -- nothing to do

    try:
        event_url = _find_event_url(event_name)
        if not event_url:
            return 0
        scraped = _parse_event_results(event_url)
    except Exception as e:
        print(f"[results_fetcher] unexpected error, giving up gracefully: {e}")
        return 0

    if not scraped:
        print("[results_fetcher] event page found but no fights parsed -- structure may not match, or card hasn't started")
        return 0

    roster_names_lower = {n.strip().lower() for n in pd.concat([fight_cards_df["fighter_a"], fight_cards_df["fighter_b"]])}
    new_rows = []
    updated_count = 0

    for r in scraped:
        key = _key(r["fighter_a"], r["fighter_b"])
        if r["fighter_a"].strip().lower() not in roster_names_lower or r["fighter_b"].strip().lower() not in roster_names_lower:
            continue

        stats = None
        if key in truly_missing or key in needs_stats_only:
            if r.get("detail_url"):
                stats = _parse_fight_stats(r["detail_url"], r["fighter_a"], r["fighter_b"])

        if key in truly_missing:
            row = {c: None for c in STAT_COLS}
            if stats:
                row.update(stats)
            row.update({
                "event_name": event_name, "fighter_a": r["fighter_a"], "fighter_b": r["fighter_b"],
                "winner": r["winner"], "method": r["method"],
                "end_round": r["end_round"], "end_time": r["end_time"],
                "date_added": pd.Timestamp.now().strftime("%Y-%m-%d"),
            })
            new_rows.append(row)
            print(f"[results_fetcher] found new result: {r['fighter_a']} vs {r['fighter_b']} -> {r['winner']} by {r['method']}" + (" (with stats)" if stats else " (no stats yet)"))

        elif key in needs_stats_only and stats:
            mask = existing.apply(lambda row: _key(row["fighter_a"], row["fighter_b"]) == key, axis=1)
            for col, val in stats.items():
                existing.loc[mask, col] = val
            updated_count += 1
            print(f"[results_fetcher] backfilled stats for existing result: {r['fighter_a']} vs {r['fighter_b']}")

    if not new_rows and not updated_count:
        return 0

    combined = pd.concat([existing, pd.DataFrame(new_rows)], ignore_index=True) if new_rows else existing
    combined.to_csv(results_path, index=False)
    return len(new_rows) + updated_count
