"""
Backfill striking / takedown columns from ESPN's PER-FIGHT statistics.

THE ENDPOINT, and why it took so long to find. ESPN publishes full per-fight
stats, but not where anyone looks first: `statsSummary.statistics` on the
athlete carries only three win-loss pairs, and reading that alone is how I
twice concluded ESPN had no rate stats at all. The real chain is:

    core/athletes/{id}/eventlog
      -> each entry has event / competition / COMPETITOR refs
        -> competitor ref, query stripped, + "/statistics"
          -> {knockDowns, sigStrikesLanded/Attempted, takedownsLanded/Attempted,
              sigDistance|Clinch|Ground x Head|Body|Leg Landed/Attempted, ...}

PER FIGHT, not career. That matters more than the extra columns: recency
weighting is the one validated model improvement this project has (+1.37pp),
and it can only be applied to stats that arrive per fight. ufcstats' career
means never could be.

TAKEDOWN DEFENCE NEEDS THE OPPONENT. "How often were you taken down" is the
other corner's takedownsLanded/Attempted in the SAME competition, so each
fight costs two fetches. Everything else comes from the fighter's own row.

TWO MODES:
  --discover   build/extend data/espn_athlete_ids.csv by reading every
               competitor off the scoreboards for dates we already know
               about. Uses only the scoreboard endpoint, which is proven --
               no guessing at a search API.
  (default)    walk the eventlog for every fighter that has an id and write
               the derived columns.

EVERY RESPONSE IS CACHED to data/.espn_cache/ keyed by URL. A full roster
backfill is thousands of requests; without a cache, one interruption means
starting over, and re-running to fix a parser bug would re-hammer ESPN.

Usage:
    python3 scripts/backfill_espn_fight_stats.py --discover
    python3 scripts/backfill_espn_fight_stats.py --card-only
    python3 scripts/backfill_espn_fight_stats.py --card-only --apply
    python3 scripts/backfill_espn_fight_stats.py --apply          # whole roster
"""

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import sys
import time
import unicodedata

import pandas as pd
import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.results_fetcher import BASE_HEADERS, REQUEST_TIMEOUT, ESPN_SCOREBOARD_URL  # noqa: E402

FIGHTERS = "data/fighters.csv"
HISTORY_FOR_TARGETS = "data/fight_history.csv"
ID_MAP = "data/espn_athlete_ids.csv"
CACHE_DIR = "data/.espn_cache"
EVENTLOG = "https://sports.core.api.espn.com/v2/sports/mma/leagues/ufc/athletes/{id}/eventlog"

# Matches the validated recency half-life already used by
# backfill_recency_stats.py. Kept identical on purpose: two different decay
# rates in one model would be indefensible, and 18 months is the swept value.
HALF_LIFE_DAYS = 548.0
REQUEST_DELAY = 0.4          # polite; thousands of calls
MIN_FIGHTS_FOR_STATS = 1     # a single tracked fight still beats a default

# MINIMUM DENOMINATORS. A percentage off a tiny denominator is noise wearing a
# number's clothes, and the first real run made that concrete: fighters with
# 3-4 tracked bouts came back at 100.0% takedown defence and 69.2% striking
# accuracy, while the 20-fight veteran on the same card sat at a thoroughly
# ordinary 40.8% / 58.2%. The extremes track sample size, not ability.
# Writing 100% defence into fighters.csv would hand the model a phantom elite
# grappler -- the same failure as the one-loss fighter scoring a perfect chin.
# Below these, the column is left None, which the both-corners gating in
# matchup_model already treats as "say nothing" rather than "zero".
# Precedent: backfill_recency_stats.py's effective-sample guard, added after
# an almost identical incident (td_per_15_r = 0.0 from almost no sample read
# as "never scores takedowns" and pushed a pick into Lock of the Week).
MIN_SIG_STRIKES_ATT = 100    # ~2 fights' worth of output
MIN_TD_ATT = 5               # own attempts, for accuracy
MIN_TD_ATT_FACED = 5         # opponents' attempts, for defence


def _fold(v) -> str:
    s = unicodedata.normalize("NFKD", str(v)).encode("ascii", "ignore").decode()
    return " ".join(s.lower().split())


def _cache_path(url: str) -> str:
    return os.path.join(CACHE_DIR, hashlib.sha1(url.encode()).hexdigest() + ".json")


def fetch(url: str, use_cache: bool = True):
    """GET with an on-disk cache. Returns parsed JSON or None."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = _cache_path(url)
    if use_cache and os.path.exists(path):
        try:
            with open(path) as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError):
            pass
    try:
        r = requests.get(url, headers=BASE_HEADERS, timeout=REQUEST_TIMEOUT)
        time.sleep(REQUEST_DELAY)
        if r.status_code != 200:
            return None
        data = r.json()
    except (requests.RequestException, ValueError):
        return None
    try:
        with open(path, "w") as f:
            json.dump(data, f)
    except OSError:
        pass
    return data


# ---------------------------------------------------------------- discover ids

def discover_ids():
    """
    Map fighter name -> ESPN athlete id by reading competitors off the
    scoreboard for every event date we already track.

    Deliberately NOT an athlete-search endpoint: guessing at undocumented
    endpoints is what produced two wrong conclusions in this project already.
    The scoreboard is proven, and every fighter we care about appeared on a
    card whose date we hold.
    """
    # EVENT dates only. The first version read `date_added` from
    # fight_results.csv, which is when a ROW WAS WRITTEN, not when the card
    # happened -- 10 distinct values, none of them a real event date, so every
    # scoreboard call came back empty. fight_history.csv carries a genuine
    # `date` per fight going back to 1994, which is exactly the set of dates
    # ESPN has cards for.
    dates = set()
    for path, cols in (("data/fight_history.csv", ("date",)),
                       ("data/fight_cards.csv", ("event_date",)),
                       ("data/future_cards.csv", ("event_date",)),
                       ("data/ufc_fight_results.csv", ("date", "event_date"))):
        try:
            df = pd.read_csv(path)
        except (FileNotFoundError, pd.errors.EmptyDataError):
            continue
        for c in cols:
            if c in df.columns:
                dates |= {str(v)[:10].replace("-", "") for v in df[c].dropna()}
                break

    dates = {d for d in dates if len(d) == 8 and d.isdigit()}
    print(f"[discover] {len(dates)} distinct event dates to scan")

    known = {}
    if os.path.exists(ID_MAP):
        # A zero-byte file is the normal residue of an earlier failed run, so
        # treat it as "no ids yet" rather than letting pandas raise. Guarding
        # only the reader in main() left this path to crash on exactly the
        # file the failed run had just created.
        try:
            prev = pd.read_csv(ID_MAP)
            known = {_fold(r["name"]): str(r["espn_id"]) for _, r in prev.iterrows()}
        except (pd.errors.EmptyDataError, KeyError):
            print(f"[discover] {ID_MAP} is empty or malformed -- starting fresh")
    before = len(known)

    for i, d in enumerate(sorted(dates), 1):
        data = fetch(f"{ESPN_SCOREBOARD_URL}?dates={d}")
        if not data:
            continue
        for ev in data.get("events") or []:
            for comp in ev.get("competitions") or []:
                for c in comp.get("competitors") or []:
                    ath = c.get("athlete") or {}
                    name = ath.get("fullName") or ath.get("displayName")
                    # THE ID IS ON THE COMPETITOR, NOT THE ATHLETE. The nested
                    # athlete object carries only names and links; the
                    # competitor wrapper holds {"id": "5097909", "type":
                    # "athlete"}. Reading athlete.id returned None for every
                    # fighter, which is why a 1,461-date scan produced zero
                    # ids from responses that were completely fine.
                    # The playercard link is a third fallback: its href ends
                    # /id/<id>/<slug>, so the id survives even if both fields
                    # move again.
                    aid = c.get("id") or ath.get("id")
                    if not aid:
                        for ln in ath.get("links") or []:
                            href = ln.get("href") or ""
                            if "/id/" in href:
                                aid = href.split("/id/")[1].split("/")[0]
                                break
                    if name and aid:
                        known.setdefault(_fold(name), str(aid))
        if i % 20 == 0:
            print(f"  [discover] {i}/{len(dates)} dates, {len(known)} ids")

    rows = [{"name": n, "espn_id": i} for n, i in sorted(known.items())]
    # Always write the header, even with zero rows -- a headerless empty file
    # makes the next read raise EmptyDataError instead of reporting "no ids".
    pd.DataFrame(rows, columns=["name", "espn_id"]).to_csv(ID_MAP, index=False)
    print(f"[discover] {len(known)} ids ({len(known) - before} new) -> {ID_MAP}")


# ------------------------------------------------------------------ stat walk

def _stats_from_competitor(comp_ref: str) -> dict:
    url = comp_ref.split("?")[0].rstrip("/") + "/statistics"
    data = fetch(url)
    if not data:
        return {}
    cats = (data.get("splits") or {}).get("categories") or data.get("categories") or []
    out = {}
    for c in cats:
        for s in c.get("stats") or []:
            name, val = s.get("name"), s.get("value")
            if name is not None and val is not None:
                out[name] = float(val)
    return out


def _opponent_ref(comp_ref: str, athlete_id: str) -> str | None:
    """The OTHER competitor under the same competition -- needed for TD defence."""
    competition_url = comp_ref.split("/competitors/")[0].split("?")[0]
    data = fetch(competition_url + "/competitors")
    if not data:
        return None
    for item in data.get("items") or []:
        ref = item.get("$ref", "")
        if f"/competitors/{athlete_id}" not in ref:
            return ref
    return None


def fighter_stats(athlete_id: str, name: str) -> dict | None:
    log = fetch(EVENTLOG.format(id=athlete_id))
    if not log:
        return None
    items = (log.get("events") or {}).get("items") or []
    if not items:
        return None

    tot = {k: 0.0 for k in ("ssl", "ssa", "tdl", "tda", "opp_tdl", "opp_tda", "kd",
                            "opp_ssl", "opp_kd")}
    wtot = dict(tot)
    fights = 0
    now = dt.datetime.now()

    for entry in items:
        if not entry.get("played"):
            continue
        comp_ref = (entry.get("competitor") or {}).get("$ref")
        if not comp_ref:
            continue
        mine = _stats_from_competitor(comp_ref)
        if not mine:
            continue
        fights += 1

        # Recency weight from the event date, same 18-month half-life as the
        # validated recency work. Undated fights fall back to weight 1 rather
        # than being dropped -- losing a fight entirely is worse than dating
        # it optimistically.
        weight = 1.0
        ev_ref = (entry.get("event") or {}).get("$ref")
        if ev_ref:
            ev = fetch(ev_ref)
            date_s = (ev or {}).get("date")
            if date_s:
                try:
                    age_days = (now - dt.datetime.fromisoformat(date_s.replace("Z", "+00:00")).replace(tzinfo=None)).days
                    weight = 0.5 ** (max(age_days, 0) / HALF_LIFE_DAYS)
                except ValueError:
                    pass

        opp_ref = _opponent_ref(comp_ref, athlete_id)
        theirs = _stats_from_competitor(opp_ref) if opp_ref else {}

        pairs = {
            "ssl": mine.get("sigStrikesLanded", 0.0),
            "ssa": mine.get("sigStrikesAttempted", 0.0),
            "tdl": mine.get("takedownsLanded", 0.0),
            "tda": mine.get("takedownsAttempted", 0.0),
            "kd": mine.get("knockDowns", 0.0),
            # The opponent's takedowns ARE this fighter's takedown defence.
            "opp_tdl": theirs.get("takedownsLanded", 0.0),
            "opp_tda": theirs.get("takedownsAttempted", 0.0),
            # And the opponent's LANDED STRIKES are this fighter's damage
            # taken -- the honest durability signal, measured every fight
            # rather than inferred from the handful of losses a fighter has.
            # Free: this payload was already being fetched for takedown
            # defence and the field was simply discarded.
            "opp_ssl": theirs.get("sigStrikesLanded", 0.0),
            "opp_kd": theirs.get("knockDowns", 0.0),
        }
        for k, v in pairs.items():
            tot[k] += v
            wtot[k] += v * weight

    if fights < MIN_FIGHTS_FOR_STATS:
        return None

    def pct(n, d, floor):
        """Percentage, or None when the denominator is too small to mean anything."""
        return round(n / d * 100, 1) if d and d >= floor else None

    return {
        "espn_fights": fights,
        "strike_accuracy_pct": pct(tot["ssl"], tot["ssa"], MIN_SIG_STRIKES_ATT),
        "td_accuracy_pct": pct(tot["tdl"], tot["tda"], MIN_TD_ATT),
        # Defence is the complement of the opponents' success rate. None when
        # too few takedowns were attempted against them -- that is "untested",
        # not "100% defence", the same absence-is-not-a-number rule the model
        # now enforces.
        "td_defense_pct": (round(100 - tot["opp_tdl"] / tot["opp_tda"] * 100, 1)
                           if tot["opp_tda"] >= MIN_TD_ATT_FACED else None),
        # Recency-weighted variants use the UNWEIGHTED denominator for the
        # threshold test: weighting shrinks old fights toward zero, so an
        # otherwise-adequate sample could fail a weighted floor purely for
        # being old, which is a different judgement than "too little data".
        "strike_accuracy_pct_r": (round(wtot["ssl"] / wtot["ssa"] * 100, 1)
                                  if tot["ssa"] >= MIN_SIG_STRIKES_ATT and wtot["ssa"] > 0 else None),
        "td_accuracy_pct_r": (round(wtot["tdl"] / wtot["tda"] * 100, 1)
                              if tot["tda"] >= MIN_TD_ATT and wtot["tda"] > 0 else None),
        "knockdowns_per_fight": round(tot["kd"] / fights, 3),
        "sig_strikes_att_per_fight": round(tot["ssa"] / fights, 2),
        "td_att_per_fight": round(tot["tda"] / fights, 2),
        # Damage taken, and knockdowns suffered. Per fight rather than per
        # minute: ESPN's payload carries no fight duration, and per-fight is
        # the honest unit for what it actually counts.
        "sig_strikes_absorbed_per_fight": round(tot["opp_ssl"] / fights, 2),
        "knockdowns_absorbed_per_fight": round(tot["opp_kd"] / fights, 3),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--discover", action="store_true", help="build the name->espn_id map first")
    ap.add_argument("--card-only", action="store_true", help="only fighters on data/fight_cards.csv")
    ap.add_argument("--from-history", action="store_true",
                    help="walk fighters from data/fight_history.csv (most-fought first) rather "
                         "than fighters.csv -- populates the cache for point-in-time validation")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    if args.discover:
        discover_ids()
        if not args.apply:
            return

    if not os.path.exists(ID_MAP):
        print(f"No {ID_MAP}. Run with --discover first.")
        sys.exit(1)

    try:
        id_df = pd.read_csv(ID_MAP)
    except pd.errors.EmptyDataError:
        id_df = pd.DataFrame(columns=["name", "espn_id"])
    if id_df.empty:
        print(f"{ID_MAP} is empty -- run --discover first (and check it reports a non-zero id count).")
        sys.exit(1)
    ids = {_fold(r["name"]): str(r["espn_id"]) for _, r in id_df.iterrows()}
    fighters = pd.read_csv(FIGHTERS)

    targets = list(fighters["name"])
    if args.from_history:
        # VALIDATION MODE. The point-in-time test needs BOTH fighters in a
        # historical bout to have a cached timeline, and walking only the
        # current roster (220 names) left an intersection of 160 scorable
        # fights out of 9,801 -- far too few to distinguish signal from
        # noise. Ordering by how often a fighter appears in history grows
        # that intersection fastest per request spent, since the most-fought
        # names are in the most bouts.
        # Writes to fighters.csv are unaffected: a fighter with no roster row
        # simply isn't written, but the CACHE is what the validator reads.
        hist = pd.read_csv(HISTORY_FOR_TARGETS)
        freq = pd.concat([hist["fighter_a"], hist["fighter_b"]]).value_counts()
        seen, ordered = set(), []
        for n in freq.index:
            if _fold(n) in ids and _fold(n) not in seen:
                seen.add(_fold(n))
                ordered.append(n)
        targets = ordered
        print(f"[from-history] {len(targets)} fighters with an ESPN id, most-fought first")
    if args.card_only:
        cards = pd.read_csv("data/fight_cards.csv")
        on = set(cards["fighter_a"]) | set(cards["fighter_b"])
        targets = [n for n in targets if n in on]
    if args.limit:
        targets = targets[:args.limit]

    print(f"{len(targets)} fighter(s) to process; {sum(1 for n in targets if _fold(n) in ids)} have an ESPN id\n")

    updates, missing_id, no_stats = {}, [], []
    for i, name in enumerate(targets, 1):
        aid = ids.get(_fold(name))
        if not aid:
            missing_id.append(name)
            continue
        s = fighter_stats(aid, name)
        if not s:
            no_stats.append(name)
            continue
        updates[name] = s
        print(f"  [{i}/{len(targets)}] {name}: {s['espn_fights']} fights, "
              f"strike {s['strike_accuracy_pct']}%, td_acc {s['td_accuracy_pct']}%, "
              f"td_def {s['td_defense_pct']}%")

    print(f"\n{len(updates)} with stats | {len(missing_id)} without an ESPN id | "
          f"{len(no_stats)} with an id but no usable stats")
    if missing_id[:10]:
        print(f"  no id: {missing_id[:10]}")

    if not args.apply:
        print("\nDRY RUN -- nothing written. Re-run with --apply.")
        return

    cols = set()
    for s in updates.values():
        cols |= set(s)
    for c in cols:
        if c not in fighters.columns:
            fighters[c] = None
        fighters[c] = fighters[c].astype("object")
    idx = {n: i for i, n in enumerate(fighters["name"])}
    for name, s in updates.items():
        if name not in idx:
            continue      # from-history target with no roster row; cache is the point
        for c, v in s.items():
            if v is not None:
                fighters.at[idx[name], c] = v
    fighters.to_csv(FIGHTERS, index=False)
    print(f"\nWrote {len(updates)} fighter(s) to {FIGHTERS}.")


if __name__ == "__main__":
    main()
