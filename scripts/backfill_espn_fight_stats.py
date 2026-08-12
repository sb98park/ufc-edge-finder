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
# Minutes of timed action before a per-minute rate means anything. Roughly
# two full three-round fights; below that a single early finish dominates.
MIN_TIMED_MINUTES = 25.0

# The nine positional x target fields ESPN already returns in the same payload
# the totals come from -- they were simply never extracted. POSITION is where
# the fight is happening (at range / clinched / on the mat); TARGET is what is
# being hit (head / body / legs). Together they are the one thing in this
# dataset that says what KIND of fighter someone is, which nothing on the site
# currently expresses: the radar says how he finishes, the waterfall says why
# he wins, neither says where he operates.
ZONE_FIELDS = {
    ("distance", "head"): "sigDistanceHeadStrikesLanded",
    ("distance", "body"): "sigDistanceBodyStrikesLanded",
    ("distance", "leg"):  "sigDistanceLegStrikesLanded",
    ("clinch", "head"):   "sigClinchHeadStrikesLanded",
    ("clinch", "body"):   "sigClinchBodyStrikesLanded",
    ("clinch", "leg"):    "sigClinchLegStrikesLanded",
    ("ground", "head"):   "sigGroundHeadStrikesLanded",
    ("ground", "body"):   "sigGroundBodyStrikesLanded",
    ("ground", "leg"):    "sigGroundLegStrikesLanded",
}

# Below this the shares are one fight's gameplan, not a fighter's habits.
MIN_ZONE_STRIKES = 60

# CONCENTRATION GUARD REMOVED after measuring what it cost.
#
# It rejected a rate when one fight supplied more than half a denominator.
# Motivated by Curtis Blaydes: 22 fights, 20 takedown attempts faced, 13 of
# them from ONE opponent, giving 35% takedown defence for a decorated
# wrestler. Measured across the roster, it threw out 58 of the 174 fighters
# who cleared the floor for td_defense_pct -- a THIRD. At these denominators
# (5-25 attempts) one fight contributing three of five is already 60%, so it
# fired on ordinary careers, not outliers.
#
# And the case that motivated it turned out to be a real measurement. The
# statistic is defence WHEN SOMEONE ATTEMPTS a takedown. Blaydes' reputation
# is that opponents don't shoot on him -- ~1 attempt per fight across 22
# fights -- and that avoidance is a different quantity from what happens when
# they do commit, where he has been taken down 13 times. The number honestly
# reports the second. It read as wrong because it was being asked the first
# question.
#
# Kept: the minimum-denominator floors, which target thin samples directly.
# The avoidance signal (attempts faced per fight) is worth surfacing on its
# own rather than smuggling into a defence rate.
# Why each column came back empty, tallied across the run. Coverage alone
# cannot separate "too few attempts" from "one fight dominated" -- and the two
# call for opposite fixes (lower the floor vs loosen the concentration rule),
# so a single coverage number is not enough to calibrate either.
REJECT_TALLY = {"floor": {}, "concentration": {}}


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


def _fight_minutes(comp_ref: str) -> float | None:
    """
    How long the fight actually lasted, in minutes.

    THE CLOCK COUNTS UP, verified rather than assumed. Two three-round
    decisions both return period=3, clock=300.0 -- a completed 15-minute
    fight. Elapsed gives 15.00; "remaining" would give 10.00, and a finished
    round would read 0:00 rather than 5:00. Getting this backwards would turn
    a round-one finish from 4.2 minutes into 0.8, a 5x error in the
    denominator of every rate, worst precisely on the finishers whose totals
    the correction exists to fix.

    `clock` is already seconds, so displayClock never needs parsing.
    """
    competition = comp_ref.split("/competitors/")[0].split("?")[0]
    st = fetch(competition + "/status")
    if not st:
        return None
    period, clock = st.get("period"), st.get("clock")
    if period is None or clock is None:
        return None
    try:
        return (int(period) - 1) * 5.0 + float(clock) / 60.0
    except (TypeError, ValueError):
        return None


def _zone_shares(zones: dict, prefix: str) -> dict:
    """
    Shares of landed strikes by TARGET (head/body/leg) and by POSITION
    (distance/clinch/ground).

    Shares, not counts, because the question is what KIND of striker this is,
    and a high-volume fighter would otherwise dominate every comparison purely
    on output. Returns None below MIN_ZONE_STRIKES rather than a share built
    from one night's gameplan.

    NOTE "distance" here means AT RANGE, the striking sense -- deliberately
    never surfaced to the reader as that word, because in betting "goes the
    distance" means reaching the judges, and the two would sit inches apart on
    the same card meaning opposite things.
    """
    total = sum(zones.values())
    if total < MIN_ZONE_STRIKES:
        return {f"{prefix}_{k}_share": None
                for k in ("head", "body", "leg", "distance", "clinch", "ground")}
    out = {}
    for target in ("head", "body", "leg"):
        v = sum(n for (pos, tgt), n in zones.items() if tgt == target)
        out[f"{prefix}_{target}_share"] = round(v / total * 100, 1)
    for position in ("distance", "clinch", "ground"):
        v = sum(n for (pos, tgt), n in zones.items() if pos == position)
        out[f"{prefix}_{position}_share"] = round(v / total * 100, 1)
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


def eventlog_items(athlete_id: str) -> list:
    """
    EVERY page of a fighter's eventlog, not just the first.

    ESPN's core API paginates at pageSize 25 and reports count / pageIndex /
    pageCount alongside the items. Reading `items` and stopping meant every
    fighter with more than 25 professional fights was silently truncated to
    his 25 most recent -- and because the oldest fights are the ones dropped,
    the loss always fell on the early-career end.

    Found via Sumudaerji: eventlog reported 24 played fights, his ESPN history
    page listed 27. count=28, pageSize=25, pageCount=2. Page two held exactly
    the three missing bouts.

    This is not a niche case. Everything derived from this endpoint inherited
    the truncation: per-fight striking and takedown stats, fight durations and
    so every per-minute rate, the zone shares behind the striking profile, and
    both validation harnesses. A veteran's numbers were computed from a career
    that stopped 25 fights ago.
    """
    log = fetch(EVENTLOG.format(id=athlete_id))
    if not log:
        return []
    ev = log.get("events") or {}
    items = list(ev.get("items") or [])
    try:
        pages = int(ev.get("pageCount") or 1)
    except (TypeError, ValueError):
        pages = 1
    for page in range(2, pages + 1):
        more = fetch(f"{EVENTLOG.format(id=athlete_id)}?page={page}")
        items += ((more or {}).get("events") or {}).get("items") or []
    return items


def fighter_stats(athlete_id: str, name: str) -> dict | None:
    items = eventlog_items(athlete_id)
    if not items:
        return None

    tot = {k: 0.0 for k in ("ssl", "ssa", "tdl", "tda", "opp_tdl", "opp_tda", "kd",
                            "opp_ssl", "opp_kd", "minutes")}
    # Per-fight denominators, so concentration is measured rather than assumed
    # even when the total looks healthy.
    per_fight = {"ssa": [], "tda": [], "opp_tda": []}
    # Offensive zones (where he lands) and defensive zones (where he is hit).
    # The defensive side is free: the opponent's row is already fetched for
    # takedown defence, and its zone fields describe damage taken.
    zones = {k: 0.0 for k in ZONE_FIELDS}
    opp_zones = {k: 0.0 for k in ZONE_FIELDS}
    # Totals restricted to fights whose duration is known, so a rate's
    # numerator and denominator always describe the same set of bouts.
    timed = {"ssl": 0.0, "tdl": 0.0, "opp_ssl": 0.0, "minutes": 0.0}
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

        # Duration, for per-minute rates. Fights where ESPN gives no usable
        # status contribute their TOTALS but not their minutes, which would
        # inflate every rate -- so they are excluded from both sides instead.
        minutes = _fight_minutes(comp_ref)

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
            "minutes": minutes or 0.0,
        }
        if minutes is None:
            # No duration: keep the per-FIGHT figures, drop this bout from the
            # per-MINUTE ones. Counting its strikes against zero minutes would
            # be worse than omitting it.
            timed_totals_skip = True
        else:
            timed_totals_skip = False
        for k, v in pairs.items():
            if timed_totals_skip and k == "minutes":
                continue
            tot[k] += v
            wtot[k] += v * weight
        if not timed_totals_skip:
            timed["ssl"] += pairs["ssl"]
            timed["tdl"] += pairs["tdl"]
            timed["opp_ssl"] += pairs["opp_ssl"]
            timed["minutes"] += minutes
        for k in per_fight:
            per_fight[k].append(pairs[k])
        for key, field in ZONE_FIELDS.items():
            zones[key] += float(mine.get(field, 0.0) or 0.0)
            opp_zones[key] += float(theirs.get(field, 0.0) or 0.0)

    if fights < MIN_FIGHTS_FOR_STATS:
        return None

    def _tally_def(total):
        """Record that td_defense_pct fell under the floor, then return None."""
        REJECT_TALLY["floor"]["opp_tda"] = REJECT_TALLY["floor"].get("opp_tda", 0) + 1
        return None

    def pct(n, d, floor, key=None):
        """Percentage, or None when the sample is too small OR too concentrated."""
        if not d or d < floor:
            if key:
                REJECT_TALLY["floor"][key] = REJECT_TALLY["floor"].get(key, 0) + 1
            return None
        return round(n / d * 100, 1)

    return {
        "espn_fights": fights,
        "strike_accuracy_pct": pct(tot["ssl"], tot["ssa"], MIN_SIG_STRIKES_ATT, "ssa"),
        "td_accuracy_pct": pct(tot["tdl"], tot["tda"], MIN_TD_ATT, "tda"),
        # Defence is the complement of the opponents' success rate. None when
        # too few takedowns were attempted against them -- that is "untested",
        # not "100% defence", the same absence-is-not-a-number rule the model
        # now enforces.
        "td_defense_pct": (round(100 - tot["opp_tdl"] / tot["opp_tda"] * 100, 1)
                           if tot["opp_tda"] >= MIN_TD_ATT_FACED
                           else _tally_def(tot["opp_tda"])),
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
        # Attempts FACED, which is a different quantity from takedown DEFENCE
        # and the one that actually captures avoidance. Blaydes shows 35%
        # defence -- damning for a wrestler -- off just 20 attempts faced in
        # 22 fights. The defence rate answers "what happens when someone
        # shoots"; this answers "does anyone dare". Conflating them cost a
        # round of analysis, so both are stored.
        "td_att_faced_per_fight": round(tot["opp_tda"] / fights, 2),
        # Damage taken, and knockdowns suffered. Per fight rather than per
        # minute: ESPN's payload carries no fight duration, and per-fight is
        # the honest unit for what it actually counts.
        "sig_strikes_absorbed_per_fight": round(tot["opp_ssl"] / fights, 2),
        **_zone_shares(zones, "strikes"),
        **_zone_shares(opp_zones, "absorbed"),
        # PER-MINUTE rates, the inputs a strike/takedown projection needs.
        # A per-FIGHT average conflates pace with fight length: a finisher's
        # totals are suppressed by the very trait that makes him dangerous.
        # Separating rate from duration lets the projection recombine them
        # with the round-survival grid the method model already produces.
        "fight_minutes_total": round(timed["minutes"], 1) if timed["minutes"] else None,
        "sig_strikes_landed_per_min": (round(timed["ssl"] / timed["minutes"], 3)
                                       if timed["minutes"] >= MIN_TIMED_MINUTES else None),
        "sig_strikes_absorbed_per_min": (round(timed["opp_ssl"] / timed["minutes"], 3)
                                         if timed["minutes"] >= MIN_TIMED_MINUTES else None),
        "td_landed_per_15": (round(timed["tdl"] / timed["minutes"] * 15, 3)
                             if timed["minutes"] >= MIN_TIMED_MINUTES else None),
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
    print("\nWHY COLUMNS CAME BACK EMPTY")
    labels = {"ssa": "strike_accuracy_pct", "tda": "td_accuracy_pct", "opp_tda": "td_defense_pct"}
    print(f"  {'column':<22}{'too few':>10}")
    for key, label in labels.items():
        print(f"  {label:<22}{REJECT_TALLY['floor'].get(key, 0):>10}")
    print("  All rejections are now sample-size only; the concentration guard was removed.")
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
            # WRITE None TOO. Skipping it meant a value could never be
            # CLEARED: once written, a stat survived even after a later run
            # judged it unusable. Curtis Blaydes kept a 35.0 takedown defence
            # that the concentration guard had just rejected, so the guard
            # looked broken when it was the write that was.
            # Safe because this script is now the sole source of these
            # columns -- the ufcstats scraper that used to fill them is dead
            # behind a JS challenge -- so there is no other writer whose work
            # a None could erase.
            fighters.at[idx[name], c] = v if v is not None else None
    fighters.to_csv(FIGHTERS, index=False)
    print(f"\nWrote {len(updates)} fighter(s) to {FIGHTERS}.")


if __name__ == "__main__":
    main()
