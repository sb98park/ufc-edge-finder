"""
Per-fighter UFC bout history, shaped for the scouting drawer.

WHAT THIS IS FOR. The site could say a lot about a fighter's rates and nothing
about how any individual fight actually went. data/ufc_fight_stats.csv holds
41,672 per-ROUND rows covering every UFC bout, and until now it was read only
by builder scripts -- nothing on the page consumed it. This turns it into the
one thing a reader scouting Saturday's card actually wants: pick a fighter,
see who they have fought, and see the shape of each of those fights.

WHY EVERY BOUT AND NOT THE LAST FIVE. Measured, because it is the obvious
place to economise and the economy is not there:

    all bouts, every booked fighter     47.7 KB raw   11.5 KB gzipped
    last 5 bouts only                   23.5 KB raw    6.2 KB gzipped

Truncating saves five kilobytes and costs the only thing that makes the
component worth building. Anthony Hernandez's history reads as five climbing
performances and then a cliff against Strickland; at "last 3" you get the
cliff, two climbs, and no trajectory. Fighters with the longest careers --
Gastelum at 27 bouts, Vera at 26 -- are exactly the ones whose arc is worth
reading. So: all of it.

THE SHARE NUMBER, AND WHAT IT IS NOT. Each round gets one figure: this
fighter's share of the round's MEASURABLE OUTPUT, being significant strikes
landed and control time. It is deliberately NOT "who won the round". Judges
score on criteria this data does not carry -- a round where someone is dropped
once and does nothing else can be scored 10-9 against the fighter who landed
more -- and a bar labelled "round winner" would be claiming knowledge the
inputs cannot support. The UI labels it as output share and says so.

    weight = sig_strikes_landed * 3 + control_seconds / 10

The weighting is a judgement call and worth stating plainly rather than hiding
in a constant: three points per landed strike against one point per ten
seconds of control puts a full five-minute round of top control (30 points)
near a ten-strike round (30 points). It is a legible exchange rate, not a
fitted one, and nothing downstream depends on it being exactly right.

COMPLETENESS, measured rather than assumed. Control time is absent on 1.1% of
all 41,672 rounds (474), and every one of those is pre-2001 -- the era before
ufcstats recorded it. From 2001 onward it is absent on 0.0%. The earliest bout
belonging to any currently booked fighter is 2010-06-19, so for the fighters
this site shows, coverage is complete. Note that roughly a third of modern
rounds record 0:00 of control: that is a real striking round, not a gap, and
treating it as missing would have understated coverage badly.
"""

from __future__ import annotations

import csv
import re
import unicodedata
from collections import defaultdict
from datetime import datetime

STATS = "data/ufc_fight_stats.csv"
RESULTS = "data/ufc_fight_results.csv"
EVENTS = "data/ufc_event_details.csv"

# Letters NFKD will not decompose, so the ascii backstop would DELETE them.
# Same map and same reason as scripts/build_pit_stats.py -- see the note there.
_STROKE_FOLD = str.maketrans({
    "ł": "l", "Ł": "L", "ø": "o", "Ø": "O",
    "đ": "d", "Đ": "D", "ħ": "h", "Ħ": "H",
    "ŧ": "t", "Ŧ": "T", "ß": "ss",
    "æ": "ae", "Æ": "AE", "œ": "oe", "Œ": "OE",
    "ı": "i",
})


def fold_name(name) -> str:
    """
    Fold a fighter name to a lookup key.

    ufcstats publishes ASCII ("Joel Alvarez") while the roster carries the
    real spelling ("Joel Álvarez"), so this has to strip diacritics or every
    accented fighter silently resolves to nothing -- which is exactly the bug
    that left 13 roster fighters with no control-time data at all.
    """
    s = str(name).strip().translate(_STROKE_FOLD)
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    return s.encode("ascii", "ignore").decode().lower()


def _landed(cell) -> int:
    """'29 of 62' -> 29. Anything unparseable -> 0."""
    try:
        return int(str(cell or "0 of 0").split(" of ")[0])
    except (ValueError, IndexError):
        return 0


def _ctrl_seconds(cell):
    """
    'm:ss' -> seconds. Returns None when the field is ABSENT ('--' or empty),
    which is different from a genuine 0:00. Only pre-2001 bouts are absent, and
    the distinction matters: counting a real striking round as missing data
    would misreport coverage as ~67% when it is ~99%.
    """
    s = str(cell or "").strip()
    if ":" not in s:
        return None
    try:
        m, sec = s.split(":")
        return int(m) * 60 + int(sec)
    except ValueError:
        return None


def _scheduled_rounds(time_format) -> int | None:
    """
    How many rounds the bout was SCHEDULED for, from ufc_fight_results'
    TIME FORMAT column.

    Counted from the parenthetical rather than the leading number, because the
    leading number lies on the old formats: "1 Rnd + OT (12-3)" is two periods
    and "3 Rnd + OT (5-5-5-5)" is four. Counting the segments is exact for
    every format in the file, modern or otherwise. "No Time Limit" (31 bouts,
    all pre-2000) has no scheduled count and returns None.
    """
    m = re.search(r"\(([^)]*)\)", str(time_format or ""))
    if not m:
        return None
    segs = [p for p in m.group(1).split("-") if p.strip()]
    return len(segs) or None


def _event_dates() -> dict:
    """EVENT -> ISO date, from ufc_event_details.csv."""
    out = {}
    try:
        for row in csv.DictReader(open(EVENTS, encoding="utf-8")):
            raw = (row.get("DATE") or "").strip()
            try:
                out[row["EVENT"].strip()] = datetime.strptime(raw, "%B %d, %Y").strftime("%Y-%m-%d")
            except ValueError:
                continue
    except FileNotFoundError:
        # The spine refresh supplies this file. Without it every bout still
        # renders, just unsorted and undated, so this is degraded not fatal.
        pass
    return out


def build_fighter_history(names, stats_path=STATS, results_path=RESULTS) -> dict:
    """
    Bout history for `names`, keyed by folded name.

    Returns {folded: [bout, ...]} newest first, where a bout is:

        o    opponent display name
        e    event name, for the expanded header
        sr   rounds the bout was SCHEDULED for (None if no time limit).
             A five-round fight that ended in the fourth still had a fifth,
             and the shape of the fight is not the same as a three-rounder
             that went the distance -- the UI draws the unfought rounds.
        w    1 won, 0 lost, None neither (draw, no contest, unrecorded)
        m    method ("KO/TKO", "Submission", "Decision - Unanimous", ...)
        er   round the fight ended in
        d    ISO date
        rs   [share of measurable output, per round, 0-100]
        sg   [this fighter's sig strikes, opponent's]
        ct   [this fighter's control seconds, opponent's]  (-1 when absent)
        td   [this fighter's takedowns, opponent's]

    Keys are short because this ships to the browser on every page load.
    """
    wanted = {fold_name(n) for n in names if str(n).strip()}
    if not wanted:
        return {}

    dates = _event_dates()

    results = {}
    for row in csv.DictReader(open(results_path, encoding="utf-8")):
        # EVENT carries a trailing space in this file and not in the stats
        # file -- an unstripped join matches exactly zero of 781 events.
        results[(row["EVENT"].strip(), row["BOUT"].strip())] = row

    # (event, bout) -> round -> fighter -> row
    grouped = defaultdict(lambda: defaultdict(dict))
    for row in csv.DictReader(open(stats_path, encoding="utf-8")):
        key = (row["EVENT"].strip(), row["BOUT"].strip())
        grouped[key][row["ROUND"].strip()][row["FIGHTER"].strip()] = row

    history = defaultdict(list)
    for (event, bout), rounds in grouped.items():
        # "A vs. B" -- the only place the two corners are named together.
        parts = [p.strip() for p in bout.split(" vs. ")]
        if len(parts) != 2:
            continue
        for i, who in enumerate(parts):
            folded = fold_name(who)
            if folded not in wanted:
                continue
            opponent = parts[1 - i]

            res = results.get((event, bout), {})
            outcome = (res.get("OUTCOME") or "").strip()
            # OUTCOME is relative to the CORNER ("W/L" means fighter A won),
            # not to the fighter being asked about.
            won = None
            if outcome == "W/L":
                won = 1 if i == 0 else 0
            elif outcome == "L/W":
                won = 0 if i == 0 else 1

            shares, sig, ctrl, tds = [], [0, 0], [0, 0], [0, 0]
            ctrl_absent = False
            for rname in sorted(rounds, key=lambda r: int(r.split()[-1]) if r.split()[-1].isdigit() else 0):
                mine = rounds[rname].get(who)
                theirs = rounds[rname].get(opponent)
                if not mine or not theirs:
                    continue
                m_sig, t_sig = _landed(mine["SIG.STR."]), _landed(theirs["SIG.STR."])
                m_ctl, t_ctl = _ctrl_seconds(mine.get("CTRL")), _ctrl_seconds(theirs.get("CTRL"))
                if m_ctl is None or t_ctl is None:
                    ctrl_absent = True
                m_ctl, t_ctl = m_ctl or 0, t_ctl or 0

                mine_w = m_sig * 3 + m_ctl / 10.0
                their_w = t_sig * 3 + t_ctl / 10.0
                total = mine_w + their_w
                # A round with no measurable output at all (it happens: a
                # 20-second knockout where the finishing strike is the only
                # entry) is drawn at 50 rather than dividing by zero.
                shares.append(round(100 * mine_w / total) if total else 50)

                sig[0] += m_sig; sig[1] += t_sig
                ctrl[0] += m_ctl; ctrl[1] += t_ctl
                tds[0] += _landed(mine.get("TD")); tds[1] += _landed(theirs.get("TD"))

            if not shares:
                continue

            history[folded].append({
                "o": opponent,
                "e": event,
                "sr": _scheduled_rounds(res.get("TIME FORMAT")),
                "w": won,
                "m": (res.get("METHOD") or "").strip(),
                "er": (res.get("ROUND") or "").strip(),
                "d": dates.get(event, ""),
                "rs": shares,
                "sg": sig,
                "ct": [-1, -1] if ctrl_absent else ctrl,
                "td": tds,
            })

    # Newest first. Undated bouts sort last rather than jumbling the top of
    # the list, which is the part a reader actually looks at.
    for folded in history:
        history[folded].sort(key=lambda b: b["d"] or "0000-00-00", reverse=True)
    return dict(history)


def summarise(history: dict) -> dict:
    """Coverage figures, for the build log. Cheap and worth printing."""
    bouts = sum(len(v) for v in history.values())
    rounds = sum(len(b["rs"]) for v in history.values() for b in v)
    no_ctrl = sum(1 for v in history.values() for b in v if b["ct"] == [-1, -1])
    return {"fighters": len(history), "bouts": bouts, "rounds": rounds,
            "bouts_without_control": no_ctrl}
