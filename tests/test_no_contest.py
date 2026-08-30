"""
Draws and no-contests, end to end.

WHAT WENT WRONG. Michael Aljarouj's last fight showed 2021-03-18 on the site
while Tapology had him fighting 2025-04-12 -- a no contest. There were ZERO
winnerless rows in 11,861 history rows and 102 result rows, which reads like
"the UFC rarely has one" and is actually "we have never been able to store
one." Three separate mechanisms:

  1. results_fetcher SKIPPED a completed bout with no winner. Nothing was
     written, and the convergence guards counted only rows with a winner, so
     the pairing stayed in truly_missing on every subsequent run: refetched
     forever, coverage stuck at n-1/n, the bout never leaving live mode.
  2. pit_roster.build_fight_index computed `won = (name == winner)`, which is
     False for BOTH fighters when the winner is empty -- so a no contest
     charged both of them a defeat in every point-in-time backtest.
  3. Nothing downstream had ever been shown the row shape, so nobody knew
     whether it was safe to start writing one.

These tests pin all three, plus the consumers that (3) turned out to be about.
"""

import sys, os, json, tempfile
import datetime as dt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd  # noqa: E402

from src import results_fetcher  # noqa: E402

FAILURES = []


def check(label, got, want):
    ok = got == want
    print(f"  {'ok  ' if ok else 'FAIL'} {label:62s} got {got!r}")
    if not ok:
        FAILURES.append(f"{label}: got {got!r}, want {want!r}")


# ---------------------------------------------------------------- _is_settled
print("\na row is settled when it has a winner OR says it never will")
check("decision", results_fetcher._is_settled(
    {"winner": "A", "method": "Decision - Unanimous"}), True)
check("no contest, empty winner", results_fetcher._is_settled(
    {"winner": "", "method": "NC"}), True)
check("draw, empty winner", results_fetcher._is_settled(
    {"winner": "", "method": "Draw"}), True)
check("no contest spelled out", results_fetcher._is_settled(
    {"winner": "", "method": "No Contest"}), True)
# NaN is what pandas hands back for an empty CSV cell, and it is TRUTHY --
# `row.get("winner") or ...` would have called this settled.
check("NaN winner, NC method", results_fetcher._is_settled(
    {"winner": float("nan"), "method": "nc"}), True)
check("NaN winner, no method -- genuinely still missing", results_fetcher._is_settled(
    {"winner": float("nan"), "method": float("nan")}), False)
check("blank winner, blank method", results_fetcher._is_settled(
    {"winner": "", "method": ""}), False)


# ------------------------------------------------- the ESPN scoreboard branch
def _scoreboard(desc, winner_flags):
    """One completed bout between Alpha and Bravo."""
    return {"events": [{
        "name": "UFC Test Event",
        "competitions": [{
            "status": {"type": {"completed": True, "description": desc}},
            "competitors": [
                {"winner": winner_flags[0], "athlete": {"fullName": "Alpha Test"}},
                {"winner": winner_flags[1], "athlete": {"fullName": "Bravo Test"}},
            ],
        }],
    }]}


class _Resp:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


def _fetch(payload, health_path):
    real_get, real_sleep = results_fetcher.requests.get, results_fetcher.time.sleep
    cwd = os.getcwd()
    results_fetcher.requests.get = lambda *a, **k: _Resp(payload)
    results_fetcher.time.sleep = lambda *a, **k: None
    try:
        # _record_no_winner writes data/source_health.json relative to cwd.
        os.chdir(health_path)
        os.makedirs("data", exist_ok=True)
        return results_fetcher._fetch_from_espn(
            "UFC Test Event", "2026-04-12", {"alpha test", "bravo test"})
    finally:
        os.chdir(cwd)
        results_fetcher.requests.get, results_fetcher.time.sleep = real_get, real_sleep


print("\na completed bout with no winner comes back as a row, not as nothing")
with tempfile.TemporaryDirectory() as tmp:
    rows = _fetch(_scoreboard("No Contest", [False, False]), tmp)
    check("one row returned", len(rows), 1)
    if rows:
        check("winner is empty", rows[0]["winner"], "")
        check("method is NC", rows[0]["method"], "NC")
        check("both fighters kept", [rows[0]["fighter_a"], rows[0]["fighter_b"]],
              ["Alpha Test", "Bravo Test"])
        check("no round claimed", rows[0]["end_round"], None)
    health = os.path.join(tmp, "data", "source_health.json")
    check("still recorded as a diagnostic", os.path.exists(health), True)
    if os.path.exists(health):
        with open(health, encoding="utf-8") as fh:
            check("under no_winner_bouts", list(json.load(fh)["no_winner_bouts"])[:1],
                  ["UFC Test Event|Alpha Test vs Bravo Test"])

with tempfile.TemporaryDirectory() as tmp:
    rows = _fetch(_scoreboard("Draw - Majority", [False, False]), tmp)
    check("a draw is a Draw, not an NC", rows[0]["method"] if rows else None, "Draw")

with tempfile.TemporaryDirectory() as tmp:
    nameless = _scoreboard("No Contest", [False, False])
    nameless["events"][0]["competitions"][0]["competitors"][1]["athlete"] = {}
    rows = _fetch(nameless, tmp)
    check("an unnamed competitor writes no row", len(rows), 0)


# ------------------------------------------------------------- it converges
print("\nand a card holding that row stops asking ESPN for it")
cols = ["event_name", "fighter_a", "fighter_b", "winner", "method"]
existing = pd.DataFrame([
    {"event_name": "UFC Test Event", "fighter_a": "Alpha Test",
     "fighter_b": "Bravo Test", "winner": None, "method": "NC"},
], columns=cols)
# The pair key is defined inside fetch_and_log_new_results; this is that
# expression, and the guard under test is _is_settled feeding it.
def _key(a, b):
    return frozenset({str(a).strip().lower(), str(b).strip().lower()})


settled = {_key(r["fighter_a"], r["fighter_b"])
           for _, r in existing.iterrows() if results_fetcher._is_settled(r)}
card_key = _key("Alpha Test", "Bravo Test")
check("the no-contest counts as known", card_key in settled, True)
check("so truly_missing is empty", sorted({card_key} - settled), [])


# ------------------------------------------------------ pit_roster's arithmetic
print("\na no contest is charged to neither fighter's record")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "scripts"))
import pit_roster  # noqa: E402

hist = pd.DataFrame([
    {"date": "2024-01-01", "fighter_a": "Alpha Test", "fighter_b": "Bravo Test",
     "winner": "Alpha Test", "method": "KO/TKO", "weight_class": "Lightweight"},
    {"date": "2025-04-12", "fighter_a": "Alpha Test", "fighter_b": "Charlie Test",
     "winner": "", "method": "NC", "weight_class": "Lightweight"},
])
hist["date"] = pd.to_datetime(hist["date"])   # what every caller hands it
idx = pit_roster.build_fight_index(hist)
alpha = pit_roster.record_as_of(idx, "Alpha Test", pd.Timestamp("2026-01-01"))
charlie = pit_roster.record_as_of(idx, "Charlie Test", pd.Timestamp("2026-01-01"))
check("the winner of the decided fight is 1-0",
      (alpha["wins"], alpha["losses"]), (1, 0))
check("his opponent in the no contest is 0-0",
      (charlie["wins"], charlie["losses"]), (0, 0))
check("but the no contest is still his last fight",
      str(charlie["last_fight_date"])[:10], "2025-04-12")
check("and it reads NC, not L", charlie["last_fight_result"], "NC")
check("the no contest is the winner's last fight too",
      str(alpha["last_fight_date"])[:10], "2025-04-12")


# ------------------------------------------------ the roster sync (fighters.csv)
print("\nthe roster clock moves, the roster record does not")
roster = pd.DataFrame([
    {"name": "Alpha Test", "wins": 5, "losses": 1, "ko_wins": 2, "sub_wins": 1,
     "dec_wins": 2, "ko_losses": 1, "sub_losses": 0, "dec_losses": 0,
     "last_fight_date": "2021-03-18", "last_fight_opponent": "Old Foe",
     "last_fight_result": "W", "last_fight_method": "KO/TKO", "short_notice": 1},
    {"name": "Charlie Test", "wins": 3, "losses": 3, "ko_wins": 1, "sub_wins": 1,
     "dec_wins": 1, "ko_losses": 1, "sub_losses": 1, "dec_losses": 1,
     "last_fight_date": "2020-01-01", "last_fight_opponent": "Someone",
     "last_fight_result": "L", "last_fight_method": "SUB", "short_notice": 0},
])
synced = results_fetcher.sync_fighter_records(
    roster.copy(), "Alpha Test", "Charlie Test", "", "NC", "2025-04-12")
a = synced[synced["name"] == "Alpha Test"].iloc[0]
c = synced[synced["name"] == "Charlie Test"].iloc[0]
check("no win is credited", (int(a["wins"]), int(a["losses"])), (5, 1))
check("no loss is charged", (int(c["wins"]), int(c["losses"])), (3, 3))
check("but the last fight date moves", str(a["last_fight_date"]), "2025-04-12")
check("for both of them", str(c["last_fight_date"]), "2025-04-12")
check("and reads NC", a["last_fight_result"], "NC")
check("with no method invented", pd.isna(a["last_fight_method"]), True)
check("short-notice flag still cleared", int(a["short_notice"]), 0)
# quick_return_penalty keys on exactly "L" plus a finish method. An NC has
# neither, so a fighter returning fast from a no contest is not penalised as
# though they had just been knocked out.
from src import matchup_model  # noqa: E402
check("no quick-return penalty after an NC",
      matchup_model.quick_return_penalty(a, dt.date(2025, 7, 1)), 0.0)
ko_loss = a.copy()
ko_loss["last_fight_result"], ko_loss["last_fight_method"] = "L", "KO/TKO"
check("but the same fighter coming off a KO loss is still penalised",
      matchup_model.quick_return_penalty(ko_loss, dt.date(2025, 7, 1)) < 0, True)

drawn = results_fetcher.sync_fighter_records(
    roster.copy(), "Alpha Test", "Charlie Test", "", "Draw", "2025-04-12")
check("a draw says Draw", drawn[drawn["name"] == "Alpha Test"].iloc[0]["last_fight_result"],
      "Draw")


# ------------------------------------------------------- recent-form scoring
print("\nan NC is not a defeat in the recent-form term")
form_hist = pd.DataFrame([
    {"date": "2025-06-01", "fighter_a": "Alpha Test", "fighter_b": "Bravo Test",
     "winner": "Alpha Test", "method": "KO/TKO"},   # inside the 2-year decay
    {"date": "2025-11-01", "fighter_a": "Alpha Test", "fighter_b": "Charlie Test",
     "winner": "", "method": "NC"},
])
ref = dt.date(2026, 1, 1)
with_nc = matchup_model.recent_form_adjustment(
    "Alpha Test", "Bravo Test", form_hist, ref)
without = matchup_model.recent_form_adjustment(
    "Alpha Test", "Bravo Test", form_hist.iloc[:1], ref)
check("adding a no contest changes nothing", round(with_nc, 6), round(without, 6))
check("and the surviving win still scores positive for him", with_nc > 0, True)


# -------------------------------------------------------- the other consumers
print("\nthe consumers that made this unsafe to write before")
from src import elo  # noqa: E402
ratings = elo.EloRatingSystem().build_from_history(hist)
check("elo invents no fighter from a winnerless row",
      sorted(n for n in ratings if "test" in n.lower()),
      ["Alpha Test", "Bravo Test"])
check("and Charlie, who only ever no-contested, gets no rating",
      "Charlie Test" in ratings, False)

from src import track_record  # noqa: E402
src = open(track_record.__file__, encoding="utf-8").read()
check("track_record still guards on winner before grading",
      'if not result.get("winner"):' in src, True)


print("\n" + ("-" * 72))
if FAILURES:
    print(f"{len(FAILURES)} FAILURE(S)")
    for f in FAILURES:
        print("   ", f)
    sys.exit(1)
print("no contests survive the fetcher, the ledger guards and the backtest")
