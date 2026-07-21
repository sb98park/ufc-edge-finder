"""
Manual last-fight backfill (July 2026).

WHY THIS EXISTS: the "Last Fight" row in the fighter comparison was blank
for essentially everyone on the card except Ankalaev. Root cause: the only
code path that fills last_fight_* is results_fetcher, which populates it as
a SIDE EFFECT of a completed fight flowing through the pipeline -- it fires
forward-only and never fetches a fighter's most-recent historical bout at
roster-build time. The path that was meant to (ESPN eventLog backfill) has
never returned usable data. So until the automated eventLog follower is
confirmed working from a real production log, this fills the gap by hand.

IMPORTANT -- MORE PERISHABLE THAN METHOD DATA: a fighter's KO/Sub/Dec
career totals barely change, but "last fight" changes every single time
they compete. These values are correct as of July 2026; each fighter's
next bout makes their entry here stale. This is a stopgap, not a
permanent data source -- the real fix is the automated last-fight source.

Only fills genuinely-empty last_fight_* cells (never overwrites existing
data), same rule as the method-breakdown backfill. Run:
    python3 scripts/manual_backfill_last_fight.py
Then regenerate + push as usual (generate_site.py, commit, pull --no-rebase,
push).

Each tuple: (result, method, opponent, date)
  result  : "W" or "L" (from this fighter's perspective)
  method  : short human label matching results_fetcher's convention
  opponent: full name
  date    : ISO YYYY-MM-DD
All nine verified directly against each fighter's Sherdog fight-history page.
"""
import pandas as pd

FIGHTERS_PATH = "data/fighters.csv"

# result, method, opponent, date  -- all Sherdog-confirmed, July 2026
MANUAL_LAST_FIGHT = {
    "Abubakar Vagaev":    ("W", "Decision (Unanimous)", "Albert Tumenov",   "2025-02-08"),
    "Saygid Izagakhmaev": ("L", "Decision (Split)",     "Nicolas Dalby",    "2025-11-22"),
    "Muhammad Said":      ("W", "KO (Punches)",         "Henrique da Silva","2026-05-09"),
    "Ismael Bonfim":      ("L", "TKO (Punches)",        "Chris Padilla",    "2025-11-08"),
    "Axel Sola":          ("L", "Decision (Unanimous)", "Mason Jones",      "2026-03-21"),
    "Valter Walker":      ("W", "Submission (Heel Hook)","Louie Sutherland","2025-10-25"),
    "Thomas Petersen":    ("W", "Decision (Majority)",  "Guilherme Pat",    "2026-04-04"),
    "Steve Erceg":        ("W", "Decision (Unanimous)", "Tim Elliott",      "2026-05-02"),
    "Rizvan Kuniev":      ("W", "Decision (Unanimous)", "Jailton Almeida",  "2026-02-07"),
}

LAST_FIGHT_COLS = ["last_fight_result", "last_fight_method", "last_fight_opponent", "last_fight_date"]


def _find_row_index(fighters: pd.DataFrame, name: str):
    exact = fighters.index[fighters["name"] == name]
    if len(exact) > 0:
        return exact[0]
    # loose fallback: first+last word, case-insensitive -- catches a slightly
    # different stored spelling the same way the method-breakdown script does
    def key(s):
        parts = str(s).split()
        return (parts[0].lower(), parts[-1].lower()) if parts else ("", "")
    target = key(name)
    for idx, csv_name in fighters["name"].items():
        if key(csv_name) == target:
            return idx
    return None


def main():
    fighters = pd.read_csv(FIGHTERS_PATH)
    for col in LAST_FIGHT_COLS:
        if col not in fighters.columns:
            fighters[col] = pd.NA

    filled_count = 0
    for name, (result, method, opponent, date) in MANUAL_LAST_FIGHT.items():
        idx = _find_row_index(fighters, name)
        if idx is None:
            print(f"[manual_last_fight] '{name}' not found in {FIGHTERS_PATH} (exact or loose) -- skipping. "
                  f"Is this fighter tracked yet?")
            continue

        values = {
            "last_fight_result": result,
            "last_fight_method": method,
            "last_fight_opponent": opponent,
            "last_fight_date": date,
        }
        filled_here = []
        for col, val in values.items():
            if pd.isna(fighters.at[idx, col]):
                fighters[col] = fighters[col].astype(object)
                fighters.at[idx, col] = val
                filled_here.append(col)

        if filled_here:
            filled_count += 1
            print(f"[manual_last_fight] filled {name}: {', '.join(filled_here)}")
        else:
            print(f"[manual_last_fight] {name} already had all last-fight fields -- left untouched")

    fighters.to_csv(FIGHTERS_PATH, index=False)
    print(f"[manual_last_fight] done -- {filled_count}/{len(MANUAL_LAST_FIGHT)} fighters had at least one field filled")
    print(f"[manual_last_fight] NOTE: last-fight data is perishable -- it goes stale the next time each of "
          f"these fighters competes. This is a stopgap until the automated eventLog source is confirmed.")
    print(f"[manual_last_fight] IMPORTANT: this only updated {FIGHTERS_PATH}. Still need generate_site.py + "
          f"commit + git pull --no-rebase + push for it to show live.")


if __name__ == "__main__":
    main()
