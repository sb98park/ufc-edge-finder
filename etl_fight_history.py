"""
ETL: raw UFC dataset (data/raw/) -> data/fight_history.csv + last_fight_date refresh.

Phase (b) of the data migration: fight HISTORY and recency only. Career
stats in fighters.csv (SLpM, accuracy, records, etc.) are deliberately
NOT touched -- that's phase (a), gated on backtest results from this
phase looking sane first.

Design decisions, each verified against the actual data before coding:

- FULL GRAPH, not roster-only: every decisive UFC fight goes into
  fight_history.csv, not just fights involving our roster. Elo's whole
  value is that opponent strength propagates through the network -- a
  win over a contender and a win over a debutant only look different if
  the opponents' own results are in the graph. This directly addresses
  the strength-of-schedule gap from the ML audit.

- Draw/NC rows EXCLUDED from fight history: build_from_history() infers
  the loser as "whichever fighter isn't the winner", so a winner of
  'Draw/NC' would create a phantom fighter named "Draw/NC" who beats
  someone every time -- poisoning ratings. (elo.py now also guards
  against this defensively.) Draws still count toward last_fight_date,
  since the fighter was genuinely active that night.

- 'Overturned' rows EXCLUDED from fight history: those results were
  legally voided (typically failed drug tests). Counting a voided win
  would credit a result that officially never happened. Also still
  counts for last_fight_date (the cage time was real).

- Name normalization maps the raw dataset's spellings to OUR roster
  spellings (e.g. "Benoit Saint Denis" -> "Benoit Saint-Denis"), because
  every downstream lookup (Elo, recency, predict_matchup) joins on the
  names in fighters.csv. Non-roster fighters keep raw spellings.

- last_fight_date is overwritten for matched roster fighters with the
  REAL most recent UFC fight date -- in whichever direction that moves
  it. If the real data says a fighter has been out for years, that's a
  fact the layoff factor should see, not a value to keep comfortable.
  Unmatched fighters (UFC debutants) keep their curated values, since
  their recent activity happened outside the UFC where this dataset
  can't see it.

Run: python3 etl_fight_history.py
"""

import pandas as pd

RAW_GOLD_PATH = "data/raw/ufc_gold_dataset_final.csv"
FIGHTERS_PATH = "data/fighters.csv"
FIGHT_HISTORY_PATH = "data/fight_history.csv"

# Raw-dataset spelling -> our roster spelling. Applied to Fighter_1,
# Fighter_2, and Winner so downstream name joins against fighters.csv work.
NAME_FIXES = {
    "Benoit Saint Denis": "Benoit Saint-Denis",
    "Kai Kamaka": "Kai Kamaka III",
}


def map_method(raw_method: str) -> str:
    if raw_method.startswith("Decision"):
        return "DEC"
    if raw_method in ("KO/TKO", "TKO - Doctor's Stoppage"):
        return "KO/TKO"
    if raw_method == "Submission":
        return "SUB"
    if raw_method == "DQ":
        return "DQ"
    return "OTHER"  # falls through to Elo's default K multiplier (1.0)


def main():
    gold = pd.read_csv(RAW_GOLD_PATH)
    fighters = pd.read_csv(FIGHTERS_PATH)
    roster = set(fighters["name"])

    for col in ("Fighter_1", "Fighter_2", "Winner"):
        gold[col] = gold[col].replace(NAME_FIXES)
    gold["Event_Date"] = pd.to_datetime(gold["Event_Date"])

    # --- Namesake guard: a roster fighter sharing a name with another
    # fighter in the raw data would silently merge two people's records.
    raw_fighters = pd.read_csv("data/raw/ufc_fighters_final.csv")
    dup_names = set(raw_fighters[raw_fighters["Fighter_Name"].duplicated(keep=False)]["Fighter_Name"])
    collisions = roster & dup_names
    if collisions:
        raise SystemExit(f"ABORT: roster names collide with duplicate names in raw data: {collisions}")

    # --- Fight history (decisive results only) ---
    total = len(gold)
    draws = gold["Winner"] == "Draw/NC"
    overturned = gold["Method"] == "Overturned"
    decisive = gold[~draws & ~overturned].copy()
    print(f"Gold rows: {total} | excluded {draws.sum()} Draw/NC + {(overturned & ~draws).sum()} additional Overturned")

    # Sanity gate: every remaining winner must be one of the two listed fighters.
    bad = decisive[(decisive["Winner"] != decisive["Fighter_1"]) & (decisive["Winner"] != decisive["Fighter_2"])]
    if not bad.empty:
        raise SystemExit(f"ABORT: {len(bad)} rows have a winner matching neither fighter:\n{bad.head()}")

    history = pd.DataFrame({
        "date": decisive["Event_Date"].dt.strftime("%Y-%m-%d"),
        "fighter_a": decisive["Fighter_1"],
        "fighter_b": decisive["Fighter_2"],
        "winner": decisive["Winner"],
        "method": decisive["Method"].map(map_method),
    }).sort_values("date").reset_index(drop=True)
    history.to_csv(FIGHT_HISTORY_PATH, index=False)
    print(f"Wrote {len(history)} fights to {FIGHT_HISTORY_PATH} ({history['date'].min()} -> {history['date'].max()})")

    # --- last_fight_date refresh (all activity counts, incl. draws) ---
    activity = pd.concat([
        gold[["Fighter_1", "Event_Date"]].rename(columns={"Fighter_1": "name"}),
        gold[["Fighter_2", "Event_Date"]].rename(columns={"Fighter_2": "name"}),
    ])
    last_fight = activity.groupby("name")["Event_Date"].max()

    changes = []
    for idx, row in fighters.iterrows():
        real = last_fight.get(row["name"])
        if real is None:
            continue  # UFC debutant -- keep curated value
        new_val = real.strftime("%Y-%m-%d")
        if str(row.get("last_fight_date", "")) != new_val:
            changes.append((row["name"], row.get("last_fight_date"), new_val))
            fighters.at[idx, "last_fight_date"] = new_val
    fighters.to_csv(FIGHTERS_PATH, index=False)

    print(f"\nlast_fight_date updated for {len(changes)} roster fighters:")
    for name, old, new in changes:
        print(f"  {name}: {old} -> {new}")
    unmatched = sorted(roster - set(last_fight.index))
    print(f"\nRoster fighters with no UFC history (curated dates kept): {unmatched}")


if __name__ == "__main__":
    main()
