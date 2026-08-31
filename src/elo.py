"""
Elo-style rating system for UFC fighters.

Ratings are built entirely from historical fight results (data/fight_history.csv).
Finishes (KO/TKO, submission) move ratings more than decisions, since a finish
is a more decisive signal of relative skill than a close decision.
"""

import pandas as pd

METHOD_K_MULTIPLIER = {
    "KO/TKO": 1.25,
    "SUB": 1.15,
    "DEC": 0.90,
    "DQ": 0.50,
}


def canonical_method(value) -> str:
    """The STORED spelling of a method: KO/TKO, SUB, DEC, DQ, Draw, NC.

    THE FILE FORMAT IS THE SHORT CODE. generate_site._method_display maps it
    to prose at render time, and its docstring is explicit that "the stored
    codes are what the model reads". Three writers ignored that and put ESPN's
    phrasings straight into the data -- "Decision - Unanimous", "Submission",
    "KO (Punches)" -- which are not wrong to a reader and are invisible to
    everything that matches on the code:

      * METHOD_K_MULTIPLIER above is an exact-match dict, so an unrecognised
        phrasing falls through to 1.0 and a knockout updates the rating as
        though it were a decision. 46 rows in fight_history.csv.
      * matchup_model.quick_return_penalty fires only on ("KO/TKO", "SUB"),
        so a fighter whose last loss is stored as "Submission" silently
        stops taking the short-turnaround penalty.

    Unrecognised input passes through unchanged rather than being forced into
    a bucket -- an unknown method should look unknown, not like a decision.
    """
    if not isinstance(value, str) or not value.strip():
        return value
    v = value.strip()
    low = v.lower()
    if low in ("nc", "no contest", "overturned"):
        return "NC"
    if "draw" in low:
        return "Draw"
    if low.startswith("dq") or "disqualif" in low:
        return "DQ"
    if "dec" in low:
        return "DEC"
    if "sub" in low or "choke" in low or "armbar" in low or "kimura" in low:
        return "SUB"
    if "ko" in low or "tko" in low or "knockout" in low or "punch" in low:
        return "KO/TKO"
    return v


# Experience-based adaptive K schedule (the core Glicko insight applied
# minimally): a fighter's first few results should move their rating a
# lot -- 1500 is a guess, not knowledge, and the fastest way out of a
# wrong guess is a big step -- while an established fighter's rating
# reflects a real body of evidence and should move less per result.
# Fully point-in-time: the fight count that picks the K is the number of
# fights ALREADY replayed for that fighter at that moment, so this is
# exactly as lookahead-safe as the base Elo itself. Validated via
# walkforward_backtest.py before shipping (see that script's output for
# the honest before/after numbers).
ADAPTIVE_K_SCHEDULE = [
    (5, 64.0),   # fights 1-5: doubled K, escape the 1500 guess quickly
    (10, 48.0),  # fights 6-10: still elevated while the picture firms up
]
# after that: the base k_factor (32.0)


def ufc_only(history_df):
    """The subset of the spine that forms a connected rating graph.

    fight_history.csv carries hand-entered regional bouts (a `promotion`
    other than blank/UFC). They are real fights and they count as ACTIVITY --
    coverage, last_fight_date, layoff all read them. They are worthless as
    evidence of RELATIVE strength, because the opponents have no other results
    in the graph.

    Every consumer that reasons about strength must use this, not just Elo.
    build_effective_ratings decides how far to trust a fighter's Elo from a
    raw row count; left unfiltered it counted 18 bouts for Michael Aljarouj
    while his Elo was built from 1, so it moved the blend weight to 1.0 and
    fully trusted a rating earned in a single fight -- a loss. That published
    his opponent at 76% against a truer 57%. Streak bonuses have the same
    exposure: a regional win streak is not a UFC win streak.
    """
    if history_df is None or getattr(history_df, "empty", True):
        return history_df
    if "promotion" not in history_df.columns:
        return history_df
    promo = history_df["promotion"].fillna("").astype(str).str.strip()
    return history_df[promo.eq("") | promo.str.upper().eq("UFC")]


class EloRatingSystem:
    def __init__(self, initial_rating: float = 1500.0, k_factor: float = 32.0, adaptive_k: bool = True):
        self.initial_rating = initial_rating
        self.k_factor = k_factor
        self.adaptive_k = adaptive_k
        self.ratings: dict[str, float] = {}
        self.fight_counts: dict[str, int] = {}
        self.history: list[dict] = []  # rating trajectory, useful for debugging/plotting

    def get_rating(self, fighter: str) -> float:
        return self.ratings.get(fighter, self.initial_rating)

    def _k_for(self, fighter: str) -> float:
        """Per-fighter K based on how many fights of theirs we've seen so far."""
        if not self.adaptive_k:
            return self.k_factor
        count = self.fight_counts.get(fighter, 0)
        for threshold, k in ADAPTIVE_K_SCHEDULE:
            if count < threshold:
                return k
        return self.k_factor

    @staticmethod
    def expected_score(rating_a: float, rating_b: float) -> float:
        """Probability fighter A beats fighter B given their ratings."""
        return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))

    def update_ratings(self, winner: str, loser: str, method: str = "DEC") -> None:
        r_w = self.get_rating(winner)
        r_l = self.get_rating(loser)

        exp_w = self.expected_score(r_w, r_l)
        exp_l = 1.0 - exp_w

        mult = METHOD_K_MULTIPLIER.get(method, 1.0)
        # Per-fighter K: an experienced fighter's rating stays steady even
        # when their opponent is a fast-moving newcomer, and vice versa --
        # asymmetric K per side is standard practice (each side's update
        # uses their own uncertainty), not a bug.
        k_w = self._k_for(winner) * mult
        k_l = self._k_for(loser) * mult

        new_r_w = r_w + k_w * (1.0 - exp_w)
        new_r_l = r_l + k_l * (0.0 - exp_l)

        self.ratings[winner] = new_r_w
        self.ratings[loser] = new_r_l
        self.fight_counts[winner] = self.fight_counts.get(winner, 0) + 1
        self.fight_counts[loser] = self.fight_counts.get(loser, 0) + 1

        self.history.append({
            "winner": winner, "loser": loser, "method": method,
            "winner_rating_before": r_w, "winner_rating_after": new_r_w,
            "loser_rating_before": r_l, "loser_rating_after": new_r_l,
        })

    def build_from_history(self, fight_history_df: pd.DataFrame) -> dict[str, float]:
        """
        Replays fight_history.csv in chronological order to build current ratings.
        Expected columns: date, fighter_a, fighter_b, winner, method

        Rows where the winner matches neither listed fighter (draws, no
        contests, malformed data) are skipped defensively -- the old
        loser-inference logic would otherwise treat the winner string
        itself (e.g. "Draw/NC") as a phantom fighter who beats fighter_a,
        which silently poisons ratings at scale.

        NON-UFC ROWS ARE EXCLUDED, and this is the only consumer that
        excludes them. Elo is a RELATIVE rating: a win is worth whatever the
        opponent's own results say they are worth, which requires the
        opponents to be in the graph. A regional bout brings in a node with
        no other results, so it is scored against the 1500 default -- beating
        an unrated fighter is arithmetically identical to beating an average
        UFC fighter.

        Measured, not assumed. Adding Michael Aljarouj's 15 decided regional
        bouts (100% FIGHT, HFC, MMAGP, Hexagone) moved him +194.5 and dragged
        269 other fighters with him through their three shared opponents --
        33 of them on the current roster, up to +36.4 each. That is the whole
        argument for the promotion column.

        Every pre-existing row has a blank promotion and is treated as UFC,
        so this filter is a no-op against the spine as it stands.
        """
        df = ufc_only(fight_history_df).copy()
        dropped = len(fight_history_df) - len(df)
        if dropped:
            print(f"[elo] excluded {dropped} non-UFC row(s) from the rating graph")
        df["date"] = pd.to_datetime(df["date"])
        # kind="stable": pandas defaults to quicksort, which is NOT stable, so a
        # plain sort_values("date") silently reshuffles rows WITHIN a date. Elo
        # replays row by row, so that alone moved 271 fighters (Don Frye +23.8)
        # with zero data change. Measured 2026-08-31.
        df = df.sort_values("date", kind="stable")

        skipped = 0
        for _, fight in df.iterrows():
            winner = fight["winner"]
            if winner == fight["fighter_a"]:
                loser = fight["fighter_b"]
            elif winner == fight["fighter_b"]:
                loser = fight["fighter_a"]
            else:
                skipped += 1
                continue
            self.update_ratings(winner, loser, method=fight.get("method", "DEC"))

        if skipped:
            print(f"[elo] skipped {skipped} rows with a winner matching neither fighter (draws/NC/malformed)")
        return self.ratings

    def rankings(self) -> pd.DataFrame:
        return (
            pd.DataFrame(
                [{"fighter": f, "elo": r} for f, r in self.ratings.items()]
            )
            .sort_values("elo", ascending=False)
            .reset_index(drop=True)
        )
