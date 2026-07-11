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


class EloRatingSystem:
    def __init__(self, initial_rating: float = 1500.0, k_factor: float = 32.0):
        self.initial_rating = initial_rating
        self.k_factor = k_factor
        self.ratings: dict[str, float] = {}
        self.history: list[dict] = []  # rating trajectory, useful for debugging/plotting

    def get_rating(self, fighter: str) -> float:
        return self.ratings.get(fighter, self.initial_rating)

    @staticmethod
    def expected_score(rating_a: float, rating_b: float) -> float:
        """Probability fighter A beats fighter B given their ratings."""
        return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))

    def update_ratings(self, winner: str, loser: str, method: str = "DEC") -> None:
        r_w = self.get_rating(winner)
        r_l = self.get_rating(loser)

        exp_w = self.expected_score(r_w, r_l)
        exp_l = 1.0 - exp_w

        k = self.k_factor * METHOD_K_MULTIPLIER.get(method, 1.0)

        new_r_w = r_w + k * (1.0 - exp_w)
        new_r_l = r_l + k * (0.0 - exp_l)

        self.ratings[winner] = new_r_w
        self.ratings[loser] = new_r_l

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
        """
        df = fight_history_df.copy()
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date")

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
