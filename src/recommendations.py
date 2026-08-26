"""
Legs the model recommends, each with the price it needs to be worth taking.

WHAT THIS IS FOR. The site's odds feed carries four markets: moneyline,
method, total rounds, and goes-the-distance. The markets actually being bet
are not those. From a winning eight-leg slip: six legs were DOUBLE CHANCE
("by KO/TKO or Submission", "by KO/TKO or on Points"), two were round-start
markets ("fight to start round 2"), one was a moneyline. None of the first
eight are in the feed.

Every one of them is nevertheless a pure derivation from what the model
already computes, so the gap is in the PRICE, not in the probability:

  Double Chance   reconcile_fighter_methods returns a 2x3 grid whose rows sum
                  to each fighter's win probability and whose columns sum to
                  the fight-level method split. Any "A by X or Y" market is a
                  two-cell sum of that grid.
  Round start     "does the fight reach round N" is one minus the chance it
                  ends before N, which is P(finish) times the division-
                  conditioned share of finishes landing before that point.

SO THIS PUBLISHES A THRESHOLD, NOT AN EDGE. Without a quoted price there is
no edge to compute and claiming one would be an invention. What the model can
say honestly is: here is the probability, and here is the price at which
backing it becomes worthwhile. The reader looks up their book and compares.
That is a smaller claim than an EV figure and it is one the data actually
supports.

THE PRICE THRESHOLD, AND WHY IT IS NOT SIMPLY 1/p. Break-even is 1/p. The
threshold demands a margin above it (REQUIRED_EDGE) for two reasons that both
push the same way: these legs are selected -- publishing the model's most
confident calls means publishing wherever its error happens to be most
favourable -- and the model's probability is itself uncertain. A leg quoted
exactly at break-even is a coin flip on the model being right, which is not
a recommendation.

WHY THE GRID IS RESCALED TOWARD THE MARKET FIRST. The grid is built from the
model's own win probability, and multiplying model-only numbers is what made
the old parlay builder overstate itself by 1.80x on two legs. The moneyline
IS quoted, so each fighter's row is rescaled so it sums to their win
probability blended toward the de-vigged market price. The method split
within a fighter is left alone -- that is the part the market says nothing
about -- while the part the market does price gets shrunk toward it.
"""

from src.method_model import finish_share_before
from src.odds_utils import implied_prob_to_american, market_blended_prob

# Margin over break-even a leg must clear before it is worth backing. 1.05 is
# roughly one moneyline leg's worth of vig, and it is the floor the betting
# audit derived independently: below it the required raw model edge stops
# being attainable, above ~1.08 almost nothing on a card qualifies.
REQUIRED_EDGE = 1.05

# Below this the model is not confident enough for the threshold to mean
# much, and the price it would demand is longer than any book offers.
MIN_PROB = 0.12

# A moneyline only counts as a confident call if the model has the fighter
# winning by a clear margin. A coin flip is not a call, and the section is
# titled for what the model likes.
MIN_MONEYLINE_PROB = 0.60

# How many legs of any ONE market type the list will show. Keeps the range of
# the card visible instead of whichever market happens to sit highest on the
# probability scale.
#
# WAS 5, AND FIVE WAS TOO MANY. "Fight to start round 2" is roughly one minus
# the chance of a first-round finish, so it is the highest-probability leg on
# almost every fight and it filled five of the six visible rows at 79, 79, 79,
# 80 and 83 percent -- five rows that are one observation about early finishes,
# not five recommendations. Two is enough to show the market exists without
# letting it own the section.
MAX_PER_MARKET = 2

_METHODS = ("KO/TKO", "Submission", "Decision")

# The Double Chance markets books actually offer, as index pairs into a
# fighter's row of the grid. "Submission or on Points" exists too but is
# rarely quoted, so it is not published rather than shown and unfindable.
_DOUBLE_CHANCE = (
    ((0, 1), "by KO/TKO or Submission"),
    ((0, 2), "by KO/TKO or on Points"),
)


def _threshold(p: float) -> int | None:
    """American price at which a leg of probability p clears REQUIRED_EDGE."""
    if not p or p <= 0 or p >= 1:
        return None
    implied_needed = p / REQUIRED_EDGE
    if implied_needed <= 0 or implied_needed >= 1:
        return None
    return int(round(implied_prob_to_american(implied_needed)))


def _blended_win_probs(preview: dict, market_probs: dict | None) -> list[float] | None:
    """
    Each fighter's win probability, shrunk toward the de-vigged moneyline
    where one is quoted. Returns None when the preview has no grid.
    """
    names = preview.get("method_grid_fighters") or []
    grid = preview.get("method_grid") or []
    if len(names) != 2 or len(grid) != 2:
        return None
    out = []
    for i, name in enumerate(names):
        p_model = sum(grid[i])
        mk = (market_probs or {}).get(name)
        out.append(market_blended_prob(p_model, mk) if mk else p_model)
    return out


def _rescaled_grid(preview: dict, market_probs: dict | None):
    grid = preview.get("method_grid") or []
    if len(grid) != 2:
        return None, None
    names = preview.get("method_grid_fighters") or []
    blended = _blended_win_probs(preview, market_probs)
    if not blended:
        return None, None
    out = []
    for i in range(2):
        total = sum(grid[i])
        # Rescale the row to the blended win probability, keeping the method
        # split inside it untouched -- the market prices WHO wins, not HOW.
        scale = (blended[i] / total) if total > 0 else 0.0
        out.append([v * scale for v in grid[i]])
    return out, names


def legs_for_fight(fight: dict, market_probs: dict | None = None) -> list[dict]:
    """
    Every derivable leg for one fight, each with its model probability and the
    price it needs. `market_probs` maps fighter name -> de-vigged moneyline
    probability, when the feed quoted one.
    """
    preview = fight.get("preview") or {}
    grid, names = _rescaled_grid(preview, market_probs)
    if not grid:
        return []
    fa, fb = fight.get("fighter_a"), fight.get("fighter_b")
    key = f"{fa}|{fb}"
    legs = []

    for i, name in enumerate(names):
        row = grid[i]
        # DOUBLE CHANCE -- the market six of eight legs on the reference slip
        # used, and the one the feed does not carry.
        for (a, b), phrase in _DOUBLE_CHANCE:
            p = row[a] + row[b]
            if p < MIN_PROB:
                continue
            th = _threshold(p)
            if th is None:
                continue
            legs.append({
                "fight_key": key, "fight_label": f"{fa} vs {fb}",
                "fighter": name, "market": "Double Chance",
                "label": f"{name} {phrase}",
                "kind": "threshold",
                "p_model": round(p, 4), "min_price": th,
                "why": f"{_METHODS[a]} {row[a]*100:.0f}% + {_METHODS[b]} {row[b]*100:.0f}%",
            })

    # ROUND-START MARKETS. "Fight to start round N" is one minus the chance it
    # ends before round N begins -- i.e. before N-1 complete rounds elapse.
    md = preview.get("method_distribution") or {}
    p_finish = 1.0 - float(md.get("decision") or 0.0) if md else None
    sched = int(preview.get("scheduled_rounds") or 3)
    division = fight.get("weight_class")
    if p_finish:
        for n in range(2, sched + 1):
            ends_before = p_finish * finish_share_before(float(n - 1), sched, division)
            p = 1.0 - ends_before
            if p < MIN_PROB or p >= 0.995:
                continue
            th = _threshold(p)
            if th is None:
                continue
            legs.append({
                "fight_key": key, "fight_label": f"{fa} vs {fb}",
                "fighter": None, "market": "Round start",
                "label": f"Fight to start round {n} — Yes",
                "kind": "threshold",
                "p_model": round(p, 4), "min_price": th,
                "why": f"{ends_before*100:.0f}% chance it ends first",
            })
    return legs


def build_recommendations(events: list[dict], tracked_edges: list[dict] | None = None,
                          limit: int = 12) -> list[dict]:
    """
    The card's derivable legs, most confident first.

    RANKED BY PROBABILITY, and that is defensible here in a way it was not in
    the old parlay builder. There, probability was maximised INSIDE a payout
    band, which pins the price and makes it algebraically a hunt for the
    model's largest optimistic error. Here there is no band and no quoted
    price, so ranking by probability is simply "the calls the model is most
    confident about" -- which is also the cohort this model is good at: it
    hits 85.2% on the 61 logged picks where it agreed with the market on the
    winner, against 39.1% on the 23 where it took the underdog.
    """
    market_probs: dict = {}
    for row in tracked_edges or []:
        if str(row.get("market")) == "Moneyline":
            p = row.get("book_fair_prob")
            if p is not None and p == p:
                market_probs[row.get("fighter")] = float(p)

    # PRICED LEGS, which this section used to exclude by charter.
    #
    # The rule was "publish only what the feed does not price", because an
    # unpriced market can be given an honest THRESHOLD while a priced one is
    # already shown elsewhere on the page. Defensible, and it produced a
    # section that was nothing but round-start rows -- a reader looking for
    # the model's actual calls found the one market that sits structurally
    # highest on the probability scale, repeated.
    #
    # So the charter widens: the best legs on the card, wherever they are
    # priced. The two claims stay VISIBLY different rather than being mixed
    # into one ranking, because they are not the same statement. A priced leg
    # says "the model disagrees with this number by this much". An unpriced
    # one says "here is the probability, here is what you must be paid".
    priced = []
    for row in tracked_edges or []:
        if str(row.get("market")) != "Moneyline":
            continue
        p, fair = row.get("model_prob"), row.get("book_fair_prob")
        odds, who = row.get("odds_american"), row.get("fighter")
        if None in (p, fair, odds, who) or p != p or fair != fair:
            continue
        edge = float(p) - float(fair)
        # A LEG THE PRICE ALREADY CONTAINS STILL BELONGS HERE, and it is
        # labelled rather than dropped. This section is the model's most
        # confident calls, so a read the market shares is still a read -- the
        # honest move is to show it and say the price is ahead, not to hide a
        # pick because it happens to be correctly priced. Dropping them left
        # the heaviest favourites off a list titled "legs the model likes",
        # which is the opposite of what a reader expects to find there.
        priced_in = edge <= 0
        # THE MODEL HAS TO ACTUALLY LIKE IT. Obvious in hindsight and it was
        # missing: the filter below only asked whether the BOOK favoured the
        # fighter, so "Yan Xiaonan, model 38%" appeared in a list of legs the
        # model likes. A call the model rates a loser is not a confident call,
        # it is the opposite one.
        if float(p) < MIN_MONEYLINE_PROB:
            continue
        # AND IT MUST AGREE WITH THE MARKET ON WHO WINS.
        #
        # Ranking by edge selects the largest model-vs-market disagreement,
        # and that disagreement is largest exactly where this model is worst.
        # Measured in this module's own docstring: 85.2% on the 61 picks where
        # it agreed with the market on the winner, 39.1% on the 23 where it
        # took the underdog. validate_market_blend puts the same split at
        # 40/47 against 9/21 -- worse than a coin flip.
        #
        # Without this the section led with seven straight underdogs at +130
        # to +614, which is not the model being bold, it is the ranking
        # function seeking the cohort the model cannot price.
        if float(fair) < 0.5:
            continue
        priced.append({
            "fight_key": row.get("fight_key") or row.get("fight_id"),
            "fight_label": f'{row.get("fighter_a")} vs {row.get("fighter_b")}',
            "fighter": who, "market": "Moneyline",
            "label": f"{who} Moneyline",
            "kind": "edge",
            "p_model": round(float(p), 4),
            "price": float(odds),
            "edge_pp": round(edge * 100, 1),
            "priced_in": priced_in,
            "source": row.get("source"),
            "why": (f"model {float(p)*100:.0f}% against a fair {float(fair)*100:.0f}%"
                    + ("" if not priced_in else " -- the price already has it")),
        })

    legs = []
    for event in events or []:
        for fight in event.get("fights") or []:
            if fight.get("cancelled"):
                continue
            try:
                legs.extend(legs_for_fight(fight, market_probs))
            except Exception:
                continue          # one bad fight must not empty the section
    # Ranked WITHIN a kind, never across it. Sorting the two together on
    # p_model is what buried everything under round-start: "fight to start
    # round 2" is roughly one minus the chance of a first-round finish, so it
    # outranks a genuine 70% moneyline read on almost every fight without
    # being a better call.
    # CONFIDENCE FIRST, NOT EDGE. Edge is the claim, but it is not the
    # ordering: sorting on it puts the thinnest, longest-priced reads at the
    # top of a section a reader treats as "what the model likes most".
    priced.sort(key=lambda l: (-l["p_model"], -l["edge_pp"]))
    legs.sort(key=lambda l: -l["p_model"])

    # ONE LEG PER FIGHT, and this is a correctness constraint rather than a
    # presentation one. Two legs on the same bout are not independent, so they
    # cannot both go in a parlay at the quoted prices -- a book would price
    # them as a same-game parlay instead. Publishing several legs from one
    # fight invites exactly the combination that cannot be taken.
    #
    # It also fixes the ranking, which without it produced a wall of
    # near-identical "Fight to start round 2" rows: that market is the highest
    # probability leg on most fights, so a pure probability sort returns the
    # same market thirteen times and buries every other option on the card.
    # ONE PASS, NOT TWO FILTERS IN SEQUENCE -- and the order was the bug.
    #
    # This used to collapse every fight to its single highest-probability leg
    # FIRST, then apply the per-market cap. But "fight to start round 2" is
    # the highest-probability leg on nearly every fight, so the first stage
    # threw away every Double Chance leg before the cap could ever admit one,
    # and the cap then just truncated a pile of identical round-start rows.
    # The card went from twelve slots to three filled.
    #
    # Walking the ranked list once, taking a leg only if its fight is unused
    # AND its market still has room, lets a fight whose round-start leg is
    # capped out contribute its next-best leg instead.
    per_market: dict = {}
    seen_fights: set = set()
    out = []

    # Priced legs lead, but they OBEY THE SAME CAP. Letting them bypass it
    # put twelve moneylines in twelve slots and pushed Double Chance and
    # round-start off the card entirely -- the same crowding-out the cap
    # exists to prevent, arriving from the other direction.
    for leg in priced:
        if leg["fight_key"] in seen_fights:
            continue
        m = leg["market"]
        if per_market.get(m, 0) >= MAX_PER_MARKET:
            continue
        per_market[m] = per_market.get(m, 0) + 1
        seen_fights.add(leg["fight_key"])
        out.append(leg)

    for leg in legs:
        if leg["fight_key"] in seen_fights:
            continue
        m = leg["market"]
        if per_market.get(m, 0) >= MAX_PER_MARKET:
            continue
        per_market[m] = per_market.get(m, 0) + 1
        seen_fights.add(leg["fight_key"])
        out.append(leg)
    return out[:limit]
