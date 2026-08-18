"""
Matches computed edges (which only know fighter names) back to the real
upcoming fight card (data/fight_cards.csv) so the site can group everything
by event -> fight, instead of one flat table.
"""

import re
import unicodedata

import pandas as pd

from src.rationale import explain_edge, explain_favorite_pick
from src.model_preview import build_fight_preview, build_full_market_projection
from src.matchup_model import normalize_division
from src.method_model import finish_share_before
from src.fight_format import is_five_round as _is_five_round, scheduled_rounds as _scheduled_rounds
from src.odds_utils import implied_prob_to_american, format_american_odds


# A consistent accent color per division, purely for faster visual scanning
# down a long card -- not tied to any model logic.
# Groups card_position into the actual broadcast segments for divider
# purposes -- Main Event and Co-Main Event are individually the most
# important fights, but they're still PART of the "Main Card" broadcast
# segment, not their own separate segments.
SEGMENT_LABELS = {
    "Main Event": "MAIN CARD",
    "Co-Main Event": "MAIN CARD",
    "Main Card": "MAIN CARD",
    "Prelims": "PRELIMINARY CARD",
    "Early Prelims": "EARLY PRELIMS",
}

WEIGHT_CLASS_COLORS = {
    "Strawweight": "#e88fc7",
    "Flyweight": "#5ec9d6",
    "Bantamweight": "#f2a65a",
    "Featherweight": "#b18af2",
    "Lightweight": "#6db3f2",
    "Welterweight": "#6ddc9a",
    "Middleweight": "#f26d6d",
    "Light Heavyweight": "#e8955e",
    "Heavyweight": "#d64545",
}

# Avatar gradient pairs per country, for the fighter-flag-colors feature.
# Deliberately muted/curated rather than literal official flag hex values
# -- picked to stay legible with white initials text on top and to fit
# the site's existing dark, restrained palette instead of clashing with
# it the way vibrant flag colors would. Falls back to the existing
# hash-based hue gradient for any fighter whose country isn't mapped.

# Avatar ring colors per country, for the fighter-flag-colors feature.
# Unlike the old 2-color gradient approach above, these are real,
# recognizable flag colors (not muted/adjusted for legibility) -- the
# ring design keeps the initials legible regardless of how bright or
# white a flag is, so there's no need to compromise the colors
# themselves anymore. Each list is 2-4 colors, rendered as equal
# conic-gradient segments.
COUNTRY_FLAG_EMOJI = {
    "USA": "🇺🇸",
    "Brazil": "🇧🇷",
    "UK": "🇬🇧",
    "Russia": "🇷🇺",
    "Australia": "🇦🇺",
    "France": "🇫🇷",
    "Ireland": "🇮🇪",
    "Canada": "🇨🇦",
    "Venezuela": "🇻🇪",
    "China": "🇨🇳",
    "Aruba": "🇦🇼",
    "South Africa": "🇿🇦",
    "Argentina": "🇦🇷",
    "Georgia": "🇬🇪",
    "Nigeria": "🇳🇬",
    "Kyrgyzstan": "🇰🇬",
    "Ukraine": "🇺🇦",
    "Uzbekistan": "🇺🇿",
    "South Korea": "🇰🇷",
}


def _normalize_name(name: str) -> str:
    """
    Strips accents and standardizes punctuation so minor spelling differences
    between sources (e.g. Polymarket listing 'Benoît Saint Denis' while our
    data has 'Benoit Saint-Denis') don't cause a real fight to silently miss
    its match and get dumped into 'unmatched' instead.
    """
    # Coerce first. Edge rows come from several finders with different key
    # sets -- fight-level markets never set "opponent" -- so a DataFrame built
    # from a mix of them fills the gap with NaN, which is a FLOAT and blows up
    # unicodedata.normalize. Returning "" for a missing name lets the caller's
    # set comparison simply fail to match, which is the correct outcome, and
    # is far better than a crash three frames away from the cause.
    if name is None or not isinstance(name, str):
        try:
            if pd.isna(name):
                return ""
        except (TypeError, ValueError):
            pass
        name = str(name) if name is not None else ""
    normalized = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9 ]", " ", normalized.lower()).strip()


def load_fight_cards(path: str = "data/fight_cards.csv") -> pd.DataFrame:
    return pd.read_csv(path)


def group_unmatched_by_fight(unmatched_df: pd.DataFrame) -> list[dict]:
    """
    Groups live odds for fights NOT on any tracked card by fighter pair, so
    they show as a genuine preview instead of a flat, hard-to-scan list.
    """
    if unmatched_df.empty:
        return []

    fights: dict[frozenset, dict] = {}
    for _, row in unmatched_df.iterrows():
        row_dict = row.to_dict()
        fighter_field = row_dict.get("fighter", "")
        opponent = row_dict.get("opponent")

        if " vs " in str(fighter_field):
            names = [n.strip() for n in fighter_field.split(" vs ")]
            if len(names) != 2:
                continue
            fighter_a, fighter_b = names
        elif opponent:
            fighter_a, fighter_b = fighter_field, opponent
        else:
            continue

        key = frozenset({fighter_a, fighter_b})
        if key not in fights:
            fights[key] = {"fighter_a": fighter_a, "fighter_b": fighter_b, "edges": []}
        fights[key]["edges"].append(row_dict)

    result = list(fights.values())
    for fight in result:
        fight["edges"].sort(key=lambda e: abs(e.get("edge_pct", 0)), reverse=True)

    # Filter out orphaned single-market noise (e.g. just a stray Under/Over
    # with no moneyline and nothing else) -- keep only fights that have
    # either a real moneyline or multiple market types, since a single
    # isolated rounds line with no other context isn't a useful preview.
    def _is_substantial(fight: dict) -> bool:
        markets = {e.get("market") for e in fight["edges"]}
        has_moneyline = "Moneyline" in markets
        return has_moneyline or len(markets) >= 2

    result = [f for f in result if _is_substantial(f)]
    result.sort(key=lambda f: len(f["edges"]), reverse=True)
    return result



def build_market_label(fighter: str, market: str) -> str:
    """
    One readable label instead of a Selection + Market pair.

    Fight-level props (rounds, distance, fight-method) carried the matchup
    "A vs B" in the Selection column of EVERY row -- redundant inside a table
    that already belongs to that fight, and it cost a full column of width on
    a phone. Fighter-specific rows keep the name because there the selection
    genuinely disambiguates; fight-level rows become a sentence instead.
    """
    fighter = str(fighter or "").strip()
    market = str(market or "").strip()
    fight_level = " vs " in fighter or not fighter

    # Phrase the fight-level markets as ONE three-way question -- KO/TKO,
    # Submission, Decision -- matching how FanDuel and DraftKings present
    # "How will the fight end?". "Goes the distance" IS the decision leg, so
    # naming it that way makes the set read as three answers to one question
    # rather than three unrelated props.
    PHRASE = {
        "goesthedistance": "Fight ends by Decision",
        "fightoutcome": "Fight ends by Decision",
        "fightmethod": "Fight ends by {}",
        "totalrounds": "Total rounds {}",
    }
    # Method abbreviations are fine in a data column but not in a sentence.
    EXPAND = {"sub": "Submission", "ko/tko": "KO/TKO", "dec": "Decision",
              "goesthedistance": "Decision", "goes the distance": "Decision"}
    if fight_level:
        # edge_finder emits "Fight Method: KO/TKO" (with a space) while the
        # projection builder uses "FightMethod" -- normalise both.
        base = market.split(":")[0].strip().lower().replace(" ", "")
        detail = market.split(":", 1)[1].strip() if ":" in market else ""
        detail = EXPAND.get(detail.lower(), detail)
        if base in PHRASE:
            tpl = PHRASE[base]
            return tpl.format(detail) if "{}" in tpl else tpl
        return market
    if market.startswith("Method:"):
        return f"{fighter} — {market.split(':', 1)[1].strip()}"
    if market == "Moneyline":
        return fighter
    return f"{fighter} — {market}"


def assign_canonical_fight_ids(upcoming_df: pd.DataFrame, cards_df: pd.DataFrame) -> pd.DataFrame:
    """
    Different odds sources assign their own internal fight IDs -- Polymarket
    might call the McGregor/Holloway fight 'e1' while DraftKings calls it
    '1'. If a moneyline comes from one source and a rounds prop for the
    SAME real fight comes from another, they'd end up with different
    fight_id values and the parlay builder would treat them as two
    unrelated fights instead of bundling them as a same-fight combo.

    This reassigns fight_id based on the normalized fighter pair matched
    against the tracked card, so every row for the same real fight shares
    one consistent ID no matter which source it came from.
    """
    if upcoming_df.empty:
        return upcoming_df

    card_pairs = {}
    for i, row in cards_df.iterrows():
        key = frozenset({_normalize_name(row["fighter_a"]), _normalize_name(row["fighter_b"])})
        card_pairs[key] = f"card_{i}"

    def canonical_id(row):
        fighter_a, fighter_b = row.get("fighter_a"), row.get("fighter_b")
        if not fighter_a or not fighter_b:
            return row.get("fight_id")
        key = frozenset({_normalize_name(fighter_a), _normalize_name(fighter_b)})
        return card_pairs.get(key, f"untracked_{'_'.join(sorted(key))}")

    df = upcoming_df.copy()
    df["fight_id"] = df.apply(canonical_id, axis=1)
    return df


def _split_total_rounds_fighter_field(fighter_field: str) -> set[str]:
    return {name.strip() for name in fighter_field.split(" vs ")}



def _reconcile_round_props(rows: list[dict], fight: dict,
                           decision_shown: float | None = None) -> list[dict]:
    """
    Recompute every round total from the SAME distribution the fight props
    show, so the two groups can't contradict each other.

    They were computed independently: edge_finder ran its own
    method_probabilities for the round lines while the displayed decision came
    from the preview's distribution. Different inputs, different answers -- on
    one card the round props implied a 51% finish rate beside a method row
    reading 77% decision, and on another the decision row said 100% while the
    rounds said the fight ended early two thirds of the time.

    A decision necessarily means the fight passed the last half-round mark, so
    these are not two estimates to be averaged; one is derived from the other.
    Deriving it here -- at display time, from the object being displayed -- is
    the same fix that stopped the headline disagreeing with its own table.
    """
    # The DISPLAYED decision first -- that is the number the reader compares
    # against, and the only one guaranteed to be on screen. The preview's
    # distribution is the fallback for fights whose Fight props group has no
    # decision row at all.
    decision = decision_shown
    if decision is None:
        dist = (fight.get("preview") or {}).get("method_distribution")
        if not dist:
            print(f"[rounds] no decision probability available for "
                  f"{fight.get('fighter_a')} vs {fight.get('fighter_b')} -- "
                  f"round totals left as computed")
            return rows
        decision = float(dist.get("decision", 0.0))
    # Guard the value however it arrived. The clamp in method_model bounds
    # the MODEL, but this figure can also come from a priced row -- and a
    # thin market can quote something at 1.00, which would zero every Under
    # line the same way.
    finish = max(1.0 - float(decision), 0.02)
    # THIRD place deriving fight length, and the one that actually renders the
    # round rows. A title fight is five rounds wherever it sits, so this has to
    # agree with the other two or the table contradicts itself: on a title
    # CO-MAIN this read 3, and finish_share_before on a 3-round table returns
    # 0.999 for BOTH Under 3.5 and Under 4.5 -- there are no rounds 4 or 5 in
    # that table to tell them apart -- so two distinct lines came out sharing
    # one probability. lint_site.py's round-monotonic check caught it.
    scheduled = _scheduled_rounds(fight)

    out = []
    matched = 0
    for r in rows:
        # Match on MARKET and LABEL both, and search rather than anchor.
        # The first version anchored on the label alone; whatever the label
        # actually holds for these rows, it didn't match, so every row fell
        # through untouched and the numbers didn't move at all.
        blob = f"{r.get('market') or ''} {r.get('label') or ''}"
        m = re.search(r"(Under|Over)\s*([\d.]+)", blob, re.IGNORECASE)
        if not m or "round" not in blob.lower():
            out.append(r)
            continue
        matched += 1
        side, line = m.group(1).capitalize(), float(m.group(2))
        # Division comes off the FIGHT rather than either roster row: this is
        # the booked weight class for the bout, which is the right answer when
        # a fighter is moving up or down and their roster division is stale.
        under = finish * finish_share_before(line, scheduled,
                                             normalize_division(fight.get("weight_class")))
        new = dict(r)
        new["model_prob"] = round(under if side == "Under" else 1.0 - under, 4)
        # The edge moves with it -- a stale edge beside a corrected
        # probability is worse than either alone.
        if new.get("has_line") and new.get("book_fair_prob") is not None:
            new["edge_pct"] = round((new["model_prob"] - float(new["book_fair_prob"])) * 100, 2)
        out.append(new)
    if not matched and rows:
        print(f"[rounds] no round rows matched for "
              f"{fight.get('fighter_a')} vs {fight.get('fighter_b')} -- "
              f"labels were {[r.get('market') for r in rows][:3]}")
    return out


def _canonical_fight_methods(rows: list[dict], fight: dict) -> list[dict]:
    """
    Force Fight props to exactly three rows: KO/TKO, Submission, Decision.

    TWO BUGS, ONE CAUSE -- the group was whatever markets happened to exist.

    Duplicates: the label map sends every GoesTheDistance row to "Fight ends
    by Decision" regardless of SELECTION, so the market's two sides both
    rendered under that name with complementary numbers (57.5% and 42.5%).

    Gaps: fight-level rows only exist for markets Polymarket published, so a
    fight with no submission market simply had no submission row -- KO 44.8%
    and Decision 43.4% with the remaining 11.8% unaccounted for and no
    indication anything was missing.

    Both are wrong for the same reason: this group answers ONE three-way
    question, so it should always show three answers. Priced rows win; any
    method without one falls back to the model's own figure with no odds,
    which the table already renders as a dash.
    """
    METHODS = ("KO/TKO", "Submission", "Decision")

    def method_of(r) -> str | None:
        label = str(r.get("label") or r.get("market") or "").lower()
        sel = str(r.get("selection") or "").lower()
        blob = f"{label} {sel}"
        # A complement is not an answer to the question -- "ends in finish"
        # and "not KO" restate the others rather than adding a fourth.
        if "not " in blob or "endsinfinish" in blob.replace(" ", ""):
            return None
        if "ko/tko" in blob or "kotko" in blob:
            return "KO/TKO"
        if "submission" in blob or "sub" in sel:
            return "Submission"
        if "decision" in blob or "distance" in blob:
            return "Decision"
        return None

    # Cap any single method before display. A fight-level method row reached
    # 100.0% on one card -- from a thinly-priced market, so the model's own
    # clamp couldn't reach it -- and a certainty on screen is wrong on its own
    # terms as well as breaking everything derived from it.
    # Applied HERE because this is what renders: bounding the model alone left
    # the displayed number untouched.
    METHOD_CAP = 0.97          # anything above this is treated as certainty
    METHOD_FLOOR = 0.015       # ...and fixed by lifting the others, not lowering it

    by_method: dict[str, dict] = {}
    for r in rows:
        m = method_of(r)
        if not m:
            continue
        prev = by_method.get(m)
        # Keep the PRICED row when two land on the same method -- a model-only
        # duplicate carries strictly less information.
        if prev is None or (r.get("has_line") and not prev.get("has_line")):
            by_method[m] = r

    dist = (fight.get("preview") or {}).get("method_distribution")
    out = []
    for m in METHODS:
        if m in by_method:
            out.append(by_method[m])
            continue
        if not dist:
            continue
        key = {"KO/TKO": "ko", "Submission": "sub", "Decision": "decision"}[m]
        prob = dist.get(key)
        if prob is None:
            continue
        # Model-only row: no odds, no edge. Completes the three-way set so a
        # missing market reads as "no line" rather than as a silent gap.
        out.append({"market": f"Fight Method: {m}", "label": f"Fight ends by {m}",
                    "selection": m, "model_prob": prob, "has_line": False,
                    "odds_american": None, "edge_pct": None, "fighter": "", "opponent": ""})

    # FLOOR each method, then renormalise. A row reached 100.0% on one card --
    # priced, so the model's own clamp couldn't reach it.
    #
    # A cap doesn't work here and the reason is worth recording: capping the
    # leader to 0.97 while the other two sit at zero gives a total of 0.97, and
    # renormalising divides straight back to 1.0. Lifting the OTHERS off zero
    # is what actually removes the certainty, and it leaves a normal
    # distribution essentially unchanged.
    # OVERRIDE model_prob from `dist` on every fight-level row.
    #
    # These rows arrive from edge_finder carrying their own model_prob, and
    # this function preferred whatever was already there -- so the Fight props
    # group showed one distribution while the per-fighter grid, reconciled
    # against `dist`, showed another. On one card that was 56.4% decision
    # beside per-fighter columns summing to 22.6% for the same fight.
    #
    # `dist` is the same distribution the per-fighter rows are reconciled to,
    # so taking it here makes the two halves agree by construction rather than
    # by both happening to compute the same thing. Odds and the priced flag
    # are kept; only the model number is unified, and the edge recomputed with
    # it so it can't be left describing the old value.
    if dist:
        key = {"KO/TKO": "ko", "Submission": "sub", "Decision": "decision"}
        for r in out:
            k = key.get(str(r.get("selection") or "").strip()) or \
                key.get(str(r.get("label") or "").replace("Fight ends by ", "").strip())
            if k and dist.get(k) is not None:
                r["model_prob"] = round(float(dist[k]), 4)
                if r.get("has_line") and r.get("book_fair_prob") is not None:
                    r["edge_pct"] = round((r["model_prob"] - float(r["book_fair_prob"])) * 100, 2)

    vals = [r.get("model_prob") for r in out]
    nums = [v for v in vals if isinstance(v, (int, float))]
    if nums and max(nums) > METHOD_CAP:
        floored = [max(float(v), METHOD_FLOOR) if isinstance(v, (int, float)) else v
                   for v in vals]
        total = sum(v for v in floored if isinstance(v, (int, float)))
        if total > 0:
            for r, v in zip(out, floored):
                if isinstance(v, (int, float)):
                    r["model_prob"] = round(v / total, 4)
                    if r.get("has_line") and r.get("book_fair_prob") is not None:
                        r["edge_pct"] = round((r["model_prob"] - float(r["book_fair_prob"])) * 100, 2)
    return out or rows


def group_edges_by_card(
    edges_df: pd.DataFrame,
    cards_df: pd.DataFrame,
    fighters_df: pd.DataFrame | None = None,
    effective_ratings: dict[str, float] | None = None,
    weight_class_history_df: pd.DataFrame | None = None,
    fight_history_df: pd.DataFrame | None = None,
) -> tuple[list[dict], pd.DataFrame]:
    """
    Returns (events, unmatched_edges):
      events: list of {event_name, event_date, fights: [{fighter_a, fighter_b,
               weight_class, card_position, edges: [...], preview: {...}}]}
      unmatched_edges: edges whose fighters aren't on data/fight_cards.csv
               (still useful, just can't be grouped into a known card)
    """
    fights = []
    for _, row in cards_df.iterrows():
        preview = None
        # A TITLE FIGHT IS FIVE ROUNDS WHEREVER IT SITS ON THE CARD, including
        # in the co-main slot -- which happens whenever a card carries two
        # belts. Deriving this from card_position alone silently modelled such
        # a fight as three rounds: wrong round distribution, wrong finish
        # probability, wrong Over/Under lines, and no error anywhere to notice.
        is_five_round = _is_five_round(row)
        if fighters_df is not None and effective_ratings is not None:
            try:
                preview = build_fight_preview(
                    row["fighter_a"], row["fighter_b"], fighters_df, effective_ratings, is_five_round=is_five_round,
                    weight_class_history_df=weight_class_history_df, fight_weight_class=row.get("weight_class"),
                    fight_history_df=fight_history_df,
                )
            except Exception as e:
                # One fight's preview failing -- bad roster data, a NaN slipping
                # past validation, anything unforeseen -- must never take down
                # site generation for every other fight on the card. Discovered
                # in production (July 2026): a single fighter with no real data
                # crashed the entire run, meaning the whole site failed to
                # publish over one incomplete row. Degrades to no preview for
                # just this fight, exactly like the existing "no fighters_df
                # supplied" case already does -- the template already handles
                # preview=None everywhere.
                print(f"[card_matcher] preview generation failed for {row['fighter_a']} vs {row['fighter_b']}, "
                      f"showing this fight without a preview rather than failing the whole run: {e}")
                preview = None
        fights.append({
            "event_name": row["event_name"],
            "event_date": row["event_date"],
            "event_start_time_et": row.get("event_start_time_et", "19:00"),
            "event_main_card_time_et": row.get("event_main_card_time_et", ""),
            "event_location": row.get("event_location", ""),
            "card_position": row["card_position"],
            "segment_label": SEGMENT_LABELS.get(row["card_position"], row["card_position"]),
            "weight_class": row["weight_class"],
            "weight_class_color": WEIGHT_CLASS_COLORS.get(row["weight_class"], "#8a8f9a"),
            "is_womens_division": bool(row.get("is_womens_division", False)),
            "cancelled": str(row.get("cancelled", "")).strip().lower() == "true",
            "is_title_fight": str(row.get("is_title_fight", "")).strip().lower() == "true",
            # Parsed the same way as `cancelled` above: these come back from
            # CSV as strings, and an all-empty flag column reads as float
            # NaN, so neither `bool(...)` nor a truthiness test works
            # directly. replaced_fighter is the DEPARTED fighter's name, set
            # alongside the flag by card_discovery's replacement detection,
            # and shown as the badge's tooltip.
            "replacement": str(row.get("replacement", "")).strip().lower() == "true",
            "replaced_fighter": (
                str(row.get("replaced_fighter", "")).strip()
                if str(row.get("replaced_fighter", "")).strip().lower() not in ("", "nan")
                else ""
            ),
            "fighter_a": row["fighter_a"],
            "fighter_b": row["fighter_b"],
            "fighters": {row["fighter_a"], row["fighter_b"]},
            "fighters_normalized": {_normalize_name(row["fighter_a"]), _normalize_name(row["fighter_b"])},
            "preview": preview,
            "edges": [],
        })

    unmatched_rows = []

    for _, edge in edges_df.iterrows():
        edge_dict = edge.to_dict()
        if fighters_df is not None:
            edge_dict["rationale"] = explain_edge(edge_dict, fighters_df)
        fighter_field = edge_dict["fighter"]

        if " vs " in fighter_field:
            row_pair = _split_total_rounds_fighter_field(fighter_field)
        elif edge_dict.get("opponent"):
            # Moneyline/Method rows: require BOTH the fighter AND their listed
            # opponent to match a tracked fight's exact pair. Matching on the
            # fighter's name alone is what let a stale/unrelated row (e.g. a
            # leftover "vs a different opponent" line) get folded into the
            # wrong fight just because one name happened to overlap.
            # Drop a missing opponent rather than admitting NaN to the set.
            row_pair = {v for v in (edge_dict["fighter"], edge_dict["opponent"])
                        if isinstance(v, str) and v.strip()}
        else:
            row_pair = {fighter_field}

        matched = False
        row_pair_normalized = {_normalize_name(n) for n in row_pair}
        for fight in fights:
            if row_pair_normalized == fight["fighters_normalized"]:
                fight["edges"].append(edge_dict)
                matched = True
                break

        if not matched:
            unmatched_rows.append(edge_dict)

    # group fights into events, preserving card order
    events_map: dict[tuple, dict] = {}
    for fight in fights:
        # sort each fight's edges by |edge_pct| descending so the juiciest line shows first
        fight["edges"].sort(key=lambda e: abs(e.get("edge_pct", 0)), reverse=True)

        # fill in model-only projections for any method/rounds markets the
        # live book didn't happen to cover for this fight, so there's always
        # something to look at beyond moneyline
        if fighters_df is not None and effective_ratings is not None:
            live_markets = {e["market"] for e in fight["edges"]}
            # Same rule as above -- see the note there.
            is_five_round = _is_five_round(fight)
            projection = build_full_market_projection(
                fight["fighter_a"], fight["fighter_b"], fighters_df, effective_ratings, is_five_round=is_five_round
            , fight_history_df=fight_history_df)
            # A 3-round fight has no 3.5 or 4.5 line -- only main events (and
            # title fights) are scheduled for 5. If a stray one ever arrives
            # from the book, or a projection is generated for the wrong
            # length, it's nonsense rather than a long shot, so drop it.
            # SINGLE SOURCE for the headline method. The preview computed its
            # own reconciled grid, and two computations of the same thing drift
            # -- one fight in sixty disagreed with the table it sat above.
            # The projection is what the table renders, so the headline reads
            # from it rather than recomputing. Duplicating a calculation to
            # display it twice is the mistake this file keeps making.
            if projection and fight.get("preview"):
                fav = fight["preview"].get("favorite")
                fav_rows = [
                    (r["market"].split(": ", 1)[1], r["model_prob"])
                    for r in projection.get("method_rows", [])
                    if r.get("fighter") == fav and ": " in r.get("market", "")
                ]
                if fav_rows:
                    fight["preview"]["likely_method"] = max(fav_rows, key=lambda t: t[1])[0]
                    fight["preview"]["likely_method_rate"] = round(
                        max(fav_rows, key=lambda t: t[1])[1], 3)

            max_line = 4.5 if is_five_round else 2.5

            def _round_line_out_of_range(e):
                import re as _re
                txt = f"{e.get('market','')} {e.get('selection','')}"
                m = _re.search(r"(\d+\.5)", txt)
                if not m or "round" not in txt.lower():
                    return False
                return float(m.group(1)) > max_line

            model_only = []
            if projection:
                # Match on FIGHTER + MARKET, not market alone. "Method: KO/TKO"
                # doesn't identify whose KO it is, so a live line on one
                # fighter's KO suppressed the projection row for BOTH -- which
                # is why the Model Projection table showed Submission and
                # Decision for each man but no KO at all. The gap was covered
                # by the per-prop prose beneath the table; with that prose
                # removed, the table has to be complete on its own.
                # Fold accents on BOTH sides of the key. Live rows carry
                # Polymarket's spelling ("Uros Medic") while projections use
                # the roster's ("Uroš Medić"), so an exact pair comparison
                # never matched and every round total rendered TWICE -- once
                # with a live line, once as a model-only row.
                import unicodedata as _ud

                def _fold(t):
                    return "".join(ch for ch in _ud.normalize("NFKD", str(t).lower())
                                   if not _ud.combining(ch)).strip()

                def _key(fighter, market):
                    f = _fold(fighter)
                    if " vs " in f:
                        # Order-independent: the two sources disagree on which
                        # fighter comes first, so a positional key duplicated
                        # every fight-level row.
                        f = " vs ".join(sorted(p.strip() for p in f.split(" vs ")))
                    return (f, _fold(market))

                live_pairs = {_key(e.get("fighter", ""), e.get("market", ""))
                              for e in fight["edges"]}
                for row in projection["method_rows"] + projection["rounds_rows"] + projection["distance_rows"]:
                    pair = _key(row.get("fighter", ""), row.get("market", ""))
                    if _round_line_out_of_range(row):
                        continue
                    if pair not in live_pairs:
                        model_only.append(row)
            fight["model_only_rows"] = model_only

            # ONE merged view. The old split was by an IMPLEMENTATION detail --
            # whether a book happens to price that market -- not by anything a
            # reader cares about, which is why a fighter's KO could sit in a
            # different table from his opponent's. Every market for both
            # fighters now appears exactly once here, with book/edge present
            # only where a line exists.
            # Drop complement rows. Polymarket prices each method as a
            # binary, so "Not KO/TKO" and "Ends In Finish" are just 1 minus a
            # row already in the table -- they double its length while adding
            # nothing. What remains reads as the three-way market FanDuel and
            # DraftKings actually offer: KO/TKO, Submission, Decision.
            def _is_complement(e):
                # Match on the NORMALISED market string. edge_finder emits
                # "Fight Method: SUB" and "Fight Outcome: Ends In Finish",
                # not the bare "FightMethod"/"GoesTheDistance" keys the
                # classifier uses -- so the earlier equality checks never
                # fired and every complement row survived.
                mkt = str(e.get("market", "") or "").lower().replace(" ", "")
                txt = f"{e.get('market','')} {e.get('selection','')}".lower()
                if mkt.startswith("fightmethod") and "not" in txt.split(":")[-1]:
                    return True
                if "endsinfinish" in txt.replace(" ", ""):
                    return True
                return False

            merged = []
            for e in fight["edges"]:
                if _is_complement(e) or _round_line_out_of_range(e):
                    continue
                merged.append({
                    "fighter": e.get("fighter"), "market": e.get("market"),
                    "model_prob": e.get("model_prob"), "odds_american": e.get("odds_american"),
                    "book_fair_prob": e.get("book_fair_prob"), "edge_pct": e.get("edge_pct"),
                    "suggested_stake_pct": e.get("suggested_stake_pct"), "has_line": True,
                    "clob_token_id": e.get("clob_token_id"),
                    # THE HEADLINE NUMBER, carried through to the table beside
                    # the edge it is so often at odds with. best_book and
                    # books_quoting travel with it because "EV +3.4%" is only
                    # meaningful next to the price and the book offering it.
                    "ev_pct": e.get("ev_pct"),
                    "vig_cost_pct": e.get("vig_cost_pct"),
                    "blended_prob": e.get("blended_prob"),
                    "best_book": e.get("best_book"),
                    "books_quoting": e.get("books_quoting"),
                    "source": e.get("source"),
                    # Without this the price cell cannot tell a vig-free
                    # reference line from a real book quote, and every row
                    # fell through to printing the raw source name.
                    "source_is_vig_free": e.get("source_is_vig_free"),
                    "label": build_market_label(e.get("fighter"), e.get("market")),
                    "selection": e.get("selection"),
                    # Kept in the Edges view but MARKED. A near-certain market
                    # is where a thin book quote does the most damage to an
                    # edge number, and the reader asked to still see these --
                    # a genuine quick-finisher can make a 0.5 line sharp.
                    # Flagging beats deleting: the disagreement is real, the
                    # size of it is what cannot be trusted.
                    "fragile_price": price_is_fragile(e),
                })
            for r in model_only:
                merged.append({
                    "fighter": r.get("fighter"), "market": r.get("market"),
                    "model_prob": r.get("model_prob"), "has_line": False,
                    "label": build_market_label(r.get("fighter"), r.get("market")),
                })
            # Priced markets first (they're actionable), then model-only;
            # within each, highest model probability first so the strongest
            # reads sit at the top of their group.
            merged.sort(key=lambda r: (not r["has_line"], -(r.get("model_prob") or 0)))
            fight["all_market_rows"] = merged

            # GROUP BY WHAT THE MODEL IS ANSWERING, not by whether a book
            # happens to price it. Splitting on has_line would scatter one
            # question across two tiers for an external reason -- a KO
            # projection landing somewhere different from a submission
            # projection purely because Polymarket priced one of them.
            # The three groups also read in a natural order: who wins, how it
            # ends, when it ends.
            def _group_of(market):
                m = str(market or "").lower().replace(" ", "")
                if m.startswith("totalrounds") or m.startswith("roundbetting"):
                    return "rounds"
                # "Fight Outcome:" is what compute_goes_the_distance_edges
                # emits; without it, goes-the-distance fell through to the
                # fighter group.
                if m.startswith("fightmethod") or m.startswith("goesthedistance") \
                        or m.startswith("fightoutcome"):
                    return "fight"
                return "fighter"          # Moneyline and per-fighter Method

            groups = []
            _decision_shown = None
            for gid, gname, gsub in (
                ("fighter", "Fighter props", "Who wins, and how"),
                ("fight", "Fight props", "How the fight ends, regardless of winner"),
                ("rounds", "Round props", "When it ends"),
            ):
                rows = [r for r in merged if _group_of(r.get("market")) == gid]
                if gid == "fight":
                    rows = _canonical_fight_methods(rows, fight)
                    # Capture what the Fight props group will actually SHOW.
                    # Reading the preview's distribution instead left 15 of 41
                    # fights untouched, silently: those have a priced decision
                    # market, so a number appeared on screen while the preview
                    # object's distribution was None and the rounds pass
                    # returned early without saying so.
                    for _r in rows:
                        if str(_r.get("label") or "").strip() == "Fight ends by Decision":
                            _decision_shown = _r.get("model_prob")
                            break
                elif gid == "rounds":
                    rows = _reconcile_round_props(rows, fight, _decision_shown)
                if not rows:
                    continue
                groups.append({
                    "id": gid, "name": gname, "sub": gsub, "rows": rows,
                    "count": len(rows),
                    # Surfaced on the closed summary so a group's worth is
                    # visible WITHOUT opening it -- otherwise collapsing just
                    # moves the scanning cost rather than removing it.
                    "live_count": sum(1 for r in rows if r.get("has_line")),
                    "best_edge": max((abs(r.get("edge_pct") or 0) for r in rows if r.get("has_line")), default=0),
                })
            fight["market_groups"] = groups

        key = (fight["event_name"], fight["event_date"])
        if key not in events_map:
            events_map[key] = {
                "event_name": fight["event_name"], "event_date": fight["event_date"],
                "event_start_time_et": fight.get("event_start_time_et", "19:00"),
                "event_main_card_time_et": fight.get("event_main_card_time_et", ""),
                "event_location": fight.get("event_location", ""),
                "fights": [],
            }
        events_map[key]["fights"].append(fight)

    events = list(events_map.values())
    unmatched_df = pd.DataFrame(unmatched_rows) if unmatched_rows else pd.DataFrame()
    return events, unmatched_df


LOW_SAMPLE_THRESHOLD = 6  # career fights below this = flagged as limited data


def _sample_size_flag(fighter_field: str, fighters_df: pd.DataFrame | None) -> dict | None:
    """
    Returns {"fighter": name, "fights": n} for whichever named fighter has
    the thinnest record, if any of them are below LOW_SAMPLE_THRESHOLD --
    None if everyone involved has a reasonable sample. fighter_field may be
    a single name or a "A vs B" fight-level string (GoesTheDistance-style
    rows), so this checks all names present, not just the first.
    """
    if fighters_df is None or not fighter_field:
        return None
    names = [n.strip() for n in fighter_field.split(" vs ")]
    thinnest = None
    for name in names:
        # Accent-tolerant, same reason as edge_finder: this resolves names
        # that arrived from Polymarket ("Uros Medic") against a roster that
        # stores them accented ("Uroš Medić"), so an exact match silently
        # skipped every fighter with a diacritic.
        from src.edge_finder import _find_fighter
        row = _find_fighter(fighters_df, name)
        if row.empty:
            continue
        r = row.iloc[0]
        wins = r.get("wins", 0)
        losses = r.get("losses", 0)
        total = int(wins if pd.notna(wins) else 0) + int(losses if pd.notna(losses) else 0)
        if total < LOW_SAMPLE_THRESHOLD and (thinnest is None or total < thinnest["fights"]):
            thinnest = {"fighter": name, "fights": total}
    return thinnest


# ---------------------------------------------------------------------------
# WHICH MARKETS BELONG IN A "PICK"
#
# Favorite Picks was filling up with "Not KO/TKO", "Not SUB" and Total Rounds
# Over/Under 0.5 -- none of which anyone would actually bet, and two of which
# are not independent opinions at all.
#
# 1. COMPLEMENTS SAY NOTHING NEW. Polymarket prices each method as a binary,
#    so "Not KO/TKO" is exactly 1 minus "KO/TKO". If the model has an edge on
#    one it has the mirror edge on the other by construction -- surfacing both
#    is the same view twice, and it crowds out real picks. The fight-card
#    tables already dropped these; picks never did.
#
# 2. NEAR-CERTAIN LINES ARE WHERE EDGES ARE LEAST TRUSTWORTHY. "Over 0.5
#    rounds" asks whether a fight passes 2:30 of round one -- true ~95% of the
#    time. At that end of the scale a small pricing error produces an enormous
#    apparent edge, and a thin Polymarket book is exactly where such errors
#    live. Observed: Magny/Brahimaj Over 0.5 quoted -178 (implying 64%) against
#    a model -551. The model is closer to right, but the "edge" is a bad quote,
#    not alpha. The payout is also negligible, so even a real edge there is
#    not worth a slot.
#    Deliberately NOT deleted from the Edges tab -- a genuine quick-finisher
#    can make a 0.5 line sharp, and the tab exists to show every disagreement.
#    It is flagged there instead, so the reader knows to distrust the price.
#
# 3. WHAT SURVIVES is what a person actually bets: moneyline, how the fight
#    ends (KO/TKO, Submission, Decision), and round totals at lines where both
#    outcomes are live.
# ---------------------------------------------------------------------------

# A 3-round fight turns on 1.5/2.5; a 5-rounder on 2.5/3.5/4.5. 0.5 is a
# formality and 5.5 cannot happen.
PICKABLE_ROUND_LINES = {1.5, 2.5, 3.5, 4.5}
CENTRAL_ROUND_LINES = {2.5, 3.5}     # the ones worth leading with

# Below/above these the market is a near-certainty and the price is fragile.
NEAR_CERTAIN_HI = 0.90
NEAR_CERTAIN_LO = 0.10


def is_complement_market(row) -> bool:
    """True for 'Not X' / 'Ends In Finish' rows, which mirror another row."""
    mkt = str(row.get("market", "") or "").lower().replace(" ", "")
    txt = f"{row.get('market','')} {row.get('selection','')}".lower()
    if mkt.startswith("fightmethod") and "not" in txt.split(":")[-1]:
        return True
    if "endsinfinish" in txt.replace(" ", ""):
        return True
    if re.search(r"\bnot\b", str(row.get("selection", "") or "").lower()):
        return True
    return False


def round_line_of(row):
    """The X.5 in a round-total market, or None if this isn't one."""
    txt = f"{row.get('market','')} {row.get('selection','')}"
    if "round" not in txt.lower():
        return None
    m = re.search(r"(\d+\.5)", txt)
    return float(m.group(1)) if m else None


def is_pickable_market(row) -> bool:
    """Would a person actually place this bet? Gate for Favorite Picks."""
    if is_complement_market(row):
        return False
    line = round_line_of(row)
    if line is not None and line not in PICKABLE_ROUND_LINES:
        return False
    return True


def price_is_fragile(row) -> bool:
    """
    Flag an edge whose NUMBER is unreliable, by either of two routes.

    Not a claim the model is wrong -- it is usually closer to right than the
    book here. It is a claim that the EDGE NUMBER is unreliable, because at
    95% true probability the difference between a good and a bad quote is
    worth tens of points of implied probability and the payout is a few
    cents on the dollar either way.

    TWO ROUTES, BECAUSE THE PROBABILITY TEST ALONE DID NOT DELIVER WHAT THE
    FENCE COMMENT ABOVE PROMISES. That comment says 0.5-round lines are
    "deliberately NOT deleted from the Edges tab... It is flagged there
    instead, so the reader knows to distrust the price." The flag was purely
    probabilistic at NEAR_CERTAIN_HI, so a 0.5 line the model read at 82-88%
    cleared the threshold and was shown unflagged.

    Measured on a live build: nine "Total Rounds Over 0.5" rows rendered
    edges of 30-36% at edge-heat-3, the loudest styling on the page, and
    EIGHT of them carried no warning of any kind -- on the exact market this
    module calls "a formality" and "a bad quote, not alpha". The fence had
    been applied to Favorite Picks, standout props and parlays; the promised
    flag on the one surface that deliberately keeps these rows had a hole in
    it the size of the market itself.

    An unpickable market is now fragile by construction, whatever the model
    says about it. The two tests are deliberately separate: the probability
    one catches a near-certain quote in ANY market, and this one catches a
    market whose price is structurally untrustworthy at any probability.
    """
    if not is_pickable_market(row):
        return True
    try:
        mp = float(row.get("model_prob"))
    except (TypeError, ValueError):
        return False
    if mp != mp:
        return False
    return mp >= NEAR_CERTAIN_HI or mp <= NEAR_CERTAIN_LO


def top_favorite_picks(
    edges_df: pd.DataFrame, fighters_df: pd.DataFrame | None = None, n: int = 5,
    min_odds: float = -220, max_odds: float = 160, min_edge: float = 3.0, min_model_prob: float = 0.55,
) -> list[dict]:
    """
    Straight, single-leg picks meant to actually be bet with real size --
    the opposite instinct from the parlay tiers. A -4000 "safe" favorite
    isn't a real pick (no real payout for the risk), and a +900 longshot
    isn't something to put 5-10 units on even if the model likes it, so
    both ends get filtered out by the odds range. Within that range, only
    picks the model has genuine conviction on qualify (min_edge), then
    sorted by model probability -- the highest-probability picks are what
    you'd actually want to size up on, not just the biggest edge number.
    Capped to one per fight so this doesn't turn into five props on the
    same two fighters.

    min_model_prob is a genuine correctness guard, not just a style
    choice: edge_pct alone measures model-vs-market disagreement, which
    says nothing about whether the model actually favors this side. A
    pick at 49.6% model probability can still clear a healthy edge
    threshold (the market may have it even lower) while the model itself
    is calling it a slight underdog -- which has no business being
    labeled a "favorite pick." Confirmed live: this was a real bug, not
    hypothetical.
    """
    if edges_df.empty:
        return []
    candidates = edges_df[
        (edges_df["edge_pct"] >= min_edge)
        & (edges_df["odds_american"] >= min_odds)
        & (edges_df["odds_american"] <= max_odds)
        & (edges_df["model_prob"] >= min_model_prob)
        & (edges_df["model_prob"] < NEAR_CERTAIN_HI)
    ].copy()
    if candidates.empty:
        return []
    # Only markets a person would actually bet -- see is_pickable_market.
    candidates = candidates[candidates.apply(is_pickable_market, axis=1)]
    if candidates.empty:
        return []

    # Lead with the markets the reader asked for -- moneyline and how the
    # fight ends -- then central round totals, then the rest. Sorting by
    # model_prob alone put a 1.5-round line above a moneyline simply because
    # near-certain outcomes carry the highest probabilities by definition.
    def _market_rank(row):
        mkt = str(row.get("market", "") or "").lower()
        if mkt.startswith("moneyline"):
            return 0
        if "method" in mkt or "outcome" in mkt:
            return 1
        line = round_line_of(row)
        if line in CENTRAL_ROUND_LINES:
            return 2
        return 3

    candidates["_rank"] = candidates.apply(_market_rank, axis=1)
    candidates = candidates.sort_values(["_rank", "model_prob"], ascending=[True, False])
    seen_fights = set()
    picks = []
    for _, row in candidates.iterrows():
        fight_id = row.get("fight_id")
        if fight_id in seen_fights:
            continue
        seen_fights.add(fight_id)
        d = row.to_dict()
        d.pop("_rank", None)
        picks.append(d)
        if len(picks) >= n:
            break

    for r in picks:
        # Fight-level rows (GoesTheDistance, "Fight Outcome") never set an
        # "opponent" field, since their "fighter" is already the full
        # matchup string. When mixed into a DataFrame with rows that DO
        # have one, pandas fills the gap with NaN -- which is truthy in
        # Python, so a template check like {% if p.opponent %} doesn't
        # actually filter it out, it just prints the literal word "nan".
        if pd.isna(r.get("opponent")):
            r["opponent"] = None
        r["low_sample"] = _sample_size_flag(r["fighter"], fighters_df)
        try:
            r["model_fair_odds"] = format_american_odds(implied_prob_to_american(r["model_prob"]))
        except (ValueError, ZeroDivisionError):
            r["model_fair_odds"] = "N/A"
        if fighters_df is not None:
            if r["market"] == "Moneyline":
                r["rationale"] = explain_favorite_pick(r, fighters_df)
            else:
                r["rationale"] = explain_edge(r, fighters_df)
    return picks


def top_standout_props(
    edges_df: pd.DataFrame, fighters_df: pd.DataFrame | None = None, n: int = 5, min_edge: float = 5.0
) -> list[dict]:
    """
    The headline 'worth a look' props. Only positive edges qualify -- a
    negative edge just means the OTHER side of that same line is the value
    play, which will already show up as its own positive-edge entry, so
    showing both is redundant and confusing (looks like two different
    findings when it's really one).
    """
    if edges_df.empty:
        return []
    standout = edges_df[edges_df["edge_pct"] >= min_edge].copy()
    standout = standout.sort_values("edge_pct", ascending=False).head(n)
    records = standout.to_dict("records")
    for r in records:
        if pd.isna(r.get("opponent")):
            r["opponent"] = None
        r["low_sample"] = _sample_size_flag(r["fighter"], fighters_df)
        try:
            r["model_fair_odds"] = format_american_odds(implied_prob_to_american(r["model_prob"]))
        except (ValueError, ZeroDivisionError):
            r["model_fair_odds"] = "N/A"
        if fighters_df is not None:
            r["rationale"] = explain_edge(r, fighters_df)
    return records
