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
        is_five_round = str(row.get("card_position", "")).strip() == "Main Event"
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
            is_five_round = str(fight.get("card_position", "")).strip() == "Main Event"
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
                    "label": build_market_label(e.get("fighter"), e.get("market")),
                    "selection": e.get("selection"),
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
            for gid, gname, gsub in (
                ("fighter", "Fighter props", "Who wins, and how"),
                ("fight", "Fight props", "How the fight ends, regardless of winner"),
                ("rounds", "Round props", "When it ends"),
            ):
                rows = [r for r in merged if _group_of(r.get("market")) == gid]
                if gid == "fight":
                    rows = _canonical_fight_methods(rows, fight)
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
    ].copy()
    if candidates.empty:
        return []

    candidates = candidates.sort_values("model_prob", ascending=False)
    seen_fights = set()
    picks = []
    for _, row in candidates.iterrows():
        fight_id = row.get("fight_id")
        if fight_id in seen_fights:
            continue
        seen_fights.add(fight_id)
        picks.append(row.to_dict())
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
