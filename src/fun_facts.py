"""
Fun-fact detection for current-card fighters: genuinely notable patterns
only, gated so nothing filler ever surfaces. A fighter with no fact
clearing the gates simply gets no fact -- an empty week is the correct
output, not a failure.

Sources, in trust order:
- data/fight_history.csv (pipeline-owned, auto-updated): source of
  truth for WHICH streaks exist. Coarse method granularity (KO/SUB/DEC).
- data/fighters.csv method breakdown columns: career-purity facts.
- data/ufc_fight_results.csv (optional, manually refreshed): used ONLY
  to enrich an already-detected submission/KO streak with the specific
  finishing move ("4 straight heel hooks") when it covers those same
  fights AND they all share one move. If it's stale or absent, facts
  gracefully stay at the coarse tier -- completed fights never change,
  so staleness can only ever mean "less detail," never a wrong fact.
"""

import os
import re

import pandas as pd

from src.elo import ufc_only

from src.card_matcher import _normalize_name

RESULTS_DETAIL_PATH = "data/ufc_fight_results.csv"

METHOD_LABEL = {"SUB": "submission", "KO/TKO": "knockout", "DEC": "decision"}
MIN_METHOD_STREAK = 3
MIN_WIN_STREAK = 5
MIN_WINS_FOR_PURITY = 8
MIN_LOSSES_FOR_IRON_CHIN = 4


def _strip_move_qualifiers(details: str) -> str:
    """'Heel Hook After Drop to Ground ' -> 'Heel Hook'."""
    s = str(details).strip()
    s = re.split(r"\s+(?:After|On|From|To)\s+", s, maxsplit=1)[0]
    return s.strip()


def _load_move_details() -> dict:
    """(normalized winner name, normalized loser name) -> specific move, when knowable."""
    if not os.path.exists(RESULTS_DETAIL_PATH):
        return {}
    try:
        df = pd.read_csv(RESULTS_DETAIL_PATH)
        df.columns = [c.strip() for c in df.columns]
    except Exception:
        return {}
    lookup = {}
    for _, row in df.iterrows():
        method = str(row.get("METHOD", "")).strip()
        if not method.startswith(("Submission", "KO/TKO")):
            continue
        bout = str(row.get("BOUT", ""))
        if " vs. " not in bout:
            continue
        a, b = (p.strip() for p in bout.split(" vs. ", 1))
        move = _strip_move_qualifiers(row.get("DETAILS", ""))
        if not move or len(move) > 40:
            continue
        key = frozenset({_normalize_name(a), _normalize_name(b)})
        # Rematch pairs would collide here -- keep only if unambiguous.
        if key in lookup and lookup[key] != move:
            lookup[key] = None
        else:
            lookup[key] = move
    return {k: v for k, v in lookup.items() if v}


def _fighter_wins_chronological(history: pd.DataFrame, name: str) -> list[dict]:
    norm = _normalize_name(name)
    rows = history[
        (history["fighter_a"].map(_normalize_name) == norm)
        | (history["fighter_b"].map(_normalize_name) == norm)
    ].sort_values("date")
    out = []
    for _, r in rows.iterrows():
        # A winnerless row (draw/no contest) lands here as False, which ENDS
        # both streaks below. That is deliberate for a published claim:
        # "riding a 5-fight win streak" is not true of a fighter whose last
        # outing was a no contest. power_rating's streak skips the row instead
        # and keeps the run alive -- a model term and a headline are allowed
        # to disagree about what a non-result means.
        won = _normalize_name(str(r["winner"])) == norm
        opponent = r["fighter_b"] if _normalize_name(str(r["fighter_a"])) == norm else r["fighter_a"]
        out.append({"won": won, "method": r["method"], "opponent": opponent})
    return out


def detect_facts_for_fighter(name: str, history: pd.DataFrame, fighter_row, move_lookup: dict) -> dict | None:
    """Best single fact for this fighter, or None if nothing clears the gates."""
    candidates = []
    fights = _fighter_wins_chronological(history, name)

    # --- Active same-method win streak (must be the CURRENT streak: the
    # fighter's most recent fights, wins all by one method, no loss inside it)
    recent_wins = []
    for f in reversed(fights):
        if not f["won"]:
            break
        recent_wins.append(f)
    if len(recent_wins) >= MIN_METHOD_STREAK:
        methods = {f["method"] for f in recent_wins}
        if len(methods) == 1 and (m := methods.pop()) in METHOD_LABEL:
            label = METHOD_LABEL[m]
            n = len(recent_wins)
            text = f"Last {n} wins all by {label}"
            # Tier ladder: hot -> gold (specific move) -> legendary (long
            # streaks: 6+ same coarse method, or 5+ by one SPECIFIC move)
            tier = "legendary" if n >= 6 else "hot"
            if m == "SUB" and move_lookup:
                moves = set()
                for f in recent_wins:
                    key = frozenset({_normalize_name(name), _normalize_name(str(f["opponent"]))})
                    moves.add(move_lookup.get(key))
                if len(moves) == 1 and (mv := moves.pop()):
                    text = f"Last {n} wins all by {mv.lower()}"
                    tier = "legendary" if n >= 5 else "gold"
            candidates.append({
                "text": text, "number": n, "tier": tier,
                "score": n * 10 + (25 if tier == "gold" else 0) + (60 if tier == "legendary" else 0),
            })

    # --- Overall win streak (method-agnostic)
    streak = 0
    for f in reversed(fights):
        if f["won"]:
            streak += 1
        else:
            break
    if streak >= MIN_WIN_STREAK:
        candidates.append({
            "text": f"Riding a {streak}-fight win streak", "number": streak,
            "tier": "legendary" if streak >= 12 else "hot",
            "score": streak * 6 + (60 if streak >= 12 else 0),
        })

    # --- Career purity from the method breakdown (covers pre-UFC too)
    if fighter_row is not None:
        ko, sub, dec = (fighter_row.get(k) for k in ("ko_wins", "sub_wins", "dec_wins"))
        if all(pd.notna(v) for v in (ko, sub, dec)):
            ko, sub, dec = int(ko), int(sub), int(dec)
            total = ko + sub + dec
            if total >= MIN_WINS_FOR_PURITY:
                # 14+ wins of pure finishing (or one pure method) is a
                # career-defining anomaly, not just a rarity -- own tier.
                purity_tier = "legendary" if total >= 14 else "gold"
                bonus = 70 if purity_tier == "legendary" else 20
                if dec == 0:
                    candidates.append({
                        "text": f"Has never won by decision — {total} finishes in {total} wins",
                        "number": total, "tier": purity_tier, "score": total * 8 + bonus,
                    })
                elif ko == total or sub == total:
                    label = METHOD_LABEL["KO/TKO" if ko == total else "SUB"]
                    candidates.append({
                        "text": f"All {total} career wins by {label}",
                        "number": total, "tier": purity_tier, "score": total * 9 + bonus + 5,
                    })
        kol, subl, decl = (fighter_row.get(k) for k in ("ko_losses", "sub_losses", "dec_losses"))
        if all(pd.notna(v) for v in (kol, subl, decl)):
            kol, subl, decl = int(kol), int(subl), int(decl)
            losses = kol + subl + decl
            if losses >= MIN_LOSSES_FOR_IRON_CHIN and kol == 0 and subl == 0:
                candidates.append({
                    "text": f"Has never been finished — all {losses} career losses went the distance",
                    "number": losses, "tier": "hot", "score": losses * 7 + 10,
                })

    if not candidates:
        return None
    best = max(candidates, key=lambda c: c["score"])
    return {"fighter": name, **best}


def compute_fun_facts(card_fighter_names: list[str], fight_history_path: str, fighters_df: pd.DataFrame) -> list[dict]:
    # UFC bouts only, for COMPARABILITY rather than for Elo's reasons. These
    # facts are superlatives ranked against each other ("longest active KO
    # streak on the card"), and the spine holds a complete career only for the
    # handful of fighters whose regional record was entered by hand. Mixing
    # the two would let one fighter's fuller record beat another's on nothing
    # but data availability, which is exactly the kind of claim CLAUDE.md s7
    # exists to stop.
    history = ufc_only(pd.read_csv(fight_history_path))
    move_lookup = _load_move_details()
    by_norm = {_normalize_name(str(r["name"])): r for _, r in fighters_df.iterrows()}
    facts = []
    for name in card_fighter_names:
        row = by_norm.get(_normalize_name(name))
        fact = detect_facts_for_fighter(name, history, row, move_lookup)
        if fact:
            facts.append(fact)
    facts.sort(key=lambda f: f["score"], reverse=True)
    return facts
