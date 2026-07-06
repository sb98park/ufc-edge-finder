"""
Pulls live UFC odds from Polymarket's Gamma API (https://gamma-api.polymarket.com).
Fully public, no authentication required. Unlike DraftKings' reverse-engineered
endpoints, this is Polymarket's actual documented API, so it should be far more
stable long-term.

Key quirks worth knowing (these caused real bugs in early testing/community
reports, so they're handled explicitly here):
  - outcomes / outcomePrices / clobTokenIds come back as STRINGIFIED JSON
    (e.g. the string '["0.62", "0.38"]', not a real array) -- must be
    json.loads()'d, or you end up indexing into individual characters.
  - Gamma has no free-text search param on /events, so discovery is done by
    pulling active/open events and filtering client-side by title.
  - Prices are share prices (0-1), which ARE probabilities directly --
    Polymarket is peer-to-peer with no bookmaker vig, unlike a sportsbook.

For a head-to-head market like "Max Holloway vs. Conor McGregor", the two
`outcomes` are typically the fighter names themselves (not "Yes"/"No").
For a prop question like "Will McGregor win by KO/TKO?", outcomes are
Yes/No and the fighter + method have to be pulled from the question text.
"""

import json
import re

import requests

from src.odds_utils import implied_prob_to_american

GAMMA_BASE = "https://gamma-api.polymarket.com"
HEADERS = {"User-Agent": "Mozilla/5.0 (personal research script)"}

METHOD_KEYWORDS = {
    "ko/tko": "KO/TKO", "knockout": "KO/TKO", "tko": "KO/TKO",
    "submission": "SUB",
    "decision": "DEC", "points": "DEC",
}


def _safe_json_loads(value, default=None):
    if value is None:
        return default if default is not None else []
    if isinstance(value, (list, dict)):
        return value  # already parsed
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return default if default is not None else []


def fetch_ufc_events(limit: int = 200) -> list[dict]:
    """
    Gamma has no text search on /events, so pull active/open events ordered
    by volume and filter client-side for UFC-titled events.
    """
    resp = requests.get(
        f"{GAMMA_BASE}/events",
        params={"active": "true", "closed": "false", "limit": limit, "order": "volume", "ascending": "false"},
        headers=HEADERS, timeout=20,
    )
    resp.raise_for_status()
    events = resp.json()
    return [e for e in events if "ufc" in (e.get("title") or "").lower() or "ufc" in (e.get("slug") or "").lower()]


def _extract_method(text: str) -> str | None:
    text_lower = text.lower()
    for keyword, method in METHOD_KEYWORDS.items():
        if keyword in text_lower:
            return method
    return None


def _extract_round_line(text: str) -> str | None:
    match = re.search(r"(\d+\.?\d*)\s*round", text.lower())
    return match.group(1) if match else None


def _extract_matchup_from_title(event_title: str) -> tuple[str, str] | None:
    """
    Event titles follow a consistent 'X vs. Y' pattern (e.g. 'UFC 329: Max
    Holloway vs. Conor McGregor (Welterweight, Main Card)'), which is a much
    more reliable source for the fighter pair than trying to parse it out of
    an individual Yes/No prop question's wording.
    """
    # strip a leading "UFC 329:" style prefix and trailing "(...)" suffix
    cleaned = re.sub(r"^[^:]+:\s*", "", event_title)
    cleaned = re.sub(r"\s*\([^)]*\)\s*$", "", cleaned).strip()
    match = re.search(r"(.+?)\s+vs\.?\s+(.+)", cleaned, re.IGNORECASE)
    if match:
        return match.group(1).strip(), match.group(2).strip()
    return None


def _classify_and_parse_market(market: dict, event_title: str) -> list[dict]:
    """Turns one Gamma market object into 0+ rows matching our upcoming-props schema."""
    question = market.get("question", "")
    outcomes = _safe_json_loads(market.get("outcomes"))
    prices = _safe_json_loads(market.get("outcomePrices"))
    if len(outcomes) != 2 or len(prices) != 2:
        return []

    try:
        price_a, price_b = float(prices[0]), float(prices[1])
    except (TypeError, ValueError):
        return []

    fight_id = event_title  # group by event, not individual market id, so all markets for one fight share a key
    title_pair = _extract_matchup_from_title(event_title)
    rows = []

    is_yes_no = {o.strip().lower() for o in outcomes} == {"yes", "no"}

    if not is_yes_no:
        # outcomes ARE the two fighter names -- a moneyline market
        fighter_a, fighter_b = outcomes[0], outcomes[1]
        for fighter, opponent, price in [(fighter_a, fighter_b, price_a), (fighter_b, fighter_a, price_b)]:
            try:
                odds = implied_prob_to_american(price)
            except (ValueError, ZeroDivisionError):
                continue
            rows.append({
                "fight_id": fight_id, "fighter_a": fighter_a, "fighter_b": fighter_b,
                "market": "Moneyline", "selection": fighter, "selection_method": "",
                "odds_american": odds,
            })
        return rows

    # Yes/No prop question -- use the event title for a reliable fighter pair,
    # since the question text alone often doesn't name the opponent
    if not title_pair:
        return []  # can't safely attribute this prop to a specific matchup
    fighter_a, fighter_b = title_pair

    method = _extract_method(question)
    round_line = _extract_round_line(question)

    # best-effort: which of the two fighters is this specific prop about?
    fighter = fighter_a if fighter_a.split()[-1].lower() in question.lower() else (
        fighter_b if fighter_b.split()[-1].lower() in question.lower() else fighter_a
    )

    try:
        yes_odds = implied_prob_to_american(price_a)
        no_odds = implied_prob_to_american(1 - price_a)
    except (ValueError, ZeroDivisionError):
        return []

    if method:
        rows.append({
            "fight_id": fight_id, "fighter_a": fighter_a, "fighter_b": fighter_b,
            "market": "Method", "selection": fighter, "selection_method": method,
            "odds_american": yes_odds,
        })
    elif "distance" in question.lower():
        rows.append({
            "fight_id": fight_id, "fighter_a": fighter_a, "fighter_b": fighter_b,
            "market": "GoesTheDistance", "selection": "Goes The Distance", "selection_method": "",
            "odds_american": yes_odds,
        })
        rows.append({
            "fight_id": fight_id, "fighter_a": fighter_a, "fighter_b": fighter_b,
            "market": "GoesTheDistance", "selection": "Ends In Finish", "selection_method": "",
            "odds_american": no_odds,
        })
    elif round_line:
        rows.append({
            "fight_id": fight_id, "fighter_a": fighter_a, "fighter_b": fighter_b,
            "market": "TotalRounds", "selection": f"Under {round_line}", "selection_method": round_line,
            "odds_american": yes_odds,
        })
        rows.append({
            "fight_id": fight_id, "fighter_a": fighter_a, "fighter_b": fighter_b,
            "market": "TotalRounds", "selection": f"Over {round_line}", "selection_method": round_line,
            "odds_american": no_odds,
        })

    return rows


def fetch_polymarket_ufc_props() -> list[dict]:
    """Convenience wrapper: find UFC events, parse every nested market."""
    events = fetch_ufc_events()
    rows = []
    for event in events:
        for market in event.get("markets", []):
            rows.extend(_classify_and_parse_market(market, event.get("title", "")))
    return rows
