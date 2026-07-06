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
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept": "application/json",
}

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


def _find_mma_tag_id() -> str | None:
    """
    Gamma's /sports endpoint returns tag metadata per sport. Filtering events
    by this tag_id is far more reliable than sorting all active events
    (across the entire platform) by volume and hoping UFC cracks the top N --
    it won't, since political/crypto markets dwarf individual MMA fights in
    platform-wide dollar volume even though MMA markets are significant
    within their own category.
    """
    resp = requests.get(f"{GAMMA_BASE}/sports", headers=HEADERS, timeout=20)
    resp.raise_for_status()
    sports = resp.json()
    for sport in sports:
        label = (sport.get("label") or sport.get("name") or sport.get("slug") or "").lower()
        if "mma" in label or "ufc" in label:
            tag_id = sport.get("id") or sport.get("tagId") or sport.get("tag_id")
            print(f"[polymarket] found MMA/UFC tag: {label!r} (tag_id={tag_id})")
            return str(tag_id) if tag_id is not None else None

    # No exact match -- show anything combat-sports-adjacent so we can spot
    # the real label name instead of just knowing the exact match failed
    combat_adjacent = [
        (s.get("label") or s.get("name") or s.get("slug") or "")
        for s in sports
        if any(kw in (s.get("label") or s.get("name") or s.get("slug") or "").lower()
               for kw in ["fight", "combat", "box", "wrestl", "martial"])
    ]
    print(f"[polymarket] no exact MMA/UFC tag found among {len(sports)} sports")
    print(f"[polymarket] combat-sports-adjacent labels found: {combat_adjacent[:15]}")
    return None


def _fetch_events_by_tag(tag_id: str, limit: int = 200) -> list[dict]:
    resp = requests.get(
        f"{GAMMA_BASE}/events",
        params={"tag_id": tag_id, "active": "true", "closed": "false", "limit": limit},
        headers=HEADERS, timeout=20,
    )
    resp.raise_for_status()
    events = resp.json()
    print(f"[polymarket] tag-based lookup returned {len(events)} events")
    return [e for e in events if "ufc" in (e.get("title") or "").lower() or "ufc" in (e.get("slug") or "").lower()]


def _fetch_events_by_volume_fallback(limit: int = 200, pages: int = 3) -> list[dict]:
    """Backup discovery if tag lookup fails: paginate through volume-sorted events instead of just the first page."""
    all_ufc_events = []
    for page in range(pages):
        resp = requests.get(
            f"{GAMMA_BASE}/events",
            params={"active": "true", "closed": "false", "limit": limit, "offset": page * limit,
                     "order": "volume", "ascending": "false"},
            headers=HEADERS, timeout=20,
        )
        resp.raise_for_status()
        events = resp.json()
        if not events:
            break
        all_ufc_events.extend(
            e for e in events if "ufc" in (e.get("title") or "").lower() or "ufc" in (e.get("slug") or "").lower()
        )
    print(f"[polymarket] volume-sorted fallback (paginated) found {len(all_ufc_events)} UFC events")
    return all_ufc_events


def fetch_ufc_events(limit: int = 200) -> list[dict]:
    try:
        tag_id = _find_mma_tag_id()
        if tag_id:
            ufc_events = _fetch_events_by_tag(tag_id, limit)
            if ufc_events:
                return ufc_events
            print("[polymarket] tag-based lookup found no UFC-titled events, falling back to volume-sorted pagination")
    except Exception as exc:
        print(f"[polymarket] tag lookup failed ({exc}), falling back to volume-sorted pagination")

    return _fetch_events_by_volume_fallback(limit)


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


def _parse_multi_outcome_market(
    outcomes: list, prices: list, question: str, event_title: str,
    title_pair: tuple[str, str] | None, fight_id: str,
) -> list[dict]:
    """
    Handles markets with 3+ outcomes in one shot -- e.g. a single 'How does
    the fight end?' market with outcomes ['KO/TKO', 'Submission', 'Decision']
    instead of three separate Yes/No questions. Also handles round-by-round
    markets (outcomes like 'Round 1', 'Round 2', ..., 'Decision') by mapping
    each round outcome into an equivalent Under/Over total-rounds price.
    """
    if not title_pair:
        return []
    fighter_a, fighter_b = title_pair
    fighter = fighter_a if fighter_a.split()[-1].lower() in question.lower() else (
        fighter_b if fighter_b.split()[-1].lower() in question.lower() else None
    )

    rows = []
    round_outcomes = []  # (round_number, price) pairs, if this looks like a round-by-round market

    for outcome_label, price_raw in zip(outcomes, prices):
        try:
            price = float(price_raw)
        except (TypeError, ValueError):
            continue

        method = _extract_method(outcome_label)
        if method and fighter:
            try:
                odds = implied_prob_to_american(price)
            except (ValueError, ZeroDivisionError):
                continue
            rows.append({
                "fight_id": fight_id, "fighter_a": fighter_a, "fighter_b": fighter_b,
                "market": "Method", "selection": fighter, "selection_method": method,
                "odds_american": odds,
            })
            continue

        round_match = re.search(r"round\s*(\d+)", outcome_label.lower())
        if round_match:
            round_outcomes.append((int(round_match.group(1)), price))

    # Round-by-round outcomes -> derive Under/Over total-rounds prices at each
    # boundary by summing cumulative probability (e.g. P(under 2.5) = P(round 1) + P(round 2))
    if round_outcomes:
        round_outcomes.sort()
        cumulative = 0.0
        for round_num, price in round_outcomes:
            cumulative += price
            line = round_num + 0.5
            try:
                under_odds = implied_prob_to_american(min(0.99, cumulative))
                over_odds = implied_prob_to_american(max(0.01, 1 - cumulative))
            except (ValueError, ZeroDivisionError):
                continue
            rows.append({
                "fight_id": fight_id, "fighter_a": fighter_a, "fighter_b": fighter_b,
                "market": "TotalRounds", "selection": f"Under {line}", "selection_method": str(line),
                "odds_american": under_odds,
            })
            rows.append({
                "fight_id": fight_id, "fighter_a": fighter_a, "fighter_b": fighter_b,
                "market": "TotalRounds", "selection": f"Over {line}", "selection_method": str(line),
                "odds_american": over_odds,
            })

    return rows


def _classify_and_parse_market(market: dict, event_title: str) -> list[dict]:
    """Turns one Gamma market object into 0+ rows matching our upcoming-props schema."""
    question = market.get("question", "")
    outcomes = _safe_json_loads(market.get("outcomes"))
    prices = _safe_json_loads(market.get("outcomePrices"))
    if len(outcomes) < 2 or len(outcomes) != len(prices):
        return []

    fight_id = event_title  # group by event, not individual market id, so all markets for one fight share a key
    title_pair = _extract_matchup_from_title(event_title)

    if len(outcomes) > 2:
        return _parse_multi_outcome_market(outcomes, prices, question, event_title, title_pair, fight_id)

    try:
        price_a, price_b = float(prices[0]), float(prices[1])
    except (TypeError, ValueError):
        return []

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
    markets_seen = 0
    outcome_count_histogram: dict[int, int] = {}
    dropped_samples = []  # actual raw content of dropped markets, to see real phrasing instead of guessing

    for event in events:
        for market in event.get("markets", []):
            markets_seen += 1
            outcomes = _safe_json_loads(market.get("outcomes"))
            outcome_count_histogram[len(outcomes)] = outcome_count_histogram.get(len(outcomes), 0) + 1

            parsed = _classify_and_parse_market(market, event.get("title", ""))
            rows.extend(parsed)

            if not parsed and len(dropped_samples) < 8:
                dropped_samples.append({
                    "event_title": event.get("title", "")[:80],
                    "question": market.get("question", "")[:100],
                    "outcomes": outcomes,
                })

    print(f"[polymarket] outcome-count breakdown across all markets: {outcome_count_histogram}")
    print(f"[polymarket] classified {markets_seen} markets into {len(rows)} usable rows")
    if dropped_samples:
        print(f"[polymarket] sample of {len(dropped_samples)} DROPPED markets (actual raw content, not a guess):")
        for s in dropped_samples:
            print(f"  event={s['event_title']!r} | question={s['question']!r} | outcomes={s['outcomes']}")
    return rows
