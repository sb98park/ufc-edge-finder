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


def _find_mma_tag_ids() -> list[str]:
    """
    Gamma's /sports endpoint returns tag metadata per sport. Filtering events
    by tag_id is far more reliable than sorting all active events (across
    the entire platform) by volume and hoping UFC cracks the top N -- it
    won't, since political/crypto markets dwarf individual MMA fights in
    platform-wide dollar volume even though MMA markets are significant
    within their own category.

    The sport identifier field is literally called "sport" (confirmed via
    live diagnostic), and tags come back as a comma-separated string. A
    sport can list MULTIPLE tags, and not all of them are sport-specific --
    e.g. tag "1" showed up on both the UFC and NCAAB entries in a live
    dump, meaning it's a shared generic "Sports" category, not a UFC-only
    one. Rather than guess which specific tag id is the meaningful one,
    this returns all of them and queries each, merging results.
    """
    resp = requests.get(f"{GAMMA_BASE}/sports", headers=HEADERS, timeout=20)
    resp.raise_for_status()
    sports = resp.json()

    for sport in sports:
        sport_code = (sport.get("sport") or "").lower()
        if sport_code in ("mma", "ufc"):
            tags_str = sport.get("tags", "")
            tag_ids = [t.strip() for t in str(tags_str).split(",") if t.strip()]
            print(f"[polymarket] found sport code {sport_code!r}, tags={tag_ids}")
            return tag_ids

    sample_codes = [s.get("sport") for s in sports[:30]]
    print(f"[polymarket] no MMA/UFC sport code found among {len(sports)} sports; sample codes: {sample_codes}")
    return []


def _is_individual_fight_event(event: dict) -> bool:
    """
    'UFC' alone in the title isn't enough -- it also matches year-end
    championship futures markets like 'Who will be UFC Flyweight champion
    at the end of 2026?' (confirmed via live diagnostic output, not a
    guess). An actual fight-vs-fight event always has a 'vs' in the title
    ('UFC 329: Max Holloway vs. Conor McGregor'); futures/ranking markets
    never do. Requiring both is what actually separates the two.
    """
    combined = f"{event.get('title') or ''} {event.get('slug') or ''}".lower()
    return "ufc" in combined and bool(re.search(r"\bvs\.?\b", combined))


def _fetch_events_by_tag(tag_id: str, limit: int = 200) -> list[dict]:
    resp = requests.get(
        f"{GAMMA_BASE}/events",
        params={"tag_id": tag_id, "active": "true", "closed": "false", "limit": limit},
        headers=HEADERS, timeout=20,
    )
    resp.raise_for_status()
    events = resp.json()
    matched = [e for e in events if _is_individual_fight_event(e)]
    sample_titles = [e.get("title", "")[:40] for e in events[:3]]
    print(f"[polymarket] tag_id={tag_id}: {len(events)} raw events, {len(matched)} matched the fight filter, sample: {sample_titles}")
    return matched


def _fetch_events_by_volume_fallback(limit: int = 200, pages: int = 10) -> list[dict]:
    """
    Backup discovery if tag lookup fails: paginate through volume-sorted
    events instead of just the first page. Each page is fetched defensively
    -- Gamma's /events endpoint has a real max offset limit (confirmed via
    a live 422 error around offset=2200), and without per-page error
    handling, hitting that limit on a later page would throw an exception
    that discards every event found on all the successful earlier pages.
    """
    all_ufc_events = []
    for page in range(pages):
        try:
            resp = requests.get(
                f"{GAMMA_BASE}/events",
                params={"active": "true", "closed": "false", "limit": limit, "offset": page * limit,
                         "order": "volume", "ascending": "false"},
                headers=HEADERS, timeout=20,
            )
            resp.raise_for_status()
            events = resp.json()
        except Exception as exc:
            print(f"[polymarket] volume-sort pagination stopped at page {page} (offset={page * limit}): {exc}")
            break

        if not events:
            break
        matched = [e for e in events if _is_individual_fight_event(e)]
        all_ufc_events.extend(matched)
    print(f"[polymarket] volume-sorted fallback found {len(all_ufc_events)} UFC events")
    return all_ufc_events


def fetch_ufc_events(limit: int = 200) -> list[dict]:
    found: dict[str, dict] = {}  # keyed by slug/title to dedupe across strategies

    try:
        tag_ids = _find_mma_tag_ids()
        for tag_id in tag_ids:
            for e in _fetch_events_by_tag(tag_id, limit):
                found[e.get("slug") or e.get("title")] = e
    except Exception as exc:
        print(f"[polymarket] tag lookup failed ({exc})")

    # Volume-sorted pagination as a backup/supplement -- confirmed live to
    # find real fight events, just needs enough depth since individual MMA
    # fights rank far below the platform's biggest political/crypto markets.
    # (End-date sorting was tried and confirmed to be a dead end -- it's
    # dominated by elections resolving soon and 5-minute crypto markets,
    # not multi-day-out events like this.)
    for e in _fetch_events_by_volume_fallback(limit, pages=10):
        found[e.get("slug") or e.get("title")] = e

    events = list(found.values())
    print(f"[polymarket] {len(events)} unique UFC fight events found after merging all discovery strategies")
    return events


def _extract_method(text: str) -> str | None:
    text_lower = text.lower()
    for keyword, method in METHOD_KEYWORDS.items():
        if keyword in text_lower:
            return method
    return None


def _extract_round_line(text: str) -> str | None:
    match = re.search(r"(\d+\.?\d*)\s*round", text.lower())
    return match.group(1) if match else None


CLOB_BASE = "https://clob.polymarket.com"


def fetch_price_history(token_id: str, interval: str = "max") -> list[dict]:
    """
    Pulls REAL historical price data for a specific outcome token from
    Polymarket's CLOB API -- this is the same data backing Polymarket's own
    price charts, going back to when the market opened, not just what we've
    accumulated ourselves since this site started tracking. Public, no auth.
    Returns [{"t": unix_timestamp, "p": price_0_to_1}, ...].
    """
    if not token_id:
        return []

    def _try_request(params: dict) -> list[dict]:
        resp = requests.get(f"{CLOB_BASE}/prices-history", params=params, headers=HEADERS, timeout=15)
        resp.raise_for_status()
        return resp.json().get("history", [])

    try:
        history = _try_request({"market": token_id, "interval": interval})

        # If interval=max returned suspiciously few points, try explicit
        # start/end timestamps instead -- a different parameter path in case
        # "max" isn't behaving as documented for this market.
        if len(history) < 5:
            import time
            fallback = _try_request({"market": token_id, "startTs": 0, "endTs": int(time.time())})
            if len(fallback) > len(history):
                print(f"[polymarket] interval=max returned only {len(history)} points for token "
                      f"{token_id[:12]}...; startTs/endTs fallback returned {len(fallback)} instead")
                history = fallback

        if history:
            from datetime import datetime, timezone
            first_dt = datetime.fromtimestamp(history[0]["t"], tz=timezone.utc).strftime("%Y-%m-%d")
            last_dt = datetime.fromtimestamp(history[-1]["t"], tz=timezone.utc).strftime("%Y-%m-%d")
            print(f"[polymarket] price history for token {token_id[:12]}...: {len(history)} points, {first_dt} to {last_dt}")
        else:
            print(f"[polymarket] price history for token {token_id[:12]}... returned ZERO points (empty history)")
        return history
    except Exception as exc:
        print(f"[polymarket] price history fetch failed for token {token_id}: {exc}")
        return []


def _fighter_name_in_text(fighter_name: str, text: str) -> bool:
    """Checks if any meaningful name part (first or last, skipping short tokens like initials) appears in the text."""
    text_lower = text.lower()
    parts = [p for p in fighter_name.lower().split() if len(p) > 2]
    return any(part in text_lower for part in parts)


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
    a_matched = _fighter_name_in_text(fighter_a, question)
    b_matched = _fighter_name_in_text(fighter_b, question)
    fighter = fighter_a if (a_matched and not b_matched) else (fighter_b if (b_matched and not a_matched) else None)

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

    # Sanity check: a real two-sided market's prices should sum close to
    # 1.0 (allowing some spread for vig/liquidity). A pair like 0.93+0.76
    # (=1.69) is a strong signal of stale/illiquid data on a thin market --
    # trusting either side individually would show a misleading price, so
    # skip it entirely rather than risk publishing a wrong number.
    price_sum = price_a + price_b
    if not (0.85 <= price_sum <= 1.15):
        print(f"[polymarket] skipping implausible market (prices sum to {price_sum:.2f}, not ~1.0): {question[:80]!r}")
        return []

    rows = []

    is_yes_no = {o.strip().lower() for o in outcomes} == {"yes", "no"}
    clob_token_ids = _safe_json_loads(market.get("clobTokenIds"))

    if not is_yes_no:
        # outcomes ARE the two fighter names -- a moneyline market
        fighter_a, fighter_b = outcomes[0], outcomes[1]
        token_a = clob_token_ids[0] if len(clob_token_ids) >= 2 else None
        token_b = clob_token_ids[1] if len(clob_token_ids) >= 2 else None
        if not token_a or not token_b:
            print(f"[polymarket] no clobTokenIds found for {fighter_a} vs {fighter_b} "
                  f"(raw field: {market.get('clobTokenIds')!r}) -- chart will fall back to accumulated snapshot data")
        for fighter, opponent, price, token_id in [
            (fighter_a, fighter_b, price_a, token_a), (fighter_b, fighter_a, price_b, token_b)
        ]:
            try:
                odds = implied_prob_to_american(price)
            except (ValueError, ZeroDivisionError):
                continue
            rows.append({
                "fight_id": fight_id, "fighter_a": fighter_a, "fighter_b": fighter_b,
                "market": "Moneyline", "selection": fighter, "selection_method": "",
                "odds_american": odds, "clob_token_id": token_id,
            })
        return rows

    # Yes/No prop question -- use the event title for a reliable fighter pair,
    # since the question text alone often doesn't name the opponent
    if not title_pair:
        return []  # can't safely attribute this prop to a specific matchup
    fighter_a, fighter_b = title_pair

    method = _extract_method(question)
    round_line = _extract_round_line(question)

    try:
        yes_odds = implied_prob_to_american(price_a)
        no_odds = implied_prob_to_american(1 - price_a)
    except (ValueError, ZeroDivisionError):
        return []

    yes_token = clob_token_ids[0] if len(clob_token_ids) >= 1 else None
    no_token = clob_token_ids[1] if len(clob_token_ids) >= 2 else None

    # Fight-level questions ("Fight to Go the Distance?") never name either
    # fighter and never needed attribution -- handle these BEFORE the
    # fighter-matching check below, which is only relevant to fighter-
    # specific method claims. Confirmed live: this ordering bug was
    # silently dropping every "Goes the Distance" market on the board.
    if "distance" in question.lower():
        rows.append({
            "fight_id": fight_id, "fighter_a": fighter_a, "fighter_b": fighter_b,
            "market": "GoesTheDistance", "selection": "Goes The Distance", "selection_method": "",
            "odds_american": yes_odds, "clob_token_id": yes_token,
        })
        rows.append({
            "fight_id": fight_id, "fighter_a": fighter_a, "fighter_b": fighter_b,
            "market": "GoesTheDistance", "selection": "Ends In Finish", "selection_method": "",
            "odds_american": no_odds, "clob_token_id": no_token,
        })
        return rows

    if round_line:
        rows.append({
            "fight_id": fight_id, "fighter_a": fighter_a, "fighter_b": fighter_b,
            "market": "TotalRounds", "selection": f"Under {round_line}", "selection_method": round_line,
            "odds_american": yes_odds, "clob_token_id": yes_token,
        })
        rows.append({
            "fight_id": fight_id, "fighter_a": fighter_a, "fighter_b": fighter_b,
            "market": "TotalRounds", "selection": f"Over {round_line}", "selection_method": round_line,
            "odds_american": no_odds, "clob_token_id": no_token,
        })
        return rows

    if not method:
        # not a method claim, not a distance claim, not a rounds claim --
        # nothing we know how to classify (e.g. a fight-level "won by
        # KO/TKO regardless of winner" question doesn't map to our
        # per-fighter Method market structure; safer to skip than force-fit it)
        return []

    # Method-of-victory claims genuinely DO need to know which fighter --
    # "Will X win by KO/TKO" is fighter-specific, unlike distance/rounds.
    # Checking both first AND last name tokens is more robust against
    # nicknames/short forms. If neither fighter is confidently matched,
    # DROP the row instead of guessing -- a wrong attribution (crediting
    # one fighter's real price to the other) is worse than a missing point.
    a_matched = _fighter_name_in_text(fighter_a, question)
    b_matched = _fighter_name_in_text(fighter_b, question)

    if a_matched and not b_matched:
        fighter = fighter_a
    elif b_matched and not a_matched:
        fighter = fighter_b
    else:
        return []  # ambiguous or unmatched -- don't guess

    rows.append({
        "fight_id": fight_id, "fighter_a": fighter_a, "fighter_b": fighter_b,
        "market": "Method", "selection": fighter, "selection_method": method,
        "odds_american": yes_odds, "clob_token_id": yes_token,
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
