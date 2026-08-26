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


# Surnames from the cards we're actually tracking, lowercased. Populated by
# set_known_fighters() before discovery; empty is safe -- the filter simply
# falls back to its original "ufc" + "vs" behaviour.
_KNOWN_FIGHT_PAIRS: list[tuple[str, str]] = []
_KNOWN_SLUG_PARTS: list[tuple[str, str, str]] = []


def _fold(text: str) -> str:
    """
    Lowercase and strip diacritics. Essential, not cosmetic: our roster holds
    "Uroš Medić" while Polymarket writes "Medic vs. Rodriguez", so a raw
    comparison misses every fighter with an accent -- which on an
    international card is most of them.
    """
    import unicodedata
    return "".join(c for c in unicodedata.normalize("NFKD", str(text).lower())
                   if not unicodedata.combining(c))


def set_known_fighters(pairs) -> None:
    """
    Register the BOUTS we're tracking as (surname_a, surname_b) pairs.

    Matching single name-parts was tried first and let real garbage through:
    "LoL: Bilibili Gaming Junior vs KT Rolster Challengers" matched because
    "Junior" appears in a tracked fighter's name (Junior Tafa). Any one token
    is a coin flip against a platform carrying esports, tennis and football.
    Requiring BOTH surnames from the SAME bout makes a false positive
    essentially impossible -- two specific surnames don't co-occur by accident.
    """
    global _KNOWN_FIGHT_PAIRS, _KNOWN_SLUG_PARTS
    _KNOWN_SLUG_PARTS = []
    for item in (pairs or []):
        a, b = item[0], item[1]
        date = item[2] if len(item) > 2 else None
        fa = _fold(a).strip().split()
        fb = _fold(b).strip().split()
        if fa and fb and date:
            _KNOWN_SLUG_PARTS.append((fa[0][:3], fb[0][:3], str(date)))
    out = []
    for item in (pairs or []):
        a, b = item[0], item[1]
        sa = _fold(a).strip().split()[-1:] or [""]
        sb = _fold(b).strip().split()[-1:] or [""]
        if len(sa[0]) >= 4 and len(sb[0]) >= 4:
            out.append((sa[0], sb[0]))
    _KNOWN_FIGHT_PAIRS = out


def _is_individual_fight_event(event: dict) -> bool:
    """
    'UFC' alone in the title isn't enough -- it also matches year-end
    championship futures markets like 'Who will be UFC Flyweight champion
    at the end of 2026?' (confirmed via live diagnostic output, not a
    guess). An actual fight-vs-fight event always has a 'vs' in the title
    ('UFC 329: Max Holloway vs. Conor McGregor'); futures/ranking markets
    never do. Requiring both is what actually separates the two.
    """
    combined = _fold(f"{event.get('title') or ''} {event.get('slug') or ''}")
    if not re.search(r"\bvs\.?\b", combined):
        return False          # futures/ranking markets never have "vs"
    if "ufc" in combined:
        return True
    # FALLBACK ADDED after a live run matched 0 of 200 events across both
    # tags AND ten pages of volume-sorted fallback. Requiring the literal
    # string "ufc" assumes Polymarket always prefixes fight titles that way
    # ("UFC 329: Holloway vs. McGregor"). When they title an event just
    # "Medic vs. Rodriguez", every real fight is silently rejected -- which
    # is indistinguishable, in the logs, from "there are no UFC markets".
    # Matching OUR OWN card's fighter names instead is robust to whatever
    # they call the event, because we already know exactly who is fighting.
    for sa, sb in _KNOWN_FIGHT_PAIRS:
        if sa in combined and sb in combined:
            return True
    return False


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
    vs_shaped = sum(1 for e in events
                    if re.search(r"\bvs\.?\b", f"{e.get('title') or ''} {e.get('slug') or ''}".lower()))
    print(f"[polymarket] tag_id={tag_id}: {len(events)} raw events, {len(matched)} matched the fight filter "
          f"({vs_shaped} were 'vs'-shaped -- if that's >0 while matched is 0, the title just lacks 'UFC'), "
          f"sample: {sample_titles}")
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


def _candidate_slugs() -> list[str]:
    """
    Build the slugs Polymarket most likely used for the bouts we track.

    WHY GUESS SLUGS INSTEAD OF ASKING WHICH TAG IS UFC. Both tag ids that
    Gamma's own /sports endpoint reports for UFC were confirmed live to return
    no MMA at all -- one gives Ballon d'Or and NFL markets, the other gives
    League of Legends and tennis (96 of 100 events were even "vs"-shaped, none
    were fights). The volume-sorted fallback doesn't rescue it either: an
    individual UFC prelim ranks far below the platform's political and crypto
    markets, so it never surfaces in the top pages.
    Their tag metadata simply can't be trusted for this sport. But we already
    know exactly who is fighting, and Polymarket slugs for fight markets are
    formulaic -- so ask for the specific events by name instead of hunting for
    them. A direct hit needs no discovery at all.
    """
    # CONFIRMED FORMAT (from real event URLs): ufc-jan-nav-2026-08-01 is
    # Jan Blachowicz vs Navajo Stirling -- three letters of each FIRST name,
    # plus the event date. Not surnames, which is why every earlier guess
    # missed. Collisions get a numeric suffix (ufc-dan7-uro-...), and that
    # number is unguessable -- so this will never find every fight. It only
    # has to find ONE: see _bootstrap_tag_from_event below.
    out = []
    for fa, fb, date in _KNOWN_SLUG_PARTS:
        out.append(f"ufc-{fa}-{fb}-{date}")
        out.append(f"ufc-{fb}-{fa}-{date}")
    return out


def _bootstrap_tag_ids(events: list[dict]) -> list[str]:
    """
    Read the REAL tag ids off events we actually found.

    Gamma's /sports endpoint claims UFC lives under tags 1 and 100639, and
    both were confirmed live to contain no MMA whatsoever (Ballon d'Or, NFL,
    League of Legends, tennis). Rather than trust that metadata, take the tag
    ids from a genuine UFC event -- self-correcting by construction, because
    it learns wherever Polymarket has actually filed the sport today.
    One slug hit is therefore enough to unlock the whole card, which matters
    because collision-suffixed slugs (ufc-dan7-uro-...) can't be guessed.
    """
    ids = []
    for e in events:
        for t in (e.get("tags") or []):
            tid = str(t.get("id") if isinstance(t, dict) else t).strip()
            if tid and tid not in ids:
                ids.append(tid)
    if ids:
        print(f"[polymarket] bootstrapped real tag ids from a confirmed UFC event: {ids}")
    return ids


def _fetch_events_by_slug(limit: int = 200) -> list[dict]:
    slugs = _candidate_slugs()
    if not slugs:
        return []
    found, tried = [], 0
    for slug in slugs:
        tried += 1
        try:
            resp = requests.get(f"{GAMMA_BASE}/events", headers=HEADERS, timeout=15,
                                params={"slug": slug, "closed": "false"})
            if resp.status_code != 200:
                continue
            for e in (resp.json() or []):
                if _is_individual_fight_event(e):
                    found.append(e)
        except Exception:
            continue    # one bad slug must never stop the sweep
    print(f"[polymarket] slug lookup: tried {tried} candidate slugs from tracked bouts, "
          f"found {len(found)} matching event(s)")
    return found


def fetch_ufc_events(limit: int = 200) -> list[dict]:
    found: dict[str, dict] = {}  # keyed by slug/title to dedupe across strategies

    # Slug lookup FIRST: it's a direct request for events we know exist,
    # rather than scanning thousands hoping a fight surfaces.
    try:
        for e in _fetch_events_by_slug(limit):
            found[e.get("slug") or e.get("title")] = e
    except Exception as exc:
        print(f"[polymarket] slug lookup failed ({exc})")

    # Prefer tags learned from a real event over Gamma's own (wrong) metadata.
    bootstrapped = _bootstrap_tag_ids(list(found.values()))
    for tag_id in bootstrapped:
        try:
            for e in _fetch_events_by_tag(tag_id, limit):
                found[e.get("slug") or e.get("title")] = e
        except Exception as exc:
            print(f"[polymarket] bootstrapped tag {tag_id} failed ({exc})")

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
    # NaN IS TRUTHY, which is how "market=nan" reached the wire. These ids
    # come out of a pandas frame, an absent one arrives as float('nan'), and
    # `if not token_id` is False for it -- so the guard passed, str() rendered
    # it "nan", and every one of those was a guaranteed 400. Checked by shape
    # rather than by falsiness: a CLOB token id is a long decimal string.
    token_id = str(token_id or "").strip()
    if not token_id.isdigit():
        return []

    def _try_request(params: dict) -> list[dict]:
        resp = requests.get(f"{CLOB_BASE}/prices-history", params=params, headers=HEADERS, timeout=15)
        resp.raise_for_status()
        return resp.json().get("history", [])

    # THE startTs/endTs FALLBACK IS GONE, and it was worse than useless.
    #
    # It fired whenever interval=max returned fewer than five points and asked
    # the same endpoint with explicit timestamps instead. Measured against the
    # live API on 2026-08-26:
    #
    #     interval=max                 -> 200
    #     startTs=0     + endTs=now    -> 400
    #     startTs=now-90d + endTs=now  -> 400
    #
    # The whole parameter path is rejected, not merely the zero -- so the
    # fallback could never succeed in any form. And because it raised inside
    # the try, the 400 landed in the except below and returned [] -- THROWING
    # AWAY the one to four points interval=max had already returned
    # successfully. A thin history became no history, and the log line blamed
    # the fetch rather than the fallback.
    try:
        history = _try_request({"market": token_id, "interval": interval})
        if history:
            from datetime import datetime, timezone
            first_dt = datetime.fromtimestamp(history[0]["t"], tz=timezone.utc).strftime("%Y-%m-%d")
            last_dt = datetime.fromtimestamp(history[-1]["t"], tz=timezone.utc).strftime("%Y-%m-%d")
            print(f"[polymarket] price history for token {token_id[:12]}...: "
                  f"{len(history)} points, {first_dt} to {last_dt}")
        # A valid token with no history is a resolved or never-traded market,
        # not a failure, and it was printing a line each. Silent.
        return history
    except Exception as exc:
        print(f"[polymarket] price history fetch failed for token {token_id[:12]}...: {exc}")
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

    # ROUND-BY-ROUND OUTCOMES BECOME A ROUND-START MARKET, NOT A TOTALS LINE.
    #
    # This used to emit "Over/Under {n}.5" from the cumulative round
    # probabilities, and the arithmetic was right while the LABEL was wrong.
    # After processing round n the cumulative is P(the fight ends in rounds
    # 1..n), so the complement is P(the fight REACHES round n+1) -- the moment
    # round n+1 begins. That is not the same bet as Over n.5, which settles at
    # the midpoint of round n+1 and therefore includes the first half of it.
    #
    # Emitting it as a totals line put a round-boundary probability into the
    # same pool as genuinely quoted Over/Under n.5 prices, where the model
    # compared them and the grader settled them as though they were the same
    # market. On a card where Polymarket ran a round-by-round market rather
    # than a totals market, every rounds edge was computed against the wrong
    # question.
    #
    # It also turns out to be a market worth having in its own right: books
    # quote "Fight to start round N" directly, and this is a real price for
    # it rather than a model derivation.
    #
    # THE COST IS REAL AND WORTH STATING: on a card whose Polymarket market is
    # round-by-round, there are now NO total-rounds prices at all, where
    # before there were wrong ones. A correct Over/Under n.5 cannot be
    # recovered from round-granular data, because it turns on whether the
    # fight passes the MIDPOINT of a round and nothing here splits a round in
    # half. The curve in method_model could estimate that split, but a
    # modelled number laundered into the "market price" column is the exact
    # error this project keeps finding. No price beats a wrong one.
    if round_outcomes:
        round_outcomes.sort()
        cumulative = 0.0
        for round_num, price in round_outcomes:
            cumulative += price
            starts_round = round_num + 1
            try:
                # Yes = the fight reaches that round. No = it ends before.
                yes_odds = implied_prob_to_american(max(0.01, 1 - cumulative))
                no_odds = implied_prob_to_american(min(0.99, cumulative))
            except (ValueError, ZeroDivisionError):
                continue
            rows.append({
                "fight_id": fight_id, "fighter_a": fighter_a, "fighter_b": fighter_b,
                "market": "RoundStart", "selection": f"Starts Round {starts_round}",
                "selection_method": str(starts_round), "odds_american": yes_odds,
            })
            rows.append({
                "fight_id": fight_id, "fighter_a": fighter_a, "fighter_b": fighter_b,
                "market": "RoundStart", "selection": f"Ends Before Round {starts_round}",
                "selection_method": str(starts_round), "odds_american": no_odds,
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
    # UNTRADED markets come back at exactly 0.5/0.5, which converts to -100
    # on both sides. That is not a price -- it's Polymarket's placeholder for
    # a market nobody has touched, common on a card still weeks out. Publishing
    # it is worse than showing nothing: the model then computes an "edge"
    # against a 50% that no one is actually offering, and a 30% projection
    # looks like a -20% edge when there is no market to be wrong about.
    _untouched = (
        abs(price_a - 0.5) < 1e-9 and abs(price_b - 0.5) < 1e-9
        and float(market.get("volumeNum") or market.get("volume") or 0) <= 0
    )
    if _untouched:
        print(f"[polymarket] skipping untraded market (0.5/0.5, no volume): {question[:70]!r}")
        return []

    price_sum = price_a + price_b
    if not (0.85 <= price_sum <= 1.15):
        print(f"[polymarket] skipping implausible market (prices sum to {price_sum:.2f}, not ~1.0): {question[:80]!r}")
        return []

    rows = []

    is_yes_no = {o.strip().lower() for o in outcomes} == {"yes", "no"}
    clob_token_ids = _safe_json_loads(market.get("clobTokenIds"))

    # OVER/UNDER markets are neither Yes/No nor a fighter pair. Polymarket
    # phrases round totals as "O/U 1.5 Rounds" with outcomes ["Over","Under"],
    # so the is_yes_no split sent them down the moneyline branch and invented
    # two fighters called "Over" and "Under" -- which is why TotalRounds
    # classified ZERO rows despite five lines being offered per fight.
    is_over_under = {o.strip().lower() for o in outcomes} == {"over", "under"}
    if is_over_under and title_pair:
        line = _extract_round_line(question)
        if line:
            f_a, f_b = title_pair
            over_i = 0 if outcomes[0].strip().lower() == "over" else 1
            prices = [price_a, 1 - price_a]
            try:
                over_odds = implied_prob_to_american(prices[over_i])
                under_odds = implied_prob_to_american(prices[1 - over_i])
            except (ValueError, ZeroDivisionError):
                return []
            rows.append({
                "fight_id": fight_id, "fighter_a": f_a, "fighter_b": f_b,
                "market": "TotalRounds", "selection": f"Over {line}", "selection_method": line,
                "odds_american": over_odds,
                "clob_token_id": clob_token_ids[over_i] if len(clob_token_ids) > over_i else None,
            })
            rows.append({
                "fight_id": fight_id, "fighter_a": f_a, "fighter_b": f_b,
                "market": "TotalRounds", "selection": f"Under {line}", "selection_method": line,
                "odds_american": under_odds,
                "clob_token_id": clob_token_ids[1 - over_i] if len(clob_token_ids) > (1 - over_i) else None,
            })
            return rows
        return []

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

    # FIGHT-LEVEL method question: "Will the fight be won by KO or TKO?"
    # names a method but NEITHER fighter. These were previously dropped --
    # correctly at the time, because a per-fighter Method row has nowhere to
    # put a claim that doesn't say who wins. They now have a natural home:
    # the discrete-time hazard model (research_survival_model.py) outputs
    # fight-level P(KO) and P(SUB) directly, which is precisely what this
    # market prices. Confirmed live on a real card -- these appear on nearly
    # every fight Polymarket lists.
    #
    # Handled BEFORE the fighter-matching check below, for the same reason
    # the distance branch had to be: that check requires exactly one fighter
    # to be named, so a fight-level question always falls through it.
    if method and not _fighter_name_in_text(fighter_a, question) \
             and not _fighter_name_in_text(fighter_b, question):
        rows.append({
            "fight_id": fight_id, "fighter_a": fighter_a, "fighter_b": fighter_b,
            "market": "FightMethod", "selection": method, "selection_method": method,
            "odds_american": yes_odds, "clob_token_id": yes_token,
        })
        rows.append({
            "fight_id": fight_id, "fighter_a": fighter_a, "fighter_b": fighter_b,
            "market": "FightMethod", "selection": f"Not {method}", "selection_method": method,
            "odds_american": no_odds, "clob_token_id": no_token,
        })
        return rows

    if not method:
        # not a method claim, not a distance claim, not a rounds claim
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



def _american_to_prob(odds) -> float:
    """American odds -> implied probability. Local so the gate has no import
    cycle back through odds_utils."""
    o = float(odds)
    return 100.0 / (o + 100.0) if o > 0 else (-o) / ((-o) + 100.0)


# Tolerances for the coherence gate below. Deliberately loose: the job is to
# catch a ladder that contradicts itself, not to police a couple of points of
# spread on a thin book.
ROUNDS_DISTANCE_SLACK = 0.01   # how far an Over line may sit below P(distance)
ROUNDS_SUM_SLACK      = 1.02   # implied finishes + P(distance) may reach this
ROUNDS_MIN_SPREAD     = 0.03   # a ladder flatter than this never traded


def _drop_incoherent_round_ladders(rows: list[dict]) -> list[dict]:
    """
    Remove a fight's Total Rounds ladder when the prices cannot all be true
    at once.

    WHY THIS EXISTS. Polymarket quotes an O/U ladder on most fights and it is
    read faithfully -- but on an untraded fight the quotes are not a market,
    they are placeholders that drift toward the distance price, and nothing
    downstream could tell the difference. Measured live on
    Song Yadong vs Umar Nurmagomedov:

        Fight to Go the Distance  0.720
        O/U 0.5 Over              0.895
        O/U 1.5 Over              0.745   <- published as -277
        O/U 2.5 Over              0.730
        O/U 3.5 Over              0.695

    DraftKings had Over 1.5 at -1400 the same day, about 93%. The gap is not
    two venues disagreeing; it is one venue not really quoting, and the proof
    is internal rather than comparative.

    TWO THINGS MUST HOLD, and both are arithmetic rather than opinion:

    1. GOING THE DISTANCE IMPLIES PASSING EVERY LINE. A fight that reaches
       the final bell has by definition gone past 0.5, 1.5, 2.5 and 3.5
       rounds. So P(Over k.5) >= P(distance) for every k. Above, Over 3.5 is
       0.695 against a distance price of 0.720 -- impossible.

    2. THE IMPLIED DISTRIBUTION MUST NOT EXCEED 100%. Consecutive Over prices
       differ by the chance the fight ends inside that round, and those plus
       P(distance) are a partition of every outcome. Above they sum to 1.055.

    A ladder flat across every line fails a third way: it never traded, and
    mirrors whatever the distance market says. Yan Xiaonan vs Denise Gomes
    read 0.730 / 0.730 / 0.725 on three different lines the same day.

    THE WHOLE LADDER GOES, not the offending rung. These prices are one
    market's opinion of one fight; if part of it is incoherent there is no
    principled way to decide which part was the real quote. A missing price
    is honest and a wrong one is not -- the same conclusion the round-start
    split reached, for the same reason.
    """
    dist_by_fight: dict = {}
    for r in rows:
        if r.get("market") == "GoesTheDistance" and r.get("selection") == "Goes The Distance":
            try:
                dist_by_fight[r["fight_id"]] = _american_to_prob(r["odds_american"])
            except Exception:
                pass

    ladder_by_fight: dict = {}
    for r in rows:
        if r.get("market") != "TotalRounds" or not str(r.get("selection", "")).startswith("Over "):
            continue
        try:
            line = float(str(r["selection"]).split()[1])
            ladder_by_fight.setdefault(r["fight_id"], {})[line] = _american_to_prob(r["odds_american"])
        except Exception:
            continue

    drop = set()
    for fid, ladder in ladder_by_fight.items():
        lines = sorted(ladder)
        vals = [ladder[k] for k in lines]
        why = None
        if len(vals) >= 2 and max(vals) - min(vals) < ROUNDS_MIN_SPREAD:
            why = f"flat across {len(vals)} lines (spread {max(vals) - min(vals):.3f})"
        dist = dist_by_fight.get(fid)
        if why is None and dist is not None:
            below = [(k, ladder[k]) for k in lines if ladder[k] < dist - ROUNDS_DISTANCE_SLACK]
            if below:
                why = (f"Over {below[0][0]} at {below[0][1]:.3f} is under P(distance) "
                       f"{dist:.3f} -- the distance implies passing it")
            else:
                prev, total = 1.0, 0.0
                for k in lines:
                    total += prev - ladder[k]
                    prev = ladder[k]
                total += dist
                if total > ROUNDS_SUM_SLACK:
                    why = f"implied outcomes sum to {total:.3f}"
        if why:
            drop.add(fid)
            print(f"[polymarket] dropping the Total Rounds ladder for {fid}: {why}")

    if not drop:
        return rows
    return [r for r in rows if not (r.get("market") == "TotalRounds" and r.get("fight_id") in drop)]


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
    # THE GATE RUNS HERE, not inside the per-market parser: coherence is a
    # property of a fight's whole ladder, and the parser only ever sees one
    # question at a time.
    rows = _drop_incoherent_round_ladders(rows)
    print(f"[polymarket] classified {markets_seen} markets into {len(rows)} usable rows")
    if dropped_samples:
        print(f"[polymarket] sample of {len(dropped_samples)} DROPPED markets (actual raw content, not a guess):")
        for s in dropped_samples:
            print(f"  event={s['event_title']!r} | question={s['question']!r} | outcomes={s['outcomes']}")
    return rows
