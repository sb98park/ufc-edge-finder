"""
Pulls live MMA odds (moneyline + method of victory + round totals props)
from DraftKings' sportsbook. This uses DraftKings' unofficial, undocumented
JSON endpoints -- the same ones their own website calls. No login/API key
needed, but also no guarantees: DraftKings can change this structure at
any time without notice, and technically their Terms of Service don't
sanction third-party use of these endpoints (they just don't lock them
down). Use at your own risk, keep request volume low, and don't rely on
this for anything commercial.

If this breaks: open sportsbook.draftkings.com/leagues/mma/ufc in a
browser, open dev tools -> Network tab -> filter for "eventgroups", and
see what URL/shape it's actually calling now.
"""

import re
import time

import requests

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://sportsbook.draftkings.com/leagues/mma/ufc",
}
LEAGUES_URL = "https://sportsbook.draftkings.com/sites/US-SB/api/v3/leagues?format=json"
EVENTGROUP_URL = "https://sportsbook.draftkings.com/sites/US-SB/api/v5/eventgroups/{event_group_id}?format=json"

REQUEST_DELAY_SECONDS = 1.5


def find_mma_event_group_id() -> str | None:
    """Looks up the current eventGroupId DraftKings uses for MMA/UFC."""
    resp = requests.get(LEAGUES_URL, headers=HEADERS, timeout=15)
    if resp.status_code != 200:
        print(f"[draftkings] leagues request returned status {resp.status_code}, body preview: {resp.text[:200]!r}")
    resp.raise_for_status()
    data = resp.json()

    for league in data.get("leagues", []):
        name = league.get("name", "")
        if re.search(r"\bmma\b|\bufc\b", name, re.IGNORECASE):
            return str(league.get("eventGroupId"))
    return None


def fetch_mma_eventgroup(event_group_id: str | None = None) -> dict:
    event_group_id = event_group_id or find_mma_event_group_id()
    if not event_group_id:
        raise RuntimeError(
            "Couldn't find a DraftKings eventGroupId for MMA. "
            "DraftKings may have changed their leagues endpoint -- "
            "check sportsbook.draftkings.com/leagues/mma/ufc in dev tools."
        )

    resp = requests.get(
        EVENTGROUP_URL.format(event_group_id=event_group_id),
        headers=HEADERS, timeout=15,
    )
    resp.raise_for_status()
    time.sleep(REQUEST_DELAY_SECONDS)
    return resp.json()


CATEGORY_MAP = {
    "moneyline": "Moneyline",
    "fight result": "Moneyline",
    "method of victory": "Method",
    "how will the fight end": "Method",
    # "Wins by finish" reuses the Method pipeline below -- structurally it's
    # the same shape as KO/TKO, SUB, DEC (a fighter + an outcome), just a
    # 4th outcome value ("FINISH" = KO/TKO or SUB combined). DK's exact
    # title for this market varies; these are reasonable guesses based on
    # common industry naming, not a verified live scrape. See the
    # DIAGNOSTIC print below for what to check if a real run reports zero
    # Finish props.
    "wins inside the distance": "Method",
    "wins by finish": "Method",
    "inside the distance": "Method",
    "total rounds": "TotalRounds",
    "round total": "TotalRounds",
    "go the distance": "GoesTheDistance",
    "fight to a decision": "GoesTheDistance",
    # Round betting: fighter-specific "wins in round N". Same caveat as
    # above -- guessed keywords, needs live confirmation.
    "round betting": "RoundBetting",
    "method of victory & round": "RoundBetting",
    "method & round": "RoundBetting",
    "which round": "RoundBetting",
}

METHOD_LABEL_MAP = {
    "ko/tko": "KO/TKO",
    "knockout": "KO/TKO",
    "submission": "SUB",
    "decision": "DEC",
    "points": "DEC",
    # Finish-market outcomes: DK labels this market's two sides something
    # like "<Fighter> Inside the Distance" / "<Fighter> by Decision" -- the
    # "decision" side already matches the DEC key above, so it's correctly
    # excluded here (that side is redundant with the existing DEC prop and
    # not something compute_method_edges needs to see twice).
    "inside the distance": "FINISH",
    "wins by finish": "FINISH",
    "by finish": "FINISH",
}

# Bounded, deduped diagnostic: logs any offerCategory name that didn't match
# CATEGORY_MAP, once per unique name per run. If a live scrape reports zero
# Method:FINISH or RoundBetting props, check this log for DK's actual
# category title text and add it to CATEGORY_MAP above -- same
# diagnostic-first approach used for the ESPN eventLog schema discovery
# (never guess at an unseen structure twice when a real log can confirm it).
_unmatched_categories_logged = set()


def _normalize_method_label(label: str) -> str | None:
    label_lower = label.lower()
    for key, value in METHOD_LABEL_MAP.items():
        if key in label_lower:
            return value
    return None


def parse_eventgroup(eventgroup_json: dict) -> list[dict]:
    """
    Walks DraftKings' eventGroup -> events / offerCategories structure and
    flattens it into rows matching the same schema as data/upcoming_props.csv:
    fight_id, fighter_a, fighter_b, market, selection, selection_method, odds_american
    """
    rows = []
    event_group = eventgroup_json.get("eventGroup", {})
    events = {e["eventId"]: e for e in event_group.get("events", [])}

    for category in event_group.get("offerCategories", []):
        category_name = category.get("name", "")
        # Pick the LONGEST matching keyword, not the first one found in dict
        # order -- "method of victory" is a substring of "method of victory
        # & round", so first-match-wins would always resolve the more
        # specific Round Betting category to the shorter Method keyword
        # instead (a real bug caught in testing). Longest-match is specific-
        # match here since every CATEGORY_MAP keyword is a literal phrase,
        # not a pattern with independent wildcards.
        matches = [(keyword, mapped) for keyword, mapped in CATEGORY_MAP.items() if keyword in category_name.lower()]
        market_key = max(matches, key=lambda kv: len(kv[0]))[1] if matches else None
        if market_key is None:
            if category_name and category_name not in _unmatched_categories_logged:
                _unmatched_categories_logged.add(category_name)
                print(f"[draftkings_scraper] DIAGNOSTIC: uncategorized offer category "
                      f"{category_name!r} -- not in CATEGORY_MAP, skipped. If this is actually "
                      f"Wins-by-Finish or Round Betting, add its real title text to CATEGORY_MAP.")
            continue  # a category we're not modeling yet (futures, parlays, etc.)

        for subcat_desc in category.get("offerSubcategoryDescriptors", []):
            subcategory = subcat_desc.get("offerSubcategory") or {}
            for offer_group in subcategory.get("offers", []):
                for offer in offer_group:
                    event_id = offer.get("eventId")
                    event = events.get(event_id, {})
                    fighter_a = event.get("team1Name") or event.get("participants", [{}])[0].get("name")
                    fighter_b = event.get("team2Name") or (
                        event.get("participants", [{}, {}])[1].get("name")
                        if len(event.get("participants", [])) > 1 else None
                    )
                    start_date = event.get("startDate")  # ISO date/time if DK provides it
                    weight_class = event.get("name") or ""  # DK sometimes includes weight class in the event name

                    for outcome in offer.get("outcomes", []):
                        label = outcome.get("label", "")
                        american_odds = outcome.get("oddsAmerican")
                        if american_odds is None:
                            continue

                        selection_method = ""
                        selection = label

                        try:
                            american_odds = float(american_odds)
                        except (TypeError, ValueError):
                            continue

                        if market_key == "Method":
                            method = _normalize_method_label(label)
                            if method is None:
                                continue
                            selection_method = method
                            # DK's exact label format varies: KO/SUB/DEC outcomes
                            # are usually "<Fighter> by <Method>", but the
                            # Finish-market outcome is more often "<Fighter>
                            # Inside the Distance" (no "by"). Try both splits
                            # rather than assuming one -- falls back to the
                            # whole label if neither separator is present,
                            # same as the pre-existing behavior.
                            if " by " in label:
                                selection = label.split(" by ")[0].strip()
                            elif method == "FINISH":
                                for sep in (" Inside the Distance", " Wins by Finish", " Inside Distance"):
                                    if sep.lower() in label.lower():
                                        selection = label[:label.lower().index(sep.lower())].strip()
                                        break
                        elif market_key == "TotalRounds":
                            # Capture the specific line (e.g. "Over 2.5") so multiple
                            # round lines on the same fight (1.5, 2.5, 3.5) don't collide
                            line_match = re.search(r"(\d+\.?\d*)", label)
                            line = line_match.group(1) if line_match else ""
                            side = "Over" if "over" in label.lower() else "Under" if "under" in label.lower() else label
                            selection = f"{side} {line}".strip()
                            selection_method = line  # stash the numeric line for grouping downstream
                        elif market_key == "GoesTheDistance":
                            selection = "Goes The Distance" if "yes" in label.lower() or "distance" in label.lower() else "Ends In Finish"
                        elif market_key == "RoundBetting":
                            # Expected label shape: "<Fighter> - Round N" or
                            # "<Fighter> Round N" (fighter-specific, unlike
                            # TotalRounds which is a fight-level over/under).
                            # Guessed separators, same caveat as the Finish
                            # market above -- verify against a real DIAGNOSTIC
                            # log if this comes back empty on a live scrape.
                            round_match = re.search(r"round\s*(\d+)", label.lower())
                            if round_match is None:
                                continue  # can't identify a round number -- skip rather than guess
                            round_num = round_match.group(1)
                            fighter_part = re.split(r"\s*-\s*round|\s+round", label, flags=re.IGNORECASE)[0].strip()
                            selection = f"{fighter_part} Round {round_num}"
                            selection_method = round_num  # numeric round, mirrors TotalRounds' line stash

                        rows.append({
                            "fight_id": event_id,
                            "fighter_a": fighter_a,
                            "fighter_b": fighter_b,
                            "event_name": "",  # DK's MMA feed usually isn't grouped by card name; derived by date downstream
                            "start_date": start_date,
                            "weight_class": weight_class,
                            "card_position": "",
                            "market": market_key,
                            "selection": selection,
                            "selection_method": selection_method,
                            "odds_american": american_odds,
                        })

    return rows


def fetch_draftkings_mma_props() -> list[dict]:
    """Convenience wrapper: find MMA, fetch it, parse it."""
    raw = fetch_mma_eventgroup()
    return parse_eventgroup(raw)


if __name__ == "__main__":
    import json
    props = fetch_draftkings_mma_props()
    print(json.dumps(props, indent=2))
