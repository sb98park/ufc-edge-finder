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
    "User-Agent": "Mozilla/5.0 (personal research script)",
    "Accept": "application/json",
}
LEAGUES_URL = "https://sportsbook.draftkings.com/sites/US-SB/api/v3/leagues?format=json"
EVENTGROUP_URL = "https://sportsbook.draftkings.com/sites/US-SB/api/v5/eventgroups/{event_group_id}?format=json"

REQUEST_DELAY_SECONDS = 1.5


def find_mma_event_group_id() -> str | None:
    """Looks up the current eventGroupId DraftKings uses for MMA/UFC."""
    resp = requests.get(LEAGUES_URL, headers=HEADERS, timeout=15)
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
    "total rounds": "TotalRounds",
    "round total": "TotalRounds",
    "go the distance": "GoesTheDistance",
    "fight to a decision": "GoesTheDistance",
}

METHOD_LABEL_MAP = {
    "ko/tko": "KO/TKO",
    "knockout": "KO/TKO",
    "submission": "SUB",
    "decision": "DEC",
    "points": "DEC",
}


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
        market_key = None
        for keyword, mapped in CATEGORY_MAP.items():
            if keyword in category_name.lower():
                market_key = mapped
                break
        if market_key is None:
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
                            # label is usually "<Fighter> by <Method>"
                            selection = label.split(" by ")[0].strip()
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
