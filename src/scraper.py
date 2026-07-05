"""
Optional helper for pulling real fighter records from ufcstats.com (public
stats pages, no login/paywall). Not run automatically -- run it yourself
on your own machine to refresh data/fighters.csv and data/fight_history.csv
with real data.

Usage (from project root, on a machine with internet access):
    pip install requests beautifulsoup4
    python -m src.scraper --event-url https://www.ufcstats.com/event-details/<id>

Be a good citizen: this adds a short delay between requests and only hits
public stat pages, not anything behind a login.
"""

import argparse
import time

import requests
from bs4 import BeautifulSoup

BASE_HEADERS = {"User-Agent": "Mozilla/5.0 (personal research script)"}
REQUEST_DELAY_SECONDS = 1.5


def fetch_event_fights(event_url: str) -> list[dict]:
    resp = requests.get(event_url, headers=BASE_HEADERS, timeout=15)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    fights = []
    rows = soup.select("tr.b-fight-details__table-row")
    for row in rows:
        names = [a.get_text(strip=True) for a in row.select("a.b-link_style_black")]
        if len(names) == 2:
            fights.append({"fighter_a": names[0], "fighter_b": names[1]})

    time.sleep(REQUEST_DELAY_SECONDS)
    return fights


def fetch_fighter_stats(fighter_url: str) -> dict:
    resp = requests.get(fighter_url, headers=BASE_HEADERS, timeout=15)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    stats = {}
    for item in soup.select("li.b-list__box-list-item"):
        text = item.get_text(strip=True)
        if ":" in text:
            key, val = text.split(":", 1)
            stats[key.strip()] = val.strip()

    time.sleep(REQUEST_DELAY_SECONDS)
    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scrape public UFCStats.com pages")
    parser.add_argument("--event-url", required=True, help="ufcstats.com event-details URL")
    args = parser.parse_args()

    result = fetch_event_fights(args.event_url)
    for fight in result:
        print(fight)
