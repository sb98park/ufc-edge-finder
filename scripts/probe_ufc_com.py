"""
Dump UFC.com's event-page structure so the cross-check parser can be written
against the real DOM instead of guessed class names.

WHY UFC.COM. ESPN is the pipeline's source of truth and it lags: on one card
it missed a bout added four days out, missed a one-day replacement, and
carried an outright wrong opponent -- three hand corrections on a single
event. UFC.com is the promotion's own listing, is server-rendered Drupal with
no bot challenge (unlike ufcstats, which now serves a JavaScript interstitial
to any non-browser client), and had all three changes correctly.

It also carries more than the fights. An event page marks bouts as e.g.
"Welterweight Title Bout" vs "Middleweight Bout", and groups them under Main
Card / Prelims / Early Prelims headings -- so card position and title status
are both derivable, and is_title_fight need not stay a manual flag.

THIS IS A PROBE, NOT THE PARSER. Writing selectors from a rendered-to-markdown
view means guessing at class names, and guessing has already cost this project
two wrong conclusions about ESPN and a scraper aimed at a JS challenge page.
Print the structure, then write the parser against what is actually there.

Usage:
    python3 scripts/probe_ufc_com.py                      # upcoming events index
    python3 scripts/probe_ufc_com.py ufc-330              # one event page
"""

import os
import re
import sys

import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.results_fetcher import BASE_HEADERS, REQUEST_TIMEOUT  # noqa: E402

EVENTS_URL = "https://www.ufc.com/events"
EVENT_URL = "https://www.ufc.com/event/{}"


def get(url: str):
    print(f"\nGET {url}")
    try:
        r = requests.get(url, headers=BASE_HEADERS, timeout=REQUEST_TIMEOUT)
    except requests.RequestException as e:
        print(f"  request failed: {e}")
        return None
    print(f"  HTTP {r.status_code}, {len(r.text)} chars")
    if r.status_code != 200:
        return None
    if "checking your browser" in r.text.lower():
        print("  BOT CHALLENGE -- same wall ufcstats puts up. Stop here.")
        return None
    return r.text


def probe_index(html: str):
    slugs = sorted(set(re.findall(r'/event/([a-z0-9\-]+)', html)))
    print(f"\n  event slugs found: {len(slugs)}")
    for s in slugs[:12]:
        print(f"    {s}")


def probe_event(html: str):
    # Class names are what the parser will key on, so show which ones actually
    # wrap the fight listings rather than assuming the usual Drupal naming.
    classes = re.findall(r'class="([^"]*(?:fight|bout|listing|card)[^"]*)"', html, re.I)
    seen, ordered = set(), []
    for c in classes:
        for token in c.split():
            if token not in seen and re.search(r'fight|bout|listing|card', token, re.I):
                seen.add(token)
                ordered.append(token)
    print(f"\n  candidate classes ({len(ordered)}):")
    for c in ordered[:40]:
        print(f"    {c}")

    print("\n  athlete links (in document order, first 20):")
    for slug in re.findall(r'/athlete/([a-z0-9\-]+)', html)[:20]:
        print(f"    {slug}")

    print("\n  bout labels found:")
    for label in sorted(set(re.findall(r'>([^<>]{0,40}?(?:Title )?Bout)<', html))):
        print(f"    {label.strip()!r}")

    print("\n  segment headings:")
    for seg in sorted(set(re.findall(r'>(Main Card|Prelims|Early Prelims)<', html))):
        print(f"    {seg!r}")

    # One raw block, so the parser can be written against real nesting.
    m = re.search(r'(<div[^>]*class="[^"]*fight[^"]*"[^>]*>.{0,2500})', html, re.I | re.S)
    if m:
        print("\n  --- first fight-ish block (raw, truncated) ---")
        print(re.sub(r'\s+', ' ', m.group(1))[:1600])


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    if args:
        html = get(EVENT_URL.format(args[0]))
        if html:
            probe_event(html)
    else:
        html = get(EVENTS_URL)
        if html:
            probe_index(html)
    print("\nPaste this back -- the class names and nesting are what the parser needs.")


if __name__ == "__main__":
    main()
