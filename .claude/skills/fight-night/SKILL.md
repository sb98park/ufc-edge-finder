---
name: fight-night
description: >
  Diagnose the live site during or right after a UFC card — wrong fight showing
  as live, results not appearing, the Record tab not updating, a stale page, or
  the site not refreshing at all. Use whenever something looks wrong while a card
  is running or on the day after. Checks causes in order of how often they are
  actually to blame.
---

# Fight-night triage

Once a week, four hours, money live. The failure is almost never where it looks.

## Check in this order. Do not skip ahead.

The order is by observed frequency, not by plausibility. Last card cost hours
because the diagnosis started at the fetcher — which was healthy the whole time —
while CI had been dead since 03:20.

### 1. Is CI alive? (this is usually it)

```bash
git fetch -q origin && python3 -c "
import subprocess,time
ts=int(subprocess.run(['git','log','-1','--format=%ct','origin/main'],capture_output=True,text=True).stdout)
age=(time.time()-ts)/60
print(f'last commit on origin/main: {age:.0f} min ago -> {\"STALLED\" if age>15 else \"alive\"}')"
```

Auto-refresh commits every ~5 minutes. **Anything over 15 minutes means CI is not
publishing.** `gh` is not installed on this machine, so use this, not `gh run list`.

If stalled, go to step 2. If alive, skip to step 3 — the data is being published, so
the problem is in what is being published, not whether.

### 2. Did a hard gate fail?

The three gates run **after** the build and **before** the commit step, so a failing
gate stops the site updating while leaving everything looking fine locally.

```bash
for g in check_free_build check_stake_schedule check_plays_ledger; do
  echo "--- $g"; python3 scripts/$g.py; done
```

`check_stake_schedule` is the one that has frozen the site before: it compares the
whole record against a frozen snapshot, and a card grading mid-run moved it. If a
gate fails for a number that legitimately moved rather than a structural problem,
that is a bug in the gate.

### 3. Are results landing?

```bash
python3 -c "
import csv
cards=[r for r in csv.DictReader(open('data/fight_cards.csv')) if str(r.get('cancelled','')).lower()!='true']
ev=cards[0]['event_name']
res=[r for r in csv.DictReader(open('data/fight_results.csv')) if r['event_name']==ev]
have={frozenset({r['fighter_a'].lower(),r['fighter_b'].lower()}) for r in res}
print(f'{len(res)} of {len(cards)} fights recorded')
for c in cards:
    k=frozenset({c['fighter_a'].lower(),c['fighter_b'].lower()})
    if k not in have: print('  MISSING:', c['fighter_a'],'vs',c['fighter_b'])"
```

A fight missing here while the browser shows its result means the **server** path
failed, not the display. The two are independent: the browser polls ESPN directly,
`results_fetcher` writes the CSV that drives Record, grading and the banner.

### 4. If one specific fight is wrong, suspect the name

ESPN said `Ce Liu`; the card said `Liu Ce`. Every live match is exact-string, so that
one bout was never fetched, its result landed out of order, and it took eight other
fights out of the live schedule with it.

```bash
python3 -c "
from src.results_fetcher import _canon" 2>/dev/null
grep -n 'fighter_a' data/fight_cards.csv | head -3
```

Compare the card spelling against what ESPN returns for the same bout. Accents and
CJK name order are the usual culprits.

### 5. If a rounds bet looks inverted, it is the clock

ESPN `displayClock` counts **down**; `fight_results.csv` stores **elapsed**. Both are
`mm:ss`. A KO at `0:34` displayed is `4:26` elapsed — 1.89 rounds, not 1.11, which
flips an Under 1.5. Decisions are stored `5:00`.

### 6. Only now suspect the browser

`pollLive` returns early when `document.visibilityState !== 'visible'`, and
`insideCardWindow` reads `schedule[0]` and `schedule[-1]` — so if the schedule is
short, the poll window is short. `results_pending` fights stay in the schedule
deliberately; they are skipped for live/next selection but must still be present.

## What NOT to do

- Do not conclude the fetcher is broken without running it. It has been blamed twice
  and was healthy both times. Run it directly and read what it returns.
- Do not edit `data/` to paper over a display bug. If the ledger is right and the page
  is wrong, fix the page.
- Do not restate the published record to make a number look right. See CLAUDE.md §1.
