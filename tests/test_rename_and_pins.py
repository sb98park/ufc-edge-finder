"""
Two ledger-integrity defects the audit found, and their fixes.

1. An ESPN card rename re-froze pick_odds / pick_fair_prob /
   pick_confidence_label on 13 bouts and orphaned 16 rows, because the prior
   row was looked up by (event_name, fighter_a, fighter_b).
2. A pinned parlay was gradeable only while its card was pinned, so all 199
   slips on file across three cards carry no result at all.
"""
import csv
import json
import os
import shutil
import sys
import tempfile

sys.path.insert(0, ".")
import src.track_record as tr                                   # noqa: E402
from src import parlay_grader as pg                             # noqa: E402

ok = fail = 0


def check(label, cond):
    global ok, fail
    if cond:
        ok += 1
    else:
        fail += 1
        print(f"  FAIL: {label}")


COLS = ['event_name', 'fighter_a', 'fighter_b', 'favorite', 'favorite_prob',
        'confidence_label', 'likely_method', 'pick_odds', 'closing_odds',
        'opponent_odds', 'pick_fair_prob', 'closing_fair_prob',
        'pick_confidence_label', 'favorite_prob_history', 'last_updated',
        'is_lock_of_week', 'voided', 'pick_falsifier']


def _fight(event):
    return {'event_name': event, 'fights': [{
        'event_name': event, 'fighter_a': 'Alpha', 'fighter_b': 'Beta',
        'preview': {'favorite': 'Alpha', 'favorite_prob': 0.7,
                    'confidence_label': 'High Confidence', 'likely_method': 'Decision'},
        'edges': [
            {'market': 'Moneyline', 'fighter': 'Alpha', 'opponent': 'Beta',
             'odds_american': 261.0, 'book_fair_prob': 0.277},
            {'market': 'Moneyline', 'fighter': 'Beta', 'opponent': 'Alpha',
             'odds_american': -310.0, 'book_fair_prob': 0.723}]}]}


tmp = tempfile.mkdtemp()
log = os.path.join(tmp, 'p.csv')
base = {c: '' for c in COLS}
with open(log, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=COLS)
    w.writeheader()
    w.writerow(base | {'event_name': 'UFC Fight Night: A vs. B',
                       'fighter_a': 'Alpha', 'fighter_b': 'Beta',
                       'favorite': 'Alpha', 'favorite_prob': '0.7',
                       'confidence_label': 'Medium Confidence',
                       'pick_odds': '295.0', 'pick_fair_prob': '0.31',
                       'opponent_odds': '-340.0',
                       'pick_confidence_label': 'Medium Confidence',
                       'last_updated': '2026-07-01 10:00 AM ET'})

_real = tr.PREDICTIONS_LOG_PATH
tr.PREDICTIONS_LOG_PATH = log
try:
    tr.log_predictions([_fight('UFC Fight Night: A vs. C')], '2026-09-01 08:00 AM ET')
finally:
    tr.PREDICTIONS_LOG_PATH = _real
rows = list(csv.DictReader(open(log, newline='')))

check("a rename migrates the row instead of duplicating it", len(rows) == 1)
check("the renamed row carries the NEW event name",
      rows and rows[0]['event_name'] == 'UFC Fight Night: A vs. C')
check("pick_odds stays frozen at the first published price",
      rows and rows[0]['pick_odds'] == '295.0')
check("pick_fair_prob stays frozen", rows and rows[0]['pick_fair_prob'] == '0.31')
check("pick_confidence_label stays frozen -- it sets the stake",
      rows and rows[0]['pick_confidence_label'] == 'Medium Confidence')
shutil.rmtree(tmp)

# A genuine rematch must NOT be collapsed into the earlier meeting.
tmp = tempfile.mkdtemp()
log = os.path.join(tmp, 'p.csv')
with open(log, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=COLS)
    w.writeheader()
    w.writerow(base | {'event_name': 'UFC 300', 'fighter_a': 'Alpha', 'fighter_b': 'Beta',
                       'favorite': 'Alpha', 'favorite_prob': '0.7',
                       'confidence_label': 'Medium Confidence', 'pick_odds': '295.0',
                       'pick_confidence_label': 'Medium Confidence',
                       'last_updated': '2026-01-01 10:00 AM ET'})
tr.PREDICTIONS_LOG_PATH = log
try:
    # decided_keys marks the first meeting as graded, which is what a rematch
    # implies -- so the fallback must not fire.
    tr.log_predictions([_fight('UFC 320')], '2026-09-01 08:00 AM ET',
                       decided_keys={frozenset({'alpha', 'beta'})})
finally:
    tr.PREDICTIONS_LOG_PATH = _real
rows = list(csv.DictReader(open(log, newline='')))
check("a decided pair is never collapsed by the rename fallback", len(rows) == 1)
shutil.rmtree(tmp)

# ---------------------------------------------------------------- parlays
# SYNTHETIC, not a copy of data/parlay_ledger.jsonl. The first version read
# the live ledger and the live pin file, and started failing the moment a
# build legitimately dropped the pin -- a test that depends on mutable
# production state is not testing anything.
tmp = tempfile.mkdtemp()
led = os.path.join(tmp, 'l.jsonl')
SLIP = {"slip_id": "committed1", "event": "UFC Fight Night: Test",
        "tier": "bankroll", "combined_decimal": 4.0, "renders": 3,
        "first_seen": "2026-08-01T00:00:00+00:00",
        "last_seen": "2026-08-08T00:00:00+00:00",
        "legs": [{"fight_key": "Nobody A|Nobody B", "label": "Nobody A ML",
                  "decimal_odds": 2.0, "is_model": True,
                  "conditions": [{"kind": "winner", "value": "Nobody A"}]}]}
UNPINNED = dict(SLIP, slip_id="never_pinned")
with open(led, 'w') as fh:
    for r in (SLIP, UNPINNED):
        fh.write(json.dumps(r) + "\n")
PINS = {"UFC Fight Night: Test": {"bankroll": {
    "pinned_at": "2026-08-01T12:00:00+00:00",
    "snapshot": {"slip_id": "committed1"}}}}

pg.grade_pinned(PINS, '2026-08-09 09:00 AM ET', path=led, results_path=os.path.join(tmp, 'none.csv'))
rows = [json.loads(l) for l in open(led) if l.strip()]
stamped = [r for r in rows if r.get('pinned_at')]
check("a currently pinned slip records its commitment", len(stamped) == 1)
check("only the pinned slip is stamped", stamped and stamped[0]['slip_id'] == 'committed1')
check("the stamp is the pin's own timestamp, not now",
      stamped and stamped[0]['pinned_at'] == '2026-08-01T12:00:00+00:00')
check("nothing is voided one day after the card",
      not any(r.get('result') == 'void' for r in rows))

# The pin rotates to another card -- the slip must STILL be gradeable.
pg.grade_pinned({"Some Other Card": {}}, '2026-09-01 09:00 AM ET', path=led,
                results_path=os.path.join(tmp, 'none.csv'))
rows = [json.loads(l) for l in open(led) if l.strip()]
voided = [r for r in rows if r.get('result') == 'void']
check("a committed slip stays gradeable after the pin rotates to another card",
      len(voided) == 1)
check("the void says why", all(r.get('void_reason') for r in voided))
check("a voided slip is worth zero units",
      all(float(r.get('units_result') or 0) == 0.0 for r in voided))
check("an UNPINNED slip is never voided -- it was never committed to",
      not any(r.get('result') == 'void' and not r.get('pinned_at') for r in rows))
shutil.rmtree(tmp)

check("the parlay void window matches the plays ledger", pg.VOID_AFTER_DAYS == 2)

print(f"test_rename_and_pins: {ok} passed, {fail} failed")
sys.exit(1 if fail else 0)
