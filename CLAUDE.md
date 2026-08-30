# Working on this repo

A UFC betting analytics site. Python builds a static site; CI republishes it every
~5 minutes. Real money is staked against what it publishes, so most of the rules
below exist because breaking them silently produced a wrong number rather than an
error.

Everything here is the non-obvious part. Code structure is discoverable; this is not.

---

## 1. The published record is never restated

`data/predictions_log.csv` and `data/plays_ledger.csv` are the model's public
track record. The past is not reopened, recomputed, or backfilled — not to correct
it, not to improve it.

**Before and after any change, confirm your edit did not move it:**

```bash
python3 -c "
from src.track_record import compute_track_record
r=compute_track_record(); us=r['units_stats']
print(us['total_units'], us['by_tier']['Lock of the Week']['count'], f\"{r['correct']}/{r['total']}\")"
```

Capture it first, compare after. The numbers move legitimately when a card grades,
so the invariant is *"my change didn't move it"*, not any fixed value. At the time
of writing: `64.87 9 72/102`.

If a genuine correction is needed, that is the owner's call, made explicitly, and
recorded in the commit message. It is never a side effect.

**Set-once fields.** `plays_ledger` may only ever change `last_seen`,
`closing_odds`, `result`, `units_result`, `graded_at`, `void_reason` — that list
lives in `scripts/check_plays_ledger.py:MUTABLE` and the gate enforces it.
`predictions_log` freezes `pick_odds`, `pick_fair_prob`, `opponent_odds`,
`pick_confidence_label` at publication.

**Write through the module, never a bare `csv.DictWriter`.** `plays_ledger._serialise`
and `write_graded` exist because `load()` coerces types; bypassing them wrote `'False'`
where `'0'` was expected and blocked CI for an hour.

---

## 2. Verify before every commit

Twelve steps, and the one most likely to be skipped is the last:

```bash
python3 generate_site.py                                   # must succeed
for g in check_free_build check_stake_schedule check_plays_ledger; do
  python3 scripts/$g.py || echo "GATE FAILED: $g"; done
for t in tests/test_*.py; do python3 "$t" || echo "FAILED: $t"; done   # 7 files
python3 scripts/lint_site.py                               # slow (~2 min); 0 failures, 3 known warnings
# then the record check from section 1
```

Tests are run as **plain scripts**, not pytest — that is how CI runs them.

The three gates run **after** the build and **before** the commit step in
`.github/workflows/refresh.yml`. A gate exiting non-zero freezes the entire site
on stale data. This has happened on a fight night. Never add a gate that can fail
for a number that legitimately moved.

---

## 3. Data commits race CI

CI rewrites `data/` every ~5 minutes. A hand edit committed and then rebased onto a
concurrent auto-refresh has been **silently lost three times** — once with git
reporting *"patch contents already upstream"* while the value at HEAD was still the
old one.

```bash
git checkout -- data/            # discard local build churn first
git add <source files>; git commit
git pull --rebase origin main
# re-apply the data edit ON TOP OF THE CURRENT FILE, not from a stale copy
git add <the one data file>; git commit
git push origin main
git show origin/main:<path> | ...   # VERIFY HERE
```

**Verify with `git show HEAD:<path>` or `git show origin/main:<path>` — never
`git status` or the working tree.** That check is the only reason today's loss was
caught. Keep a data commit to the single field you intended to change; local builds
append prob-history churn to unrelated rows.

---

## 4. Conventions that corrupt silently

- **The clock runs two ways.** ESPN `displayClock` counts **down** (time remaining);
  `fight_results.csv` stores **elapsed**. Both are `mm:ss` and neither announces which.
  Getting it backwards inverts every Over/Under rounds bet. Decisions are stored `5:00`.
- **pandas NaN is truthy.** `x or {}` does not catch it and `len(NaN)` raises. Use
  `isinstance(x, dict)`. This shipped a card with no parlay at all, swallowed by a
  catch-all.
- **Fight identity is `frozenset({a, b})`**, order-insensitive, because cards get
  re-scraped with the corners swapped. It carries **no event**, so a rematch collides
  with the first meeting — disambiguate on event only where a pair is recorded twice.
- **Names differ between sources.** ESPN said `Ce Liu`; the card said `Liu Ce`. Exact
  matching cost a whole card's live mode. There are already ~12 name-folding helpers;
  do not add a thirteenth.
- **`bout_order`** on a card row is the chronology. Do not re-derive fight order from
  CSV row position.

---

## 5. Two clocks decide which card is shown

- **Card screen** — holds the concluded card through Sunday, hands over Monday
  (`promote_card_if_stale`, `days_since >= 2`).
- **Everything forward-looking** (Reads, Locks, Parlay, Plays, Fight Facts) — moves
  when `card_is_over`: results all in **OR** `days_since >= 2` as a backstop.

Neither condition is safe alone: results can genuinely never confirm, and the
calendar alone costs a day and a half of content. Every section names its own card
in its heading; keep it that way.

---

## 6. Frontend facts

No `package.json`, no bundler, no React, **zero external `<script src>`**. The app
ships as one self-contained ~5.8MB HTML file (member payload to R2, free to Pages).
Charts are built server-side in Python and injected as SVG strings.

- Colour comes from the `--chart-*` palette, defined **identically** in
  `templates/site.html` and `templates/landing.html`. Those two files are separate
  design systems that disagree about `--red` and `--muted`, so never point a shared
  chart builder at a semantic token.
- `requestAnimationFrame` does not run in a hidden tab. Anything animated must write
  its final value synchronously first, or it renders blank for anyone who opens the
  page in a background tab.
- `scripts/lint_site.py` has a `deferred-init` rule: an `init*` function doing
  document-rooted queries must be registered on `DOMContentLoaded`, not called inline.

---

## 7. Claims must survive a second look

Findings reported from this data have been confidently wrong more than once: a
"5-sigma" method bias that was an argmax artefact, a "4.2× worse than baseline"
computed from the single narrowest card of six, a reproduction showing zero effect
because the test passed non-lowercased keys.

Before reporting a finding: state the sample size, name the artefact you ruled out,
and confirm it holds on more than one card. One card is ~13 fights and almost
nothing is significant at that size.
