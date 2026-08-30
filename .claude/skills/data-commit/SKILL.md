---
name: data-commit
description: >
  Commit a change to a file under data/ in this repo safely. Use whenever editing
  data/plays_ledger.csv, predictions_log.csv, fight_cards.csv, bankroll.json or any
  other tracked data file. CI rewrites these every ~5 minutes, so the ordinary
  commit-then-rebase flow has silently lost changes three times.
---

# Committing a data file

CI auto-commits `data/` every ~5 minutes. A hand edit committed and then rebased onto
a concurrent auto-refresh **has been silently lost three times** — including one where
git reported `patch contents already upstream` while the value at HEAD was plainly
still the old one. Verifying against the working tree would have missed all three.

## The sequence

### 1. Discard build churn first

Running `generate_site.py` appends prob-history and rewrites timestamps on rows you
did not touch. Those must not ride along in a commit that restates a record.

```bash
git checkout -- data/
git status --short
```

### 2. Commit source changes separately

Source and data get separate commits. Source rarely conflicts; data always might.

```bash
git add <source files>
git commit
```

### 3. Rebase BEFORE writing the data edit

```bash
git pull --rebase origin main
```

### 4. Apply the edit on top of the *current* file

Not from a copy you made earlier, and not by replaying an old diff. Read the file as
it exists now, change the one field, write it back through the module's own writer:

- `data/plays_ledger.csv` → `src.plays_ledger.write_graded` (never a bare `DictWriter`;
  `load()` coerces types and bypassing `_serialise` wrote `'False'` for `'0'` and
  blocked CI for an hour)
- `data/predictions_log.csv` → `csv.DictWriter(f, fieldnames=track_record.FIELDNAMES)`,
  after asserting the header still matches `FIELDNAMES`

### 5. Prove the diff is only what you intended

```bash
python3 - <<'PY'
import csv, io, subprocess
old=list(csv.DictReader(io.StringIO(subprocess.run(
    ["git","show","HEAD:data/<FILE>"],capture_output=True,text=True).stdout)))
new=list(csv.DictReader(open("data/<FILE>",encoding="utf-8")))
d=[(i,a.get("favorite") or a.get("selection"),k,a.get(k),b.get(k))
   for i,(a,b) in enumerate(zip(old,new)) for k in a if (a.get(k) or "")!=(b.get(k) or "")]
print(f"{len(d)} field-level difference(s)")
for x in d: print("  idx=%s %s  %s: %r -> %r" % x)
PY
```

Expect the exact count you intended. Anything else is churn — go back to step 1.

### 6. Commit, push, then VERIFY AT ORIGIN

```bash
git add data/<FILE> && git commit
git push origin main
git fetch -q origin
git show origin/main:data/<FILE> | grep <the value you changed>
```

**This last step is not optional.** `git status`, `git diff` and the working tree all
looked correct in every one of the three losses. Only reading the committed object
caught it.

If the change is missing at origin: the rebase dropped your commit. Re-apply it on top
of the current file (step 4) and push again — do not `git cherry-pick` the lost commit,
because its parent no longer matches.

## Before you touch the file at all

Read CLAUDE.md §1. Most data edits are not allowed: the published record is never
restated, and `plays_ledger` may only change six fields. If the edit corrects the
public record, that is the owner's explicit call and belongs in the commit message.
