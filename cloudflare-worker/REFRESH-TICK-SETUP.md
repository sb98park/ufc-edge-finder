# Refresh tick — setup

## Why this exists

`refresh.yml` asks GitHub for a five-minute cron. GitHub delivers about one
tick an hour.

Measured across 2026-08-12..14, every scheduled run landed 43–137 minutes
apart — roughly one tick in twelve. GitHub documents `schedule` as
best-effort and drops ticks under load, so this is their scheduler working as
designed, not a misconfiguration in this repo.

The real cost isn't a slightly stale site. `scripts/should_refresh.py` exists
to *raise* the cadence to every 5 minutes inside 12h of a card, when late
money moves lines. At one tick an hour that tier is unreachable: by the time
the gate is asked, the last build is always older than the interval, so it
answers BUILD every single time. **The throttle has been a no-op for days and
the fight-night cadence has never once fired.**

This Worker fires the tick from Cloudflare instead, every 5 minutes,
unconditionally. It holds no cadence logic on purpose — `should_refresh.py`
stays the sole decision-maker, and the Worker's only job is guaranteeing the
gate gets *asked* on time.

## Deploy (~10 minutes, same account as the other two Workers)

1. Cloudflare dashboard → Workers & Pages → **Create**.

   The create screen offers several paths. Pick **Start with Hello World!**
   — *not* "Connect GitHub", tempting as that looks given the code lives in
   a GitHub repo. Connecting the repo sets up build-and-deploy-from-git,
   which expects a Workers project layout (`wrangler.jsonc`, a build step).
   `ufc-edge-finder` is a Python project with a few loose `.js` files in a
   folder, so that build fails. The other two Workers here are
   dashboard-managed; keep this one consistent.

   Name it `refresh-tick`, deploy the Hello World template as-is, then
   **Edit code**, select all, and paste in the full contents of
   `cloudflare-worker/refresh-tick.js`. Deploy. The file already uses the
   modern `export default {}` module format, so it drops into the scaffold
   with no adaptation.

2. **Settings → Variables and Secrets** — add two, both with type **Secret**,
   not Text:

   - `GITHUB_TOKEN` — a fine-grained GitHub PAT (next step)
   - `TICK_TOKEN` — protects the manual test endpoint so that finding the
     Worker's URL doesn't hand someone a build button

   Generate `TICK_TOKEN` with `openssl rand -hex 32 | pbcopy` rather than
   inventing one. A handmade token containing `&`, `#`, `+` or a space
   breaks when passed in a URL query string — `#` truncates the URL, `&`
   starts a new parameter, `+` decodes as a space — so the Worker receives a
   mangled value and returns 403 while everything else is perfectly fine.
   Hex is URL-safe and avoids the whole class of problem. Piping to `pbcopy`
   keeps it off your screen and out of your shell history.

   Names are read as `env.GITHUB_TOKEN` / `env.TICK_TOKEN` and are
   case-sensitive; a typo reads as `undefined`, which the Worker reports as
   a missing secret.

3. Create the PAT at **github.com/settings/personal-access-tokens** →
   Generate new token:

   - Repository access: **Only select repositories** → `ufc-edge-finder`
   - Permissions → Repository permissions → **Contents: Read and write**

   Contents is the permission `repository_dispatch` is filed under — *not*
   Actions, which is the one `workflow_dispatch` needs. Getting this wrong
   produces a 403. If you set only one permission, set this one.

   **Metadata: Read-only** gets added automatically and greyed out. That's a
   mandatory dependency of Contents, not a mistake.

   Set the longest expiry you're comfortable with. When it does expire the
   Worker fails loudly (see below) rather than going quiet.

4. **Settings → Triggers** (Cloudflare sometimes labels this panel *Trigger
   Events*) → **Cron Triggers → Add**, expression:

   ```
   */5 * * * *
   ```

   You want a *Cron* trigger, not a Route. Cloudflare renames these panels
   fairly often — go by the landmarks (a **Secret**-typed variable, a
   **Cron** trigger) rather than the exact labels above.

That's the whole setup. `refresh.yml` already listens for
`repository_dispatch: types: [tick]` — nothing else to change.

## How to tell if it worked

**Check GitHub, not the Worker.** The authoritative signal is whether
`repository_dispatch` runs are appearing about 5 minutes apart:

```bash
curl -s "https://api.github.com/repos/sb98park/ufc-edge-finder/actions/workflows/refresh.yml/runs?per_page=15" \
  | python3 -c "import json,sys; [print(r['created_at'], r['event'], r['conclusion']) for r in json.load(sys.stdin)['workflow_runs']]"
```

(Or just open the repo's Actions tab and read the trigger column.)

If those runs are landing on a 5-minute cadence, **everything works** — the
Cron Trigger fires, the PAT is valid, and GitHub accepted the dispatch. There
is nothing else to check.

Measured on first deploy, 2026-08-14: `14:16:01`, `14:21:03`, `14:26:01` —
5m02s and 4m58s apart, against the 43–137 minute gaps GitHub's own scheduler
had been delivering.

If no `repository_dispatch` runs appear at all, check the Worker's **Cron
Events** log. Failures throw with the reason attached (see Failure behavior
below).

### The manual test endpoint — optional, and a known foot-gun

The Worker also exposes a token-protected trigger so you can test without
waiting out a cron window:

```bash
read -rsp "TICK_TOKEN: " TT && echo && curl -i "https://refresh-tick.<your-subdomain>.workers.dev/?token=$TT"
```

**A 403 here tells you nothing about whether the pipeline works.**
`TICK_TOKEN` guards only this endpoint; the production path is Cron Trigger →
`scheduled()` → dispatch and never reads it. During the first deploy several
rounds were spent debugging a 403 here while the real pipeline was already
running correctly. Check the Actions tab *first*.

Decoding the responses:

- **`200`** + `OK -- repository_dispatch (event_type=tick) accepted, HTTP 204.`
  — working end to end.
- **`502` + `HTTP 403`** — the PAT lacks **Contents: write**, or is scoped to
  the wrong repository.
- **`502` + `HTTP 401`** — the PAT is wrong, revoked, or expired.
- **`403 Forbidden`** from the Worker itself, no upstream detail — your
  `?token=` doesn't match `TICK_TOKEN`. Cosmetic; see the warning above.

Then confirm the *throttle* is doing its job, which is the whole point. In the
Actions log of a `repository_dispatch` run, the **Decide whether this run
needs to build** step prints one of:

```
[cadence] no card close (9999h) -- last build 7m ago, interval 30m -- skipping
[cadence] within 12h of a card (4h) -- last build 6m ago, interval 5m -- BUILDING
```

On a quiet day you should see roughly 11 skips for every build. If every run
says BUILDING, the ticks aren't arriving any faster than before — check the
Worker's **Cron Events** log.

## Failure behavior

A failed tick throws, which marks the invocation failed and surfaces it with
the error message in the Worker's Cron Events log. This is deliberate: the
expensive failure mode is a silently expired PAT, where dispatches quietly
stop and GitHub's Actions tab looks perfectly healthy — because from GitHub's
side nothing *is* wrong, it simply stopped being asked. That's the same
invisible starvation that cost 12 hours of stale odds two days before a card.

The `schedule:` trigger in `refresh.yml` is deliberately kept as a backstop.
If this Worker is down, misconfigured, or its PAT expires, the site falls back
to GitHub's ~1 tick/hour rather than stopping entirely.

## Notes

- `repository_dispatch` always runs the workflow from the **default branch**
  (`main`). It can't be pointed at a feature branch.
- 288 invocations/day is far inside the Workers free tier.
- Most ticks exit at the gate in ~90s without calling the odds API,
  committing, or deploying. Only real builds cost a Pages deployment.
