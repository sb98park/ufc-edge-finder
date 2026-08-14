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

1. Cloudflare dashboard → Workers & Pages → Create → Create Worker. Name it
   `refresh-tick`. Deploy the default template, then **Edit code**, delete
   everything, and paste in the full contents of
   `cloudflare-worker/refresh-tick.js`. Save and deploy.

2. **Settings → Variables → add two Secrets** (Secret, *not* plain-text
   variable):

   - `GITHUB_TOKEN` — a fine-grained GitHub PAT (next step)
   - `TICK_TOKEN` — any long random string you invent; it protects the manual
     test endpoint so that finding the Worker's URL doesn't hand someone a
     build button

3. Create the PAT at **github.com/settings/personal-access-tokens** →
   Generate new token:

   - Repository access: **Only select repositories** → `ufc-edge-finder`
   - Permissions → Repository permissions → **Contents: Read and write**

   Contents is the permission `repository_dispatch` is filed under — *not*
   Actions, which is the one `workflow_dispatch` needs. Getting this wrong
   produces a 403, which the test in the next section will show you plainly.

   Set the longest expiry you're comfortable with. When it does expire the
   Worker fails loudly (see below) rather than going quiet.

4. **Settings → Triggers → Cron Triggers → Add Cron Trigger**, expression:

   ```
   */5 * * * *
   ```

That's the whole setup. `refresh.yml` already listens for
`repository_dispatch: types: [tick]` — nothing else to change.

## How to tell if it worked

Don't wait out the cron. Hit the test endpoint (substitute your Worker URL
and the `TICK_TOKEN` you chose):

```bash
curl -i "https://refresh-tick.<your-subdomain>.workers.dev/?token=YOUR_TICK_TOKEN"
```

- **Working**: `HTTP/1.1 200` and `OK -- repository_dispatch (event_type=tick)
  accepted, HTTP 204.` A run appears in the repo's Actions tab within seconds,
  triggered by `repository_dispatch`.
- **`HTTP/1.1 502` + `HTTP 403`**: the PAT lacks **Contents: write**, or the
  token is scoped to the wrong repository.
- **`HTTP/1.1 502` + `HTTP 401`**: the PAT is wrong, revoked, or expired.
- **`HTTP/1.1 403 Forbidden`** (from the Worker itself, no upstream detail):
  your `?token=` doesn't match `TICK_TOKEN`, or you saved it as a plain-text
  variable instead of a Secret.

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
