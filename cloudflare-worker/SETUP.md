# Combat Edge relay — setup

## Why this exists
Combat Edge blocks GitHub Actions' shared IP range specifically (confirmed:
works fine from other networks, 429s only from GH Actions runners — an
IP-reputation block, not a per-request rate limit). This Worker runs the
fetch from Cloudflare's IP range instead, which Combat Edge has no known
reason to block.

**This is a genuinely unverified attempt**, not a confirmed fix. Deploy it,
run the pipeline once, and check the log for the difference described below.

## Deploy (~5 minutes, same account you already used for the access gate)

1. Go to the Cloudflare dashboard → Workers & Pages → Create → Create Worker.
2. Give it any name (e.g. `combat-edge-relay`). Deploy the default template first — you'll overwrite the code next.
3. Click **Edit code**, delete everything, and paste in the full contents of `cloudflare-worker/combat-edge-relay.js` from this repo. Save and deploy.
4. Go to the Worker's **Settings → Variables**. Add a **Secret** (not a plain text variable) named `RELAY_TOKEN`, and set it to any long random string you make up — this is what stops anyone else from finding your Worker's URL and using it as an open proxy. Save.
5. Copy your Worker's URL from the dashboard (looks like `https://combat-edge-relay.yoursubdomain.workers.dev`).

## Connect it to the pipeline

Add two repo secrets (GitHub repo → Settings → Secrets and variables → Actions → New repository secret):

- `COMBAT_EDGE_RELAY_URL` = your Worker's URL from step 5 above
- `COMBAT_EDGE_RELAY_TOKEN` = the same random string you set as `RELAY_TOKEN` in step 4

That's it — `refresh.yml` already passes both through, and `src/fighter_backfill.py` already checks for them automatically. Nothing else to change. If you *don't* set these two secrets, everything behaves exactly as it does today — this is fully opt-in and can't break anything by existing unused.

## How to tell if it worked

Run the pipeline (next scheduled run, or trigger it manually) and check the log:

- **Before / not working**: `[fighter_backfill] combat-edge rate-limited (429) fetching directory for ...`
- **Working**: `[fighter_backfill] combat-edge: <name> not found on the ... directory page` (a real "not found" instead of a 429) or actual filled data — either means the request went through.

Run `python3 scripts/audit_fighter_data.py` a few refresh cycles later — the STUCK count in section 5 should start dropping if this is working.
