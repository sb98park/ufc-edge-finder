# UFC Edge Finder

A tool that estimates fight outcome probabilities from historical data and
compares them against sportsbook lines to surface where the two disagree
most — a starting point for finding value, not a black-box "sure thing"
generator.

> ⚠️ **For analysis and education, not betting advice.** The model is only
> as good as the data you feed it, and it can't see injuries, weight cuts,
> camp changes, or recent form the way you can. Sports betting carries real
> financial risk — bet only what you can afford to lose, and if it ever
> stops being fun, that's worth paying attention to. If you're in the US,
> the National Council on Problem Gambling helpline is 1-800-522-4700.

## How it works

1. **Elo ratings** are built from `data/fight_history.csv`. Finishes
   (KO/TKO, submissions) move ratings more than decisions since they're a
   more decisive signal.
2. **Moneyline edges**: your model's implied win probability vs. the
   sportsbook's *vig-removed* fair probability.
3. **Method / total-rounds props**: priced off each fighter's historical
   finish rate (simplified — a good first thing to improve with better
   data).
4. Everything is ranked by **Edge %** (model probability minus the book's
   fair probability) plus a suggested half-Kelly stake size, purely as a
   reference for bankroll math — not a recommendation.

## Setup

```bash
pip install -r requirements.txt
python app.py
```

Then open **http://127.0.0.1:5000**.

## Using real data

The `data/` CSVs ship with small sample datasets so the app runs out of
the box. To use it for real:

- Update `data/fighters.csv` and `data/fight_history.csv` with real
  records (see `src/scraper.py` for a starting point that pulls public
  stats from ufcstats.com — run it yourself, it's not wired into the app
  automatically since this needs a live internet connection).
- Update `data/upcoming_props.csv` with the current lines from your
  sportsbook of choice for the card you're analyzing.

### CSV formats

**fighters.csv**: `name, weight_class, height_in, reach_in, stance, wins, losses, ko_wins, sub_wins, dec_wins`

**fight_history.csv**: `date, fighter_a, fighter_b, winner, method` (method is `KO/TKO`, `SUB`, or `DEC`)

**upcoming_props.csv**: `fight_id, fighter_a, fighter_b, market, selection, selection_method, odds_american`
Markets supported: `Moneyline`, `Method`, `TotalRounds`.

## Project structure

```
ufc-edge-finder/
├── app.py                  # Flask app / entry point
├── data/                   # CSV inputs (sample data included)
├── src/
│   ├── elo.py              # Elo rating engine
│   ├── odds_utils.py       # American odds, vig removal, Kelly sizing
│   ├── edge_finder.py      # Combines model + odds into ranked edges
│   └── scraper.py          # Optional: pull public fighter data
└── templates/index.html    # Dashboard
```

## Finding mispriced props (the main tool)

```bash
python find_ev_bets.py                  # default: flag anything with |edge| >= 5%
python find_ev_bets.py --min-edge 8     # stricter threshold
```

This pulls live odds through a three-way fallback chain, runs every line
through the model, and flags anything where the model disagrees with the
market by more than your threshold.

### Data source priority: Polymarket → DraftKings → The Odds API

1. **Polymarket (`src/polymarket_source.py`) — primary source.** Polymarket's
   Gamma API (`gamma-api.polymarket.com`) is fully public, no auth required,
   and is Polymarket's actual documented API (unlike DraftKings below), so
   it should be far more stable long-term. It's a peer-to-peer prediction
   market, not a sportsbook, so prices carry no bookmaker vig. Prices come
   back as 0–1 probabilities and are converted to familiar American odds
   format (+150, -180) before anything touches the site — you'll never see
   a raw percentage where an odds line should be.
2. **DraftKings (`src/draftkings_scraper.py`) — fallback.** DraftKings has
   no official public API at all. This uses unofficial, reverse-engineered
   endpoints their own website calls. Real risks: it can break without
   warning if DraftKings changes their JSON structure, and — worth knowing
   specifically because this runs on GitHub Actions — cloud/CI IP ranges
   are exactly the kind of traffic sites like this tend to block or
   rate-limit. If Polymarket doesn't have a market for something DraftKings
   does, this is the fallback.
3. **The Odds API (`src/live_odds.py`) — last resort.** Moneyline only for
   MMA on their platform, used only if both of the above fail.

**FanDuel**: not scraped directly — their site has much heavier bot
detection with no clean unofficial pattern like DraftKings has. If you
want FanDuel specifically, a paid aggregator (OpticOdds, OddsPapi) is the
realistic path.

### Honest limitations of the Polymarket integration

Polymarket's API is properly documented, which makes this meaningfully
more trustworthy than the DraftKings scraper — but a few things are still
best-effort:
- **Market discovery** is done by pulling active events sorted by volume
  and filtering client-side for "UFC" in the title (Gamma's `/events`
  endpoint has no free-text search), so a very low-volume or oddly-titled
  event could be missed.
- **Prop questions** (method of victory, rounds, goes-the-distance) are
  classified by keyword-matching the market's question text, and the
  specific fighter/opponent is pulled from the event title rather than the
  question itself, since Yes/No prop questions don't always name the
  opponent. This is a reasonable approach but hasn't been verified against
  Polymarket's live servers from this dev environment (no internet access
  here) — if props come back empty or misattributed, that's the first
  place to check.
- Fighter names are matched to your tracked card with accent/punctuation
  normalization (e.g. Polymarket's "Benoît Saint Denis" correctly matches
  your data's "Benoit Saint-Denis"), so minor spelling differences between
  sources shouldn't cause a real fight to silently disappear into
  "unmatched."

## Getting the live, auto-updating GitHub Pages site

This repo also includes a **static site generator** (`generate_site.py`) and
a **GitHub Actions workflow** that runs it every 4 hours, so you get a link
you can open on your phone that keeps itself current — no server needed.

### Setup (one-time)

1. **Nothing required for odds data** — Polymarket (primary) and
   DraftKings (fallback) both need zero setup or API keys.
2. **Optional**: get a free key at [the-odds-api.com](https://the-odds-api.com)
   as a last-resort fallback if both of the above ever fail, and add it as
   a repo secret (Settings → Secrets and variables → Actions → New
   repository secret → name `ODDS_API_KEY`). Skippable — the site works
   without it as long as Polymarket or DraftKings are reachable.
3. **Enable GitHub Pages**: Settings → Pages → under "Build and deployment",
   set Source to "Deploy from a branch", branch `main`, folder `/docs`.
4. Push to `main`. The workflow runs automatically on push, on a schedule
   after that, and any time you trigger it manually from the Actions tab.
5. Your live link will be `https://<your-username>.github.io/<repo-name>/`

### Important limitation

Coverage depends on which source actually responds: Polymarket has real
prop markets (method, rounds, goes-the-distance) when they exist for a
given fight, DraftKings similarly covers moneyline + props when its
endpoints are reachable, but The Odds API (the last-resort fallback) is
moneyline-only for MMA. Check the "Odds via ___" badge at the top of the
site to see which source actually served that refresh.

## Ideas to extend

- Weight the Elo model by weight class changes, layoffs, or age curves
- Pull live odds via an odds API instead of manual CSV updates
- Add strike/grappling differential stats instead of just win/loss records
- Backtest the model against closing lines from past cards to check calibration
