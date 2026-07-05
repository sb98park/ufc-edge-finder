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

This pulls **live moneyline, method of victory, and total rounds props
directly from DraftKings** (via their unofficial JSON endpoints — same
data their own site displays, no login needed), runs every line through
the model, and flags anything where the model disagrees with the market
by more than your threshold. Falls back to The Odds API (moneyline only)
if DraftKings' endpoints are unreachable or have changed shape.

**Important honesty note about the DraftKings scraper**
(`src/draftkings_scraper.py`): DraftKings has no official public API.
This uses documented-by-the-community, unofficial endpoints their own
website calls. That means:
- It could break without warning if DraftKings changes their JSON structure
- It's a legal/ToS gray area — you're not hacking anything (no login,
  no auth bypass), but it's also not sanctioned use, so keep request
  volume low and don't build anything commercial on top of it
- I built and unit-tested the parser against DraftKings' documented JSON
  shape, but couldn't test it against their live servers (this dev
  environment has no internet access) — if it returns nothing, the
  likely fix is opening DevTools on sportsbook.draftkings.com/leagues/mma/ufc
  and checking whether the endpoint URL or JSON field names have moved

**FanDuel**: their site is much more locked down (heavy bot detection,
no clean unofficial JSON pattern like DK's), so I didn't build a scraper
for it. If you want FanDuel props specifically, the realistic options are
a paid aggregator (OpticOdds, OddsPapi, or an Apify actor) that normalizes
FanDuel + DraftKings + others into one clean feed — worth it if you want
this to be reliable long-term rather than best-effort.

## Getting the live, auto-updating GitHub Pages site

This repo also includes a **static site generator** (`generate_site.py`) and
a **GitHub Actions workflow** that runs it every 4 hours, so you get a link
you can open on your phone that keeps itself current — no server needed.

### Setup (one-time)

1. **Get a free API key** at [the-odds-api.com](https://the-odds-api.com)
   (free tier covers this fine at low request volume).
2. **Add it as a repo secret**: on GitHub, go to your repo →
   Settings → Secrets and variables → Actions → New repository secret →
   name it `ODDS_API_KEY`, paste your key.
3. **Enable GitHub Pages**: Settings → Pages → under "Build and deployment",
   set Source to "Deploy from a branch", branch `main`, folder `/docs`.
4. Push to `main`. The workflow runs automatically on push, on a 4-hour
   schedule after that, and any time you trigger it manually from the
   Actions tab.
5. Your live link will be `https://<your-username>.github.io/<repo-name>/`

### Important limitation

The Odds API currently only provides **moneyline (h2h)** odds for MMA —
no method-of-victory or round-totals markets yet. So the live auto-updating
page only shows moneyline edges. If you want prop edges for a specific
upcoming card, update `data/upcoming_props.csv` manually with lines from
your sportsbook and run the local Flask app (`python app.py`) — that part
still supports the fuller prop analysis.

## Ideas to extend

- Weight the Elo model by weight class changes, layoffs, or age curves
- Pull live odds via an odds API instead of manual CSV updates
- Add strike/grappling differential stats instead of just win/loss records
- Backtest the model against closing lines from past cards to check calibration
