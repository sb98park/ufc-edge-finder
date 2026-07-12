<div align="center">

# 🥊 Octane Alpha

### Quantitative UFC Analytics & Edge Detection

A from-scratch Elo + style-matchup prediction model, checked live against real market pricing — rebuilt automatically every 15 minutes, with zero server and zero hosting cost.

[![Live Site](https://img.shields.io/badge/live_site-octanealpha.com-d4af37?style=for-the-badge)](https://octanealpha.com)

![Python](https://img.shields.io/badge/python-3.11-3776AB?style=flat-square&logo=python&logoColor=white)
![Auto Refresh](https://img.shields.io/badge/auto--refresh-every_15_min-3ddc84?style=flat-square)
![Hosted on GitHub Pages](https://img.shields.io/badge/hosted_on-GitHub_Pages-181717?style=flat-square&logo=github)
![No Server Required](https://img.shields.io/badge/server-none_required-blue?style=flat-square)

</div>

<br>

> ⚠️ **For analysis and education, not betting advice.** The model is only as good as the data it's fed, and it can't see injuries, weight cuts, camp changes, or last-minute news the way you can. Sports betting carries real financial risk — bet only what you can afford to lose, and if it ever stops being fun, that's worth paying attention to. If you're in the US, the National Council on Problem Gambling helpline is 1-800-522-4700.

## What this is

Octane Alpha predicts UFC fight outcomes with a real, from-scratch model — Elo ratings built from full fight history, adjusted for style matchups (wrestling, striking volume, submission threat, durability, layoff, age, stance) — and checks those predictions against live market pricing to surface where the two genuinely disagree.

Every pick gets tracked against the real result afterward, in public, including the misses. There's no cherry-picking: the Track Record section shows the full history, the calibration curve, and a real units/ROI ledger priced off actual market odds at pick time, not the model's own confidence.

The whole site is static — no database, no backend, no hosting bill. A GitHub Action rebuilds it from scratch every 15 minutes.

## ✨ Features

| | |
|---|---|
| 🥊 **This Weekend** | Every fight on the current card, grouped by segment (Main Card / Prelims / Early Prelims), with live status that self-corrects off confirmed results instead of trusting a static pre-card time estimate |
| 📊 **Track Record** | Full public history of every tracked pick — confidence tier, predicted vs. actual method, a calibration curve checking whether "70% confident" actually wins ~70% of the time, and a real units/ROI ledger |
| 💰 **Units tracker** | Picks sized by confidence (5U high / 3U medium / 1U low), priced at real market odds — a running P&L, not just a win/loss count |
| 🎯 **Standout Props** | Method and round-total props where the model's read disagrees meaningfully with the market |
| 🎰 **Parlay builder** | Auto-generated bankroll, lotto, and moonshot parlays from the model's own edges |
| 📈 **Live line movement** | Odds refresh client-side during fight windows, no page reload needed |
| 🔄 **Self-updating** | A GitHub Action rebuilds the whole site every 15 minutes — results, odds, and predictions all refresh on their own |

## 🧠 How the model works

1. **Elo ratings** are built from `data/fight_history.csv` — thousands of real, decisive UFC fights. Finishes (KO/TKO, submissions) move ratings more than decisions, since they're a more decisive signal.
2. **Style-matchup adjustments** layer on top of the Elo gap: takedown accuracy vs. defense (or control-time when available), striking accuracy and volume differential, submission-finish rate, historical durability (finish-loss rate), layoff/ring-rust penalties, age-cliff risk by weight class, missed-weight history, and stance mismatches — each with a documented, capped scale so no single factor can dominate the prediction.
3. **Recent form** applies a decaying bonus/penalty based on how recently and how impressively each fighter last won or lost.
4. **Moneyline edges**: the model's implied win probability vs. the market's *vig-removed* fair probability.
5. **Method / round-total props**: priced off each fighter's historical finish-method distribution.
6. Everything is ranked by **Edge %** and comes with a suggested (capped) Kelly stake size, purely as a bankroll-math reference — not a recommendation.

<details>
<summary><b>Full adjustment-factor list, with rationale</b></summary>

| Factor | What it captures |
|---|---|
| Wrestling | Takedown accuracy vs. opponent's takedown defense, or control-time % when available |
| Striking | Striking accuracy differential, plus significant-strikes-landed vs. -absorbed volume gap |
| Submission threat | Career submission-win rate — a distinct skill from generic wrestling/control stats, since a fighter can have modest takedown numbers but a live finishing threat off scrambles or guard |
| Durability | Historical finish-loss rate (how often each fighter has been finished, by any method) |
| Layoff | Penalty scaling with time since last fight, past a grace period |
| Quick return | Penalty for fighting again unusually soon after the last bout |
| Age cliff | Penalty past a weight-class-specific age threshold |
| Missed weight | Penalty per historical missed-weight instance |
| Stance | Bonus for a southpaw/switch fighter against an orthodox opponent (a real, well-documented stylistic edge) |

All adjustments are capped in aggregate so the Elo base gap — the part actually validated by walk-forward backtesting — still dominates the final number.

</details>

## 🏗️ Architecture

```
ufc-edge-finder/
├── generate_site.py          # Builds the static site (docs/index.html) — the actual entry point
├── app.py                    # Local Flask dev server (optional, for testing without regenerating)
├── find_ev_bets.py           # CLI: flag mispriced props from the command line
├── etl_fight_history.py      # Rerunnable ETL: raw data → data/fight_history.csv
├── backtest_model.py         # Smell-test backtest against historical results
├── walkforward_backtest.py   # Genuine point-in-time walk-forward backtest + calibration table
│
├── src/
│   ├── elo.py                 # Elo rating engine, built from full fight history
│   ├── power_rating.py        # Blends Elo with raw stats by tracked-fight count
│   ├── matchup_model.py       # Style-matchup adjustment layer (see table above)
│   ├── edge_finder.py         # Combines model + live odds into ranked edges
│   ├── card_matcher.py        # Groups edges onto known fight cards
│   ├── schedule.py            # Self-correcting fight-time estimation + card auto-rotation
│   ├── results_fetcher.py     # Best-effort automated results scraper (ufcstats.com + Wikipedia fallback)
│   ├── track_record.py        # Prediction logging, matching against results, ROI/units math
│   ├── rationale.py           # Prose explanations for standout picks
│   ├── model_preview.py       # Per-fight "Model Preview" narrative generation
│   ├── parlay_builder.py      # Bankroll / lotto / moonshot parlay construction
│   ├── polymarket_source.py   # Primary live-odds source (Polymarket Gamma API)
│   ├── draftkings_scraper.py  # Fallback live-odds source (unofficial DraftKings endpoints)
│   ├── live_odds.py           # Last-resort fallback (The Odds API)
│   ├── line_movement.py       # Moneyline movement chart (SVG)
│   ├── calibration_chart.py   # Calibration curve (SVG)
│   ├── sparkline_chart.py     # Accuracy/units trend sparklines (SVG)
│   ├── units_chart.py         # Full ROI time-series chart with zero baseline (SVG)
│   ├── donut_chart.py         # Post-fight significant-strikes donut (SVG)
│   ├── damage_silhouette.py   # Post-fight damage-by-target silhouette (SVG)
│   ├── radar_chart.py         # Tale-of-the-tape radar comparison (SVG)
│   └── scraper.py             # Manual fighter-stat scraper (run yourself, not wired in automatically)
│
├── templates/site.html       # The entire static site — Jinja2 template, single file
├── data/                     # CSV inputs: fighters, fight history, cards, results, predictions log
├── docs/                     # Generated output — what GitHub Pages actually serves
└── .github/workflows/        # The GitHub Action that rebuilds everything every 15 minutes
```

## 🚀 Getting the live, auto-updating site

This is the primary way to run this project — a static site that keeps itself current with no server to maintain.

1. **Nothing required for odds data** — Polymarket (primary) and DraftKings (fallback) both need zero setup or API keys.
2. **Optional**: get a free key at [the-odds-api.com](https://the-odds-api.com) as a last-resort fallback, and add it as a repo secret (`Settings → Secrets and variables → Actions → New repository secret`, name it `ODDS_API_KEY`). Skippable — the site works without it as long as Polymarket or DraftKings are reachable.
3. **Enable GitHub Pages**: `Settings → Pages` → under "Build and deployment," set Source to "Deploy from a branch," branch `main`, folder `/docs`.
4. Push to `main`. The workflow runs automatically on push, every 15 minutes after that, and any time you trigger it manually from the Actions tab (including from the GitHub mobile app).
5. Your live link will be `https://<your-username>.github.io/<repo-name>/`

### Running locally instead

```bash
pip install -r requirements.txt
python generate_site.py   # builds docs/index.html once
python app.py              # or: run the live Flask dev server at http://127.0.0.1:5000
```

<details>
<summary><b>CSV data formats</b></summary>

**`fighters.csv`**: `name, weight_class, height_in, reach_in, stance, wins, losses, ko_wins, sub_wins, dec_wins, ko_losses, sub_losses, dec_losses, strike_accuracy_pct, td_accuracy_pct, td_defense_pct, last_fight_date, first_round_finish_pct, age, last_fight_result, last_fight_method, missed_weight_count, slpm, sapm, control_time_pct, last_fight_opponent`

**`fight_history.csv`**: `date, fighter_a, fighter_b, winner, method` (method is `KO/TKO`, `SUB`, or `DEC`)

**`fight_cards.csv`** / **`future_cards.csv`**: `event_name, event_date, card_position, weight_class, fighter_a, fighter_b, event_start_time_et, is_womens_division`

**`fight_results.csv`**: `event_name, fighter_a, fighter_b, winner, method, end_round, end_time, date_added` plus 20 optional strike-breakdown columns (`fa_sig_landed`, `fa_head`, etc.) — only the first 8 are required; the rest unlock the post-fight strike/damage visuals when populated.

**`predictions_log.csv`**: auto-generated by `track_record.py` on every site build — one row per tracked fight, locked once a result exists so ongoing model tuning can't retroactively rewrite what was actually predicted before the outcome was known.

</details>

## 📊 Live odds: source priority

**Polymarket → DraftKings → The Odds API**

1. **Polymarket (primary).** Polymarket's Gamma API is fully public, documented, and requires no auth — a peer-to-peer prediction market, so prices carry no bookmaker vig. Converted to familiar American odds format before anything touches the site.
2. **DraftKings (fallback).** No official public API — this uses unofficial, reverse-engineered endpoints their own site calls internally. Real risk: it can break without warning if DraftKings changes their JSON structure, and cloud/CI IP ranges are exactly the kind of traffic sites like this tend to rate-limit.
3. **The Odds API (last resort).** Moneyline only for MMA on their platform.

**FanDuel** isn't scraped directly — their bot detection is heavier with no clean unofficial pattern like DraftKings has. A paid aggregator (OpticOdds, OddsPapi) is the realistic path if you specifically need it.

Check the "Odds via ___" badge at the top of the live site to see which source actually served that refresh.

## 🛠️ Ideas to extend

- Round-by-round strike decay (needs per-round data, not just per-fight totals)
- Weight the model by camp/coaching changes, not just fighter-level stats
- Backtest the submission-threat and stance factors specifically once more events accumulate
- A second, fully independent results source beyond ufcstats.com + Wikipedia

---

<div align="center">

Made with love, by yours truly, SBP 🤍

</div>
