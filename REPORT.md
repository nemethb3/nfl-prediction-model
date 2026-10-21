# Juke Weekly Picks — Calibration & Backtest Report

Generated 2026-10-06. Walk-forward test seasons: 2023, 2024, 2025 (816 real completed games). Production model: this project's own O/D-Elo (k=180.0), the exact same fit that drives the live 2026 `our_spread`/`win_prob_home`. Each test season's predictions come from a model trained **only** on seasons strictly before it (train 2015–2022→test 2023; 2015–2023→test 2024; 2015–2024→test 2025) — no lookahead.

## One-line verdict

**No.** The model does not show a real out-of-sample edge over the market. The literal log-loss-minimizing blend weight is **w = 0.00 for both moneyline and spread** — shrinking fully to the market's own no-vig price beats any blend with the model. The resulting live strategy (see below, run with a disclosed non-zero `w` so it can actually place bets) loses money overall (-3.11% ROI) and underperforms the "bet the Vegas favorite flat" baseline (+4.01% ROI) across the 3-season backtest.

## Sigma (spread residual std, out-of-sample)

**σ = 13.25 points** — std of (actual home margin − model's predicted margin) across all 816 held-out games. This is *larger* than a typical in-sample NFL spread residual (~10–11 pts), consistent with the model being somewhat overconfident out-of-sample.

## Blend weight w

| Market | w (log-loss optimal) | w (used live) | Why used ≠ optimal |
|---|---|---|---|
| Moneyline | 0.00 | **0.56** | See note below |
| Spread | 0.00 | **0.10** | See note below |

**Note — this is a disclosed methodology deviation, not a data-driven result.** The literal optimum (w=0) makes `p_final` mathematically identical to the market's own no-vig price for every bet, which drives `edge` and `EV` to exactly 0.00 on every game — a model run that way can never produce a real pick. The user asked for the best usable allocation now, with model improvement ongoing. `w_used` is the standard **"within one standard error of the best"** selection (the same rule used to pick a lasso/ridge regularization strength): the most model-trusting `w` whose pooled log-loss is still statistically indistinguishable from the true optimum (SE = 0.0123 moneyline, 0.0008 spread). It is explicitly **not** evidence the model is 56%/10% as good as the market — it's the least-damaging way to generate non-degenerate picks.

Log-loss comparison (lower is better):

| Market | Model only | Market only | At w_used |
|---|---|---|---|
| Moneyline | 0.6387 | 0.6077 | 0.6197 |
| Spread | 0.7199 | 0.6936 | 0.6943 |

## Calibration tables (predicted-probability bucket vs. actual hit rate)

**Moneyline, model only (n=816):**

| Bucket | n | Mean predicted | Actual hit rate |
|---|---|---|---|
| 20-30% | 49 | 0.253 | 0.347 |
| 30-40% | 96 | 0.360 | 0.302 |
| 40-50% | 175 | 0.454 | 0.440 |
| 50-60% | 171 | 0.548 | 0.509 |
| 60-70% | 164 | 0.649 | 0.671 |
| 70-80% | 104 | 0.742 | 0.721 |
| 80-90% | 50 | 0.839 | 0.880 |

Reasonably well-calibrated on its own (no bucket is wildly off), which is why the "model is broken" explanation was ruled out — it's just not *better* than the market, which is the relevant bar here.

**Moneyline, final blend (w=0.56, n=816):** calibration holds up similarly (50-60%: predicted 0.548 vs actual 0.561; 60-70%: predicted 0.650 vs actual 0.694) — blending doesn't visibly break calibration, it just doesn't improve on the market's own log-loss.

**Spread, model only (n=797):** noisier — 20-30% bucket predicts 0.267 but hits 0.467; 60-70% predicts 0.636 but hits only 0.466. The spread-cover model is the weaker of the two signals, consistent with w_spread's optimum landing at 0.00 with a much tighter standard error (0.0008 vs 0.0123) — the market's spread pricing is very hard to beat.

**Spread, final blend (w=0.10, n=797):** compresses almost entirely into the 40-60% bands (397 + 400 of 797 games), which tracks reality closely (50.4% / 50.0%) — a low-information blend that's honest about how little edge the spread side contributes.

Full bucket tables (all 10 buckets, both markets, both model/final) and the full weekly backtest ledger are in `output/juke_calibration_and_backtest_report.json` if needed for deeper review.

## Backtest vs. baselines (fresh 1000 credits every week, same strategy/staking as the live system)

| Season | Strategy bets | Strategy win% | Strategy ROI% | Baseline A (our pick, flat) ROI% | Baseline B (Vegas favorite, flat) ROI% |
|---|---|---|---|---|---|
| 2023 | 196 | 33.7 | **-2.13** | -7.09 | **+4.97** |
| 2024 | 215 | 36.3 | **-8.89** | -10.59 | **+10.16** |
| 2025 | 213 | 35.2 | **+1.69** | -3.15 | **-3.09** |
| **Overall** | 624 | 35.1 | **-3.11** | -6.94 | **+4.01** |

**The strategy does not beat either baseline out of sample, overall.** It beats Baseline A (flat stakes on our own picks — Kelly sizing + selection help some) in every season, but loses to Baseline B (flat stakes on the market's own moneyline favorite) in 2 of 3 seasons and overall. Baseline B winning is itself a real, somewhat surprising finding — NFL moneyline favorites covering their vigged price at a 4%+ ROI clip over 3 seasons is not something this backtest set out to find, but it's visible directly in the numbers and worth independent follow-up.

Worst single week: **-$698.41** (2024, week 15). Largest within-season drawdown: **-$2,830.73** (2023). Credits allocated every single week = exactly 1000 (54/54 weeks, confirming the staking math holds under real data, not just unit tests).

## Core vs. filler

**Filler was never used in the 3-season backtest (0 of 624 bets).** With w_used=0.56 for moneyline, enough games per week clear the 3% MIN_EV core threshold on their own (avg ~11.6 bets/week) that the "lower the bar to reach MIN_PICKS=7" fallback never triggered. This is itself informative: it confirms MIN_PICKS=7 was already a non-binding constraint for the chosen `w`, not a lever that's quietly doing a lot of the selection work.

## Where this falls short, and how to improve it

1. **No real out-of-sample edge exists yet.** This is the core finding, not a side detail. Until the model beats the market's log-loss on fresh data, any allocation (including this week's) is noise dressed as a strategy. Priority: feature work on the O/D-Elo model itself (QB-adjusted ratings, injury-aware adjustments, situational rest/travel factors) rather than more calibration tuning — the current gap is model quality, not shrinkage weight.
2. **Spread signal is weaker than moneyline** (SE an order of magnitude tighter, meaning the market's spread edge over the model is more statistically certain). If forced to choose where to invest modeling effort first, moneyline has more realistic upside.
3. **w_used is a disclosed workaround, not a result.** Every week this runs with w > 0, it's realistically more likely to cost credits than earn them, per this exact backtest. This should be revisited the moment there's a real walk-forward season where w_optimal > 0.
4. **ESPN's "current week" scoreboard default is unreliable** (found live during this build — it was still reporting the prior week as current a day after it ended). Fixed by passing an explicit week number, but worth monitoring — if ESPN's week numbering ever drifts from this project's own `real_current_week_2026()`, odds could silently be pulled for the wrong week again.
5. **Only one book (DraftKings) has been observed per game so far.** The no-vig price is only as good as its single source; a second/third book would make the no-vig estimate noticeably more robust, and would let the de-vig step median across books instead of trusting one.
6. **No real per-side spread betting price has ever been collected** — `-110` is used as a disclosed default for every spread bet (see `used_default_spread_juice` in the internal log). ESPN's API does expose a real one (`awayTeamOdds.spreadOdds`); worth adding to the collector.
7. **The reference-line observation timestamp is the odds-collector's run time, not a true "line as of kickoff minus Xh" snapshot.** Good enough for "was this fetched before kickoff" (now enforced), not yet good enough for "how much did the line move after we looked."
8. **Baseline B's +4.01% ROI deserves its own investigation**, independent of this picks engine — if real, it's a more useful finding than anything else in this report.

## Addendum (2026-10-06, same day) — ensemble, "Our System", and parlays

Three follow-ups, each tested with real data rather than assumed:

**1. Elo ensemble (use more of the models already built).** Added single-Elo (`elo_game_prediction.py` — this project's other real, already-fitted game model) alongside O/D-Elo, blended with a real walk-forward-fit weight rather than an assumed 50/50: **e_moneyline = 0.87, e_spread = 0.74** (O/D-Elo's own weight — it's the better of the two models, but single-Elo adds a little). Result: model-only spread MAE improved slightly (10.27 → 10.24 pts), but the core finding is unchanged — market-shrinkage `w_log_loss_optimal` is still 0.00 for both markets, and the 3-season backtest ROI barely moved (-3.11% → -3.56% overall). Using more of the model didn't surface a hidden edge; it's now live in the picks engine anyway since it's a real, if small, improvement and costs nothing to keep.

**2. "Our System" validated against 3 full seasons, not 40 bets.** The live Betting Analysis dashboard's "Our System" strategy is currently showing +10.2% ML / +5.5% ATS ROI on 40 real 2026 bets. I ran that *exact* heuristic (`betting_backtest.py`'s own `should_bet`/`get_bet_direction`, imported directly, not reimplemented) across all 677 qualifying bets in the 2023-2025 walk-forward sample:

| Season | Bets | ML ROI% | ATS ROI% |
|---|---|---|---|
| 2023 | 223 | -4.65 | -3.37 |
| 2024 | 216 | -13.76 | -2.40 |
| 2025 | 238 | -4.16 | -8.48 |

**It loses money in every season, by a wide margin.** This is the honest read: 40 bets is not enough to distinguish a real edge from variance, and the larger, out-of-sample sample says the live dashboard's current positive number is very likely noise, not a durable strategy. I did not gate picks on this signal — doing so would mean filtering toward something 677 games says loses money. It's still tagged (`confirmed_by_our_system`) in the internal log for ongoing tracking, in case the live number keeps diverging from this historical read in a way worth another look.

**3. Parlays: tested directly, clear no.** I built 2-leg and 3-leg parlay combinations from each week's actual top-EV candidates and ran the single best 2-leg parlay through the exact same walk-forward backtest (54 weeks, Kelly-sized, real settlement). Result: **2 wins out of 54, -51.15% ROI.** Parlays look attractive on paper EV (compounding two ~0.42 EV legs can show "EV ≈ 1.9" per credit) precisely because they compound the model's already-demonstrated overconfidence multiplicatively instead of averaging it out — a small, likely-noisy edge on each leg becomes a much worse bet combined, not a better one. Not implemented in the live picks.

## Addendum 2 (2026-10-06, same day) — weak-point audit

You asked me to find weak points and fix no-brainers without touching the original models. Found 3 real bugs (fixed, none of them in the Elo models themselves) and several open limitations (disclosed, not fixed).

### Fixed

**1. Real per-side spread juice was never captured — a genuine bug, not just a disclosed gap.** ESPN's API returns an asymmetric per-side spread price (`homeTeamOdds.spreadOdds`/`awayTeamOdds.spreadOdds`, e.g. home -112 / away -108) in the exact same response the collector already fetches, right next to `moneyLine` — but `espn_odds_client.parse_odds` never read it, so every spread bet in this project (the juke engine *and* the existing `betting_backtest_2026.py` ATS settlement) was silently using a hardcoded symmetric -110/-110. That made the spread market's de-vigged "price" exactly 50.0% by construction for every single game, regardless of the real line. Fixed — now captures the real value and falls back to -110 only when a provider genuinely doesn't expose one. **Honest caveat on impact:** real spread juice rarely strays far from -110 anyway (books balance via the line, not the price) — this week's real prices came back 0.489-0.504, so the fix is correct and now real rather than hardcoded, but it was never going to be a big lever on its own.

**2. A related second bug this surfaced: the odds-history dedup key would have silently swallowed the fix.** `espn_odds_storage.append_history` dedupes same-day re-fetches by a tuple of fields that didn't include the new `spreadOdds` field — so the first re-run after fixing bug #1 was discarded as "no change," even though it carried real, previously-missing data. Fixed by adding the new fields to the dedup key, then manually backfilled today's already-collected rows.

**3. A real methodology bug in my own calibration code (not the Elo models): hyperparameter leakage across the walk-forward seasons.** σ, the market-shrinkage weights, and the Elo ensemble weights were all fit by pooling 2023+2024+2025 together, then that single fit was applied back to score all 3 seasons — meaning the reported 2023 backtest number was partly chosen *using* 2024-2025 outcomes, which hadn't happened yet. Restructured into a real dev/holdout split: every hyperparameter is now fit on 2023-2024 only, then evaluated unchanged on 2025, which never informed them at all. The honest, uncontaminated number: **2025 holdout strategy ROI is -1.03%** (vs. the old pooled estimate of roughly -3 to -4%) — closer to breakeven than previously reported, and this is now the number to trust, not the old pooled one. Updated σ=13.32, w_moneyline=0.61, w_spread=0.21 (both fit on dev only).

### Disclosed, not fixed (real weak points, listed by how much they likely matter)

1. **Single-book pricing is a genuine external data limitation, not a code bug** — checked: the collector already calls ESPN's real multi-book endpoint, and nflverse's own `schedules_2026.csv` has zero real lines for Week 5 either (real Vegas totals there only cover weeks 1-4). Only DraftKings had a posted line for any Week 5 game as of today. Nothing to fix in our code; worth re-checking closer to kickoff when more books typically post.
2. **Line staleness between generation and actual bet placement** — prices are captured once per run; if there's a gap between generating picks and submitting them to Juke, the line may have moved. Not addressed (would need a real-time re-check at placement time, which this project's static-JSON, no-backend architecture isn't built for).
3. **σ is assumed constant across all games (homoscedastic).** Blowouts and close games likely have different real error variance. Left disclosed, not fixed — the whole project's existing CI convention (every model's own 90% band) is built the same homoscedastic way, so fixing this just for juke would mean inventing a new, untested variance model rather than reusing anything that exists, and would make juke inconsistent with how every other confidence interval in this project is computed.

## Addendum 3 (2026-10-06, same day) — fixing the open limitations

Fixed 2 of the remaining items with real, already-tested work. Found a third candidate fix that turned out to already exist in the codebase, ran its own built-in validation, and it didn't pass — so it stayed unfixed, correctly.

**1. Standard error now uses a real block bootstrap instead of assuming independence.** The per-game log-loss values aren't independent (games sharing a week share that week's exact Elo ratings) — the original SE (`std/sqrt(n)`) assumed they were. Replaced with a real block bootstrap: resample whole (season, week) blocks with replacement, 2000 times, and take the std of the resulting means. Honest result: the fix was methodologically correct to make, but it barely moved the number in practice — SE went from 0.0145 → 0.0147 (moneyline) and 0.0009 → 0.0008 (spread). Worth having done since it's now measuring the right thing, even though this particular dataset turned out not to have much serial correlation to matter.

**2. Calibration caching.** `fit_calibration()` now caches its result to `data/processed/juke_calibration_cache.json`, keyed on the real git commit hash plus the 3 parameters that affect the result — any of those changing invalidates the cache automatically. Cuts repeat runs from ~15-20s down to ~3s (mostly Python startup). Pure performance, no behavior change (verified identical output before/after).

**3. Investigated the injury-signal gap before touching anything — found real, already-built, already-fitted infrastructure for exactly this (`injury_model.py`: real nflreadpy injury reports correlated to real offensive EPA loss by position group, plus `game_predictions.fit_game_level_epa_to_points` for a real, already-fitted EPA→points conversion) that was never wired into any live pipeline. Before plugging it in, I ran its own built-in validation function on real 2025 data: MAE with the adjustment (0.1928) was *worse* than predicting zero injury effect at all (0.1927), and the correlation between predicted and actual EPA loss was -0.02 — essentially zero, slightly negative. This model was built, genuinely tested, and correctly never wired in because it doesn't work. Plugging it into the juke engine would mean reusing something its own validation already rejected, so I didn't. This remains the most plausible place a real edge could exist (the market reacts to injuries in real time; nothing in this project's models currently does), but closing it for real would mean building and validating a *new* injury-impact model from scratch — which is exactly the kind of change to the original models you asked me not to make without a no-brainer justification. Flagged as the single best candidate for a dedicated follow-up task, not attempted here.

Re-verified after all 3 changes: 23/23 Python tests pass, 8/8 frontend tests pass, backtest and Week 5 output regenerated and unchanged in shape (still 11 picks, exactly 1000 credits).

## Privacy scope (per your instruction)

`output/juke_picks_2026_week05.json` (the file meant for the third party) contains only: event id, kickoff time, market, selection, line, assumed price, credits, tier, and the real market-line source/timestamp. It does **not** contain `model_prob`, `final_prob`, `edge`, `expected_value_per_credit`, or `forecast_value` — those stay in the internal, append-only `data/forecast_log/juke_picks.jsonl` for this project's own ROI tracking only.

## Addendum 4 (2026-10-20) — weekly results tracker, a real sign bug, and a 10-pick cap

**New file: `output/juke_weekly_results.json`** (`src/track_juke_results.py`). Regenerated fresh each run from the real, canonical `output/juke_picks_*_week*.json` files, settled against games_2026.json's own real results. A week with any game not yet played is reported `"fully_settled": false` and excluded from the season-to-date total, rather than guessed. Real settled results so far:

| Week | Record (W-L-P) | Net credits | ROI% |
|---|---|---|---|
| 5 | 4-7-0 | +170.22 | +17.02 |
| 6 | 4-8-0 | -187.31 | -18.73 |
| **Season to date** | | **-17.09** | **-0.85%** |

Notably close to the -1.03% true-holdout backtest estimate from Addendum 2 — two weeks is nowhere near enough to confirm that, but it's not contradicting it either.

**Found and fixed a real sign bug while building the tracker**, before it could corrupt spread settlement: the exported `line` field in every output JSON was in the *opposite* sign from what the submission declares (`"spread_convention": "sportsbook (negative = favorite)"`) — a favored team's own line was showing positive instead of negative. This never affected any EV/Kelly/staking math (that uses the no-vig price, not this display field) and never corrupted a real pick, because every pick generated so far happened to be moneyline. Fixed in `juke/pricing.py`; the test that had encoded the bug as "expected" is fixed too.

**10-game cap implemented** (`MAX_PICKS=10` in `juke/config.py`, enforced in `juke/staking.py`): keeps only the highest-EV picks when more than 10 qualify, never pads up to 10 when fewer genuinely do (Week 7 came back with 9 real picks on its own, validating that the "fewer is fine" half of this actually works). Honest side effect on the backtest: capping to 10 made the true 2025 holdout ROI *worse* (-1.03% → -3.74%) — fewer, more concentrated bets raised variance without improving the average outcome in this sample. Implementing the cap was the right call per your instruction either way; flagging the backtest number moved so it's not mistaken for an improvement.
