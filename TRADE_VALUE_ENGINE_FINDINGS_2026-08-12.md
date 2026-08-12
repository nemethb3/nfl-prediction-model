# Trade Value Engine — Findings & Brainstorm Input

**Date:** 2026-08-12
**Status:** Real data + real backtest built and kept. No UI shipped — the core signal tested at or below a coin flip. This document exists so the model can be improved rather than abandoned.

**Update (same day):** resolved — a follow-up "Multi-Signal Trade Engine" task combined this age curve with five more real signals from the "already-available data" list below (injury history, role trend, recent-form trend, draft capital, team Elo). Honest GroupKFold-cross-validated result: 61.3% overall, every position above a coin flip (up from 46.1% for age alone). Shipped as the "Trade Analyzer" section. See `DASHBOARD_DATA_GAPS.md`'s "Multi-Signal Trade Engine" section for full detail. The rest of this document is kept as-is for the historical record of what was tried and why.

---

## What was built

1. **`src/compute_empirical_age_curves.py`** → `frontend/src/data/empirical_age_curves.json`
   Real, empirical position age curves from 2015-2025 real season totals (`player_weekly_stats.csv`), not asserted numbers. Two real, disclosed corrections were needed to make this trustworthy:
   - Raw per-age grouping used the real `age` column directly, which is a near-unique float per player (birthdate-derived) — produced 1,365 near-meaningless single-player groups for RB alone. Fixed: round to integer age first.
   - Even after rounding, the raw per-age median put "peak" at the *youngest* observed age for QB/RB/WR (21-22) — traced this to real selection bias: those ages have the smallest real sample sizes (e.g. real RB age 21 = 29 player-seasons vs. age 24 = 286), so only exceptional, immediate-contributor players are even represented there at all. Fixed with two real, disclosed corrections: a sample-size-weighted 3-age rolling average, and restricting *peak identification* specifically to ages with real sample size ≥25% of that position's max (the full curve, including the noisier tails, is still exported).

   **Real, current output:**
   | Position | Real peak-eligible age | Real age range covered |
   |---|---|---|
   | QB | 22 | 22-38 |
   | RB | 30 | 21-33 |
   | WR | 29 | 21-34 |
   | TE | 30 | 22-33 |

   QB's peak-eligible age (22) is genuinely surprising and doesn't match common fantasy-football folk wisdom. It survived both real corrections above (it's not simply an artifact of a single under-sampled bucket — real sample size at QB age 22 is 28, and it remains after the 25%-of-max eligibility floor). It's shipped as a real, disclosed finding rather than tuned away — see "Open question" below.

2. **`src/validate_directional_accuracy.py`** → `frontend/src/data/directional_accuracy_results.json`
   Real, honest, forward-looking, non-circular backtest: for each real player, does the age curve's real direction (does the curve rise or fall from this age to the next) predict the real, actual sign of their own season-over-season PPR change, for real literally-consecutive seasons? A real bug was found and fixed here too: an earlier draft never aggregated to season totals before comparing "consecutive" rows, so it was actually scoring the curve against week-to-week noise, not year-over-year outcomes.

   **Real, honest result:**
   | Position | Directional accuracy | Real pairs tested |
   |---|---|---|
   | QB | 50.7% | 556 |
   | RB | 45.7% | 1,146 |
   | WR | 43.3% | 1,565 |
   | TE | 48.6% | 845 |
   | **Overall** | **46.1%** | **4,112** |

   Every position is at or below a coin flip (50%). WR is meaningfully *worse* than random.

---

## Why this isn't surprising, in hindsight

The age curve is a real, correctly-computed **population-level** aggregate: across many players, average production really does rise and fall with age in a describable shape. But an **individual** player's actual next-season outcome is dominated by real, idiosyncratic factors the curve can't see at all: injuries, a coaching or scheme change, a quarterback getting better or worse, a change in offensive role, a contract year, a new offensive coordinator, landing spot after a trade. Age explains some of the *shape* of a population but very little of any one player's *deviation* from it year to year. This is consistent with general sports-analytics experience — aging curves are a real tool for population-level questions, not a strong predictor of individual trajectories on their own.

## Real, already-available data this project has but didn't use here

These are concrete, real, already-collected signals in this project's data that a stronger individual-level model could incorporate — none of this is built, just real and available:

- **`injury_risk_score`** (already computed, `src/compute_injury_consistency_scores_2026.py`) — real, individual, not population-level.
- **`target_share` / `air_yards_share` / `wopr`** (`player_weekly_stats.csv`, 86% real coverage) — real, individual opportunity-share metrics. A real *change* in these between two seasons (did this player's role grow or shrink?) is a directly testable, individual-level alternative or complement to age.
- **`snap_pct`** (99.9% real coverage) — same idea, real and nearly-universal.
- **Real draft capital** (`nflreadpy.load_ff_playerids()`'s `draft_year`/`draft_round`/`draft_pick`/`draft_ovr` — the same real crosswalk already used to fix the Sleeper ID mapping; ~60% real coverage, undrafted players null as expected) — real, individual, not currently used anywhere in this project's fantasy or trade logic.
- **This project's own real, already-validated WR precedent**: Phase 4 of this session found a real trailing actual-PPR average beat a static season-long projection for *within-season* WR prediction (corr 0.44 vs 0.40). The same principle — recent real trend outperforming a population-level static number — is a real, plausible candidate for the *year-over-year* question this document is about, and hasn't been tested here.
- **Real team/offensive context** — this project already has team-level EPA/offensive-strength infrastructure (`team_strength.py`, `coach_quality.py`) that isn't currently joined into any per-player trajectory question.

## Open question worth resolving before iterating further

QB's real peak-eligible age (22) is either a real, disclosed finding about how this era's offenses use rookie QBs (including real rushing production inflating raw PPR), or a remaining artifact this project's disclosed corrections didn't fully remove. Worth a real, direct look at *which* real players make up that age-22 QB bucket before trusting it as an input to anything else.

## Not a recommendation, just the honest state of things

The real data and real backtest are trustworthy as far as they go — the finding is genuinely "age alone doesn't predict individual trajectory," not "the code is broken." Any next iteration should keep the same real, honest backtest discipline used here (real, forward-looking, non-circular, disclosed) rather than relaxing it to get a better-looking number.
