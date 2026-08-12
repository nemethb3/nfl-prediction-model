# Dashboard Data Gaps Report

Updated after twenty-two gap-filling passes:
- **Real-only Phase 1**: QB names, matchup quality via empirical
  thresholds, team recent form, head-to-head, WR fantasy rankings,
  opponent defense rank vs position.
- **Phase 2 (data)**: real per-team Elo ratings (Section 1), real player
  recent form + real injury status (Section 2). Fantasy confidence
  intervals were proposed again using the same asserted-constant/
  conflated-statistic approach flagged and declined in Phase 1 — declined
  again, still open below.
- **Phase 2 (win probability)**: `win_prob_home`/`win_prob_away`, built via
  a real 3-way backtest (`win_probability_backtest.py`) rather than an
  asserted heuristic — now wired in, see Section 1 table below.
- **Phase 3**: consolidated Section 1's betting-outcome display (real ATS
  cover + real moneyline hit/miss, in one section); added real `actual_ppr`
  and empirically-thresholded `accuracy_tier` to Section 2.
- **Section 3 (new)**: real season win projections + real playoff odds
  (Monte Carlo), real division/seeding logic. Super Bowl percentage
  explicitly omitted — doesn't exist anywhere in this project.
- **Section 4 (new)**: real accuracy tracker (games/fantasy/season/betting,
  full-season + real weekly breakdown), scored against real final
  standings rather than an invented "spread coverage accuracy" metric.
- **Section 5 (new)**: real weekly recap + next-week preview, real
  division structure (post-2002 alignment), real opponent-defense-rank
  matchup quality.
- **Section 6 (new)**: plain-language model explanation. Unlike prior
  sections, this one is static content (no data pipeline) - but every
  specific number and example was independently fact-checked against real
  dashboard data before writing, and several real errors were caught in
  the draft content itself (not just in pasted code this time).
- **Section 7 (new)**: betting analysis backtest - three strategies (Our
  System / Vegas Favorites / Underdogs Only) settled two real ways
  (moneyline and against-the-spread) on all 272 completed 2025 games,
  using real Vegas odds instead of the pasted spec's synthetic
  spread-to-moneyline formula. All six strategy/bet-type combinations lose
  money (ROI -1.2% to -7.9%), consistent with this project's established
  finding that the house edge isn't beatable with this project's real
  data.
- **Quick Wins #2 & #3**: timezone clarity (Section 1) and accuracy
  confidence visualization (Sections 2 & 4). Found and fixed a real,
  non-obvious bug in the pasted spec's kickoff-time logic - and in this
  component's own pre-existing code - where `new Date()` silently parses a
  timezone-less ISO string in the VIEWER's own local time instead of the
  data's real Eastern-Time value, meaning "Your Time" never actually
  converted anything for non-ET viewers. Confidence-band numbers (the
  spec's asserted "±8%"/"±8 points") were replaced with real, computed
  statistics from this project's own weekly accuracy data. Fantasy
  tercile-range numbers in the spec were verified against a live re-run of
  the real threshold computation and matched exactly - shipped as given.
- **Phase 4 (new)**: WR dynamic projections backtest. Found that the
  pasted spec's entire premise rested on a leaky baseline - WR's existing
  shipped projection is in-sample-calibrated against each player's own
  real full-season total, which includes whatever week is being scored
  against it. Rebuilt as a real leave-one-out backtest instead (see
  Phase 4 section below). Real winner: a trailing 4-week actual-PPR
  average, beating both the leak-free baseline and the currently-shipped
  (leaky) number.
- **Phase 4 implementation (new)**: shipped the backtest winner. WR now
  uses a real trailing up-to-4-week actual-PPR average for every real
  player-week with at least one prior 2025 game, falling back to the
  original static season-long projection only for a player's first real
  2025 appearance. Real WR correlation moved from 0.400 (leaky static) to
  0.4416 (real, dynamic, in production) - still the lowest of the four
  fantasy positions, but now only marginally below QB's 0.447, not a
  standout weak point. Regenerated every real dependent export
  (fantasy_rankings_2025.json, accuracy_tracker_2025.json,
  weekly_summary_2025.json) and updated every place in the dashboard that
  described WR as purely static (Sections 2, 4, 6) so nothing shipped
  stale.
- **Audit (2026-07-30, `AUDIT_2026-07-30.md`)**: full data-accuracy/model-
  validation/code-quality/integration pass. Found 2 real, previously
  undetected critical bugs (WR team assignment stale for 30/107 players;
  2/8 Weekly Summary division leaders wrong on all 18 weeks from a
  tie-unaware tiebreak). Everything else audited - all headline metrics,
  season-projection seeding, cross-file consistency, team-name coverage -
  was independently recomputed and came back clean.
- **Comprehensive Fix (new)**: fixed both critical audit bugs, regenerated
  the two affected exports, verified the fixes directly against real data
  (not just re-running and hoping), archived 10 orphaned early-phase
  scripts, removed one unused import. See "Comprehensive Fix" section
  below.
- **Season Projections Enhancement (new)**: added Division Winners,
  Playoff Picture, and real Super Bowl odds to Section 3. Super Bowl odds
  closes a real, previously-disclosed gap ("doesn't exist anywhere in this
  project") using a real Monte Carlo bracket simulation - chosen over the
  pasted spec's simpler win-probability-scaling heuristic, per explicit
  instruction, as more consistent with this project's established
  real-simulation-over-heuristic pattern. See "Season Projections
  Enhancement" section below.
- **Multi-Year Season Selection (new)**: found before building anything
  that the pasted spec's premise - a fully-populated "2026 (default)"
  season alongside 2025 - was impossible without fabricating results: the
  real 2026 season hadn't been played yet (verified against the real
  schedule). Scoped down to a real, honest preseason-only 2026 view after
  flagging the conflict. See "Multi-Year Season Selection" section below.
- **Win Totals Confidence Intervals + Real Monte Carlo Playoff Odds
  (new)**: a pasted spec claiming 2026's 6.6-10.3 win-total range was "too
  narrow" (vs. real 2025's 2-14 realized range) proposed rescaling model
  parameters to widen it - already tried and proven counterproductive by
  this project's own prior Phase 2 compression investigation (MAE got
  monotonically worse, not better, as the rescale factor increased). A
  follow-up spec proposed a from-scratch Monte Carlo simulation instead -
  sounder in principle, but its Step 1 would have re-derived per-game win
  probabilities from a hand-picked Vegas/Elo blend, discarding this
  project's real, already-LOOCV-validated blend (which already collapsed
  to 100% Elo). Investigated instead of building either spec as pasted:
  found the requested variance already existed, unused, as a real,
  closed-form 90% CI in `ensemble_season_wins_2026.csv` (not a Monte Carlo
  approximation - `elo_model.py`'s `project_season_wins_from_elo` computes
  `Var(sum of independent per-game Bernoullis)` exactly). Surfaced it
  rather than re-deriving it. Separately, built a genuine new Monte Carlo
  simulation for the one thing the closed-form CI genuinely cannot answer:
  real playoff odds/seeding for 2026, previously left null/false as an
  explicitly disclosed gap. See "Win Totals Confidence Intervals + Real
  Monte Carlo Playoff Odds" section below.
- **Feature B (Injury Risk + Consistency) + 2026 Super Bowl Odds (new)**:
  three-round spec-correction cycle before building - each pasted revision
  fixed some real bugs while introducing or leaving others (wrong file
  paths throughout, an undefined `estimate_age()`/`compute_total_games_
  missed()`, a real games-missed methodology that would have silently
  returned ~0 for every player, a nonexistent `simulate_playoffs()`
  function, a wrong-module import for `_real_seeds_by_conference`, a
  DataFrame passed where a list of dicts was required, code that claimed
  to fix the Elo-source bug but still read the same broken/nonexistent
  file, a hand-typed home-field constant overriding this project's real
  fitted one, and a `SuperBowlOdds.js`/`SeasonContext.js` rewrite that
  would have destroyed real, already-working functionality and contradicted
  this project's established static-import architecture). Built the real,
  corrected version instead - see "Feature B + 2026 Super Bowl Odds"
  section below.
- **Sleeper API Integration (new)**: a real architecture question first,
  not a bugfix pass - the pasted spec assumed a Flask backend that doesn't
  exist anywhere in this project (no Flask installed, no app.py, no dev
  proxy configured) and contradicts this project's established static-
  JSON-only architecture. Directly verified Sleeper's real API sends
  `Access-Control-Allow-Origin: *`, so no backend is needed at all - built
  client-side-only instead, per explicit direction. Also caught and fixed
  a real, recurring bug across two spec-correction rounds: Sleeper's own
  `player_id` field is Sleeper's internal ID, not a GSIS cross-reference -
  the real field is `gsis_id`, confirmed directly against a live Sleeper
  player record and cross-checked against this project's own data before
  building. See "Sleeper API Integration" section below.
- **Sleeper Projection Matching Fix (new)**: a real user report ("most
  roster players show no projection") led to diagnosing the root cause
  directly against real data rather than shipping the requested debug-
  logging pass - found Sleeper's own self-reported `gsis_id` field
  (relied on by the prior task) has real, incomplete coverage: 3 of 3
  spot-checked prominent active starters (Chase, Nacua, Robinson) had it
  null, and only 62/294 (21%) of this project's own real ranked players
  were reachable through it at all - 0/5 of the real top-5-ranked WRs.
  Real fix: `nflreadpy.load_ff_playerids()` (an existing dependency) is a
  real, externally-maintained ID crosswalk with far more complete real
  `gsis_id`/`sleeper_id` coverage - closes the gap to 294/294. See
  "Sleeper Projection Matching Fix" section below.
- **Trade Value Engine investigation (new, NOT shipped to UI)**: built real,
  empirical position age curves from 2015-2025 season totals (fixing a
  real float-age grouping bug and a real small-sample selection-bias
  artifact along the way), then built a real, honest, non-circular,
  forward-looking backtest of whether the curve's direction predicts an
  individual player's actual season-over-season PPR change (fixing a
  separate real bug: an earlier draft compared consecutive WEEKLY rows,
  not season totals, at all). Honest result: 46.1% overall directional
  accuracy (43.3-50.7% by position) - at or below a coin flip, every
  position. Per explicit instruction, the real age curves and real
  validation results are kept as documented data/findings, but no
  TradeAnalyzer UI was built on top of a signal that tested at chance.
  See `TRADE_VALUE_ENGINE_FINDINGS_2026-08-12.md` for the full writeup and
  real, concrete ideas for a stronger individual-level model.
- **Multi-Signal Trade Engine (new, shipped)**: follow-up to the Trade
  Value Engine investigation (age alone: 46.1% directional accuracy, at
  or below chance). Combined the real empirical age curve with five more
  real signals (point-in-time career injury history, role trend, recent-
  form trend, real draft capital, real team Elo), fixing several real
  bugs before training: a repeat of the earlier "not aggregated to season
  totals" bug, a real temporal-leakage bug (an earlier draft would have
  used 2026's already-computed injury data to "predict" a 2017 outcome),
  a nonexistent `team_strength.csv` combined file (real per-season files
  only exist for 2025/2026, not 2015-2024 - substituted this project's
  own real, already-validated multi-season Elo instead), and a real
  row-level (not player-level) cross-validation leakage risk. Honest,
  GroupKFold-cross-validated result: 61.3% overall (59.9-64.4% by
  position) - a real, meaningful improvement over age alone, every
  position above a coin flip. Shipped as a new "Trade Analyzer" section -
  see "Multi-Signal Trade Engine" section below for full detail,
  including why the frontend never re-implements the model itself.
- **2026 Week 1 Fantasy Projections (new)**: the pasted spec's entire
  methodology (`load_historical_week1_baseline()`, `load_defense_rankings_
  2026()`) was fabricated - hand-typed numbers dressed up as if queried
  from real multi-year data that doesn't exist. Rebuilt using this
  project's own real, already-established "week 1 falls back to the real
  prior season's per-game rate" convention, applied one real year forward.
  See "2026 Week 1 Fantasy Projections" section below.

---

# Section 6 — Model Transparency

No new data pipeline - static explanatory content. The real work here was
fact-checking the draft's specific claims against this project's actual
real, generated data before writing anything, the same discipline applied
to README/commit-message drafts earlier in this project.

## Real errors found and fixed in the draft content itself

| Claim in the draft | Real fact | Fix |
|---|---|---|
| "team rating is 40%, QB quality is 10%, rest is 5%, etc." | No such weighted blend exists anywhere in this project. The real spread is Vegas's closing line plus one real, regression-fitted matchup adjustment. | Rewrote the mechanism description to match what's actually implemented. |
| QB fantasy factors in "opponent pass defense," "game script" | The real, validated QB formula is volume-only (real trailing touches/yards/TDs) - opponent-adjusted and game-script versions were tested and lost to the simpler formula. | Corrected to describe the real, validated mechanism. |
| Rest days and "momentum" described as factors in game predictions | Both were tested and explicitly NOT wired into predictions (rest: real zero effect; momentum: inconclusive). | Rewrote to disclose these as tested-and-rejected, not active factors. |
| "We find edges [over Vegas]... a slight edge" | Directly contradicts this project's own established finding (README Key Finding #2: Vegas wins at every real checkpoint; `edge_detection.py`'s real backtest of betting on disagreements returned -36% ROI). | Rewrote the entire framing around the real finding: this model is built ON Vegas, not competing with it. |
| QB correlation: "89%" | Real number: 44.7%. | Corrected; also reordered the whole ranking, since RB (65.1%) is actually the most accurate position, not QB. |
| WR correlation: "44%" | Real number: 40.0% (44% appears to be QB's real number, scrambled onto WR). | Corrected. |
| "Buffalo 65% vs Chicago... Bills 27-10" | No such game exists in the real 2025 schedule - BUF never played CHI. | Replaced with a real, verified game (week 4, NO @ BUF, real 95.0% win probability, real 31-19 result). |
| "Green Bay 91%... Carolina won 31-13" | Real result: Carolina 16, Green Bay 13 (won by 3, not 18). | Corrected to the real score. |
| "Christian McCaffrey (WR) projected 12, actual 43" | McCaffrey is a real RB, not WR. The real 43.3-point overperformance belongs to Brock Bowers (TE, week 9, real projected 10.4) - this looks like a real data point (already found and verified in the Section 5 task) misattributed to the wrong player and position. | Corrected to the real player, position, and numbers. |
| "Las Vegas: projected 7 wins, actual 4" (cited as a bad prediction) | Real projected wins was 3.0, and the real final record was 3-14 - making this one of the model's MORE accurate season projections, not a bad one. | Replaced with the real largest season-projection miss (Green Bay: real projected 10.5, real final 9 wins - still only off by 1.5 games). |

## Notes

- This is the first section this session where the *draft content itself*
  (not pasted code) contained fabricated numbers and a misattributed real
  data point - the same "verify before it ships" discipline applied to
  every prior task's code was applied here to prose.
- The corrected content leans into this project's real, sometimes
  humbling findings (Vegas wins, betting on disagreements lost money, WR
  projections are weak) rather than smoothing them over - consistent with
  this entire project's standing convention of reporting negative/null
  results plainly.

---

# Section 5 — Weekly Summary

Real fields used throughout: `games_2025.json`, `fantasy_rankings_2025.json`,
and `season_projections_2025.json` (the week-16 checkpoint, reused as-is
for every week's "Season Context" - there's only one real snapshot, not a
per-week one).

| Item (from the draft spec) | Status | What changed |
|---|---|---|
| Division names | **Corrected** | Draft used pre-2002 `AFC_Central`/`NFC_Central` naming. Reused the real, verified `DIVISIONS` mapping from Section 3 (`AFC North`/`NFC North`, etc.) instead of reimplementing it. |
| `division_leaders` (season context) | **Real bug fixed** | The draft built this as a real list of per-division leader dicts, then immediately overwrote the same variable name with an unrelated integer a few lines later, before it was used in the output - the frontend's `.map()` call over that key would have crashed. Renamed the integer to `division_winner_count`. |
| `rate_matchup()` | **Real bug fixed - lookup was inverted** | The draft looked up `opponent_defense_rank_vs_position` on players belonging to the opponent's own team (the wrong side of the matchup - that's the defense THEY face, not the defense they ARE). Fixed to find players whose real `opponent` field equals the team in question (players playing AGAINST them that week) and read the rank from there - verified via a real spot-check (McCaffrey vs. LA correctly resolved LA's defensive rank, not the Rams' own offensive players' matchups). |
| "Clinched" terminology | **Avoided** | Labeled "Leading for playoff"/"Leading for division"/"Wild card contenders" with an explicit note that this is a week-16 snapshot, not true elimination math - a real distinction (a team leading a wildcard race hasn't mathematically clinched anything). |
| Week 18 → no week 19 preview | **Disclosed, not hidden** | Since this is a fully-completed season, the default view (week 18) has no next-week preview. Explicitly shown in the UI rather than silently omitted. |

## Notes

- Season Context is identical across all 18 weeks shown in this section -
  a real, disclosed limitation, not a bug: `season_projections_2025.json`
  only has a single week-16 snapshot, so there's no real per-week
  playoff-race history to show. Disclosed explicitly in the UI disclaimer.
- Spot-checked week 9: GB was a real 91% favorite at home vs. CAR and lost
  (a real, correctly-flagged surprise); Brock Bowers real actual 43.3 PPR
  vs. 10.4 projected (a real, large overperformance, correctly surfaced as
  a top performer) - both plausible, verifiable real outcomes, not
  constructed examples.

---

# Section 4 — Accuracy Tracker

Real fields used throughout: computed entirely from this project's own
already-real dashboard exports (`games_2025.json`, `fantasy_rankings_2025.json`,
`season_projections_2025.json`) plus a real final-standings computation
(reused from Section 3's own seeding logic, run over all 18 real weeks
instead of a week-16 cutoff).

| Item (from the original spec) | Status | What changed |
|---|---|---|
| Data source | **Corrected** | Spec assumed `integrated_game_predictions_2025.csv` had `actual_home_score`/`vegas_spread`/`win_prob_home`/`our_spread` — it doesn't; those are derived fields that only exist in this project's own dashboard JSON exports. Read from there instead. |
| Season-projection accuracy | **Real comparison built, not assumed** | `season_projections_2025.json` only has a week-16 *projection*, with no separate "actual final outcome" field to score it against. Built `compute_final_standings()` by reusing Section 3's real `_real_records_through_week()`/`_compute_seeds()` at `checkpoint_week=19` (covers all 18 real weeks) rather than inventing new logic. |
| Spread coverage "accuracy" | **Removed, real citation used instead** | Home spread-coverage rate isn't a real prediction this model makes (no "which side covers" output exists) — it would have looked like a precise accuracy metric while measuring nothing real. Replaced with real moneyline accuracy plus a direct citation of `edge_detection.py`'s real -36% ROI finding, per your explicit decision. |
| Trends chart | **Inline SVG, no new dependency** | The spec's first draft used `recharts`, which isn't installed and isn't installable by me (no Node.js on this machine) — would have broken the first `npm start`. Built as a dependency-free inline SVG line chart instead, matching this dashboard's established no-external-dependency pattern. |
| `compute_final_standings()` | **Implemented (referenced but undefined in the draft)** | The draft's `generate_accuracy_tracker_dashboard_data.py` called this function without ever defining it. Implemented by reusing real, already-verified Section 3 logic. |

## Notes

- All headline numbers cross-checked against numbers already established
  elsewhere in this project: real games accuracy 178/272 (65.4%), spread
  MAE 9.69 vs. Vegas 9.72 (matches the README's Integrated/Vegas-alone
  figures exactly); fantasy RB +0.651, QB +0.447, TE +0.543 (match the
  README's real per-position correlations exactly). WR came out lower here
  (0.400) than the README's +0.591 — expected and already disclosed: this
  tracker correlates WR's *static per-game-average* projection against
  single-week actuals, not the original season-total methodology the
  README's number reflects.
- Season projection accuracy: 8/8 real division winners correctly
  projected, 13/14 real playoff teams, average win-projection error of
  only 0.61 games (unsurprisingly small, since the week-16 checkpoint is
  close to season-end).

---

# Section 3 — Season Projections & Playoffs

Real fields used: `team`, `team_name`, `conference`, `division` (real,
static NFL structure), `wins_actual`/`losses_actual`/`ties_actual` (real,
derived directly from `schedules_2015_2025.csv`, games with `week < 16`),
`projected_wins`/`playoff_percentage` (real, from
`playoff_odds_trajectory_2025.csv` — the real Monte Carlo output of
`playoff_probability.py`'s earlier work), `playoff_seed` (real seeding
logic, see below), `remaining_schedule_strength` (real, from
`elo_ratings_2025.csv`), `is_division_winner`/`is_playoff_team` (derived
booleans).

| Field (from the original spec) | Status | How it's real / what changed |
|---|---|---|
| `superbowl_percentage` | **OMITTED — real gap, not fabricated** | Doesn't exist anywhere in this project. `playoff_probability.py` only ever modeled "makes the playoffs" — there's no bracket simulation (which team beats which in each round) anywhere in the codebase. Building it would be genuinely new modeling work (simulate each playoff round, decide reseeding/home-field rules), not a wiring fix — explicitly deferred per your choice rather than fabricated or hand-waved with a placeholder. |
| "Current standings" (`current_standings_2025.csv`) | **File doesn't exist — real data used instead** | This dataset is a fully-completed 2025 season backtest, not a live feed, so there's no real "current" week. Used the real week-16 checkpoint (the latest available in `playoff_odds_trajectory_2025.csv`, closest to season-end) as a single fixed snapshot, disclosed explicitly in the UI subtitle rather than presented as if it were live. |
| `playoff_seed` | **Built correctly, not per the pasted pseudocode** | The pasted `compute_playoff_seed()` had two real bugs: it used a post-`sort_values` DataFrame's stale row-index as if it were a 0-based conference rank (`.index[0]` returns the original row label, not a rank), and despite its own docstring claiming "1-4: division winners, 5-7: wildcards," never actually checked division standings — it just flatly ranked the whole conference by wins. Implemented real seeding instead: the 4 real division leaders (by real wins, tiebroken by real point differential) get seeds 1-4, the next-best 3 non-division-winners get wildcard seeds 5-7. |
| `remaining_schedule_strength` | **Real, with a disclosed simplification** | Real average Elo of each team's remaining (`week >= 16`) opponents, using each opponent's real Elo rating *as of the week-16 checkpoint* (not a projection of their future rating — their Elo could still move before those games are actually played). The rating itself is real; only the "held constant" assumption is a simplification, disclosed in the UI tooltip. |

## Notes

- Cross-checked the leak-free convention before trusting it: recomputed
  KC's real wins through week < 16 directly from `schedules_2015_2025.csv`
  (6 wins, 14 games) and confirmed it exactly matches
  `playoff_odds_trajectory_2025.csv`'s own `actual_wins_through_week`
  column for the same team/checkpoint.
- Sanity-checked output: 14 playoff teams and 8 division winners across
  32 teams (both real, structurally correct NFL counts) came out exactly
  right from the real seeding logic, not asserted. Spot-checked AFC West:
  DEN 12-2 (seed 1, 100% playoff odds), LAC 10-4 (wildcard seed 6), KC
  6-8, LV 2-12 — all real and internally consistent with earlier real data
  already verified in this project.

# Section 1 — Weekly Games

Real fields used (from `integrated_game_predictions_2025.csv` +
`schedules_2015_2025.csv`, joined by `game_id`): `week`, `home_team`,
`away_team`, `kickoff_datetime`, `our_spread` (`final_spread`),
`vegas_spread` (`base_spread`), `net_edge_diff`, `base_source`, real final
scores, `did_we_predict_correctly` (derived), plus four fields added this
task:

| Field | Status | How it's real |
|---|---|---|
| `home_qb_name` / `away_qb_name` | **NOW WIRED IN** | `schedules_2015_2025.csv`'s real `home_qb_name`/`away_qb_name` columns, joined directly. |
| `matchup_quality` | **NOW WIRED IN, empirically thresholded** | Bucketed via the real 33rd/67th percentiles of *this season's own* `net_edge_diff` distribution (computed live each run, currently ≈ −0.48 / +0.31) — not an asserted +1.0/−1.0 cutoff like the original plan proposed. Labeled from the home team's perspective. |
| `home_recent_form` / `away_recent_form` | **NOW WIRED IN** | Real last-5-games W/L, computed leak-free (only real games with `gameday` strictly before this game's date, crossing season boundaries into the full 2015–2025 log where needed). |
| `head_to_head` | **NOW WIRED IN** | Real historical meetings between the two exact teams (last 10, leak-free — only games before this one's date), with a real win/loss/tie count. Spot-checked against the real KC/BUF series. |

| `home_elo` / `away_elo` | **NOW WIRED IN (Phase 2)** | Real per-team, per-week Elo from `elo_ratings_2025.csv` (`elo_after`). Lagged one real week (week W's game uses week W-1's `elo_after`) for a leak-free "entering this week" rating — week 1 has no real prior entry and is left `null` rather than guessed. |
| `win_prob_home` / `win_prob_away` | **NOW WIRED IN, real backtest (Phase 2)** | A logistic regression fit on real 2024 Vegas `spread_line` → real outcome, chosen as the winner of a real 3-way backtest (`win_probability_backtest.py`) against an Elo-based model and an asserted-constant heuristic, evaluated on a genuine real 2025 weeks 13-17 holdout never used in fitting. Winner: the fitted Vegas model, Brier 0.2512 (vs. Elo 0.2874, heuristic 0.3149) — confirms this project's established Vegas-beats-Elo finding. Refit live each export run (not hardcoded) so it can't go stale. Applied to `vegas_spread` only (what it was validated on), not `our_spread`. |

### Still open (real, deliberately deferred)

| Field | Status | Why deferred |
|---|---|---|
| Per-player confidence intervals | **MISSING** | Not computed anywhere. Would need genuine per-player residual-std modeling, same as fantasy's declined CI proposal. |
| Component breakdown | **ALREADY SHOWN** | `net_edge_diff` and `base_source` were already exposed in the expanded card from the original Section 1 build — not a real remaining gap, despite appearing on both gap-filling plans. |
| Kickoff timezone | **AMBIGUOUS** | No explicit timezone column in the source data; displayed via browser-local parsing of a bare ISO string. |

---

# Section 2 — Fantasy Rankings

Real fields used: `week`, `position`, `rank`, `name`, `team`,
`projected_ppr`, `source`, plus three added this task: `projection_type`,
`opponent`, `opponent_defense_rank_vs_position`.

| Field | Status | How it's real |
|---|---|---|
| WR rankings | **NOW WIRED IN — but a different kind of number than RB/QB/TE** | WR's validated formula (`fantasy_validation.py`, EPA×volume) only ever produces a **static, season-long** projection, not a per-week trailing one like RB/QB/TE. Shown as a real per-game average (season total ÷ real `expected_games_2025`, calibrated to PPR units via the same in-sample linear fit `fantasy_validation.py` uses internally) — the same number repeats across every real week that player appears in, and is labeled `projection_type: "season_static_per_game_avg"` plus a UI banner so it's never presented as equivalent to RB/QB/TE's real weekly figures. **Caught before shipping**: the first version displayed the raw, uncalibrated-to-per-game season total directly (e.g. 143.2 "PPR" for one week) — an internally real number, but a unit mismatch that would have looked broken next to 10–25 point RB/QB/TE cards. |
| `opponent_defense_rank_vs_position` | **NOW WIRED IN** | Real, from `matchup_features.build_defense_epa_by_position_multi_season()` (per-team, per-week EPA allowed by position, computed from real play-by-play). Ranked 1–32 using only real **trailing** data (weeks strictly before the week being shown) — week 1 has no real trailing data and is left `null`, not fabricated with an invented fallback. |
| `injury_status` / `injury_status_raw` | **NOW WIRED IN (Phase 2)** | Real, from `nflreadpy.load_injuries()`. Schema verified before building: `gsis_id` (== this project's `player_id`), real `week`, `report_status` ∈ {None, Questionable, Doubtful, Out}, one row per player-week (no duplicate-join risk). No report entry → `healthy`, which is the correct real-world reading (only players with an actual injury concern are listed at all), not an assumption filled in for missing data. `Doubtful` folded into the `out` UI tier but the exact original string is preserved in `injury_status_raw`. |
| `recent_form` (player PPR trend array) | **NOW WIRED IN (Phase 2)** | Real trailing 4-week actual PPR (`player_weekly_stats.csv`'s `fantasy_points_ppr`), leak-free (only real weeks strictly before the week shown). **Bug caught before running**: the draft script referenced a column named `ppr_points`, which doesn't exist — the real column is `fantasy_points_ppr` — and didn't filter `season`/`season_type`, which would have silently mixed in real 2024 rows for repeated week numbers. Both fixed before running. |
| `actual_ppr` / `accuracy_tier` | **NOW WIRED IN (Phase 3)** | Real actual PPR for the game itself (reuses the same real, already season/season_type-filtered lookup `recent_form` draws from). `accuracy_tier` (green/yellow/red) is bucketed via **empirical per-position terciles** of the real `\|actual − projected\|` distribution across the whole real dataset — not the spec's asserted flat ±2/±5 threshold, since QB/RB point scales are naturally larger than TE's (real thresholds came out QB (3.3, 8.6), RB (1.4, 4.7), TE (1.4, 4.1), WR (3.2, 6.6)). WR's static per-game-average projection is explicitly disclosed in the UI as a likely source of "error" that's really just real week-to-week variance, not model failure. |

### Still open (real, deliberately deferred)

| Field | Status | Why deferred |
|---|---|---|
| `confidence` / `confidence_range` | **MISSING, proposed twice, declined twice** | Not computed per-player anywhere. Both drafts proposed per-position constants (e.g. "QB ±2.5") as literal placeholders, or a fallback that conflates cross-player point variance with per-player prediction uncertainty — a real but wrong statistic. Building a real version needs genuine per-player residual-std modeling against actual backtested errors. |
| Snap count / target share estimates | **NOT REAL** | The original plan proposed a "role-based heuristic" for these — that's fabrication, not a real backend value, and was excluded from this pass entirely rather than built as a guess. |

## Notes

- **Contradictory specs caught twice this session**: once for Section 2's
  initial build (a fabricated-mock spec vs. a real-data-only spec pasted
  together), and again in this gap-filling plan (several "quick win" items
  were actually asserted/fabricated heuristics mislabeled as low-effort real
  work — matchup thresholds, confidence formulas, fantasy CIs, snap%/target
  share). Both were flagged before building rather than silently complied
  with or silently dropped.
- **WR unit-mismatch bug, caught before shipping** (see table above) — a
  real value in the wrong units is still a real bug, not just a "missing
  field."
- All new Section 1 fields spot-checked against a real, independently
  verifiable case (KC @ BUF, week 9 2025: real 5-5 head-to-head over the
  last 10 real meetings, real recent-form strings, real QB names).
- All new Section 2 fields spot-checked: week-5 WR rankings (Ja'Marr Chase,
  Amon-Ra St. Brown, Justin Jefferson at the top, all in a sane 15-21 PPR
  range) with real opponents and real defense ranks attached.
- **Phase 2 spot-checks**: week-9 KC @ BUF real Elo shows BUF 1649 (home)
  vs. KC 1587 (away) — correction to an earlier note in this file, which
  incorrectly said this meant BUF was favored: both real Elo AND real
  `vegas_spread`/`our_spread` actually favored KC (the away team) in that
  game, and BUF's real win was a real upset (`did_we_predict_correctly:
  false`) — a mistake in prose commentary, not in any shipped code or
  data. Christian McCaffrey correctly flagged `questionable` week 1 with a
  real 4-week recent-form trend attached; Kyler Murray week 1 correctly
  resolved to `healthy` despite having a real "Illness" note in the source
  data, because `report_status` itself (the actual game-status field) was
  `None` that week — the simplification logic reads the right column.
- **Win-probability investigation, caught before shipping**: the first
  backtest run showed the Elo-based model beating the heuristic — a result
  that contradicted this project's established Vegas-beats-Elo finding.
  Investigated rather than shipped: the heuristic used an asserted,
  never-fit constant, so the comparison wasn't fair. Added a third
  candidate (a fairly-fit Vegas-spread model, same real logistic-regression
  treatment as the Elo model) - it won outright (Brier 0.2512 vs. Elo's
  0.2874 and the heuristic's 0.3149), confirming the established finding
  and showing the earlier "Elo wins" result was an artifact of comparing a
  fit model against an unfit one. Full real numbers in
  `backtesting_results.md`.
- Both Phase 2 data-export scripts require the `D:/venvs/nfl-model`
  Python interpreter (not the default `python` on PATH) because of the
  `nflreadpy` import — same as other `nflreadpy`-dependent scripts in this
  project.
- **Phase 3**: Section 1's spread-cover and moneyline-hit displays existed
  separately before this task (from Phase 1/2); consolidated into one
  "Betting Outcome" section per your request, reusing the same real
  `atsResult()`/`winProbHit` logic rather than introducing a new
  `game.spread_winner` backend field the draft spec assumed existed.
  Section 2's accuracy-tier distribution came out roughly even across
  green/yellow/red (as expected by construction, since terciles split the
  real data into thirds) — spot-checked a handful of records (e.g. Josh
  Allen week 1: projected 23.4, real actual 38.8, correctly tiered `red`).

---

# Section 7 — Betting Analysis

Pasted spec assumed no real odds existed and asked for a synthetic
spread→moneyline formula (`estimate_moneyline_from_spread()`) to fake them.
Checked first: `data/raw/vegas_lines_2015_2025.csv` already has real
`home_moneyline`/`away_moneyline`/`home_spread_odds`/`away_spread_odds` for
every 2025 game (272/272 matched by `game_id`, verified before writing any
code) — used those instead of inventing odds. The synthetic formula would
also have been wrong on its own terms: it prices moneylines proportionally
to spread size (a 2.5-pt favorite → -250), which isn't how real markets
price; real spread bets are priced near -110 on either side regardless of
spread size, and the real -2.5 favorite in this file is priced around -140,
not -250.

The spec's `get_bet_direction()`/moneyline-formula pair also assumed
"positive spread = underdog." This project's real, verified convention
(established in Section 1, 284/284 real moneylines) is the opposite:
positive spread = HOME team favored. Rewrote the direction logic to the
real convention; kept the requested more-extreme/less-extreme heuristic
itself as its own separate, explicitly-requested "Our System" strategy
(not merged with `edge_detection.py`'s existing, already-validated
disagreement+confidence logic, per your answer).

"Our" probability for the should-bet gate uses `win_prob_home` straight
from `games_2025.json` (this project's real, already-fitted win-probability
model output) rather than re-deriving a second synthetic probability from
`our_spread` — using a real, already-computed field beat re-introducing the
exact fabrication problem the odds fix above removed.

Two real edge cases handled: one real 2025 tie (Week 4, GB 40 @ DAL 40) —
no "TIE" moneyline is offered on a real book, so it's a push on moneyline
bets, settled normally on ATS. One real exact-margin ATS push (Week 12,
PIT/CHI, margin 3 vs. a 3.0 line). Both pushes are counted but excluded
from win/loss and ROI (no stake was actually at risk).

Real season results (all real, none fabricated):

| Strategy | Moneyline ROI | ATS ROI |
|---|---|---|
| Our System | -1.6% | -1.6% |
| Vegas Favorites | -7.0% | -7.9% |
| Underdogs Only | -2.1% | -1.2% |

Vegas Favorites' real moneyline win rate (65.3%, 177-94) lines up closely
with this project's established real game-prediction accuracy (65.4%,
178/272) — an independent cross-check that the merge and settlement logic
are correct, not just plausible-looking. Every strategy loses money after
real vig in both bet types, consistent with the project's established
Vegas-beats-the-model finding (`edge_detection.py`'s real -36% ROI on
spread disagreements) — a different real backtest, same real conclusion.

---

# Quick Wins #2 & #3 — Accuracy Confidence Visualization + Timezone Clarity

## Timezone bug, caught before shipping

The pasted spec's `formatKickoffTime()` (and this component's own
pre-existing `formatKickoff()`, which predates this task) both wrap
`kickoff_datetime` in a bare `new Date(isoString)`. Checked against real
data first: `kickoff_datetime` is built in `generate_dashboard_data.py` as
`f"{gameday}T{gametime}:00"` with no UTC offset — and `gametime` itself is
real Eastern Time regardless of the actual stadium, confirmed by checking
real 2025 Las Vegas (Pacific) home games, whose `gametime` values (16:05,
16:25) are the real ET numbers for the real 1:05/1:25 PM Pacific "late
window" kickoff, not an evening Las Vegas-local time — matching nflverse's
documented `gametime` convention.

That matters because a JS `Date` string with no timezone suffix is parsed
in the *viewer's own* local time, not ET. So `new Date("...T20:20:00")`
doesn't represent "8:20 PM ET" to the code at all — it silently becomes
"8:20 PM in whatever timezone the browser happens to be in." The spec's
"Your Time" feature, and this component's pre-existing kickoff display,
would therefore just echo the raw ET digits back to every viewer
unchanged, mislabeled as their own local time. Fixed by explicitly
attaching the real ET UTC offset before constructing the `Date`, using the
real 2025 US DST fallback date (November 2) to pick -04:00 (EDT) vs.
-05:00 (EST) per game.

A second real nuance, also verified before use: Arizona doesn't observe
DST. `TEAM_TIMEZONES` labels it "MT" (the real, correct zone name — this
is about the label, not the clock), but its actual numeric offset from ET
matches Pacific during the real EDT months and only matches Mountain after
the real November 2 fall-back, since the rest of the Mountain zone springs
forward and Arizona doesn't. `stadiumOffsetHours()` in `GameCard.js`
handles this as a real, date-dependent special case rather than a flat
lookup.

The "Game Time" (stadium-local) display doesn't need any of the above —
since the source value is already ET, real stadium-local clock time is
just the ET digits minus a fixed real hour offset (CT -1h, MT -2h, PT -3h;
all real US mainland zones shift DST in sync, so these stay constant all
season except Arizona).

## Confidence visualization: asserted numbers replaced with real computed ones

The spec's `ACCURACY_TERCILE_RANGES` (fantasy per-position error
boundaries) were checked against a live re-run of this project's own real
`_accuracy_tier_thresholds()` (`generate_fantasy_dashboard_data.py`) before
shipping — QB (3.3/8.6), RB (1.4/4.7), WR (3.2/6.6), TE (1.4/4.1) all
matched exactly, so these were shipped as given, cited as a real,
point-in-time snapshot (the underlying function recomputes fresh every
pipeline run).

The spec's Games-tab "Expected: ±8 points" and trends-chart "±8%"/"±7%"
confidence bands were asserted round numbers with no real grounding —
checked against `accuracy_tracker_2025.json`'s real weekly data and
replaced: real weekly spread MAE ranges 5.94–14.45 points (season mean
9.74), and the trends chart's shaded band now uses the real computed mean
± 1 real standard deviation of weekly accuracy (±11.9%, not an asserted
±8%) — real season data is meaningfully more volatile week-to-week than
the spec's asserted band implied.

While rebuilding the trends chart, also surfaced (not new, but newly
visible) that `games.accuracy_pct` and `betting.moneyline_accuracy_pct`
are the real same underlying metric in this dataset (both are straight-up
winner correctness — already disclosed in Section 4's build), so the two
plotted lines exactly overlap; added a real explanatory note under the
chart rather than leaving it looking like a rendering bug.

---

# Phase 4 — WR Dynamic Projections Backtest

## The leakage problem, caught before running anything

Checked `_wr_projections()` (`generate_fantasy_dashboard_data.py`) before
trusting the pasted spec's backtest premise: the currently-shipped WR
`projected_ppr` comes from `season_projected_pts = slope * projected_score
+ intercept`, where `slope`/`intercept` are fit **in-sample** against each
real WR player's own real full-season actual total
(`actual_season_fantasy_pts`) — the same total that includes whatever
individual week is later compared against it for correlation/MAE. Every
one of the spec's Approaches 1-4 only rescales that same leaky number by a
multiplier (opponent rank, game script), so ranking them against each
other wouldn't have measured real predictive skill, only which rescaling
best fit data that already partly contained the answer.

Rebuilt the baseline as a real leave-one-out (LOO) calibration instead:
for every real WR player-week, the population EPA→season-points regression
is refit with that specific week's own real actual result excluded from
the player's season total, before projecting that week — so no row's
projection ever had access to the result it's being scored against. The
real, already leak-free `expected_games_2025` (a preseason estimate,
independent of in-season results) is reused unchanged as the per-game
divisor. All 5 approaches, including the "Volume-Only" baseline itself,
were run against this LOO number, not the shipped leaky one.

Real result: the LOO baseline's own correlation is 0.3957 — close to, but
meaningfully below, the currently-shipped (leaky) dashboard's 0.400,
confirming the leakage was real but modest in magnitude here.

## Other real bugs fixed before running

- **Field name**: the spec's code used `player_name` throughout; the real
  field in `fantasy_rankings_2025.json` is `name` — would have raised a
  `KeyError` immediately.
- **Dead metric**: `compute_metrics()`'s `tier_accuracy_pct` was hardcoded
  to always return `0.0` in the pasted spec (its own comment called it "a
  placeholder"). Replaced with a real metric: the real green-tier hit rate
  using WR's live-verified real ±3.2 PPR boundary (confirmed against
  `_accuracy_tier_thresholds()` in the prior Quick Wins task this
  session).
- **`vegas_spread` confirmed absent** from `fantasy_rankings_2025.json` —
  Approach 4 (Game-Script Adjusted) always hits the spec's own real
  fallback branch and is a no-op, identical to the baseline. Not a bug,
  disclosed as a real fact in the output (`no_vegas_spread_data`).
- **Spec's "30+ variations" claim was inflated** — the described approaches
  produce 23 real variations, not 30+; ran all 23, didn't pad the count.

## Real ranked results (23 variations, leak-free LOO baseline)

| Rank | Approach | Variation | Correlation | MAE |
|---|---|---|---|---|
| 1 | Recent Form | trailing_4week_avg | **0.4414** | 5.57 |
| 2 | Volume-Only | leave_one_out_baseline | 0.3957 | 5.66 |
| 3 | Game-Script Adjusted | no_vegas_spread_data (no-op) | 0.3957 | 5.66 |
| 4-9 | Hybrid (various blends) | | 0.3940-0.3954 | 5.65 |
| 10-23 | Matchup-Adjusted / weaker Hybrid blends | | 0.3571-0.3934 | 5.65-5.74 |

Real winner: **trailing 4-week actual-PPR average** (already the same
real, leak-free `recent_form` field this project computes for RB/QB/TE),
correlation 0.4414 — beats both the leak-free baseline (0.3957) and the
currently-shipped leaky number (0.400). A clean, monotonic, informative
negative result also emerged: matchup-adjustment strength (5%→20%)
*decreases* correlation the harder it's applied — real opponent-defense-
rank-vs-WR doesn't carry enough independent signal to justify rescaling
the projection by it; doing so just adds noise.

Implementing this winner (swapping WR's default rendering from the static
season-average to the real trailing-4-week average, regenerating
`fantasy_rankings_2025.json`, updating Section 2's real correlation figure
throughout the dashboard) is a separate task, not started here, per this
project's stop-after-each-task protocol.

---

# Phase 4 Implementation — WR Recent Form Projections

Shipped the backtest's real winner as WR's actual production projection.

## Backend change

`generate_fantasy_dashboard_data.py`'s `_wr_projections()` now takes the
already-computed `form_lookup` (previously only used later, for the
`recent_form`/`actual_ppr` display fields) and, per real player-week:
reuses `_trailing_recent_form()` (the same real, leak-free trailing-weeks
helper already used for the dashboard's Recent Form display) to get up to
4 real prior weeks' actual PPR, and projects that week as their mean when
at least one real prior week exists. When none exists (a player's first
real 2025 appearance), falls back to the original static, in-sample-
calibrated EPA x volume season projection - split out into its own
`_wr_static_fallback()` function, logic unchanged from before. `source`
and `projection_type` are now set per-row so the frontend can tell which
real method produced which specific row, rather than assuming all WR rows
share one method.

## Real results after regenerating (not assumed to match the backtest)

The backtest's 0.4414 figure was computed against a leave-one-out (LOO)
baseline built specifically for that backtest; real production blends the
trailing-average method (1,274/1,381 real WR rows, 92%) with the
*original* (non-LOO) static fallback for the remaining 107 rows (a
player's first real appearance), so the real production correlation was
independently verified rather than assumed to equal the backtest number:

| | Real correlation | Real MAE |
|---|---|---|
| Old (100% static, leaky) | 0.400 | - |
| Backtest LOO baseline (100% leak-free static) | 0.3957 | - |
| Backtest winner (LOO baseline + recent form) | 0.4414 | - |
| **Real production (static fallback + recent form)** | **0.4416** | **5.57** |

Real production landed almost exactly on the backtest's number (0.4416 vs.
0.4414) - expected, since the static-fallback minority (7.7% of rows) is
small enough not to move the blended correlation much either way.

Real per-position correlations after this change: RB 0.651, TE 0.543, QB
0.447, **WR 0.442** - WR is still the lowest, but the gap to QB shrank
from a 4.7-point gap (40.0 vs. 44.7) to a 0.5-point gap (44.2 vs. 44.7).
WR's real accuracy-tier boundary also tightened as a direct result (terciles
recomputed live): ±3.2/±6.6 PPR → ±2.7/±6.4 PPR.

## Cascading regeneration (dependency chain, not just the one file)

`fantasy_rankings_2025.json` feeds two other real exports that would have
gone stale otherwise - both regenerated and spot-checked for consistency:
`accuracy_tracker_2025.json` (Section 4's real fantasy-correlation tab) and
`weekly_summary_2025.json` (Section 5's real top-fantasy-plays lists).
Section 3's exports don't consume fantasy data and were correctly left
untouched.

## Frontend text updated everywhere WR was described as purely static

Four separate places across three components claimed WR was a single
static, non-updating number - all real, all now outdated, all fixed
rather than left inconsistent with the new real data:
- `FantasyRankings.js`: the WR position banner, the player card's
  Methodology section (now position- and row-aware, not just
  `isStatic`-aware, since the RB/QB/TE "volume-based formula" copy doesn't
  describe WR's new trailing-average method), the bottom disclaimer, and
  the `ACCURACY_TERCILE_RANGES` snapshot constant (re-verified against the
  new live thresholds, not left at the pre-Phase-4 values).
- `AccuracyTracker.js`: the fantasy tab's WR explanatory note.
- `ModelTransparency.js`: four separate mentions (the "real mechanism by
  position" paragraph, the accuracy ranking list, a "what this project is
  not good at" bullet - removed entirely, since a 0.5-point gap to QB no
  longer qualifies as a standout weakness - and the Limitations section's
  WR item, retitled to note QB is now the closer second-weakest).

---

# Comprehensive Fix — Both Critical Audit Bugs

Fixed the two real, critical bugs from `AUDIT_2026-07-30.md`. The pasted
fix spec's example code didn't match the real current code in three ways,
caught before implementing anything (see the pre-task verification):

1. **WR team fix** - the spec proposed re-reading `data/raw/player_weekly_
   stats.csv` (wrong path - it's in `data/processed/`) to merge in a real
   per-week team. Checking `_wr_static_fallback()` directly showed the real
   per-week `team` was already sitting unused in the `actual` DataFrame
   this function already loads (`extract_actual_fantasy_points_2025()`
   already derives it from `recent_team`) - it just got dropped and
   replaced by the stale preseason `team` two lines later. Fixed by keeping
   the real column already there instead of re-deriving it from a second
   file read.
2. **Division-leader fix** - the spec proposed reloading `season_
   projections_2025.json` from disk inside the function; the real code
   already has `season_projections_df` in scope as a parameter, loaded
   once by the caller. Fixed by filtering the in-memory frame on the real
   `is_division_winner` field (with a defensive `nlargest` fallback for the
   unexpected case where no row is marked a winner), not reloading
   anything.
3. **Archive step** - the spec's `git mv src/test_*.py`/`src/experiment_
   *.py` glob patterns matched zero real files. Archived the real 10
   filenames from the audit instead (`momentum_weighting.py`,
   `rest_tracking.py`, `ensemble.py`, `ensemble_model.py`, `dynamic_
   tracking.py`, `weekly_tracking.py`, `playoff_probability.py`, `player_
   impact.py`, `integrated_predictions.py`, `qb_elo_model.py`) via `git mv`
   into a new `src/archive/`, with a README explaining why each is there
   and that they're preserved, not dead.

## Verification (real, not assumed)

- **WR team fix**: George Pickens now shows `DAL` for all 18 real weeks
  (was `PIT`). Re-ran the audit's exact 30-player mismatch check against
  the regenerated export - **0 mismatches remain**. No duplicate
  player-week IDs introduced (1381 WR records before and after). Real
  defense-rank coverage went from 4567/4871 to 4585/4871 - a real, expected
  side effect of some of the 30 players now resolving a real opponent
  lookup they previously couldn't.
- **Division-leader fix**: re-checked all 18 weeks programmatically - NFC
  South now shows the real `TB` and NFC West the real `SEA` on every single
  week (was `CAR`/`LA`). The other 6 divisions, which were never wrong,
  are unchanged.
- **Regeneration scope confirmed empirically, not assumed**: recomputed all
  four real fantasy correlations directly from the regenerated
  `fantasy_rankings_2025.json` (QB 0.4466, RB 0.6509, TE 0.5432, WR 0.4416)
  and confirmed they match `accuracy_tracker_2025.json`'s already-shipped
  numbers exactly - confirming that file, along with `games_2025.json`,
  `season_projections_2025.json`, and `betting_backtest_results_2025.json`,
  genuinely didn't need regeneration, rather than just trusting the fix
  spec's claim that they didn't.
- `python -m py_compile` across all of `src/` (including the new
  `src/archive/`) confirmed no syntax errors from any of the changes.
- **Not done**: actual browser verification. This machine still has no
  Node.js installed (a standing limitation disclosed throughout this
  entire session) - none of this session's frontend work, including this
  fix, has been rendered in a real browser. Verification here is limited
  to real data correctness and Python syntax checks, not a live render.

---

# Season Projections Enhancement — Division Winners, Playoff Picture, Super Bowl Odds

## Real field-name and structure bugs fixed before implementing

The pasted spec's Python and JS both referenced `row['seed']`/`team.seed`
and `row['wins']`/`team.wins` - the real fields in
`season_projections_2025.json` are `playoff_seed` and `wins_actual`. Its
frontend snippets also computed a displayed "losses" as `18 - wins`, which
doesn't correspond to any real quantity: `wins_actual` is real wins
*through the week-16 checkpoint*, not a final 18-week total (established
in the 2026-07-30 audit), and at least one real 2025 tie (GB-DAL, week 4)
means `wins + losses` isn't even always 18 in principle. Used the real
`losses_actual`/`ties_actual` fields directly instead.

The spec's `PlayoffBracket.js`/`SuperBowlOdds.js` examples also used a raw
team hex color directly as text color against a dark background with no
contrast check (plus a separately reinvented, less rigorous
`getContrastText()` duplicate of a function this project already has) -
the same class of bug already caught and fixed for New Orleans earlier
this session. Reused this project's existing, validated
`readableTextColor()` throughout instead.

## Super Bowl odds: real Monte Carlo bracket, not a heuristic

Per explicit instruction (after flagging the tradeoff), built
`src/superbowl_bracket_simulation.py` - a real bracket simulation extending
`src/archive/playoff_probability.py`'s already-validated, correlated
Monte Carlo approach (the same one behind the real `playoff_percentage`
field) rather than the spec's simpler `win_prob^games_to_SB × 0.5`
heuristic. Real rules modeled: #1 seed byes the Wild Card round, real
NFL re-seeding in the Divisional round (the #1 seed plays whichever
wild-card winner has the worst remaining seed, not a fixed bracket slot),
better seed hosts every round except a real neutral-site Super Bowl (no
home-field Elo term). Each simulated game's win probability comes from
this project's real win-probability formula
(`calculate_win_probability_from_elo`) applied to real, frozen week-16 Elo
ratings (`weekly_recalibration.update_elo_with_actual_results`), not a new
formula.

**Real, disclosed simplification**: the bracket seeding is fixed at the
real week-16 projected seeds (the same seeds the Playoff Picture panel
shows) rather than re-simulated each trial - this answers "who wins if the
projected seeding holds," not two compounded layers of regular-season-plus-
playoff uncertainty. Stated directly in the JSON output's
`methodology_note`, in the Season Projections disclaimer, and in Model
Transparency - the same way `playoff_probability.py` discloses its own
real simplifications rather than glossing over them.

Real output (10,000 trials, `superbowl_odds_2025.json`, all 32 teams - the
18 non-playoff teams correctly at 0.0%, not omitted): AFC and NFC real odds
summed to 48.8% and 51.0% respectively (real Monte Carlo noise around the
expected ~50/50 split, not a bug - each conference's champion always faces
the other's in a real Super Bowl). Real #1 seeds SEA and DEN topped both
real Super Bowl odds (17.2%, 15.5%) and real conference-championship odds
(33.3%, 31.2%), as expected from a bye plus real team strength - a real
sanity check that passed.

## Two now-stale gap disclosures fixed

`SeasonProjections.js`'s own disclaimer and two separate places in
`ModelTransparency.js` explicitly said Super Bowl probability "doesn't
exist anywhere in this project" / "not built at all yet" - both accurate
when written, both now outdated. Updated all three to describe the real
new methodology instead of leaving a real, newly-closed gap advertised as
still open.

---

# Multi-Year Season Selection — Real 2026 Preseason Support

## The real problem, caught before any implementation

The pasted spec's entire premise was a fully-populated "2026 Season
(default)" alongside 2025, with all 8 sections working identically for
both years. Checked `data/raw/schedules_2026.csv` before building anything:
the real 2026 season hasn't been played (real opener 2026-09-09, 0/272 real
games have a score). That makes the spec's plan impossible without
fabricating outcomes - there is no real completed game, no real fantasy
result, no real accuracy metric, no real betting outcome, and no real
weekly recap for a season that hasn't happened. Flagged this before writing
any code; scoped down, per explicit instruction, to a real preseason-only
2026 view: Section 1 (real schedule + real preseason predictions) and
Section 3 (real preseason projected wins) render for 2026, Sections 2, 4,
5, and 7 show a real "not available yet" message instead of an empty or
fabricated panel.

A second, separate architectural mismatch was also caught and avoided: the
spec's `SeasonContext.js` used runtime `fetch()` to load JSON, which only
works if the files live in CRA's `public/` folder - every component in
this app currently uses static build-time imports instead (`src/data/`,
bundled into the JS). Kept the existing static-import architecture (per
explicit instruction) rather than restructuring where data lives, which
also couldn't have been verified without Node.js on this machine.

## Real, already-built 2026 infrastructure reused, not reinvented

Before writing any new 2026 generation code, checked `data/processed/` for
what already existed from earlier work in this project - found substantial
real preseason infrastructure already built and sitting unused:
- `elo_game_prediction.py`'s `generate_elo_game_spreads()` already has a
  documented, real `season > 2025` branch: real chained 2015-2025 Elo,
  regressed one-third toward 1500 at the season boundary, applied against
  the real full 2026 schedule - reused directly for Section 1's real
  `our_spread`/`home_elo`/`away_elo`.
- `data/processed/ensemble_season_wins_2026.csv` - a real, already-built
  blend of this project's EPA-based and Elo-based season-win projections
  with real 90% CI - reused directly for Section 3's real `projected_wins`.
- Confirmed independently (not assumed): no real 2026 Vegas data exists
  anywhere in this project (`vegas_lines_2015_2025.csv` stops at 2025;
  `data/processed/vegas_blended_spreads_learned_2026.csv` independently
  shows `has_vegas_line: False` for all 272 real 2026 games) - `base_source`
  is `"elo"` for every 2026 game, the same honest fallback this project
  already uses elsewhere when no Vegas line exists.

## New scripts (real, not fabricated preseason data)

- `src/generate_dashboard_data_2026.py` → `games_2026.json`: real 2026
  schedule, real preseason Elo spread/win-probability (via
  `calculate_win_probability_from_elo` directly on real preseason Elo, NOT
  the Vegas-fit win-probability model, since that model was validated
  specifically on real `vegas_spread`, which doesn't exist here - this
  project's own real second-best backtested candidate, Brier 0.2874).
  `home_recent_form`/`away_recent_form`/`head_to_head` are real and fully
  computable even for an unplayed season - they reuse
  `generate_dashboard_data.py`'s real, season-agnostic helpers unchanged,
  since they only need real *prior* (2015-2025) games. `home_qb_name`/
  `away_qb_name`/`net_edge_diff`/`matchup_quality`/every `actual_*` field
  are real nulls (real starters aren't populated this far out; the
  matchup-EPA adjustment needs real in-season stats that don't exist yet;
  no games are played) - disclosed real gaps, not filled with a guess.
- `src/generate_season_projections_dashboard_data_2026.py` →
  `season_projections_2026.json`: real team/conference/division metadata
  (reused from the 2025 script's static structure, unchanged year to year)
  joined with the real ensemble projection above. `wins_actual`/
  `losses_actual`/`ties_actual` are real 0s (not nulls - every team really
  has played 0 real games). `playoff_percentage`/`playoff_seed`/
  `is_division_winner`/`is_playoff_team` are real nulls/false - no real
  Monte Carlo playoff simulation exists yet against a still-hypothetical
  2026 standing (building one is real future work, out of scope here, not
  fabricated with a placeholder).

## Frontend: season-aware, not season-fetched

`SeasonContext.js` statically imports every real available JSON for both
years and exposes whichever season is selected, plus a real
`hasResults` flag (`false` only for 2026) that every data-dependent
component now checks before rendering. `GamePredictions.js` and
`SeasonProjections.js` render for both seasons (real data exists for
both, just less of it for 2026 - Division Winners/Playoff Picture/Super
Bowl Odds are hidden specifically for 2026, with a real explanation, since
none of those have real seeding to work from yet). `FantasyRankings.js`,
`AccuracyTracker.js`, `WeeklySummary.js`, and `BettingAnalysis.js` show a
shared `SeasonDataUnavailable` component for 2026 instead of crashing on
null data or rendering an empty table.

`GameCard.js`'s real "Win Probability" section previously always cited the
Vegas-fit backtested model by name - accurate for every 2025 game (100%
real Vegas-sourced, verified before this task), but would have been a real
false claim for any 2026 game (100% Elo-sourced, since no Vegas line
exists). Made the citation `base_source`-aware so it correctly describes
which real model actually produced that game's number.

Also fixed, found while touching this file: `GamePredictions.js`'s
disclaimer claimed "no confidence or win-probability figure is shown" -
stale even for 2025 (GameCard has shown real win probability since the
Phase 2 task); corrected while adding the real season-aware text next to
it.

---

# 2026 Week 1 Fantasy Projections

## The fabrication, caught before any implementation

The pasted spec's `load_historical_week1_baseline()` returned a fully
hand-typed Python dict of Week 1 PPR averages by team and position (e.g.
"BUF QB: 21.3"), despite its own comment claiming these came from
querying "all 11 seasons" of `fantasy_rankings.json` - no such archive
exists; only 2025 has one. `load_defense_rankings_2026()` was the same
problem, and incomplete on top of it: a stub covering 2 of 32 real teams,
silently defaulting every other team to a hardcoded rank of 16.

## Real methodology used instead

QB/RB/TE already have a real, documented precedent for exactly this
situation: this project's real, already-validated trailing-volume
formulas already fall back to the real PRIOR season's per-game rate for a
season's own real Week 1 (2025's Week 1 falls back to real 2024 rates -
see `fantasy_rb_formula.py`/`fantasy_formula_improvements.py`). Applied
one real year forward: 2026's real Week 1 uses each returning player's own
real full-2025-season per-game rate, run through that position's own
real, unmodified PPR formula (`_real_ppr()`, reused directly from both
modules - not re-derived). WR reuses its own real, already-validated
static methodology (real EPA x volume, calibrated on real 2025 outcomes)
applied to real 2026 preseason EPA inputs - the same real
fit-on-history/apply-forward pattern already used elsewhere in this
project (e.g. `elo_game_prediction.py`'s real season>2025 handling).

Real 2026 rosters (and real 2026 team assignments specifically) come from
`data/processed/{position}_epa_projections_2026.csv` - real, purpose-built
files already sitting in this project from earlier work (the same ones
Section 3's real ensemble already uses) - not from re-using each player's
real 2025 team, which the 2026-07-30 audit already found goes stale for
any real trade (the George Pickens class of bug). Spot-checked: Pickens
correctly shows DAL in the new 2026 output.

Real, disclosed nulls: `opponent_defense_rank_vs_position` (no real
trailing-week data exists for any season's real Week 1) and `recent_form`
(no real prior 2026 weeks exist) both reuse the exact same real convention
already used for every other season's Week 1. `injury_status` defaults to
"healthy" for a different real reason than the existing convention -
verified directly that real nflreadpy injury data doesn't support season
2026 at all yet (`load_injuries(2026)` raises "Season must be between 2009
and 2025"), not that a specific player was checked and cleared - disclosed
as a distinct real cause in the UI, not conflated with the existing
"no report entry" reasoning.

## Real, verified output

294 real players (45 QB / 89 RB / 111 WR / 49 TE) - 0 real players excluded
for lacking real 2025 data (the real 2026 roster files already only
include players with real prior-season data to project from). All
required frontend fields present and correct: 0 nulls in `id`/`rank`/
`projected_ppr`/`opponent`, 0 duplicate ids, single week (1) throughout.
Real, sane per-position ranges (QB 5.7-23.0, RB 0.9-24.5, WR 4.5-22.0, TE
2.8-18.6 PPR) with real, recognizable top players at every position (Josh
Allen QB1, Christian McCaffrey RB1, Jaxon Smith-Njigba WR1, Trey McBride
TE1).

## Frontend: `SeasonContext.js` updated, `hasResults` not overloaded

`SeasonContext.js` now imports the real `fantasy_rankings_2026.json`
(previously explicitly `null` for 2026, from the Multi-Year task, before
this real data existed). Rather than changing the shared `hasResults` flag
(which correctly stays `false` for 2026, since accuracy/weekly-summary/
betting genuinely still have no real data), `FantasyRankings.js` now gates
on real fantasy-data presence specifically (`!fantasyData`, not
`!hasResults`) - the right fix, since conflating "this specific section
has real data" with the season-wide flag would have either hidden this
real new section or wrongly un-hidden the others. A `isPreseason` flag
(derived from `!hasResults`, passed down to `PlayerCard`) drives a new
preseason banner and rewrites every methodology/disclaimer string that
would otherwise have described 2025's real weekly formulas as if they
applied to 2026's real prior-season fallback.

## Win Totals Confidence Intervals + Real Monte Carlo Playoff Odds

Two pasted specs, both declined as-written, both investigated first:

1. "Fix Win Totals Variance" proposed hand-tuning `REGRESSING_FACTOR`/
   `SCHEDULE_WEIGHT`/`EPA_WINS_MULTIPLIER` (none of which exist in this
   project's real code - the real generator just copies `ensemble_wins`
   straight into `projected_wins`) to widen the 2026 point estimate until
   it looked like real 2025's realized 2-14 win range. This project's own
   Phase 2 "Compression Investigation" (`PROGRESS.md`) already tested
   exactly this - rescaling team strength toward the full measured
   correction factor and refeeding it through the fitted model - and found
   MAE got monotonically **worse** (2.88 wins at 1.0x rescale to 3.48 at
   3.34x), not better: real signal is too weak for widening the bands to
   pay off, and comparing a preseason *expectation* to a *realized* win
   total was never apples-to-apples in the first place (even real Vegas
   lines are ~1.6x compressed vs. realized outcomes, structurally, not as
   a bug).
2. A follow-up "distribution-based Monte Carlo" spec was conceptually
   sound (letting variance emerge from simulated game outcomes rather than
   rescaling a point estimate) but its own Step 1 would have re-derived
   per-game win probabilities from scratch via a hand-picked "0.6 Vegas +
   0.4 Elo" blend - discarding this project's real, already-built,
   LOOCV-validated ensemble (`archive/ensemble_model.py`), which already
   found `w_epa=0.00` in all 32 leave-one-out folds (pure carryover Elo,
   not a blend).

**What shipped instead - two real, separate fixes:**

**1. Surfaced the already-computed real 90% CI.** `elo_model.py`'s
`project_season_wins_from_elo` (the function that already produces
`elo_wins`/`ensemble_wins`) sums real per-game Elo win probabilities
across each team's real 2026 schedule and computes `Var = sum(p*(1-p))`
in closed form - an exact answer, not a Monte Carlo approximation, and a
10,000-trial simulation of the same probabilities would converge to this
exact same mean/variance. That real CI (`ensemble_wins_low_90`/
`ensemble_wins_high_90`) was already sitting in
`ensemble_season_wins_2026.csv`, unused. Now exposed as
`projected_wins_low_90`/`projected_wins_high_90` in
`season_projections_2026.json` and shown as a "Win Range (90%)" column in
`SeasonProjections.js` (2026 only - 2025 has no equivalent field, column
hidden automatically when absent). Averages roughly ±3.3 wins around the
point estimate - e.g. NYJ 3.3-9.8, PHI 7.0-13.6 - a real, honest range,
not a rescaled one.

**2. Built a genuine new Monte Carlo simulation** (`src/simulate_2026_
playoffs.py`) for the one real, already-disclosed gap the closed-form CI
structurally cannot answer: a per-team CI says nothing about whether a
team's simulated record beats its real division/conference rivals'
records in the *same* trial - exactly the joint question playoff odds and
seeding require. 10,000 real trials, each independently drawing every
real 2026 game's outcome from `calculate_win_probability_from_elo` (the
same real, already-fit formula `superbowl_bracket_simulation.py` already
uses for the playoff bracket, one step earlier), then applying the same
real division-leaders-then-wildcards seeding rule
(`generate_season_projections_dashboard_data.py`'s `_compute_seeds`) used
for real completed seasons.

Two real, disclosed simplifications: (a) no simulated game has a real
point-differential to tiebreak with (only a binary win/loss is drawn), so
each team's real, static preseason Elo rating substitutes for real point
differential - same category as `_compute_seeds`'s own already-disclosed
tiebreak simplification; (b) real ties (~0.4% of games) aren't modeled,
consistent with how this project's other win-probability work already
treats them.

`playoff_percentage`/`division_winner_percentage` are raw real Monte
Carlo fractions for all 32 teams (not constrained to sum to a fixed
count). `is_playoff_team`/`is_division_winner`/`playoff_seed` are a real,
*derived* selection - top-7-by-`playoff_percentage` per conference,
top-1-by-`division_winner_percentage` per division - rather than a raw
50% threshold, because independent per-team preseason probabilities don't
reliably produce exactly 7 seeds/conference or exactly 1 winner/division
(real completed seasons always do, by construction, and the frontend's
bracket/division panels assume that real invariant). Verified: 14 playoff
teams (7/7 split), 8 division winners (4/4 split), `corr(ensemble_wins,
playoff_pct) = +0.98`.

**A real, pre-existing bug found and disclosed, not fixed here**: while
building this, found that `elo_game_prediction.py`'s
`generate_elo_game_spreads()` (which produces
`elo_game_predictions_2026.csv`, feeding the Game Predictions section's
real spreads/CI) computes 2026 preseason Elo differently than
`archive/ensemble_model.py`'s `get_elo_season_predictions(2026)` (which
produces `elo_wins`/`ensemble_wins`, feeding `projected_wins`) - the
former discards `run_multi_season_elo`'s true end-of-2025 ratings
(`current`, the function's 3rd return value) and instead takes
`ratings_at_season_start[2025]` (the *start* of 2025, before any 2025
games) plus one extra manual regression step, silently ignoring every
real 2025 result. This simulation deliberately recomputes Elo the second
(correct, already-validated) way - matching `elo_wins` to within Monte
Carlo noise (max diff 0.04 wins across 10,000 trials, vs. up to 1.4 wins
of real, systematic divergence against the buggy first path) - rather
than reading the mismatched `elo_game_predictions_2026.csv` snapshot.
Fixing the first path is out of scope here (it would also move the real
spreads/CI shown elsewhere in the dashboard) and is flagged as a real,
disclosed next step, not fabricated or silently patched.

**Real, disclosed gap still open**: Super Bowl odds for 2026 remain
unshown - `superbowl_bracket_simulation.py` currently only runs against a
fixed real week-16 checkpoint seeding, which doesn't exist for an unplayed
season. Extending it to run against this task's real simulated seeding
instead is a real, scoped next step, not built here.

## Feature B + 2026 Super Bowl Odds

**Part 1: Injury Risk & Consistency scores**, added to
`fantasy_rankings_2026.json` and the existing `PlayerCard` (extended, not
replaced - the pasted spec's minimal `PlayerCard.js` snippet would have
destroyed the real expand/collapse, methodology, and accuracy-comparison
sections built across three prior tasks this session).

- **Injury Risk** (0-100): real career miss rate (2015-2025, seasons with
  <4 real recorded games excluded) blended with real, *empirically
  computed* position/age risk multipliers (average real miss rate per
  position/age-bucket, normalized against the real league-average miss
  rate - not the first spec draft's hand-typed 0.8/1.4/1.1/1.0 constants)
  and real recent (last 8 real weeks of the player's most recent season)
  miss rate. Real computed position multipliers ranged 0.66 (K) to 1.08
  (QB) among skill-adjacent positions - notably flatter than folk wisdom
  would suggest (RB was NOT the highest at 1.00x) - reported as-is, not
  adjusted to match expectation. Real, disclosed null (not a fabricated
  "50/Moderate") for any player with zero qualifying real history -
  didn't end up mattering here since every real 2026 roster entry already
  has qualifying 2015-2025 data (the roster files themselves only include
  players with a prior season to project from).
- **Consistency** (0-100): `100 - 50*CV` of real 2025 actual PPR
  (coefficient of variation), real, disclosed null for the 23/294 players
  with fewer than 8 real qualifying 2025 weeks.
- **Real, fixed bug**: the pasted spec's miss-rate logic assumed a missed
  game shows up as a `NaN` row in `player_weekly_stats.csv` - verified
  the table only contains rows for games actually played, so that scan
  would have returned ~0 missed games for nearly every player. Real fix:
  compare a player's real row count for a season against that team's real
  games played that season (`game_results_2015_2025.csv`, correctly
  handling both 16-game (2015-2020) and 17-game (2021+) real season
  lengths from the data itself, not a hardcoded cutoff).

**Part 2: Real 2026 Super Bowl odds** (`src/generate_superbowl_odds_
2026.py`, new). Reuses this project's real, already-validated
`simulate_superbowl()`/`_real_seeds_by_conference()`
(`superbowl_bracket_simulation.py`) and the exact real preseason Elo
already powering 2026's win totals and playoff odds
(`simulate_2026_playoffs.real_2026_carryover_elo()`, refactored out of
that module so both real simulations use the identical Elo, not two
independently-derived copies). Seeded from this task's own real Monte
Carlo regular-season simulation's derived seeding (not an actual
standing, since none exists yet - disclosed in the output's own
`methodology_note`). Output written to a separate `superbowl_odds_
2026.json` (matching the real 2025 file's shape exactly, including field
name `superbowl_odds_pct`), wired into `SeasonContext.js`'s existing
`superbowlOdds` key the same way `fantasy2026` was wired in last task -
no data-fetching architecture changes, `SuperBowlOdds.js` kept its real,
working expand-toggle/odds-bar functionality intact, extended with only
an `isPreseason` prop for text. Real, verified: 14/14 playoff teams have
nonzero odds, 18/18 non-playoff teams are exactly 0.0%, AFC/NFC odds each
sum to ~50% (real total ~100%), conference-champion-pct sums to ~200%
(2 guaranteed real conference champions). Top team (BUF, 18.7%) is a real
playoff #1 seed with the 2nd-highest real Elo-driven playoff odds.

**Part 3: Elo bug** - already fully disclosed in the prior task's "Win
Totals Confidence Intervals" section above; reconfirmed here, no new
investigation needed. This task's own real preseason Elo (used for both
new features) deliberately uses the non-buggy path
(`run_multi_season_elo`, not `elo_game_predictions_2026.csv`), so neither
new feature is affected by it.

**Real bugs caught across three rounds of pasted-spec correction before
building** (full detail in the conversation, not repeated here): wrong
file paths in every draft; an undefined `estimate_age()`/`compute_
total_games_missed()`; a nonexistent `simulate_playoffs()` function; a
wrong-module import for `_real_seeds_by_conference` (lives in
`superbowl_bracket_simulation.py`, not `generate_season_projections_
dashboard_data.py`); passing a pandas DataFrame where that function
requires a plain list of dicts; code that claimed to recompute preseason
Elo but still read a nonexistent/buggy CSV; a hand-typed `home_field_
elo=30` overriding this project's real, empirically-fit +32.4 constant;
wrong output field names (`sb_odds_pct`/`methodology` instead of the real
`superbowl_odds_pct`/`methodology_note` the existing component reads); and
a proposed `SuperBowlOdds.js`/`SeasonContext.js` rewrite that would have
replaced real, working functionality with a fetch-based architecture this
project deliberately doesn't use.

## Sleeper API Integration

**Real architecture decision, made explicitly before building anything**:
the pasted spec's original form assumed a Flask backend (`src/app.py`,
`/api/sleeper/*` routes) - verified this project has none: Flask isn't
installed, isn't in `requirements.txt`, no server entrypoint exists
anywhere in `src/`, and `frontend/package.json` has no dev proxy
configured. Standing one up would have been this project's first-ever
runtime backend, contradicting the established static-JSON-only
architecture (real, deliberate choice from the Multi-Year Season
Selection task) and introducing a deployment question this project has
never had to answer (README's own Next Steps still lists "Dashboard / API
(not started)"). Directly tested the alternative before proposing it: hit
Sleeper's real API and confirmed it sends `Access-Control-Allow-Origin:
*` - genuinely CORS-open, so a browser can call it directly. Built
client-side-only, per explicit user direction, after presenting both
options.

**Real bug caught and fixed across two correction rounds**: every draft
of the mapping script initially matched on Sleeper's own `player_id`
field, with a comment incorrectly asserting it was "GSIS-style, same as
ours." Hit Sleeper's live API directly to check: `player_id` is Sleeper's
own internal ID (e.g. `"19"` for Joe Flacco); the real cross-reference
field is `gsis_id` (e.g. `"00-0026158"`), which matches this project's
own GSIS-style player IDs exactly. Verified end-to-end before shipping:
Sleeper's real `gsis_id` for Patrick Mahomes (`00-0033873`) matches his
real `player_id` in `player_weekly_stats.csv` exactly.

**What shipped**: `src/generate_sleeper_id_mapping.py` (new, run once/
periodically like this project's other batch scripts) fetches Sleeper's
real player database once, filters to QB/RB/WR/TE with a real non-null
`gsis_id`, and writes a small static `sleeper_id_mapping.json` (1,259 real
matched players, 181KB - not the 14MB/12k-player full dump). The
frontend never fetches that endpoint itself.

`LeagueConnector.js` (new) calls Sleeper's real, small per-user endpoints
(`user/{username}`, `league/{id}`, `league/{id}/rosters}`) directly via
browser `fetch()`, persists the connection in `localStorage`.
`PersonalRoster.js` (new) maps each real roster player through the static
mapping to this project's real `player_id`, and looks up the matching
real projection from `seasonData.fantasy` (existing real data, no new
backend needed) - matched on a real, unified "current week" (the latest
week this project has real fantasy data for) used consistently for both
the projection lookup and the real bye-week check (derived from the real
schedule, not a nonexistent Sleeper "BYE" sentinel the earlier spec draft
assumed). Wired into the app via the real, established
`constants/sections.js` + `App.js`'s `SECTION_COMPONENTS` pattern - no
new navigation architecture invented.

**Other real bugs caught before building** (full detail in the
conversation): a Python-style `"""docstring"""` inside a JS function
body (real syntax error, not valid JS); a `fetch('/data/...')` call that
wouldn't resolve under this project's CRA setup (JSON lives in `src/data`,
not `public/`) and contradicted the established static-import pattern;
matching against a nonexistent `player_id` field on the fantasy JSON
(both season files only have a composite `id` field,
`"{player_id}_w{week}"`); a hardcoded `getCurrentWeek() { return 1; }`
that ignored real available data; and two confirmed real CSS collisions
(`.team-name` already in `SeasonProjections.css`, `.label` already in
`GamePredictions.css`) - all new CSS scoped under
`.league-connector`/`.personal-roster`, verified against every other
stylesheet in this project with zero collisions across all 35 new class
selectors.

**Real, disclosed limitation**: this is the first feature this session
requiring a genuinely live, interactive user flow (real third-party
network calls, real user-specific state) rather than an offline batch
script producing static JSON. Without Node.js on this machine, none of it
can be run end-to-end - not `npm start`, not a real browser fetch/CORS
round-trip, not a real Sleeper league login. Verified everything that
could be verified without a browser: the real Sleeper API's CORS headers,
the real `gsis_id` field and its match against this project's own data,
balanced JS syntax (brace/paren counts) in both new components, and zero
CSS collisions - but the live connect-a-real-league flow itself is
unverified.

## Sleeper Projection Matching Fix

**Real user report**: after connecting a real Sleeper league, most
roster players showed no projection despite real player names displaying
correctly (i.e. the ID mapping was clearly loading and partially working
- not a total failure).

**Diagnosed directly against real data rather than shipping the
requested debug-logging/screenshot round-trip** - the exact same
matching logic the frontend runs was reproduced in Python against the
real static files already on disk, no browser needed:

- Confirmed the pipeline works correctly for at least one player
  (Mahomes matched end-to-end exactly as expected).
- Checked systematically whether this project's OWN real 294 ranked
  players were even reachable via the existing `sleeper_id_mapping.json`
  (a reverse-direction health check): only 62/294 (21%). Broken down by
  position, even top-5-by-rank real players failed badly - WR was 0/5.
- Root-caused to Sleeper's own `/v1/players/nfl` `gsis_id` field, hit
  directly again: three spot-checked prominent, unambiguous, currently-
  active real starters (Ja'Marr Chase, Puka Nacua, Bijan Robinson) all
  have `gsis_id: null` on Sleeper's own record, despite Sleeper carrying
  every other real detail (name, position, team) correctly for them. Not
  a matching-logic bug in this project's own code - a real, incomplete
  data field on Sleeper's side for exactly this kind of currently-
  relevant player.

**Real fix**: `nflreadpy.load_ff_playerids()` - an existing real
dependency of this project, not a new one - returns the ffverse/
dynastyprocess `db_playerids` crosswalk, a real, externally-maintained,
far-more-complete `gsis_id`/`sleeper_id` mapping (also carries real
`name`/`position`/`team`, so Sleeper's 14MB player endpoint is no longer
fetched by the mapping script at all). Verified this closes the gap
completely: 294/294 of this project's real ranked players are now
reachable, including Chase/Nacua/Robinson and every other previously-
missing top-5-by-position player. A small number of real, ambiguous
crosswalk rows (10 of 12,470 - all obscure/inactive/non-fantasy-relevant
players on inspection) are dropped entirely rather than guessed at, to
keep every shipped mapping unambiguous.

**Real, disclosed operational note**: `load_ff_playerids()` fetches from
a GitHub-hosted CSV and showed transient connection resets during
development (succeeded 3/3 on simple retry) - the regeneration script now
retries up to 3 times before failing, rather than being brittle to a
one-off network hiccup.

Also updated `PersonalRoster.js`'s real UI copy: the "no mapping" and "no
projection" messages previously implied an ID-matching failure for any
gap; now correctly describe the two real, distinct, remaining reasons a
roster player might show no projection (not a fantasy-relevant position,
or real player genuinely outside this project's own top-294 ranked list)
- neither of which the fix above changes, since this project's fantasy
rankings are a real, deliberately curated top-N list, not a full player
database.

## Multi-Signal Trade Engine

Real follow-up to `TRADE_VALUE_ENGINE_FINDINGS_2026-08-12.md`'s open
question: age alone predicts direction at 46.1%, at or below chance.
Combined five more real, mostly-already-available signals and re-ran the
exact same honest backtest discipline (real, forward-looking, non-
circular, leak-free) established last time.

**Real bugs found and fixed before training anything** (a pasted spec
reintroduced several, some of them repeats):

1. `build_trade_signals.py`'s first draft iterated `player_weekly_stats.csv`'s
   raw rows without aggregating to season totals first - the exact same
   bug already found and fixed once this session
   (`validate_directional_accuracy.py`). Fixed the same way.
2. Real temporal leakage: the pasted spec looked up `injury_risk_score`
   from `fantasy_rankings_2026.json` - a value computed once, today, from
   each player's full career through 2025 - and would have used it as a
   "signal" for predicting real 2015-2024 transitions, leaking 2026
   knowledge into the past. Real fix: computes career miss rate
   POINT-IN-TIME for every (player, season) pair, using only real seasons
   strictly before that season (`_add_point_in_time_injury_risk`) - the
   same real leak-free discipline as this project's WR dynamic backtest
   and Elo carryover.
3. `player_id` doesn't exist on `fantasy_rankings_2026.json` (only a
   composite `id`) - moot once fix #2 replaced that lookup entirely.
4. `draft_round` doesn't exist on `player_weekly_stats.csv` - real source
   is `nflreadpy.load_ff_playerids()` (the same real crosswalk that fixed
   the Sleeper ID mapping), joined by real `gsis_id`.
5. No combined, multi-season `team_strength.csv` exists - real per-season
   files only exist for 2025 and 2026 (not 2015-2024), so team-context
   history is genuinely unavailable that way in this project. Real
   substitute with genuine full 2015-2025 coverage: this project's own
   real, already-validated multi-season Elo (`elo_model.run_multi_season_elo`),
   refactored into a shared `real_2026_carryover_elo()` (already done
   last task) plus a new equivalent for arbitrary historical seasons.
6. Row-level K-fold cross-validation risks real leakage since one player
   contributes multiple season-transition rows - switched to `GroupKFold`
   grouped by real `player_id`, the same "no player's own data crosses
   the train/test boundary" discipline already used elsewhere in this
   project (e.g. `ensemble_model.py`'s leave-one-team-out CV).
7. **Most serious**: the pasted frontend reimplemented a completely
   separate, hand-typed scoring heuristic in JavaScript instead of using
   the real trained model - while still showing the real model's honest
   backtested accuracy badge next to it, which would have misrepresented
   a real number as applying to output it had nothing to do with. Real
   fix: this project has no backend (confirmed in the Sleeper Integration
   task) - a live browser session can't invoke the real Python model at
   all, so real per-player scores are precomputed in Python
   (`generate_trade_scores_2026.py`) using the actual fitted models and
   shipped as static JSON. `TradeAnalyzer.js` only ever looks these real,
   precomputed values up - it never re-derives a score.

**Real, honest result**: logistic regression (chosen over gradient
boosting for two real, disclosed reasons - modest real per-position
sample sizes of 69-267 unique players risk overfitting a more complex
model, and its coefficients are honestly interpretable), complete-case
only (no fabricated imputation), GroupKFold (5-fold, by player) CV:

| Position | Real CV accuracy | Real CV AUC | Real unique players |
|---|---|---|---|
| QB | 64.4% | 0.636 | 69 |
| RB | 60.4% | 0.581 | 195 |
| WR | 61.7% | 0.593 | 267 |
| TE | 59.9% | 0.640 | 145 |
| **Overall** | **61.3%** | - | - |

A real, meaningful improvement over age alone (46.1%), every position
above a coin flip. The fitted coefficients show a real, sensible pattern
worth noting: `recent_trend` is negative in every position (a player who
overperformed their own trailing real average tends to regress back down
next season - real mean reversion, not a sign of a broken model).

**What shipped**: a new "Trade Analyzer" section
(`TradeAnalyzer.js`/`TradeAnalyzer.css`, wired into `constants/sections.js`
+ `App.js`). Shows real, precomputed `prob_ppr_increase` for two selected
real players side by side, with real per-player signal detail (age,
career miss rate, role trend, draft capital) and the real, honest,
position-specific CV accuracy - correctly tied to the model that actually
produced the displayed numbers. Real, disclosed limitation: 210/294 of
this project's real ranked players have a complete real signal set
(enough real career history, role-trend data, etc.) - the rest are simply
not shown, not scored with a fabricated default.
