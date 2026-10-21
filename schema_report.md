# Spread Forecast Output Schema — Report for Third-Party Review

**Scope:** read-only investigation. No code was modified to produce this report. All format examples below use placeholder values only (`KC @ BUF`, `-3.5`, `0.57`, etc.) — no real forecast values appear anywhere in this document.

---

## 1. Entry points that generate live spread predictions

These are the scripts that run for upcoming (not-yet-played) games, as part of the weekly live pipeline — not the backtest scripts.

| File | Role |
|---|---|
| `src/refresh_weekly.py` | Top-level weekly entry point. Invokes everything below, in order, as subprocess steps. |
| `src/orchestrate_2026_pipeline.py` | Enforces the correct write order for the two scripts below (step 2 must always run after step 1, or it silently overwrites step 2's output — see its own module docstring). |
| `src/generate_dashboard_data_2026.py` | Writes the core per-game fields: `our_spread`, `win_prob_home`, `ci_low_90`, `ci_high_90`, Elo ratings, `single_elo_spread`. Calls into `compute_offensive_defensive_elo.generate_od_elo_game_spreads()` for the actual math. |
| `src/compute_offensive_defensive_elo.py` (`generate_od_elo_game_spreads`, `predict_od_spread`, `od_elo_win_probability`) | The real forecast math: Elo differential → win probability (logistic) → spread (fitted linear/logit/probit regression on win probability). |
| `src/apply_point_totals_2026.py` | Enriches with `predicted_total_value`, `vegas_total`, `predicted_total_diff`, `predicted_total_direction`. Separate model from the spread. |
| `src/join_espn_odds_2026.py` | Joins a real market spread (`vegas_spread`) and moneylines from ESPN's odds feed into the game record. |
| `src/recalibrate_2026_elo.py` | Refreshes the current (not frozen-preseason) O/D-Elo ratings used by `generate_od_elo_game_spreads` once any real games have completed. |
| `src/elo_game_prediction.py` (`generate_elo_game_spreads`) | A second, single-Elo model used only as a "Matchup Strength" comparison field (`single_elo_spread` etc.) — not the primary shipped forecast. |

---

## 2. Where predictions are saved

**File:** `frontend/src/data/games_2026.json` — one JSON file, top-level array, one object per scheduled game, written fresh each pipeline run (not a database, not append-only).

**All fields currently in that object** (37 total; types/units inferred directly from the producing code):

| Field | Type | Unit / meaning |
|---|---|---|
| `id` | string | `"{season}_{week:02d}_{away}_{home}"`, e.g. format `2026_01_AAA_BBB` |
| `week` | integer | NFL week number |
| `weekday` | string | e.g. `"Thursday"` |
| `kickoff_datetime` | string | naive datetime string, no timezone marker (see §6) |
| `home_team` / `away_team` | string | team abbreviation |
| `home_qb_name` / `away_qb_name` | string or null | |
| `home_recent_form` / `away_recent_form` | float or null | |
| `head_to_head` | object or null | |
| `home_elo` / `away_elo` | float | single-Elo rating, pre-game |
| `home_o_elo` / `home_d_elo` / `away_o_elo` / `away_d_elo` | float | O/D-Elo split ratings, pre-game |
| `single_elo_spread` | float | points, single-Elo model's spread (comparison field, not primary) |
| `single_elo_win_prob_home` | float | probability [0,1] |
| `single_elo_ci_low_90` / `single_elo_ci_high_90` | float | points |
| `our_spread` | float | points — **the primary shipped forecast** (see §4 for sign convention) |
| `win_prob_home` / `win_prob_away` | float | probability [0,1], sum to 1.0 |
| `ci_low_90` / `ci_high_90` | float | points — 90% interval around `our_spread` |
| `predicted_total_value` | float or null | points, combined-score forecast (separate model) |
| `vegas_total` | float or null | points — real market total (nflverse `total_line`), only posted this far out for weeks 1-4 |
| `predicted_total_diff` | float or null | `predicted_total_value - vegas_total` |
| `predicted_total_direction` | string or null | `"OVER"` / `"UNDER"` |
| `vegas_spread` | float or null | points — real market spread (ESPN odds feed; see §4-5) |
| `matchup_quality` | string or null | |
| `net_edge_diff` | float or null | |
| `base_source` | string or null | |
| `actual_home_score` / `actual_away_score` | integer or null | filled post-game |
| `actual_winner` | string or null | filled post-game |
| `actual_spread_margin` | integer or null | filled post-game |
| `did_we_predict_correctly` | boolean or null | filled post-game |

---

## 3. Field-by-field status against the partner's 6 requirements

| # | Partner's field | Status | Our field name | Type | Format example (placeholder) | Notes |
|---|---|---|---|---|---|---|
| 1 | Event identifier | **EXISTS** | `id` | string | `"2026_01_AAA_BBB"` | `{season}_{week}_{away}_{home}`, built in `generate_dashboard_data_2026.py` |
| 1 | Kickoff time + timezone | **PARTIAL** | `kickoff_datetime` | string | `"2026-10-04T13:00:00"` | Present, but **naive** — no timezone offset or marker anywhere in the value or a sibling field. Built by plain string concatenation (`f"{gameday.date()}T{gametime}:00"`) from nflverse's schedule columns, which carry no timezone metadata of their own. The de-facto convention (nflverse community practice, not asserted in this project's code) is US/Eastern local time — unverified, not encoded. |
| 2 | Market/spread sign convention | **EXISTS** (as metadata-free convention, not a stored field) | — | — | — | The convention itself isn't a stored field — it's a fixed rule in code; see §4 for the exact rule and source lines. |
| 3 | Forecast value + what it measures | **EXISTS** | `our_spread` | float | `"-3.5"` | Predicted **home-minus-away point margin** (not a cover probability, not a pick-side label). `win_prob_home`/`win_prob_away` separately give the probability the home/away team wins outright — also not a cover probability. No field expresses "probability of covering this specific line." |
| 4 | Reference market line: source + observation time | **PARTIAL** | `vegas_spread` (+ `vegas_total`) | float or null | `"-3.0"` | Source is known and real (see §5), but the **observation timestamp is not stored on the game record at all**. A timestamp (`fetched_at`) exists upstream in `data/espn_odds/history.json` and in the intermediate `data/processed/vegas_lines_2026.csv` (as `closing_line_fetched_at`), but neither of those is gitignored-safe long-term storage, and the join into `games_2026.json` keeps only the numeric value, discarding the timestamp. `vegas_total` comes from a **different** source (nflverse schedule data) with no observation timestamp anywhere. |
| 5 | Forecast generation timestamp | **PARTIAL** | `generated_at.json["games_2026"]` | string (ISO 8601, UTC, timezone-aware) | `"2026-10-04T15:02:33.000000+00:00"` | Exists, but is **per dataset/batch, not per game or per prediction**. One timestamp covers the entire weekly games file, not each individual forecast. Not embedded in the game object itself — it's a separate sidecar file the frontend doesn't currently surface. |
| 6 | Model version / run identifier | **MISSING** | — | — | — | No git hash, package/model version string, config hash, or run ID is stored anywhere in the predictions pipeline. Confirmed by direct search of the codebase and every shipped JSON file — zero matches for any such field. The closest adjacent thing is `data/refresh_logs/week_N_refresh_<timestamp>.json`, a per-run step-status log (not a run ID, and not referenced from the prediction record itself). |

---

## 4. Spread sign convention

**In one sentence:** a positive `our_spread` (and a positive `vegas_spread`, after this project's own sign flip — see below) means the **home team is favored** by that many points; a negative value means the **away team is favored**.

Quoted from the code:

- `src/ingest_completed_results_2026.py`:
  ```python
  predicted_winner = home if our_spread > 0 else away
  ```
- `src/betting_backtest.py` (module docstring, describing the established, verified real convention):
  > "Real sign convention. ... This project's real, verified convention ... is the opposite: positive vegas_spread/our_spread means the HOME team is favored."
- `src/join_espn_odds_2026.py` (module docstring) — note this is the **opposite** of the raw ESPN convention, which this project deliberately flips on ingestion:
  > "ESPN's own home_spread_current field uses the standard sportsbook convention (negative = home favored, e.g. '-3' means home favored by 3). This project's own vegas_spread/our_spread convention is the opposite (positive = home favored) ... Every real spread pulled from ESPN is negated here to match."

**Risk flag:** this means `vegas_spread` as stored in `games_2026.json` does **not** use the sign convention a sportsbook or a typical odds API would hand a partner directly — it's already been flipped to match this project's internal `our_spread` convention. If the partner's own systems expect the standard sportsbook sign (negative = favorite), this flip must be explicitly documented in the schema handed over, or re-flipped on export.

---

## 5. Vegas/market line: source and timing

**In one sentence:** `vegas_spread` is sourced from ESPN's unofficial odds API (a single sportsbook per game, commonly DraftKings, not a consensus line) and reflects whichever snapshot was most recently fetched before the join ran — functioning as a late/near-closing line in practice but not a guaranteed, formally-defined closing line — while `vegas_total` is sourced separately from nflverse's own schedule data (`total_line` column), which is only populated for the first several weeks of the season this far in advance.

Supporting detail:
- `src/espn_odds_client.py` fetches real per-game odds from ESPN's `sports.core.api.espn.com` endpoint; `src/espn_odds_orchestrate.py` runs this once per week, for whichever games are on the "current week" scoreboard at run time.
- `src/join_espn_odds_2026.py` keeps only the single **latest** `fetched_at` snapshot per matchup from the accumulated history (`data/espn_odds/history.json`), discarding older snapshots and the `fetched_at` timestamp itself when writing into `games_2026.json`.
- A given game can have several real snapshots on file between line-open and kickoff (the odds collector runs every week); which one ends up as "latest" depends entirely on when the pipeline happened to run, not a formal "closing line" cutoff tied to kickoff.
- `vegas_total` instead comes from `data/raw/schedules_2026.csv`'s `total_line` column (nflverse), confirmed real for 2026 weeks 1–4 only (per `apply_point_totals_2026.py`'s own module docstring) — a structurally different source from `vegas_spread`, with its own, separate (and currently narrower) real-data coverage.

---

## 6. Timestamp timezone-awareness

| Timestamp | Naive or aware | Timezone |
|---|---|---|
| `kickoff_datetime` (in `games_2026.json`) | **Naive** | None stored; presumed US/Eastern by nflverse convention, not asserted in code |
| `generated_at.json["<dataset>"]` | **Aware** | UTC (`datetime.now(timezone.utc).isoformat()`) |
| `data/refresh_logs/week_N_refresh_*.json` timestamps | **Naive** | Local system clock of whatever machine ran the refresh (`datetime.now().isoformat()`, no timezone attached) |

**Risk flag:** the codebase is inconsistent — one timestamp source is UTC-aware, two others are naive and implicitly tied to different, unstated clocks (nflverse's convention vs. the local machine's). A partner reconciling kickoff time against a generation timestamp cannot currently do so reliably without an unstated assumption on both sides.

---

## 7. Versioning

No model version, git commit hash, config hash, or run identifier is captured or stored anywhere in the prediction pipeline or its outputs. Confirmed by direct search (zero matches for any such field/string across `src/*.py` and every shipped `frontend/src/data/*.json` file). The fitted model coefficients themselves (e.g. `fitted["spread_model"]`, `fitted["k_factor"]`) are loaded from static JSON model files in the repo, but those files carry no version tag of their own either, and the forecast output doesn't reference which model file / coefficient set produced it.

---

## 8. Lookahead check

**Game-level spread/win-probability (`our_spread`, `win_prob_home`, Elo ratings):** no lookahead risk found. `compute_offensive_defensive_elo.generate_od_elo_game_spreads()` explicitly branches on whether a game has already been played: an already-played game's rating is pinned to its real, leak-free pre-game value from history and is never recomputed using later information; a not-yet-played game uses the current rating as of the latest real completed game. This was directly verified (quoted from the function's own docstring):

> "a game that's already been played uses its real, leak-free home_o_elo_before/home_d_elo_before/etc. ... never recomputed using a later week's information. a game not yet played uses the CURRENT real rating."

No injury data of any kind feeds into this pipeline — confirmed by direct search; the only injury-adjustment code in this project is a separate, fantasy-projection-only overlay that never touches `games_2026.json` or any Elo-computing script.

**Market-line fields (`vegas_spread`, `vegas_total`) — real, conditional risk found:**
- `vegas_spread`'s "latest fetched snapshot" selection (§5) has no explicit cutoff tied to kickoff. If the weekly refresh is run **after** some of that week's games have already kicked off (e.g., a Sunday afternoon run after the early games), the ESPN scoreboard call could return in-game or closed odds for games already underway, and the join would pick that up as the "pre-game" reference line for a game that, by the time of the run, is no longer actually upcoming. This is a timing/operational risk, not a current, observed defect — it depends entirely on when the pipeline happens to be run relative to that week's kickoffs.
- `vegas_total`'s source (nflverse schedule data) is a static snapshot with no visible real-time risk of this kind, since it's not scraped live mid-week by this project — but it also carries no observation timestamp, so a partner cannot independently confirm when that number was true.

---

## Gaps, ranked by importance for a credible pre-game record

| Rank | Gap | Why it matters | Rough effort to fix |
|---|---|---|---|
| 1 | No per-prediction generation timestamp or run/version identifier (§5, §7) | Without this, a third party cannot prove a given forecast was genuinely produced before kickoff rather than reconstructed after the fact — this is usually the single most important field for a credible pre-game record. | Small–medium: add a `generated_at` + a run identifier (even a simple incrementing run counter or the refresh timestamp already computed in `refresh_weekly.py`) to each game object at write time. |
| 2 | No observation timestamp stored alongside `vegas_spread`/`vegas_total` on the game record (§4, §5) | A reference line without a timestamp can't be judged as "the line at kickoff" vs. "the line whenever it happened to be fetched" — undermines the credibility of any backtest or audit built on it. | Small: the timestamp already exists upstream (`fetched_at` in the ESPN history, `closing_line_fetched_at` in the intermediate CSV) — just needs to be carried through the join into `games_2026.json` instead of being dropped. |
| 3 | `kickoff_datetime` has no timezone (§1, §6) | Any downstream consumer (including a partner) has to guess the timezone; a misread here silently shifts every game's lookahead-safety judgment by hours. | Small: attach an explicit UTC offset or IANA zone when the field is built, confirming nflverse's actual convention first rather than assuming it. |
| 4 | No model version / git hash / config hash on the output (§7) | Without this, "which model produced this forecast" can only be answered by cross-referencing commit history by hand — not reproducible or auditable at scale. | Medium: would need a deliberate convention (e.g., stamp the current git commit short-hash into `generated_at.json` or a new sidecar at refresh time) — small in code, but needs a decision on exactly what "version" should mean here. |
| 5 | `vegas_spread`'s "latest fetched snapshot" has no explicit pre-kickoff cutoff (§8) | A pipeline run mid-week, after some games have kicked off, could silently pick up a post-kickoff odds snapshot as the "pre-game" reference line for an already-started game. | Small–medium: filter the join to snapshots fetched strictly before that specific game's `kickoff_datetime` (once that field itself is timezone-correct — see gap 3). |
| 6 | Inconsistent timestamp timezone-awareness across the codebase (§6) | Two different "naive, implicitly-local" timestamp sources plus one UTC-aware one makes any cross-field time comparison fragile and easy to get wrong. | Small: standardize all new timestamp-writing code on `datetime.now(timezone.utc)`, matching `generation_timestamps.py`'s existing (correct) convention. |
| 7 | `vegas_spread` and `vegas_total` come from two structurally different sources with no shared field documenting that split (§5) | Not a bug, but worth disclosing explicitly to a partner so they don't assume one unified "Vegas data" pipeline. | Documentation only — no code change. |
