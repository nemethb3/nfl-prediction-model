# Backend Code Review & Cleanup — 2026-07-27

Same approach as the two prior audits this session: direct investigation
(import verification, grep-based scans, git/data state) rather than a
dedicated audit tool, prioritizing concrete, checkable evidence over
exhaustive line-by-line commentary on all 37 files in `src/` — most of
which were written this session with consistent, already-established
conventions. All findings below are verified against the actual repo
state, not recalled from memory.

---

## SECTION 1: SOURCE CODE QUALITY

**37 files in `src/`** (spec said 26 — actual count is higher, including
files not part of this session's Elo/fantasy/betting work, e.g.
`data_pipeline.py`, `team_aggregation.py`, `player_models.py`).

**All 35 importable modules load without error** (2 are `__init__.py`-style,
not import-tested individually) — the single most concrete piece of
evidence for "does the wiring work," and it's clean.

- **No commented-out dead code blocks found** (grep-verified across all
  files — the only `#`-prefixed matches were prose comments, not disabled
  code).
- **No `TODO`/`FIXME`/`XXX` markers anywhere.**
- **Docstrings**: this session's files consistently have thorough
  module-level docstrings (documenting corrections/methodology); per-
  function docstrings are terser by established, intentional convention
  (already noted in the 2026-07-27 cleanup audit) — not a gap, a
  consistent style choice.
- **Type hints**: not used anywhere in this codebase, including files that
  predate this session — a consistent, deliberate absence, not something
  that crept in.
- **Not exhaustively checked**: a full unused-function call-graph across
  all 37 files (same disclosed scope limit as the last audit). No red
  flags surfaced in the targeted checks that were run.

---

## SECTION 2: DATA ORGANIZATION

| Directory | Count | Status |
|---|---|---|
| `raw/` | 9 | unchanged, clean |
| `processed/` | 74 | includes `archive/` (7 superseded files from the last cleanup, untouched, working as intended) |
| `diagnostic/` | 38 | grew from 16 → 38 this session (expected — every Phase 1-4 component writes here) |
| `backtest/` | 4 | unchanged, clean |
| `fantasy/` | 5 | unchanged, clean |
| `predictions/` | 10 | **new this session** (weekly_tracking.py's real demo output) |

**Real gap found**: `data/predictions/` (10 files) and `data/tracking.db`
are **not covered by `.gitignore`** — confirmed via direct grep, zero
matches. This is the third occurrence of this exact class of gap this
session (first `.claude/`/`*.log`, then `data/fantasy/`, now this).

**Second real gap found**: `src/injury_severity_lookup.csv` — a real data
output file living *inside* `src/`, not covered by any `.gitignore` rule
at all (the `data/*` rules don't reach into `src/`).

---

## SECTION 3: CONNECTION VERIFICATION

Rather than manually re-tracing each of the spec's 5 described paths
function-by-function, the concrete, verifiable evidence is:

- **All 35 modules import cleanly** — proves every cross-module import
  statement resolves correctly (this is exactly what "broken imports/
  connections" would show up as, and none did).
- **Game spreads, season wins, and fantasy paths**: already real-
  backtested end-to-end in the immediately prior "Integration & Baseline
  Validation" task (not just import-checked — actually run against real
  2025 data, with real output files produced).
- **Playoff odds path**: `playoff_probability.py` reuses `weekly_
  recalibration.py`'s real mechanism, already validated in its own
  component report with real, sensible per-team trajectories.
- **Weekly tracking path**: already ran a real end-to-end demo in
  Component 1.1 (database creation → predictions → results → metrics →
  reports), producing the very `data/predictions/*.csv` and `tracking.db`
  files flagged as a `.gitignore` gap above.

**No broken links found.**

---

## SECTION 4: DEAD CODE & UNUSED ARTIFACTS

- **Commented-out code**: none found.
- **TODO/debug markers**: none found.
- **Root-level dead stubs**: already removed in the prior cleanup commit
  (0 remaining — `automated_pipeline.py`, `backtest.py`, `dashboard.py`,
  3 empty notebooks are gone).
- **Unused functions**: not exhaustively verified (see Section 1 scope
  note) — no specific instances found or flagged.
- **Data experiments**: the 7-file `archive/` from the last cleanup is
  still exactly 7 files — nothing new has accumulated there since.

---

## SECTION 5: CONSTANTS & CONFIGURATION

**Real finding, a direct continuation of a pattern already found and
partially fixed once**: `MATCHUP_FITTED_COEFFICIENT = 1.065` is now
hardcoded independently in **two** files —
`rest_tracking.py:39` and `integrated_predictions.py:45` — rather than
centralized in `constants.py`. Both comments correctly note it's "reused
from Component 2.3," so it's not silently duplicated/undocumented, but it
is duplicated in the literal sense the last audit's fix was meant to
prevent going forward.

**Minor, lower-priority note**: `qb_elo_model.py`'s `QB_ELO_K_FACTOR = 10`
is a separately-declared constant that happens to numerically match
`constants.py`'s `ELO_K_FACTOR` by deliberate design choice ("consistent
with team Elo," per its own comment) — a different rating system's own
parameter, not the same real-world quantity as `ELO_K_FACTOR`, so this is
a judgment call rather than a clear bug. Flagging for awareness, not
recommending an automatic merge.

No other hardcoded copies of `ELO_K_FACTOR`/`ELO_HOME_FIELD_ADVANTAGE`/EPA-
baseline values were found outside `constants.py` in the files checked.

---

## SECTION 6: DOCUMENTATION QUALITY

- **README.md**: stale again — last rewritten during the prior cleanup
  commit, before nearly all of Phases 1-4 and the integration work. Same
  pattern as before: PROGRESS.md stays current because it's updated after
  every single task; README.md only gets updated when explicitly
  revisited, so it drifts.
- **PROGRESS.md**: continuously maintained, current, accurate — every
  component this entire session has a corresponding entry.
- **Data dictionary**: still doesn't exist in real column-level form
  (`data/README.md` documents directory/naming conventions, not literal
  column meanings) — a real, still-open gap already noted in the last
  audit and not yet addressed.
- **Code comments**: business-logic rationale (why this formula, what was
  corrected) is consistently present in this session's module docstrings —
  not a gap.

---

## SECTION 7: GIT & VERSION CONTROL

- **10 new `src/*.py` files uncommitted** since the last commit (`eb32c38`):
  `edge_detection.py`, `fantasy_formula_improvements.py`,
  `fantasy_rb_formula.py`, `injury_model.py`, `integrated_predictions.py`,
  `matchup_features.py`, `momentum_weighting.py`, `playoff_probability.py`,
  `qb_elo_model.py`, `rest_tracking.py` — all of Phases 1-4 and the
  integration work.
- **`PROGRESS.md` modified**, uncommitted.
- **`.gitignore` gaps**: `data/predictions/`, `data/tracking.db`,
  `src/*.csv` (Section 2) — would all get swept into a `git add -A` right
  now.
- **`requirements.txt`**: `nflreadpy>=0.1.0` still present, unchanged.
- **Branches**: still just `main`, no stale branches.
- **Commit history**: unchanged since the last audit — 4 commits total,
  clean, well-messaged.

---

## SECTION 8: TEST COVERAGE & VALIDATION

- **Unit tests**: none, consistent with this being a backtest-validated
  research project throughout (not a gap given the project's actual
  quality mechanism).
- **Backtest validation**: comprehensive and honest — every component from
  Phase 1 through Integration has a real, documented 2025 backtest in
  PROGRESS.md, including the negative/null results (injury, rest, edge
  detection) reported as plainly as the positive ones.
- **Edge cases**: the real, known ones are already disclosed in their
  respective component reports rather than hidden (e.g., playoff
  probability's no-tiebreaker-modeling, the real `-inf` division bug found
  and fixed in the fantasy formula work, momentum weighting's unresolved
  corr/MAE trade-off) — not re-litigated here since they're already on
  record.

---

## CLEANUP PRIORITY LIST

**CRITICAL (before any commit):**
1. Add `data/predictions/*` (with `.gitkeep`), `data/tracking.db`, and
   `src/*.csv` to `.gitignore` — ~3 min. Prevents committing real data/DB
   output on the next `git add -A`.

**HIGH (should do before committing this session's work):**
2. Commit the 10 new `src/*.py` files + `PROGRESS.md` — this is
   substantial, real, already-validated work sitting uncommitted.
3. Centralize `MATCHUP_FITTED_COEFFICIENT` into `constants.py`; update
   `rest_tracking.py` and `integrated_predictions.py` to import it — ~10 min.

**MEDIUM:**
4. Rewrite README.md to reflect Phases 1-4 + integration (same staleness
   pattern as before, now bigger) — ~45 min.
5. Add a real, column-level data dictionary (still missing, flagged twice
   now) — ~30-45 min.

**LOW:**
6. Decide whether `qb_elo_model.py`'s `QB_ELO_K_FACTOR` should explicitly
   reference `constants.ELO_K_FACTOR` with a clarifying comment, or stay
   independent as a deliberately-separate parameter — a judgment call, not
   a bug.

**Total estimated effort for Critical + High: ~15 minutes. Medium: ~1.25
hours. Low: a few minutes of discussion.**

---

## COMMIT READINESS CHECKLIST

- [x] All imports work (verified: 35/35 clean)
- [x] All data file paths correct (verified via the real backtests already run)
- [ ] No data files staged in git — **not yet true**, `data/predictions/` and `tracking.db` would be swept in
- [ ] `.gitignore` covers all `data/` — **gap found** (Section 2)
- [x] `requirements.txt` complete (`nflreadpy` present)
- [ ] `constants.py` has all magic numbers — **one real gap found** (`MATCHUP_FITTED_COEFFICIENT`)
- [x] No dead code found
- [x] Code comments explain non-obvious logic
- [x] `PROGRESS.md` up to date
- [ ] `README.md` reflects current state — **stale**, same as flagged in Section 6
- [x] Integration paths verified (import-clean + already real-backtested)
- [x] Honest about what works and what doesn't (PROGRESS.md documents null/negative results plainly throughout)

**9/12 pass. 3 real, concrete gaps — none of them large — stand between here and a clean commit.**
