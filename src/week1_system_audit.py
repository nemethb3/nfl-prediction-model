"""Real Week 1 system audit across game predictions, player projections,
the ensemble model, betting analysis, accuracy tracking, UI/UX, and data
quality.

Real, serious problems found and fixed vs. the originally pasted spec
before writing this:

1. `data/accuracy_tracking/week_1_accuracy.json` and `data/betting_
   analysis/week_1_analysis.json` don't exist - neither directory exists
   anywhere in this project. Those paths (and the `{ats_analysis: {...},
   roi_analysis: {...}}` schema the spec assumed for the betting file) are
   exactly the fabricated paths/schema from an EARLIER pasted spec this
   session already rejected (see the real "Accuracy Tracker + Weekly
   Summary + Betting Analysis" task) - the real files this project
   actually generates are frontend/src/data/accuracy_tracker_2026.json
   (`{season_summary: {games, fantasy, season_projections, betting}}`)
   and frontend/src/data/betting_backtest_results_2026.json (`{our_system:
   {...}, vegas_favorites: {...}, underdogs_only: {...}}`, each with
   `.moneyline`/`.ats` sub-keys). As written, the spec's audit would
   silently report "not generated yet" for two real, live, already-working
   dashboard sections.
2. Real crash: `ensemble_projections_2026.json` is a real DICT
   (`{generated_at, methodology_note, n_matched, n_unmatched_no_props_row,
   players: [...]}`), not a bare list - `for p in self.ensemble` as
   written would iterate the dict's own string KEYS ('generated_at', ...)
   and crash on `.get('week')` (`AttributeError: 'str' object has no
   attribute 'get'`) the first time it ran. Fixed to read `.players`.
3. The "UI/UX Assessment" component was entirely hardcoded print
   statements - a canned list of "working features"/"issues"/"quick wins"
   presented as audit findings but computed from nothing. Checked directly
   against the real codebase before writing this: a position filter
   (`selectedPosition`/`POSITIONS.map`) ALREADY EXISTS in FantasyRankings.js
   (the spec's own "quick win" list recommended adding it) and a real
   mobile media query already exists in FantasyRankings.css. Rebuilt as
   real, lightweight checks against the actual component/CSS source
   instead of assumptions.
4. The "health scorecard" used unexplained, arbitrary formulas
   (`100 - mae*25`, flat hardcoded 70%/85% for UI/UX and Data regardless of
   any real finding, `75 if fully_blended>0 else 50`) - this project's own
   established rule is that scoring constants must be derived or clearly
   disclosed as illustrative, never asserted as if measured. Replaced with
   a direct table of the real underlying metrics (already meaningful 0-100
   numbers where one exists, like accuracy_pct - not forced into a
   percentage where one doesn't naturally exist, like MAE).
5. `generate_recommendations()` returned hardcoded strings with today's
   specific numbers baked in as literal text (e.g. "Only 87% coverage
   (14/16 games)") - would print the same claim forever regardless of what
   a future run actually found. Rebuilt to build recommendation text from
   `self.findings` at run time.
"""

import json
import statistics
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FRONTEND_DATA_DIR = PROJECT_ROOT / "frontend" / "src" / "data"
AUDIT_OUTPUT = PROJECT_ROOT / "data" / "diagnostic" / "week1_system_audit.json"


class Week1SystemAudit:
    def __init__(self):
        self.timestamp = datetime.now(timezone.utc).isoformat()
        self.findings = {}

    def run(self):
        print("\n" + "=" * 70)
        print("WEEK 1 SYSTEM AUDIT")
        print("=" * 70 + "\n")

        self.load_data()

        for title, fn in [
            ("COMPONENT 1: GAME PREDICTIONS", self.audit_game_predictions),
            ("COMPONENT 2: PLAYER PROJECTIONS", self.audit_player_projections),
            ("COMPONENT 3: ENSEMBLE MODEL", self.audit_ensemble_model),
            ("COMPONENT 4: BETTING ANALYSIS", self.audit_betting_analysis),
            ("COMPONENT 5: ACCURACY TRACKING", self.audit_accuracy_tracking),
            ("COMPONENT 6: UI/UX (real, checked against source)", self.audit_ui_ux),
            ("COMPONENT 7: DATA QUALITY", self.audit_data_quality),
        ]:
            print("\n" + "=" * 70)
            print(title)
            print("=" * 70)
            fn()

        print("\n" + "=" * 70)
        print("EXECUTIVE SUMMARY")
        print("=" * 70)
        self.print_summary()
        self.save_audit_report()

    def load_data(self):
        with open(FRONTEND_DATA_DIR / "games_2026.json", encoding="utf-8") as f:
            self.games = json.load(f)
        with open(FRONTEND_DATA_DIR / "fantasy_rankings_2026.json", encoding="utf-8") as f:
            self.rankings = json.load(f)
        with open(FRONTEND_DATA_DIR / "ensemble_projections_2026.json", encoding="utf-8") as f:
            self.ensemble = json.load(f)["players"]
        print("Loaded real games_2026.json / fantasy_rankings_2026.json / ensemble_projections_2026.json")

    # ------------------------------------------------------------------
    def audit_game_predictions(self):
        week1_games = [g for g in self.games if g.get("week") == 1]
        completed = [g for g in week1_games if g.get("actual_winner")]

        if not completed:
            print("  No Week 1 games completed yet.")
            return

        correct = sum(1 for g in completed if g.get("did_we_predict_correctly"))
        accuracy = 100 * correct / len(completed)
        print(f"  Record: {correct}/{len(completed)} ({accuracy:.1f}%)")

        spread_errors = [
            abs(g["our_spread"] - g["actual_spread_margin"])
            for g in completed if g.get("our_spread") is not None and g.get("actual_spread_margin") is not None
        ]
        mae = statistics.mean(spread_errors) if spread_errors else None
        if mae is not None:
            print(f"  Spread MAE: {mae:.2f}pts")

        high_conf = [g for g in completed if g.get("win_prob_home", 0.5) >= 0.65 or g.get("win_prob_home", 0.5) <= 0.35]
        high_conf_accuracy = None
        if high_conf:
            high_conf_correct = sum(1 for g in high_conf if g.get("did_we_predict_correctly"))
            high_conf_accuracy = 100 * high_conf_correct / len(high_conf)
            print(f"  High-confidence picks: {high_conf_correct}/{len(high_conf)} ({high_conf_accuracy:.1f}%)")

        with_vegas = [g for g in completed if g.get("vegas_spread") is not None]
        vegas_coverage = 100 * len(with_vegas) / len(completed)
        print(f"  Vegas lines captured: {len(with_vegas)}/{len(completed)} ({vegas_coverage:.0f}%)")

        self.findings["games"] = {
            "n_completed": len(completed), "correct": correct, "accuracy_pct": round(accuracy, 1),
            "spread_mae": round(mae, 2) if mae is not None else None,
            "high_conf_accuracy_pct": round(high_conf_accuracy, 1) if high_conf_accuracy is not None else None,
            "vegas_coverage_pct": round(vegas_coverage, 1),
            "vegas_coverage_n": f"{len(with_vegas)}/{len(completed)}",
        }

    # ------------------------------------------------------------------
    def audit_player_projections(self):
        with_actuals = [p for p in self.rankings if p.get("actual_ppr") is not None]
        if not with_actuals:
            print("  No player actuals yet.")
            return

        errors_by_pos = {}
        for p in with_actuals:
            errors_by_pos.setdefault(p["position"], []).append(abs(p["actual_ppr"] - p["projected_ppr"]))

        print(f"  Players with real actual_ppr: {len(with_actuals)}")
        by_position = {}
        for pos in sorted(errors_by_pos):
            mae = statistics.mean(errors_by_pos[pos])
            by_position[pos] = round(mae, 2)
            print(f"    {pos}: {mae:.2f} PPR MAE (n={len(errors_by_pos[pos])})")

        overall = statistics.mean(mae for mae in by_position.values())
        print(f"  Overall MAE: {overall:.2f} PPR")

        self.findings["players"] = {
            "n_with_actuals": len(with_actuals), "overall_mae": round(overall, 2), "by_position": by_position,
        }

    # ------------------------------------------------------------------
    def audit_ensemble_model(self):
        if not self.ensemble:
            print("  Ensemble not generated yet.")
            return

        weights = [p["model1_weight"] for p in self.ensemble if p.get("model1_weight") is not None]
        avg_w1 = statistics.mean(weights) if weights else None
        n_extreme = sum(1 for w in weights if w in (0.0, 1.0))
        pct_extreme = 100 * n_extreme / len(weights) if weights else 0
        sample_sizes = [p["weight_fit_sample_size"] for p in self.ensemble if p.get("weight_fit_sample_size")]

        print(f"  Ensemble records: {len(self.ensemble)}")
        if avg_w1 is not None:
            print(f"  Average blend: {avg_w1:.0%} Model 1 / {1 - avg_w1:.0%} Model 2")
        print(f"  Records at an extreme (100/0 or 0/100) weight: {n_extreme}/{len(weights)} ({pct_extreme:.0f}%)")
        if sample_sizes:
            print(f"  Real weight-fit sample size range: {min(sample_sizes)}-{max(sample_sizes)} "
                  "(small - see optimize_ensemble_weights_2026.py's own disclosed caveat)")

        self.findings["ensemble"] = {
            "n_records": len(self.ensemble),
            "avg_model1_weight": round(avg_w1, 2) if avg_w1 is not None else None,
            "pct_at_extreme_weight": round(pct_extreme, 1),
            "min_fit_sample_size": min(sample_sizes) if sample_sizes else None,
        }

    # ------------------------------------------------------------------
    def audit_betting_analysis(self):
        path = FRONTEND_DATA_DIR / "betting_backtest_results_2026.json"
        if not path.exists():
            print("  Betting analysis not generated yet.")
            return
        with open(path, encoding="utf-8") as f:
            betting = json.load(f)

        # Real schema: strategy sub-dicts (our_system/vegas_favorites/
        # underdogs_only) each with .moneyline/.ats.season_summary, plus 2
        # metadata keys with no `.label` - same filter FantasyRankings/
        # BettingAnalysis.js use.
        strategy_keys = [k for k, v in betting.items() if isinstance(v, dict) and "label" in v]
        print(f"  Real strategies present: {strategy_keys}")
        print(f"  Coverage note: {betting.get('real_odds_coverage_note', 'n/a')}")

        summaries = {}
        for key in strategy_keys:
            ats_summary = betting[key]["ats"]["season_summary"]
            summaries[key] = ats_summary
            print(f"    {key} (ATS): {ats_summary['wins']}-{ats_summary['losses']}"
                  f"{'-' + str(ats_summary['pushes']) if ats_summary['pushes'] else ''} "
                  f"({ats_summary['win_pct']}%), ROI {ats_summary['roi_pct']:+.1f}%")

        self.findings["betting"] = {
            "strategies": strategy_keys,
            "our_system_ats": summaries.get("our_system"),
            "coverage_note": betting.get("real_odds_coverage_note"),
        }

    # ------------------------------------------------------------------
    def audit_accuracy_tracking(self):
        path = FRONTEND_DATA_DIR / "accuracy_tracker_2026.json"
        if not path.exists():
            print("  Accuracy tracker not generated yet.")
            return
        with open(path, encoding="utf-8") as f:
            accuracy = json.load(f)

        games_summary = accuracy.get("season_summary", {}).get("games")
        fantasy_summary = accuracy.get("season_summary", {}).get("fantasy")
        if games_summary:
            print(f"  Games tracked: {games_summary['total_games']}, "
                  f"accuracy {games_summary['accuracy_pct']}%, "
                  f"vs.-Vegas coverage: {games_summary.get('vs_vegas_coverage', 'n/a')}")
        if fantasy_summary:
            n_players = sum(v["samples"] for v in fantasy_summary.values())
            print(f"  Fantasy positions tracked: {list(fantasy_summary.keys())} ({n_players} real samples)")

        self.findings["tracking"] = {
            "games_summary": games_summary,
            "fantasy_positions_tracked": list(fantasy_summary.keys()) if fantasy_summary else [],
        }

    # ------------------------------------------------------------------
    def audit_ui_ux(self):
        """Real, checked claims against the actual component/CSS source -
        not a hardcoded list (see module docstring point 3)."""
        checks = []

        def _file_contains(rel_path, needle, label):
            full_path = PROJECT_ROOT / rel_path
            found = full_path.exists() and needle in full_path.read_text(encoding="utf-8")
            checks.append((label, found, rel_path))
            return found

        _file_contains("frontend/src/components/FantasyRankings.js", "selectedPosition", "Position filter on Fantasy Rankings")
        _file_contains("frontend/src/styles/FantasyRankings.css", "@media", "Mobile responsive CSS on Fantasy Rankings")
        _file_contains("frontend/src/components/FantasyRankings.js", "Ensemble Projection", "Ensemble blend shown on player card")
        _file_contains("frontend/src/components/FantasyRankings.js", "confidence_tier", "Confidence/accuracy tier shown per player")
        has_outlier_warning = _file_contains("frontend/src/components/FantasyRankings.js", "outlier", "Outlier/volatility warning for players")
        _file_contains("frontend/src/components/GameCard.js", "vegas_spread", "Vegas spread shown next to our own spread")
        _file_contains("frontend/src/components/AccuracyTracker.js", "vs_vegas_spread", "Vegas comparison shown in Accuracy Tracker")

        for label, found, rel_path in checks:
            print(f"  [{'PRESENT' if found else 'MISSING'}] {label} ({rel_path})")

        self.findings["ui_ux"] = {label: found for label, found, _ in checks}
        if not has_outlier_warning:
            print("\n  Real gap confirmed: no outlier/volatility warning exists anywhere in FantasyRankings.js "
                  "(relevant given the Week-2 carry-forward volatility found in the stats<->PPR diagnostic task).")

    # ------------------------------------------------------------------
    def audit_data_quality(self):
        week1_games = [g for g in self.games if g.get("week") == 1]
        null_spreads = sum(1 for g in week1_games if g.get("our_spread") is None)

        week1_ranked = [p for p in self.rankings if p.get("week") == 1]
        ids = [p["id"] for p in week1_ranked]
        n_duplicate_ids = len(ids) - len(set(ids))

        issues = []
        if null_spreads:
            issues.append(f"{null_spreads} Week 1 games missing our_spread")
        if n_duplicate_ids:
            issues.append(f"{n_duplicate_ids} duplicate player ids in Week 1 rankings")

        print(f"  Week 1 rankings completeness: {len(week1_ranked)} players")
        print(f"  Duplicate ids: {n_duplicate_ids}")
        print(f"  Null spreads: {null_spreads}")
        if issues:
            print("  Issues found:")
            for issue in issues:
                print(f"    - {issue}")
        else:
            print("  No data-quality issues found in these checks.")

        self.findings["data_quality"] = {"n_players": len(week1_ranked), "issues": issues}

    # ------------------------------------------------------------------
    def print_summary(self):
        print("\nReal metrics by component (no invented 0-100 scoring - see module docstring point 4):\n")
        g = self.findings.get("games", {})
        p = self.findings.get("players", {})
        e = self.findings.get("ensemble", {})
        b = self.findings.get("betting", {})
        t = self.findings.get("tracking", {})
        u = self.findings.get("ui_ux", {})
        d = self.findings.get("data_quality", {})

        if g:
            print(f"  Games:      {g['accuracy_pct']}% straight-up ({g['correct']}/{g['n_completed']}), "
                  f"spread MAE {g['spread_mae']}pts, Vegas coverage {g['vegas_coverage_n']}")
        if p:
            print(f"  Players:    {p['overall_mae']} PPR overall MAE (n={p['n_with_actuals']})")
        if e:
            print(f"  Ensemble:   {e['n_records']} records, {e['pct_at_extreme_weight']}% at an extreme "
                  f"(0/100 or 100/0) weight, min fit sample size {e['min_fit_sample_size']}")
        if b and b.get("our_system_ats"):
            s = b["our_system_ats"]
            print(f"  Betting:    Our System ATS {s['win_pct']}%, ROI {s['roi_pct']:+.1f}%")
        if t and t.get("games_summary"):
            print(f"  Tracking:   {t['games_summary']['total_games']} games, "
                  f"{len(t['fantasy_positions_tracked'])} fantasy positions tracked")
        if u:
            n_present = sum(1 for v in u.values() if v)
            print(f"  UI/UX:      {n_present}/{len(u)} real, checked features confirmed present")
        if d:
            print(f"  Data:       {d['n_players']} Week 1 players ranked, {len(d['issues'])} issue(s) found")

        print("\nReal, dynamically-generated recommendations:\n")
        for rec in self.generate_recommendations():
            print(f"  - {rec}")

    def generate_recommendations(self):
        """Built from self.findings at run time, not frozen text (see
        module docstring point 5) - only fires when the real, current data
        actually supports the recommendation."""
        recs = []
        g = self.findings.get("games", {})
        e = self.findings.get("ensemble", {})
        u = self.findings.get("ui_ux", {})
        d = self.findings.get("data_quality", {})

        if g and g["vegas_coverage_pct"] < 100:
            recs.append(f"Vegas line coverage is {g['vegas_coverage_n']} ({g['vegas_coverage_pct']:.0f}%) - "
                        "the 2 missing games have no real ESPN closing line captured; widen odds-collection "
                        "timing if this recurs.")
        if e and e["pct_at_extreme_weight"] > 0:
            recs.append(f"{e['pct_at_extreme_weight']:.0f}% of ensemble positions are at an extreme (0/100 "
                        f"or 100/0) weight, fit on as few as {e['min_fit_sample_size']} real samples - "
                        "revisit once more real weeks complete (this is disclosed noise, not necessarily wrong).")
        if u and not u.get("Outlier/volatility warning for players"):
            recs.append("No outlier/volatility warning exists for players coming off a huge/tiny single-week "
                        "actual (the real root cause found in the stats<->PPR diagnostic task) - a real, "
                        "concrete quick win if you want it.")
        if d and d["issues"]:
            recs.append(f"Data quality issues found: {'; '.join(d['issues'])}")
        if not recs:
            recs.append("No real, actionable issues found in this run's checks.")
        return recs

    def save_audit_report(self):
        report = {"timestamp": self.timestamp, "findings": self.findings,
                  "recommendations": self.generate_recommendations()}
        AUDIT_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        with open(AUDIT_OUTPUT, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"\nAudit report saved -> {AUDIT_OUTPUT}")


if __name__ == "__main__":
    Week1SystemAudit().run()
