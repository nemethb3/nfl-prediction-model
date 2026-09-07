"""Real project readiness audit - checks real files/schemas against what
this project actually has, not the assumed shapes a pasted spec had.

Real corrections made vs. the originally pasted spec before writing this
(all verified directly this session):

1. power_rankings_2026.json / superbowl_odds_2026.json / award_races_2026.json
   are real DICTS with a nested list (`teams` / `candidates`), not bare
   top-level lists - `len(json.load(f))` on those gives the wrong number
   (top-level key count, e.g. 4-5), not the real per-team/per-candidate
   count. Fixed to look inside the real nested key for each file.
2. award_races_2026.json is MVP-only (see DECISIONS_LOG.md - DPOY/OROY/
   DROY have no real per-player projection data anywhere in this
   project), not "4 awards" as the spec assumed.
3. `team_elo_2026.json` doesn't exist - the real 2026 Elo file is
   power_rankings_2026.json (already checked above).
4. Required backend modules: dropped `build_team_elo_2026.py` (no such
   file - real Elo pipeline is compute_offensive_defensive_elo.py /
   apply_season_regression_od_elo.py / generate_power_rankings_2026.py),
   `league_settings.py` and `lineup_optimizer.py` (both real FEATURES,
   but real client-side JS - frontend/src/utils/sleeperLeagueSettings.js
   and lineupOptimizer.js - not Python; see RUNNING_INSTRUCTIONS.md).
   Added orchestrate_2026_pipeline.py/update_rosters_2026.py/
   generate_trade_scores_2026.py/generate_mvp_race_2026.py instead - the
   real scripts a deployment actually depends on.
5. Python syntax check: `python -m py_compile src/*.py` with shell=True
   doesn't glob on Windows (cmd.exe doesn't expand `*`) - this machine is
   Windows. Fixed to iterate real files via pathlib instead.
6. Dependency check: dropped `pulp` (not a real dependency - real
   lineup optimization is client-side JS, not ILP; see
   RUNNING_INSTRUCTIONS.md). Matches requirements.txt instead.
7. Also warns if this script isn't running under the real project venv
   (D:/venvs/nfl-model) - real, previously-hit failure mode: `python` on
   this machine's PATH resolves to a stale miniconda install missing
   nflreadpy/etc, which would make every dependency check below fail for
   an unrelated reason.
"""

import importlib
import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "frontend" / "src" / "data"


class ProjectAudit:
    def __init__(self):
        self.errors = []
        self.warnings = []
        self.passed = []

    def run_all(self):
        print("Starting real project readiness audit...\n")
        self.check_interpreter()
        self.check_frontend_data()
        self.check_backend_modules()
        self.check_python_syntax()
        self.check_dependencies()
        self.check_git_status()
        self.print_report()

    def check_interpreter(self):
        print("Checking Python interpreter...")
        exe = sys.executable.replace("\\", "/")
        if "nfl-model" in exe.lower() or "venv" in exe.lower():
            self.passed.append(f"Running under project venv: {exe}")
        else:
            self.warnings.append(
                f"Running under {exe} - if this isn't the project venv "
                "(D:/venvs/nfl-model/Scripts/python.exe on this machine), "
                "the dependency checks below may fail for the wrong reason."
            )

    def check_frontend_data(self):
        print("Checking frontend data...")
        # (filename, real path-to-the-countable-list inside the JSON,
        # real minimum count, real description)
        checks = [
            ("games_2026.json", None, 272, "real 2026 REG-season games"),
            ("fantasy_rankings_2026.json", None, 300, "real ranked player-weeks (Week 1 only right now)"),
            ("player_props_2026.json", None, 5000, "real per-player-week stat projections"),
            ("season_projections_2026.json", None, 32, "real teams"),
            ("power_rankings_2026.json", "teams", 32, "real teams"),
            ("superbowl_odds_2026.json", "teams", 32, "real teams"),
            ("award_races_2026.json", "candidates", 1, "real MVP candidates (MVP-only, see DECISIONS_LOG.md)"),
            ("sleeper_id_mapping.json", None, 100, "real mapped players"),
            ("trade_scores_2026.json", "players", 100, "real scored players"),
        ]

        for filename, nested_key, min_count, desc in checks:
            path = DATA_DIR / filename
            if not path.exists():
                self.errors.append(f"Missing: {filename}")
                continue
            try:
                with open(path, encoding="utf-8") as f:
                    data = json.load(f)
                container = data[nested_key] if nested_key else data
                count = len(container)
                if count < min_count:
                    self.warnings.append(f"{filename}: {count} {desc} (expected {min_count}+)")
                else:
                    self.passed.append(f"{filename}: {count} {desc}")
            except Exception as e:
                self.errors.append(f"Error reading {filename}: {e}")

    def check_backend_modules(self):
        print("Checking backend modules...")
        required_modules = [
            "compute_offensive_defensive_elo.py",
            "apply_season_regression_od_elo.py",
            "generate_power_rankings_2026.py",
            "build_player_props_signals.py",
            "train_player_props_models.py",
            "generate_fantasy_rankings_2026_week1.py",
            "simulate_2026_playoffs.py",
            "orchestrate_2026_pipeline.py",
            "update_rosters_2026.py",
            "espn_odds_client.py",
            "espn_odds_orchestrate.py",
            "generate_trade_scores_2026.py",
            "generate_mvp_race_2026.py",
        ]
        for module in required_modules:
            if (PROJECT_ROOT / "src" / module).exists():
                self.passed.append(module)
            else:
                self.errors.append(f"Missing backend module: {module}")

    def check_python_syntax(self):
        print("Checking Python syntax...")
        py_files = sorted((PROJECT_ROOT / "src").glob("*.py"))
        failed = []
        for f in py_files:
            result = subprocess.run(
                [sys.executable, "-m", "py_compile", str(f)],
                capture_output=True, text=True,
            )
            if result.returncode != 0:
                failed.append((f.name, result.stderr.strip()))
        if failed:
            for name, err in failed:
                self.errors.append(f"Syntax error in {name}: {err}")
        else:
            self.passed.append(f"All {len(py_files)} files in src/ compile")

    def check_dependencies(self):
        print("Checking dependencies...")
        # Matches requirements.txt (import names, not pip package names -
        # nfl-data-py's real import name is nfl_data_py).
        packages = ["nfl_data_py", "nflreadpy", "pandas", "numpy", "sklearn", "scipy", "requests"]
        for package in packages:
            try:
                importlib.import_module(package)
                self.passed.append(package)
            except ImportError:
                self.errors.append(f"Missing package: {package}")

    def check_git_status(self):
        print("Checking git status...")
        try:
            result = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=PROJECT_ROOT, capture_output=True, text=True,
            )
            if result.returncode == 0:
                if result.stdout.strip():
                    self.warnings.append(f"Uncommitted changes:\n{result.stdout}")
                else:
                    self.passed.append("Git working tree clean")
            else:
                self.warnings.append("Could not check git status")
        except Exception as e:
            self.warnings.append(f"Git check failed: {e}")

    def print_report(self):
        print("\n" + "=" * 60)
        print("AUDIT REPORT")
        print("=" * 60)

        if self.passed:
            print(f"\nPASSED ({len(self.passed)}):")
            for msg in self.passed:
                print(f"  OK  {msg}")

        if self.warnings:
            print(f"\nWARNINGS ({len(self.warnings)}):")
            for msg in self.warnings:
                print(f"  !!  {msg}")

        if self.errors:
            print(f"\nERRORS ({len(self.errors)}):")
            for msg in self.errors:
                print(f"  XX  {msg}")

        print("\n" + "=" * 60)
        if not self.errors:
            print("READY (no blocking errors - see warnings above for anything non-fatal)")
        else:
            print("NOT READY - fix errors above")
        print("=" * 60)


if __name__ == "__main__":
    ProjectAudit().run_all()
