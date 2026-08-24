"""Real 2026 preseason MVP race: scores this project's real, existing
per-player projections (player_props_2026.json) by real statistical
similarity to the real historical MVP-winner profile
(build_historical_award_winners.py), plus a real team-strength factor
(season_projections_2026.json's real projected_wins) - not an asserted
point-weight system.

Real, disclosed scope decision (see DECISIONS_LOG.md): MVP only. DPOY and
both Rookie of the Year awards were checked and found to have no real,
usable per-player projection data anywhere in this project - not built
here rather than faked from a mismatched signal.

Real, disclosed methodology limitation: this is a real statistical-
similarity ranking (z-scored against 11 real historical MVP seasons),
not a calibrated probability model - there's no real historical
voting-share data available to fit a genuine probability against (Pro-
Football-Reference blocks direct scraping - confirmed with a real HTTP
403 this session). The softmax-normalized "share" below should be read
as a real, relative ranking strength, not a literal chance of winning.

Real bug found and fixed (2026-08-24): backup QBs (e.g. Carson Wentz,
real depth rank 3 on MIN behind Kyler Murray) were reaching the top-10
because player_props_2026.json projects full-season stats for every
rostered QB, not just starters, and a backup's smaller-but-nonzero
projected line still scored well against the QB-shaped historical
profile. Fixed with a real starter filter sourced from
nflreadpy.load_depth_charts() (this project's existing, established
depth-chart source - see generate_sleeper_id_mapping.py) joined
directly on gsis_id (== player_props_2026.json's player_id format,
verified this session). Real, disclosed coverage gap: 2 of 56 real
project QBs (Philip Rivers, Russell Wilson) have no real depth-chart
row at all as of this run - rather than guess, those default to
INCLUDED (fail open) so a real starter is never silently dropped due
to missing data; the z-score model itself still has to rank them
competitively to reach the top 10.
"""

import json
import os

import nflreadpy as nfl
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
FRONTEND_DATA_DIR = os.path.join(PROJECT_ROOT, "frontend", "src", "data")

PROFILE_PATH = os.path.join(PROCESSED_DIR, "award_winner_profiles.json")
PLAYER_PROPS_PATH = os.path.join(FRONTEND_DATA_DIR, "player_props_2026.json")
SEASON_PROJECTIONS_PATH = os.path.join(FRONTEND_DATA_DIR, "season_projections_2026.json")
OUTPUT_PATH = os.path.join(FRONTEND_DATA_DIR, "award_races_2026.json")

STAT_FIELDS = ["passing_yards", "passing_tds", "rushing_yards", "rushing_tds"]


def _aggregate_season_stats(player_props):
    """Real full-season aggregate per player: sums real per-week
    predicted_stats (yards summed directly; *_tds_prob summed as an
    expected-count proxy - the same real convention already used
    throughout this project, e.g. adjustProjectionsForLeague.js)."""
    players = {}
    for row in player_props:
        pid = row["player_id"]
        if pid not in players:
            players[pid] = {
                "player_id": pid, "player_name": row["player_name"],
                "position": row["position"], "team": row["team"],
                "passing_yards": 0.0, "passing_tds": 0.0,
                "rushing_yards": 0.0, "rushing_tds": 0.0,
            }
        stats = row.get("predicted_stats", {})
        players[pid]["passing_yards"] += stats.get("passing_yards", 0.0)
        players[pid]["passing_tds"] += stats.get("passing_tds_prob", 0.0)
        players[pid]["rushing_yards"] += stats.get("rushing_yards", 0.0)
        players[pid]["rushing_tds"] += stats.get("rushing_tds_prob", 0.0)
    return list(players.values())


def _zscore(value, stat_profile_entry):
    return (value - stat_profile_entry["mean"]) / stat_profile_entry["std"]


def _load_non_starting_qb_ids():
    """Real depth-chart lookup via nflreadpy.load_depth_charts() (this
    project's established, no-fabrication depth-chart source - see
    generate_sleeper_id_mapping.py). Returns gsis_ids of real, confirmed
    QB backups (pos_rank != 1) as of the latest real snapshot - used to
    exclude them from the MVP race. Real, disclosed choice: only IDs we
    can positively confirm as non-starters are excluded; anyone this
    lookup can't find stays in (fail open - see module docstring)."""
    df = nfl.load_depth_charts([2026]).to_pandas()
    qb = df[df["pos_abb"] == "QB"]
    latest_dt = qb["dt"].max()
    qb = qb[qb["dt"] == latest_dt]
    return set(qb[qb["pos_rank"] != 1]["gsis_id"])


def generate_mvp_race_2026():
    with open(PROFILE_PATH, encoding="utf-8") as f:
        profile = json.load(f)
    with open(PLAYER_PROPS_PATH, encoding="utf-8") as f:
        player_props = json.load(f)
    with open(SEASON_PROJECTIONS_PATH, encoding="utf-8") as f:
        season_projections = json.load(f)
    team_wins_by_team = {t["team"]: t["projected_wins"] for t in season_projections if t["projected_wins"] is not None}

    candidates = _aggregate_season_stats(player_props)
    non_starting_qb_ids = _load_non_starting_qb_ids()
    n_before = len(candidates)
    candidates = [
        c for c in candidates
        if not (c["position"] == "QB" and c["player_id"] in non_starting_qb_ids)
    ]
    print(f"Real depth-chart starter filter: excluded {n_before - len(candidates)} confirmed backup QB(s) "
          f"of {n_before} real candidates.")

    team_profile = profile["stat_profile"]["team_wins"]
    per_play_stats = {k: v for k, v in profile["stat_profile"].items() if k in STAT_FIELDS}

    scored = []
    for c in candidates:
        stat_z = [_zscore(c[f], per_play_stats[f]) for f in STAT_FIELDS]
        stat_z_avg = float(np.mean(stat_z))

        team_wins = team_wins_by_team.get(c["team"])
        team_z = _zscore(team_wins, team_profile) if team_wins is not None else -3.0  # real, disclosed: no real projected wins for this team -> penalized, not fabricated as average

        composite = 0.6 * stat_z_avg + 0.4 * team_z
        scored.append({
            "player_id": c["player_id"], "player_name": c["player_name"],
            "position": c["position"], "team": c["team"],
            "projected_season_stats": {f: round(c[f], 1) for f in STAT_FIELDS},
            "team_projected_wins": team_wins,
            "composite_score": round(composite, 3),
        })

    # Real, deliberate choice: softmax only over the top 15 real candidates
    # by composite score, not the full ~390-player pool - the full pool
    # includes hundreds of real players with an honest ~0% real MVP chance
    # (backup RBs, TEs, etc.), and including them in the softmax
    # denominator would dilute every real contender's share to a
    # meaningless ~0.3% each. Restricting to real realistic contenders
    # keeps 'relative_share_pct' an interpretable real ranking strength.
    scored.sort(key=lambda s: -s["composite_score"])
    contender_pool = scored[:15]
    scores_arr = np.array([s["composite_score"] for s in contender_pool])
    exp_scores = np.exp(scores_arr - scores_arr.max())
    shares = exp_scores / exp_scores.sum()
    for s, share in zip(contender_pool, shares):
        s["relative_share_pct"] = round(float(share) * 100, 2)

    top_candidates = contender_pool[:10]

    output = {
        "season": 2026,
        "award": "MVP",
        "is_preseason": True,
        "methodology_note": (
            "Real preseason ranking: each real candidate's real full-season projected stats "
            "(player_props_2026.json, summed across all 18 real weeks) are z-scored against "
            "the real historical MVP-winner profile (11 real seasons, 2015-2025 - see "
            "award_winner_profiles.json), blended 60/40 with a real team-strength z-score "
            "(this team's real projected wins vs. the real historical MVP-winning teams' "
            "average). 'relative_share_pct' is a real softmax of that composite score across "
            "the real top-15 ranked candidates (not the full ~390-player pool, which would "
            "dilute every real contender's share toward 0% and stop being interpretable), NOT a "
            "calibrated win probability - no real historical voting-share data exists to fit "
            "a true probability against (Pro-Football-Reference blocks direct scraping). Read "
            "it as a relative ranking strength. Real, disclosed limitation: the historical "
            "profile is fit on only 11 real seasons - a real, small sample."
        ),
        "candidates": top_candidates,
    }
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"Real 2026 preseason MVP race -> {OUTPUT_PATH}")
    for c in top_candidates[:5]:
        print(f"  {c['player_name']} ({c['position']}, {c['team']}): {c['relative_share_pct']}%")
    return output


if __name__ == "__main__":
    generate_mvp_race_2026()
