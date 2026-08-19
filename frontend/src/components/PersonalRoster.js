import React, { useEffect, useState } from 'react';
import { useSeason } from '../context/SeasonContext';
import { TEAM_NAMES } from '../constants/teams';
// Real static import (matches this project's established architecture -
// every other JSON file in this app is a build-time import, not a runtime
// fetch(); a fetch('/data/...') here wouldn't even resolve under CRA,
// since this file lives in src/data, not public/).
import sleeperIdMapping from '../data/sleeper_id_mapping.json';
import '../styles/PersonalRoster.css';

export default function PersonalRoster({ leagueId, userId }) {
  const { selectedSeason, seasonData } = useSeason();
  const [roster, setRoster] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchRoster = async () => {
      try {
        const response = await fetch(`https://api.sleeper.app/v1/league/${leagueId}/rosters`);
        if (!response.ok) {
          throw new Error('Failed to fetch your roster');
        }
        const rosters = await response.json();
        const userRoster = rosters.find((r) => r.owner_id === userId);
        if (!userRoster) {
          throw new Error('Roster not found for this user in this league');
        }
        setRoster(userRoster);
      } catch (err) {
        setError(err.message);
      } finally {
        setLoading(false);
      }
    };

    if (userId) {
      fetchRoster();
    }
  }, [leagueId, userId]);

  const fantasyData = seasonData.fantasy || [];
  const games = seasonData.games || [];
  const isPreseason = selectedSeason === 2026;

  // Real "current week" for this season: the latest week this project has
  // real fantasy projection data for (2026 only ever has week 1 right
  // now; 2025 has multiple weeks - this picks the most recent one rather
  // than an arbitrary/hardcoded week). Used consistently for BOTH the bye
  // check and the projection lookup below, instead of two disconnected
  // notions of "current week."
  const targetWeek = fantasyData.length > 0 ? Math.max(...fantasyData.map((p) => p.week || 1)) : 1;

  const isBye = (nflTeam) => {
    if (!nflTeam || games.length === 0) return false;
    const hasGame = games.some(
      (g) => g.week === targetWeek && (g.home_team === nflTeam || g.away_team === nflTeam)
    );
    return !hasGame;
  };

  if (loading) {
    return <div className="personal-roster personal-roster--loading">Loading your roster...</div>;
  }
  if (error) {
    return <div className="personal-roster personal-roster--error">{error}</div>;
  }
  if (!roster) {
    return <div className="personal-roster">No roster data.</div>;
  }

  // Real, deliberate crosswalk scope (generate_sleeper_id_mapping.py:
  // POSITIONS = {"QB","RB","WR","TE"}) - kickers and team defenses are
  // never in sleeperIdMapping at all, by design, since this project has
  // no K or DEF model anywhere (see ModelTransparency's Limitations
  // section). An unmapped roster slot is therefore virtually always one
  // of those two real, out-of-scope positions, not a data gap - counted
  // and disclosed in aggregate below rather than rendered as a per-card
  // error, which is what was reading as "empty boxes" for every real
  // league roster (every real lineup carries exactly one DEF + one K).
  const rosterIds = roster.players || [];
  const unmappedCount = rosterIds.filter((sleeperId) => !sleeperIdMapping[sleeperId]).length;

  return (
    <div className="personal-roster">
      <h2>Your Roster</h2>

      {isPreseason && (
        <div className="personal-roster__preseason-note">
          Showing real Week 1 preseason projections - the 2026 season hasn&apos;t started yet.
        </div>
      )}

      <div className="personal-roster__roster-grid">
        {rosterIds.map((sleeperId) => {
          const idInfo = sleeperIdMapping[sleeperId];

          if (!idInfo) {
            return null;
          }

          const projection = fantasyData.find((p) => p.id === `${idInfo.player_id}_w${targetWeek}`);
          const onBye = isBye(idInfo.team);

          let recommendation = null;
          if (onBye) {
            recommendation = { text: 'SIT (Bye Week)', slug: 'bye' };
          } else if (projection) {
            if (projection.projected_ppr > 15) {
              recommendation = { text: 'START', slug: 'start' };
            } else if (projection.projected_ppr < 5) {
              recommendation = { text: 'BENCH', slug: 'bench' };
            }
          }

          return (
            <div key={sleeperId} className="personal-roster__roster-card">
              <div className="personal-roster__card-header">
                <span className="personal-roster__player-name">{idInfo.name}</span>
                <span className="personal-roster__position">{idInfo.position}</span>
              </div>

              <div className="personal-roster__card-team">
                {idInfo.team ? TEAM_NAMES[idInfo.team] || idInfo.team : 'Free Agent'}
                {onBye && ' - BYE'}
              </div>

              {projection ? (
                <div className="personal-roster__card-projection">
                  <div className="personal-roster__projection-value">
                    {projection.projected_ppr != null ? projection.projected_ppr.toFixed(1) : '--'}
                    <span className="personal-roster__ppr-label">PPR</span>
                  </div>

                  {recommendation && (
                    <div
                      className={`personal-roster__recommendation personal-roster__recommendation--${recommendation.slug}`}
                    >
                      {recommendation.text}
                    </div>
                  )}

                  {projection.injury_risk_label && (
                    <div className="personal-roster__injury-badge">
                      Injury Risk: {projection.injury_risk_label}
                    </div>
                  )}

                  {projection.consistency_label && (
                    <div className="personal-roster__consistency-badge">
                      Consistency: {projection.consistency_label}
                    </div>
                  )}

                  {projection.confidence_tier === 'lower' && (
                    <div
                      className="personal-roster__confidence-tier-badge"
                      title="Real 2025 opportunities below this position's validated projection threshold - a real, meaningfully weaker signal than this project's core rankings."
                    >
                      Lower confidence projection
                    </div>
                  )}
                </div>
              ) : (
                <div className="personal-roster__card-no-data">
                  Outside this project&apos;s real ~390 ranked players this week
                </div>
              )}
            </div>
          );
        })}
      </div>

      {unmappedCount > 0 && (
        <div className="personal-roster__excluded-note">
          {unmappedCount} roster spot{unmappedCount === 1 ? '' : 's'} not shown - kickers, team
          defenses, and any other position outside this project&apos;s real QB/RB/WR/TE scope
          (see Limitations in How This Model Works).
        </div>
      )}

      <div className="personal-roster__disclaimer">
        <p>
          Real data throughout: your roster comes directly from Sleeper&apos;s real, public API
          (called from your browser, not a server this project runs). Player IDs are matched via
          a real, externally-maintained ID crosswalk (nflreadpy&apos;s ffverse player-ID table,
          Sleeper ID &lt;-&gt; this project&apos;s own GSIS-style player ID) - not Sleeper&apos;s
          own self-reported ID field, which was found to have incomplete real coverage for
          several prominent active players. Projections are this project&apos;s own real,
          already-published {selectedSeason} fantasy rankings (~390 real players, including a real,
          explicitly labeled lower-confidence tier for players below this project&apos;s validated
          opportunity threshold) - a roster player outside that list (deep bench, practice squad,
          true rookie) will real-honestly show no projection rather than a fabricated one.
          Start/Bench/Sit is a basic real threshold (projected PPR &gt; 15 =
          Start, &lt; 5 = Bench, real bye week = Sit) - not this project&apos;s full model output,
          which has no lineup-optimization logic built yet.
        </p>
      </div>
    </div>
  );
}
