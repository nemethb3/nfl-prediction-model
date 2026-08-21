import React, { useEffect, useState } from 'react';
import { useSeason } from '../context/SeasonContext';
import sleeperIdMapping from '../data/sleeper_id_mapping.json';
import { fetchSleeperLeague, parseScoringSettings, parseRosterPositions } from '../utils/sleeperLeagueSettings';
import { adjustRosterForLeague } from '../utils/adjustProjectionsForLeague';
import { optimizeLineup } from '../utils/lineupOptimizer';
import '../styles/SuggestedLineups.css';

const SLEEPER_BASE = 'https://api.sleeper.app/v1';

export default function SuggestedLineups() {
  const { seasonData, selectedSeason } = useSeason();
  const [state, setState] = useState({ status: 'idle' }); // idle | loading | error | ready

  const savedUsername = typeof window !== 'undefined' ? localStorage.getItem('sleeper_username') : null;
  const savedLeagueId = typeof window !== 'undefined' ? localStorage.getItem('sleeper_league_id') : null;

  useEffect(() => {
    if (!savedUsername || !savedLeagueId) {
      setState({ status: 'not-connected' });
      return;
    }
    // Real, disclosed gap (same pattern as playerProps/breakoutAlerts/
    // rookieScores/powerRankings elsewhere in this app): the per-stat
    // projections this feature needs only exist for 2026.
    if (!seasonData.playerProps) {
      setState({ status: 'unavailable-season' });
      return;
    }

    let cancelled = false;
    setState({ status: 'loading' });

    (async () => {
      try {
        const userResponse = await fetch(`${SLEEPER_BASE}/user/${savedUsername}`);
        if (!userResponse.ok) throw new Error('Sleeper user not found');
        const user = await userResponse.json();

        const [league, rostersResponse] = await Promise.all([
          fetchSleeperLeague(savedLeagueId),
          fetch(`${SLEEPER_BASE}/league/${savedLeagueId}/rosters`),
        ]);
        if (!rostersResponse.ok) throw new Error('Failed to fetch rosters');
        const rosters = await rostersResponse.json();
        const userRoster = rosters.find((r) => r.owner_id === user.user_id);
        if (!userRoster) throw new Error('Roster not found for this user in this league');

        if (cancelled) return;

        const scoring = parseScoringSettings(league.scoring_settings);
        const { slotCounts, unsupportedSlots } = parseRosterPositions(league.roster_positions);

        const rosterIds = userRoster.players || [];
        const mappedPlayers = rosterIds
          .map((sleeperId) => sleeperIdMapping[sleeperId])
          .filter(Boolean); // real, deliberate scope: DEF/K/unmapped IDs are never in this crosswalk

        const excludedCount = rosterIds.length - mappedPlayers.length;

        const fantasyData = seasonData.fantasy || [];
        const targetWeek = fantasyData.length > 0 ? Math.max(...fantasyData.map((p) => p.week || 1)) : 1;

        const allProps = seasonData.playerProps || [];
        const weeklyProps = allProps.filter((p) => p.week === targetWeek);

        const adjustedRoster = adjustRosterForLeague(mappedPlayers, weeklyProps, targetWeek, scoring);
        const withProjection = adjustedRoster.filter((p) => typeof p.leaguePoints === 'number');
        const withoutProjection = adjustedRoster.filter((p) => typeof p.leaguePoints !== 'number');

        const lineup = optimizeLineup(withProjection, slotCounts);

        setState({
          status: 'ready',
          leagueName: league.name,
          scoring,
          slotCounts,
          unsupportedSlots,
          targetWeek,
          lineup,
          withoutProjection,
          excludedCount,
        });
      } catch (err) {
        if (!cancelled) setState({ status: 'error', message: err.message });
      }
    })();

    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [savedUsername, savedLeagueId, seasonData]);

  if (state.status === 'not-connected') {
    return (
      <div className="suggested-lineups suggested-lineups--empty">
        <p>Connect a Sleeper league first (League Connection tab) to get a suggested lineup.</p>
      </div>
    );
  }

  if (state.status === 'unavailable-season') {
    return (
      <div className="suggested-lineups suggested-lineups--empty">
        <p>Suggested lineups aren&apos;t available for {selectedSeason} - the real per-stat projections this feature needs only exist for 2026.</p>
      </div>
    );
  }

  if (state.status === 'loading' || state.status === 'idle') {
    return <div className="suggested-lineups suggested-lineups--loading">Loading your league and roster...</div>;
  }

  if (state.status === 'error') {
    return <div className="suggested-lineups suggested-lineups--error">{state.message}</div>;
  }

  const { leagueName, unsupportedSlots, targetWeek, lineup, withoutProjection, excludedCount } = state;
  const unsupportedSlotNames = Object.keys(unsupportedSlots || {});

  return (
    <div className="suggested-lineups">
      <h2>Suggested Lineup</h2>
      <p className="suggested-lineups__meta">
        {leagueName} - real week {targetWeek} projections{selectedSeason === 2026 ? ' (2026 preseason)' : ''}, scored
        using your real league&apos;s scoring settings from Sleeper.
      </p>

      <div className="suggested-lineups__total">
        {lineup.totalPoints.toFixed(1)} <span>projected PPR points (starters only)</span>
      </div>

      <div className="suggested-lineups__section">
        <h3>Starting</h3>
        <div className="suggested-lineups__list">
          {lineup.starting.map((s) => (
            <div key={s.slot} className="suggested-lineups__row">
              <span className="suggested-lineups__slot">{s.slot}</span>
              <span className="suggested-lineups__name">{s.player.name}</span>
              <span className="suggested-lineups__pos">{s.player.position}</span>
              <span className="suggested-lineups__pts">{s.player.leaguePoints.toFixed(1)}</span>
            </div>
          ))}
        </div>
        {lineup.unfilled.length > 0 && (
          <p className="suggested-lineups__warning">
            Could not fill every slot from your real roster: {lineup.unfilled.map((u) => `${u.slot} (missing ${u.missing})`).join(', ')}.
          </p>
        )}
      </div>

      {lineup.bench.length > 0 && (
        <div className="suggested-lineups__section">
          <h3>Bench</h3>
          <div className="suggested-lineups__list">
            {lineup.bench.map((p) => (
              <div key={p.player_id} className="suggested-lineups__row suggested-lineups__row--bench">
                <span className="suggested-lineups__name">{p.name}</span>
                <span className="suggested-lineups__pos">{p.position}</span>
                <span className="suggested-lineups__pts">{p.leaguePoints.toFixed(1)}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      <div className="suggested-lineups__disclaimer">
        <p>
          QB/RB/WR/TE only - this project has no real defense or kicker model, so those roster
          slots (and any other slot type Sleeper reports beyond QB/RB/WR/TE/FLEX
          {unsupportedSlotNames.length > 0 ? ` - your league also has: ${unsupportedSlotNames.join(', ')}` : ''})
          aren&apos;t included here. Points come from this project&apos;s real, opponent-adjusted
          per-stat projections, rescored using your real league&apos;s Sleeper scoring settings -
          not this project&apos;s own default PPR total. Interception-based scoring (if your league
          uses it) isn&apos;t applied: this project has no real interception projection to score
          against.
          {excludedCount > 0 &&
            ` ${excludedCount} roster spot${excludedCount === 1 ? '' : 's'} excluded (kickers/defenses/unmapped players).`}
          {withoutProjection.length > 0 &&
            ` ${withoutProjection.length} rostered player${withoutProjection.length === 1 ? '' : 's'} (${withoutProjection
              .map((p) => p.name)
              .join(', ')}) outside this project's real ~390 ranked players this week - not included in the optimization.`}
        </p>
      </div>
    </div>
  );
}
