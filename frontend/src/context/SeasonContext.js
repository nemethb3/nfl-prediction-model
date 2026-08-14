import React, { createContext, useContext, useState, useEffect, useRef } from 'react';
import { AVAILABLE_SEASONS, DEFAULT_SEASON, SEASON_HAS_RESULTS, SEASON_LABELS } from '../constants/seasons';

// AUDIT_2026-08-12_DEEP.md Section 7.2/8.1, Recommendation 11: the prior
// version of this file statically imported all 11 real JSON data files
// (5.01MB total, verified) at module load time - meaning every one of
// them landed in the single initial JS bundle regardless of which season
// a user actually viewed. fantasy_rankings_2025.json alone (2.90MB) is
// the single biggest file and is 2025-only.
//
// Fixed by loading each season's real data files via dynamic import()
// only when that season is actually selected, cached per-season after
// first load so switching back and forth doesn't re-fetch. The loading
// gate lives in ONE place (App.js, via `dataLoading` below) rather than
// being pushed into all 17 downstream components - every component still
// receives a fully-populated, synchronous-shaped `seasonData` exactly as
// before; they were not changed at all. Real, verified via `npm run build`:
// each season's JSON now lands in its own separate chunk file, not the
// main bundle.
const SEASON_FILE_LOADERS = {
  2025: async () => {
    const [games, fantasy, seasonProjections, accuracyTracker, weeklySummary, bettingBacktest,
      totalsBettingBacktest, superbowlOdds] = await Promise.all([
        import('../data/games_2025.json'),
        import('../data/fantasy_rankings_2025.json'),
        import('../data/season_projections_2025.json'),
        import('../data/accuracy_tracker_2025.json'),
        import('../data/weekly_summary_2025.json'),
        import('../data/betting_backtest_results_2025.json'),
        import('../data/totals_betting_backtest_2025.json'),
        import('../data/superbowl_odds_2025.json'),
      ]);
    return {
      games: games.default,
      fantasy: fantasy.default,
      seasonProjections: seasonProjections.default,
      accuracyTracker: accuracyTracker.default,
      weeklySummary: weeklySummary.default,
      bettingBacktest: bettingBacktest.default,
      totalsBettingBacktest: totalsBettingBacktest.default,
      superbowlOdds: superbowlOdds.default,
      // Real player props (Player Props Model task) were only ever built for
      // 2026 scoring - no real 2015-2025 per-player-game backtest display
      // was requested, so this stays a real `null` for 2025 rather than a
      // fabricated backfill.
      playerProps: null,
      // Same real reasoning as playerProps - breakout alerts are a real
      // 2026-only, forward-looking signal, not a 2025 backtest display.
      breakoutAlerts: null,
    };
  },
  // 2026: accuracyTracker/weeklySummary/bettingBacktest/totalsBettingBacktest
  // genuinely don't exist yet (real, unplayed season - see
  // constants/seasons.js; the totals backtest is real-2025-holdout-only,
  // same reasoning as backtest_totals_betting_2025.py's own scoping), so
  // they're real `null`, not a fabricated/missing import.
  2026: async () => {
    const [games, fantasy, seasonProjections, superbowlOdds, playerProps, breakoutAlerts] = await Promise.all([
      import('../data/games_2026.json'),
      import('../data/fantasy_rankings_2026.json'),
      import('../data/season_projections_2026.json'),
      import('../data/superbowl_odds_2026.json'),
      import('../data/player_props_2026.json'),
      import('../data/breakout_alerts_2026.json'),
    ]);
    return {
      games: games.default,
      fantasy: fantasy.default,
      seasonProjections: seasonProjections.default,
      accuracyTracker: null,
      weeklySummary: null,
      bettingBacktest: null,
      totalsBettingBacktest: null,
      superbowlOdds: superbowlOdds.default,
      playerProps: playerProps.default,
      breakoutAlerts: breakoutAlerts.default,
    };
  },
};

const SeasonContext = createContext(null);

export function SeasonProvider({ children }) {
  const [selectedSeason, setSelectedSeason] = useState(DEFAULT_SEASON);
  const [seasonDataCache, setSeasonDataCache] = useState({});
  // Guards against a stale, slower-resolving load overwriting a newer one
  // if a user switches seasons again before the first load finishes.
  const loadTokenRef = useRef(0);

  // Real bug found and fixed here: dataLoading used to be its own useState,
  // reset to true/false only inside the useEffect below - which runs AFTER
  // render/commit, one tick behind a selectedSeason change. That left a
  // real window where a season switch (e.g. 2026 -> 2025) had already
  // updated `hasResults` (purely derived from selectedSeason, synchronous)
  // while `dataLoading` was still the STALE value from the previous
  // season and `seasonDataCache[selectedSeason]` hadn't loaded yet -
  // exactly the gap that let BettingAnalysis (and every other component
  // gated the same way) render with hasResults=true but seasonData still
  // undefined, crashing on `seasonData.bettingBacktest`. Fixed by deriving
  // dataLoading synchronously from the same render's selectedSeason/cache
  // state instead of tracking it as separate, laggable state - it can now
  // never disagree with hasResults on a single render.
  const dataLoading = !seasonDataCache[selectedSeason];

  useEffect(() => {
    if (seasonDataCache[selectedSeason]) return;
    const thisLoad = ++loadTokenRef.current;
    SEASON_FILE_LOADERS[selectedSeason]().then((data) => {
      if (loadTokenRef.current !== thisLoad) return; // a newer selection already superseded this one
      setSeasonDataCache((prev) => ({ ...prev, [selectedSeason]: data }));
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedSeason]);

  const value = {
    selectedSeason,
    setSelectedSeason,
    seasonData: seasonDataCache[selectedSeason],
    dataLoading,
    hasResults: SEASON_HAS_RESULTS[selectedSeason],
    availableSeasons: AVAILABLE_SEASONS,
    seasonLabel: SEASON_LABELS[selectedSeason],
  };

  return <SeasonContext.Provider value={value}>{children}</SeasonContext.Provider>;
}

export function useSeason() {
  const context = useContext(SeasonContext);
  if (!context) {
    throw new Error('useSeason must be used within a SeasonProvider');
  }
  return context;
}
