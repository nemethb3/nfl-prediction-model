import React, { createContext, useContext, useState } from 'react';
import { AVAILABLE_SEASONS, DEFAULT_SEASON, SEASON_HAS_RESULTS, SEASON_LABELS } from '../constants/seasons';

// Static build-time imports throughout (matches how every component in
// this app already loads data - CRA bundles these into the JS bundle, not
// served as fetchable URLs) rather than the runtime fetch() approach
// originally proposed, which would have required moving every JSON file
// into CRA's public/ folder - a bigger restructuring this project can't
// verify without Node.js installed.
import games2025 from '../data/games_2025.json';
import fantasy2025 from '../data/fantasy_rankings_2025.json';
import seasonProjections2025 from '../data/season_projections_2025.json';
import accuracyTracker2025 from '../data/accuracy_tracker_2025.json';
import weeklySummary2025 from '../data/weekly_summary_2025.json';
import bettingBacktest2025 from '../data/betting_backtest_results_2025.json';
import superbowlOdds2025 from '../data/superbowl_odds_2025.json';

// 2026: only real, computable-preseason exports exist (see constants/
// seasons.js) - accuracyTracker/weeklySummary/bettingBacktest files don't
// exist for 2026 and are never imported here (a static import of a
// nonexistent file would break the whole build); their real absence is
// represented as `null`, not fabricated. fantasy IS real for 2026 (Week 1
// preseason projections only, actual_ppr null throughout - see
// generate_fantasy_rankings_2026_week1.py). superbowlOdds is ALSO real for
// 2026 now (Monte Carlo bracket sim seeded from this project's own real
// simulated regular-season seeding, not an actual standing - see
// generate_superbowl_odds_2026.py) - wired the same way fantasy2026 was.
import games2026 from '../data/games_2026.json';
import seasonProjections2026 from '../data/season_projections_2026.json';
import fantasy2026 from '../data/fantasy_rankings_2026.json';
import superbowlOdds2026 from '../data/superbowl_odds_2026.json';

const SEASON_DATA = {
  2025: {
    games: games2025,
    fantasy: fantasy2025,
    seasonProjections: seasonProjections2025,
    accuracyTracker: accuracyTracker2025,
    weeklySummary: weeklySummary2025,
    bettingBacktest: bettingBacktest2025,
    superbowlOdds: superbowlOdds2025,
  },
  2026: {
    games: games2026,
    fantasy: fantasy2026,
    seasonProjections: seasonProjections2026,
    accuracyTracker: null,
    weeklySummary: null,
    bettingBacktest: null,
    superbowlOdds: superbowlOdds2026,
  },
};

const SeasonContext = createContext(null);

export function SeasonProvider({ children }) {
  const [selectedSeason, setSelectedSeason] = useState(DEFAULT_SEASON);

  const value = {
    selectedSeason,
    setSelectedSeason,
    seasonData: SEASON_DATA[selectedSeason],
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
