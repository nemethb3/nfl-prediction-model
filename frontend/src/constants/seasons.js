// Real fact, verified against data/raw/schedules_2026.csv before building
// this: the 2026 NFL season has not been played yet (real season opener
// 2026-09-09; 0/272 real 2026 games have a score as of this build). So
// unlike 2025 (a fully-completed real season), 2026 can only ever show
// real PRESEASON data - no real accuracy, fantasy outcomes, weekly recap,
// or betting backtest exist for it yet. SEASON_HAS_RESULTS drives which
// sections render normally vs. show a real "not available yet" message
// instead of an empty or broken panel.
export const AVAILABLE_SEASONS = [2026, 2025];
export const DEFAULT_SEASON = 2026;

export const SEASON_LABELS = {
  2025: '2025 Season (Final)',
  2026: '2026 Season (Preseason)',
};

export const SEASON_HAS_RESULTS = {
  2025: true,
  2026: false,
};
