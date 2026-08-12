export const DASHBOARD_SECTIONS = [
  {
    id: 'games',
    label: 'Weekly Games',
    description: 'Game predictions and Vegas comparison',
    component: 'GamePredictions',
  },
  {
    id: 'fantasy',
    label: 'Fantasy Rankings',
    description: 'Player rankings by position',
    component: 'FantasyRankings',
  },
  {
    id: 'projections',
    label: 'Season Projections',
    description: 'Team wins and playoff odds',
    component: 'SeasonProjections',
  },
  {
    id: 'accuracy',
    label: 'Accuracy Tracker',
    description: 'Model performance vs Vegas',
    component: 'AccuracyTracker',
  },
  {
    id: 'summary',
    label: 'Weekly Summary',
    description: 'Key insights and takeaways',
    component: 'WeeklySummary',
  },
  {
    id: 'transparency',
    label: 'Model Transparency',
    description: 'Component breakdown and methodology',
    component: 'ModelTransparency',
  },
  {
    id: 'betting',
    label: 'Betting Analysis',
    description: 'Strategy backtests vs. real Vegas odds',
    component: 'BettingAnalysis',
  },
  {
    id: 'sleeper',
    label: 'My League',
    description: 'Connect your Sleeper league and match your roster to real projections',
    component: 'LeagueConnector',
  },
  {
    id: 'trade-analyzer',
    label: 'Trade Analyzer',
    description: 'Multi-signal, backtested year-over-year trajectory model',
    component: 'TradeAnalyzer',
  },
];

export const DEFAULT_SECTION = 'games';
