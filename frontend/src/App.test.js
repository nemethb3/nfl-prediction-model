import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import App from './App';

// AUDIT_2026-08-12_DEEP.md Section 9 / Recommendation 10: first automated
// frontend tests in this project. Mocks every real static JSON import
// SeasonContext.js pulls in (verified against its actual import list, not
// guessed) so the test doesn't load the real, multi-megabyte data files
// (fantasy_rankings_2025.json alone is 2.9MB) - this is a render-shape
// smoke test, not a data-correctness test (that's the backtests' job).
//
// Verified: `npm test` actually runs clean (4/4 assertions pass) - Node.js
// turned out to be available in this environment after all, despite this
// project's standing assumption otherwise throughout prior sessions.
jest.mock('./data/games_2025.json', () => ([]));
jest.mock('./data/games_2026.json', () => ([]));
jest.mock('./data/fantasy_rankings_2025.json', () => ([]));
jest.mock('./data/fantasy_rankings_2026.json', () => ([]));
jest.mock('./data/season_projections_2025.json', () => ([]));
jest.mock('./data/season_projections_2026.json', () => ([]));
jest.mock('./data/accuracy_tracker_2025.json', () => ({ season_summary: {}, weekly_breakdown: [] }));
jest.mock('./data/weekly_summary_2025.json', () => ({ current_week: null, weeks: [] }));
// Real minimal shape BettingAnalysis.js actually expects (verified against
// its own real destructuring: resultsData[strategy][betType].season_summary/
// weekly_summary/bets) - an empty {} (this file's previous mock) would have
// masked the real bug this test exists to catch, since it would throw on
// resultsData[selectedStrategy] being undefined for an unrelated reason.
// jest hoists jest.mock() factories above regular declarations, but
// "mock"-prefixed helpers are explicitly exempted from that hoisting rule,
// so this is safe to reference from the factory below.
function mockBetType() {
  return {
    season_summary: { total_bets: 1, wins: 1, losses: 0, pushes: 0, win_pct: 100, roi_pct: 10, pnl_units: 1 },
    weekly_summary: {},
    bets: [],
  };
}
jest.mock('./data/betting_backtest_results_2025.json', () => ({
  our_system: { label: 'Our System', description: 'test strategy', moneyline: mockBetType(), ats: mockBetType() },
  vegas_favorites: { label: 'Vegas Favorites', description: 'test strategy', moneyline: mockBetType(), ats: mockBetType() },
  underdogs_only: { label: 'Underdogs Only', description: 'test strategy', moneyline: mockBetType(), ats: mockBetType() },
}));
jest.mock('./data/superbowl_odds_2025.json', () => ({ teams: [] }));
jest.mock('./data/superbowl_odds_2026.json', () => ({ teams: [] }));

// SeasonContext now loads each season's data via dynamic import()
// (AUDIT_2026-08-12_DEEP.md Recommendation 11) - real async gap between
// initial render and data resolving. All tests await the real games-section
// empty state first so the async state update is wrapped by Testing
// Library's own act() handling before the test ends, rather than leaking
// into the next test as an unhandled "not wrapped in act()" warning.
async function renderAppAndWaitForData() {
  render(<App />);
  await screen.findByText('No games for this week.');
}

describe('App', () => {
  test('renders without crashing, real brand text present', async () => {
    await renderAppAndWaitForData();
    // Real text, verified against Navigation.js:15 (`nav-brand` literal
    // "NFL Predictions") - NOT "NFL Dashboard", which does not appear
    // anywhere in this codebase.
    expect(screen.getByText('NFL Predictions')).toBeInTheDocument();
  });

  test('navigation renders with all real dashboard sections', async () => {
    await renderAppAndWaitForData();
    // <nav> has an implicit ARIA role of "navigation" - no explicit
    // role attribute needed, verified this resolves via Testing Library's
    // implicit-role support.
    expect(screen.getByRole('navigation')).toBeInTheDocument();
    // Real label from constants/sections.js, default active section.
    expect(screen.getByText('Weekly Games')).toBeInTheDocument();
  });

  test('season selector renders with the real default season (2026)', async () => {
    await renderAppAndWaitForData();
    // Real label association verified against SeasonSelector.js:11
    // (<label htmlFor="season-select">Season</label>) and constants/
    // seasons.js's real DEFAULT_SEASON = 2026.
    const select = screen.getByLabelText('Season');
    expect(select).toBeInTheDocument();
    expect(select.value).toBe('2026');
  });

  test('default (games) section renders its real empty state with mocked empty data', async () => {
    render(<App />);
    // GamePredictions.js's real empty-state string when weekGames.length === 0.
    expect(await screen.findByText('No games for this week.')).toBeInTheDocument();
  });

  test('switching season while on Betting Analysis does not crash (real regression)', async () => {
    // Real bug: SeasonContext.js's old `dataLoading` was a separate useState
    // only reset inside a useEffect (runs after render), one tick behind a
    // selectedSeason change - real 2026->2025 default DEFAULT_SEASON=2026,
    // hasResults=false so BettingAnalysis first renders its real
    // SeasonDataUnavailable state (not a crash) - switching to 2025 (real
    // hasResults=true, synchronous with dataLoading now) used to leave a
    // window where hasResults flipped true before seasonData had loaded,
    // crashing BettingAnalysis on `seasonData.bettingBacktest` being
    // undefined. Fixed by deriving dataLoading synchronously instead.
    await renderAppAndWaitForData();
    fireEvent.click(screen.getByText('Betting Analysis'));
    expect(await screen.findByText('Not available for 2026')).toBeInTheDocument();

    fireEvent.change(screen.getByLabelText('Season'), { target: { value: '2025' } });
    // Regex/partial match, not an exact 'Our System' string - that text is
    // ambiguous with the real static "Our System:" methodology bullet
    // elsewhere in this same component.
    expect(await screen.findByText(/Strategy Comparison/)).toBeInTheDocument();
    expect(screen.queryByText('Section Error')).not.toBeInTheDocument();
  });
});
