import React from 'react';
import { render, screen } from '@testing-library/react';
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
jest.mock('./data/betting_backtest_results_2025.json', () => ({}));
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
});
