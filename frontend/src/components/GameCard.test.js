import React from 'react';
import { render, screen } from '@testing-library/react';
import GameCard from './GameCard';

// AUDIT_2026-08-12_DEEP.md Section 9 / Recommendation 10. Verified:
// `npm test` actually runs clean (3/3 assertions pass). Uses this
// project's real field names (home_team, away_team, our_spread,
// ci_low_90/ci_high_90, etc.), not the approximate `home_spread`/
// `model_confidence` names from the original pasted spec, which don't
// exist anywhere in this project's real games_2025.json/games_2026.json
// schema.
const mockGame = {
  id: '2026_01_BAL_KC',
  week: 1,
  weekday: 'Thursday',
  kickoff_datetime: '2026-09-10T20:20:00',
  home_team: 'KC',
  away_team: 'BAL',
  home_qb_name: null,
  away_qb_name: null,
  home_elo: 1550.0,
  away_elo: 1520.0,
  our_spread: 3.5,
  ci_low_90: -2.0,
  ci_high_90: 9.0,
  vegas_spread: null,
  win_prob_home: 0.62,
  win_prob_away: 0.38,
  base_source: 'elo',
  net_edge_diff: null,
  matchup_quality: null,
  home_recent_form: [],
  away_recent_form: [],
  head_to_head: null,
  actual_home_score: null,
  actual_away_score: null,
  actual_winner: null,
  actual_spread_margin: null,
  did_we_predict_correctly: null,
};

describe('GameCard', () => {
  test('renders both real team codes', () => {
    render(<GameCard game={mockGame} isExpanded={false} onToggle={() => {}} />);
    // Real: home_team/away_team render as isolated text inside their own
    // .team-box span (GameCard.js:189-207) - clean single-text-node match.
    expect(screen.getByText('KC')).toBeInTheDocument();
    expect(screen.getByText('BAL')).toBeInTheDocument();
  });

  test('renders the real formatted spread', () => {
    const { container } = render(<GameCard game={mockGame} isExpanded={false} onToggle={() => {}} />);
    // formatSpread(3.5, 'KC', 'BAL') -> real sign convention (positive =
    // home favored, GameCard.js:4-6) -> "KC -3.5". Checked against
    // container text rather than getByText, since "Our" and the spread
    // value are separate text nodes inside the same div.
    expect(container.textContent).toContain('KC -3.5');
  });

  test('handles a minimal/incomplete game object without throwing', () => {
    // Real, disclosed gap from AUDIT_2026-08-12_DEEP.md Section 4.2:
    // formatSpread has no null guard on `spread` itself, so this renders a
    // real "null -NaN"-style string rather than crashing - this test
    // verifies the non-crash behavior, not the display quality.
    expect(() => {
      render(<GameCard game={{ week: 1 }} isExpanded={false} onToggle={() => {}} />);
    }).not.toThrow();
  });
});
