import React, { useState } from 'react';
import '../styles/ModelTransparency.css';

// Every specific number and example below was checked against this
// project's real, generated dashboard data before writing (games_2025.json,
// fantasy_rankings_2025.json, season_projections_2025.json,
// accuracy_tracker_2025.json) - the original draft of this content had
// several real errors caught in that pass:
//   - Fabricated model weighting percentages ("team rating 40%, QB 10%,
//     rest 5%") that don't correspond to anything this project actually
//     computes - the real spread is Vegas's closing line plus ONE real,
//     regression-fitted matchup adjustment, not a weighted blend of many
//     asserted factors. Rest and recent-form/momentum were both tested
//     (rest_tracking.py, momentum_weighting.py) and explicitly NOT wired
//     into predictions (zero effect / inconclusive real findings).
//   - QB correlation stated as 89% - the real number is 44.7%. WR/QB
//     numbers were also swapped/scrambled relative to the real per-
//     position correlations.
//   - "We find edges over Vegas" - contradicts this project's own real,
//     established finding (README Key Finding #2: Vegas wins at every
//     real checkpoint tested; edge_detection.py's real backtest of betting
//     on disagreements returned -36% ROI).
//   - A fabricated Buffalo-vs-Chicago game example - no such game exists
//     in the real 2025 schedule.
//   - Green Bay/Carolina score stated as "31-13" - the real score was
//     13-16 (Carolina by 3, not 18).
//   - A "Christian McCaffrey (WR) projected 12, actual 43" example - two
//     errors: McCaffrey is a real RB, and the real 43.3-point
//     overperformance belongs to Brock Bowers (TE, week 9, real projected
//     10.4).
//   - Las Vegas cited as "projected 7 wins, actual 4" as a bad-prediction
//     example - the real projected figure was 3.0 wins, and the real
//     final record was 3-14, making it one of the MORE accurate season
//     projections, not a bad one. Replaced with the real largest miss
//     (Green Bay: real projected 10.5, real final 9 wins).
const SECTIONS = [
  {
    id: 'overview',
    title: 'How It Works',
    icon: '🧠',
    paragraphs: [
      { h: 'What is this?' },
      { p: 'This is an NFL prediction system built and backtested on 10 years of real historical data (2015-2025): who wins each game, how many fantasy points players score, and each team’s projected final record and playoff odds.' },
      { h: 'The honest version of "why build this"' },
      { p: 'The real, tested finding behind this whole project is that Vegas’s own closing lines are very hard to beat. Several independent approaches were built and backtested against Vegas - a team-strength rating system (Elo), play-level efficiency stats (EPA), blends of the two, rest-day adjustments, recent-form/momentum weighting - and in every real backtest, Vegas alone came out on top or tied. Betting specifically on the games where this model disagreed with Vegas was tested directly (a dedicated real backtest) and came back at -36% ROI - actively worse than doing nothing.' },
      { p: 'So this isn’t a "beat the market" tool. It’s Vegas’s own closing line, plus one real, additional adjustment for a specific matchup effect that did hold up in backtesting (see "Game Predictions"), reported honestly including the parts that didn’t work.' },
      { h: 'What does it predict?' },
      { p: '1. Who wins games (winner + point spread)\n2. Fantasy points for individual players\n3. Final season win totals and playoff odds for every team' },
    ],
  },
  {
    id: 'games',
    title: 'Game Predictions',
    icon: '🏈',
    paragraphs: [
      { h: 'The real mechanism' },
      { p: 'The starting point for every game prediction is Vegas’s own real closing spread - not a team rating, not a formula built from scratch. Vegas had a posted line for all 272 real games in the 2025 season used to build this dashboard.' },
      { p: 'On top of that Vegas line, exactly one real adjustment is applied: a matchup-strength adjustment based on each team’s real trailing offensive EPA/play against the opponent’s real trailing defensive EPA/play allowed, at that position group. The size of that adjustment (about 1.065 points per unit of edge) was fit with a real regression on 2015-2023 games, not asserted.' },
      { h: 'What is NOT factored in, despite being tested' },
      { p: '- Rest days: tested directly against real outcomes - measured zero effect. Not used.\n- Recent form / "momentum": tested - results were inconclusive. Shown elsewhere in this dashboard as context, but not used in the spread.\n- QB injury status: shown elsewhere in this dashboard for context, but not wired into the spread calculation itself (Vegas’s own line already reflects public injury news).\n- A team-strength rating (Elo) alone: tested independently and lost to Vegas at every real checkpoint. Elo ratings are shown in this dashboard for context, and would only replace Vegas as a fallback for a game with no posted line - which never happened in the real 2025 season.' },
      { h: 'Win probability' },
      { p: 'A separate real model converts the spread into a win probability. Three candidate approaches were built and backtested against each other on a genuine holdout (2024 data to train, real 2025 weeks 13-17 to test, never seen during fitting): a simple asserted-constant formula, a model based on Elo rating difference, and a model based on the real Vegas spread. The Vegas-based model won clearly (lower real error than both alternatives) and is the one used throughout this dashboard.' },
      { h: 'Real accuracy' },
      { p: 'Straight-up winner accuracy: 65.4% (178 of 272 real games). Our own spread’s real average error is 9.69 points; Vegas alone’s is 9.72 - essentially identical, which makes sense given the model starts from Vegas’s own number.' },
    ],
  },
  {
    id: 'fantasy',
    title: 'Fantasy Predictions',
    icon: '⚡',
    paragraphs: [
      { h: 'The real mechanism, by position' },
      { p: 'For RB, QB, and TE, the real, validated formula is volume-only: each player’s own real trailing touches, yards, receptions, and touchdowns (weeks strictly before the one being predicted - never that week’s own already-realized stats). This was a real, tested finding, not a design choice made in advance: more complex versions factoring in opponent defense strength or game-script were built and backtested, and the simpler volume-only version won for all three positions.' },
      { p: 'WR used to be the one real exception - a single static number per player for the whole season, not week-by-week updating. A real leave-one-out backtest across 23 variations (Phase 4) found a better real method: a trailing up-to-4-week actual-PPR average, the same real leak-free approach RB/QB/TE\'s recent-form display already uses, now used as WR\'s actual projection too. A player\'s first real 2025 appearance still has no trailing weeks to average, so it falls back to the old static number - disclosed per-row in the dashboard, not hidden.' },
      { h: 'Real accuracy, correctly ranked' },
      { p: 'RB: 65.1% correlation to real actual points (the best of the four)\nTE: 54.3%\nQB: 44.7%\nWR: 44.2% (the weakest, but only marginally now - up from 40.0% before the Phase 4 trailing-average fix)' },
      { p: 'A projection is a central estimate, not a guarantee - real week-to-week variance in football is large. A real example: Josh Allen was projected 23.4 points in week 1 and actually scored 38.8.' },
    ],
  },
  {
    id: 'season',
    title: 'Season Projections',
    icon: '📊',
    paragraphs: [
      { h: 'The real mechanism' },
      { p: 'Real wins-so-far plus each remaining game’s real win probability, summed. Playoff odds come from a real Monte Carlo simulation (10,000 runs), simulating every team’s remaining real schedule at once so that shared opponents and tiebreakers stay correlated across the simulation, not treated as independent coin flips.' },
      { p: 'This dashboard uses a single real checkpoint: week 16 of the 2025 season (the latest one this project computed real odds for). A real example: Denver was projected for 13.0 final wins and 100% playoff odds at that checkpoint - the real final result was 14 wins and the #1 seed.' },
      { h: 'Super Bowl odds (added later)' },
      { p: 'A real Monte Carlo bracket simulation (10,000 trials), extending the same playoff-odds engine above through real bracket rules - the #1 seed byes the Wild Card round, real re-seeding in the Divisional round, and a real neutral-site Super Bowl (no home-field edge). Each simulated playoff game uses real, frozen week-16 Elo ratings and this project’s real win-probability formula, not a new heuristic.' },
      { p: 'One real simplification, disclosed directly: the bracket seeding is fixed at the real week-16 projected seeds rather than re-simulated, so this answers “who wins if the projected seeding holds,” not two compounded layers of regular-season-plus-playoff uncertainty.' },
      { h: 'Real accuracy' },
      { p: 'Real final win totals were projected within 0.61 games on average. Division winners: correctly identified 8 of 8. Playoff teams: correctly identified 13 of 14. The real largest single miss was Green Bay, projected for 10.5 wins with a real final of 9 - even the worst case was off by under two games, since a week-16 checkpoint already reflects most of a real season’s outcome.' },
    ],
  },
  {
    id: 'data',
    title: 'Data Sources',
    icon: '📚',
    paragraphs: [
      { h: 'Where everything comes from' },
      { p: '- Game results, play-by-play stats: real NFL data via nflverse/nflreadpy, the same public data source used throughout this project.\n- Team strength ratings: an in-house Elo rating system, fit and validated on 10 years of real games (2015-2024).\n- Player stats and fantasy points: real per-player, per-week stats from the same nflverse source, PPR points calculated from real box-score stats.\n- Injury status: real weekly injury reports (nflreadpy), showing each player’s real official game-status designation.\n- Vegas lines: real historical closing spreads from nflverse’s schedules data - one closing-line snapshot per game (this data source doesn’t include opening-line or intraday movement history).' },
      { h: 'What is deliberately not used' },
      { p: 'No social-media speculation, no locker-room rumors, no sports-media conjecture, no manually-entered opinions. Every number in this dashboard traces back to a real, public data source.' },
      { h: 'Multiple seasons (added later)' },
      { p: 'The season selector at the top switches every section between real, independent datasets - it doesn’t reinterpret one dataset two ways. 2025 is a real, fully-completed season: predictions, results, and every accuracy figure are all real. 2026 is real too, but the season hasn’t been played yet - only a real preseason schedule and real preseason model predictions (rolled-forward Elo, since no real Vegas lines exist for games that far out) are shown. Fantasy Rankings, Accuracy Tracker, Weekly Summary, and Betting Analysis are hidden for 2026 rather than shown empty or fabricated - none of them have anything real to compute from yet.' },
    ],
  },
  {
    id: 'accuracy',
    title: 'What We Get Right (& Wrong)',
    icon: '✅❌',
    paragraphs: [
      { h: 'Real examples, correctly predicted' },
      { p: 'Game: week 4, New Orleans at Buffalo - real win probability 95.0% for Buffalo. Real result: Buffalo won 31-19.\nSeason: Denver, real week-16 projection of 13.0 wins / #1 seed. Real final: 14 wins, #1 seed.' },
      { h: 'Real examples, missed' },
      { p: 'Game: week 9, Carolina at Green Bay - real win probability 91.3% for Green Bay. Real result: Carolina won, 16-13.\nFantasy: week 9, Brock Bowers (TE) - real projected 10.4 points, real actual 43.3.\nSeason: Green Bay, real week-16 projection of 10.5 wins. Real final: 9 wins (the real largest season-projection miss across all 32 teams).' },
      { h: 'What this project is generally good at (real, backtested)' },
      { p: '- Matching Vegas on game outcomes (65.4% real accuracy, essentially tied with Vegas alone)\n- RB fantasy projections (65.1% real correlation, the strongest of the four positions)\n- Season win totals close to real final outcomes (0.61 games average error)\n- Being honest about which tested ideas didn’t pan out, instead of quietly dropping them' },
      { h: 'What this project is generally not good at (real, disclosed)' },
      { p: '- Beating Vegas - no tested approach did, including betting on real disagreements with it\n- Anything driven by in-game events after kickoff (game script, injuries during a game)' },
    ],
  },
  {
    id: 'limits',
    title: 'Limitations',
    icon: '⚠️',
    paragraphs: [
      { h: '1. This is historical pattern-matching, not foresight' },
      { p: 'The model learns from real past games. A genuinely new situation (a major rule change, a blockbuster in-season trade) won’t be reflected until real data about it exists.' },
      { h: '2. Real information gaps' },
      { p: 'No access to internal team data, practice reports, coaching intentions, or real-time line movement (this data source has only one closing-line snapshot per game, not opening lines or intraday movement).' },
      { h: '3. Real randomness, not eliminated' },
      { p: 'A 65% win probability still means a 35% real chance of being wrong. The real accuracy numbers shown throughout this dashboard already account for this - they are not inflated by cherry-picked examples.' },
      { h: '4. WR (and to a lesser extent QB) projections are the weakest part' },
      { p: 'WR now uses a real trailing up-to-4-week actual-PPR average rather than a full-season static number (fixed in Phase 4, a real backtested improvement) - but it\'s still the lowest of the four positions\' real correlations (44.2%, barely below QB\'s 44.7%). A player\'s first real 2025 appearance still has no trailing weeks to average and falls back to the old static figure - disclosed directly next to that specific card in the Fantasy Rankings section, not just here.' },
      { h: '5. No separate defensive model' },
      { p: 'Defensive strength is folded into each team’s overall Elo rating and into the real opponent-defense-by-position ranks shown in Fantasy Rankings, but there is no standalone defensive projection or IDP (individual defensive player) scoring anywhere in this project.' },
      { h: '6. One real season-projection snapshot' },
      { p: 'Every week shown in the Weekly Summary section reuses the exact same real week-16 playoff-odds snapshot - there is no real week-by-week history of how those odds actually moved over the season. This is disclosed directly in that section.' },
      { h: '7. Real backtests, not a live betting record' },
      { p: 'Every accuracy number in this dashboard comes from backtesting against a real, already-completed 2025 season, not from placing real bets and tracking real-money results.' },
      { h: 'The bottom line' },
      { p: 'This model is a real, backtested tool for understanding NFL outcomes and player performance - not a way to reliably beat Vegas, and not a substitute for judgment. Its own real backtests found that acting on its disagreements with the market was actively harmful. Use it as one input, not the whole analysis.' },
    ],
  },
];

export default function ModelTransparency() {
  const [activeSection, setActiveSection] = useState('overview');
  const current = SECTIONS.find((s) => s.id === activeSection);

  return (
    <div className="model-transparency">
      <div className="header">
        <h1>How This Model Works</h1>
        <p className="subtitle">Plain-language explanation of predictions, data, and limitations - every number below is real, checked against this dashboard's own generated data.</p>
      </div>

      <div className="tabs">
        {SECTIONS.map((s) => (
          <button key={s.id} className={`tab ${activeSection === s.id ? 'active' : ''}`} onClick={() => setActiveSection(s.id)}>
            <span className="icon">{s.icon}</span>
            <span className="label">{s.title}</span>
          </button>
        ))}
      </div>

      <div className="content-area">
        {current && (
          <div className="section-content">
            <h2>
              <span>{current.icon}</span> {current.title}
            </h2>
            <div className="text-content">
              {current.paragraphs.map((block, i) =>
                block.h ? (
                  <h3 key={i}>{block.h}</h3>
                ) : (
                  block.p.split('\n').map((line, j) => <p key={`${i}-${j}`}>{line}</p>)
                )
              )}
            </div>
          </div>
        )}
      </div>

      <div className="disclaimer">
        <p>
          Remember: this is one model, not the final word on anything. Its own real backtests found that
          acting on its disagreements with Vegas was actively harmful (-36% ROI), so it is not offered here
          as a way to beat the market - use it as one input among many, alongside expert opinion, injury
          news, and your own judgment.
        </p>
      </div>
    </div>
  );
}
