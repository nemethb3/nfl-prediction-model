import React, { useState } from 'react';
import '../styles/BettingAnalysis.css';
import '../styles/SeasonDataUnavailable.css';
import { useSeason } from '../context/SeasonContext';
import SeasonDataUnavailable from './SeasonDataUnavailable';
import strategyComparison from '../data/betting_strategies_comparison_od_elo.json';

// Real 2024-holdout rows built from strategyComparison.results at module
// scope (static data, not season-dependent - same real train/holdout split
// as od_elo_production_validation.json, see backtest_betting_strategies_
// comparison_od_elo.py's methodology field for the real, disclosed
// train/holdout convention).
const STRATEGY_COMPARISON_ROWS = (() => {
  const r = strategyComparison.results;
  const rows = [
    { label: 'Moneyline', single: r.single_elo.moneyline.holdout_2024, od: r.od_elo.moneyline.holdout_2024 },
    { label: 'Against the Spread', single: r.single_elo.ats.holdout_2024, od: r.od_elo.ats.holdout_2024 },
  ];
  for (const threshold of ['1.0', '2.0', '3.0']) {
    rows.push({
      label: `Totals (edge > ${parseFloat(threshold)} pt${threshold === '1.0' ? '' : 's'})`,
      single: r.single_elo.totals.holdout_2024[threshold],
      od: r.od_elo.totals.holdout_2024[threshold],
    });
  }
  return rows;
})();

const STRATEGY_COLORS = {
  our_system: '#0080c6',
  vegas_favorites: '#4ade80',
  underdogs_only: '#f59e0b',
};

const BET_TYPE_LABELS = {
  moneyline: 'Moneyline (straight-up)',
  ats: 'Against the Spread',
  totals: 'Over/Under (Point Totals)',
};

export default function BettingAnalysis() {
  const { seasonData, selectedSeason, hasResults } = useSeason();
  const resultsData = seasonData.bettingBacktest;
  const totalsBettingBacktest = seasonData.totalsBettingBacktest;

  const [unitSize, setUnitSize] = useState(100);
  const [betType, setBetType] = useState('moneyline');
  const [selectedStrategy, setSelectedStrategy] = useState('our_system');
  const [expandedWeek, setExpandedWeek] = useState(null);

  if (!hasResults) {
    return (
      <div className="betting-analysis">
        <SeasonDataUnavailable season={selectedSeason} sectionName="Betting Analysis" />
      </div>
    );
  }

  // Over/Under is a real, different strategy shape (one edge-threshold axis,
  // no our_system/vegas_favorites/underdogs_only rows, no per-bet moneyline
  // odds) - not just a third betType of the existing 3 strategies, so the
  // strategy-comparison/weekly-breakdown computations below only apply to
  // the other two bet types.
  const isTotals = betType === 'totals';
  const strategyKeys = isTotals ? [] : Object.keys(resultsData);
  const current = isTotals ? null : resultsData[selectedStrategy][betType];
  const seasonSummary = isTotals ? null : current.season_summary;
  const dollarPnL = isTotals ? null : seasonSummary.pnl_units * unitSize;

  const weeklyData = isTotals ? [] : Object.entries(current.weekly_summary)
    .map(([week, data]) => ({ week: parseInt(week, 10), ...data, pnl_dollars: data.pnl_units * unitSize }))
    .sort((a, b) => a.week - b.week);

  return (
    <div className="betting-analysis">
      <div className="header">
        <h1>Betting Analysis</h1>
        <p className="subtitle">
          Three win/loss strategies (moneyline and against-the-spread) plus a separate point-totals
          edge strategy, all backtested on real completed 2025 games using real Vegas odds/lines.
        </p>
      </div>

      <div className="controls">
        <div className="unit-size-control">
          <label>Bet Size Per Game:</label>
          <input
            type="range"
            min="10"
            max="1000"
            step="10"
            value={unitSize}
            onChange={(e) => setUnitSize(parseFloat(e.target.value))}
            className="slider"
          />
          <span className="unit-value">${unitSize}</span>
        </div>

        <div className="bet-type-toggle">
          {Object.entries(BET_TYPE_LABELS).map(([key, label]) => (
            <button
              key={key}
              className={`bet-type-btn ${betType === key ? 'active' : ''}`}
              onClick={() => setBetType(key)}
            >
              {label}
            </button>
          ))}
        </div>
      </div>

      {isTotals ? (
        totalsBettingBacktest && (
          <div className="strategy-detail">
            <div className="strategy-header">
              <h2>Over/Under Edge Betting (Point Totals)</h2>
              <p className="strategy-desc">{totalsBettingBacktest.methodology}</p>
            </div>

            {totalsBettingBacktest.highest_roi_threshold_for_disclosure && (
              <div className="season-metrics">
                <div className="metric-card">
                  <span className="metric-label">Highest-ROI Threshold Tested</span>
                  <span className="metric-value">
                    {totalsBettingBacktest.highest_roi_threshold_for_disclosure.threshold} pts
                  </span>
                  <span className="metric-detail">for disclosure only, not a recommendation</span>
                </div>
                <div className="metric-card">
                  <span className="metric-label">Bets at that threshold</span>
                  <span className="metric-value">
                    {totalsBettingBacktest.highest_roi_threshold_for_disclosure.total_bets}
                  </span>
                  <span className="metric-detail">
                    {totalsBettingBacktest.highest_roi_threshold_for_disclosure.wins}W-
                    {totalsBettingBacktest.highest_roi_threshold_for_disclosure.losses}L
                  </span>
                </div>
                <div className="metric-card">
                  <span className="metric-label">Win Rate</span>
                  <span className="metric-value">
                    {(totalsBettingBacktest.highest_roi_threshold_for_disclosure.win_rate * 100).toFixed(1)}%
                  </span>
                </div>
                <div className="metric-card highlight">
                  <span className="metric-label">ROI</span>
                  <span className={`metric-value ${
                    totalsBettingBacktest.highest_roi_threshold_for_disclosure.roi_pct >= 0 ? 'positive' : 'negative'
                  }`}>
                    {totalsBettingBacktest.highest_roi_threshold_for_disclosure.roi_pct > 0 ? '+' : ''}
                    {totalsBettingBacktest.highest_roi_threshold_for_disclosure.roi_pct.toFixed(1)}%
                  </span>
                </div>
              </div>
            )}

            <div className="table-scroll">
              <table className="comparison-table">
                <thead>
                  <tr>
                    <th>Edge Threshold</th>
                    <th>Bets</th>
                    <th>Record</th>
                    <th>Win %</th>
                    <th>ROI</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(totalsBettingBacktest.results_by_threshold).map(([threshold, result]) => (
                    <tr key={threshold}>
                      <td>{threshold} pts</td>
                      <td>{result.total_bets}</td>
                      <td>{result.wins}-{result.losses}</td>
                      <td>{(result.win_rate * 100).toFixed(1)}%</td>
                      <td className={result.roi_pct >= 0 ? 'positive' : 'negative'}>
                        {result.roi_pct > 0 ? '+' : ''}{result.roi_pct.toFixed(1)}%
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            <div className="weekly-breakdown">
              <p className="strategy-desc">{totalsBettingBacktest.disclosure}</p>
            </div>
          </div>
        )
      ) : (
        <>
          <div className="comparison-section">
            <h2>Strategy Comparison — {BET_TYPE_LABELS[betType]}</h2>
            <div className="table-scroll">
              <table className="comparison-table">
                <thead>
                  <tr>
                    <th>Strategy</th>
                    <th>Bets</th>
                    <th>Record</th>
                    <th>Win %</th>
                    <th>ROI</th>
                    <th>P&amp;L (units)</th>
                    <th>P&amp;L (${unitSize})</th>
                  </tr>
                </thead>
                <tbody>
                  {strategyKeys.map((key) => {
                    const strategy = resultsData[key];
                    const summary = strategy[betType].season_summary;
                    const rowDollarPnL = summary.pnl_units * unitSize;
                    return (
                      <tr
                        key={key}
                        className={`strategy-row ${key === selectedStrategy ? 'active' : ''}`}
                        onClick={() => setSelectedStrategy(key)}
                      >
                        <td className="strategy-name">
                          <span className="color-dot" style={{ backgroundColor: STRATEGY_COLORS[key] }} />
                          {strategy.label}
                        </td>
                        <td>{summary.total_bets}</td>
                        <td>
                          {summary.wins}-{summary.losses}
                          {summary.pushes > 0 ? `-${summary.pushes}` : ''}
                        </td>
                        <td>{summary.win_pct}%</td>
                        <td className={summary.roi_pct >= 0 ? 'positive' : 'negative'}>
                          {summary.roi_pct > 0 ? '+' : ''}
                          {summary.roi_pct}%
                        </td>
                        <td className={summary.pnl_units >= 0 ? 'positive' : 'negative'}>
                          {summary.pnl_units > 0 ? '+' : ''}
                          {summary.pnl_units}
                        </td>
                        <td className={rowDollarPnL >= 0 ? 'positive' : 'negative'}>
                          {rowDollarPnL > 0 ? '+' : ''}${rowDollarPnL.toFixed(0)}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>

          <div className="strategy-detail">
            <div className="strategy-header">
              <h2>{resultsData[selectedStrategy].label}</h2>
              <p className="strategy-desc">{resultsData[selectedStrategy].description}</p>
            </div>

            <div className="season-metrics">
              <div className="metric-card">
                <span className="metric-label">Total Bets</span>
                <span className="metric-value">{seasonSummary.total_bets}</span>
                {seasonSummary.pushes > 0 && (
                  <span className="metric-detail">{seasonSummary.pushes} push{seasonSummary.pushes > 1 ? 'es' : ''}</span>
                )}
              </div>
              <div className="metric-card">
                <span className="metric-label">Win Rate</span>
                <span className="metric-value">{seasonSummary.win_pct}%</span>
                <span className="metric-detail">
                  {seasonSummary.wins}W-{seasonSummary.losses}L
                </span>
              </div>
              <div className="metric-card">
                <span className="metric-label">ROI</span>
                <span className={`metric-value ${seasonSummary.roi_pct >= 0 ? 'positive' : 'negative'}`}>
                  {seasonSummary.roi_pct > 0 ? '+' : ''}
                  {seasonSummary.roi_pct}%
                </span>
              </div>
              <div className="metric-card highlight">
                <span className="metric-label">P&amp;L (${unitSize}/bet)</span>
                <span className={`metric-value ${dollarPnL >= 0 ? 'positive' : 'negative'}`}>
                  {dollarPnL > 0 ? '+' : ''}${dollarPnL.toFixed(0)}
                </span>
              </div>
            </div>

            <div className="weekly-breakdown">
              <h3>Weekly Breakdown</h3>
              <div className="weekly-list">
                {weeklyData.map((week) => (
                  <div key={week.week} className={`weekly-row ${expandedWeek === week.week ? 'expanded' : ''}`}>
                    <div
                      className="weekly-row-summary"
                      onClick={() => setExpandedWeek(expandedWeek === week.week ? null : week.week)}
                    >
                      <span className="week-label">Week {week.week}</span>
                      <span className="week-stats">
                        {week.total_bets} bets · {week.win_pct}% wins
                        {week.pushes > 0 ? ` · ${week.pushes} push` : ''}
                      </span>
                      <span className={`week-pnl ${week.pnl_units >= 0 ? 'positive' : 'negative'}`}>
                        {week.pnl_dollars > 0 ? '+' : ''}${week.pnl_dollars.toFixed(0)}
                      </span>
                    </div>

                    {expandedWeek === week.week && (
                      <div className="week-bets">
                        {current.bets
                          .filter((bet) => bet.week === week.week)
                          .map((bet, i) => (
                            <div key={i} className={`bet-row ${bet.result}`}>
                              <span className="matchup">{bet.matchup}</span>
                              <span className="bet-detail">
                                Bet: {bet.bet_team} ({bet.odds > 0 ? '+' : ''}
                                {bet.odds})
                              </span>
                              <span className="result">
                                {bet.result === 'win' ? '✅ Won' : bet.result === 'loss' ? '❌ Loss' : '➖ Push'}
                              </span>
                              <span className={`pnl ${bet.pnl_units >= 0 ? 'positive' : 'negative'}`}>
                                {bet.pnl_units > 0 ? '+' : ''}${(bet.pnl_units * unitSize).toFixed(0)}
                              </span>
                            </div>
                          ))}
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </div>
          </div>
        </>
      )}

      <section className="model-evolution">
        <h3>🔬 Model Evolution: Offensive/Defensive Elo</h3>

        <div className="evolution-intro">
          <p>
            This project tests an <strong>Offensive/Defensive Elo split</strong> model that
            separates team strength into offense (scoring ability) and defense (preventing
            scores), instead of one combined rating. This captures matchup dynamics the combined
            rating can&apos;t: an elite offense vs. a weak defense plays out differently than an
            average offense vs. an elite defense, even at the same combined team strength.
          </p>
        </div>

        <div className="model-comparison">
          <div className="comparison-header">
            <h4>Real Backtest Results</h4>
          </div>

          <div className="table-scroll">
            <table className="comparison-table">
              <thead>
                <tr>
                  <th>Metric</th>
                  <th>Single-Elo</th>
                  <th>O/D Elo</th>
                  <th>Winner</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td>Win/Loss Accuracy (5-fold CV, 2015-2025)</td>
                  <td>62.9%</td>
                  <td><strong>64.2%</strong></td>
                  <td>✅ O/D Elo (+1.3pp)</td>
                </tr>
                <tr>
                  <td>Spread Accuracy (MAE, 2024 holdout)</td>
                  <td>10.21 pts</td>
                  <td><strong>10.14 pts</strong></td>
                  <td>✅ O/D Elo</td>
                </tr>
                <tr>
                  <td>Probabilistic Calibration (Brier, 2024 holdout)</td>
                  <td>0.2272</td>
                  <td><strong>0.2182</strong></td>
                  <td>✅ O/D Elo</td>
                </tr>
                <tr>
                  <td>90% CI Coverage (2024 holdout, target 90%)</td>
                  <td><strong>89.3%</strong></td>
                  <td>86.4%</td>
                  <td>✅ Single-Elo</td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>

        <div className="model-comparison">
          <div className="comparison-header">
            <h4>Single-Elo vs. O/D Elo: Real Betting Results (2024 Holdout)</h4>
          </div>
          <p className="evolution-strategy-note">
            Same &quot;bet when we disagree with Vegas&quot; strategy shape used in Betting Analysis
            above, run once with Single-Elo predictions and once with O/D Elo predictions, settled
            with real per-game Vegas odds — genuinely out-of-sample (both models fit on 2015-2023
            only, scored on 2024).
          </p>

          <div className="table-scroll">
            <table className="comparison-table">
              <thead>
                <tr>
                  <th>Strategy</th>
                  <th>Single-Elo</th>
                  <th>O/D Elo</th>
                  <th>Winner</th>
                </tr>
              </thead>
              <tbody>
                {STRATEGY_COMPARISON_ROWS.map((row) => {
                  const odWinsRoi = row.od.roi_pct > row.single.roi_pct;
                  return (
                    <tr key={row.label}>
                      <td>{row.label}</td>
                      <td>
                        {row.single.win_pct}% win rate,{' '}
                        <span className={row.single.roi_pct >= 0 ? 'positive' : 'negative'}>
                          {row.single.roi_pct > 0 ? '+' : ''}
                          {row.single.roi_pct}% ROI
                        </span>
                        <span className="small-text"> ({row.single.total_bets} bets)</span>
                      </td>
                      <td>
                        {row.od.win_pct}% win rate,{' '}
                        <span className={row.od.roi_pct >= 0 ? 'positive' : 'negative'}>
                          {row.od.roi_pct > 0 ? '+' : ''}
                          {row.od.roi_pct}% ROI
                        </span>
                        <span className="small-text"> ({row.od.total_bets} bets)</span>
                      </td>
                      <td>{odWinsRoi ? '✅ O/D Elo' : '✅ Single-Elo'}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
          <p className="evolution-strategy-note">
            Most rows are net-negative ROI for both models — consistent with this project&apos;s
            already-disclosed finding that betting on disagreements with Vegas loses money over a
            real backtest (see the disclaimer at the bottom of this page). This table is a real,
            honest head-to-head, not a recommendation to bet either model. 2025 numbers (context
            only, same models, not re-fit) are in the underlying data file if needed.
          </p>
        </div>

        <div className="evolution-status">
          <h4>Current Status</h4>
          <ul>
            <li>
              <strong>Game Predictions (2026):</strong> O/D Elo is now the primary model — after
              the results above, it was swapped in to generate the spread, win probability, and
              confidence interval for every real 2026 game. Single-Elo is still shown separately
              as &quot;Matchup Strength&quot; context on each game card, but no longer drives the
              prediction.
            </li>
            <li>
              <strong>Betting Analysis (above):</strong> the moneyline/ATS/totals results shown at
              the top of this page are still Single-Elo based — they cover the completed 2025
              season, which predates the O/D Elo swap. The real head-to-head comparison below
              shows how O/D Elo would have done on the same three strategies over a genuine
              out-of-sample holdout.
            </li>
            <li>
              <strong>Trade Model, Player Props, Breakout Alerts:</strong> not yet integrated with
              O/D Elo. These would be real, separate follow-up work, not something this swap
              already includes.
            </li>
          </ul>
        </div>

        <div className="evolution-next-steps">
          <h4>Next: 2026 Live Monitoring</h4>
          <p>
            As real 2026 games are played, this project will be able to compare O/D Elo&apos;s
            actual predictions against real results directly, the same way Model Transparency
            already tracks single-Elo accuracy for 2025. That real data — not a backtest — is
            what determines whether the swap holds up.
          </p>
        </div>

        <div className="evolution-caveat">
          <p>
            One disclosed trade-off: O/D Elo&apos;s 90% confidence bands covered the real 2024
            holdout only 86.4% of the time, vs. Single-Elo&apos;s 89.3% (closer to the 90% target)
            — bands are a bit tighter than ideal. Everything else favored the swap; this one
            metric didn&apos;t, and is shown here rather than left out.
          </p>
        </div>
      </section>

      <div className="methodology">
        <h3>Methodology</h3>
        <ul>
          <li>
            <strong>Real odds:</strong> every bet is priced with this project&apos;s real 2025 Vegas
            moneylines and real spread odds (not estimated from the spread size).
          </li>
          <li>
            <strong>Our System:</strong> bets when our model&apos;s win probability and Vegas&apos;s
            implied probability differ by more than 2%. Direction is the favorite when our spread
            is more extreme than Vegas&apos;s, the underdog when it&apos;s less extreme — a
            different, separately-tested rule from this dashboard&apos;s spread-disagreement note
            elsewhere, not the same strategy.
          </li>
          <li>
            <strong>Vegas Favorites / Underdogs Only:</strong> every game, bet the Vegas favorite or
            underdog at real Vegas odds.
          </li>
          <li>
            <strong>Moneyline vs. ATS:</strong> the same three strategies, settled two different real
            ways — straight-up winner (moneyline) or covering the real spread (ATS). Toggle above to
            compare.
          </li>
          <li>
            <strong>Pushes:</strong> one real 2025 tie (Week 4, GB 40 @ DAL 40) pushes on moneyline
            bets; one real exact-margin push exists on ATS (Week 12, PIT/CHI). Pushes return the
            stake and are excluded from win/loss and ROI.
          </li>
          <li>
            <strong>Over/Under (Point Totals):</strong> a separate strategy, not one of the three
            above - bets the direction our point-totals model disagrees with the real Vegas total by
            more than an edge threshold. Backtested with a genuine holdout (model refit on
            2015-2024 only, scored on 2025), not reused in-sample predictions.
          </li>
        </ul>
      </div>

      <div className="disclaimer">
        <p>
          Historical backtest on completed 2025 games only — not a live betting recommendation.
          This project has already found, separately (see Model Transparency), that betting on
          disagreements with Vegas loses money over a real backtest; these three strategies are
          shown here as an additional, real cross-check, not as advice.
        </p>
      </div>
    </div>
  );
}
