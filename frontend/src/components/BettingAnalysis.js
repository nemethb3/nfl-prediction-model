import React, { useState } from 'react';
import '../styles/BettingAnalysis.css';
import '../styles/SeasonDataUnavailable.css';
import { useSeason } from '../context/SeasonContext';
import SeasonDataUnavailable from './SeasonDataUnavailable';

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
