import React, { useState } from 'react';
import '../styles/AccuracyTracker.css';
import '../styles/SeasonDataUnavailable.css';
import { useSeason } from '../context/SeasonContext';
import SeasonDataUnavailable from './SeasonDataUnavailable';

const TABS = ['games', 'fantasy', 'season', 'betting', 'trends', 'comparison'];

export default function AccuracyTracker() {
  const { seasonData, selectedSeason, hasResults } = useSeason();
  const accuracyData = seasonData.accuracyTracker;

  const [activeTab, setActiveTab] = useState('games');
  const [selectedWeek, setSelectedWeek] = useState(null);

  if (!hasResults) {
    return (
      <div className="accuracy-tracker">
        <SeasonDataUnavailable season={selectedSeason} sectionName="Accuracy Tracker" />
      </div>
    );
  }

  const summary = accuracyData.season_summary;
  const weeklyData = accuracyData.weekly_breakdown;

  const currentData = selectedWeek ? weeklyData.find((w) => w.week === selectedWeek) : summary;

  // Real, computed from the real weekly breakdown - not an asserted "expected
  // range." Spread MAE swings well outside a flat +-8pt band across a real
  // season (2025's real weekly range is roughly 6-14.5 points).
  const maeValues = weeklyData.map((w) => w.games.mae_spread);
  const maeRange = { min: Math.min(...maeValues), max: Math.max(...maeValues) };

  return (
    <div className="accuracy-tracker">
      <div className="header">
        <h1>Accuracy Tracker</h1>
        <p className="subtitle">Real model performance vs. real Vegas lines and real outcomes, 2025 season</p>
      </div>

      <div className="week-selector">
        <button className={`week-btn ${!selectedWeek ? 'active' : ''}`} onClick={() => setSelectedWeek(null)}>
          Full Season
        </button>
        {weeklyData.map((w) => (
          <button
            key={w.week}
            className={`week-btn ${selectedWeek === w.week ? 'active' : ''}`}
            onClick={() => setSelectedWeek(w.week)}
          >
            W{w.week}
          </button>
        ))}
      </div>

      <div className="tabs">
        {TABS.map((tab) => (
          <button key={tab} className={`tab ${activeTab === tab ? 'active' : ''}`} onClick={() => setActiveTab(tab)}>
            {tab.charAt(0).toUpperCase() + tab.slice(1)}
          </button>
        ))}
      </div>

      <div className="tab-content">
        {activeTab === 'games' && currentData.games && (
          <div className="metrics-grid">
            <MetricCard
              title="Accuracy"
              value={`${currentData.games.accuracy_pct}%`}
              subtitle={`${currentData.games.correct_predictions ?? currentData.games.correct}/${currentData.games.total_games ?? currentData.games.total} correct`}
            />
            <MetricCard title="MAE (Our Spread)" value={`${currentData.games.mae_spread}`} subtitle="Points off on average" />
            <MetricCard title="MAE (Vegas)" value={`${currentData.games.vs_vegas_spread ?? currentData.games.vs_vegas_mae}`} subtitle="For comparison" />
            <div className="chart-confidence-note">
              📊 Real weekly spread MAE has ranged {maeRange.min.toFixed(1)}–{maeRange.max.toFixed(1)} points across the
              2025 season - a single week's number is expected to vary well outside the season average shown above.
            </div>
          </div>
        )}

        {activeTab === 'fantasy' && currentData.fantasy && (
          <div className="fantasy-grid">
            {Object.entries(currentData.fantasy).map(([position, stats]) => (
              <div key={position} className="fantasy-card">
                <h3>{position}</h3>
                <div className="stat">
                  <span className="label">Correlation</span>
                  <span className="value">{stats.correlation}</span>
                </div>
                <div className="stat">
                  <span className="label">MAE</span>
                  <span className="value">{stats.mae} PPR</span>
                </div>
                {stats.samples !== undefined && (
                  <div className="stat">
                    <span className="label">Samples</span>
                    <span className="value">{stats.samples}</span>
                  </div>
                )}
              </div>
            ))}
            {activeTab === 'fantasy' && !selectedWeek && (
              <p className="tab-note">
                WR now uses a real trailing up-to-4-week actual-PPR average (Phase 4 backtest
                winner) rather than a static season-long figure - only a player's first real 2025
                appearance still falls back to the static number (see Section 2). WR's real
                correlation is still the lowest of the four positions, but only marginally below
                QB's now, not because of a static-projection artifact.
              </p>
            )}
          </div>
        )}

        {activeTab === 'season' && !selectedWeek && summary.season_projections && (
          <div className="metrics-grid">
            <MetricCard
              title="Division Winners"
              value={`${summary.season_projections.correct_division_winners}/${summary.season_projections.teams}`}
              subtitle="Week-16 projection vs. real final outcome"
            />
            <MetricCard
              title="Playoff Teams"
              value={`${summary.season_projections.correct_playoff_teams}/14`}
              subtitle="Week-16 projection vs. real final outcome"
            />
            <MetricCard
              title="Win Projection Error"
              value={`${summary.season_projections.avg_wins_error}`}
              subtitle="Avg. |projected − real final wins| per team"
            />
          </div>
        )}
        {activeTab === 'season' && selectedWeek && (
          <p className="tab-note">Season projection accuracy is a single season-level metric (week-16 snapshot vs. real final outcome) - switch to Full Season to view it.</p>
        )}

        {activeTab === 'betting' && currentData.betting && (
          <div className="betting-section">
            <MetricCard title="Moneyline Accuracy" value={`${currentData.betting.moneyline_accuracy_pct}%`} subtitle="Correct straight-up winner prediction" />
            {currentData.betting.note && (
              <div className="betting-note">
                <p>{currentData.betting.note}</p>
              </div>
            )}
          </div>
        )}

        {activeTab === 'trends' && !selectedWeek && (
          <div className="chart-container">
            <TrendsChart data={weeklyData} />
            <p className="tab-note">
              Game Accuracy and Moneyline % track exactly together here because they're the real
              same underlying metric (straight-up winner correctness) - the model's only real
              moneyline prediction is its win-probability favorite, so the two lines overlap
              exactly rather than showing two independent signals.
            </p>
          </div>
        )}
        {activeTab === 'trends' && selectedWeek && (
          <p className="tab-note">Trends show the full-season week-by-week view - switch to Full Season to see it.</p>
        )}

        {activeTab === 'comparison' && currentData.games && (
          <div className="comparison-table">
            <table>
              <thead>
                <tr>
                  <th>Metric</th>
                  <th>Our Model</th>
                  <th>Vegas</th>
                  <th>Difference</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td>Spread MAE</td>
                  <td>{currentData.games.mae_spread}</td>
                  <td>{currentData.games.vs_vegas_spread ?? currentData.games.vs_vegas_mae}</td>
                  <td className={currentData.games.mae_spread < (currentData.games.vs_vegas_spread ?? currentData.games.vs_vegas_mae) ? 'win' : 'loss'}>
                    {((currentData.games.vs_vegas_spread ?? currentData.games.vs_vegas_mae) - currentData.games.mae_spread).toFixed(2)}
                  </td>
                </tr>
              </tbody>
            </table>
          </div>
        )}
      </div>

      <div className="disclaimer">
        <p>
          Real 2025 data throughout. Games/betting/fantasy metrics are computed from this
          project's own dashboard exports (games_2025.json, fantasy_rankings_2025.json), not
          re-derived from raw pipeline files. Season projection accuracy compares the real
          week-16 checkpoint snapshot against the real final 18-week outcome. Weekly data covers
          completed games only - this is a fully-completed historical season, so all 18 weeks
          have real data.
        </p>
      </div>
    </div>
  );
}

function MetricCard({ title, value, subtitle }) {
  return (
    <div className="metric-card">
      <div className="metric-title">{title}</div>
      <div className="metric-value">{value}</div>
      {subtitle && <div className="metric-subtitle">{subtitle}</div>}
    </div>
  );
}

function TrendsChart({ data }) {
  if (!data || data.length === 0) return <p>No data available</p>;

  const width = 800;
  const height = 400;
  const padding = 60;
  const chartWidth = width - 2 * padding;
  const chartHeight = height - 2 * padding;
  const minAccuracy = 50;
  const maxAccuracy = 100;

  // Real confidence band: mean +/- 1 real standard deviation of this
  // season's own weekly game-accuracy values (not an asserted round
  // number). Game accuracy and moneyline accuracy are the real same
  // underlying metric here (both are straight-up-winner correctness - see
  // accuracyData.season_summary.betting.note), so one band covers both
  // lines.
  const accuracyValues = data.map((d) => d.games.accuracy_pct);
  const accuracyMean = accuracyValues.reduce((a, b) => a + b, 0) / accuracyValues.length;
  const accuracyStdev = Math.sqrt(
    accuracyValues.reduce((sum, v) => sum + (v - accuracyMean) ** 2, 0) / accuracyValues.length
  );
  const bandHigh = Math.min(accuracyMean + accuracyStdev, maxAccuracy);
  const bandLow = Math.max(accuracyMean - accuracyStdev, minAccuracy);
  const yFor = (val) => padding + chartHeight - ((val - minAccuracy) / (maxAccuracy - minAccuracy)) * chartHeight;

  const points = data.map((d, i) => {
    const x = padding + (data.length === 1 ? 0 : (i / (data.length - 1)) * chartWidth);
    const gameAccY = yFor(d.games.accuracy_pct);
    const moneylineY = yFor(d.betting.moneyline_accuracy_pct);
    return { x, gameAccY, moneylineY, week: d.week };
  });

  const gameAccPath = points.map((p, i) => `${i === 0 ? 'M' : 'L'} ${p.x} ${p.gameAccY}`).join(' ');
  const moneylinePath = points.map((p, i) => `${i === 0 ? 'M' : 'L'} ${p.x} ${p.moneylineY}`).join(' ');
  const bandPath =
    `M ${padding} ${yFor(bandHigh)} L ${width - padding} ${yFor(bandHigh)} ` +
    `L ${width - padding} ${yFor(bandLow)} L ${padding} ${yFor(bandLow)} Z`;

  return (
    <svg width="100%" height={height} viewBox={`0 0 ${width} ${height}`} className="trends-svg">
      <path d={bandPath} fill="#4ade80" fillOpacity="0.1" />

      {[50, 60, 70, 80, 90, 100].map((val) => (
        <g key={`grid-${val}`}>
          <line
            x1={padding}
            y1={padding + chartHeight - ((val - minAccuracy) / (maxAccuracy - minAccuracy)) * chartHeight}
            x2={width - padding}
            y2={padding + chartHeight - ((val - minAccuracy) / (maxAccuracy - minAccuracy)) * chartHeight}
            stroke="#333"
            strokeDasharray="4"
          />
          <text
            x={padding - 10}
            y={padding + chartHeight - ((val - minAccuracy) / (maxAccuracy - minAccuracy)) * chartHeight + 4}
            textAnchor="end"
            fill="#666"
            fontSize="12"
          >
            {val}%
          </text>
        </g>
      ))}

      <line x1={padding} y1={padding} x2={padding} y2={height - padding} stroke="#666" strokeWidth="2" />
      <line x1={padding} y1={height - padding} x2={width - padding} y2={height - padding} stroke="#666" strokeWidth="2" />

      {points.map((p, i) => (
        <text key={`week-${i}`} x={p.x} y={height - padding + 20} textAnchor="middle" fill="#666" fontSize="11">
          W{p.week}
        </text>
      ))}

      <path d={gameAccPath} stroke="#4ade80" strokeWidth="2" fill="none" />
      <path d={moneylinePath} stroke="#0080c6" strokeWidth="2" fill="none" />

      {points.map((p, i) => (
        <g key={`dots-${i}`}>
          <circle cx={p.x} cy={p.gameAccY} r="4" fill="#4ade80" />
          <circle cx={p.x} cy={p.moneylineY} r="4" fill="#0080c6" />
        </g>
      ))}

      <g>
        <rect x={width - 300} y={padding + 5} width="290" height="70" fill="#1a1a1a" stroke="#333" strokeWidth="1" rx="4" />

        <rect x={width - 290} y={padding + 15} width="12" height="12" fill="#4ade80" fillOpacity="0.3" />
        <text x={width - 270} y={padding + 25} fill="#999" fontSize="11">
          Real weekly range (mean ±{accuracyStdev.toFixed(1)}%)
        </text>

        <line x1={width - 290} y1={padding + 42} x2={width - 278} y2={padding + 42} stroke="#4ade80" strokeWidth="2" />
        <text x={width - 270} y={padding + 47} fill="#999" fontSize="11">
          Game Accuracy %
        </text>

        <line x1={width - 290} y1={padding + 60} x2={width - 278} y2={padding + 60} stroke="#0080c6" strokeWidth="2" />
        <text x={width - 270} y={padding + 65} fill="#999" fontSize="11">
          Moneyline % (same metric, see note below)
        </text>
      </g>
    </svg>
  );
}
