import React, { useState, useEffect } from 'react';
import { useSeason } from '../context/SeasonContext';
import SeasonDataUnavailable from './SeasonDataUnavailable';
import { getCurrentWeek } from '../utils/getCurrentWeek';
import '../styles/WeeklyPicks.css';

// "Our Picks" and "Credit Picks" are two real, separate pick sources -
// the first is the exact same "Our System" selection Betting Analysis
// tracks (generate_weekly_picks_2026.py), the second is the credit-
// weighted allocation from a separate paper-trading exercise
// (track_juke_results.py) - kept under a generic name here per the
// user's own instruction to keep it visibly distinct from that
// competition, not named after it anywhere in this UI.
const TABS = [
  { id: 'our-picks', label: 'Our Picks' },
  { id: 'credit-picks', label: 'Credit Picks' },
];

function resultEmoji(result) {
  if (result === 'win') return '✅';
  if (result === 'loss') return '❌';
  if (result === 'push') return '➖';
  return '⏳';
}

function formatOdds(odds) {
  if (odds == null) return '—';
  return odds > 0 ? `+${odds}` : `${odds}`;
}

function OurPicksPanel({ weeklyPicks, selectedSeason, hasResults }) {
  const weeks = [...new Set(weeklyPicks.map((p) => p.week))].sort((a, b) => a - b);
  const defaultWeek = () => (hasResults ? weeks[0] : getCurrentWeek(weeklyPicks));
  const [selectedWeek, setSelectedWeek] = useState(defaultWeek);

  useEffect(() => {
    setSelectedWeek(defaultWeek());
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedSeason]);

  const weekPicks = weeklyPicks
    .filter((p) => p.week === selectedWeek)
    .slice()
    .sort((a, b) => new Date(a.kickoff_datetime || 0) - new Date(b.kickoff_datetime || 0));

  return (
    <div className="weekly-picks-panel">
      <div className="week-selector">
        <label htmlFor="our-picks-week-select">Week</label>
        <select
          id="our-picks-week-select"
          value={selectedWeek}
          onChange={(e) => setSelectedWeek(Number(e.target.value))}
        >
          {weeks.map((w) => (
            <option key={w} value={w}>Week {w}</option>
          ))}
        </select>
      </div>
      <p className="weekly-picks-subtitle">
        Our System&apos;s pick for each real {selectedSeason} game, posted here before kickoff — the exact
        same pick Betting Analysis tracks once the game is final.
      </p>
      {weekPicks.length === 0 ? (
        <div className="weekly-picks-empty">No real pick cleared our entry gate this week.</div>
      ) : (
        <div className="weekly-picks-list">
          {weekPicks.map((p) => (
            <div key={p.id} className="weekly-pick-row">
              <div className="weekly-pick-matchup">{p.away_team} @ {p.home_team}</div>
              <div className="weekly-pick-badge">Our Pick: {p.pick_team}</div>
              <div className="weekly-pick-detail">
                <span className="weekly-pick-odds">ML {formatOdds(p.moneyline_odds)}</span>
                <span className={`weekly-pick-result ${p.result_moneyline}`}>
                  {resultEmoji(p.result_moneyline)} ML
                </span>
                <span className={`weekly-pick-result ${p.result_ats}`}>
                  {resultEmoji(p.result_ats)} ATS
                </span>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function CreditPicksPanel({ creditPicks }) {
  const weeks = creditPicks.weeks.map((w) => w.week);
  const [selectedWeek, setSelectedWeek] = useState(weeks[weeks.length - 1]);
  const week = creditPicks.weeks.find((w) => w.week === selectedWeek) || creditPicks.weeks[weeks.length - 1];
  const seasonToDate = creditPicks.season_to_date;

  return (
    <div className="weekly-picks-panel">
      <div className="credit-picks-summary">
        <div className="weekly-picks-summary-card">
          <span className="weekly-picks-summary-label">Season to date</span>
          <span className={`weekly-picks-summary-value ${seasonToDate.net_credits_won >= 0 ? 'positive' : 'negative'}`}>
            {seasonToDate.net_credits_won >= 0 ? '+' : ''}{seasonToDate.net_credits_won} credits
          </span>
          <span className="weekly-picks-summary-detail">
            {seasonToDate.roi_pct != null ? `${seasonToDate.roi_pct}% ROI · ` : ''}
            {seasonToDate.weeks_fully_settled} settled week{seasonToDate.weeks_fully_settled === 1 ? '' : 's'}
          </span>
        </div>
      </div>

      <div className="week-selector">
        <label htmlFor="credit-picks-week-select">Week</label>
        <select
          id="credit-picks-week-select"
          value={selectedWeek}
          onChange={(e) => setSelectedWeek(Number(e.target.value))}
        >
          {weeks.map((w) => (
            <option key={w} value={w}>Week {w}</option>
          ))}
        </select>
      </div>

      {week && (
        <>
          <p className="weekly-picks-subtitle">
            A separate, credit-weighted allocation across this week&apos;s games (1000 credits, sized by
            confidence) — {week.fully_settled
              ? `settled ${week.net_credits_won >= 0 ? '+' : ''}${week.net_credits_won} credits (${week.roi_pct}% ROI).`
              : 'results pending.'}
          </p>
          <div className="weekly-picks-list">
            {week.picks.map((p, i) => (
              <div key={i} className="weekly-pick-row">
                <div className="weekly-pick-matchup">{p.away_team ?? '?'} @ {p.home_team ?? '?'}</div>
                <div className="weekly-pick-badge">
                  {p.selection} ({p.market}{p.line != null ? ` ${p.line > 0 ? '+' : ''}${p.line}` : ''})
                  <span className={`weekly-pick-tier ${p.tier}`}>{p.tier}</span>
                </div>
                <div className="weekly-pick-detail">
                  <span className="weekly-pick-odds">{p.credits} credits</span>
                  <span className={`weekly-pick-result ${p.result}`}>
                    {resultEmoji(p.result)}{' '}
                    {p.result === 'pending' ? 'pending' : `${p.pnl >= 0 ? '+' : ''}${p.pnl}`}
                  </span>
                </div>
              </div>
            ))}
          </div>
        </>
      )}
    </div>
  );
}

export default function WeeklyPicks() {
  const { seasonData, selectedSeason, hasResults } = useSeason();
  const [activeTab, setActiveTab] = useState('our-picks');
  const weeklyPicks = seasonData.weeklyPicks;
  const creditPicks = seasonData.creditPicks;

  if (!weeklyPicks && !creditPicks) {
    return (
      <div className="weekly-picks-container">
        <SeasonDataUnavailable season={selectedSeason} sectionName="Weekly Picks" />
      </div>
    );
  }

  return (
    <div className="weekly-picks-container">
      <div className="header">
        <h1>Weekly Picks</h1>
        <p className="subtitle">
          Picks posted here before kickoff, so they can be checked against the result honestly —
          never added after the fact.
        </p>
      </div>

      {/* Real, deliberate choice: NOT reusing Fantasy.css's .fantasy-tabs-
          header/.fantasy-tab-button classes even though they're unscoped
          globals - Fantasy.js is lazy-loaded and not the default section,
          so its CSS chunk may never have loaded if a visitor opens Weekly
          Picks directly. GamePredictions.css's .header/.subtitle/.week-
          selector/.metric-card ARE safe to reuse below - GamePredictions
          is DEFAULT_SECTION, so its CSS always loads on first render. */}
      <div className="weekly-picks-tabs-header" role="tablist">
        {TABS.map((tab) => (
          <button
            key={tab.id}
            role="tab"
            aria-selected={activeTab === tab.id}
            className={`weekly-picks-tab-button ${activeTab === tab.id ? 'active' : ''}`}
            onClick={() => setActiveTab(tab.id)}
          >
            {tab.label}
          </button>
        ))}
      </div>

      <div className="weekly-picks-tabs-content">
        {activeTab === 'our-picks' && (
          weeklyPicks && weeklyPicks.length > 0
            ? <OurPicksPanel weeklyPicks={weeklyPicks} selectedSeason={selectedSeason} hasResults={hasResults} />
            : <div className="weekly-picks-empty">No real Our System picks recorded yet for {selectedSeason}.</div>
        )}
        {activeTab === 'credit-picks' && (
          creditPicks && creditPicks.weeks?.length > 0
            ? <CreditPicksPanel creditPicks={creditPicks} />
            : <div className="weekly-picks-empty">No real credit-allocation picks recorded yet for {selectedSeason}.</div>
        )}
      </div>
    </div>
  );
}
