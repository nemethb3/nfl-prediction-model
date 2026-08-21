import React, { useState } from 'react';
import { useSeason } from '../context/SeasonContext';
import FantasyRankings from './FantasyRankings';
import TradeAnalyzer from './TradeAnalyzer';
import LeagueConnector from './LeagueConnector';
import SuggestedLineups from './SuggestedLineups';
import './Fantasy.css';

// Rankings and League Connection real-y need season data (FantasyRankings
// calls useSeason() directly; LeagueConnector renders PersonalRoster once
// connected, which also calls useSeason() - verified via grep). Trade
// Analyzer doesn't (TradeAnalyzer statically imports its own precomputed
// JSON, independent of the selected season). Gating ALL subtabs behind the
// season-data loading placeholder (the originally pasted spec's Fantasy.js
// has no loading-state handling at all) would be a real regression for
// the one that doesn't need it.
//
// Breakout Alerts tab removed (BreakoutAlerts.js component and its
// breakout_alerts_2026.json data both left in place, unused, in case this
// comes back later) - the real per-player breakout badges inside
// FantasyRankings itself are a separate feature and are unaffected.
const TABS = [
  { id: 'rankings', label: 'Rankings', needsSeasonData: true },
  { id: 'trade-analyzer', label: 'Trade Analyzer', needsSeasonData: false },
  { id: 'league-connection', label: 'League Connection', needsSeasonData: true },
  { id: 'suggested-lineup', label: 'Suggested Lineup', needsSeasonData: true },
];

export default function Fantasy() {
  const [activeTab, setActiveTab] = useState('rankings');
  const { dataLoading } = useSeason();
  const activeTabMeta = TABS.find((t) => t.id === activeTab);
  const waitingOnSeasonData = dataLoading && activeTabMeta.needsSeasonData;

  return (
    <div className="fantasy-container">
      <div className="fantasy-tabs-header" role="tablist">
        {TABS.map((tab) => (
          <button
            key={tab.id}
            role="tab"
            aria-selected={activeTab === tab.id}
            className={`fantasy-tab-button ${activeTab === tab.id ? 'active' : ''}`}
            onClick={() => setActiveTab(tab.id)}
          >
            {tab.label}
          </button>
        ))}
      </div>

      <div className="fantasy-tabs-content">
        {waitingOnSeasonData ? (
          <div className="section-loading-placeholder">Loading season data...</div>
        ) : (
          <>
            {activeTab === 'rankings' && <FantasyRankings />}
            {activeTab === 'trade-analyzer' && <TradeAnalyzer />}
            {/* LeagueConnector already renders PersonalRoster itself once connected
                (with the real leagueId/userId it collects) - rendering a second,
                separate <PersonalRoster /> here (as the originally pasted spec did)
                would duplicate it with no props, which PersonalRoster's own
                `if (userId) fetchRoster()` guard would just leave permanently
                stuck on "loading". */}
            {activeTab === 'league-connection' && <LeagueConnector />}
            {activeTab === 'suggested-lineup' && <SuggestedLineups />}
          </>
        )}
      </div>
    </div>
  );
}
