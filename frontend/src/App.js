import React, { useState } from 'react';
import Navigation from './components/Navigation';
import SeasonSelector from './components/SeasonSelector';
import GamePredictions from './components/GamePredictions';
import FantasyRankings from './components/FantasyRankings';
import SeasonProjections from './components/SeasonProjections';
import AccuracyTracker from './components/AccuracyTracker';
import WeeklySummary from './components/WeeklySummary';
import ModelTransparency from './components/ModelTransparency';
import BettingAnalysis from './components/BettingAnalysis';
import LeagueConnector from './components/LeagueConnector';
import TradeAnalyzer from './components/TradeAnalyzer';
import { SeasonProvider } from './context/SeasonContext';
import { DEFAULT_SECTION } from './constants/sections';
import './App.css';

const SECTION_COMPONENTS = {
  games: GamePredictions,
  fantasy: FantasyRankings,
  projections: SeasonProjections,
  accuracy: AccuracyTracker,
  summary: WeeklySummary,
  transparency: ModelTransparency,
  betting: BettingAnalysis,
  sleeper: LeagueConnector,
  'trade-analyzer': TradeAnalyzer,
};

export default function App() {
  const [activeSection, setActiveSection] = useState(DEFAULT_SECTION);

  const ActiveComponent = SECTION_COMPONENTS[activeSection] || GamePredictions;

  return (
    <SeasonProvider>
      <div className="app-container">
        <Navigation activeSection={activeSection} onSectionChange={setActiveSection} />
        <SeasonSelector />
        <main className="section-content">
          <ActiveComponent />
        </main>
      </div>
    </SeasonProvider>
  );
}
