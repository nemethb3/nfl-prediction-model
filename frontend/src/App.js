import React, { useState, Suspense, lazy } from 'react';
import Navigation from './components/Navigation';
import SeasonSelector from './components/SeasonSelector';
import { SeasonProvider, useSeason } from './context/SeasonContext';
import { DEFAULT_SECTION } from './constants/sections';
import generatedAt from './data/generated_at.json';
import './App.css';

// Real shape (generation_timestamps.py's record_generation, called by
// every pipeline script that writes a dashboard data file) is a flat
// {script_output_name: iso_timestamp} map, not the spec's assumed
// {generated_at, scripts_run: [...]} - there is no single top-level
// "generated_at" field. The real most-recent pipeline run is the max
// timestamp across all real values in the file (ISO 8601 strings sort
// correctly as plain strings).
const LAST_UPDATED = Object.values(generatedAt).sort().pop();

function formatLastUpdated(iso) {
  if (!iso) return null;
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return null;
  return `${d.toLocaleDateString(undefined, { year: 'numeric', month: 'short', day: 'numeric' })} ` +
    `${d.toLocaleTimeString(undefined, { hour: 'numeric', minute: '2-digit', timeZoneName: 'short' })}`;
}

// AUDIT_2026-08-12_DEEP.md Section 7.2/8.1, Recommendation 11: every
// section's own code is lazy-loaded so it lands in its own chunk,
// downloaded only when that section is actually selected - verified via
// `npm run build` (separate chunk files exist per section, not one
// bundle). A prior pass of this fix left ModelTransparency/LeagueConnector/
// TradeAnalyzer as plain static imports (comment claimed "still lazy-
// loaded", code didn't match) - caught by grepping the real build output
// for actual player data (Josh Allen/Matthew Stafford/etc. showed up in
// main.js) rather than trusting the comment. TradeAnalyzer statically
// imports its own real data files directly, so leaving its COMPONENT
// import eager was enough to drag fantasy_rankings_2026.json/
// trade_scores_2026.json into the main bundle even though none of them
// call useSeason(). Since the Fantasy Tab Consolidation task, LeagueConnector
// and TradeAnalyzer are no longer imported here directly - they're static
// imports inside the single lazy-loaded Fantasy.js instead, so they still
// land in a real, separate chunk (Fantasy's), just shared with
// FantasyRankings/BreakoutAlerts rather than each getting their own.
const GamePredictions = lazy(() => import('./components/GamePredictions'));
const Fantasy = lazy(() => import('./components/Fantasy'));
const SeasonProjections = lazy(() => import('./components/SeasonProjections'));
const AccuracyTracker = lazy(() => import('./components/AccuracyTracker'));
const WeeklySummary = lazy(() => import('./components/WeeklySummary'));
const BettingAnalysis = lazy(() => import('./components/BettingAnalysis'));
const ModelTransparency = lazy(() => import('./components/ModelTransparency'));

// ModelTransparency is genuinely season-independent (doesn't call
// useSeason() anywhere) - its code is still lazy-loaded (above), just
// deliberately NOT gated behind the season-data loading placeholder below,
// since gating it on data it never uses would be a real, needless
// regression. Fantasy (Rankings/Trade Analyzer/League Connection/Breakout
// Alerts) mixes season-dependent and season-independent subtabs under one
// section id, so it isn't gated here either - it gates its own subtabs
// internally instead (see Fantasy.js) to avoid blocking the
// season-independent ones on data they don't need.
const SEASON_DATA_SECTIONS = new Set(['games', 'projections', 'accuracy', 'summary', 'betting']);

const SECTION_COMPONENTS = {
  games: GamePredictions,
  fantasy: Fantasy,
  projections: SeasonProjections,
  accuracy: AccuracyTracker,
  summary: WeeklySummary,
  transparency: ModelTransparency,
  betting: BettingAnalysis,
};

// AUDIT_2026-08-12_DEEP.md Section 7.2: no error boundary existed anywhere
// in the tree, so one bad render in any section would crash the whole app
// instead of just that section. Class component because getDerivedStateFromError/
// componentDidCatch have no hook equivalent.
class SectionErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error) {
    return { hasError: true, error };
  }

  componentDidCatch(error, errorInfo) {
    console.error('Section render failed:', error, errorInfo);
  }

  componentDidUpdate(prevProps) {
    if (this.state.hasError && prevProps.sectionKey !== this.props.sectionKey) {
      // eslint-disable-next-line react/no-did-update-set-state
      this.setState({ hasError: false, error: null });
    }
  }

  render() {
    if (this.state.hasError) {
      return (
        <div className="section-error-boundary">
          <h3>Section Error</h3>
          <p>This section encountered an error. Try selecting another section or refreshing the page.</p>
          <details className="section-error-boundary__details">
            <summary>Details</summary>
            <pre>{this.state.error?.toString()}</pre>
          </details>
        </div>
      );
    }

    return this.props.children;
  }
}

function LoadingPlaceholder({ label }) {
  return <div className="section-loading-placeholder">{label}</div>;
}

function AppContent({ activeSection, setActiveSection }) {
  const { dataLoading } = useSeason();
  const ActiveComponent = SECTION_COMPONENTS[activeSection] || GamePredictions;
  const waitingOnSeasonData = dataLoading && SEASON_DATA_SECTIONS.has(activeSection);

  return (
    <div className="app-container">
      <Navigation activeSection={activeSection} onSectionChange={setActiveSection} />
      <SeasonSelector />
      <main className="section-content">
        <SectionErrorBoundary sectionKey={activeSection}>
          {waitingOnSeasonData ? (
            <LoadingPlaceholder label="Loading season data..." />
          ) : (
            <Suspense fallback={<LoadingPlaceholder label="Loading section..." />}>
              <ActiveComponent />
            </Suspense>
          )}
        </SectionErrorBoundary>
      </main>
      {LAST_UPDATED && (
        <footer className="app-footer">
          Data last updated: {formatLastUpdated(LAST_UPDATED)}
        </footer>
      )}
    </div>
  );
}

export default function App() {
  const [activeSection, setActiveSection] = useState(DEFAULT_SECTION);

  return (
    <SeasonProvider>
      <AppContent activeSection={activeSection} setActiveSection={setActiveSection} />
    </SeasonProvider>
  );
}
