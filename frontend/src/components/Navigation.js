import React, { useMemo, useState } from 'react';
import { DASHBOARD_SECTIONS } from '../constants/sections';
import { useSeason } from '../context/SeasonContext';
import { getCurrentWeek } from '../utils/getCurrentWeek';
import '../styles/Navigation.css';

export default function Navigation({ activeSection, onSectionChange }) {
  const [menuOpen, setMenuOpen] = useState(false);
  const { seasonData, hasResults } = useSeason();
  // Real, live-season-only indicator (see getCurrentWeek.js) - not shown
  // for a completed, historical season (hasResults - e.g. 2025), where
  // "currently" doesn't mean anything real.
  const currentWeek = useMemo(
    () => (!hasResults && seasonData ? getCurrentWeek(seasonData.games) : null),
    [hasResults, seasonData]);

  const handleSectionChange = (sectionId) => {
    onSectionChange(sectionId);
    setMenuOpen(false);
  };

  return (
    <nav className="dashboard-nav">
      <div className="nav-brand">NFL Predictions</div>
      {currentWeek != null && (
        <div className="current-week-indicator">Currently: Week {currentWeek}</div>
      )}

      <button
        className="hamburger"
        onClick={() => setMenuOpen(!menuOpen)}
        aria-label="Toggle menu"
        aria-expanded={menuOpen}
      >
        ☰
      </button>

      <ul className={`nav-sections ${menuOpen ? 'open' : ''}`}>
        {DASHBOARD_SECTIONS.map((section) => (
          <li key={section.id}>
            <button
              className={activeSection === section.id ? 'active' : ''}
              onClick={() => handleSectionChange(section.id)}
            >
              {section.label}
            </button>
          </li>
        ))}
      </ul>
    </nav>
  );
}
