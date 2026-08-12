import React from 'react';
import { useSeason } from '../context/SeasonContext';
import '../styles/SeasonSelector.css';

export default function SeasonSelector() {
  const { selectedSeason, setSelectedSeason, availableSeasons, seasonLabel, hasResults } = useSeason();

  return (
    <div className="season-selector-container">
      <div className="season-selector">
        <label htmlFor="season-select">Season</label>
        <select
          id="season-select"
          value={selectedSeason}
          onChange={(e) => setSelectedSeason(Number(e.target.value))}
          className="season-select"
        >
          {availableSeasons.map((year) => (
            <option key={year} value={year}>
              {year}
            </option>
          ))}
        </select>
        <span className="season-label-text">{seasonLabel}</span>
        {!hasResults && (
          <span className="preseason-note">
            Real schedule + preseason model predictions only - the {selectedSeason} season hasn&apos;t
            been played yet, so no real results/accuracy/fantasy/betting data exists.
          </span>
        )}
      </div>
    </div>
  );
}
