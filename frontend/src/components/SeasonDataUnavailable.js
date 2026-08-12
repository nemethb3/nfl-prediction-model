import React from 'react';

// Shared "not available yet" state for sections that need real completed-
// game data (fantasy outcomes, accuracy, weekly recap, betting results) -
// none of which exist for a season that hasn't been played. Shown instead
// of an empty or broken panel, same honesty standard as every other real
// gap disclosed throughout this project.
export default function SeasonDataUnavailable({ season, sectionName }) {
  return (
    <div className="season-data-unavailable">
      <div className="season-data-unavailable-icon">📅</div>
      <h2>Not available for {season}</h2>
      <p>
        {sectionName} needs real completed games to compute from, and the real {season} season
        hasn&apos;t been played yet. Switch to a completed season using the selector above to see
        this section, or check back once real {season} games start.
      </p>
    </div>
  );
}
