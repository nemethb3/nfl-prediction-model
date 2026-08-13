// Enter/Space keyboard activation for a clickable, role="button" element -
// extracted after the same onKeyDown block was duplicated verbatim in
// GameCard.js and FantasyRankings.js (AUDIT_2026-08-12_DEEP.md Section 4.2/
// Recommendation 6).
export function useKeyboardToggle(onToggle) {
  return (e) => {
    if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault();
      onToggle();
    }
  };
}
