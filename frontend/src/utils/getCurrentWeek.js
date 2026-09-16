// Real current week: the first week with any game whose real outcome is
// still unknown (actual_winner still null) - i.e. the week to default the
// UI to. Deliberately NOT keyed on a `game_completed` field - no such
// field exists anywhere in games_2026.json (verified against the real
// schema; GameCard.js's own real completion check is
// `actual_home_score !== null`) - a spec that assumed one would have had
// every week look "incomplete" and always returned week 1, silently
// defeating the whole point of this function.
export function getCurrentWeek(games) {
  if (!games || games.length === 0) return 1;
  const weeks = [...new Set(games.map((g) => g.week))].sort((a, b) => a - b);
  for (const week of weeks) {
    const weekGames = games.filter((g) => g.week === week);
    const hasIncomplete = weekGames.some((g) => g.actual_winner == null);
    if (hasIncomplete) return week;
  }
  // Every real game in the data has a result - default to the latest week.
  return weeks[weeks.length - 1] ?? 1;
}
