// Static reference data (team full names + official public brand colors) -
// not backend model output, so the no-fabrication rule for prediction/stat
// fields doesn't apply here. No logo images are used - there's no real logo
// asset in the backend, so team identity is shown as the abbreviation
// itself inside a colored box, rather than an emoji standing in for a logo.
//
// Primary/secondary pairs are the official NFL palette as supplied for this
// task. Keyed by the abbreviations that actually appear in the real 2025
// data (games_2025.json) - notably "LA" for the Rams, not "LAR", which is
// what the real schedules_2015_2025.csv / integrated_game_predictions_2025.csv
// use.
export const TEAM_NAMES = {
  ARI: 'Arizona Cardinals',
  ATL: 'Atlanta Falcons',
  BAL: 'Baltimore Ravens',
  BUF: 'Buffalo Bills',
  CAR: 'Carolina Panthers',
  CHI: 'Chicago Bears',
  CIN: 'Cincinnati Bengals',
  CLE: 'Cleveland Browns',
  DAL: 'Dallas Cowboys',
  DEN: 'Denver Broncos',
  DET: 'Detroit Lions',
  GB: 'Green Bay Packers',
  HOU: 'Houston Texans',
  IND: 'Indianapolis Colts',
  JAX: 'Jacksonville Jaguars',
  KC: 'Kansas City Chiefs',
  LA: 'Los Angeles Rams',
  LAC: 'Los Angeles Chargers',
  LV: 'Las Vegas Raiders',
  MIA: 'Miami Dolphins',
  MIN: 'Minnesota Vikings',
  NE: 'New England Patriots',
  NO: 'New Orleans Saints',
  NYG: 'New York Giants',
  NYJ: 'New York Jets',
  PHI: 'Philadelphia Eagles',
  PIT: 'Pittsburgh Steelers',
  SEA: 'Seattle Seahawks',
  SF: 'San Francisco 49ers',
  TB: 'Tampa Bay Buccaneers',
  TEN: 'Tennessee Titans',
  WAS: 'Washington Commanders',
};

export const TEAM_COLORS = {
  ARI: '#97233F',
  ATL: '#A71930',
  BAL: '#241773',
  BUF: '#00338D',
  CAR: '#0085CA',
  CHI: '#0B162A',
  CIN: '#FB4F14',
  CLE: '#311D00',
  DAL: '#003594',
  DEN: '#FB4F14',
  DET: '#0076B6',
  GB: '#203731',
  HOU: '#03202F',
  IND: '#002C5F',
  JAX: '#006687',
  KC: '#E31828',
  LA: '#003594',
  LAC: '#0080B4',
  LV: '#000000',
  MIA: '#008E97',
  MIN: '#4F2683',
  NE: '#002244',
  NO: '#D3BC8D',
  NYG: '#0B2265',
  NYJ: '#125740',
  PHI: '#004C54',
  PIT: '#27251A',
  SEA: '#002244',
  SF: '#FFB612',
  TB: '#092C5F',
  TEN: '#0C2C56',
  WAS: '#5A1D34',
};

export const TEAM_COLORS_SECONDARY = {
  ARI: '#000000',
  ATL: '#000000',
  BAL: '#000000',
  BUF: '#C60C30',
  CAR: '#000000', // overridden from Process Blue #0085CA - that duplicates CAR's primary color
  CHI: '#C83803',
  CIN: '#000000',
  CLE: '#FF3C00',
  DAL: '#869397',
  DEN: '#002244',
  DET: '#B0B7BC',
  GB: '#FFB612',
  HOU: '#A71930',
  IND: '#FFFFFF',
  JAX: '#D7A22A',
  KC: '#FFB81C',
  LA: '#FFA300',
  LAC: '#FFC20E',
  LV: '#A5ACAF',
  MIA: '#FC4C02',
  MIN: '#FFC62F',
  NE: '#C60C30',
  NO: '#D3BC8D',
  NYG: '#A71930',
  NYJ: '#FFFFFF',
  PHI: '#A5ACAF',
  PIT: '#FFB612',
  SEA: '#69BE28',
  SF: '#B3995D',
  TB: '#34302B',
  TEN: '#4B92DB', // light blue accent - "Red" label in the source table was a typo, hex is correct
  WAS: '#FFB612',
};

export function teamName(abbr) {
  return TEAM_NAMES[abbr] || abbr;
}

export function teamColor(abbr) {
  return TEAM_COLORS[abbr] || '#888888';
}

export function teamSecondaryColor(abbr) {
  return TEAM_COLORS_SECONDARY[abbr] || '#333333';
}

// WCAG-ish relative luminance -> pick black or white text so team boxes
// stay readable even where the secondary color is light (several teams'
// secondaries are gold/silver/light blue, e.g. WAS/DAL/SEA - white text on
// those would be low-contrast, which defeats the point of this fix).
export function readableTextColor(hexColor) {
  const hex = (hexColor || '#333333').replace('#', '');
  const r = parseInt(hex.substring(0, 2), 16) / 255;
  const g = parseInt(hex.substring(2, 4), 16) / 255;
  const b = parseInt(hex.substring(4, 6), 16) / 255;
  const toLinear = (c) => (c <= 0.03928 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4);
  const luminance = 0.2126 * toLinear(r) + 0.7152 * toLinear(g) + 0.0722 * toLinear(b);
  return luminance > 0.45 ? '#0a0a0a' : '#ffffff';
}

// NFL team -> stadium timezone label (real NFL facts; spot-checked against
// this project's real 2025 schedule - e.g. LV's real "late window" home
// games show gametime 16:05/16:25, the real ET value for a 1:05/1:25 PM
// Pacific kickoff, confirming gametime is stored in ET, not stadium-local -
// see GameCard.js). ARI is labeled MT per convention, but note: Arizona
// doesn't observe DST, so its real numeric offset from ET actually matches
// Pacific during the real EDT months and Mountain only after the real fall
// DST change - handled in GameCard.js's stadiumOffsetHours(), not here.
export const TEAM_TIMEZONES = {
  BUF: 'ET', MIA: 'ET', NE: 'ET', NYJ: 'ET',
  BAL: 'ET', CIN: 'ET', CLE: 'ET', PIT: 'ET',
  HOU: 'CT', IND: 'ET', JAX: 'ET', TEN: 'CT',
  DEN: 'MT', KC: 'CT', LV: 'PT', LAC: 'PT',
  DAL: 'CT', NYG: 'ET', PHI: 'ET', WAS: 'ET',
  CHI: 'CT', DET: 'ET', GB: 'CT', MIN: 'CT',
  ATL: 'ET', CAR: 'ET', NO: 'CT', TB: 'ET',
  ARI: 'MT', LA: 'PT', SF: 'PT', SEA: 'PT',
};

export const TIMEZONE_LABELS = {
  ET: 'Eastern',
  CT: 'Central',
  MT: 'Mountain',
  PT: 'Pacific',
};
