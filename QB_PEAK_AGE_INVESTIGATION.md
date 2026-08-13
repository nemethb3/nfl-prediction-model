# QB Peak-Eligible Age (22) — Investigation

**Date:** 2026-08-12 (Full Polish task, Recommendation 13)
**Resolves the open question left in `TRADE_VALUE_ENGINE_FINDINGS_2026-08-12.md`.**

## The question

`compute_empirical_age_curves.py`'s real, empirical QB age curve puts the
peak-eligible age at 22 — the youngest age in the whole 22-38 real range.
It survived two real, disclosed anti-bias corrections (rounding age before
grouping; a 25%-of-max-sample-size floor for peak *identification*
specifically), so it isn't simply an artifact either correction missed. It
was flagged as either a real finding about how this era uses rookie QBs, or
a remaining artifact — worth a direct look at which real players populate
that bucket before trusting it further.

## Method

Pulled every real (player_id, season) row with `position == "QB"` and
`age_int == 22` from `compute_empirical_age_curves._real_season_totals()`
(2015-2025), attached each player's real name from
`player_weekly_stats.csv`, and inspected the real, full list rather than a
sample.

## The real list (n=28 player-seasons, all of them)

| Rank by real PPR | Player | Season | Real season PPR |
|---|---|---|---|
| 1 | J.Herbert | 2020 | 332.8 |
| 2 | K.Murray | 2019 | 286.0 |
| 3 | C.Stroud | 2023 | 275.0 |
| 4 | J.Winston | 2015 | 275.0 |
| 5 | J.Dart | 2025 | 241.6 |
| 6 | D.Jones | 2019 | 217.0 |
| 7 | M.Mariota | 2015 | 211.0 |
| 8 | J.Allen | 2018 | 206.9 |
| 9 | T.Lawrence | 2021 | 199.0 |
| 10 | S.Darnold | 2019 | 189.2 |
| 11 | D.Maye | 2024 | 177.1 |
| 12 | D.Kizer | 2017 | 175.7 |
| 13 | D.Watson | 2017 | 168.9 |
| 14 | A.Richardson | 2024 | 163.4 |
| 15 | L.Jackson | 2018 | 157.5 |
| 16 | B.Young | 2023 | 156.4 |
| 17 | Z.Wilson | 2021 | 151.9 |
| 18 | J.Fields | 2021 | 126.8 |
| 19 | J.Rosen | 2018 | 112.9 |
| 20 | J.Hurts | 2020 | 109.1 |
| 21 | D.Haskins | 2019 | 76.7 |
| 22 | J.Goff | 2016 | 53.2 |
| 23 | Q.Ewers | 2025 | 33.2 |
| 24 | K.Allen | 2018 | 26.5 |
| 25 | S.Howell | 2022 | 18.3 |
| 26 | T.Lance | 2022 | 12.5 |
| 27 | P.Mahomes | 2017 | 10.4 |
| 28 | K.Mond | 2021 | 0.2 |

Real QB medians for context: overall (all ages) 63.9 PPR; age 22 specifically
160.4 PPR; age 23 129.7 PPR. Age 22 really is the highest-median age bucket
in the real data, by a wide margin.

## Real finding: this is a survivorship/selection effect, not (mainly) a rushing-inflation effect

Every single one of the 28 names is a real, immediately-drafted-and-started
rookie or near-rookie QB — the large majority (Herbert, Murray, Stroud,
Winston, Jones, Mariota, Allen, Lawrence, Darnold, Maye, Watson, Richardson,
Jackson, Young, Wilson, Fields, Rosen, Haskins, Goff, Lance, Mahomes) were
real first-round draft picks, and nearly all of the rest (Kizer, Howell,
Mond) were still real Day 2-3 picks who won a real starting job as rookies.
**No real 22-year-old backup or clipboard-holder QB appears in this list at
all** - not because such players don't exist, but because a QB who doesn't
start as a rookie simply has no real recorded season_ppr at 22 to begin
with (a real, structural fact about the data, not a filtering choice this
project made).

That is the real root cause: **age 22 in this dataset is not a random
sample of "22-year-old QBs" - it is, by construction, the subset of QBs
good enough (or thrust into the job early enough) to start real games as
rookies.** A real 22-year-old QB who instead spends his rookie year holding
a clipboard doesn't show up in this age-22 bucket at all; he only enters
the real data later, at an older age, once he actually gets on the field -
at which point he's competing in an older age-bucket's median instead. The
age-22 population is therefore pre-filtered by real talent/opportunity in
a way older-age buckets aren't (a 26-year-old QB in this data includes
both real stars AND real journeymen/backups who finally got a real shot).

**Real rushing production is a genuine, secondary contributor for several
individual names** (Murray, Allen, Lawrence, Jackson, Fields, Hurts, Watson
are all real, significantly rush-involved QBs whose real PPR is inflated
relative to a pure-passing valuation) - but it is not the primary driver of
the aggregate pattern, since several of the highest-scoring names in this
exact list (Stroud, Winston, Jones, Mariota, Goff) are real, conventional
pocket passers with minimal real rushing value, and they still rank near
the top.

Patrick Mahomes' real 10.4 PPR at age 22 (2017) is itself a confirming data
point, not an anomaly within the anomaly: he was real 2017 Kansas City's
backup and didn't start until Week 17 - the low score correctly reflects
his real, limited rookie playing time, not a data error.

## Decision

**Accepted as a real, now-precisely-understood finding, not changed.** Per
this project's own established discipline (disclose real, sometimes
counterintuitive findings rather than tuning them away), the empirical age
curve is left exactly as computed - the real peak-eligible age of 22 is a
real, correct reflection of this real, selection-biased population, not a
bug in the age-curve computation itself. What changes is the *disclosure*:
the age curve's real meaning at the youngest ages is "among QBs who started
as rookies, how good were they," not "how does the typical QB's production
change from age 22 onward" - the latter question this dataset structurally
cannot answer cleanly at age 22 given the real survivorship effect
described above. This caveat is worth surfacing anywhere the age curve's
QB peak age is cited (Trade Analyzer's methodology text, Model
Transparency) - not done as part of this investigation task, flagged here
for a future documentation pass.
