# Win Probability Model Backtesting Results

## Test Data (real, held out from any fitting)
- Train: real 2024 season, 272 games (`elo_game_spreads_2024.csv` + real Vegas `spread_line`)
- Test: real 2025 weeks 13-17, 71 games
- Sign-convention sanity check: real home-favorite win rate = 0.514

## Approach A: Heuristic (Vegas spread -> probability, asserted constant)
- Formula: p_home = 1 / (1 + exp(-spread / 2.5)) (scale asserted, not fit)
- Brier Score: 0.3149
- Log Loss: 0.9345

Calibration (predicted vs. real actual win rate, by bucket):
                         bucket  mean_predicted  mean_actual  n
(0.0034999999999999996, 0.0998]        0.046775     0.266667 15
                (0.0998, 0.269]        0.223038     0.400000 15
                 (0.269, 0.769]        0.597515     0.461538 13
                 (0.769, 0.931]        0.871971     0.466667 15
                 (0.931, 0.997]        0.968327     0.692308 13

## Approach B: Elo-based (real logistic regression, fit on real 2024 outcomes)
- Coefficients: intercept=0.1587, elo_diff_coef=0.018947
- Brier Score: 0.2874
- Log Loss: 0.9060

Calibration (predicted vs. real actual win rate, by bucket):
           bucket  mean_predicted  mean_actual  n
(0.00252, 0.0712]        0.027958     0.266667 15
  (0.0712, 0.291]        0.187666     0.500000 14
   (0.291, 0.669]        0.510178     0.214286 14
   (0.669, 0.866]        0.788917     0.571429 14
   (0.866, 0.993]        0.952486     0.714286 14

## Approach C: Vegas spread, fairly fit (added after investigating A losing to B)
A's initial loss to B contradicted this project's established finding that Vegas beats Elo at every checkpoint tested (README Key Finding #2). Investigated rather than shipped: A used an ASSERTED conversion constant while B was properly fit - not a fair comparison. Fit a logistic regression on real Vegas spread_line the same way, instead.
- Coefficients: intercept=-0.1266, spread_coef=0.198011
- Brier Score: 0.2512
- Log Loss: 0.6930

Calibration (predicted vs. real actual win rate, by bucket):
                       bucket  mean_predicted  mean_actual  n
(0.056299999999999996, 0.229]        0.153773     0.266667 15
               (0.229, 0.349]        0.320700     0.400000 15
               (0.349, 0.615]        0.521476     0.461538 13
               (0.615, 0.761]        0.700137     0.466667 15
                (0.761, 0.94]        0.845998     0.692308 13

## Winner
**VEGAS_FIT** (lowest real Brier score on the real holdout: heuristic=0.3149, elo=0.2874, vegas_fit=0.2512).

Confirms this project's established finding: a fairly-fit Vegas-based model beats both the unfit heuristic and the Elo-based model. The earlier apparent 'Elo beats Vegas' result was an artifact of comparing a fit model against an asserted-constant one, not a genuine reversal - caught by adding the fair comparison rather than shipping the first result.

## Deployed
`vegas_fit` is the model wired into generate_dashboard_data.py's win_prob_home/win_prob_away.