# Yearly Backtest Summary — nifty50

_Generated: 2026-05-12T19:30:53_

- Capital: INR 1,000,000
- Max concurrent (headline): 2
- Years: 3 (3 windows)
- Models: ['ema_9_21', 'ema_200_400']

## Penny stock filter
- ``swing_pullback``: ``min_adv_inr`` liquidity floor (₹5cr ADV default).
- ``ema_200_400`` / ``ema_9_21`` / ``orb_15min``: no min_price field yet — penny filter applies only at the swing_pullback strategy level.

## Yearly Headlines

| Model | Year | Window | RC | Taken | Skip | Final₹ | ROI% | MaxDD% | Open@End |
|-------|------|--------|---:|------:|-----:|-------:|-----:|-------:|---------:|
| ema_9_21 | 2025_2026 | 2025-05-12..2026-05-12 | 0 | 213 | 2720 | 929,028 | -7.10 | 26.44 | 2 |
| ema_9_21 | 2024_2025 | 2024-05-12..2025-05-12 | 0 | 396 | 5730 | 797,902 | -20.21 | 39.29 | 2 |
| ema_9_21 | 2023_2024 | 2023-05-13..2024-05-12 | 0 | 554 | 8680 | 990,619 | -0.94 | 39.20 | 2 |
| ema_200_400 | 2025_2026 | 2025-05-12..2026-05-12 | 0 | 54 | 337 | 1,067,675 | +6.77 | 13.01 | 2 |
| ema_200_400 | 2024_2025 | 2024-05-12..2025-05-12 | 0 | 125 | 730 | 1,548,797 | +54.88 | 13.06 | 2 |
| ema_200_400 | 2023_2024 | 2023-05-13..2024-05-12 | 0 | 179 | 951 | 1,981,301 | +98.13 | 13.06 | 2 |

## Per-model 3-year aggregate

| Model | Years run | Avg ROI% | Worst MaxDD% | Best ROI% | Worst ROI% |
|-------|----------:|---------:|-------------:|----------:|-----------:|
| ema_9_21 | 3 | -9.42 | 39.29 | -0.94 | -20.21 |
| ema_200_400 | 3 | +53.26 | 13.06 | +98.13 | +6.77 |
