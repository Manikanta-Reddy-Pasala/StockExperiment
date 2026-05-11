# Yearly Backtest Summary — nifty50

_Generated: 2026-05-11T01:32:39_

- Capital: INR 200,000
- Max concurrent (headline): 2
- Years: 3 (3 windows)
- Models: ['ema_200_400']

## Penny stock filter
- ``swing_pullback``: ``min_adv_inr`` liquidity floor (₹5cr ADV default).
- ``ema_200_400`` / ``ema_9_21`` / ``orb_15min``: no min_price field yet — penny filter applies only at the swing_pullback strategy level.

## Yearly Headlines

| Model | Year | Window | RC | Taken | Skip | Final₹ | ROI% | MaxDD% | Open@End |
|-------|------|--------|---:|------:|-----:|-------:|-----:|-------:|---------:|
| ema_200_400 | 2025_2026 | 2025-05-11..2026-05-11 | 0 | 39 | 173 | 171,983 | -14.01 | 19.97 | 2 |
| ema_200_400 | 2024_2025 | 2024-05-11..2025-05-11 | 0 | 108 | 446 | 249,640 | +24.82 | 16.82 | 2 |
| ema_200_400 | 2023_2024 | 2023-05-12..2024-05-11 | -9 | - | - | 0 | +0.00 | 0.00 | - |

## Per-model 3-year aggregate

| Model | Years run | Avg ROI% | Worst MaxDD% | Best ROI% | Worst ROI% |
|-------|----------:|---------:|-------------:|----------:|-----------:|
| ema_200_400 | 2 | +5.41 | 19.97 | +24.82 | -14.01 |
