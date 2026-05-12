# Yearly Backtest Summary — nifty500

_Generated: 2026-05-12T15:48:04_

- Capital: INR 200,000
- Max concurrent (headline): 2
- Years: 1 (1 windows)
- Models: ['ema_200_400', 'ema_9_21', 'swing_pullback', 'orb_15min']

## Penny stock filter
- ``swing_pullback``: ``min_adv_inr`` liquidity floor (₹5cr ADV default).
- ``ema_200_400`` / ``ema_9_21`` / ``orb_15min``: no min_price field yet — penny filter applies only at the swing_pullback strategy level.

## Yearly Headlines

| Model | Year | Window | RC | Taken | Skip | Final₹ | ROI% | MaxDD% | Open@End |
|-------|------|--------|---:|------:|-----:|-------:|-----:|-------:|---------:|
| ema_200_400 | 2025_2026 | 2025-05-12..2026-05-12 | 0 | 90 | 3233 | 132,933 | -33.53 | 34.94 | 2 |
| ema_9_21 | 2025_2026 | 2025-05-12..2026-05-12 | 0 | 208 | 25890 | 143,389 | -28.31 | 41.33 | 2 |
| swing_pullback | 2025_2026 | 2025-05-12..2026-05-12 | 0 | 18 | 8 | 200,266 | +0.13 | 9.83 | 2 |
| orb_15min | 2025_2026 | 2025-05-12..2026-05-12 | 0 | 0 | 0 | 200,000 | +0.00 | 0.00 | 0 |

## Per-model 3-year aggregate

| Model | Years run | Avg ROI% | Worst MaxDD% | Best ROI% | Worst ROI% |
|-------|----------:|---------:|-------------:|----------:|-----------:|
| ema_200_400 | 1 | -33.53 | 34.94 | -33.53 | -33.53 |
| ema_9_21 | 1 | -28.31 | 41.33 | -28.31 | -28.31 |
| swing_pullback | 1 | +0.13 | 9.83 | +0.13 | +0.13 |
| orb_15min | 1 | +0.00 | 0.00 | +0.00 | +0.00 |
