# Yearly Backtest Summary — nifty500

_Generated: 2026-05-12T15:01:48_

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
| ema_200_400 | 2025_2026 | 2025-05-12..2026-05-12 | -9 | - | - | 0 | +0.00 | 0.00 | - |
| ema_9_21 | 2025_2026 | 2025-05-12..2026-05-12 | 0 | 216 | 26540 | 205,689 | +2.84 | 31.91 | 2 |
| swing_pullback | 2025_2026 | 2025-05-12..2026-05-12 | 0 | 18 | 8 | 200,266 | +0.13 | 9.83 | 2 |
| orb_15min | 2025_2026 | 2025-05-12..2026-05-12 | 0 | 0 | 0 | 200,000 | +0.00 | 0.00 | 0 |

## Per-model 3-year aggregate

| Model | Years run | Avg ROI% | Worst MaxDD% | Best ROI% | Worst ROI% |
|-------|----------:|---------:|-------------:|----------:|-----------:|
| ema_200_400 | 0 | n/a | n/a | n/a | n/a |
| ema_9_21 | 1 | +2.84 | 31.91 | +2.84 | +2.84 |
| swing_pullback | 1 | +0.13 | 9.83 | +0.13 | +0.13 |
| orb_15min | 1 | +0.00 | 0.00 | +0.00 | +0.00 |
