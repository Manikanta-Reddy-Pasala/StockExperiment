# Yearly Backtest Summary — nifty50

_Generated: 2026-05-12T14:20:29_

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
| ema_9_21 | 2025_2026 | 2025-05-12..2026-05-12 | 0 | 213 | 2720 | 184,953 | -7.52 | 26.74 | 2 |
| swing_pullback | 2025_2026 | 2025-05-12..2026-05-12 | 0 | 3 | 0 | 197,810 | -1.09 | 2.02 | 0 |
| orb_15min | 2025_2026 | 2025-05-12..2026-05-12 | 0 | 0 | 0 | 200,000 | +0.00 | 0.00 | 0 |

## Per-model 3-year aggregate

| Model | Years run | Avg ROI% | Worst MaxDD% | Best ROI% | Worst ROI% |
|-------|----------:|---------:|-------------:|----------:|-----------:|
| ema_200_400 | 0 | n/a | n/a | n/a | n/a |
| ema_9_21 | 1 | -7.52 | 26.74 | -7.52 | -7.52 |
| swing_pullback | 1 | -1.09 | 2.02 | -1.09 | -1.09 |
| orb_15min | 1 | +0.00 | 0.00 | +0.00 | +0.00 |
