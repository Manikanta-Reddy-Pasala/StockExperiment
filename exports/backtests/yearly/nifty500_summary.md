# Yearly Backtest Summary — nifty500

_Generated: 2026-05-11T17:34:24_

- Capital: INR 200,000
- Max concurrent (headline): 2
- Years: 1 (1 windows)
- Models: ['ema_9_21']

## Penny stock filter
- ``swing_pullback``: ``min_adv_inr`` liquidity floor (₹5cr ADV default).
- ``ema_200_400`` / ``ema_9_21`` / ``orb_15min``: no min_price field yet — penny filter applies only at the swing_pullback strategy level.

## Yearly Headlines

| Model | Year | Window | RC | Taken | Skip | Final₹ | ROI% | MaxDD% | Open@End |
|-------|------|--------|---:|------:|-----:|-------:|-----:|-------:|---------:|
| ema_9_21 | 2023_2024 | 2023-05-12..2024-05-11 | 0 | 652 | 66681 | 262,186 | +31.09 | 43.55 | 2 |

## Per-model 3-year aggregate

| Model | Years run | Avg ROI% | Worst MaxDD% | Best ROI% | Worst ROI% |
|-------|----------:|---------:|-------------:|----------:|-----------:|
| ema_9_21 | 1 | +31.09 | 43.55 | +31.09 | +31.09 |
