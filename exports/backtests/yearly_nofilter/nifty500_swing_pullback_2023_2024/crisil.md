# CRISIL Ltd. (CRISIL)

## Backtest Summary

- **Window:** 2022-09-05 00:00:00 → 2026-05-08 00:00:00 (911 bars)
- **Last close:** 4162.30
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 0 |
| ALERT1 | 0 |
| ALERT2 | 0 |
| ALERT2_SKIP | 0 |
| ALERT3 | 0 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 5 |
| ENTRY2 | 0 |
| PARTIAL | 0 |
| TARGET_HIT | 0 |
| STOP_HIT | 5 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 5 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 0 / 5
- **Target hits / Stop hits / Partials:** 0 / 5 / 0
- **Avg / median % per leg:** -3.91% / -3.90%
- **Sum % (uncompounded):** -19.57%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 5 | 0 | 0.0% | 0 | 5 | 0 | -3.91% | -19.6% |
| BUY @ 2nd Alert (retest1) | 5 | 0 | 0.0% | 0 | 5 | 0 | -3.91% | -19.6% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 5 | 0 | 0.0% | 0 | 5 | 0 | -3.91% | -19.6% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-08-10 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-10 00:00:00 | 3899.25 | 3494.29 | 3857.79 | Stage2 pullback-breakout RSI=55 vol=1.8x ATR=81.80 |
| Stop hit — per-position SL triggered | 2023-08-14 00:00:00 | 3776.54 | 3501.90 | 3861.05 | SL hit (bars_held=2) |

### Cycle 2 — BUY (started 2023-08-22 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-22 00:00:00 | 4023.50 | 3521.88 | 3883.57 | Stage2 pullback-breakout RSI=62 vol=1.8x ATR=104.59 |
| Stop hit — per-position SL triggered | 2023-08-28 00:00:00 | 3866.62 | 3539.18 | 3907.97 | SL hit (bars_held=4) |

### Cycle 3 — BUY (started 2023-10-10 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-10 00:00:00 | 4070.65 | 3626.36 | 3895.91 | Stage2 pullback-breakout RSI=68 vol=3.3x ATR=89.00 |
| Stop hit — per-position SL triggered | 2023-10-12 00:00:00 | 3937.14 | 3633.49 | 3912.47 | SL hit (bars_held=2) |

### Cycle 4 — BUY (started 2023-10-19 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-19 00:00:00 | 4141.50 | 3649.81 | 3939.37 | Stage2 pullback-breakout RSI=65 vol=2.8x ATR=113.73 |
| Stop hit — per-position SL triggered | 2023-10-23 00:00:00 | 3970.90 | 3658.90 | 3969.48 | SL hit (bars_held=2) |

### Cycle 5 — BUY (started 2023-11-06 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-06 00:00:00 | 4383.55 | 3694.39 | 4050.06 | Stage2 pullback-breakout RSI=67 vol=2.1x ATR=149.89 |
| Stop hit — per-position SL triggered | 2023-11-09 00:00:00 | 4158.72 | 3711.05 | 4102.72 | SL hit (bars_held=3) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-08-10 00:00:00 | 3899.25 | 2023-08-14 00:00:00 | 3776.54 | STOP_HIT | 1.00 | -3.15% |
| BUY | retest1 | 2023-08-22 00:00:00 | 4023.50 | 2023-08-28 00:00:00 | 3866.62 | STOP_HIT | 1.00 | -3.90% |
| BUY | retest1 | 2023-10-10 00:00:00 | 4070.65 | 2023-10-12 00:00:00 | 3937.14 | STOP_HIT | 1.00 | -3.28% |
| BUY | retest1 | 2023-10-19 00:00:00 | 4141.50 | 2023-10-23 00:00:00 | 3970.90 | STOP_HIT | 1.00 | -4.12% |
| BUY | retest1 | 2023-11-06 00:00:00 | 4383.55 | 2023-11-09 00:00:00 | 4158.72 | STOP_HIT | 1.00 | -5.13% |
