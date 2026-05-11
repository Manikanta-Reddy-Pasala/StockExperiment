# Balkrishna Industries Ltd. (BALKRISIND)

## Backtest Summary

- **Window:** 2022-09-05 00:00:00 → 2026-05-11 00:00:00 (912 bars)
- **Last close:** 2161.50
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
- **Winners / losers:** 1 / 4
- **Target hits / Stop hits / Partials:** 0 / 5 / 0
- **Avg / median % per leg:** -2.46% / -3.55%
- **Sum % (uncompounded):** -12.31%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 5 | 1 | 20.0% | 0 | 5 | 0 | -2.46% | -12.3% |
| BUY @ 2nd Alert (retest1) | 5 | 1 | 20.0% | 0 | 5 | 0 | -2.46% | -12.3% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 5 | 1 | 20.0% | 0 | 5 | 0 | -2.46% | -12.3% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-07-26 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-26 00:00:00 | 2502.35 | 2172.39 | 2398.67 | Stage2 pullback-breakout RSI=62 vol=3.3x ATR=64.10 |
| Stop hit — per-position SL triggered | 2023-08-07 00:00:00 | 2406.20 | 2195.65 | 2437.65 | SL hit (bars_held=8) |

### Cycle 2 — BUY (started 2023-09-14 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-14 00:00:00 | 2502.60 | 2240.35 | 2402.94 | Stage2 pullback-breakout RSI=65 vol=2.9x ATR=56.56 |
| Stop hit — per-position SL triggered | 2023-09-29 00:00:00 | 2557.80 | 2269.54 | 2495.19 | Time-stop (10d <3%) |

### Cycle 3 — BUY (started 2023-11-08 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-08 00:00:00 | 2600.50 | 2338.17 | 2563.50 | Stage2 pullback-breakout RSI=58 vol=1.7x ATR=50.69 |
| Stop hit — per-position SL triggered | 2023-11-20 00:00:00 | 2524.47 | 2357.59 | 2578.41 | SL hit (bars_held=8) |

### Cycle 4 — BUY (started 2024-01-09 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-09 00:00:00 | 2623.50 | 2413.42 | 2548.62 | Stage2 pullback-breakout RSI=58 vol=1.8x ATR=62.00 |
| Stop hit — per-position SL triggered | 2024-01-18 00:00:00 | 2530.49 | 2427.45 | 2581.65 | SL hit (bars_held=7) |

### Cycle 5 — BUY (started 2024-01-20 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-20 00:00:00 | 2766.40 | 2432.93 | 2604.44 | Stage2 pullback-breakout RSI=65 vol=2.4x ATR=77.62 |
| Stop hit — per-position SL triggered | 2024-01-23 00:00:00 | 2649.98 | 2434.81 | 2606.11 | SL hit (bars_held=1) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-07-26 00:00:00 | 2502.35 | 2023-08-07 00:00:00 | 2406.20 | STOP_HIT | 1.00 | -3.84% |
| BUY | retest1 | 2023-09-14 00:00:00 | 2502.60 | 2023-09-29 00:00:00 | 2557.80 | STOP_HIT | 1.00 | 2.21% |
| BUY | retest1 | 2023-11-08 00:00:00 | 2600.50 | 2023-11-20 00:00:00 | 2524.47 | STOP_HIT | 1.00 | -2.92% |
| BUY | retest1 | 2024-01-09 00:00:00 | 2623.50 | 2024-01-18 00:00:00 | 2530.49 | STOP_HIT | 1.00 | -3.55% |
| BUY | retest1 | 2024-01-20 00:00:00 | 2766.40 | 2024-01-23 00:00:00 | 2649.98 | STOP_HIT | 1.00 | -4.21% |
