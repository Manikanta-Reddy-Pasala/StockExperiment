# eClerx Services Ltd. (ECLERX)

## Backtest Summary

- **Window:** 2022-09-05 00:00:00 → 2026-05-11 00:00:00 (912 bars)
- **Last close:** 1630.00
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
| ENTRY1 | 6 |
| ENTRY2 | 0 |
| PARTIAL | 1 |
| TARGET_HIT | 1 |
| STOP_HIT | 5 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 7 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 2 / 5
- **Target hits / Stop hits / Partials:** 1 / 5 / 1
- **Avg / median % per leg:** -0.41% / -5.22%
- **Sum % (uncompounded):** -2.87%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 7 | 2 | 28.6% | 1 | 5 | 1 | -0.41% | -2.9% |
| BUY @ 2nd Alert (retest1) | 7 | 2 | 28.6% | 1 | 5 | 1 | -0.41% | -2.9% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 7 | 2 | 28.6% | 1 | 5 | 1 | -0.41% | -2.9% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-07-17 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-17 00:00:00 | 913.85 | 741.38 | 852.26 | Stage2 pullback-breakout RSI=64 vol=1.7x ATR=33.49 |
| Stop hit — per-position SL triggered | 2023-07-25 00:00:00 | 863.61 | 749.82 | 866.91 | SL hit (bars_held=6) |

### Cycle 2 — BUY (started 2023-09-11 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-11 00:00:00 | 895.45 | 774.90 | 838.86 | Stage2 pullback-breakout RSI=68 vol=2.4x ATR=25.66 |
| Stop hit — per-position SL triggered | 2023-09-12 00:00:00 | 856.96 | 775.91 | 842.35 | SL hit (bars_held=1) |

### Cycle 3 — BUY (started 2023-09-15 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-15 00:00:00 | 925.48 | 779.33 | 855.68 | Stage2 pullback-breakout RSI=69 vol=7.6x ATR=32.18 |
| Stop hit — per-position SL triggered | 2023-09-22 00:00:00 | 877.21 | 784.61 | 874.40 | SL hit (bars_held=4) |

### Cycle 4 — BUY (started 2023-11-10 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-10 00:00:00 | 1075.95 | 844.68 | 1008.94 | Stage2 pullback-breakout RSI=66 vol=4.3x ATR=43.25 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-13 00:00:00 | 1162.44 | 850.35 | 1031.25 | T1 booked 50% @ 1162.44 |
| Target hit | 2023-12-12 00:00:00 | 1238.85 | 922.99 | 1241.59 | Trail-exit close<EMA20 |

### Cycle 5 — BUY (started 2024-01-12 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-12 00:00:00 | 1354.53 | 994.64 | 1288.52 | Stage2 pullback-breakout RSI=64 vol=5.5x ATR=48.73 |
| Stop hit — per-position SL triggered | 2024-01-18 00:00:00 | 1281.44 | 1009.04 | 1311.82 | SL hit (bars_held=4) |

### Cycle 6 — BUY (started 2024-04-04 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-04 00:00:00 | 1238.65 | 1101.07 | 1208.04 | Stage2 pullback-breakout RSI=55 vol=2.6x ATR=46.59 |
| Stop hit — per-position SL triggered | 2024-04-22 00:00:00 | 1168.76 | 1112.57 | 1213.69 | SL hit (bars_held=10) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-07-17 00:00:00 | 913.85 | 2023-07-25 00:00:00 | 863.61 | STOP_HIT | 1.00 | -5.50% |
| BUY | retest1 | 2023-09-11 00:00:00 | 895.45 | 2023-09-12 00:00:00 | 856.96 | STOP_HIT | 1.00 | -4.30% |
| BUY | retest1 | 2023-09-15 00:00:00 | 925.48 | 2023-09-22 00:00:00 | 877.21 | STOP_HIT | 1.00 | -5.22% |
| BUY | retest1 | 2023-11-10 00:00:00 | 1075.95 | 2023-11-13 00:00:00 | 1162.44 | PARTIAL | 0.50 | 8.04% |
| BUY | retest1 | 2023-11-10 00:00:00 | 1075.95 | 2023-12-12 00:00:00 | 1238.85 | TARGET_HIT | 0.50 | 15.14% |
| BUY | retest1 | 2024-01-12 00:00:00 | 1354.53 | 2024-01-18 00:00:00 | 1281.44 | STOP_HIT | 1.00 | -5.40% |
| BUY | retest1 | 2024-04-04 00:00:00 | 1238.65 | 2024-04-22 00:00:00 | 1168.76 | STOP_HIT | 1.00 | -5.64% |
