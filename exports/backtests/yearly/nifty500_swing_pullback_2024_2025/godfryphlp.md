# Godfrey Phillips India Ltd. (GODFRYPHLP)

## Backtest Summary

- **Window:** 2023-09-04 05:30:00 → 2026-05-08 05:30:00 (663 bars)
- **Last close:** 2424.80
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
| ENTRY1 | 4 |
| ENTRY2 | 0 |
| PARTIAL | 1 |
| TARGET_HIT | 0 |
| STOP_HIT | 4 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 5 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 1 / 4
- **Target hits / Stop hits / Partials:** 0 / 4 / 1
- **Avg / median % per leg:** -1.70% / -7.05%
- **Sum % (uncompounded):** -8.48%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 5 | 1 | 20.0% | 0 | 4 | 1 | -1.70% | -8.5% |
| BUY @ 2nd Alert (retest1) | 5 | 1 | 20.0% | 0 | 4 | 1 | -1.70% | -8.5% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 5 | 1 | 20.0% | 0 | 4 | 1 | -1.70% | -8.5% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-07-15 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-15 05:30:00 | 1578.08 | 1036.25 | 1398.65 | Stage2 pullback-breakout RSI=68 vol=12.3x ATR=74.20 |
| Stop hit — per-position SL triggered | 2024-07-18 05:30:00 | 1466.79 | 1044.55 | 1408.71 | SL hit (bars_held=2) |

### Cycle 2 — BUY (started 2024-11-04 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-04 05:30:00 | 2279.75 | 1588.31 | 2220.89 | Stage2 pullback-breakout RSI=54 vol=1.8x ATR=110.88 |
| Stop hit — per-position SL triggered | 2024-11-13 05:30:00 | 2113.43 | 1632.52 | 2225.01 | SL hit (bars_held=7) |

### Cycle 3 — BUY (started 2025-02-07 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-07 05:30:00 | 1823.48 | 1661.48 | 1574.63 | Stage2 pullback-breakout RSI=66 vol=10.8x ATR=103.73 |
| Stop hit — per-position SL triggered | 2025-02-11 05:30:00 | 1667.89 | 1663.91 | 1612.01 | SL hit (bars_held=2) |

### Cycle 4 — BUY (started 2025-02-14 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-14 05:30:00 | 1999.83 | 1668.81 | 1669.94 | Stage2 pullback-breakout RSI=64 vol=6.2x ATR=143.98 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-17 05:30:00 | 2287.79 | 1675.55 | 1734.32 | T1 booked 50% @ 2287.79 |
| Stop hit — per-position SL triggered | 2025-02-20 05:30:00 | 1999.83 | 1689.60 | 1840.14 | SL hit (bars_held=4) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-07-15 05:30:00 | 1578.08 | 2024-07-18 05:30:00 | 1466.79 | STOP_HIT | 1.00 | -7.05% |
| BUY | retest1 | 2024-11-04 05:30:00 | 2279.75 | 2024-11-13 05:30:00 | 2113.43 | STOP_HIT | 1.00 | -7.30% |
| BUY | retest1 | 2025-02-07 05:30:00 | 1823.48 | 2025-02-11 05:30:00 | 1667.89 | STOP_HIT | 1.00 | -8.53% |
| BUY | retest1 | 2025-02-14 05:30:00 | 1999.83 | 2025-02-17 05:30:00 | 2287.79 | PARTIAL | 0.50 | 14.40% |
| BUY | retest1 | 2025-02-14 05:30:00 | 1999.83 | 2025-02-20 05:30:00 | 1999.83 | STOP_HIT | 0.50 | 0.00% |
