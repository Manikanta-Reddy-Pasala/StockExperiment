# Info Edge (India) Ltd. (NAUKRI)

## Backtest Summary

- **Window:** 2023-09-04 05:30:00 → 2026-05-08 05:30:00 (663 bars)
- **Last close:** 978.35
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
| PARTIAL | 2 |
| TARGET_HIT | 2 |
| STOP_HIT | 2 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 6 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 4 / 2
- **Target hits / Stop hits / Partials:** 2 / 2 / 2
- **Avg / median % per leg:** 1.91% / 4.78%
- **Sum % (uncompounded):** 11.46%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 6 | 4 | 66.7% | 2 | 2 | 2 | 1.91% | 11.5% |
| BUY @ 2nd Alert (retest1) | 6 | 4 | 66.7% | 2 | 2 | 2 | 1.91% | 11.5% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 6 | 4 | 66.7% | 2 | 2 | 2 | 1.91% | 11.5% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-09-11 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-11 05:30:00 | 1533.95 | 1224.57 | 1486.42 | Stage2 pullback-breakout RSI=64 vol=2.2x ATR=36.64 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-19 05:30:00 | 1607.24 | 1244.24 | 1521.08 | T1 booked 50% @ 1607.24 |
| Target hit | 2024-10-17 05:30:00 | 1595.92 | 1311.48 | 1619.84 | Trail-exit close<EMA20 |

### Cycle 2 — BUY (started 2024-11-06 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-06 05:30:00 | 1599.40 | 1341.96 | 1561.06 | Stage2 pullback-breakout RSI=55 vol=1.6x ATR=48.79 |
| Stop hit — per-position SL triggered | 2024-11-08 05:30:00 | 1526.22 | 1346.19 | 1559.87 | SL hit (bars_held=2) |

### Cycle 3 — BUY (started 2024-11-22 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-22 05:30:00 | 1599.99 | 1361.63 | 1552.26 | Stage2 pullback-breakout RSI=55 vol=3.6x ATR=55.66 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-05 05:30:00 | 1711.31 | 1388.18 | 1625.08 | T1 booked 50% @ 1711.31 |
| Target hit | 2025-01-07 05:30:00 | 1687.73 | 1456.93 | 1728.85 | Trail-exit close<EMA20 |

### Cycle 4 — BUY (started 2025-02-05 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-05 05:30:00 | 1588.23 | 1469.17 | 1544.72 | Stage2 pullback-breakout RSI=54 vol=1.6x ATR=55.27 |
| Stop hit — per-position SL triggered | 2025-02-11 05:30:00 | 1505.33 | 1472.81 | 1549.24 | SL hit (bars_held=4) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-09-11 05:30:00 | 1533.95 | 2024-09-19 05:30:00 | 1607.24 | PARTIAL | 0.50 | 4.78% |
| BUY | retest1 | 2024-09-11 05:30:00 | 1533.95 | 2024-10-17 05:30:00 | 1595.92 | TARGET_HIT | 0.50 | 4.04% |
| BUY | retest1 | 2024-11-06 05:30:00 | 1599.40 | 2024-11-08 05:30:00 | 1526.22 | STOP_HIT | 1.00 | -4.58% |
| BUY | retest1 | 2024-11-22 05:30:00 | 1599.99 | 2024-12-05 05:30:00 | 1711.31 | PARTIAL | 0.50 | 6.96% |
| BUY | retest1 | 2024-11-22 05:30:00 | 1599.99 | 2025-01-07 05:30:00 | 1687.73 | TARGET_HIT | 0.50 | 5.48% |
| BUY | retest1 | 2025-02-05 05:30:00 | 1588.23 | 2025-02-11 05:30:00 | 1505.33 | STOP_HIT | 1.00 | -5.22% |
