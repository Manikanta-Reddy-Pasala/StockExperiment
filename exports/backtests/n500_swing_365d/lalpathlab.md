# Dr. Lal Path Labs Ltd. (LALPATHLAB)

## Backtest Summary

- **Window:** 2024-09-02 05:30:00 → 2026-05-08 05:30:00 (417 bars)
- **Last close:** 1649.80
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
| TARGET_HIT | 1 |
| STOP_HIT | 3 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 5 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 2 / 3
- **Target hits / Stop hits / Partials:** 1 / 3 / 1
- **Avg / median % per leg:** -0.73% / -4.09%
- **Sum % (uncompounded):** -3.65%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 5 | 2 | 40.0% | 1 | 3 | 1 | -0.73% | -3.6% |
| BUY @ 2nd Alert (retest1) | 5 | 2 | 40.0% | 1 | 3 | 1 | -0.73% | -3.6% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 5 | 2 | 40.0% | 1 | 3 | 1 | -0.73% | -3.6% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-07-31 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-31 05:30:00 | 1575.45 | 1455.30 | 1498.99 | Stage2 pullback-breakout RSI=70 vol=1.8x ATR=41.98 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-14 05:30:00 | 1659.41 | 1470.11 | 1574.46 | T1 booked 50% @ 1659.41 |
| Target hit | 2025-09-01 05:30:00 | 1636.05 | 1490.00 | 1636.65 | Trail-exit close<EMA20 |

### Cycle 2 — BUY (started 2025-09-19 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-19 05:30:00 | 1736.75 | 1509.80 | 1645.12 | Stage2 pullback-breakout RSI=65 vol=2.8x ATR=49.87 |
| Stop hit — per-position SL triggered | 2025-09-23 05:30:00 | 1661.95 | 1513.16 | 1651.26 | SL hit (bars_held=2) |

### Cycle 3 — BUY (started 2025-11-03 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-03 05:30:00 | 1627.45 | 1526.89 | 1572.58 | Stage2 pullback-breakout RSI=58 vol=2.4x ATR=48.05 |
| Stop hit — per-position SL triggered | 2025-11-06 05:30:00 | 1555.37 | 1527.66 | 1571.18 | SL hit (bars_held=2) |

### Cycle 4 — BUY (started 2025-11-19 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-19 05:30:00 | 1604.05 | 1530.95 | 1571.04 | Stage2 pullback-breakout RSI=57 vol=2.2x ATR=43.68 |
| Stop hit — per-position SL triggered | 2025-11-24 05:30:00 | 1538.52 | 1531.96 | 1569.47 | SL hit (bars_held=3) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-07-31 05:30:00 | 1575.45 | 2025-08-14 05:30:00 | 1659.41 | PARTIAL | 0.50 | 5.33% |
| BUY | retest1 | 2025-07-31 05:30:00 | 1575.45 | 2025-09-01 05:30:00 | 1636.05 | TARGET_HIT | 0.50 | 3.85% |
| BUY | retest1 | 2025-09-19 05:30:00 | 1736.75 | 2025-09-23 05:30:00 | 1661.95 | STOP_HIT | 1.00 | -4.31% |
| BUY | retest1 | 2025-11-03 05:30:00 | 1627.45 | 2025-11-06 05:30:00 | 1555.37 | STOP_HIT | 1.00 | -4.43% |
| BUY | retest1 | 2025-11-19 05:30:00 | 1604.05 | 2025-11-24 05:30:00 | 1538.52 | STOP_HIT | 1.00 | -4.09% |
