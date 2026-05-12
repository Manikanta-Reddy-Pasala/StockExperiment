# M&M (M&M)

## Backtest Summary

- **Window:** 2022-09-05 00:00:00 → 2026-05-11 00:00:00 (912 bars)
- **Last close:** 3261.90
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
| PARTIAL | 3 |
| TARGET_HIT | 1 |
| STOP_HIT | 4 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 8 (incl. partial bookings)
- **Trades open at end:** 1
- **Winners / losers:** 6 / 2
- **Target hits / Stop hits / Partials:** 1 / 4 / 3
- **Avg / median % per leg:** 2.88% / 3.99%
- **Sum % (uncompounded):** 23.01%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 8 | 6 | 75.0% | 1 | 4 | 3 | 2.88% | 23.0% |
| BUY @ 2nd Alert (retest1) | 8 | 6 | 75.0% | 1 | 4 | 3 | 2.88% | 23.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 8 | 6 | 75.0% | 1 | 4 | 3 | 2.88% | 23.0% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-08-07 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-07 00:00:00 | 1526.90 | 1343.92 | 1497.33 | Stage2 pullback-breakout RSI=57 vol=2.8x ATR=41.54 |
| Stop hit — per-position SL triggered | 2023-08-22 00:00:00 | 1550.50 | 1363.17 | 1529.70 | Time-stop (10d <3%) |

### Cycle 2 — BUY (started 2023-09-15 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-15 00:00:00 | 1601.10 | 1396.47 | 1562.10 | Stage2 pullback-breakout RSI=60 vol=2.5x ATR=33.92 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-18 00:00:00 | 1668.93 | 1398.90 | 1569.65 | T1 booked 50% @ 1668.93 |
| Stop hit — per-position SL triggered | 2023-09-21 00:00:00 | 1601.10 | 1403.05 | 1576.49 | SL hit (bars_held=3) |

### Cycle 3 — BUY (started 2023-11-09 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-09 00:00:00 | 1552.05 | 1440.07 | 1515.18 | Stage2 pullback-breakout RSI=56 vol=2.2x ATR=33.56 |
| Stop hit — per-position SL triggered | 2023-11-23 00:00:00 | 1545.85 | 1450.45 | 1537.70 | Time-stop (10d <3%) |

### Cycle 4 — BUY (started 2023-11-29 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-29 00:00:00 | 1619.10 | 1454.27 | 1549.11 | Stage2 pullback-breakout RSI=66 vol=1.5x ATR=32.32 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-05 00:00:00 | 1683.73 | 1462.11 | 1584.35 | T1 booked 50% @ 1683.73 |
| Stop hit — per-position SL triggered | 2023-12-13 00:00:00 | 1666.15 | 1474.18 | 1621.49 | Time-stop (10d <3%) |

### Cycle 5 — BUY (started 2024-02-15 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-15 00:00:00 | 1765.05 | 1540.27 | 1668.94 | Stage2 pullback-breakout RSI=65 vol=3.4x ATR=50.39 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-21 00:00:00 | 1865.84 | 1552.39 | 1728.77 | T1 booked 50% @ 1865.84 |
| Target hit | 2024-03-13 00:00:00 | 1853.70 | 1604.07 | 1875.02 | Trail-exit close<EMA20 |

### Cycle 6 — BUY (started 2024-04-30 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-30 00:00:00 | 2156.35 | 1705.38 | 2040.54 | Stage2 pullback-breakout RSI=68 vol=1.9x ATR=60.84 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-08-07 00:00:00 | 1526.90 | 2023-08-22 00:00:00 | 1550.50 | STOP_HIT | 1.00 | 1.55% |
| BUY | retest1 | 2023-09-15 00:00:00 | 1601.10 | 2023-09-18 00:00:00 | 1668.93 | PARTIAL | 0.50 | 4.24% |
| BUY | retest1 | 2023-09-15 00:00:00 | 1601.10 | 2023-09-21 00:00:00 | 1601.10 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-11-09 00:00:00 | 1552.05 | 2023-11-23 00:00:00 | 1545.85 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest1 | 2023-11-29 00:00:00 | 1619.10 | 2023-12-05 00:00:00 | 1683.73 | PARTIAL | 0.50 | 3.99% |
| BUY | retest1 | 2023-11-29 00:00:00 | 1619.10 | 2023-12-13 00:00:00 | 1666.15 | STOP_HIT | 0.50 | 2.91% |
| BUY | retest1 | 2024-02-15 00:00:00 | 1765.05 | 2024-02-21 00:00:00 | 1865.84 | PARTIAL | 0.50 | 5.71% |
| BUY | retest1 | 2024-02-15 00:00:00 | 1765.05 | 2024-03-13 00:00:00 | 1853.70 | TARGET_HIT | 0.50 | 5.02% |
