# Grasim Industries Ltd. (GRASIM)

## Backtest Summary

- **Window:** 2022-09-05 00:00:00 → 2026-05-08 00:00:00 (911 bars)
- **Last close:** 2968.60
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
| ENTRY1 | 9 |
| ENTRY2 | 0 |
| PARTIAL | 3 |
| TARGET_HIT | 2 |
| STOP_HIT | 7 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 12 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 8 / 4
- **Target hits / Stop hits / Partials:** 2 / 7 / 3
- **Avg / median % per leg:** 1.01% / 2.49%
- **Sum % (uncompounded):** 12.17%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 12 | 8 | 66.7% | 2 | 7 | 3 | 1.01% | 12.2% |
| BUY @ 2nd Alert (retest1) | 12 | 8 | 66.7% | 2 | 7 | 3 | 1.01% | 12.2% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 12 | 8 | 66.7% | 2 | 7 | 3 | 1.01% | 12.2% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-07-03 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-03 00:00:00 | 1786.60 | 1692.78 | 1739.34 | Stage2 pullback-breakout RSI=61 vol=1.7x ATR=33.17 |
| Stop hit — per-position SL triggered | 2023-07-07 00:00:00 | 1736.85 | 1695.32 | 1745.02 | SL hit (bars_held=4) |

### Cycle 2 — BUY (started 2023-07-17 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-17 00:00:00 | 1785.95 | 1699.03 | 1751.94 | Stage2 pullback-breakout RSI=58 vol=1.7x ATR=31.95 |
| Stop hit — per-position SL triggered | 2023-08-01 00:00:00 | 1830.42 | 1710.73 | 1794.91 | Time-stop (10d <3%) |

### Cycle 3 — BUY (started 2023-08-07 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-07 00:00:00 | 1848.25 | 1715.25 | 1805.54 | Stage2 pullback-breakout RSI=63 vol=1.6x ATR=32.68 |
| Stop hit — per-position SL triggered | 2023-08-14 00:00:00 | 1799.23 | 1720.58 | 1812.42 | SL hit (bars_held=5) |

### Cycle 4 — BUY (started 2023-09-04 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-04 00:00:00 | 1833.76 | 1730.15 | 1799.89 | Stage2 pullback-breakout RSI=60 vol=1.5x ATR=30.71 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-12 00:00:00 | 1895.18 | 1736.93 | 1821.54 | T1 booked 50% @ 1895.18 |
| Target hit | 2023-10-04 00:00:00 | 1888.34 | 1761.46 | 1896.42 | Trail-exit close<EMA20 |

### Cycle 5 — BUY (started 2023-10-12 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-12 00:00:00 | 1993.87 | 1770.90 | 1911.51 | Stage2 pullback-breakout RSI=66 vol=2.3x ATR=39.20 |
| Stop hit — per-position SL triggered | 2023-10-19 00:00:00 | 1935.07 | 1780.34 | 1932.19 | SL hit (bars_held=5) |

### Cycle 6 — BUY (started 2023-11-17 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-17 00:00:00 | 1962.75 | 1803.13 | 1920.56 | Stage2 pullback-breakout RSI=62 vol=1.7x ATR=33.33 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-01 00:00:00 | 2029.42 | 1817.81 | 1954.29 | T1 booked 50% @ 2029.42 |
| Target hit | 2023-12-22 00:00:00 | 2038.10 | 1853.58 | 2048.76 | Trail-exit close<EMA20 |

### Cycle 7 — BUY (started 2024-02-20 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-20 00:00:00 | 2192.25 | 1934.73 | 2113.28 | Stage2 pullback-breakout RSI=60 vol=1.9x ATR=58.60 |
| Stop hit — per-position SL triggered | 2024-03-04 00:00:00 | 2234.15 | 1960.91 | 2177.25 | Time-stop (10d <3%) |

### Cycle 8 — BUY (started 2024-03-28 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-28 00:00:00 | 2287.30 | 1997.85 | 2207.16 | Stage2 pullback-breakout RSI=63 vol=2.1x ATR=55.04 |
| Stop hit — per-position SL triggered | 2024-04-15 00:00:00 | 2237.40 | 2024.66 | 2250.79 | Time-stop (10d <3%) |

### Cycle 9 — BUY (started 2024-04-23 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-23 00:00:00 | 2370.15 | 2037.07 | 2264.22 | Stage2 pullback-breakout RSI=66 vol=2.5x ATR=53.01 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-03 00:00:00 | 2476.16 | 2061.33 | 2334.74 | T1 booked 50% @ 2476.16 |
| Stop hit — per-position SL triggered | 2024-05-08 00:00:00 | 2377.35 | 2071.84 | 2355.47 | Time-stop (10d <3%) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-07-03 00:00:00 | 1786.60 | 2023-07-07 00:00:00 | 1736.85 | STOP_HIT | 1.00 | -2.78% |
| BUY | retest1 | 2023-07-17 00:00:00 | 1785.95 | 2023-08-01 00:00:00 | 1830.42 | STOP_HIT | 1.00 | 2.49% |
| BUY | retest1 | 2023-08-07 00:00:00 | 1848.25 | 2023-08-14 00:00:00 | 1799.23 | STOP_HIT | 1.00 | -2.65% |
| BUY | retest1 | 2023-09-04 00:00:00 | 1833.76 | 2023-09-12 00:00:00 | 1895.18 | PARTIAL | 0.50 | 3.35% |
| BUY | retest1 | 2023-09-04 00:00:00 | 1833.76 | 2023-10-04 00:00:00 | 1888.34 | TARGET_HIT | 0.50 | 2.98% |
| BUY | retest1 | 2023-10-12 00:00:00 | 1993.87 | 2023-10-19 00:00:00 | 1935.07 | STOP_HIT | 1.00 | -2.95% |
| BUY | retest1 | 2023-11-17 00:00:00 | 1962.75 | 2023-12-01 00:00:00 | 2029.42 | PARTIAL | 0.50 | 3.40% |
| BUY | retest1 | 2023-11-17 00:00:00 | 1962.75 | 2023-12-22 00:00:00 | 2038.10 | TARGET_HIT | 0.50 | 3.84% |
| BUY | retest1 | 2024-02-20 00:00:00 | 2192.25 | 2024-03-04 00:00:00 | 2234.15 | STOP_HIT | 1.00 | 1.91% |
| BUY | retest1 | 2024-03-28 00:00:00 | 2287.30 | 2024-04-15 00:00:00 | 2237.40 | STOP_HIT | 1.00 | -2.18% |
| BUY | retest1 | 2024-04-23 00:00:00 | 2370.15 | 2024-05-03 00:00:00 | 2476.16 | PARTIAL | 0.50 | 4.47% |
| BUY | retest1 | 2024-04-23 00:00:00 | 2370.15 | 2024-05-08 00:00:00 | 2377.35 | STOP_HIT | 0.50 | 0.30% |
