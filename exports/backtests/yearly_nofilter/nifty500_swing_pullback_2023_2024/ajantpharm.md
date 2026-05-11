# Ajanta Pharmaceuticals Ltd. (AJANTPHARM)

## Backtest Summary

- **Window:** 2022-09-05 00:00:00 → 2026-05-11 00:00:00 (912 bars)
- **Last close:** 3006.10
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
| PARTIAL | 4 |
| TARGET_HIT | 3 |
| STOP_HIT | 2 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 9 (incl. partial bookings)
- **Trades open at end:** 1
- **Winners / losers:** 7 / 2
- **Target hits / Stop hits / Partials:** 3 / 2 / 4
- **Avg / median % per leg:** 4.14% / 6.06%
- **Sum % (uncompounded):** 37.25%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 9 | 7 | 77.8% | 3 | 2 | 4 | 4.14% | 37.2% |
| BUY @ 2nd Alert (retest1) | 9 | 7 | 77.8% | 3 | 2 | 4 | 4.14% | 37.2% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 9 | 7 | 77.8% | 3 | 2 | 4 | 4.14% | 37.2% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-07-25 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-25 00:00:00 | 1511.75 | 1317.62 | 1434.85 | Stage2 pullback-breakout RSI=66 vol=7.1x ATR=50.22 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-28 00:00:00 | 1612.19 | 1324.74 | 1467.75 | T1 booked 50% @ 1612.19 |
| Target hit | 2023-08-25 00:00:00 | 1694.65 | 1396.70 | 1703.47 | Trail-exit close<EMA20 |

### Cycle 2 — BUY (started 2023-09-13 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-13 00:00:00 | 1821.25 | 1437.96 | 1730.29 | Stage2 pullback-breakout RSI=68 vol=8.2x ATR=61.33 |
| Stop hit — per-position SL triggered | 2023-09-15 00:00:00 | 1729.26 | 1444.18 | 1733.87 | SL hit (bars_held=2) |

### Cycle 3 — BUY (started 2023-11-01 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-01 00:00:00 | 1827.55 | 1523.24 | 1757.51 | Stage2 pullback-breakout RSI=62 vol=3.3x ATR=55.34 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-15 00:00:00 | 1938.23 | 1555.05 | 1826.57 | T1 booked 50% @ 1938.23 |
| Target hit | 2023-12-11 00:00:00 | 1887.00 | 1616.16 | 1922.01 | Trail-exit close<EMA20 |

### Cycle 4 — BUY (started 2023-12-22 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-22 00:00:00 | 1968.20 | 1640.79 | 1912.59 | Stage2 pullback-breakout RSI=62 vol=1.8x ATR=53.81 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-29 00:00:00 | 2075.82 | 1655.50 | 1947.76 | T1 booked 50% @ 2075.82 |
| Target hit | 2024-01-30 00:00:00 | 2111.45 | 1758.92 | 2164.26 | Trail-exit close<EMA20 |

### Cycle 5 — BUY (started 2024-02-29 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-29 00:00:00 | 2208.00 | 1837.66 | 2148.13 | Stage2 pullback-breakout RSI=57 vol=2.3x ATR=72.38 |
| Stop hit — per-position SL triggered | 2024-03-04 00:00:00 | 2099.43 | 1845.80 | 2138.57 | SL hit (bars_held=3) |

### Cycle 6 — BUY (started 2024-04-26 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-26 00:00:00 | 2193.50 | 1930.91 | 2138.62 | Stage2 pullback-breakout RSI=58 vol=2.0x ATR=70.38 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-03 00:00:00 | 2334.26 | 1943.90 | 2181.40 | T1 booked 50% @ 2334.26 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-07-25 00:00:00 | 1511.75 | 2023-07-28 00:00:00 | 1612.19 | PARTIAL | 0.50 | 6.64% |
| BUY | retest1 | 2023-07-25 00:00:00 | 1511.75 | 2023-08-25 00:00:00 | 1694.65 | TARGET_HIT | 0.50 | 12.10% |
| BUY | retest1 | 2023-09-13 00:00:00 | 1821.25 | 2023-09-15 00:00:00 | 1729.26 | STOP_HIT | 1.00 | -5.05% |
| BUY | retest1 | 2023-11-01 00:00:00 | 1827.55 | 2023-11-15 00:00:00 | 1938.23 | PARTIAL | 0.50 | 6.06% |
| BUY | retest1 | 2023-11-01 00:00:00 | 1827.55 | 2023-12-11 00:00:00 | 1887.00 | TARGET_HIT | 0.50 | 3.25% |
| BUY | retest1 | 2023-12-22 00:00:00 | 1968.20 | 2023-12-29 00:00:00 | 2075.82 | PARTIAL | 0.50 | 5.47% |
| BUY | retest1 | 2023-12-22 00:00:00 | 1968.20 | 2024-01-30 00:00:00 | 2111.45 | TARGET_HIT | 0.50 | 7.28% |
| BUY | retest1 | 2024-02-29 00:00:00 | 2208.00 | 2024-03-04 00:00:00 | 2099.43 | STOP_HIT | 1.00 | -4.92% |
| BUY | retest1 | 2024-04-26 00:00:00 | 2193.50 | 2024-05-03 00:00:00 | 2334.26 | PARTIAL | 0.50 | 6.42% |
