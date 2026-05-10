# Amber Enterprises India Ltd. (AMBER)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 8851.00
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
| ENTRY1 | 13 |
| ENTRY2 | 0 |
| PARTIAL | 5 |
| TARGET_HIT | 1 |
| STOP_HIT | 12 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 18 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 6 / 12
- **Target hits / Stop hits / Partials:** 1 / 12 / 5
- **Avg / median % per leg:** 0.14% / 0.00%
- **Sum % (uncompounded):** 2.49%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 12 | 5 | 41.7% | 1 | 7 | 4 | 0.29% | 3.5% |
| BUY @ 2nd Alert (retest1) | 12 | 5 | 41.7% | 1 | 7 | 4 | 0.29% | 3.5% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 6 | 1 | 16.7% | 0 | 5 | 1 | -0.16% | -1.0% |
| SELL @ 2nd Alert (retest1) | 6 | 1 | 16.7% | 0 | 5 | 1 | -0.16% | -1.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 18 | 6 | 33.3% | 1 | 12 | 5 | 0.14% | 2.5% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2026-02-13 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-13 09:30:00 | 7723.00 | 7756.47 | 0.00 | ORB-short ORB[7740.00,7820.50] vol=2.2x ATR=25.62 |
| Stop hit — per-position SL triggered | 2026-02-13 09:40:00 | 7748.62 | 7747.52 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2026-02-17 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-17 09:35:00 | 7821.00 | 7778.69 | 0.00 | ORB-long ORB[7686.00,7790.00] vol=2.0x ATR=22.99 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-17 09:50:00 | 7855.48 | 7797.67 | 0.00 | T1 1.5R @ 7855.48 |
| Stop hit — per-position SL triggered | 2026-02-17 10:05:00 | 7821.00 | 7802.24 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2026-02-20 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-20 10:45:00 | 7760.00 | 7721.16 | 0.00 | ORB-long ORB[7646.00,7732.00] vol=1.5x ATR=23.95 |
| Stop hit — per-position SL triggered | 2026-02-20 11:05:00 | 7736.05 | 7725.17 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2026-02-24 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-24 10:00:00 | 7836.50 | 7773.21 | 0.00 | ORB-long ORB[7699.00,7770.50] vol=1.6x ATR=24.80 |
| Stop hit — per-position SL triggered | 2026-02-24 10:10:00 | 7811.70 | 7781.13 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2026-03-05 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-05 10:45:00 | 7556.00 | 7653.33 | 0.00 | ORB-short ORB[7691.50,7745.00] vol=1.6x ATR=26.42 |
| Stop hit — per-position SL triggered | 2026-03-05 11:10:00 | 7582.42 | 7634.04 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2026-03-19 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-19 11:00:00 | 6637.00 | 6678.19 | 0.00 | ORB-short ORB[6642.50,6730.00] vol=1.9x ATR=27.23 |
| Stop hit — per-position SL triggered | 2026-03-19 11:30:00 | 6664.23 | 6674.17 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2026-04-08 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-08 09:40:00 | 6776.00 | 6726.43 | 0.00 | ORB-long ORB[6659.00,6759.50] vol=1.5x ATR=41.09 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-08 10:20:00 | 6837.64 | 6755.78 | 0.00 | T1 1.5R @ 6837.64 |
| Target hit | 2026-04-08 15:20:00 | 6944.50 | 6900.80 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 8 — BUY (started 2026-04-16 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-16 10:30:00 | 7670.00 | 7610.88 | 0.00 | ORB-long ORB[7556.50,7668.00] vol=1.8x ATR=35.95 |
| Stop hit — per-position SL triggered | 2026-04-16 11:40:00 | 7634.05 | 7623.30 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2026-04-17 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-17 09:40:00 | 7811.00 | 7776.17 | 0.00 | ORB-long ORB[7714.00,7798.50] vol=2.1x ATR=30.85 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-17 10:00:00 | 7857.28 | 7796.17 | 0.00 | T1 1.5R @ 7857.28 |
| Stop hit — per-position SL triggered | 2026-04-17 12:30:00 | 7811.00 | 7818.80 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2026-04-23 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-23 09:35:00 | 7715.00 | 7752.92 | 0.00 | ORB-short ORB[7740.00,7804.50] vol=2.1x ATR=24.95 |
| Stop hit — per-position SL triggered | 2026-04-23 09:40:00 | 7739.95 | 7751.27 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2026-05-05 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-05 09:30:00 | 8095.50 | 8039.57 | 0.00 | ORB-long ORB[7961.00,8030.50] vol=2.3x ATR=30.18 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-05 09:35:00 | 8140.76 | 8062.10 | 0.00 | T1 1.5R @ 8140.76 |
| Stop hit — per-position SL triggered | 2026-05-05 09:40:00 | 8095.50 | 8063.62 | 0.00 | SL hit |

### Cycle 12 — SELL (started 2026-05-06 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-06 11:05:00 | 8230.00 | 8257.85 | 0.00 | ORB-short ORB[8239.00,8333.00] vol=2.0x ATR=24.36 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-06 11:30:00 | 8193.47 | 8253.50 | 0.00 | T1 1.5R @ 8193.47 |
| Stop hit — per-position SL triggered | 2026-05-06 12:20:00 | 8230.00 | 8240.08 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2026-05-08 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-08 09:55:00 | 8855.00 | 8805.36 | 0.00 | ORB-long ORB[8736.50,8814.50] vol=1.7x ATR=38.17 |
| Stop hit — per-position SL triggered | 2026-05-08 10:00:00 | 8816.83 | 8806.81 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2026-02-13 09:30:00 | 7723.00 | 2026-02-13 09:40:00 | 7748.62 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2026-02-17 09:35:00 | 7821.00 | 2026-02-17 09:50:00 | 7855.48 | PARTIAL | 0.50 | 0.44% |
| BUY | retest1 | 2026-02-17 09:35:00 | 7821.00 | 2026-02-17 10:05:00 | 7821.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-20 10:45:00 | 7760.00 | 2026-02-20 11:05:00 | 7736.05 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2026-02-24 10:00:00 | 7836.50 | 2026-02-24 10:10:00 | 7811.70 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2026-03-05 10:45:00 | 7556.00 | 2026-03-05 11:10:00 | 7582.42 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2026-03-19 11:00:00 | 6637.00 | 2026-03-19 11:30:00 | 6664.23 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest1 | 2026-04-08 09:40:00 | 6776.00 | 2026-04-08 10:20:00 | 6837.64 | PARTIAL | 0.50 | 0.91% |
| BUY | retest1 | 2026-04-08 09:40:00 | 6776.00 | 2026-04-08 15:20:00 | 6944.50 | TARGET_HIT | 0.50 | 2.49% |
| BUY | retest1 | 2026-04-16 10:30:00 | 7670.00 | 2026-04-16 11:40:00 | 7634.05 | STOP_HIT | 1.00 | -0.47% |
| BUY | retest1 | 2026-04-17 09:40:00 | 7811.00 | 2026-04-17 10:00:00 | 7857.28 | PARTIAL | 0.50 | 0.59% |
| BUY | retest1 | 2026-04-17 09:40:00 | 7811.00 | 2026-04-17 12:30:00 | 7811.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-04-23 09:35:00 | 7715.00 | 2026-04-23 09:40:00 | 7739.95 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2026-05-05 09:30:00 | 8095.50 | 2026-05-05 09:35:00 | 8140.76 | PARTIAL | 0.50 | 0.56% |
| BUY | retest1 | 2026-05-05 09:30:00 | 8095.50 | 2026-05-05 09:40:00 | 8095.50 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-05-06 11:05:00 | 8230.00 | 2026-05-06 11:30:00 | 8193.47 | PARTIAL | 0.50 | 0.44% |
| SELL | retest1 | 2026-05-06 11:05:00 | 8230.00 | 2026-05-06 12:20:00 | 8230.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-05-08 09:55:00 | 8855.00 | 2026-05-08 10:00:00 | 8816.83 | STOP_HIT | 1.00 | -0.43% |
