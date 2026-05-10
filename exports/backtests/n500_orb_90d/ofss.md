# Oracle Financial Services Software Ltd. (OFSS)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 9321.00
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
| ENTRY1 | 12 |
| ENTRY2 | 0 |
| PARTIAL | 3 |
| TARGET_HIT | 1 |
| STOP_HIT | 11 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 15 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 4 / 11
- **Target hits / Stop hits / Partials:** 1 / 11 / 3
- **Avg / median % per leg:** -0.03% / -0.24%
- **Sum % (uncompounded):** -0.46%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 8 | 2 | 25.0% | 0 | 6 | 2 | -0.02% | -0.1% |
| BUY @ 2nd Alert (retest1) | 8 | 2 | 25.0% | 0 | 6 | 2 | -0.02% | -0.1% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 7 | 2 | 28.6% | 1 | 5 | 1 | -0.05% | -0.3% |
| SELL @ 2nd Alert (retest1) | 7 | 2 | 28.6% | 1 | 5 | 1 | -0.05% | -0.3% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 15 | 4 | 26.7% | 1 | 11 | 3 | -0.03% | -0.5% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-10 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-10 09:40:00 | 7363.50 | 7333.76 | 0.00 | ORB-long ORB[7256.00,7349.50] vol=1.7x ATR=17.39 |
| Stop hit — per-position SL triggered | 2026-02-10 09:50:00 | 7346.11 | 7336.65 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2026-02-17 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-17 09:40:00 | 6805.50 | 6767.70 | 0.00 | ORB-long ORB[6672.50,6765.00] vol=1.8x ATR=24.05 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-17 09:55:00 | 6841.58 | 6787.23 | 0.00 | T1 1.5R @ 6841.58 |
| Stop hit — per-position SL triggered | 2026-02-17 10:00:00 | 6805.50 | 6789.63 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2026-02-18 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-18 10:20:00 | 6725.50 | 6777.71 | 0.00 | ORB-short ORB[6785.00,6877.50] vol=1.6x ATR=24.00 |
| Stop hit — per-position SL triggered | 2026-02-18 10:35:00 | 6749.50 | 6768.13 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2026-02-19 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-19 09:45:00 | 6733.50 | 6741.80 | 0.00 | ORB-short ORB[6734.50,6799.50] vol=3.0x ATR=21.09 |
| Stop hit — per-position SL triggered | 2026-02-19 10:05:00 | 6754.59 | 6739.97 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2026-02-20 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-20 09:30:00 | 6658.00 | 6622.81 | 0.00 | ORB-long ORB[6593.00,6650.00] vol=1.6x ATR=23.43 |
| Stop hit — per-position SL triggered | 2026-02-20 10:00:00 | 6634.57 | 6640.68 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2026-02-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-24 11:15:00 | 6521.00 | 6566.94 | 0.00 | ORB-short ORB[6566.50,6646.50] vol=1.5x ATR=18.89 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-24 11:40:00 | 6492.66 | 6557.95 | 0.00 | T1 1.5R @ 6492.66 |
| Target hit | 2026-02-24 15:20:00 | 6462.00 | 6481.72 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 7 — BUY (started 2026-03-09 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-09 09:30:00 | 6675.50 | 6623.97 | 0.00 | ORB-long ORB[6569.50,6651.50] vol=2.4x ATR=28.30 |
| Stop hit — per-position SL triggered | 2026-03-09 09:35:00 | 6647.20 | 6628.26 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2026-03-10 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-10 11:05:00 | 6786.00 | 6757.40 | 0.00 | ORB-long ORB[6724.00,6770.50] vol=3.3x ATR=14.94 |
| Stop hit — per-position SL triggered | 2026-03-10 11:15:00 | 6771.06 | 6759.04 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2026-03-12 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-12 09:30:00 | 6703.50 | 6733.12 | 0.00 | ORB-short ORB[6710.00,6796.00] vol=1.8x ATR=26.01 |
| Stop hit — per-position SL triggered | 2026-03-12 11:00:00 | 6729.51 | 6714.97 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2026-03-13 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-13 09:55:00 | 6587.50 | 6612.78 | 0.00 | ORB-short ORB[6591.00,6686.00] vol=1.9x ATR=21.60 |
| Stop hit — per-position SL triggered | 2026-03-13 10:00:00 | 6609.10 | 6612.55 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2026-03-25 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-25 09:30:00 | 6642.50 | 6606.57 | 0.00 | ORB-long ORB[6554.50,6641.00] vol=1.6x ATR=25.34 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-25 10:00:00 | 6680.50 | 6632.65 | 0.00 | T1 1.5R @ 6680.50 |
| Stop hit — per-position SL triggered | 2026-03-25 11:25:00 | 6642.50 | 6646.68 | 0.00 | SL hit |

### Cycle 12 — SELL (started 2026-05-08 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-08 11:10:00 | 9341.00 | 9493.12 | 0.00 | ORB-short ORB[9472.50,9580.00] vol=2.9x ATR=26.51 |
| Stop hit — per-position SL triggered | 2026-05-08 11:40:00 | 9367.51 | 9465.80 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-02-10 09:40:00 | 7363.50 | 2026-02-10 09:50:00 | 7346.11 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2026-02-17 09:40:00 | 6805.50 | 2026-02-17 09:55:00 | 6841.58 | PARTIAL | 0.50 | 0.53% |
| BUY | retest1 | 2026-02-17 09:40:00 | 6805.50 | 2026-02-17 10:00:00 | 6805.50 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-02-18 10:20:00 | 6725.50 | 2026-02-18 10:35:00 | 6749.50 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2026-02-19 09:45:00 | 6733.50 | 2026-02-19 10:05:00 | 6754.59 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2026-02-20 09:30:00 | 6658.00 | 2026-02-20 10:00:00 | 6634.57 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2026-02-24 11:15:00 | 6521.00 | 2026-02-24 11:40:00 | 6492.66 | PARTIAL | 0.50 | 0.43% |
| SELL | retest1 | 2026-02-24 11:15:00 | 6521.00 | 2026-02-24 15:20:00 | 6462.00 | TARGET_HIT | 0.50 | 0.90% |
| BUY | retest1 | 2026-03-09 09:30:00 | 6675.50 | 2026-03-09 09:35:00 | 6647.20 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest1 | 2026-03-10 11:05:00 | 6786.00 | 2026-03-10 11:15:00 | 6771.06 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2026-03-12 09:30:00 | 6703.50 | 2026-03-12 11:00:00 | 6729.51 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest1 | 2026-03-13 09:55:00 | 6587.50 | 2026-03-13 10:00:00 | 6609.10 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2026-03-25 09:30:00 | 6642.50 | 2026-03-25 10:00:00 | 6680.50 | PARTIAL | 0.50 | 0.57% |
| BUY | retest1 | 2026-03-25 09:30:00 | 6642.50 | 2026-03-25 11:25:00 | 6642.50 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-05-08 11:10:00 | 9341.00 | 2026-05-08 11:40:00 | 9367.51 | STOP_HIT | 1.00 | -0.28% |
