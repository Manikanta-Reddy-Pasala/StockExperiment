# Niva Bupa Health Insurance Company Ltd. (NIVABUPA)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 81.25
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
| PARTIAL | 4 |
| TARGET_HIT | 2 |
| STOP_HIT | 11 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 17 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 6 / 11
- **Target hits / Stop hits / Partials:** 2 / 11 / 4
- **Avg / median % per leg:** 0.14% / -0.21%
- **Sum % (uncompounded):** 2.30%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 11 | 5 | 45.5% | 2 | 6 | 3 | 0.28% | 3.1% |
| BUY @ 2nd Alert (retest1) | 11 | 5 | 45.5% | 2 | 6 | 3 | 0.28% | 3.1% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 6 | 1 | 16.7% | 0 | 5 | 1 | -0.14% | -0.8% |
| SELL @ 2nd Alert (retest1) | 6 | 1 | 16.7% | 0 | 5 | 1 | -0.14% | -0.8% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 17 | 6 | 35.3% | 2 | 11 | 4 | 0.14% | 2.3% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2026-02-12 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-12 10:20:00 | 77.05 | 77.13 | 0.00 | ORB-short ORB[77.11,77.74] vol=1.8x ATR=0.18 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-12 11:25:00 | 76.79 | 77.07 | 0.00 | T1 1.5R @ 76.79 |
| Stop hit — per-position SL triggered | 2026-02-12 11:40:00 | 77.05 | 77.05 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2026-02-18 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-18 09:40:00 | 76.65 | 76.53 | 0.00 | ORB-long ORB[76.25,76.62] vol=2.8x ATR=0.18 |
| Stop hit — per-position SL triggered | 2026-02-18 09:45:00 | 76.47 | 76.52 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2026-02-19 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-19 09:55:00 | 75.88 | 76.16 | 0.00 | ORB-short ORB[76.12,76.67] vol=1.9x ATR=0.16 |
| Stop hit — per-position SL triggered | 2026-02-19 10:00:00 | 76.04 | 76.14 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2026-02-23 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-23 10:05:00 | 76.51 | 76.75 | 0.00 | ORB-short ORB[76.75,77.17] vol=1.9x ATR=0.19 |
| Stop hit — per-position SL triggered | 2026-02-23 10:10:00 | 76.70 | 76.75 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2026-02-25 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-25 09:40:00 | 76.76 | 76.70 | 0.00 | ORB-long ORB[76.32,76.71] vol=5.5x ATR=0.19 |
| Stop hit — per-position SL triggered | 2026-02-25 10:00:00 | 76.57 | 76.71 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2026-03-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-12 10:15:00 | 69.96 | 70.17 | 0.00 | ORB-short ORB[70.00,71.00] vol=1.6x ATR=0.20 |
| Stop hit — per-position SL triggered | 2026-03-12 10:40:00 | 70.16 | 70.12 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2026-03-20 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-20 09:35:00 | 70.84 | 70.75 | 0.00 | ORB-long ORB[70.11,70.74] vol=1.7x ATR=0.34 |
| Stop hit — per-position SL triggered | 2026-03-20 09:45:00 | 70.50 | 70.70 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2026-03-24 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-24 10:50:00 | 67.87 | 68.33 | 0.00 | ORB-short ORB[68.60,69.56] vol=1.6x ATR=0.30 |
| Stop hit — per-position SL triggered | 2026-03-24 11:40:00 | 68.17 | 68.26 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2026-03-25 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-25 10:10:00 | 70.93 | 70.34 | 0.00 | ORB-long ORB[69.33,70.31] vol=1.9x ATR=0.34 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-25 10:15:00 | 71.44 | 70.58 | 0.00 | T1 1.5R @ 71.44 |
| Target hit | 2026-03-25 13:50:00 | 72.31 | 72.35 | 0.00 | Trail-exit close<VWAP |

### Cycle 10 — BUY (started 2026-04-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-09 11:15:00 | 74.36 | 73.59 | 0.00 | ORB-long ORB[73.10,73.75] vol=11.5x ATR=0.32 |
| Stop hit — per-position SL triggered | 2026-04-09 11:45:00 | 74.04 | 73.84 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2026-04-13 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-13 10:00:00 | 73.11 | 72.57 | 0.00 | ORB-long ORB[71.86,72.67] vol=3.1x ATR=0.31 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-13 10:20:00 | 73.57 | 72.65 | 0.00 | T1 1.5R @ 73.57 |
| Target hit | 2026-04-13 15:20:00 | 74.03 | 73.71 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 12 — BUY (started 2026-04-15 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-15 10:35:00 | 76.46 | 75.65 | 0.00 | ORB-long ORB[74.78,75.84] vol=7.6x ATR=0.33 |
| Stop hit — per-position SL triggered | 2026-04-15 10:40:00 | 76.13 | 75.68 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2026-04-21 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-21 09:45:00 | 78.30 | 78.10 | 0.00 | ORB-long ORB[77.49,78.28] vol=2.1x ATR=0.21 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-21 11:10:00 | 78.61 | 78.28 | 0.00 | T1 1.5R @ 78.61 |
| Stop hit — per-position SL triggered | 2026-04-21 12:10:00 | 78.30 | 78.39 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2026-02-12 10:20:00 | 77.05 | 2026-02-12 11:25:00 | 76.79 | PARTIAL | 0.50 | 0.34% |
| SELL | retest1 | 2026-02-12 10:20:00 | 77.05 | 2026-02-12 11:40:00 | 77.05 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-18 09:40:00 | 76.65 | 2026-02-18 09:45:00 | 76.47 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2026-02-19 09:55:00 | 75.88 | 2026-02-19 10:00:00 | 76.04 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2026-02-23 10:05:00 | 76.51 | 2026-02-23 10:10:00 | 76.70 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2026-02-25 09:40:00 | 76.76 | 2026-02-25 10:00:00 | 76.57 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2026-03-12 10:15:00 | 69.96 | 2026-03-12 10:40:00 | 70.16 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2026-03-20 09:35:00 | 70.84 | 2026-03-20 09:45:00 | 70.50 | STOP_HIT | 1.00 | -0.48% |
| SELL | retest1 | 2026-03-24 10:50:00 | 67.87 | 2026-03-24 11:40:00 | 68.17 | STOP_HIT | 1.00 | -0.44% |
| BUY | retest1 | 2026-03-25 10:10:00 | 70.93 | 2026-03-25 10:15:00 | 71.44 | PARTIAL | 0.50 | 0.71% |
| BUY | retest1 | 2026-03-25 10:10:00 | 70.93 | 2026-03-25 13:50:00 | 72.31 | TARGET_HIT | 0.50 | 1.95% |
| BUY | retest1 | 2026-04-09 11:15:00 | 74.36 | 2026-04-09 11:45:00 | 74.04 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest1 | 2026-04-13 10:00:00 | 73.11 | 2026-04-13 10:20:00 | 73.57 | PARTIAL | 0.50 | 0.63% |
| BUY | retest1 | 2026-04-13 10:00:00 | 73.11 | 2026-04-13 15:20:00 | 74.03 | TARGET_HIT | 0.50 | 1.26% |
| BUY | retest1 | 2026-04-15 10:35:00 | 76.46 | 2026-04-15 10:40:00 | 76.13 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest1 | 2026-04-21 09:45:00 | 78.30 | 2026-04-21 11:10:00 | 78.61 | PARTIAL | 0.50 | 0.40% |
| BUY | retest1 | 2026-04-21 09:45:00 | 78.30 | 2026-04-21 12:10:00 | 78.30 | STOP_HIT | 0.50 | 0.00% |
