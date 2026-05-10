# NHPC Ltd. (NHPC)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 80.70
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
| ENTRY1 | 19 |
| ENTRY2 | 0 |
| PARTIAL | 7 |
| TARGET_HIT | 3 |
| STOP_HIT | 16 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 26 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 10 / 16
- **Target hits / Stop hits / Partials:** 3 / 16 / 7
- **Avg / median % per leg:** 0.14% / 0.00%
- **Sum % (uncompounded):** 3.59%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 11 | 4 | 36.4% | 1 | 7 | 3 | 0.08% | 0.9% |
| BUY @ 2nd Alert (retest1) | 11 | 4 | 36.4% | 1 | 7 | 3 | 0.08% | 0.9% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 15 | 6 | 40.0% | 2 | 9 | 4 | 0.18% | 2.7% |
| SELL @ 2nd Alert (retest1) | 15 | 6 | 40.0% | 2 | 9 | 4 | 0.18% | 2.7% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 26 | 10 | 38.5% | 3 | 16 | 7 | 0.14% | 3.6% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2026-02-10 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-10 09:55:00 | 77.00 | 77.35 | 0.00 | ORB-short ORB[77.16,78.14] vol=1.6x ATR=0.23 |
| Stop hit — per-position SL triggered | 2026-02-10 10:25:00 | 77.23 | 77.29 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2026-02-16 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-16 11:00:00 | 76.20 | 75.49 | 0.00 | ORB-long ORB[74.65,75.50] vol=2.7x ATR=0.22 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-16 11:20:00 | 76.53 | 75.61 | 0.00 | T1 1.5R @ 76.53 |
| Target hit | 2026-02-16 15:20:00 | 76.91 | 76.48 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 3 — SELL (started 2026-02-18 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-18 10:20:00 | 76.01 | 76.41 | 0.00 | ORB-short ORB[76.41,76.88] vol=1.5x ATR=0.15 |
| Stop hit — per-position SL triggered | 2026-02-18 10:40:00 | 76.16 | 76.38 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2026-02-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-19 10:15:00 | 75.91 | 76.06 | 0.00 | ORB-short ORB[75.92,76.73] vol=1.8x ATR=0.16 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-19 11:45:00 | 75.67 | 75.96 | 0.00 | T1 1.5R @ 75.67 |
| Target hit | 2026-02-19 15:20:00 | 74.22 | 75.26 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 5 — SELL (started 2026-02-23 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-23 09:55:00 | 73.87 | 74.47 | 0.00 | ORB-short ORB[74.47,75.14] vol=1.8x ATR=0.22 |
| Stop hit — per-position SL triggered | 2026-02-23 10:05:00 | 74.09 | 74.41 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2026-02-24 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-24 10:40:00 | 74.69 | 74.15 | 0.00 | ORB-long ORB[73.70,74.19] vol=2.2x ATR=0.15 |
| Stop hit — per-position SL triggered | 2026-02-24 10:50:00 | 74.54 | 74.18 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2026-02-26 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-26 10:00:00 | 75.42 | 75.64 | 0.00 | ORB-short ORB[75.47,75.89] vol=3.5x ATR=0.19 |
| Stop hit — per-position SL triggered | 2026-02-26 10:15:00 | 75.61 | 75.65 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2026-03-06 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-06 10:40:00 | 74.98 | 74.62 | 0.00 | ORB-long ORB[73.65,74.75] vol=1.8x ATR=0.21 |
| Stop hit — per-position SL triggered | 2026-03-06 10:45:00 | 74.77 | 74.65 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2026-03-12 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-12 09:50:00 | 73.67 | 73.32 | 0.00 | ORB-long ORB[72.76,73.49] vol=1.7x ATR=0.25 |
| Stop hit — per-position SL triggered | 2026-03-12 09:55:00 | 73.42 | 73.33 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2026-03-16 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-16 10:50:00 | 74.67 | 75.18 | 0.00 | ORB-short ORB[74.82,75.92] vol=1.8x ATR=0.31 |
| Stop hit — per-position SL triggered | 2026-03-16 11:05:00 | 74.98 | 75.14 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2026-03-20 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-20 09:30:00 | 77.78 | 77.34 | 0.00 | ORB-long ORB[76.60,77.62] vol=1.6x ATR=0.24 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-20 09:35:00 | 78.15 | 77.59 | 0.00 | T1 1.5R @ 78.15 |
| Stop hit — per-position SL triggered | 2026-03-20 10:00:00 | 77.78 | 77.73 | 0.00 | SL hit |

### Cycle 12 — SELL (started 2026-04-09 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-09 09:30:00 | 77.46 | 77.86 | 0.00 | ORB-short ORB[77.50,78.54] vol=2.0x ATR=0.30 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-09 09:40:00 | 77.01 | 77.78 | 0.00 | T1 1.5R @ 77.01 |
| Stop hit — per-position SL triggered | 2026-04-09 10:55:00 | 77.46 | 77.59 | 0.00 | SL hit |

### Cycle 13 — SELL (started 2026-04-10 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-10 10:55:00 | 77.20 | 77.79 | 0.00 | ORB-short ORB[77.34,78.28] vol=2.2x ATR=0.20 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-10 11:05:00 | 76.89 | 77.69 | 0.00 | T1 1.5R @ 76.89 |
| Stop hit — per-position SL triggered | 2026-04-10 11:40:00 | 77.20 | 77.62 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2026-04-21 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-21 09:35:00 | 84.38 | 84.03 | 0.00 | ORB-long ORB[83.04,84.28] vol=1.9x ATR=0.25 |
| Stop hit — per-position SL triggered | 2026-04-21 10:00:00 | 84.13 | 84.07 | 0.00 | SL hit |

### Cycle 15 — BUY (started 2026-04-22 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-22 10:30:00 | 83.05 | 82.93 | 0.00 | ORB-long ORB[82.35,82.96] vol=4.8x ATR=0.22 |
| Stop hit — per-position SL triggered | 2026-04-22 10:55:00 | 82.83 | 82.94 | 0.00 | SL hit |

### Cycle 16 — BUY (started 2026-04-28 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-28 09:35:00 | 84.00 | 83.69 | 0.00 | ORB-long ORB[83.15,83.85] vol=3.8x ATR=0.27 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-28 10:15:00 | 84.40 | 83.88 | 0.00 | T1 1.5R @ 84.40 |
| Stop hit — per-position SL triggered | 2026-04-28 10:25:00 | 84.00 | 83.89 | 0.00 | SL hit |

### Cycle 17 — SELL (started 2026-04-30 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-30 10:30:00 | 81.78 | 82.64 | 0.00 | ORB-short ORB[82.50,83.51] vol=2.0x ATR=0.32 |
| Stop hit — per-position SL triggered | 2026-04-30 11:25:00 | 82.10 | 82.42 | 0.00 | SL hit |

### Cycle 18 — SELL (started 2026-05-06 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-06 11:00:00 | 82.56 | 82.96 | 0.00 | ORB-short ORB[82.88,83.77] vol=1.5x ATR=0.15 |
| Stop hit — per-position SL triggered | 2026-05-06 11:05:00 | 82.71 | 82.95 | 0.00 | SL hit |

### Cycle 19 — SELL (started 2026-05-07 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-07 11:05:00 | 82.25 | 82.88 | 0.00 | ORB-short ORB[83.25,84.08] vol=4.9x ATR=0.20 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-07 11:30:00 | 81.95 | 82.69 | 0.00 | T1 1.5R @ 81.95 |
| Target hit | 2026-05-07 15:20:00 | 81.57 | 82.19 | 0.00 | EOD square-off @ 15:20:00 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2026-02-10 09:55:00 | 77.00 | 2026-02-10 10:25:00 | 77.23 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2026-02-16 11:00:00 | 76.20 | 2026-02-16 11:20:00 | 76.53 | PARTIAL | 0.50 | 0.43% |
| BUY | retest1 | 2026-02-16 11:00:00 | 76.20 | 2026-02-16 15:20:00 | 76.91 | TARGET_HIT | 0.50 | 0.93% |
| SELL | retest1 | 2026-02-18 10:20:00 | 76.01 | 2026-02-18 10:40:00 | 76.16 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest1 | 2026-02-19 10:15:00 | 75.91 | 2026-02-19 11:45:00 | 75.67 | PARTIAL | 0.50 | 0.31% |
| SELL | retest1 | 2026-02-19 10:15:00 | 75.91 | 2026-02-19 15:20:00 | 74.22 | TARGET_HIT | 0.50 | 2.23% |
| SELL | retest1 | 2026-02-23 09:55:00 | 73.87 | 2026-02-23 10:05:00 | 74.09 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2026-02-24 10:40:00 | 74.69 | 2026-02-24 10:50:00 | 74.54 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2026-02-26 10:00:00 | 75.42 | 2026-02-26 10:15:00 | 75.61 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2026-03-06 10:40:00 | 74.98 | 2026-03-06 10:45:00 | 74.77 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2026-03-12 09:50:00 | 73.67 | 2026-03-12 09:55:00 | 73.42 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2026-03-16 10:50:00 | 74.67 | 2026-03-16 11:05:00 | 74.98 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest1 | 2026-03-20 09:30:00 | 77.78 | 2026-03-20 09:35:00 | 78.15 | PARTIAL | 0.50 | 0.47% |
| BUY | retest1 | 2026-03-20 09:30:00 | 77.78 | 2026-03-20 10:00:00 | 77.78 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-04-09 09:30:00 | 77.46 | 2026-04-09 09:40:00 | 77.01 | PARTIAL | 0.50 | 0.59% |
| SELL | retest1 | 2026-04-09 09:30:00 | 77.46 | 2026-04-09 10:55:00 | 77.46 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-04-10 10:55:00 | 77.20 | 2026-04-10 11:05:00 | 76.89 | PARTIAL | 0.50 | 0.40% |
| SELL | retest1 | 2026-04-10 10:55:00 | 77.20 | 2026-04-10 11:40:00 | 77.20 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-21 09:35:00 | 84.38 | 2026-04-21 10:00:00 | 84.13 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2026-04-22 10:30:00 | 83.05 | 2026-04-22 10:55:00 | 82.83 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2026-04-28 09:35:00 | 84.00 | 2026-04-28 10:15:00 | 84.40 | PARTIAL | 0.50 | 0.48% |
| BUY | retest1 | 2026-04-28 09:35:00 | 84.00 | 2026-04-28 10:25:00 | 84.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-04-30 10:30:00 | 81.78 | 2026-04-30 11:25:00 | 82.10 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest1 | 2026-05-06 11:00:00 | 82.56 | 2026-05-06 11:05:00 | 82.71 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest1 | 2026-05-07 11:05:00 | 82.25 | 2026-05-07 11:30:00 | 81.95 | PARTIAL | 0.50 | 0.36% |
| SELL | retest1 | 2026-05-07 11:05:00 | 82.25 | 2026-05-07 15:20:00 | 81.57 | TARGET_HIT | 0.50 | 0.83% |
