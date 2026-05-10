# NMDC Ltd. (NMDC)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 88.75
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
| ENTRY1 | 15 |
| ENTRY2 | 0 |
| PARTIAL | 7 |
| TARGET_HIT | 3 |
| STOP_HIT | 12 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 22 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 10 / 12
- **Target hits / Stop hits / Partials:** 3 / 12 / 7
- **Avg / median % per leg:** 0.16% / 0.00%
- **Sum % (uncompounded):** 3.48%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 12 | 7 | 58.3% | 3 | 5 | 4 | 0.25% | 3.0% |
| BUY @ 2nd Alert (retest1) | 12 | 7 | 58.3% | 3 | 5 | 4 | 0.25% | 3.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 10 | 3 | 30.0% | 0 | 7 | 3 | 0.04% | 0.4% |
| SELL @ 2nd Alert (retest1) | 10 | 3 | 30.0% | 0 | 7 | 3 | 0.04% | 0.4% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 22 | 10 | 45.5% | 3 | 12 | 7 | 0.16% | 3.5% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-09 11:15:00 | 85.20 | 84.67 | 0.00 | ORB-long ORB[83.95,85.01] vol=2.0x ATR=0.33 |
| Stop hit — per-position SL triggered | 2026-02-09 11:45:00 | 84.87 | 84.78 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2026-02-11 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-11 11:10:00 | 84.63 | 85.35 | 0.00 | ORB-short ORB[85.10,86.07] vol=2.5x ATR=0.24 |
| Stop hit — per-position SL triggered | 2026-02-11 11:25:00 | 84.87 | 85.32 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2026-02-17 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-17 10:45:00 | 78.92 | 79.46 | 0.00 | ORB-short ORB[79.25,80.39] vol=1.7x ATR=0.23 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-17 11:05:00 | 78.58 | 79.35 | 0.00 | T1 1.5R @ 78.58 |
| Stop hit — per-position SL triggered | 2026-02-17 11:15:00 | 78.92 | 79.33 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2026-02-19 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-19 09:40:00 | 80.20 | 80.54 | 0.00 | ORB-short ORB[80.25,81.05] vol=3.5x ATR=0.21 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-19 10:05:00 | 79.88 | 80.46 | 0.00 | T1 1.5R @ 79.88 |
| Stop hit — per-position SL triggered | 2026-02-19 11:45:00 | 80.20 | 80.22 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2026-03-11 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-11 09:30:00 | 80.98 | 80.62 | 0.00 | ORB-long ORB[79.95,80.87] vol=1.7x ATR=0.26 |
| Stop hit — per-position SL triggered | 2026-03-11 09:50:00 | 80.72 | 80.69 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2026-03-16 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-16 10:35:00 | 77.79 | 78.12 | 0.00 | ORB-short ORB[77.81,78.96] vol=2.4x ATR=0.34 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-16 10:55:00 | 77.28 | 78.05 | 0.00 | T1 1.5R @ 77.28 |
| Stop hit — per-position SL triggered | 2026-03-16 11:15:00 | 77.79 | 78.01 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2026-04-13 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-13 10:50:00 | 85.22 | 84.01 | 0.00 | ORB-long ORB[83.12,84.20] vol=2.0x ATR=0.32 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-13 11:25:00 | 85.70 | 84.32 | 0.00 | T1 1.5R @ 85.70 |
| Target hit | 2026-04-13 15:20:00 | 85.76 | 85.27 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 8 — BUY (started 2026-04-16 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-16 09:35:00 | 88.39 | 88.07 | 0.00 | ORB-long ORB[87.53,88.38] vol=1.6x ATR=0.28 |
| Stop hit — per-position SL triggered | 2026-04-16 09:55:00 | 88.11 | 88.16 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2026-04-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-17 11:15:00 | 88.45 | 87.96 | 0.00 | ORB-long ORB[87.26,88.07] vol=1.6x ATR=0.22 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-17 11:20:00 | 88.78 | 88.13 | 0.00 | T1 1.5R @ 88.78 |
| Target hit | 2026-04-17 15:20:00 | 89.83 | 89.10 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 10 — BUY (started 2026-04-22 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-22 09:30:00 | 89.48 | 88.95 | 0.00 | ORB-long ORB[88.25,89.31] vol=2.1x ATR=0.29 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-22 09:35:00 | 89.91 | 89.07 | 0.00 | T1 1.5R @ 89.91 |
| Stop hit — per-position SL triggered | 2026-04-22 10:20:00 | 89.48 | 89.34 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2026-04-23 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-23 09:45:00 | 86.95 | 87.58 | 0.00 | ORB-short ORB[87.42,88.54] vol=1.6x ATR=0.28 |
| Stop hit — per-position SL triggered | 2026-04-23 10:20:00 | 87.23 | 87.22 | 0.00 | SL hit |

### Cycle 12 — SELL (started 2026-04-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-24 11:15:00 | 87.07 | 87.21 | 0.00 | ORB-short ORB[87.20,88.09] vol=1.9x ATR=0.20 |
| Stop hit — per-position SL triggered | 2026-04-24 11:20:00 | 87.27 | 87.21 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2026-04-27 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-27 11:10:00 | 91.10 | 90.65 | 0.00 | ORB-long ORB[89.81,90.70] vol=1.6x ATR=0.22 |
| Stop hit — per-position SL triggered | 2026-04-27 11:30:00 | 90.88 | 90.68 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2026-05-07 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-07 09:35:00 | 90.08 | 89.71 | 0.00 | ORB-long ORB[89.14,89.85] vol=1.8x ATR=0.24 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-07 09:40:00 | 90.44 | 89.91 | 0.00 | T1 1.5R @ 90.44 |
| Target hit | 2026-05-07 10:45:00 | 90.34 | 90.36 | 0.00 | Trail-exit close<VWAP |

### Cycle 15 — SELL (started 2026-05-08 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-08 10:50:00 | 89.22 | 89.56 | 0.00 | ORB-short ORB[89.52,90.50] vol=1.8x ATR=0.19 |
| Stop hit — per-position SL triggered | 2026-05-08 10:55:00 | 89.41 | 89.55 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-02-09 11:15:00 | 85.20 | 2026-02-09 11:45:00 | 84.87 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest1 | 2026-02-11 11:10:00 | 84.63 | 2026-02-11 11:25:00 | 84.87 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2026-02-17 10:45:00 | 78.92 | 2026-02-17 11:05:00 | 78.58 | PARTIAL | 0.50 | 0.43% |
| SELL | retest1 | 2026-02-17 10:45:00 | 78.92 | 2026-02-17 11:15:00 | 78.92 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-02-19 09:40:00 | 80.20 | 2026-02-19 10:05:00 | 79.88 | PARTIAL | 0.50 | 0.40% |
| SELL | retest1 | 2026-02-19 09:40:00 | 80.20 | 2026-02-19 11:45:00 | 80.20 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-03-11 09:30:00 | 80.98 | 2026-03-11 09:50:00 | 80.72 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2026-03-16 10:35:00 | 77.79 | 2026-03-16 10:55:00 | 77.28 | PARTIAL | 0.50 | 0.66% |
| SELL | retest1 | 2026-03-16 10:35:00 | 77.79 | 2026-03-16 11:15:00 | 77.79 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-13 10:50:00 | 85.22 | 2026-04-13 11:25:00 | 85.70 | PARTIAL | 0.50 | 0.56% |
| BUY | retest1 | 2026-04-13 10:50:00 | 85.22 | 2026-04-13 15:20:00 | 85.76 | TARGET_HIT | 0.50 | 0.63% |
| BUY | retest1 | 2026-04-16 09:35:00 | 88.39 | 2026-04-16 09:55:00 | 88.11 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2026-04-17 11:15:00 | 88.45 | 2026-04-17 11:20:00 | 88.78 | PARTIAL | 0.50 | 0.37% |
| BUY | retest1 | 2026-04-17 11:15:00 | 88.45 | 2026-04-17 15:20:00 | 89.83 | TARGET_HIT | 0.50 | 1.56% |
| BUY | retest1 | 2026-04-22 09:30:00 | 89.48 | 2026-04-22 09:35:00 | 89.91 | PARTIAL | 0.50 | 0.48% |
| BUY | retest1 | 2026-04-22 09:30:00 | 89.48 | 2026-04-22 10:20:00 | 89.48 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-04-23 09:45:00 | 86.95 | 2026-04-23 10:20:00 | 87.23 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2026-04-24 11:15:00 | 87.07 | 2026-04-24 11:20:00 | 87.27 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2026-04-27 11:10:00 | 91.10 | 2026-04-27 11:30:00 | 90.88 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2026-05-07 09:35:00 | 90.08 | 2026-05-07 09:40:00 | 90.44 | PARTIAL | 0.50 | 0.40% |
| BUY | retest1 | 2026-05-07 09:35:00 | 90.08 | 2026-05-07 10:45:00 | 90.34 | TARGET_HIT | 0.50 | 0.29% |
| SELL | retest1 | 2026-05-08 10:50:00 | 89.22 | 2026-05-08 10:55:00 | 89.41 | STOP_HIT | 1.00 | -0.21% |
