# Capri Global Capital Ltd. (CGCL)

## Backtest Summary

- **Window:** 2024-02-06 09:15:00 → 2026-05-08 15:25:00 (37021 bars)
- **Last close:** 197.75
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
| TARGET_HIT | 4 |
| STOP_HIT | 15 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 26 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 11 / 15
- **Target hits / Stop hits / Partials:** 4 / 15 / 7
- **Avg / median % per leg:** 0.70% / 0.00%
- **Sum % (uncompounded):** 18.20%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 8 | 3 | 37.5% | 1 | 5 | 2 | 1.80% | 14.4% |
| BUY @ 2nd Alert (retest1) | 8 | 3 | 37.5% | 1 | 5 | 2 | 1.80% | 14.4% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 18 | 8 | 44.4% | 3 | 10 | 5 | 0.21% | 3.8% |
| SELL @ 2nd Alert (retest1) | 18 | 8 | 44.4% | 3 | 10 | 5 | 0.21% | 3.8% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 26 | 11 | 42.3% | 4 | 15 | 7 | 0.70% | 18.2% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-02-08 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-08 10:55:00 | 230.30 | 231.04 | 0.00 | ORB-short ORB[230.36,231.90] vol=5.2x ATR=0.46 |
| Stop hit — per-position SL triggered | 2024-02-08 11:30:00 | 230.76 | 230.93 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2024-02-09 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-09 10:50:00 | 230.78 | 231.25 | 0.00 | ORB-short ORB[231.00,232.91] vol=3.9x ATR=1.34 |
| Stop hit — per-position SL triggered | 2024-02-09 11:40:00 | 232.12 | 231.26 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2024-02-12 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-12 10:30:00 | 228.24 | 229.51 | 0.00 | ORB-short ORB[230.20,233.00] vol=1.7x ATR=0.97 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-12 12:05:00 | 226.78 | 228.62 | 0.00 | T1 1.5R @ 226.78 |
| Stop hit — per-position SL triggered | 2024-02-12 13:50:00 | 228.24 | 228.50 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2024-02-19 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-19 11:00:00 | 239.51 | 240.45 | 0.00 | ORB-short ORB[239.75,241.65] vol=1.9x ATR=0.71 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-19 13:15:00 | 238.44 | 240.12 | 0.00 | T1 1.5R @ 238.44 |
| Target hit | 2024-02-19 15:20:00 | 238.44 | 239.81 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 5 — BUY (started 2024-02-21 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-21 09:50:00 | 241.39 | 239.61 | 0.00 | ORB-long ORB[238.24,240.49] vol=5.6x ATR=0.70 |
| Stop hit — per-position SL triggered | 2024-02-21 09:55:00 | 240.69 | 239.70 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2024-02-23 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-23 11:00:00 | 239.45 | 240.81 | 0.00 | ORB-short ORB[239.64,242.68] vol=1.6x ATR=0.50 |
| Stop hit — per-position SL triggered | 2024-02-23 11:05:00 | 239.95 | 240.80 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2024-02-28 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-28 09:45:00 | 247.90 | 246.11 | 0.00 | ORB-long ORB[244.21,247.88] vol=1.6x ATR=1.05 |
| Stop hit — per-position SL triggered | 2024-02-28 09:55:00 | 246.85 | 246.17 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2024-03-01 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-03-01 09:30:00 | 242.51 | 244.08 | 0.00 | ORB-short ORB[243.23,245.99] vol=2.5x ATR=1.03 |
| Stop hit — per-position SL triggered | 2024-03-01 09:35:00 | 243.54 | 243.99 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2024-03-02 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-02 09:35:00 | 252.00 | 250.97 | 0.00 | ORB-long ORB[249.36,251.50] vol=1.9x ATR=1.04 |
| Stop hit — per-position SL triggered | 2024-03-02 11:35:00 | 250.96 | 251.19 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2024-03-21 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-03-21 10:45:00 | 205.00 | 207.77 | 0.00 | ORB-short ORB[207.15,209.75] vol=2.0x ATR=1.28 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-21 11:10:00 | 203.09 | 205.57 | 0.00 | T1 1.5R @ 203.09 |
| Target hit | 2024-03-21 13:00:00 | 202.90 | 202.46 | 0.00 | Trail-exit close>VWAP |

### Cycle 11 — SELL (started 2024-03-26 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-03-26 09:30:00 | 201.05 | 202.40 | 0.00 | ORB-short ORB[202.05,203.65] vol=3.3x ATR=0.93 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-26 09:35:00 | 199.66 | 202.08 | 0.00 | T1 1.5R @ 199.66 |
| Stop hit — per-position SL triggered | 2024-03-26 09:45:00 | 201.05 | 201.32 | 0.00 | SL hit |

### Cycle 12 — SELL (started 2024-03-27 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-03-27 10:50:00 | 200.50 | 201.82 | 0.00 | ORB-short ORB[202.30,203.95] vol=2.0x ATR=0.68 |
| Stop hit — per-position SL triggered | 2024-03-27 10:55:00 | 201.18 | 201.80 | 0.00 | SL hit |

### Cycle 13 — SELL (started 2024-04-01 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-04-01 10:20:00 | 201.00 | 201.58 | 0.00 | ORB-short ORB[202.20,204.50] vol=2.1x ATR=1.36 |
| Stop hit — per-position SL triggered | 2024-04-01 12:45:00 | 202.36 | 201.51 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2024-04-02 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-02 10:20:00 | 204.50 | 203.56 | 0.00 | ORB-long ORB[201.55,203.80] vol=1.8x ATR=0.64 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-02 10:25:00 | 205.47 | 203.98 | 0.00 | T1 1.5R @ 205.47 |
| Stop hit — per-position SL triggered | 2024-04-02 10:40:00 | 204.50 | 204.17 | 0.00 | SL hit |

### Cycle 15 — BUY (started 2024-04-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-03 10:15:00 | 207.50 | 206.07 | 0.00 | ORB-long ORB[204.35,206.15] vol=4.3x ATR=0.90 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-03 10:25:00 | 208.84 | 207.00 | 0.00 | T1 1.5R @ 208.84 |
| Target hit | 2024-04-03 15:20:00 | 238.30 | 233.93 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 16 — SELL (started 2024-04-08 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-04-08 10:55:00 | 220.70 | 222.08 | 0.00 | ORB-short ORB[221.65,224.15] vol=1.9x ATR=0.75 |
| Stop hit — per-position SL triggered | 2024-04-08 11:10:00 | 221.45 | 221.96 | 0.00 | SL hit |

### Cycle 17 — SELL (started 2024-04-18 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-04-18 10:55:00 | 234.75 | 237.82 | 0.00 | ORB-short ORB[238.85,241.20] vol=2.9x ATR=1.02 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-18 12:05:00 | 233.22 | 236.89 | 0.00 | T1 1.5R @ 233.22 |
| Target hit | 2024-04-18 15:20:00 | 229.80 | 234.58 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 18 — BUY (started 2024-04-25 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-25 09:55:00 | 229.30 | 228.25 | 0.00 | ORB-long ORB[226.10,227.90] vol=4.8x ATR=1.04 |
| Stop hit — per-position SL triggered | 2024-04-25 10:00:00 | 228.26 | 228.30 | 0.00 | SL hit |

### Cycle 19 — SELL (started 2024-05-03 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-03 10:25:00 | 221.55 | 223.05 | 0.00 | ORB-short ORB[223.00,224.45] vol=2.5x ATR=0.80 |
| Stop hit — per-position SL triggered | 2024-05-03 12:45:00 | 222.35 | 222.43 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2024-02-08 10:55:00 | 230.30 | 2024-02-08 11:30:00 | 230.76 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2024-02-09 10:50:00 | 230.78 | 2024-02-09 11:40:00 | 232.12 | STOP_HIT | 1.00 | -0.58% |
| SELL | retest1 | 2024-02-12 10:30:00 | 228.24 | 2024-02-12 12:05:00 | 226.78 | PARTIAL | 0.50 | 0.64% |
| SELL | retest1 | 2024-02-12 10:30:00 | 228.24 | 2024-02-12 13:50:00 | 228.24 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-02-19 11:00:00 | 239.51 | 2024-02-19 13:15:00 | 238.44 | PARTIAL | 0.50 | 0.45% |
| SELL | retest1 | 2024-02-19 11:00:00 | 239.51 | 2024-02-19 15:20:00 | 238.44 | TARGET_HIT | 0.50 | 0.45% |
| BUY | retest1 | 2024-02-21 09:50:00 | 241.39 | 2024-02-21 09:55:00 | 240.69 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2024-02-23 11:00:00 | 239.45 | 2024-02-23 11:05:00 | 239.95 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2024-02-28 09:45:00 | 247.90 | 2024-02-28 09:55:00 | 246.85 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest1 | 2024-03-01 09:30:00 | 242.51 | 2024-03-01 09:35:00 | 243.54 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest1 | 2024-03-02 09:35:00 | 252.00 | 2024-03-02 11:35:00 | 250.96 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest1 | 2024-03-21 10:45:00 | 205.00 | 2024-03-21 11:10:00 | 203.09 | PARTIAL | 0.50 | 0.93% |
| SELL | retest1 | 2024-03-21 10:45:00 | 205.00 | 2024-03-21 13:00:00 | 202.90 | TARGET_HIT | 0.50 | 1.02% |
| SELL | retest1 | 2024-03-26 09:30:00 | 201.05 | 2024-03-26 09:35:00 | 199.66 | PARTIAL | 0.50 | 0.69% |
| SELL | retest1 | 2024-03-26 09:30:00 | 201.05 | 2024-03-26 09:45:00 | 201.05 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-03-27 10:50:00 | 200.50 | 2024-03-27 10:55:00 | 201.18 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2024-04-01 10:20:00 | 201.00 | 2024-04-01 12:45:00 | 202.36 | STOP_HIT | 1.00 | -0.68% |
| BUY | retest1 | 2024-04-02 10:20:00 | 204.50 | 2024-04-02 10:25:00 | 205.47 | PARTIAL | 0.50 | 0.47% |
| BUY | retest1 | 2024-04-02 10:20:00 | 204.50 | 2024-04-02 10:40:00 | 204.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-04-03 10:15:00 | 207.50 | 2024-04-03 10:25:00 | 208.84 | PARTIAL | 0.50 | 0.65% |
| BUY | retest1 | 2024-04-03 10:15:00 | 207.50 | 2024-04-03 15:20:00 | 238.30 | TARGET_HIT | 0.50 | 14.84% |
| SELL | retest1 | 2024-04-08 10:55:00 | 220.70 | 2024-04-08 11:10:00 | 221.45 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2024-04-18 10:55:00 | 234.75 | 2024-04-18 12:05:00 | 233.22 | PARTIAL | 0.50 | 0.65% |
| SELL | retest1 | 2024-04-18 10:55:00 | 234.75 | 2024-04-18 15:20:00 | 229.80 | TARGET_HIT | 0.50 | 2.11% |
| BUY | retest1 | 2024-04-25 09:55:00 | 229.30 | 2024-04-25 10:00:00 | 228.26 | STOP_HIT | 1.00 | -0.45% |
| SELL | retest1 | 2024-05-03 10:25:00 | 221.55 | 2024-05-03 12:45:00 | 222.35 | STOP_HIT | 1.00 | -0.36% |
