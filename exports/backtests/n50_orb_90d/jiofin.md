# JIOFIN (JIOFIN)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 249.01
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
| PARTIAL | 8 |
| TARGET_HIT | 3 |
| STOP_HIT | 16 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 27 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 11 / 16
- **Target hits / Stop hits / Partials:** 3 / 16 / 8
- **Avg / median % per leg:** 0.05% / 0.00%
- **Sum % (uncompounded):** 1.24%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 13 | 6 | 46.2% | 2 | 7 | 4 | 0.06% | 0.7% |
| BUY @ 2nd Alert (retest1) | 13 | 6 | 46.2% | 2 | 7 | 4 | 0.06% | 0.7% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 14 | 5 | 35.7% | 1 | 9 | 4 | 0.04% | 0.5% |
| SELL @ 2nd Alert (retest1) | 14 | 5 | 35.7% | 1 | 9 | 4 | 0.04% | 0.5% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 27 | 11 | 40.7% | 3 | 16 | 8 | 0.05% | 1.2% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-10 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-10 09:30:00 | 272.40 | 271.78 | 0.00 | ORB-long ORB[269.00,271.90] vol=5.0x ATR=0.54 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-10 09:35:00 | 273.21 | 272.17 | 0.00 | T1 1.5R @ 273.21 |
| Target hit | 2026-02-10 10:40:00 | 272.60 | 272.97 | 0.00 | Trail-exit close<VWAP |

### Cycle 2 — SELL (started 2026-02-16 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-16 09:40:00 | 260.15 | 261.28 | 0.00 | ORB-short ORB[260.50,262.50] vol=1.5x ATR=0.70 |
| Stop hit — per-position SL triggered | 2026-02-16 09:55:00 | 260.85 | 260.97 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2026-02-17 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-17 10:20:00 | 262.95 | 261.82 | 0.00 | ORB-long ORB[261.25,262.40] vol=1.6x ATR=0.50 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-17 10:25:00 | 263.70 | 262.16 | 0.00 | T1 1.5R @ 263.70 |
| Stop hit — per-position SL triggered | 2026-02-17 10:45:00 | 262.95 | 262.40 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2026-02-19 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-19 11:05:00 | 261.20 | 262.75 | 0.00 | ORB-short ORB[263.15,265.55] vol=1.7x ATR=0.50 |
| Stop hit — per-position SL triggered | 2026-02-19 11:15:00 | 261.70 | 262.70 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2026-02-20 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-20 10:55:00 | 260.25 | 259.62 | 0.00 | ORB-long ORB[257.55,259.80] vol=1.6x ATR=0.53 |
| Stop hit — per-position SL triggered | 2026-02-20 12:35:00 | 259.72 | 260.07 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2026-02-23 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-23 10:00:00 | 258.20 | 259.28 | 0.00 | ORB-short ORB[259.10,260.40] vol=1.6x ATR=0.51 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-23 10:20:00 | 257.44 | 258.98 | 0.00 | T1 1.5R @ 257.44 |
| Stop hit — per-position SL triggered | 2026-02-23 11:45:00 | 258.20 | 258.32 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2026-02-24 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-24 09:30:00 | 255.25 | 255.91 | 0.00 | ORB-short ORB[255.35,257.20] vol=3.0x ATR=0.53 |
| Stop hit — per-position SL triggered | 2026-02-24 11:25:00 | 255.78 | 255.55 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2026-03-11 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-11 10:40:00 | 241.40 | 239.35 | 0.00 | ORB-long ORB[237.50,240.20] vol=1.9x ATR=0.78 |
| Stop hit — per-position SL triggered | 2026-03-11 10:45:00 | 240.62 | 239.45 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2026-03-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-27 11:15:00 | 232.80 | 234.09 | 0.00 | ORB-short ORB[233.35,236.50] vol=1.8x ATR=0.63 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-27 11:30:00 | 231.85 | 233.84 | 0.00 | T1 1.5R @ 231.85 |
| Stop hit — per-position SL triggered | 2026-03-27 11:35:00 | 232.80 | 233.81 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2026-04-16 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-16 11:00:00 | 242.17 | 243.16 | 0.00 | ORB-short ORB[242.70,245.00] vol=2.3x ATR=0.50 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-16 11:45:00 | 241.41 | 242.80 | 0.00 | T1 1.5R @ 241.41 |
| Target hit | 2026-04-16 15:20:00 | 241.40 | 241.54 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 11 — BUY (started 2026-04-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-17 11:15:00 | 242.69 | 242.39 | 0.00 | ORB-long ORB[240.64,242.23] vol=5.0x ATR=0.55 |
| Stop hit — per-position SL triggered | 2026-04-17 12:00:00 | 242.14 | 242.43 | 0.00 | SL hit |

### Cycle 12 — SELL (started 2026-04-21 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-21 10:20:00 | 235.00 | 236.05 | 0.00 | ORB-short ORB[235.64,237.60] vol=1.6x ATR=0.47 |
| Stop hit — per-position SL triggered | 2026-04-21 10:40:00 | 235.47 | 235.85 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2026-04-23 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-23 09:45:00 | 241.80 | 240.13 | 0.00 | ORB-long ORB[238.54,240.13] vol=4.0x ATR=0.76 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-23 09:50:00 | 242.94 | 241.61 | 0.00 | T1 1.5R @ 242.94 |
| Target hit | 2026-04-23 11:10:00 | 243.40 | 243.49 | 0.00 | Trail-exit close<VWAP |

### Cycle 14 — BUY (started 2026-04-27 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-27 09:55:00 | 250.34 | 248.57 | 0.00 | ORB-long ORB[247.02,249.28] vol=1.7x ATR=0.97 |
| Stop hit — per-position SL triggered | 2026-04-27 10:05:00 | 249.37 | 248.76 | 0.00 | SL hit |

### Cycle 15 — BUY (started 2026-04-29 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-29 10:10:00 | 254.64 | 253.03 | 0.00 | ORB-long ORB[251.21,254.46] vol=1.7x ATR=0.86 |
| Stop hit — per-position SL triggered | 2026-04-29 10:15:00 | 253.78 | 253.11 | 0.00 | SL hit |

### Cycle 16 — BUY (started 2026-05-04 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-04 11:10:00 | 251.24 | 249.78 | 0.00 | ORB-long ORB[247.87,251.00] vol=2.9x ATR=0.68 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-04 11:15:00 | 252.27 | 250.03 | 0.00 | T1 1.5R @ 252.27 |
| Stop hit — per-position SL triggered | 2026-05-04 12:10:00 | 251.24 | 250.66 | 0.00 | SL hit |

### Cycle 17 — SELL (started 2026-05-05 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-05 10:25:00 | 249.00 | 251.11 | 0.00 | ORB-short ORB[250.50,252.34] vol=2.4x ATR=0.88 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-05 10:55:00 | 247.67 | 250.36 | 0.00 | T1 1.5R @ 247.67 |
| Stop hit — per-position SL triggered | 2026-05-05 11:25:00 | 249.00 | 250.06 | 0.00 | SL hit |

### Cycle 18 — SELL (started 2026-05-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-06 11:15:00 | 250.54 | 251.32 | 0.00 | ORB-short ORB[250.88,252.25] vol=1.6x ATR=0.57 |
| Stop hit — per-position SL triggered | 2026-05-06 12:00:00 | 251.11 | 251.17 | 0.00 | SL hit |

### Cycle 19 — SELL (started 2026-05-07 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-07 10:55:00 | 251.24 | 252.04 | 0.00 | ORB-short ORB[251.53,254.15] vol=1.6x ATR=0.62 |
| Stop hit — per-position SL triggered | 2026-05-07 11:55:00 | 251.86 | 251.86 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-02-10 09:30:00 | 272.40 | 2026-02-10 09:35:00 | 273.21 | PARTIAL | 0.50 | 0.30% |
| BUY | retest1 | 2026-02-10 09:30:00 | 272.40 | 2026-02-10 10:40:00 | 272.60 | TARGET_HIT | 0.50 | 0.07% |
| SELL | retest1 | 2026-02-16 09:40:00 | 260.15 | 2026-02-16 09:55:00 | 260.85 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2026-02-17 10:20:00 | 262.95 | 2026-02-17 10:25:00 | 263.70 | PARTIAL | 0.50 | 0.29% |
| BUY | retest1 | 2026-02-17 10:20:00 | 262.95 | 2026-02-17 10:45:00 | 262.95 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-02-19 11:05:00 | 261.20 | 2026-02-19 11:15:00 | 261.70 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest1 | 2026-02-20 10:55:00 | 260.25 | 2026-02-20 12:35:00 | 259.72 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2026-02-23 10:00:00 | 258.20 | 2026-02-23 10:20:00 | 257.44 | PARTIAL | 0.50 | 0.30% |
| SELL | retest1 | 2026-02-23 10:00:00 | 258.20 | 2026-02-23 11:45:00 | 258.20 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-02-24 09:30:00 | 255.25 | 2026-02-24 11:25:00 | 255.78 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2026-03-11 10:40:00 | 241.40 | 2026-03-11 10:45:00 | 240.62 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2026-03-27 11:15:00 | 232.80 | 2026-03-27 11:30:00 | 231.85 | PARTIAL | 0.50 | 0.41% |
| SELL | retest1 | 2026-03-27 11:15:00 | 232.80 | 2026-03-27 11:35:00 | 232.80 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-04-16 11:00:00 | 242.17 | 2026-04-16 11:45:00 | 241.41 | PARTIAL | 0.50 | 0.31% |
| SELL | retest1 | 2026-04-16 11:00:00 | 242.17 | 2026-04-16 15:20:00 | 241.40 | TARGET_HIT | 0.50 | 0.32% |
| BUY | retest1 | 2026-04-17 11:15:00 | 242.69 | 2026-04-17 12:00:00 | 242.14 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2026-04-21 10:20:00 | 235.00 | 2026-04-21 10:40:00 | 235.47 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2026-04-23 09:45:00 | 241.80 | 2026-04-23 09:50:00 | 242.94 | PARTIAL | 0.50 | 0.47% |
| BUY | retest1 | 2026-04-23 09:45:00 | 241.80 | 2026-04-23 11:10:00 | 243.40 | TARGET_HIT | 0.50 | 0.66% |
| BUY | retest1 | 2026-04-27 09:55:00 | 250.34 | 2026-04-27 10:05:00 | 249.37 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest1 | 2026-04-29 10:10:00 | 254.64 | 2026-04-29 10:15:00 | 253.78 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2026-05-04 11:10:00 | 251.24 | 2026-05-04 11:15:00 | 252.27 | PARTIAL | 0.50 | 0.41% |
| BUY | retest1 | 2026-05-04 11:10:00 | 251.24 | 2026-05-04 12:10:00 | 251.24 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-05-05 10:25:00 | 249.00 | 2026-05-05 10:55:00 | 247.67 | PARTIAL | 0.50 | 0.53% |
| SELL | retest1 | 2026-05-05 10:25:00 | 249.00 | 2026-05-05 11:25:00 | 249.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-05-06 11:15:00 | 250.54 | 2026-05-06 12:00:00 | 251.11 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2026-05-07 10:55:00 | 251.24 | 2026-05-07 11:55:00 | 251.86 | STOP_HIT | 1.00 | -0.25% |
