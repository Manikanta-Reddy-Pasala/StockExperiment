# IRCON International Ltd. (IRCON)

## Backtest Summary

- **Window:** 2024-05-13 09:15:00 → 2026-05-08 15:25:00 (36871 bars)
- **Last close:** 158.99
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
| ENTRY1 | 30 |
| ENTRY2 | 0 |
| PARTIAL | 13 |
| TARGET_HIT | 6 |
| STOP_HIT | 24 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 43 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 19 / 24
- **Target hits / Stop hits / Partials:** 6 / 24 / 13
- **Avg / median % per leg:** 0.39% / 0.00%
- **Sum % (uncompounded):** 16.80%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 23 | 8 | 34.8% | 2 | 15 | 6 | 0.31% | 7.0% |
| BUY @ 2nd Alert (retest1) | 23 | 8 | 34.8% | 2 | 15 | 6 | 0.31% | 7.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 20 | 11 | 55.0% | 4 | 9 | 7 | 0.49% | 9.8% |
| SELL @ 2nd Alert (retest1) | 20 | 11 | 55.0% | 4 | 9 | 7 | 0.49% | 9.8% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 43 | 19 | 44.2% | 6 | 24 | 13 | 0.39% | 16.8% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-05-16 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-16 10:00:00 | 248.60 | 246.44 | 0.00 | ORB-long ORB[243.75,247.15] vol=1.6x ATR=1.31 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-16 10:10:00 | 250.57 | 248.20 | 0.00 | T1 1.5R @ 250.57 |
| Target hit | 2024-05-16 15:20:00 | 262.90 | 256.66 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 2 — BUY (started 2024-05-30 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-30 10:50:00 | 274.75 | 271.53 | 0.00 | ORB-long ORB[269.50,273.00] vol=3.7x ATR=1.31 |
| Stop hit — per-position SL triggered | 2024-05-30 11:20:00 | 273.44 | 272.36 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2024-06-27 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-27 09:30:00 | 274.30 | 273.07 | 0.00 | ORB-long ORB[271.15,273.90] vol=3.0x ATR=0.93 |
| Stop hit — per-position SL triggered | 2024-06-27 10:30:00 | 273.37 | 273.62 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2024-07-01 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-01 09:45:00 | 271.25 | 269.39 | 0.00 | ORB-long ORB[267.70,270.40] vol=1.5x ATR=1.22 |
| Stop hit — per-position SL triggered | 2024-07-01 10:10:00 | 270.03 | 269.83 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2024-07-02 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-02 11:10:00 | 276.40 | 273.16 | 0.00 | ORB-long ORB[270.70,273.50] vol=7.4x ATR=0.93 |
| Stop hit — per-position SL triggered | 2024-07-02 11:15:00 | 275.47 | 273.54 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2024-08-01 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-01 09:30:00 | 288.80 | 289.45 | 0.00 | ORB-short ORB[288.85,291.00] vol=2.0x ATR=0.85 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-01 09:40:00 | 287.52 | 289.23 | 0.00 | T1 1.5R @ 287.52 |
| Stop hit — per-position SL triggered | 2024-08-01 10:15:00 | 288.80 | 288.97 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2024-08-16 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-16 09:30:00 | 270.20 | 267.62 | 0.00 | ORB-long ORB[265.35,268.65] vol=2.7x ATR=1.21 |
| Stop hit — per-position SL triggered | 2024-08-16 09:35:00 | 268.99 | 267.82 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2024-08-27 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-27 10:45:00 | 262.75 | 263.80 | 0.00 | ORB-short ORB[263.70,265.90] vol=1.8x ATR=0.63 |
| Stop hit — per-position SL triggered | 2024-08-27 11:05:00 | 263.38 | 263.72 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2024-08-28 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-28 09:55:00 | 265.90 | 264.88 | 0.00 | ORB-long ORB[263.50,265.70] vol=2.3x ATR=1.04 |
| Stop hit — per-position SL triggered | 2024-08-28 10:00:00 | 264.86 | 264.90 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2024-08-29 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-29 10:35:00 | 262.60 | 264.73 | 0.00 | ORB-short ORB[264.80,266.70] vol=1.7x ATR=0.64 |
| Stop hit — per-position SL triggered | 2024-08-29 11:40:00 | 263.24 | 264.31 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2024-08-30 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-30 09:30:00 | 261.90 | 263.02 | 0.00 | ORB-short ORB[262.15,264.35] vol=1.7x ATR=0.62 |
| Stop hit — per-position SL triggered | 2024-08-30 09:40:00 | 262.52 | 262.88 | 0.00 | SL hit |

### Cycle 12 — SELL (started 2024-09-03 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-03 09:35:00 | 255.80 | 258.13 | 0.00 | ORB-short ORB[256.85,259.75] vol=2.6x ATR=0.91 |
| Stop hit — per-position SL triggered | 2024-09-03 09:50:00 | 256.71 | 257.28 | 0.00 | SL hit |

### Cycle 13 — SELL (started 2024-09-05 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-05 09:30:00 | 255.20 | 256.18 | 0.00 | ORB-short ORB[255.30,257.40] vol=1.6x ATR=0.76 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-05 10:00:00 | 254.06 | 255.32 | 0.00 | T1 1.5R @ 254.06 |
| Target hit | 2024-09-05 15:20:00 | 252.25 | 253.91 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 14 — SELL (started 2024-09-11 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-11 10:00:00 | 242.25 | 243.32 | 0.00 | ORB-short ORB[242.55,245.55] vol=1.7x ATR=0.72 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-11 10:10:00 | 241.17 | 243.15 | 0.00 | T1 1.5R @ 241.17 |
| Target hit | 2024-09-11 15:20:00 | 238.50 | 240.87 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 15 — SELL (started 2024-09-17 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-17 09:35:00 | 237.30 | 238.41 | 0.00 | ORB-short ORB[237.40,240.60] vol=1.6x ATR=0.81 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-17 10:05:00 | 236.09 | 237.62 | 0.00 | T1 1.5R @ 236.09 |
| Stop hit — per-position SL triggered | 2024-09-17 10:35:00 | 237.30 | 237.43 | 0.00 | SL hit |

### Cycle 16 — SELL (started 2024-09-26 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-26 09:35:00 | 223.00 | 224.65 | 0.00 | ORB-short ORB[224.40,227.35] vol=2.5x ATR=0.86 |
| Stop hit — per-position SL triggered | 2024-09-26 10:10:00 | 223.86 | 223.82 | 0.00 | SL hit |

### Cycle 17 — BUY (started 2024-09-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-27 10:15:00 | 227.85 | 226.89 | 0.00 | ORB-long ORB[225.00,227.60] vol=2.5x ATR=0.70 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-27 10:30:00 | 228.90 | 227.18 | 0.00 | T1 1.5R @ 228.90 |
| Stop hit — per-position SL triggered | 2024-09-27 10:55:00 | 227.85 | 227.69 | 0.00 | SL hit |

### Cycle 18 — SELL (started 2024-10-07 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-07 09:30:00 | 215.98 | 217.76 | 0.00 | ORB-short ORB[216.76,219.99] vol=2.3x ATR=0.88 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-07 09:35:00 | 214.67 | 217.27 | 0.00 | T1 1.5R @ 214.67 |
| Target hit | 2024-10-07 15:20:00 | 206.27 | 209.05 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 19 — BUY (started 2024-10-09 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-09 11:05:00 | 224.00 | 222.17 | 0.00 | ORB-long ORB[220.60,222.85] vol=2.7x ATR=0.84 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-09 11:30:00 | 225.26 | 222.67 | 0.00 | T1 1.5R @ 225.26 |
| Stop hit — per-position SL triggered | 2024-10-09 11:40:00 | 224.00 | 222.75 | 0.00 | SL hit |

### Cycle 20 — BUY (started 2024-10-11 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-11 09:40:00 | 221.29 | 219.75 | 0.00 | ORB-long ORB[218.60,221.00] vol=2.6x ATR=0.80 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-11 09:45:00 | 222.50 | 220.59 | 0.00 | T1 1.5R @ 222.50 |
| Target hit | 2024-10-11 11:10:00 | 225.86 | 225.91 | 0.00 | Trail-exit close<VWAP |

### Cycle 21 — BUY (started 2024-11-27 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-27 09:40:00 | 201.56 | 199.09 | 0.00 | ORB-long ORB[196.60,198.50] vol=5.9x ATR=1.03 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-27 09:45:00 | 203.11 | 200.23 | 0.00 | T1 1.5R @ 203.11 |
| Stop hit — per-position SL triggered | 2024-11-27 09:50:00 | 201.56 | 200.43 | 0.00 | SL hit |

### Cycle 22 — BUY (started 2024-12-09 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-09 09:30:00 | 229.01 | 228.05 | 0.00 | ORB-long ORB[226.02,228.95] vol=2.2x ATR=1.00 |
| Stop hit — per-position SL triggered | 2024-12-09 09:35:00 | 228.01 | 228.07 | 0.00 | SL hit |

### Cycle 23 — BUY (started 2024-12-11 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-11 09:30:00 | 224.47 | 223.65 | 0.00 | ORB-long ORB[222.43,224.16] vol=2.1x ATR=0.63 |
| Stop hit — per-position SL triggered | 2024-12-11 09:40:00 | 223.84 | 223.89 | 0.00 | SL hit |

### Cycle 24 — BUY (started 2024-12-24 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-24 10:00:00 | 214.88 | 212.47 | 0.00 | ORB-long ORB[210.33,213.00] vol=3.2x ATR=1.14 |
| Stop hit — per-position SL triggered | 2024-12-24 10:30:00 | 213.74 | 212.91 | 0.00 | SL hit |

### Cycle 25 — BUY (started 2024-12-30 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-30 09:55:00 | 211.99 | 208.92 | 0.00 | ORB-long ORB[206.86,209.30] vol=3.6x ATR=1.06 |
| Stop hit — per-position SL triggered | 2024-12-30 10:05:00 | 210.93 | 209.64 | 0.00 | SL hit |

### Cycle 26 — BUY (started 2025-01-09 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-09 09:50:00 | 207.55 | 206.35 | 0.00 | ORB-long ORB[204.51,206.33] vol=5.4x ATR=0.90 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-09 10:25:00 | 208.89 | 207.06 | 0.00 | T1 1.5R @ 208.89 |
| Stop hit — per-position SL triggered | 2025-01-09 10:45:00 | 207.55 | 207.36 | 0.00 | SL hit |

### Cycle 27 — SELL (started 2025-02-06 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-06 09:30:00 | 192.32 | 193.48 | 0.00 | ORB-short ORB[192.80,195.00] vol=2.2x ATR=0.56 |
| Stop hit — per-position SL triggered | 2025-02-06 09:35:00 | 192.88 | 193.40 | 0.00 | SL hit |

### Cycle 28 — SELL (started 2025-02-10 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-10 09:30:00 | 189.52 | 191.26 | 0.00 | ORB-short ORB[190.25,193.03] vol=1.7x ATR=0.94 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-10 10:00:00 | 188.11 | 190.36 | 0.00 | T1 1.5R @ 188.11 |
| Stop hit — per-position SL triggered | 2025-02-10 15:10:00 | 189.52 | 188.78 | 0.00 | SL hit |

### Cycle 29 — BUY (started 2025-04-21 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-21 09:50:00 | 162.44 | 161.05 | 0.00 | ORB-long ORB[159.50,161.80] vol=2.9x ATR=0.60 |
| Stop hit — per-position SL triggered | 2025-04-21 10:10:00 | 161.84 | 161.46 | 0.00 | SL hit |

### Cycle 30 — SELL (started 2025-04-23 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-23 09:30:00 | 163.32 | 164.26 | 0.00 | ORB-short ORB[163.35,165.25] vol=2.3x ATR=0.61 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-23 09:35:00 | 162.40 | 164.08 | 0.00 | T1 1.5R @ 162.40 |
| Target hit | 2025-04-23 14:05:00 | 162.40 | 162.27 | 0.00 | Trail-exit close>VWAP |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-05-16 10:00:00 | 248.60 | 2024-05-16 10:10:00 | 250.57 | PARTIAL | 0.50 | 0.79% |
| BUY | retest1 | 2024-05-16 10:00:00 | 248.60 | 2024-05-16 15:20:00 | 262.90 | TARGET_HIT | 0.50 | 5.75% |
| BUY | retest1 | 2024-05-30 10:50:00 | 274.75 | 2024-05-30 11:20:00 | 273.44 | STOP_HIT | 1.00 | -0.48% |
| BUY | retest1 | 2024-06-27 09:30:00 | 274.30 | 2024-06-27 10:30:00 | 273.37 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2024-07-01 09:45:00 | 271.25 | 2024-07-01 10:10:00 | 270.03 | STOP_HIT | 1.00 | -0.45% |
| BUY | retest1 | 2024-07-02 11:10:00 | 276.40 | 2024-07-02 11:15:00 | 275.47 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2024-08-01 09:30:00 | 288.80 | 2024-08-01 09:40:00 | 287.52 | PARTIAL | 0.50 | 0.44% |
| SELL | retest1 | 2024-08-01 09:30:00 | 288.80 | 2024-08-01 10:15:00 | 288.80 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-08-16 09:30:00 | 270.20 | 2024-08-16 09:35:00 | 268.99 | STOP_HIT | 1.00 | -0.45% |
| SELL | retest1 | 2024-08-27 10:45:00 | 262.75 | 2024-08-27 11:05:00 | 263.38 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2024-08-28 09:55:00 | 265.90 | 2024-08-28 10:00:00 | 264.86 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest1 | 2024-08-29 10:35:00 | 262.60 | 2024-08-29 11:40:00 | 263.24 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2024-08-30 09:30:00 | 261.90 | 2024-08-30 09:40:00 | 262.52 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2024-09-03 09:35:00 | 255.80 | 2024-09-03 09:50:00 | 256.71 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2024-09-05 09:30:00 | 255.20 | 2024-09-05 10:00:00 | 254.06 | PARTIAL | 0.50 | 0.45% |
| SELL | retest1 | 2024-09-05 09:30:00 | 255.20 | 2024-09-05 15:20:00 | 252.25 | TARGET_HIT | 0.50 | 1.16% |
| SELL | retest1 | 2024-09-11 10:00:00 | 242.25 | 2024-09-11 10:10:00 | 241.17 | PARTIAL | 0.50 | 0.45% |
| SELL | retest1 | 2024-09-11 10:00:00 | 242.25 | 2024-09-11 15:20:00 | 238.50 | TARGET_HIT | 0.50 | 1.55% |
| SELL | retest1 | 2024-09-17 09:35:00 | 237.30 | 2024-09-17 10:05:00 | 236.09 | PARTIAL | 0.50 | 0.51% |
| SELL | retest1 | 2024-09-17 09:35:00 | 237.30 | 2024-09-17 10:35:00 | 237.30 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-09-26 09:35:00 | 223.00 | 2024-09-26 10:10:00 | 223.86 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest1 | 2024-09-27 10:15:00 | 227.85 | 2024-09-27 10:30:00 | 228.90 | PARTIAL | 0.50 | 0.46% |
| BUY | retest1 | 2024-09-27 10:15:00 | 227.85 | 2024-09-27 10:55:00 | 227.85 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-10-07 09:30:00 | 215.98 | 2024-10-07 09:35:00 | 214.67 | PARTIAL | 0.50 | 0.61% |
| SELL | retest1 | 2024-10-07 09:30:00 | 215.98 | 2024-10-07 15:20:00 | 206.27 | TARGET_HIT | 0.50 | 4.50% |
| BUY | retest1 | 2024-10-09 11:05:00 | 224.00 | 2024-10-09 11:30:00 | 225.26 | PARTIAL | 0.50 | 0.56% |
| BUY | retest1 | 2024-10-09 11:05:00 | 224.00 | 2024-10-09 11:40:00 | 224.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-10-11 09:40:00 | 221.29 | 2024-10-11 09:45:00 | 222.50 | PARTIAL | 0.50 | 0.55% |
| BUY | retest1 | 2024-10-11 09:40:00 | 221.29 | 2024-10-11 11:10:00 | 225.86 | TARGET_HIT | 0.50 | 2.07% |
| BUY | retest1 | 2024-11-27 09:40:00 | 201.56 | 2024-11-27 09:45:00 | 203.11 | PARTIAL | 0.50 | 0.77% |
| BUY | retest1 | 2024-11-27 09:40:00 | 201.56 | 2024-11-27 09:50:00 | 201.56 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-12-09 09:30:00 | 229.01 | 2024-12-09 09:35:00 | 228.01 | STOP_HIT | 1.00 | -0.44% |
| BUY | retest1 | 2024-12-11 09:30:00 | 224.47 | 2024-12-11 09:40:00 | 223.84 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2024-12-24 10:00:00 | 214.88 | 2024-12-24 10:30:00 | 213.74 | STOP_HIT | 1.00 | -0.53% |
| BUY | retest1 | 2024-12-30 09:55:00 | 211.99 | 2024-12-30 10:05:00 | 210.93 | STOP_HIT | 1.00 | -0.50% |
| BUY | retest1 | 2025-01-09 09:50:00 | 207.55 | 2025-01-09 10:25:00 | 208.89 | PARTIAL | 0.50 | 0.65% |
| BUY | retest1 | 2025-01-09 09:50:00 | 207.55 | 2025-01-09 10:45:00 | 207.55 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-02-06 09:30:00 | 192.32 | 2025-02-06 09:35:00 | 192.88 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2025-02-10 09:30:00 | 189.52 | 2025-02-10 10:00:00 | 188.11 | PARTIAL | 0.50 | 0.74% |
| SELL | retest1 | 2025-02-10 09:30:00 | 189.52 | 2025-02-10 15:10:00 | 189.52 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-04-21 09:50:00 | 162.44 | 2025-04-21 10:10:00 | 161.84 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest1 | 2025-04-23 09:30:00 | 163.32 | 2025-04-23 09:35:00 | 162.40 | PARTIAL | 0.50 | 0.56% |
| SELL | retest1 | 2025-04-23 09:30:00 | 163.32 | 2025-04-23 14:05:00 | 162.40 | TARGET_HIT | 0.50 | 0.56% |
