# Castrol India Ltd. (CASTROLIND)

## Backtest Summary

- **Window:** 2024-05-13 09:15:00 → 2025-03-06 15:25:00 (15408 bars)
- **Last close:** 244.90
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
| ENTRY1 | 40 |
| ENTRY2 | 0 |
| PARTIAL | 18 |
| TARGET_HIT | 7 |
| STOP_HIT | 33 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 58 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 25 / 33
- **Target hits / Stop hits / Partials:** 7 / 33 / 18
- **Avg / median % per leg:** 0.21% / 0.00%
- **Sum % (uncompounded):** 12.36%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 33 | 13 | 39.4% | 3 | 20 | 10 | 0.13% | 4.3% |
| BUY @ 2nd Alert (retest1) | 33 | 13 | 39.4% | 3 | 20 | 10 | 0.13% | 4.3% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 25 | 12 | 48.0% | 4 | 13 | 8 | 0.32% | 8.1% |
| SELL @ 2nd Alert (retest1) | 25 | 12 | 48.0% | 4 | 13 | 8 | 0.32% | 8.1% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 58 | 25 | 43.1% | 7 | 33 | 18 | 0.21% | 12.4% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-05-22 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-22 09:35:00 | 192.90 | 194.10 | 0.00 | ORB-short ORB[193.55,195.20] vol=1.5x ATR=0.65 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-22 09:40:00 | 191.92 | 193.67 | 0.00 | T1 1.5R @ 191.92 |
| Stop hit — per-position SL triggered | 2024-05-22 09:50:00 | 192.90 | 193.47 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2024-05-24 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-24 09:35:00 | 192.40 | 193.11 | 0.00 | ORB-short ORB[192.50,194.50] vol=1.8x ATR=0.58 |
| Stop hit — per-position SL triggered | 2024-05-24 10:10:00 | 192.98 | 192.75 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2024-05-27 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-27 10:40:00 | 190.00 | 191.52 | 0.00 | ORB-short ORB[192.00,194.00] vol=2.7x ATR=0.64 |
| Stop hit — per-position SL triggered | 2024-05-27 13:45:00 | 190.64 | 190.83 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2024-05-28 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-28 09:35:00 | 189.60 | 190.24 | 0.00 | ORB-short ORB[189.80,191.65] vol=1.7x ATR=0.51 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-28 10:10:00 | 188.83 | 189.99 | 0.00 | T1 1.5R @ 188.83 |
| Stop hit — per-position SL triggered | 2024-05-28 10:15:00 | 189.60 | 189.98 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2024-05-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-30 10:15:00 | 185.85 | 187.19 | 0.00 | ORB-short ORB[186.95,189.00] vol=1.7x ATR=0.62 |
| Stop hit — per-position SL triggered | 2024-05-30 10:20:00 | 186.47 | 187.14 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2024-05-31 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-31 11:05:00 | 190.30 | 188.72 | 0.00 | ORB-long ORB[188.20,189.65] vol=3.3x ATR=0.74 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-31 11:15:00 | 191.41 | 189.34 | 0.00 | T1 1.5R @ 191.41 |
| Stop hit — per-position SL triggered | 2024-05-31 11:25:00 | 190.30 | 189.45 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2024-06-07 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-07 09:55:00 | 192.20 | 190.74 | 0.00 | ORB-long ORB[188.45,190.60] vol=2.8x ATR=0.86 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-07 10:05:00 | 193.49 | 191.36 | 0.00 | T1 1.5R @ 193.49 |
| Target hit | 2024-06-07 15:20:00 | 194.90 | 194.78 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 8 — BUY (started 2024-06-11 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-11 10:05:00 | 203.51 | 202.76 | 0.00 | ORB-long ORB[201.55,203.50] vol=1.6x ATR=0.90 |
| Stop hit — per-position SL triggered | 2024-06-11 11:00:00 | 202.61 | 202.88 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2024-06-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-13 11:15:00 | 203.36 | 204.84 | 0.00 | ORB-short ORB[204.40,207.00] vol=1.6x ATR=0.54 |
| Stop hit — per-position SL triggered | 2024-06-13 12:10:00 | 203.90 | 204.55 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2024-06-14 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-14 10:00:00 | 205.63 | 204.04 | 0.00 | ORB-long ORB[202.11,204.74] vol=3.0x ATR=0.86 |
| Stop hit — per-position SL triggered | 2024-06-14 10:25:00 | 204.77 | 204.60 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2024-06-20 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-20 09:45:00 | 207.13 | 206.17 | 0.00 | ORB-long ORB[205.01,206.50] vol=2.0x ATR=0.71 |
| Stop hit — per-position SL triggered | 2024-06-20 09:55:00 | 206.42 | 206.29 | 0.00 | SL hit |

### Cycle 12 — SELL (started 2024-06-25 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-25 11:00:00 | 210.83 | 212.63 | 0.00 | ORB-short ORB[213.01,214.30] vol=2.5x ATR=0.55 |
| Stop hit — per-position SL triggered | 2024-06-25 11:45:00 | 211.38 | 211.93 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2024-07-11 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-11 11:00:00 | 256.63 | 252.35 | 0.00 | ORB-long ORB[250.22,253.40] vol=6.6x ATR=1.52 |
| Stop hit — per-position SL triggered | 2024-07-11 11:05:00 | 255.11 | 253.01 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2024-07-29 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-29 09:40:00 | 271.34 | 268.22 | 0.00 | ORB-long ORB[265.41,268.90] vol=3.3x ATR=1.59 |
| Stop hit — per-position SL triggered | 2024-07-29 09:45:00 | 269.75 | 268.43 | 0.00 | SL hit |

### Cycle 15 — BUY (started 2024-07-30 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-30 10:30:00 | 274.12 | 269.61 | 0.00 | ORB-long ORB[266.67,269.90] vol=4.8x ATR=1.33 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-30 10:35:00 | 276.12 | 271.40 | 0.00 | T1 1.5R @ 276.12 |
| Target hit | 2024-07-30 11:25:00 | 274.55 | 274.65 | 0.00 | Trail-exit close<VWAP |

### Cycle 16 — BUY (started 2024-08-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-08 10:15:00 | 255.95 | 253.96 | 0.00 | ORB-long ORB[252.35,255.70] vol=1.6x ATR=1.27 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-08 10:20:00 | 257.86 | 254.39 | 0.00 | T1 1.5R @ 257.86 |
| Stop hit — per-position SL triggered | 2024-08-08 10:55:00 | 255.95 | 255.93 | 0.00 | SL hit |

### Cycle 17 — BUY (started 2024-08-09 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-09 09:40:00 | 261.05 | 258.93 | 0.00 | ORB-long ORB[257.00,259.70] vol=2.7x ATR=1.47 |
| Stop hit — per-position SL triggered | 2024-08-09 09:55:00 | 259.58 | 259.68 | 0.00 | SL hit |

### Cycle 18 — BUY (started 2024-08-16 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-16 09:45:00 | 250.70 | 249.13 | 0.00 | ORB-long ORB[247.50,250.50] vol=1.9x ATR=1.15 |
| Stop hit — per-position SL triggered | 2024-08-16 09:50:00 | 249.55 | 249.20 | 0.00 | SL hit |

### Cycle 19 — SELL (started 2024-08-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-20 10:15:00 | 252.80 | 254.08 | 0.00 | ORB-short ORB[253.65,256.00] vol=1.8x ATR=0.86 |
| Stop hit — per-position SL triggered | 2024-08-20 10:30:00 | 253.66 | 254.02 | 0.00 | SL hit |

### Cycle 20 — SELL (started 2024-09-19 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-19 09:45:00 | 253.00 | 254.17 | 0.00 | ORB-short ORB[253.40,256.35] vol=2.6x ATR=1.09 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-19 10:05:00 | 251.36 | 253.70 | 0.00 | T1 1.5R @ 251.36 |
| Target hit | 2024-09-19 15:15:00 | 247.60 | 247.31 | 0.00 | Trail-exit close>VWAP |

### Cycle 21 — BUY (started 2024-09-23 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-23 10:35:00 | 253.50 | 251.78 | 0.00 | ORB-long ORB[250.05,252.55] vol=1.5x ATR=0.91 |
| Stop hit — per-position SL triggered | 2024-09-23 10:40:00 | 252.59 | 251.87 | 0.00 | SL hit |

### Cycle 22 — BUY (started 2024-09-24 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-24 10:05:00 | 254.25 | 253.11 | 0.00 | ORB-long ORB[251.00,253.80] vol=2.0x ATR=0.84 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-24 10:20:00 | 255.51 | 253.50 | 0.00 | T1 1.5R @ 255.51 |
| Stop hit — per-position SL triggered | 2024-09-24 11:15:00 | 254.25 | 254.31 | 0.00 | SL hit |

### Cycle 23 — SELL (started 2024-09-25 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-25 10:05:00 | 248.40 | 249.67 | 0.00 | ORB-short ORB[248.70,251.85] vol=1.9x ATR=0.94 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-25 14:15:00 | 246.99 | 248.84 | 0.00 | T1 1.5R @ 246.99 |
| Target hit | 2024-09-25 15:20:00 | 247.15 | 248.41 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 24 — BUY (started 2024-09-26 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-26 10:35:00 | 247.70 | 245.47 | 0.00 | ORB-long ORB[244.25,247.40] vol=2.3x ATR=0.84 |
| Stop hit — per-position SL triggered | 2024-09-26 11:10:00 | 246.86 | 245.89 | 0.00 | SL hit |

### Cycle 25 — SELL (started 2024-10-03 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-03 10:35:00 | 240.37 | 242.30 | 0.00 | ORB-short ORB[240.40,243.72] vol=3.0x ATR=0.92 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-03 12:30:00 | 238.98 | 240.68 | 0.00 | T1 1.5R @ 238.98 |
| Target hit | 2024-10-03 15:20:00 | 235.27 | 237.42 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 26 — SELL (started 2024-10-07 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-07 09:55:00 | 224.91 | 227.70 | 0.00 | ORB-short ORB[229.61,232.47] vol=1.6x ATR=1.46 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-07 10:35:00 | 222.73 | 226.62 | 0.00 | T1 1.5R @ 222.73 |
| Stop hit — per-position SL triggered | 2024-10-07 11:20:00 | 224.91 | 225.21 | 0.00 | SL hit |

### Cycle 27 — BUY (started 2024-10-11 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-11 10:00:00 | 229.73 | 227.32 | 0.00 | ORB-long ORB[225.25,228.65] vol=1.5x ATR=0.98 |
| Stop hit — per-position SL triggered | 2024-10-11 10:05:00 | 228.75 | 227.39 | 0.00 | SL hit |

### Cycle 28 — SELL (started 2024-10-16 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-16 10:05:00 | 226.16 | 227.21 | 0.00 | ORB-short ORB[226.40,229.79] vol=3.1x ATR=0.72 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-16 11:50:00 | 225.09 | 226.59 | 0.00 | T1 1.5R @ 225.09 |
| Stop hit — per-position SL triggered | 2024-10-16 12:00:00 | 226.16 | 226.58 | 0.00 | SL hit |

### Cycle 29 — BUY (started 2024-10-30 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-30 10:00:00 | 212.68 | 210.75 | 0.00 | ORB-long ORB[209.09,211.39] vol=2.2x ATR=1.03 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-30 10:50:00 | 214.23 | 211.90 | 0.00 | T1 1.5R @ 214.23 |
| Stop hit — per-position SL triggered | 2024-10-30 14:05:00 | 212.68 | 212.56 | 0.00 | SL hit |

### Cycle 30 — SELL (started 2024-12-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-10 11:15:00 | 220.61 | 221.06 | 0.00 | ORB-short ORB[220.80,223.60] vol=2.5x ATR=0.62 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-10 11:35:00 | 219.68 | 220.99 | 0.00 | T1 1.5R @ 219.68 |
| Target hit | 2024-12-10 15:20:00 | 216.83 | 219.20 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 31 — BUY (started 2024-12-24 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-24 09:50:00 | 200.32 | 198.45 | 0.00 | ORB-long ORB[197.05,198.97] vol=1.9x ATR=0.91 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-24 10:25:00 | 201.68 | 199.55 | 0.00 | T1 1.5R @ 201.68 |
| Stop hit — per-position SL triggered | 2024-12-24 11:10:00 | 200.32 | 199.91 | 0.00 | SL hit |

### Cycle 32 — SELL (started 2024-12-26 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-26 10:50:00 | 196.60 | 197.69 | 0.00 | ORB-short ORB[197.70,199.60] vol=1.8x ATR=0.58 |
| Stop hit — per-position SL triggered | 2024-12-26 11:00:00 | 197.18 | 197.63 | 0.00 | SL hit |

### Cycle 33 — BUY (started 2025-01-01 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-01 09:30:00 | 200.32 | 198.98 | 0.00 | ORB-long ORB[196.87,199.85] vol=2.3x ATR=0.93 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-01 09:35:00 | 201.71 | 201.47 | 0.00 | T1 1.5R @ 201.71 |
| Target hit | 2025-01-01 09:45:00 | 203.60 | 203.76 | 0.00 | Trail-exit close<VWAP |

### Cycle 34 — BUY (started 2025-01-03 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-03 09:55:00 | 204.60 | 203.68 | 0.00 | ORB-long ORB[202.75,204.28] vol=3.8x ATR=0.70 |
| Stop hit — per-position SL triggered | 2025-01-03 10:05:00 | 203.90 | 203.80 | 0.00 | SL hit |

### Cycle 35 — SELL (started 2025-01-15 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-15 09:30:00 | 182.00 | 183.49 | 0.00 | ORB-short ORB[183.60,185.00] vol=2.2x ATR=0.64 |
| Stop hit — per-position SL triggered | 2025-01-15 09:40:00 | 182.64 | 183.10 | 0.00 | SL hit |

### Cycle 36 — BUY (started 2025-01-17 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-17 09:50:00 | 187.20 | 185.22 | 0.00 | ORB-long ORB[182.60,184.55] vol=4.2x ATR=0.68 |
| Stop hit — per-position SL triggered | 2025-01-17 09:55:00 | 186.52 | 185.70 | 0.00 | SL hit |

### Cycle 37 — SELL (started 2025-01-24 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-24 10:00:00 | 177.28 | 179.69 | 0.00 | ORB-short ORB[179.80,182.10] vol=1.8x ATR=0.87 |
| Stop hit — per-position SL triggered | 2025-01-24 10:20:00 | 178.15 | 179.33 | 0.00 | SL hit |

### Cycle 38 — BUY (started 2025-01-29 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-29 09:30:00 | 173.67 | 172.06 | 0.00 | ORB-long ORB[170.00,172.47] vol=2.4x ATR=0.94 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-29 09:50:00 | 175.09 | 173.41 | 0.00 | T1 1.5R @ 175.09 |
| Stop hit — per-position SL triggered | 2025-01-29 11:05:00 | 173.67 | 173.84 | 0.00 | SL hit |

### Cycle 39 — BUY (started 2025-01-31 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-31 09:50:00 | 177.40 | 175.76 | 0.00 | ORB-long ORB[174.35,175.85] vol=1.5x ATR=0.64 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-31 09:55:00 | 178.37 | 176.33 | 0.00 | T1 1.5R @ 178.37 |
| Stop hit — per-position SL triggered | 2025-01-31 10:05:00 | 177.40 | 176.82 | 0.00 | SL hit |

### Cycle 40 — BUY (started 2025-02-25 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-25 09:30:00 | 217.79 | 216.71 | 0.00 | ORB-long ORB[214.55,217.62] vol=3.3x ATR=0.77 |
| Stop hit — per-position SL triggered | 2025-02-25 10:05:00 | 217.02 | 217.21 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2024-05-22 09:35:00 | 192.90 | 2024-05-22 09:40:00 | 191.92 | PARTIAL | 0.50 | 0.51% |
| SELL | retest1 | 2024-05-22 09:35:00 | 192.90 | 2024-05-22 09:50:00 | 192.90 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-05-24 09:35:00 | 192.40 | 2024-05-24 10:10:00 | 192.98 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2024-05-27 10:40:00 | 190.00 | 2024-05-27 13:45:00 | 190.64 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2024-05-28 09:35:00 | 189.60 | 2024-05-28 10:10:00 | 188.83 | PARTIAL | 0.50 | 0.41% |
| SELL | retest1 | 2024-05-28 09:35:00 | 189.60 | 2024-05-28 10:15:00 | 189.60 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-05-30 10:15:00 | 185.85 | 2024-05-30 10:20:00 | 186.47 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2024-05-31 11:05:00 | 190.30 | 2024-05-31 11:15:00 | 191.41 | PARTIAL | 0.50 | 0.58% |
| BUY | retest1 | 2024-05-31 11:05:00 | 190.30 | 2024-05-31 11:25:00 | 190.30 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-06-07 09:55:00 | 192.20 | 2024-06-07 10:05:00 | 193.49 | PARTIAL | 0.50 | 0.67% |
| BUY | retest1 | 2024-06-07 09:55:00 | 192.20 | 2024-06-07 15:20:00 | 194.90 | TARGET_HIT | 0.50 | 1.40% |
| BUY | retest1 | 2024-06-11 10:05:00 | 203.51 | 2024-06-11 11:00:00 | 202.61 | STOP_HIT | 1.00 | -0.44% |
| SELL | retest1 | 2024-06-13 11:15:00 | 203.36 | 2024-06-13 12:10:00 | 203.90 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2024-06-14 10:00:00 | 205.63 | 2024-06-14 10:25:00 | 204.77 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest1 | 2024-06-20 09:45:00 | 207.13 | 2024-06-20 09:55:00 | 206.42 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2024-06-25 11:00:00 | 210.83 | 2024-06-25 11:45:00 | 211.38 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2024-07-11 11:00:00 | 256.63 | 2024-07-11 11:05:00 | 255.11 | STOP_HIT | 1.00 | -0.59% |
| BUY | retest1 | 2024-07-29 09:40:00 | 271.34 | 2024-07-29 09:45:00 | 269.75 | STOP_HIT | 1.00 | -0.59% |
| BUY | retest1 | 2024-07-30 10:30:00 | 274.12 | 2024-07-30 10:35:00 | 276.12 | PARTIAL | 0.50 | 0.73% |
| BUY | retest1 | 2024-07-30 10:30:00 | 274.12 | 2024-07-30 11:25:00 | 274.55 | TARGET_HIT | 0.50 | 0.16% |
| BUY | retest1 | 2024-08-08 10:15:00 | 255.95 | 2024-08-08 10:20:00 | 257.86 | PARTIAL | 0.50 | 0.74% |
| BUY | retest1 | 2024-08-08 10:15:00 | 255.95 | 2024-08-08 10:55:00 | 255.95 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-08-09 09:40:00 | 261.05 | 2024-08-09 09:55:00 | 259.58 | STOP_HIT | 1.00 | -0.56% |
| BUY | retest1 | 2024-08-16 09:45:00 | 250.70 | 2024-08-16 09:50:00 | 249.55 | STOP_HIT | 1.00 | -0.46% |
| SELL | retest1 | 2024-08-20 10:15:00 | 252.80 | 2024-08-20 10:30:00 | 253.66 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2024-09-19 09:45:00 | 253.00 | 2024-09-19 10:05:00 | 251.36 | PARTIAL | 0.50 | 0.65% |
| SELL | retest1 | 2024-09-19 09:45:00 | 253.00 | 2024-09-19 15:15:00 | 247.60 | TARGET_HIT | 0.50 | 2.13% |
| BUY | retest1 | 2024-09-23 10:35:00 | 253.50 | 2024-09-23 10:40:00 | 252.59 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2024-09-24 10:05:00 | 254.25 | 2024-09-24 10:20:00 | 255.51 | PARTIAL | 0.50 | 0.49% |
| BUY | retest1 | 2024-09-24 10:05:00 | 254.25 | 2024-09-24 11:15:00 | 254.25 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-09-25 10:05:00 | 248.40 | 2024-09-25 14:15:00 | 246.99 | PARTIAL | 0.50 | 0.57% |
| SELL | retest1 | 2024-09-25 10:05:00 | 248.40 | 2024-09-25 15:20:00 | 247.15 | TARGET_HIT | 0.50 | 0.50% |
| BUY | retest1 | 2024-09-26 10:35:00 | 247.70 | 2024-09-26 11:10:00 | 246.86 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2024-10-03 10:35:00 | 240.37 | 2024-10-03 12:30:00 | 238.98 | PARTIAL | 0.50 | 0.58% |
| SELL | retest1 | 2024-10-03 10:35:00 | 240.37 | 2024-10-03 15:20:00 | 235.27 | TARGET_HIT | 0.50 | 2.12% |
| SELL | retest1 | 2024-10-07 09:55:00 | 224.91 | 2024-10-07 10:35:00 | 222.73 | PARTIAL | 0.50 | 0.97% |
| SELL | retest1 | 2024-10-07 09:55:00 | 224.91 | 2024-10-07 11:20:00 | 224.91 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-10-11 10:00:00 | 229.73 | 2024-10-11 10:05:00 | 228.75 | STOP_HIT | 1.00 | -0.43% |
| SELL | retest1 | 2024-10-16 10:05:00 | 226.16 | 2024-10-16 11:50:00 | 225.09 | PARTIAL | 0.50 | 0.47% |
| SELL | retest1 | 2024-10-16 10:05:00 | 226.16 | 2024-10-16 12:00:00 | 226.16 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-10-30 10:00:00 | 212.68 | 2024-10-30 10:50:00 | 214.23 | PARTIAL | 0.50 | 0.73% |
| BUY | retest1 | 2024-10-30 10:00:00 | 212.68 | 2024-10-30 14:05:00 | 212.68 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-12-10 11:15:00 | 220.61 | 2024-12-10 11:35:00 | 219.68 | PARTIAL | 0.50 | 0.42% |
| SELL | retest1 | 2024-12-10 11:15:00 | 220.61 | 2024-12-10 15:20:00 | 216.83 | TARGET_HIT | 0.50 | 1.71% |
| BUY | retest1 | 2024-12-24 09:50:00 | 200.32 | 2024-12-24 10:25:00 | 201.68 | PARTIAL | 0.50 | 0.68% |
| BUY | retest1 | 2024-12-24 09:50:00 | 200.32 | 2024-12-24 11:10:00 | 200.32 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-12-26 10:50:00 | 196.60 | 2024-12-26 11:00:00 | 197.18 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2025-01-01 09:30:00 | 200.32 | 2025-01-01 09:35:00 | 201.71 | PARTIAL | 0.50 | 0.69% |
| BUY | retest1 | 2025-01-01 09:30:00 | 200.32 | 2025-01-01 09:45:00 | 203.60 | TARGET_HIT | 0.50 | 1.64% |
| BUY | retest1 | 2025-01-03 09:55:00 | 204.60 | 2025-01-03 10:05:00 | 203.90 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2025-01-15 09:30:00 | 182.00 | 2025-01-15 09:40:00 | 182.64 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2025-01-17 09:50:00 | 187.20 | 2025-01-17 09:55:00 | 186.52 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2025-01-24 10:00:00 | 177.28 | 2025-01-24 10:20:00 | 178.15 | STOP_HIT | 1.00 | -0.49% |
| BUY | retest1 | 2025-01-29 09:30:00 | 173.67 | 2025-01-29 09:50:00 | 175.09 | PARTIAL | 0.50 | 0.82% |
| BUY | retest1 | 2025-01-29 09:30:00 | 173.67 | 2025-01-29 11:05:00 | 173.67 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-01-31 09:50:00 | 177.40 | 2025-01-31 09:55:00 | 178.37 | PARTIAL | 0.50 | 0.54% |
| BUY | retest1 | 2025-01-31 09:50:00 | 177.40 | 2025-01-31 10:05:00 | 177.40 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-02-25 09:30:00 | 217.79 | 2025-02-25 10:05:00 | 217.02 | STOP_HIT | 1.00 | -0.36% |
