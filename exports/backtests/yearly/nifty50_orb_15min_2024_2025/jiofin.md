# JIOFIN (JIOFIN)

## Backtest Summary

- **Window:** 2024-10-08 09:15:00 → 2026-05-08 15:25:00 (29275 bars)
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
| ENTRY1 | 31 |
| ENTRY2 | 0 |
| PARTIAL | 14 |
| TARGET_HIT | 9 |
| STOP_HIT | 22 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 45 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 23 / 22
- **Target hits / Stop hits / Partials:** 9 / 22 / 14
- **Avg / median % per leg:** 0.37% / 0.26%
- **Sum % (uncompounded):** 16.74%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 23 | 10 | 43.5% | 4 | 13 | 6 | 0.17% | 4.0% |
| BUY @ 2nd Alert (retest1) | 23 | 10 | 43.5% | 4 | 13 | 6 | 0.17% | 4.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 22 | 13 | 59.1% | 5 | 9 | 8 | 0.58% | 12.8% |
| SELL @ 2nd Alert (retest1) | 22 | 13 | 59.1% | 5 | 9 | 8 | 0.58% | 12.8% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 45 | 23 | 51.1% | 9 | 22 | 14 | 0.37% | 16.7% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-10-08 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-08 11:05:00 | 339.90 | 336.88 | 0.00 | ORB-long ORB[333.00,337.50] vol=2.0x ATR=1.53 |
| Stop hit — per-position SL triggered | 2024-10-08 12:55:00 | 338.37 | 337.83 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2024-10-17 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-17 09:40:00 | 331.55 | 333.05 | 0.00 | ORB-short ORB[333.30,335.00] vol=2.5x ATR=0.89 |
| Stop hit — per-position SL triggered | 2024-10-17 09:55:00 | 332.44 | 332.83 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2024-10-22 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-22 09:30:00 | 324.10 | 326.29 | 0.00 | ORB-short ORB[325.10,328.40] vol=2.0x ATR=1.00 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-22 09:40:00 | 322.60 | 325.32 | 0.00 | T1 1.5R @ 322.60 |
| Stop hit — per-position SL triggered | 2024-10-22 09:45:00 | 324.10 | 325.25 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2024-10-25 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-25 09:35:00 | 312.70 | 315.10 | 0.00 | ORB-short ORB[314.75,317.85] vol=1.8x ATR=0.81 |
| Stop hit — per-position SL triggered | 2024-10-25 09:40:00 | 313.51 | 314.76 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2024-11-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-11 11:15:00 | 315.20 | 313.48 | 0.00 | ORB-long ORB[311.55,314.45] vol=2.0x ATR=0.77 |
| Stop hit — per-position SL triggered | 2024-11-11 11:35:00 | 314.43 | 313.65 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2024-11-13 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-13 09:30:00 | 304.85 | 307.06 | 0.00 | ORB-short ORB[306.80,310.00] vol=3.3x ATR=0.99 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-13 09:40:00 | 303.37 | 306.04 | 0.00 | T1 1.5R @ 303.37 |
| Target hit | 2024-11-13 15:20:00 | 300.50 | 302.13 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 7 — BUY (started 2024-11-19 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-19 10:00:00 | 323.20 | 321.00 | 0.00 | ORB-long ORB[318.45,321.55] vol=1.7x ATR=1.04 |
| Stop hit — per-position SL triggered | 2024-11-19 10:20:00 | 322.16 | 321.38 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2024-11-27 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-27 11:00:00 | 327.85 | 325.73 | 0.00 | ORB-long ORB[322.40,325.70] vol=1.9x ATR=0.94 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-27 11:15:00 | 329.25 | 326.20 | 0.00 | T1 1.5R @ 329.25 |
| Target hit | 2024-11-27 15:20:00 | 328.70 | 328.71 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 9 — BUY (started 2024-12-03 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-03 09:30:00 | 332.55 | 331.69 | 0.00 | ORB-long ORB[329.85,332.25] vol=1.9x ATR=0.65 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-03 09:35:00 | 333.52 | 332.39 | 0.00 | T1 1.5R @ 333.52 |
| Target hit | 2024-12-03 15:20:00 | 340.35 | 337.17 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 10 — BUY (started 2024-12-04 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-04 11:00:00 | 345.05 | 342.89 | 0.00 | ORB-long ORB[340.65,342.90] vol=3.3x ATR=1.06 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-04 11:05:00 | 346.64 | 343.34 | 0.00 | T1 1.5R @ 346.64 |
| Stop hit — per-position SL triggered | 2024-12-04 11:10:00 | 345.05 | 343.38 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2024-12-11 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-11 10:45:00 | 335.95 | 334.99 | 0.00 | ORB-long ORB[333.30,334.80] vol=1.9x ATR=0.65 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-11 11:05:00 | 336.92 | 335.27 | 0.00 | T1 1.5R @ 336.92 |
| Target hit | 2024-12-11 15:20:00 | 342.45 | 340.06 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 12 — SELL (started 2024-12-18 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-18 09:50:00 | 331.75 | 333.86 | 0.00 | ORB-short ORB[333.15,337.00] vol=1.8x ATR=0.88 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-18 10:05:00 | 330.43 | 333.13 | 0.00 | T1 1.5R @ 330.43 |
| Target hit | 2024-12-18 15:20:00 | 324.55 | 327.96 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 13 — BUY (started 2025-01-01 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-01 09:35:00 | 304.20 | 301.67 | 0.00 | ORB-long ORB[298.80,301.90] vol=1.6x ATR=0.88 |
| Stop hit — per-position SL triggered | 2025-01-01 09:40:00 | 303.32 | 301.84 | 0.00 | SL hit |

### Cycle 14 — SELL (started 2025-01-02 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-02 09:50:00 | 304.30 | 304.84 | 0.00 | ORB-short ORB[304.45,306.50] vol=1.6x ATR=0.68 |
| Stop hit — per-position SL triggered | 2025-01-02 10:05:00 | 304.98 | 304.78 | 0.00 | SL hit |

### Cycle 15 — SELL (started 2025-01-09 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-09 10:50:00 | 294.75 | 296.62 | 0.00 | ORB-short ORB[295.25,298.50] vol=2.5x ATR=0.75 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-09 11:15:00 | 293.63 | 296.05 | 0.00 | T1 1.5R @ 293.63 |
| Target hit | 2025-01-09 15:20:00 | 289.40 | 292.63 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 16 — SELL (started 2025-01-21 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-21 10:45:00 | 271.10 | 274.58 | 0.00 | ORB-short ORB[275.15,277.45] vol=2.7x ATR=0.83 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-21 10:50:00 | 269.85 | 274.38 | 0.00 | T1 1.5R @ 269.85 |
| Target hit | 2025-01-21 15:20:00 | 259.50 | 266.24 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 17 — BUY (started 2025-01-23 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-23 10:25:00 | 264.60 | 264.59 | 0.00 | ORB-long ORB[261.00,264.10] vol=3.2x ATR=1.02 |
| Stop hit — per-position SL triggered | 2025-01-23 10:50:00 | 263.58 | 264.58 | 0.00 | SL hit |

### Cycle 18 — SELL (started 2025-01-24 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-24 09:45:00 | 251.55 | 254.34 | 0.00 | ORB-short ORB[254.55,257.35] vol=1.5x ATR=1.02 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-24 09:50:00 | 250.01 | 253.18 | 0.00 | T1 1.5R @ 250.01 |
| Stop hit — per-position SL triggered | 2025-01-24 10:25:00 | 251.55 | 252.14 | 0.00 | SL hit |

### Cycle 19 — SELL (started 2025-01-28 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-28 10:25:00 | 232.60 | 234.79 | 0.00 | ORB-short ORB[235.30,238.50] vol=1.6x ATR=1.37 |
| Stop hit — per-position SL triggered | 2025-01-28 11:45:00 | 233.97 | 234.00 | 0.00 | SL hit |

### Cycle 20 — BUY (started 2025-01-29 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-29 10:05:00 | 237.00 | 236.19 | 0.00 | ORB-long ORB[233.80,235.65] vol=4.0x ATR=0.89 |
| Stop hit — per-position SL triggered | 2025-01-29 10:30:00 | 236.11 | 236.34 | 0.00 | SL hit |

### Cycle 21 — BUY (started 2025-02-07 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-07 10:10:00 | 250.20 | 249.04 | 0.00 | ORB-long ORB[247.76,249.43] vol=1.6x ATR=0.97 |
| Stop hit — per-position SL triggered | 2025-02-07 10:15:00 | 249.23 | 248.98 | 0.00 | SL hit |

### Cycle 22 — BUY (started 2025-02-13 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-13 09:30:00 | 230.45 | 229.14 | 0.00 | ORB-long ORB[227.66,229.94] vol=1.6x ATR=0.85 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-13 09:50:00 | 231.72 | 229.66 | 0.00 | T1 1.5R @ 231.72 |
| Stop hit — per-position SL triggered | 2025-02-13 10:25:00 | 230.45 | 230.04 | 0.00 | SL hit |

### Cycle 23 — BUY (started 2025-02-21 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-21 09:30:00 | 237.55 | 236.87 | 0.00 | ORB-long ORB[235.00,237.00] vol=2.6x ATR=0.62 |
| Stop hit — per-position SL triggered | 2025-02-21 09:40:00 | 236.93 | 237.02 | 0.00 | SL hit |

### Cycle 24 — SELL (started 2025-03-26 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-26 09:40:00 | 225.54 | 227.34 | 0.00 | ORB-short ORB[226.60,228.90] vol=1.6x ATR=0.79 |
| Stop hit — per-position SL triggered | 2025-03-26 09:50:00 | 226.33 | 227.13 | 0.00 | SL hit |

### Cycle 25 — BUY (started 2025-04-11 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-11 09:30:00 | 227.33 | 225.72 | 0.00 | ORB-long ORB[223.72,226.95] vol=1.6x ATR=0.86 |
| Stop hit — per-position SL triggered | 2025-04-11 10:20:00 | 226.47 | 226.23 | 0.00 | SL hit |

### Cycle 26 — BUY (started 2025-04-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-22 11:15:00 | 248.20 | 245.51 | 0.00 | ORB-long ORB[243.70,247.05] vol=2.5x ATR=0.67 |
| Stop hit — per-position SL triggered | 2025-04-22 11:25:00 | 247.53 | 245.63 | 0.00 | SL hit |

### Cycle 27 — SELL (started 2025-04-25 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-25 09:35:00 | 257.55 | 259.74 | 0.00 | ORB-short ORB[258.40,261.62] vol=2.5x ATR=0.89 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-25 09:45:00 | 256.22 | 258.95 | 0.00 | T1 1.5R @ 256.22 |
| Target hit | 2025-04-25 13:05:00 | 254.15 | 254.14 | 0.00 | Trail-exit close>VWAP |

### Cycle 28 — BUY (started 2025-04-28 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-28 09:30:00 | 256.50 | 254.48 | 0.00 | ORB-long ORB[252.30,255.52] vol=1.8x ATR=1.09 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-28 11:30:00 | 258.13 | 255.87 | 0.00 | T1 1.5R @ 258.13 |
| Target hit | 2025-04-28 15:20:00 | 257.66 | 256.93 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 29 — SELL (started 2025-04-29 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-29 09:40:00 | 256.43 | 258.52 | 0.00 | ORB-short ORB[257.62,259.75] vol=1.5x ATR=0.84 |
| Stop hit — per-position SL triggered | 2025-04-29 09:45:00 | 257.27 | 258.29 | 0.00 | SL hit |

### Cycle 30 — BUY (started 2025-04-30 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-30 09:30:00 | 259.00 | 257.13 | 0.00 | ORB-long ORB[255.20,257.97] vol=2.7x ATR=0.79 |
| Stop hit — per-position SL triggered | 2025-04-30 09:35:00 | 258.21 | 257.26 | 0.00 | SL hit |

### Cycle 31 — SELL (started 2025-05-08 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-08 09:35:00 | 255.75 | 256.85 | 0.00 | ORB-short ORB[256.10,258.30] vol=1.9x ATR=0.79 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-08 10:45:00 | 254.57 | 256.11 | 0.00 | T1 1.5R @ 254.57 |
| Stop hit — per-position SL triggered | 2025-05-08 12:10:00 | 255.75 | 255.89 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-10-08 11:05:00 | 339.90 | 2024-10-08 12:55:00 | 338.37 | STOP_HIT | 1.00 | -0.45% |
| SELL | retest1 | 2024-10-17 09:40:00 | 331.55 | 2024-10-17 09:55:00 | 332.44 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2024-10-22 09:30:00 | 324.10 | 2024-10-22 09:40:00 | 322.60 | PARTIAL | 0.50 | 0.46% |
| SELL | retest1 | 2024-10-22 09:30:00 | 324.10 | 2024-10-22 09:45:00 | 324.10 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-10-25 09:35:00 | 312.70 | 2024-10-25 09:40:00 | 313.51 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2024-11-11 11:15:00 | 315.20 | 2024-11-11 11:35:00 | 314.43 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2024-11-13 09:30:00 | 304.85 | 2024-11-13 09:40:00 | 303.37 | PARTIAL | 0.50 | 0.48% |
| SELL | retest1 | 2024-11-13 09:30:00 | 304.85 | 2024-11-13 15:20:00 | 300.50 | TARGET_HIT | 0.50 | 1.43% |
| BUY | retest1 | 2024-11-19 10:00:00 | 323.20 | 2024-11-19 10:20:00 | 322.16 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2024-11-27 11:00:00 | 327.85 | 2024-11-27 11:15:00 | 329.25 | PARTIAL | 0.50 | 0.43% |
| BUY | retest1 | 2024-11-27 11:00:00 | 327.85 | 2024-11-27 15:20:00 | 328.70 | TARGET_HIT | 0.50 | 0.26% |
| BUY | retest1 | 2024-12-03 09:30:00 | 332.55 | 2024-12-03 09:35:00 | 333.52 | PARTIAL | 0.50 | 0.29% |
| BUY | retest1 | 2024-12-03 09:30:00 | 332.55 | 2024-12-03 15:20:00 | 340.35 | TARGET_HIT | 0.50 | 2.35% |
| BUY | retest1 | 2024-12-04 11:00:00 | 345.05 | 2024-12-04 11:05:00 | 346.64 | PARTIAL | 0.50 | 0.46% |
| BUY | retest1 | 2024-12-04 11:00:00 | 345.05 | 2024-12-04 11:10:00 | 345.05 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-12-11 10:45:00 | 335.95 | 2024-12-11 11:05:00 | 336.92 | PARTIAL | 0.50 | 0.29% |
| BUY | retest1 | 2024-12-11 10:45:00 | 335.95 | 2024-12-11 15:20:00 | 342.45 | TARGET_HIT | 0.50 | 1.93% |
| SELL | retest1 | 2024-12-18 09:50:00 | 331.75 | 2024-12-18 10:05:00 | 330.43 | PARTIAL | 0.50 | 0.40% |
| SELL | retest1 | 2024-12-18 09:50:00 | 331.75 | 2024-12-18 15:20:00 | 324.55 | TARGET_HIT | 0.50 | 2.17% |
| BUY | retest1 | 2025-01-01 09:35:00 | 304.20 | 2025-01-01 09:40:00 | 303.32 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2025-01-02 09:50:00 | 304.30 | 2025-01-02 10:05:00 | 304.98 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2025-01-09 10:50:00 | 294.75 | 2025-01-09 11:15:00 | 293.63 | PARTIAL | 0.50 | 0.38% |
| SELL | retest1 | 2025-01-09 10:50:00 | 294.75 | 2025-01-09 15:20:00 | 289.40 | TARGET_HIT | 0.50 | 1.82% |
| SELL | retest1 | 2025-01-21 10:45:00 | 271.10 | 2025-01-21 10:50:00 | 269.85 | PARTIAL | 0.50 | 0.46% |
| SELL | retest1 | 2025-01-21 10:45:00 | 271.10 | 2025-01-21 15:20:00 | 259.50 | TARGET_HIT | 0.50 | 4.28% |
| BUY | retest1 | 2025-01-23 10:25:00 | 264.60 | 2025-01-23 10:50:00 | 263.58 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest1 | 2025-01-24 09:45:00 | 251.55 | 2025-01-24 09:50:00 | 250.01 | PARTIAL | 0.50 | 0.61% |
| SELL | retest1 | 2025-01-24 09:45:00 | 251.55 | 2025-01-24 10:25:00 | 251.55 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-01-28 10:25:00 | 232.60 | 2025-01-28 11:45:00 | 233.97 | STOP_HIT | 1.00 | -0.59% |
| BUY | retest1 | 2025-01-29 10:05:00 | 237.00 | 2025-01-29 10:30:00 | 236.11 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2025-02-07 10:10:00 | 250.20 | 2025-02-07 10:15:00 | 249.23 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest1 | 2025-02-13 09:30:00 | 230.45 | 2025-02-13 09:50:00 | 231.72 | PARTIAL | 0.50 | 0.55% |
| BUY | retest1 | 2025-02-13 09:30:00 | 230.45 | 2025-02-13 10:25:00 | 230.45 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-02-21 09:30:00 | 237.55 | 2025-02-21 09:40:00 | 236.93 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2025-03-26 09:40:00 | 225.54 | 2025-03-26 09:50:00 | 226.33 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2025-04-11 09:30:00 | 227.33 | 2025-04-11 10:20:00 | 226.47 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2025-04-22 11:15:00 | 248.20 | 2025-04-22 11:25:00 | 247.53 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2025-04-25 09:35:00 | 257.55 | 2025-04-25 09:45:00 | 256.22 | PARTIAL | 0.50 | 0.52% |
| SELL | retest1 | 2025-04-25 09:35:00 | 257.55 | 2025-04-25 13:05:00 | 254.15 | TARGET_HIT | 0.50 | 1.32% |
| BUY | retest1 | 2025-04-28 09:30:00 | 256.50 | 2025-04-28 11:30:00 | 258.13 | PARTIAL | 0.50 | 0.63% |
| BUY | retest1 | 2025-04-28 09:30:00 | 256.50 | 2025-04-28 15:20:00 | 257.66 | TARGET_HIT | 0.50 | 0.45% |
| SELL | retest1 | 2025-04-29 09:40:00 | 256.43 | 2025-04-29 09:45:00 | 257.27 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2025-04-30 09:30:00 | 259.00 | 2025-04-30 09:35:00 | 258.21 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2025-05-08 09:35:00 | 255.75 | 2025-05-08 10:45:00 | 254.57 | PARTIAL | 0.50 | 0.46% |
| SELL | retest1 | 2025-05-08 09:35:00 | 255.75 | 2025-05-08 12:10:00 | 255.75 | STOP_HIT | 0.50 | 0.00% |
