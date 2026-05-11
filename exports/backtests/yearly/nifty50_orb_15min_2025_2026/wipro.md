# WIPRO (WIPRO)

## Backtest Summary

- **Window:** 2025-07-10 09:15:00 → 2026-05-08 15:25:00 (13800 bars)
- **Last close:** 197.88
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
| ENTRY1 | 41 |
| ENTRY2 | 0 |
| PARTIAL | 20 |
| TARGET_HIT | 9 |
| STOP_HIT | 32 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 61 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 29 / 32
- **Target hits / Stop hits / Partials:** 9 / 32 / 20
- **Avg / median % per leg:** 0.13% / 0.00%
- **Sum % (uncompounded):** 7.74%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 37 | 20 | 54.1% | 7 | 17 | 13 | 0.16% | 6.1% |
| BUY @ 2nd Alert (retest1) | 37 | 20 | 54.1% | 7 | 17 | 13 | 0.16% | 6.1% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 24 | 9 | 37.5% | 2 | 15 | 7 | 0.07% | 1.7% |
| SELL @ 2nd Alert (retest1) | 24 | 9 | 37.5% | 2 | 15 | 7 | 0.07% | 1.7% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 61 | 29 | 47.5% | 9 | 32 | 20 | 0.13% | 7.7% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-07-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-14 11:15:00 | 253.70 | 255.87 | 0.00 | ORB-short ORB[256.00,259.55] vol=1.7x ATR=0.54 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-14 11:50:00 | 252.90 | 255.30 | 0.00 | T1 1.5R @ 252.90 |
| Stop hit — per-position SL triggered | 2025-07-14 13:50:00 | 253.70 | 254.28 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2025-07-15 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-15 10:30:00 | 255.75 | 254.37 | 0.00 | ORB-long ORB[254.00,255.55] vol=1.8x ATR=0.72 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-15 10:35:00 | 256.83 | 254.64 | 0.00 | T1 1.5R @ 256.83 |
| Target hit | 2025-07-15 14:50:00 | 256.90 | 256.95 | 0.00 | Trail-exit close<VWAP |

### Cycle 3 — SELL (started 2025-07-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-23 10:15:00 | 258.70 | 259.15 | 0.00 | ORB-short ORB[258.80,260.70] vol=1.7x ATR=0.50 |
| Stop hit — per-position SL triggered | 2025-07-23 10:40:00 | 259.20 | 259.11 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2025-08-06 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-06 11:10:00 | 241.19 | 242.75 | 0.00 | ORB-short ORB[243.07,246.00] vol=1.5x ATR=0.47 |
| Stop hit — per-position SL triggered | 2025-08-06 12:20:00 | 241.66 | 242.33 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2025-08-11 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-11 11:05:00 | 240.84 | 239.96 | 0.00 | ORB-long ORB[238.62,240.63] vol=2.2x ATR=0.39 |
| Stop hit — per-position SL triggered | 2025-08-11 11:10:00 | 240.45 | 239.98 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2025-08-12 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-12 09:30:00 | 244.70 | 243.57 | 0.00 | ORB-long ORB[242.12,244.25] vol=1.7x ATR=0.68 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-12 09:45:00 | 245.71 | 244.26 | 0.00 | T1 1.5R @ 245.71 |
| Stop hit — per-position SL triggered | 2025-08-12 10:15:00 | 244.70 | 244.83 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2025-08-14 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-14 10:30:00 | 246.69 | 244.62 | 0.00 | ORB-long ORB[242.41,244.70] vol=1.8x ATR=0.63 |
| Stop hit — per-position SL triggered | 2025-08-14 10:40:00 | 246.06 | 244.91 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2025-08-20 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-20 09:50:00 | 247.58 | 246.74 | 0.00 | ORB-long ORB[245.30,247.36] vol=2.0x ATR=0.61 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-20 10:00:00 | 248.50 | 246.97 | 0.00 | T1 1.5R @ 248.50 |
| Target hit | 2025-08-20 15:20:00 | 250.74 | 250.23 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 9 — BUY (started 2025-08-25 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-25 09:30:00 | 255.14 | 253.83 | 0.00 | ORB-long ORB[250.60,254.34] vol=3.8x ATR=0.77 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-25 09:45:00 | 256.30 | 254.74 | 0.00 | T1 1.5R @ 256.30 |
| Stop hit — per-position SL triggered | 2025-08-25 10:00:00 | 255.14 | 254.92 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2025-08-26 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-26 09:35:00 | 254.03 | 253.38 | 0.00 | ORB-long ORB[252.56,253.84] vol=3.6x ATR=0.54 |
| Stop hit — per-position SL triggered | 2025-08-26 10:00:00 | 253.49 | 253.82 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2025-09-02 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-02 11:10:00 | 251.64 | 250.87 | 0.00 | ORB-long ORB[249.60,251.30] vol=1.6x ATR=0.38 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-02 11:45:00 | 252.21 | 251.01 | 0.00 | T1 1.5R @ 252.21 |
| Stop hit — per-position SL triggered | 2025-09-02 13:50:00 | 251.64 | 251.62 | 0.00 | SL hit |

### Cycle 12 — SELL (started 2025-09-05 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-05 10:10:00 | 242.80 | 245.63 | 0.00 | ORB-short ORB[245.83,247.32] vol=2.7x ATR=0.77 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-05 10:15:00 | 241.64 | 244.91 | 0.00 | T1 1.5R @ 241.64 |
| Stop hit — per-position SL triggered | 2025-09-05 10:20:00 | 242.80 | 244.63 | 0.00 | SL hit |

### Cycle 13 — SELL (started 2025-09-12 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-12 09:30:00 | 252.30 | 253.47 | 0.00 | ORB-short ORB[252.70,255.90] vol=1.9x ATR=0.57 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-12 09:40:00 | 251.44 | 252.97 | 0.00 | T1 1.5R @ 251.44 |
| Stop hit — per-position SL triggered | 2025-09-12 11:25:00 | 252.30 | 252.27 | 0.00 | SL hit |

### Cycle 14 — SELL (started 2025-09-15 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-15 10:45:00 | 249.94 | 250.59 | 0.00 | ORB-short ORB[250.15,251.79] vol=1.6x ATR=0.33 |
| Stop hit — per-position SL triggered | 2025-09-15 11:05:00 | 250.27 | 250.54 | 0.00 | SL hit |

### Cycle 15 — SELL (started 2025-09-18 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-18 11:10:00 | 256.42 | 258.41 | 0.00 | ORB-short ORB[256.55,259.80] vol=1.6x ATR=0.52 |
| Stop hit — per-position SL triggered | 2025-09-18 12:20:00 | 256.94 | 258.16 | 0.00 | SL hit |

### Cycle 16 — SELL (started 2025-09-24 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-24 11:05:00 | 245.02 | 246.46 | 0.00 | ORB-short ORB[246.78,249.12] vol=1.6x ATR=0.54 |
| Target hit | 2025-09-24 15:20:00 | 244.65 | 245.57 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 17 — BUY (started 2025-11-10 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-10 10:20:00 | 239.17 | 238.02 | 0.00 | ORB-long ORB[237.02,238.10] vol=2.1x ATR=0.40 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-10 10:25:00 | 239.78 | 238.18 | 0.00 | T1 1.5R @ 239.78 |
| Target hit | 2025-11-10 15:15:00 | 239.80 | 239.87 | 0.00 | Trail-exit close<VWAP |

### Cycle 18 — BUY (started 2025-11-12 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-12 10:50:00 | 245.33 | 244.30 | 0.00 | ORB-long ORB[242.10,244.69] vol=2.8x ATR=0.51 |
| Stop hit — per-position SL triggered | 2025-11-12 11:00:00 | 244.82 | 244.40 | 0.00 | SL hit |

### Cycle 19 — BUY (started 2025-11-19 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-19 10:05:00 | 243.30 | 241.89 | 0.00 | ORB-long ORB[240.54,242.10] vol=1.7x ATR=0.49 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-19 10:30:00 | 244.04 | 242.46 | 0.00 | T1 1.5R @ 244.04 |
| Target hit | 2025-11-19 15:20:00 | 246.15 | 245.13 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 20 — BUY (started 2025-11-26 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-26 09:35:00 | 248.17 | 246.92 | 0.00 | ORB-long ORB[245.62,247.50] vol=1.6x ATR=0.63 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-26 09:50:00 | 249.12 | 247.57 | 0.00 | T1 1.5R @ 249.12 |
| Stop hit — per-position SL triggered | 2025-11-26 09:55:00 | 248.17 | 247.62 | 0.00 | SL hit |

### Cycle 21 — BUY (started 2025-11-27 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-27 09:55:00 | 251.10 | 250.26 | 0.00 | ORB-long ORB[249.23,250.97] vol=1.9x ATR=0.54 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-27 10:00:00 | 251.91 | 250.42 | 0.00 | T1 1.5R @ 251.91 |
| Stop hit — per-position SL triggered | 2025-11-27 10:05:00 | 251.10 | 250.47 | 0.00 | SL hit |

### Cycle 22 — BUY (started 2025-12-03 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-03 10:05:00 | 254.53 | 253.20 | 0.00 | ORB-long ORB[250.55,253.89] vol=2.7x ATR=0.67 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-03 10:25:00 | 255.54 | 253.66 | 0.00 | T1 1.5R @ 255.54 |
| Target hit | 2025-12-03 13:15:00 | 255.02 | 255.11 | 0.00 | Trail-exit close<VWAP |

### Cycle 23 — BUY (started 2025-12-04 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-04 10:20:00 | 258.12 | 257.36 | 0.00 | ORB-long ORB[255.50,257.40] vol=1.5x ATR=0.54 |
| Stop hit — per-position SL triggered | 2025-12-04 10:25:00 | 257.58 | 257.38 | 0.00 | SL hit |

### Cycle 24 — SELL (started 2025-12-16 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-16 11:00:00 | 260.14 | 260.56 | 0.00 | ORB-short ORB[260.19,261.75] vol=1.6x ATR=0.41 |
| Stop hit — per-position SL triggered | 2025-12-16 11:45:00 | 260.55 | 260.50 | 0.00 | SL hit |

### Cycle 25 — BUY (started 2025-12-17 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-17 09:30:00 | 260.84 | 259.94 | 0.00 | ORB-long ORB[258.31,260.44] vol=1.6x ATR=0.45 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-17 09:35:00 | 261.52 | 260.46 | 0.00 | T1 1.5R @ 261.52 |
| Stop hit — per-position SL triggered | 2025-12-17 09:45:00 | 260.84 | 260.59 | 0.00 | SL hit |

### Cycle 26 — BUY (started 2026-01-02 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-02 09:40:00 | 268.65 | 267.63 | 0.00 | ORB-long ORB[266.15,267.85] vol=4.1x ATR=0.62 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-02 10:30:00 | 269.58 | 268.42 | 0.00 | T1 1.5R @ 269.58 |
| Target hit | 2026-01-02 13:45:00 | 269.85 | 270.08 | 0.00 | Trail-exit close<VWAP |

### Cycle 27 — BUY (started 2026-02-01 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-01 11:05:00 | 237.87 | 236.53 | 0.00 | ORB-long ORB[235.44,236.80] vol=2.0x ATR=0.45 |
| Stop hit — per-position SL triggered | 2026-02-01 11:40:00 | 237.42 | 236.81 | 0.00 | SL hit |

### Cycle 28 — SELL (started 2026-02-02 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-02 10:20:00 | 240.38 | 241.94 | 0.00 | ORB-short ORB[241.27,244.08] vol=1.6x ATR=1.04 |
| Stop hit — per-position SL triggered | 2026-02-02 10:35:00 | 241.42 | 241.67 | 0.00 | SL hit |

### Cycle 29 — BUY (started 2026-02-10 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-10 10:05:00 | 230.40 | 229.08 | 0.00 | ORB-long ORB[227.20,229.98] vol=1.6x ATR=0.52 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-10 10:15:00 | 231.19 | 229.35 | 0.00 | T1 1.5R @ 231.19 |
| Target hit | 2026-02-10 15:20:00 | 231.29 | 230.60 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 30 — SELL (started 2026-02-11 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-11 10:35:00 | 230.60 | 231.47 | 0.00 | ORB-short ORB[231.46,233.00] vol=2.6x ATR=0.44 |
| Stop hit — per-position SL triggered | 2026-02-11 10:40:00 | 231.04 | 231.41 | 0.00 | SL hit |

### Cycle 31 — BUY (started 2026-02-17 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-17 10:05:00 | 216.92 | 214.92 | 0.00 | ORB-long ORB[212.63,215.50] vol=1.7x ATR=0.80 |
| Stop hit — per-position SL triggered | 2026-02-17 10:25:00 | 216.12 | 215.23 | 0.00 | SL hit |

### Cycle 32 — SELL (started 2026-02-18 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-18 09:30:00 | 213.04 | 214.33 | 0.00 | ORB-short ORB[213.59,216.40] vol=2.3x ATR=0.76 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-18 09:45:00 | 211.90 | 213.60 | 0.00 | T1 1.5R @ 211.90 |
| Target hit | 2026-02-18 15:00:00 | 211.94 | 211.73 | 0.00 | Trail-exit close>VWAP |

### Cycle 33 — BUY (started 2026-02-20 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-20 10:55:00 | 211.00 | 210.20 | 0.00 | ORB-long ORB[208.25,210.00] vol=1.9x ATR=0.61 |
| Stop hit — per-position SL triggered | 2026-02-20 12:55:00 | 210.39 | 210.41 | 0.00 | SL hit |

### Cycle 34 — SELL (started 2026-02-23 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-23 10:30:00 | 205.32 | 207.76 | 0.00 | ORB-short ORB[209.54,211.74] vol=1.5x ATR=0.63 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-23 10:45:00 | 204.37 | 207.36 | 0.00 | T1 1.5R @ 204.37 |
| Stop hit — per-position SL triggered | 2026-02-23 11:05:00 | 205.32 | 207.11 | 0.00 | SL hit |

### Cycle 35 — SELL (started 2026-02-24 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-24 09:30:00 | 200.93 | 201.53 | 0.00 | ORB-short ORB[201.10,203.29] vol=2.4x ATR=0.70 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-24 09:40:00 | 199.89 | 201.20 | 0.00 | T1 1.5R @ 199.89 |
| Stop hit — per-position SL triggered | 2026-02-24 09:45:00 | 200.93 | 201.18 | 0.00 | SL hit |

### Cycle 36 — BUY (started 2026-03-09 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-09 10:50:00 | 196.14 | 194.50 | 0.00 | ORB-long ORB[193.03,195.40] vol=2.0x ATR=0.61 |
| Stop hit — per-position SL triggered | 2026-03-09 10:55:00 | 195.53 | 194.68 | 0.00 | SL hit |

### Cycle 37 — SELL (started 2026-03-16 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-16 11:00:00 | 194.05 | 196.12 | 0.00 | ORB-short ORB[196.24,198.29] vol=2.7x ATR=0.63 |
| Stop hit — per-position SL triggered | 2026-03-16 11:15:00 | 194.68 | 195.91 | 0.00 | SL hit |

### Cycle 38 — BUY (started 2026-04-21 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-21 10:45:00 | 205.15 | 203.84 | 0.00 | ORB-long ORB[202.48,203.90] vol=2.3x ATR=0.36 |
| Stop hit — per-position SL triggered | 2026-04-21 10:50:00 | 204.79 | 203.98 | 0.00 | SL hit |

### Cycle 39 — SELL (started 2026-04-24 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-24 09:45:00 | 199.37 | 200.88 | 0.00 | ORB-short ORB[201.11,202.75] vol=2.4x ATR=0.52 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-24 10:15:00 | 198.59 | 200.06 | 0.00 | T1 1.5R @ 198.59 |
| Stop hit — per-position SL triggered | 2026-04-24 11:05:00 | 199.37 | 199.54 | 0.00 | SL hit |

### Cycle 40 — BUY (started 2026-04-30 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-30 09:30:00 | 201.90 | 200.91 | 0.00 | ORB-long ORB[200.05,201.80] vol=1.5x ATR=0.48 |
| Stop hit — per-position SL triggered | 2026-04-30 10:00:00 | 201.42 | 201.36 | 0.00 | SL hit |

### Cycle 41 — SELL (started 2026-05-08 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-08 09:30:00 | 197.43 | 198.10 | 0.00 | ORB-short ORB[197.49,198.94] vol=1.8x ATR=0.40 |
| Stop hit — per-position SL triggered | 2026-05-08 09:35:00 | 197.83 | 198.02 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2025-07-14 11:15:00 | 253.70 | 2025-07-14 11:50:00 | 252.90 | PARTIAL | 0.50 | 0.32% |
| SELL | retest1 | 2025-07-14 11:15:00 | 253.70 | 2025-07-14 13:50:00 | 253.70 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-07-15 10:30:00 | 255.75 | 2025-07-15 10:35:00 | 256.83 | PARTIAL | 0.50 | 0.42% |
| BUY | retest1 | 2025-07-15 10:30:00 | 255.75 | 2025-07-15 14:50:00 | 256.90 | TARGET_HIT | 0.50 | 0.45% |
| SELL | retest1 | 2025-07-23 10:15:00 | 258.70 | 2025-07-23 10:40:00 | 259.20 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest1 | 2025-08-06 11:10:00 | 241.19 | 2025-08-06 12:20:00 | 241.66 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest1 | 2025-08-11 11:05:00 | 240.84 | 2025-08-11 11:10:00 | 240.45 | STOP_HIT | 1.00 | -0.16% |
| BUY | retest1 | 2025-08-12 09:30:00 | 244.70 | 2025-08-12 09:45:00 | 245.71 | PARTIAL | 0.50 | 0.41% |
| BUY | retest1 | 2025-08-12 09:30:00 | 244.70 | 2025-08-12 10:15:00 | 244.70 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-08-14 10:30:00 | 246.69 | 2025-08-14 10:40:00 | 246.06 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2025-08-20 09:50:00 | 247.58 | 2025-08-20 10:00:00 | 248.50 | PARTIAL | 0.50 | 0.37% |
| BUY | retest1 | 2025-08-20 09:50:00 | 247.58 | 2025-08-20 15:20:00 | 250.74 | TARGET_HIT | 0.50 | 1.28% |
| BUY | retest1 | 2025-08-25 09:30:00 | 255.14 | 2025-08-25 09:45:00 | 256.30 | PARTIAL | 0.50 | 0.45% |
| BUY | retest1 | 2025-08-25 09:30:00 | 255.14 | 2025-08-25 10:00:00 | 255.14 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-08-26 09:35:00 | 254.03 | 2025-08-26 10:00:00 | 253.49 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2025-09-02 11:10:00 | 251.64 | 2025-09-02 11:45:00 | 252.21 | PARTIAL | 0.50 | 0.22% |
| BUY | retest1 | 2025-09-02 11:10:00 | 251.64 | 2025-09-02 13:50:00 | 251.64 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-09-05 10:10:00 | 242.80 | 2025-09-05 10:15:00 | 241.64 | PARTIAL | 0.50 | 0.48% |
| SELL | retest1 | 2025-09-05 10:10:00 | 242.80 | 2025-09-05 10:20:00 | 242.80 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-09-12 09:30:00 | 252.30 | 2025-09-12 09:40:00 | 251.44 | PARTIAL | 0.50 | 0.34% |
| SELL | retest1 | 2025-09-12 09:30:00 | 252.30 | 2025-09-12 11:25:00 | 252.30 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-09-15 10:45:00 | 249.94 | 2025-09-15 11:05:00 | 250.27 | STOP_HIT | 1.00 | -0.13% |
| SELL | retest1 | 2025-09-18 11:10:00 | 256.42 | 2025-09-18 12:20:00 | 256.94 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2025-09-24 11:05:00 | 245.02 | 2025-09-24 15:20:00 | 244.65 | TARGET_HIT | 1.00 | 0.15% |
| BUY | retest1 | 2025-11-10 10:20:00 | 239.17 | 2025-11-10 10:25:00 | 239.78 | PARTIAL | 0.50 | 0.25% |
| BUY | retest1 | 2025-11-10 10:20:00 | 239.17 | 2025-11-10 15:15:00 | 239.80 | TARGET_HIT | 0.50 | 0.26% |
| BUY | retest1 | 2025-11-12 10:50:00 | 245.33 | 2025-11-12 11:00:00 | 244.82 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2025-11-19 10:05:00 | 243.30 | 2025-11-19 10:30:00 | 244.04 | PARTIAL | 0.50 | 0.30% |
| BUY | retest1 | 2025-11-19 10:05:00 | 243.30 | 2025-11-19 15:20:00 | 246.15 | TARGET_HIT | 0.50 | 1.17% |
| BUY | retest1 | 2025-11-26 09:35:00 | 248.17 | 2025-11-26 09:50:00 | 249.12 | PARTIAL | 0.50 | 0.38% |
| BUY | retest1 | 2025-11-26 09:35:00 | 248.17 | 2025-11-26 09:55:00 | 248.17 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-11-27 09:55:00 | 251.10 | 2025-11-27 10:00:00 | 251.91 | PARTIAL | 0.50 | 0.32% |
| BUY | retest1 | 2025-11-27 09:55:00 | 251.10 | 2025-11-27 10:05:00 | 251.10 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-12-03 10:05:00 | 254.53 | 2025-12-03 10:25:00 | 255.54 | PARTIAL | 0.50 | 0.40% |
| BUY | retest1 | 2025-12-03 10:05:00 | 254.53 | 2025-12-03 13:15:00 | 255.02 | TARGET_HIT | 0.50 | 0.19% |
| BUY | retest1 | 2025-12-04 10:20:00 | 258.12 | 2025-12-04 10:25:00 | 257.58 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2025-12-16 11:00:00 | 260.14 | 2025-12-16 11:45:00 | 260.55 | STOP_HIT | 1.00 | -0.16% |
| BUY | retest1 | 2025-12-17 09:30:00 | 260.84 | 2025-12-17 09:35:00 | 261.52 | PARTIAL | 0.50 | 0.26% |
| BUY | retest1 | 2025-12-17 09:30:00 | 260.84 | 2025-12-17 09:45:00 | 260.84 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-01-02 09:40:00 | 268.65 | 2026-01-02 10:30:00 | 269.58 | PARTIAL | 0.50 | 0.34% |
| BUY | retest1 | 2026-01-02 09:40:00 | 268.65 | 2026-01-02 13:45:00 | 269.85 | TARGET_HIT | 0.50 | 0.45% |
| BUY | retest1 | 2026-02-01 11:05:00 | 237.87 | 2026-02-01 11:40:00 | 237.42 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest1 | 2026-02-02 10:20:00 | 240.38 | 2026-02-02 10:35:00 | 241.42 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest1 | 2026-02-10 10:05:00 | 230.40 | 2026-02-10 10:15:00 | 231.19 | PARTIAL | 0.50 | 0.34% |
| BUY | retest1 | 2026-02-10 10:05:00 | 230.40 | 2026-02-10 15:20:00 | 231.29 | TARGET_HIT | 0.50 | 0.39% |
| SELL | retest1 | 2026-02-11 10:35:00 | 230.60 | 2026-02-11 10:40:00 | 231.04 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest1 | 2026-02-17 10:05:00 | 216.92 | 2026-02-17 10:25:00 | 216.12 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest1 | 2026-02-18 09:30:00 | 213.04 | 2026-02-18 09:45:00 | 211.90 | PARTIAL | 0.50 | 0.54% |
| SELL | retest1 | 2026-02-18 09:30:00 | 213.04 | 2026-02-18 15:00:00 | 211.94 | TARGET_HIT | 0.50 | 0.52% |
| BUY | retest1 | 2026-02-20 10:55:00 | 211.00 | 2026-02-20 12:55:00 | 210.39 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2026-02-23 10:30:00 | 205.32 | 2026-02-23 10:45:00 | 204.37 | PARTIAL | 0.50 | 0.46% |
| SELL | retest1 | 2026-02-23 10:30:00 | 205.32 | 2026-02-23 11:05:00 | 205.32 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-02-24 09:30:00 | 200.93 | 2026-02-24 09:40:00 | 199.89 | PARTIAL | 0.50 | 0.52% |
| SELL | retest1 | 2026-02-24 09:30:00 | 200.93 | 2026-02-24 09:45:00 | 200.93 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-03-09 10:50:00 | 196.14 | 2026-03-09 10:55:00 | 195.53 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2026-03-16 11:00:00 | 194.05 | 2026-03-16 11:15:00 | 194.68 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2026-04-21 10:45:00 | 205.15 | 2026-04-21 10:50:00 | 204.79 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest1 | 2026-04-24 09:45:00 | 199.37 | 2026-04-24 10:15:00 | 198.59 | PARTIAL | 0.50 | 0.39% |
| SELL | retest1 | 2026-04-24 09:45:00 | 199.37 | 2026-04-24 11:05:00 | 199.37 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-30 09:30:00 | 201.90 | 2026-04-30 10:00:00 | 201.42 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2026-05-08 09:30:00 | 197.43 | 2026-05-08 09:35:00 | 197.83 | STOP_HIT | 1.00 | -0.20% |
