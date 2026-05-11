# Redington Ltd. (REDINGTON)

## Backtest Summary

- **Window:** 2025-08-11 09:15:00 → 2026-05-08 15:25:00 (13588 bars)
- **Last close:** 223.29
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
| ENTRY1 | 32 |
| ENTRY2 | 0 |
| PARTIAL | 13 |
| TARGET_HIT | 7 |
| STOP_HIT | 25 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 45 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 20 / 25
- **Target hits / Stop hits / Partials:** 7 / 25 / 13
- **Avg / median % per leg:** 0.13% / 0.00%
- **Sum % (uncompounded):** 5.98%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 16 | 6 | 37.5% | 2 | 10 | 4 | 0.10% | 1.5% |
| BUY @ 2nd Alert (retest1) | 16 | 6 | 37.5% | 2 | 10 | 4 | 0.10% | 1.5% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 29 | 14 | 48.3% | 5 | 15 | 9 | 0.15% | 4.4% |
| SELL @ 2nd Alert (retest1) | 29 | 14 | 48.3% | 5 | 15 | 9 | 0.15% | 4.4% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 45 | 20 | 44.4% | 7 | 25 | 13 | 0.13% | 6.0% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-08-14 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-14 09:30:00 | 237.35 | 239.28 | 0.00 | ORB-short ORB[238.25,241.35] vol=1.6x ATR=0.81 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-14 09:50:00 | 236.14 | 237.82 | 0.00 | T1 1.5R @ 236.14 |
| Target hit | 2025-08-14 11:30:00 | 237.15 | 236.78 | 0.00 | Trail-exit close>VWAP |

### Cycle 2 — SELL (started 2025-08-18 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-18 09:30:00 | 236.50 | 238.18 | 0.00 | ORB-short ORB[237.10,240.10] vol=1.5x ATR=0.77 |
| Stop hit — per-position SL triggered | 2025-08-18 10:00:00 | 237.27 | 237.34 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2025-08-22 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-22 09:30:00 | 240.15 | 240.90 | 0.00 | ORB-short ORB[240.50,243.55] vol=2.6x ATR=0.94 |
| Stop hit — per-position SL triggered | 2025-08-22 09:35:00 | 241.09 | 240.77 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2025-08-25 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-25 10:55:00 | 242.90 | 243.12 | 0.00 | ORB-short ORB[243.00,245.15] vol=3.5x ATR=0.79 |
| Stop hit — per-position SL triggered | 2025-08-25 11:05:00 | 243.69 | 243.11 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2025-08-26 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-26 09:30:00 | 240.60 | 242.33 | 0.00 | ORB-short ORB[241.80,244.00] vol=1.7x ATR=0.90 |
| Stop hit — per-position SL triggered | 2025-08-26 09:40:00 | 241.50 | 241.88 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2025-08-29 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-29 10:20:00 | 237.80 | 235.63 | 0.00 | ORB-long ORB[233.05,236.00] vol=1.9x ATR=0.99 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-29 10:35:00 | 239.29 | 236.18 | 0.00 | T1 1.5R @ 239.29 |
| Target hit | 2025-08-29 15:20:00 | 241.05 | 239.91 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 7 — SELL (started 2025-09-02 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-02 09:30:00 | 243.03 | 244.33 | 0.00 | ORB-short ORB[243.59,246.80] vol=1.6x ATR=1.03 |
| Stop hit — per-position SL triggered | 2025-09-02 09:50:00 | 244.06 | 243.99 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2025-09-05 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-05 09:40:00 | 240.23 | 241.68 | 0.00 | ORB-short ORB[241.00,244.00] vol=1.9x ATR=0.97 |
| Stop hit — per-position SL triggered | 2025-09-05 09:50:00 | 241.20 | 241.61 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2025-09-09 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-09 10:25:00 | 240.10 | 241.36 | 0.00 | ORB-short ORB[240.15,243.33] vol=1.5x ATR=0.85 |
| Stop hit — per-position SL triggered | 2025-09-09 10:55:00 | 240.95 | 241.07 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2025-09-10 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-10 10:10:00 | 248.44 | 246.58 | 0.00 | ORB-long ORB[244.50,247.29] vol=1.6x ATR=1.03 |
| Stop hit — per-position SL triggered | 2025-09-10 10:30:00 | 247.41 | 246.98 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2025-09-12 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-12 09:45:00 | 246.49 | 244.75 | 0.00 | ORB-long ORB[242.20,245.60] vol=2.2x ATR=0.85 |
| Stop hit — per-position SL triggered | 2025-09-12 09:50:00 | 245.64 | 244.81 | 0.00 | SL hit |

### Cycle 12 — SELL (started 2025-09-15 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-15 09:30:00 | 241.03 | 242.50 | 0.00 | ORB-short ORB[242.50,245.35] vol=4.5x ATR=0.68 |
| Stop hit — per-position SL triggered | 2025-09-15 09:55:00 | 241.71 | 242.15 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2025-10-03 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-03 09:35:00 | 282.20 | 280.11 | 0.00 | ORB-long ORB[277.70,281.55] vol=2.2x ATR=1.38 |
| Stop hit — per-position SL triggered | 2025-10-03 09:40:00 | 280.82 | 280.14 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2025-10-07 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-07 10:55:00 | 279.40 | 277.59 | 0.00 | ORB-long ORB[276.60,279.20] vol=2.2x ATR=0.81 |
| Stop hit — per-position SL triggered | 2025-10-07 11:00:00 | 278.59 | 277.68 | 0.00 | SL hit |

### Cycle 15 — SELL (started 2025-10-30 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-30 09:30:00 | 261.20 | 262.27 | 0.00 | ORB-short ORB[261.55,265.00] vol=2.0x ATR=0.80 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-30 10:00:00 | 259.99 | 261.51 | 0.00 | T1 1.5R @ 259.99 |
| Stop hit — per-position SL triggered | 2025-10-30 10:25:00 | 261.20 | 261.23 | 0.00 | SL hit |

### Cycle 16 — SELL (started 2025-10-31 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-31 09:35:00 | 257.70 | 259.44 | 0.00 | ORB-short ORB[258.10,261.80] vol=2.6x ATR=0.94 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-31 10:05:00 | 256.29 | 258.59 | 0.00 | T1 1.5R @ 256.29 |
| Target hit | 2025-10-31 13:55:00 | 256.80 | 256.57 | 0.00 | Trail-exit close>VWAP |

### Cycle 17 — BUY (started 2025-11-14 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-14 09:35:00 | 297.70 | 295.90 | 0.00 | ORB-long ORB[294.20,296.50] vol=1.7x ATR=1.14 |
| Stop hit — per-position SL triggered | 2025-11-14 09:40:00 | 296.56 | 296.01 | 0.00 | SL hit |

### Cycle 18 — BUY (started 2025-11-27 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-27 09:35:00 | 289.60 | 288.36 | 0.00 | ORB-long ORB[287.15,289.20] vol=2.5x ATR=0.85 |
| Stop hit — per-position SL triggered | 2025-11-27 09:40:00 | 288.75 | 288.35 | 0.00 | SL hit |

### Cycle 19 — SELL (started 2025-12-18 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-18 10:45:00 | 270.20 | 271.12 | 0.00 | ORB-short ORB[271.20,274.65] vol=1.5x ATR=0.75 |
| Stop hit — per-position SL triggered | 2025-12-18 11:10:00 | 270.95 | 270.98 | 0.00 | SL hit |

### Cycle 20 — SELL (started 2025-12-19 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-19 11:00:00 | 268.25 | 270.08 | 0.00 | ORB-short ORB[268.50,272.00] vol=2.7x ATR=0.67 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-19 11:30:00 | 267.24 | 269.78 | 0.00 | T1 1.5R @ 267.24 |
| Stop hit — per-position SL triggered | 2025-12-19 12:20:00 | 268.25 | 269.29 | 0.00 | SL hit |

### Cycle 21 — BUY (started 2025-12-24 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-24 10:30:00 | 276.50 | 274.92 | 0.00 | ORB-long ORB[273.75,276.20] vol=1.9x ATR=0.85 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-24 10:45:00 | 277.77 | 276.29 | 0.00 | T1 1.5R @ 277.77 |
| Stop hit — per-position SL triggered | 2025-12-24 11:20:00 | 276.50 | 276.69 | 0.00 | SL hit |

### Cycle 22 — SELL (started 2025-12-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-26 11:15:00 | 273.00 | 273.68 | 0.00 | ORB-short ORB[273.10,275.45] vol=4.8x ATR=0.63 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-26 11:50:00 | 272.06 | 273.50 | 0.00 | T1 1.5R @ 272.06 |
| Target hit | 2025-12-26 15:20:00 | 270.35 | 272.41 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 23 — BUY (started 2025-12-29 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-29 09:30:00 | 273.85 | 271.50 | 0.00 | ORB-long ORB[269.95,272.40] vol=1.6x ATR=0.80 |
| Stop hit — per-position SL triggered | 2025-12-29 09:45:00 | 273.05 | 272.07 | 0.00 | SL hit |

### Cycle 24 — SELL (started 2025-12-30 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-30 09:40:00 | 265.30 | 266.87 | 0.00 | ORB-short ORB[266.50,268.90] vol=2.1x ATR=0.84 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-30 10:00:00 | 264.04 | 266.02 | 0.00 | T1 1.5R @ 264.04 |
| Stop hit — per-position SL triggered | 2025-12-30 10:15:00 | 265.30 | 265.67 | 0.00 | SL hit |

### Cycle 25 — BUY (started 2026-01-01 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-01 11:05:00 | 275.00 | 273.53 | 0.00 | ORB-long ORB[269.15,273.15] vol=1.7x ATR=0.74 |
| Stop hit — per-position SL triggered | 2026-01-01 12:15:00 | 274.26 | 273.91 | 0.00 | SL hit |

### Cycle 26 — SELL (started 2026-01-29 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-29 10:55:00 | 257.55 | 258.55 | 0.00 | ORB-short ORB[258.65,262.25] vol=1.8x ATR=1.06 |
| Stop hit — per-position SL triggered | 2026-01-29 11:05:00 | 258.61 | 258.52 | 0.00 | SL hit |

### Cycle 27 — BUY (started 2026-01-30 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-30 09:30:00 | 262.35 | 260.60 | 0.00 | ORB-long ORB[259.10,261.50] vol=1.6x ATR=1.16 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-30 09:45:00 | 264.09 | 261.55 | 0.00 | T1 1.5R @ 264.09 |
| Stop hit — per-position SL triggered | 2026-01-30 10:10:00 | 262.35 | 261.96 | 0.00 | SL hit |

### Cycle 28 — BUY (started 2026-02-17 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-17 10:25:00 | 259.05 | 257.01 | 0.00 | ORB-long ORB[254.50,256.90] vol=2.0x ATR=0.81 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-17 10:35:00 | 260.26 | 257.56 | 0.00 | T1 1.5R @ 260.26 |
| Target hit | 2026-02-17 15:20:00 | 260.95 | 259.30 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 29 — SELL (started 2026-02-19 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-19 11:05:00 | 257.45 | 258.36 | 0.00 | ORB-short ORB[257.95,259.85] vol=3.5x ATR=0.61 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-19 11:50:00 | 256.53 | 258.16 | 0.00 | T1 1.5R @ 256.53 |
| Target hit | 2026-02-19 15:20:00 | 254.85 | 256.26 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 30 — SELL (started 2026-02-23 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-23 09:40:00 | 250.80 | 251.79 | 0.00 | ORB-short ORB[250.85,254.00] vol=1.5x ATR=1.00 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-23 09:50:00 | 249.31 | 251.37 | 0.00 | T1 1.5R @ 249.31 |
| Target hit | 2026-02-23 15:20:00 | 246.35 | 247.41 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 31 — SELL (started 2026-02-24 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-24 09:30:00 | 241.80 | 243.41 | 0.00 | ORB-short ORB[243.00,246.50] vol=2.4x ATR=1.03 |
| Stop hit — per-position SL triggered | 2026-02-24 09:45:00 | 242.83 | 243.07 | 0.00 | SL hit |

### Cycle 32 — SELL (started 2026-04-29 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-29 10:20:00 | 212.76 | 214.50 | 0.00 | ORB-short ORB[215.40,217.50] vol=2.0x ATR=0.81 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-29 10:25:00 | 211.54 | 214.12 | 0.00 | T1 1.5R @ 211.54 |
| Stop hit — per-position SL triggered | 2026-04-29 10:50:00 | 212.76 | 213.72 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2025-08-14 09:30:00 | 237.35 | 2025-08-14 09:50:00 | 236.14 | PARTIAL | 0.50 | 0.51% |
| SELL | retest1 | 2025-08-14 09:30:00 | 237.35 | 2025-08-14 11:30:00 | 237.15 | TARGET_HIT | 0.50 | 0.08% |
| SELL | retest1 | 2025-08-18 09:30:00 | 236.50 | 2025-08-18 10:00:00 | 237.27 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2025-08-22 09:30:00 | 240.15 | 2025-08-22 09:35:00 | 241.09 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest1 | 2025-08-25 10:55:00 | 242.90 | 2025-08-25 11:05:00 | 243.69 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2025-08-26 09:30:00 | 240.60 | 2025-08-26 09:40:00 | 241.50 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2025-08-29 10:20:00 | 237.80 | 2025-08-29 10:35:00 | 239.29 | PARTIAL | 0.50 | 0.63% |
| BUY | retest1 | 2025-08-29 10:20:00 | 237.80 | 2025-08-29 15:20:00 | 241.05 | TARGET_HIT | 0.50 | 1.37% |
| SELL | retest1 | 2025-09-02 09:30:00 | 243.03 | 2025-09-02 09:50:00 | 244.06 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest1 | 2025-09-05 09:40:00 | 240.23 | 2025-09-05 09:50:00 | 241.20 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest1 | 2025-09-09 10:25:00 | 240.10 | 2025-09-09 10:55:00 | 240.95 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2025-09-10 10:10:00 | 248.44 | 2025-09-10 10:30:00 | 247.41 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest1 | 2025-09-12 09:45:00 | 246.49 | 2025-09-12 09:50:00 | 245.64 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2025-09-15 09:30:00 | 241.03 | 2025-09-15 09:55:00 | 241.71 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2025-10-03 09:35:00 | 282.20 | 2025-10-03 09:40:00 | 280.82 | STOP_HIT | 1.00 | -0.49% |
| BUY | retest1 | 2025-10-07 10:55:00 | 279.40 | 2025-10-07 11:00:00 | 278.59 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2025-10-30 09:30:00 | 261.20 | 2025-10-30 10:00:00 | 259.99 | PARTIAL | 0.50 | 0.46% |
| SELL | retest1 | 2025-10-30 09:30:00 | 261.20 | 2025-10-30 10:25:00 | 261.20 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-10-31 09:35:00 | 257.70 | 2025-10-31 10:05:00 | 256.29 | PARTIAL | 0.50 | 0.55% |
| SELL | retest1 | 2025-10-31 09:35:00 | 257.70 | 2025-10-31 13:55:00 | 256.80 | TARGET_HIT | 0.50 | 0.35% |
| BUY | retest1 | 2025-11-14 09:35:00 | 297.70 | 2025-11-14 09:40:00 | 296.56 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2025-11-27 09:35:00 | 289.60 | 2025-11-27 09:40:00 | 288.75 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2025-12-18 10:45:00 | 270.20 | 2025-12-18 11:10:00 | 270.95 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2025-12-19 11:00:00 | 268.25 | 2025-12-19 11:30:00 | 267.24 | PARTIAL | 0.50 | 0.38% |
| SELL | retest1 | 2025-12-19 11:00:00 | 268.25 | 2025-12-19 12:20:00 | 268.25 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-12-24 10:30:00 | 276.50 | 2025-12-24 10:45:00 | 277.77 | PARTIAL | 0.50 | 0.46% |
| BUY | retest1 | 2025-12-24 10:30:00 | 276.50 | 2025-12-24 11:20:00 | 276.50 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-12-26 11:15:00 | 273.00 | 2025-12-26 11:50:00 | 272.06 | PARTIAL | 0.50 | 0.35% |
| SELL | retest1 | 2025-12-26 11:15:00 | 273.00 | 2025-12-26 15:20:00 | 270.35 | TARGET_HIT | 0.50 | 0.97% |
| BUY | retest1 | 2025-12-29 09:30:00 | 273.85 | 2025-12-29 09:45:00 | 273.05 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2025-12-30 09:40:00 | 265.30 | 2025-12-30 10:00:00 | 264.04 | PARTIAL | 0.50 | 0.47% |
| SELL | retest1 | 2025-12-30 09:40:00 | 265.30 | 2025-12-30 10:15:00 | 265.30 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-01-01 11:05:00 | 275.00 | 2026-01-01 12:15:00 | 274.26 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2026-01-29 10:55:00 | 257.55 | 2026-01-29 11:05:00 | 258.61 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest1 | 2026-01-30 09:30:00 | 262.35 | 2026-01-30 09:45:00 | 264.09 | PARTIAL | 0.50 | 0.66% |
| BUY | retest1 | 2026-01-30 09:30:00 | 262.35 | 2026-01-30 10:10:00 | 262.35 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-17 10:25:00 | 259.05 | 2026-02-17 10:35:00 | 260.26 | PARTIAL | 0.50 | 0.47% |
| BUY | retest1 | 2026-02-17 10:25:00 | 259.05 | 2026-02-17 15:20:00 | 260.95 | TARGET_HIT | 0.50 | 0.73% |
| SELL | retest1 | 2026-02-19 11:05:00 | 257.45 | 2026-02-19 11:50:00 | 256.53 | PARTIAL | 0.50 | 0.36% |
| SELL | retest1 | 2026-02-19 11:05:00 | 257.45 | 2026-02-19 15:20:00 | 254.85 | TARGET_HIT | 0.50 | 1.01% |
| SELL | retest1 | 2026-02-23 09:40:00 | 250.80 | 2026-02-23 09:50:00 | 249.31 | PARTIAL | 0.50 | 0.60% |
| SELL | retest1 | 2026-02-23 09:40:00 | 250.80 | 2026-02-23 15:20:00 | 246.35 | TARGET_HIT | 0.50 | 1.77% |
| SELL | retest1 | 2026-02-24 09:30:00 | 241.80 | 2026-02-24 09:45:00 | 242.83 | STOP_HIT | 1.00 | -0.43% |
| SELL | retest1 | 2026-04-29 10:20:00 | 212.76 | 2026-04-29 10:25:00 | 211.54 | PARTIAL | 0.50 | 0.57% |
| SELL | retest1 | 2026-04-29 10:20:00 | 212.76 | 2026-04-29 10:50:00 | 212.76 | STOP_HIT | 0.50 | 0.00% |
