# Mahindra & Mahindra Financial Services Ltd. (M&MFIN)

## Backtest Summary

- **Window:** 2025-05-12 09:15:00 → 2026-04-02 15:25:00 (15300 bars)
- **Last close:** 285.00
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
| ENTRY1 | 62 |
| ENTRY2 | 0 |
| PARTIAL | 26 |
| TARGET_HIT | 13 |
| STOP_HIT | 49 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 88 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 39 / 49
- **Target hits / Stop hits / Partials:** 13 / 49 / 26
- **Avg / median % per leg:** 0.17% / 0.00%
- **Sum % (uncompounded):** 15.13%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 45 | 24 | 53.3% | 11 | 21 | 13 | 0.28% | 12.6% |
| BUY @ 2nd Alert (retest1) | 45 | 24 | 53.3% | 11 | 21 | 13 | 0.28% | 12.6% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 43 | 15 | 34.9% | 2 | 28 | 13 | 0.06% | 2.6% |
| SELL @ 2nd Alert (retest1) | 43 | 15 | 34.9% | 2 | 28 | 13 | 0.06% | 2.6% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 88 | 39 | 44.3% | 13 | 49 | 26 | 0.17% | 15.1% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-16 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-16 10:40:00 | 267.60 | 266.28 | 0.00 | ORB-long ORB[264.60,267.30] vol=1.6x ATR=0.78 |
| Stop hit — per-position SL triggered | 2025-05-16 11:05:00 | 266.82 | 266.43 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2025-05-20 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-20 10:50:00 | 261.90 | 263.24 | 0.00 | ORB-short ORB[263.60,266.30] vol=4.8x ATR=0.77 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-20 11:10:00 | 260.75 | 263.02 | 0.00 | T1 1.5R @ 260.75 |
| Target hit | 2025-05-20 15:20:00 | 258.25 | 259.91 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 3 — BUY (started 2025-05-21 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-21 10:35:00 | 260.15 | 258.23 | 0.00 | ORB-long ORB[255.80,258.90] vol=2.0x ATR=0.74 |
| Stop hit — per-position SL triggered | 2025-05-21 10:55:00 | 259.41 | 258.57 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2025-05-23 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-23 10:40:00 | 255.00 | 256.59 | 0.00 | ORB-short ORB[255.85,257.35] vol=2.6x ATR=0.68 |
| Stop hit — per-position SL triggered | 2025-05-23 11:45:00 | 255.68 | 255.75 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2025-05-27 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-27 10:55:00 | 255.00 | 255.06 | 0.00 | ORB-short ORB[255.70,258.80] vol=5.0x ATR=0.77 |
| Stop hit — per-position SL triggered | 2025-05-27 11:10:00 | 255.77 | 255.09 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2025-05-28 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-28 09:40:00 | 261.65 | 260.46 | 0.00 | ORB-long ORB[258.50,260.35] vol=1.6x ATR=0.63 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-28 09:45:00 | 262.59 | 260.86 | 0.00 | T1 1.5R @ 262.59 |
| Target hit | 2025-05-28 14:10:00 | 263.45 | 263.53 | 0.00 | Trail-exit close<VWAP |

### Cycle 7 — SELL (started 2025-05-30 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-30 10:10:00 | 262.50 | 263.13 | 0.00 | ORB-short ORB[263.00,265.00] vol=1.6x ATR=0.77 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-30 10:35:00 | 261.34 | 262.84 | 0.00 | T1 1.5R @ 261.34 |
| Stop hit — per-position SL triggered | 2025-05-30 11:00:00 | 262.50 | 262.69 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2025-06-04 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-04 11:05:00 | 262.20 | 262.57 | 0.00 | ORB-short ORB[262.90,264.50] vol=2.5x ATR=0.57 |
| Stop hit — per-position SL triggered | 2025-06-04 11:55:00 | 262.77 | 262.45 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2025-06-06 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-06 10:05:00 | 264.10 | 262.34 | 0.00 | ORB-long ORB[260.95,263.70] vol=4.1x ATR=0.73 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-06 10:10:00 | 265.20 | 263.83 | 0.00 | T1 1.5R @ 265.20 |
| Stop hit — per-position SL triggered | 2025-06-06 10:15:00 | 264.10 | 264.14 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2025-06-11 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-11 10:50:00 | 282.30 | 281.33 | 0.00 | ORB-long ORB[279.25,281.00] vol=2.9x ATR=0.67 |
| Stop hit — per-position SL triggered | 2025-06-11 11:25:00 | 281.63 | 281.48 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2025-06-12 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-12 09:35:00 | 279.85 | 280.26 | 0.00 | ORB-short ORB[280.00,282.45] vol=3.7x ATR=0.80 |
| Stop hit — per-position SL triggered | 2025-06-12 09:50:00 | 280.65 | 280.17 | 0.00 | SL hit |

### Cycle 12 — SELL (started 2025-06-13 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-13 11:10:00 | 268.10 | 269.57 | 0.00 | ORB-short ORB[268.30,271.25] vol=3.5x ATR=0.70 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-13 11:15:00 | 267.05 | 269.42 | 0.00 | T1 1.5R @ 267.05 |
| Stop hit — per-position SL triggered | 2025-06-13 11:25:00 | 268.10 | 269.38 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2025-06-16 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-16 10:55:00 | 272.15 | 269.98 | 0.00 | ORB-long ORB[270.00,271.80] vol=1.6x ATR=0.77 |
| Stop hit — per-position SL triggered | 2025-06-16 11:30:00 | 271.38 | 270.46 | 0.00 | SL hit |

### Cycle 14 — SELL (started 2025-06-17 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-17 10:45:00 | 271.90 | 273.26 | 0.00 | ORB-short ORB[272.50,274.80] vol=1.7x ATR=0.67 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-17 11:05:00 | 270.90 | 272.72 | 0.00 | T1 1.5R @ 270.90 |
| Stop hit — per-position SL triggered | 2025-06-17 11:20:00 | 271.90 | 272.66 | 0.00 | SL hit |

### Cycle 15 — BUY (started 2025-06-18 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-18 10:20:00 | 270.00 | 269.61 | 0.00 | ORB-long ORB[267.60,269.50] vol=1.8x ATR=0.54 |
| Stop hit — per-position SL triggered | 2025-06-18 10:25:00 | 269.46 | 269.57 | 0.00 | SL hit |

### Cycle 16 — BUY (started 2025-06-20 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-20 10:30:00 | 263.65 | 262.15 | 0.00 | ORB-long ORB[260.50,263.35] vol=1.9x ATR=0.73 |
| Stop hit — per-position SL triggered | 2025-06-20 11:45:00 | 262.92 | 262.74 | 0.00 | SL hit |

### Cycle 17 — BUY (started 2025-06-24 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-24 09:45:00 | 266.30 | 264.82 | 0.00 | ORB-long ORB[263.55,265.00] vol=2.2x ATR=0.84 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-24 11:05:00 | 267.56 | 265.87 | 0.00 | T1 1.5R @ 267.56 |
| Target hit | 2025-06-24 13:40:00 | 267.35 | 267.35 | 0.00 | Trail-exit close<VWAP |

### Cycle 18 — BUY (started 2025-06-26 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-26 09:45:00 | 268.90 | 268.00 | 0.00 | ORB-long ORB[266.65,268.30] vol=2.0x ATR=0.55 |
| Stop hit — per-position SL triggered | 2025-06-26 10:00:00 | 268.35 | 268.12 | 0.00 | SL hit |

### Cycle 19 — BUY (started 2025-06-27 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-27 10:55:00 | 272.50 | 270.38 | 0.00 | ORB-long ORB[268.60,271.75] vol=2.4x ATR=0.69 |
| Stop hit — per-position SL triggered | 2025-06-27 11:30:00 | 271.81 | 270.78 | 0.00 | SL hit |

### Cycle 20 — SELL (started 2025-06-30 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-30 10:40:00 | 270.80 | 272.02 | 0.00 | ORB-short ORB[271.50,274.60] vol=1.6x ATR=0.68 |
| Stop hit — per-position SL triggered | 2025-06-30 11:55:00 | 271.48 | 271.52 | 0.00 | SL hit |

### Cycle 21 — SELL (started 2025-07-01 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-01 10:10:00 | 267.35 | 269.31 | 0.00 | ORB-short ORB[269.65,270.75] vol=1.8x ATR=0.68 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-01 11:45:00 | 266.32 | 268.31 | 0.00 | T1 1.5R @ 266.32 |
| Stop hit — per-position SL triggered | 2025-07-01 15:05:00 | 267.35 | 267.37 | 0.00 | SL hit |

### Cycle 22 — BUY (started 2025-07-04 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-04 09:30:00 | 270.55 | 268.80 | 0.00 | ORB-long ORB[265.90,269.55] vol=1.5x ATR=0.81 |
| Stop hit — per-position SL triggered | 2025-07-04 09:35:00 | 269.74 | 268.99 | 0.00 | SL hit |

### Cycle 23 — BUY (started 2025-07-07 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-07 09:45:00 | 268.60 | 267.11 | 0.00 | ORB-long ORB[264.85,267.75] vol=2.5x ATR=0.70 |
| Stop hit — per-position SL triggered | 2025-07-07 10:00:00 | 267.90 | 267.32 | 0.00 | SL hit |

### Cycle 24 — SELL (started 2025-07-08 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-08 10:00:00 | 266.95 | 267.92 | 0.00 | ORB-short ORB[267.00,268.90] vol=1.5x ATR=0.47 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-08 10:05:00 | 266.24 | 267.64 | 0.00 | T1 1.5R @ 266.24 |
| Stop hit — per-position SL triggered | 2025-07-08 10:10:00 | 266.95 | 267.54 | 0.00 | SL hit |

### Cycle 25 — SELL (started 2025-07-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-10 10:15:00 | 269.75 | 270.64 | 0.00 | ORB-short ORB[270.35,271.95] vol=1.6x ATR=0.55 |
| Stop hit — per-position SL triggered | 2025-07-10 10:35:00 | 270.30 | 270.56 | 0.00 | SL hit |

### Cycle 26 — SELL (started 2025-07-11 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-11 10:00:00 | 265.60 | 266.98 | 0.00 | ORB-short ORB[266.60,268.35] vol=2.4x ATR=0.63 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-11 11:00:00 | 264.66 | 265.98 | 0.00 | T1 1.5R @ 264.66 |
| Stop hit — per-position SL triggered | 2025-07-11 11:15:00 | 265.60 | 265.91 | 0.00 | SL hit |

### Cycle 27 — BUY (started 2025-07-14 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-14 10:20:00 | 267.70 | 266.36 | 0.00 | ORB-long ORB[265.05,267.45] vol=3.4x ATR=0.60 |
| Stop hit — per-position SL triggered | 2025-07-14 10:25:00 | 267.10 | 266.41 | 0.00 | SL hit |

### Cycle 28 — SELL (started 2025-07-17 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-17 10:05:00 | 263.80 | 264.33 | 0.00 | ORB-short ORB[264.25,265.45] vol=1.6x ATR=0.45 |
| Stop hit — per-position SL triggered | 2025-07-17 11:00:00 | 264.25 | 264.06 | 0.00 | SL hit |

### Cycle 29 — SELL (started 2025-07-18 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-18 10:20:00 | 259.90 | 261.34 | 0.00 | ORB-short ORB[261.60,263.00] vol=2.2x ATR=0.52 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-18 10:30:00 | 259.12 | 261.09 | 0.00 | T1 1.5R @ 259.12 |
| Stop hit — per-position SL triggered | 2025-07-18 11:35:00 | 259.90 | 260.27 | 0.00 | SL hit |

### Cycle 30 — SELL (started 2025-07-22 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-22 11:00:00 | 261.10 | 262.54 | 0.00 | ORB-short ORB[262.75,264.45] vol=1.5x ATR=0.62 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-22 11:05:00 | 260.18 | 262.44 | 0.00 | T1 1.5R @ 260.18 |
| Stop hit — per-position SL triggered | 2025-07-22 11:10:00 | 261.10 | 262.38 | 0.00 | SL hit |

### Cycle 31 — SELL (started 2025-07-24 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-24 10:50:00 | 255.75 | 257.47 | 0.00 | ORB-short ORB[258.25,260.50] vol=1.6x ATR=0.56 |
| Stop hit — per-position SL triggered | 2025-07-24 12:10:00 | 256.31 | 256.59 | 0.00 | SL hit |

### Cycle 32 — SELL (started 2025-08-04 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-04 10:40:00 | 254.85 | 256.75 | 0.00 | ORB-short ORB[257.90,259.80] vol=5.2x ATR=0.76 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-04 11:55:00 | 253.70 | 256.03 | 0.00 | T1 1.5R @ 253.70 |
| Stop hit — per-position SL triggered | 2025-08-04 12:00:00 | 254.85 | 255.99 | 0.00 | SL hit |

### Cycle 33 — SELL (started 2025-08-06 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-06 10:35:00 | 254.70 | 255.14 | 0.00 | ORB-short ORB[255.10,256.90] vol=3.5x ATR=0.64 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-06 10:45:00 | 253.74 | 255.02 | 0.00 | T1 1.5R @ 253.74 |
| Stop hit — per-position SL triggered | 2025-08-06 12:15:00 | 254.70 | 254.53 | 0.00 | SL hit |

### Cycle 34 — SELL (started 2025-08-07 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-07 11:05:00 | 251.45 | 253.07 | 0.00 | ORB-short ORB[252.00,254.75] vol=2.1x ATR=0.63 |
| Stop hit — per-position SL triggered | 2025-08-07 11:50:00 | 252.08 | 252.84 | 0.00 | SL hit |

### Cycle 35 — BUY (started 2025-08-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-08 10:15:00 | 257.25 | 256.13 | 0.00 | ORB-long ORB[253.75,256.90] vol=2.9x ATR=0.74 |
| Stop hit — per-position SL triggered | 2025-08-08 10:20:00 | 256.51 | 256.30 | 0.00 | SL hit |

### Cycle 36 — BUY (started 2025-08-12 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-12 10:55:00 | 256.70 | 255.81 | 0.00 | ORB-long ORB[253.80,256.40] vol=1.6x ATR=0.46 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-12 13:10:00 | 257.40 | 256.23 | 0.00 | T1 1.5R @ 257.40 |
| Stop hit — per-position SL triggered | 2025-08-12 15:00:00 | 256.70 | 256.50 | 0.00 | SL hit |

### Cycle 37 — BUY (started 2025-08-22 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-22 11:10:00 | 263.70 | 262.56 | 0.00 | ORB-long ORB[260.00,262.55] vol=3.0x ATR=0.69 |
| Stop hit — per-position SL triggered | 2025-08-22 11:20:00 | 263.01 | 262.78 | 0.00 | SL hit |

### Cycle 38 — BUY (started 2025-08-25 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-25 09:30:00 | 265.30 | 264.29 | 0.00 | ORB-long ORB[263.00,265.10] vol=1.7x ATR=0.70 |
| Stop hit — per-position SL triggered | 2025-08-25 09:55:00 | 264.60 | 264.68 | 0.00 | SL hit |

### Cycle 39 — SELL (started 2025-08-26 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-26 09:30:00 | 262.35 | 263.16 | 0.00 | ORB-short ORB[262.80,265.50] vol=1.9x ATR=0.63 |
| Stop hit — per-position SL triggered | 2025-08-26 09:45:00 | 262.98 | 263.01 | 0.00 | SL hit |

### Cycle 40 — SELL (started 2025-08-28 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-28 11:10:00 | 257.55 | 258.71 | 0.00 | ORB-short ORB[258.10,260.40] vol=7.9x ATR=0.64 |
| Stop hit — per-position SL triggered | 2025-08-28 12:35:00 | 258.19 | 257.81 | 0.00 | SL hit |

### Cycle 41 — SELL (started 2025-09-02 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-02 10:45:00 | 256.40 | 257.51 | 0.00 | ORB-short ORB[256.85,258.65] vol=2.5x ATR=0.59 |
| Stop hit — per-position SL triggered | 2025-09-02 11:10:00 | 256.99 | 257.51 | 0.00 | SL hit |

### Cycle 42 — BUY (started 2025-09-05 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-05 11:00:00 | 266.40 | 264.97 | 0.00 | ORB-long ORB[263.00,265.75] vol=5.3x ATR=0.80 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-05 11:10:00 | 267.60 | 266.05 | 0.00 | T1 1.5R @ 267.60 |
| Target hit | 2025-09-05 13:10:00 | 269.25 | 269.71 | 0.00 | Trail-exit close<VWAP |

### Cycle 43 — BUY (started 2025-09-16 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-16 09:50:00 | 273.65 | 272.10 | 0.00 | ORB-long ORB[269.50,272.20] vol=3.1x ATR=0.74 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-16 10:10:00 | 274.77 | 274.34 | 0.00 | T1 1.5R @ 274.77 |
| Target hit | 2025-09-16 10:55:00 | 275.45 | 275.52 | 0.00 | Trail-exit close<VWAP |

### Cycle 44 — BUY (started 2025-09-17 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-17 11:05:00 | 278.70 | 278.32 | 0.00 | ORB-long ORB[276.05,278.50] vol=2.2x ATR=0.84 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-17 11:20:00 | 279.96 | 278.44 | 0.00 | T1 1.5R @ 279.96 |
| Target hit | 2025-09-17 15:20:00 | 281.55 | 280.76 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 45 — BUY (started 2025-09-18 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-18 10:00:00 | 284.30 | 282.45 | 0.00 | ORB-long ORB[280.05,283.25] vol=2.3x ATR=1.08 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-18 10:25:00 | 285.92 | 283.53 | 0.00 | T1 1.5R @ 285.92 |
| Target hit | 2025-09-18 12:40:00 | 287.35 | 287.40 | 0.00 | Trail-exit close<VWAP |

### Cycle 46 — BUY (started 2025-09-19 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-19 09:30:00 | 288.30 | 287.86 | 0.00 | ORB-long ORB[284.60,288.15] vol=5.2x ATR=1.18 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-19 10:50:00 | 290.07 | 288.72 | 0.00 | T1 1.5R @ 290.07 |
| Target hit | 2025-09-19 14:00:00 | 289.40 | 289.66 | 0.00 | Trail-exit close<VWAP |

### Cycle 47 — SELL (started 2025-09-23 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-23 10:00:00 | 285.70 | 286.74 | 0.00 | ORB-short ORB[286.15,289.50] vol=1.5x ATR=1.15 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-23 11:30:00 | 283.97 | 286.08 | 0.00 | T1 1.5R @ 283.97 |
| Target hit | 2025-09-23 15:20:00 | 282.70 | 284.83 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 48 — BUY (started 2025-09-24 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-24 10:35:00 | 284.60 | 282.21 | 0.00 | ORB-long ORB[279.20,281.95] vol=2.5x ATR=0.92 |
| Stop hit — per-position SL triggered | 2025-09-24 10:45:00 | 283.68 | 282.45 | 0.00 | SL hit |

### Cycle 49 — SELL (started 2025-10-03 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-03 10:00:00 | 270.10 | 272.96 | 0.00 | ORB-short ORB[273.50,276.15] vol=2.6x ATR=1.09 |
| Stop hit — per-position SL triggered | 2025-10-03 10:45:00 | 271.19 | 271.83 | 0.00 | SL hit |

### Cycle 50 — SELL (started 2025-11-10 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-10 10:45:00 | 312.80 | 314.00 | 0.00 | ORB-short ORB[313.35,316.90] vol=12.6x ATR=1.22 |
| Stop hit — per-position SL triggered | 2025-11-10 11:00:00 | 314.02 | 313.99 | 0.00 | SL hit |

### Cycle 51 — SELL (started 2025-11-11 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-11 11:00:00 | 312.00 | 312.46 | 0.00 | ORB-short ORB[312.10,315.35] vol=3.9x ATR=0.87 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-11 11:20:00 | 310.69 | 312.36 | 0.00 | T1 1.5R @ 310.69 |
| Stop hit — per-position SL triggered | 2025-11-11 11:45:00 | 312.00 | 312.31 | 0.00 | SL hit |

### Cycle 52 — BUY (started 2025-11-14 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-14 11:00:00 | 311.80 | 310.67 | 0.00 | ORB-long ORB[309.00,311.55] vol=2.7x ATR=0.62 |
| Stop hit — per-position SL triggered | 2025-11-14 11:35:00 | 311.18 | 310.77 | 0.00 | SL hit |

### Cycle 53 — BUY (started 2025-11-20 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-20 10:25:00 | 337.75 | 334.70 | 0.00 | ORB-long ORB[331.45,335.75] vol=2.9x ATR=1.39 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-20 11:20:00 | 339.83 | 336.10 | 0.00 | T1 1.5R @ 339.83 |
| Target hit | 2025-11-20 15:20:00 | 346.00 | 343.89 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 54 — SELL (started 2025-11-26 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-26 10:40:00 | 355.75 | 358.13 | 0.00 | ORB-short ORB[357.70,363.05] vol=2.3x ATR=1.66 |
| Stop hit — per-position SL triggered | 2025-11-26 10:45:00 | 357.41 | 358.05 | 0.00 | SL hit |

### Cycle 55 — BUY (started 2025-12-18 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-18 10:05:00 | 356.85 | 353.65 | 0.00 | ORB-long ORB[349.15,353.75] vol=1.7x ATR=1.61 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-18 10:25:00 | 359.26 | 355.38 | 0.00 | T1 1.5R @ 359.26 |
| Target hit | 2025-12-18 15:20:00 | 364.45 | 361.40 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 56 — BUY (started 2026-01-01 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-01 11:00:00 | 403.45 | 402.37 | 0.00 | ORB-long ORB[399.00,402.50] vol=7.3x ATR=1.41 |
| Stop hit — per-position SL triggered | 2026-01-01 11:25:00 | 402.04 | 402.49 | 0.00 | SL hit |

### Cycle 57 — SELL (started 2026-01-14 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-14 09:45:00 | 354.90 | 358.02 | 0.00 | ORB-short ORB[359.00,363.80] vol=1.5x ATR=1.52 |
| Stop hit — per-position SL triggered | 2026-01-14 09:55:00 | 356.42 | 357.54 | 0.00 | SL hit |

### Cycle 58 — BUY (started 2026-01-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-23 10:15:00 | 363.25 | 361.32 | 0.00 | ORB-long ORB[357.45,362.20] vol=2.6x ATR=1.23 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-23 10:25:00 | 365.09 | 362.57 | 0.00 | T1 1.5R @ 365.09 |
| Target hit | 2026-01-23 11:15:00 | 365.40 | 365.54 | 0.00 | Trail-exit close<VWAP |

### Cycle 59 — BUY (started 2026-01-28 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-28 09:30:00 | 368.55 | 366.90 | 0.00 | ORB-long ORB[363.90,367.80] vol=1.8x ATR=1.63 |
| Stop hit — per-position SL triggered | 2026-01-28 09:45:00 | 366.92 | 367.20 | 0.00 | SL hit |

### Cycle 60 — BUY (started 2026-02-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-18 11:15:00 | 383.85 | 383.02 | 0.00 | ORB-long ORB[379.20,383.75] vol=2.1x ATR=0.98 |
| Stop hit — per-position SL triggered | 2026-02-18 11:50:00 | 382.87 | 383.07 | 0.00 | SL hit |

### Cycle 61 — SELL (started 2026-02-20 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-20 10:55:00 | 375.70 | 378.90 | 0.00 | ORB-short ORB[376.00,380.00] vol=3.6x ATR=1.63 |
| Stop hit — per-position SL triggered | 2026-02-20 12:35:00 | 377.33 | 377.55 | 0.00 | SL hit |

### Cycle 62 — BUY (started 2026-02-25 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-25 09:55:00 | 375.20 | 371.04 | 0.00 | ORB-long ORB[364.70,369.15] vol=1.7x ATR=1.51 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-25 10:30:00 | 377.46 | 372.72 | 0.00 | T1 1.5R @ 377.46 |
| Target hit | 2026-02-25 14:40:00 | 378.50 | 378.92 | 0.00 | Trail-exit close<VWAP |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-05-16 10:40:00 | 267.60 | 2025-05-16 11:05:00 | 266.82 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2025-05-20 10:50:00 | 261.90 | 2025-05-20 11:10:00 | 260.75 | PARTIAL | 0.50 | 0.44% |
| SELL | retest1 | 2025-05-20 10:50:00 | 261.90 | 2025-05-20 15:20:00 | 258.25 | TARGET_HIT | 0.50 | 1.39% |
| BUY | retest1 | 2025-05-21 10:35:00 | 260.15 | 2025-05-21 10:55:00 | 259.41 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2025-05-23 10:40:00 | 255.00 | 2025-05-23 11:45:00 | 255.68 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2025-05-27 10:55:00 | 255.00 | 2025-05-27 11:10:00 | 255.77 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2025-05-28 09:40:00 | 261.65 | 2025-05-28 09:45:00 | 262.59 | PARTIAL | 0.50 | 0.36% |
| BUY | retest1 | 2025-05-28 09:40:00 | 261.65 | 2025-05-28 14:10:00 | 263.45 | TARGET_HIT | 0.50 | 0.69% |
| SELL | retest1 | 2025-05-30 10:10:00 | 262.50 | 2025-05-30 10:35:00 | 261.34 | PARTIAL | 0.50 | 0.44% |
| SELL | retest1 | 2025-05-30 10:10:00 | 262.50 | 2025-05-30 11:00:00 | 262.50 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-06-04 11:05:00 | 262.20 | 2025-06-04 11:55:00 | 262.77 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2025-06-06 10:05:00 | 264.10 | 2025-06-06 10:10:00 | 265.20 | PARTIAL | 0.50 | 0.42% |
| BUY | retest1 | 2025-06-06 10:05:00 | 264.10 | 2025-06-06 10:15:00 | 264.10 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-06-11 10:50:00 | 282.30 | 2025-06-11 11:25:00 | 281.63 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2025-06-12 09:35:00 | 279.85 | 2025-06-12 09:50:00 | 280.65 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2025-06-13 11:10:00 | 268.10 | 2025-06-13 11:15:00 | 267.05 | PARTIAL | 0.50 | 0.39% |
| SELL | retest1 | 2025-06-13 11:10:00 | 268.10 | 2025-06-13 11:25:00 | 268.10 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-06-16 10:55:00 | 272.15 | 2025-06-16 11:30:00 | 271.38 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2025-06-17 10:45:00 | 271.90 | 2025-06-17 11:05:00 | 270.90 | PARTIAL | 0.50 | 0.37% |
| SELL | retest1 | 2025-06-17 10:45:00 | 271.90 | 2025-06-17 11:20:00 | 271.90 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-06-18 10:20:00 | 270.00 | 2025-06-18 10:25:00 | 269.46 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2025-06-20 10:30:00 | 263.65 | 2025-06-20 11:45:00 | 262.92 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2025-06-24 09:45:00 | 266.30 | 2025-06-24 11:05:00 | 267.56 | PARTIAL | 0.50 | 0.47% |
| BUY | retest1 | 2025-06-24 09:45:00 | 266.30 | 2025-06-24 13:40:00 | 267.35 | TARGET_HIT | 0.50 | 0.39% |
| BUY | retest1 | 2025-06-26 09:45:00 | 268.90 | 2025-06-26 10:00:00 | 268.35 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2025-06-27 10:55:00 | 272.50 | 2025-06-27 11:30:00 | 271.81 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2025-06-30 10:40:00 | 270.80 | 2025-06-30 11:55:00 | 271.48 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2025-07-01 10:10:00 | 267.35 | 2025-07-01 11:45:00 | 266.32 | PARTIAL | 0.50 | 0.38% |
| SELL | retest1 | 2025-07-01 10:10:00 | 267.35 | 2025-07-01 15:05:00 | 267.35 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-07-04 09:30:00 | 270.55 | 2025-07-04 09:35:00 | 269.74 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2025-07-07 09:45:00 | 268.60 | 2025-07-07 10:00:00 | 267.90 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2025-07-08 10:00:00 | 266.95 | 2025-07-08 10:05:00 | 266.24 | PARTIAL | 0.50 | 0.27% |
| SELL | retest1 | 2025-07-08 10:00:00 | 266.95 | 2025-07-08 10:10:00 | 266.95 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-07-10 10:15:00 | 269.75 | 2025-07-10 10:35:00 | 270.30 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2025-07-11 10:00:00 | 265.60 | 2025-07-11 11:00:00 | 264.66 | PARTIAL | 0.50 | 0.35% |
| SELL | retest1 | 2025-07-11 10:00:00 | 265.60 | 2025-07-11 11:15:00 | 265.60 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-07-14 10:20:00 | 267.70 | 2025-07-14 10:25:00 | 267.10 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2025-07-17 10:05:00 | 263.80 | 2025-07-17 11:00:00 | 264.25 | STOP_HIT | 1.00 | -0.17% |
| SELL | retest1 | 2025-07-18 10:20:00 | 259.90 | 2025-07-18 10:30:00 | 259.12 | PARTIAL | 0.50 | 0.30% |
| SELL | retest1 | 2025-07-18 10:20:00 | 259.90 | 2025-07-18 11:35:00 | 259.90 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-07-22 11:00:00 | 261.10 | 2025-07-22 11:05:00 | 260.18 | PARTIAL | 0.50 | 0.35% |
| SELL | retest1 | 2025-07-22 11:00:00 | 261.10 | 2025-07-22 11:10:00 | 261.10 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-07-24 10:50:00 | 255.75 | 2025-07-24 12:10:00 | 256.31 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2025-08-04 10:40:00 | 254.85 | 2025-08-04 11:55:00 | 253.70 | PARTIAL | 0.50 | 0.45% |
| SELL | retest1 | 2025-08-04 10:40:00 | 254.85 | 2025-08-04 12:00:00 | 254.85 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-08-06 10:35:00 | 254.70 | 2025-08-06 10:45:00 | 253.74 | PARTIAL | 0.50 | 0.38% |
| SELL | retest1 | 2025-08-06 10:35:00 | 254.70 | 2025-08-06 12:15:00 | 254.70 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-08-07 11:05:00 | 251.45 | 2025-08-07 11:50:00 | 252.08 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2025-08-08 10:15:00 | 257.25 | 2025-08-08 10:20:00 | 256.51 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2025-08-12 10:55:00 | 256.70 | 2025-08-12 13:10:00 | 257.40 | PARTIAL | 0.50 | 0.27% |
| BUY | retest1 | 2025-08-12 10:55:00 | 256.70 | 2025-08-12 15:00:00 | 256.70 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-08-22 11:10:00 | 263.70 | 2025-08-22 11:20:00 | 263.01 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2025-08-25 09:30:00 | 265.30 | 2025-08-25 09:55:00 | 264.60 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2025-08-26 09:30:00 | 262.35 | 2025-08-26 09:45:00 | 262.98 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2025-08-28 11:10:00 | 257.55 | 2025-08-28 12:35:00 | 258.19 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2025-09-02 10:45:00 | 256.40 | 2025-09-02 11:10:00 | 256.99 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2025-09-05 11:00:00 | 266.40 | 2025-09-05 11:10:00 | 267.60 | PARTIAL | 0.50 | 0.45% |
| BUY | retest1 | 2025-09-05 11:00:00 | 266.40 | 2025-09-05 13:10:00 | 269.25 | TARGET_HIT | 0.50 | 1.07% |
| BUY | retest1 | 2025-09-16 09:50:00 | 273.65 | 2025-09-16 10:10:00 | 274.77 | PARTIAL | 0.50 | 0.41% |
| BUY | retest1 | 2025-09-16 09:50:00 | 273.65 | 2025-09-16 10:55:00 | 275.45 | TARGET_HIT | 0.50 | 0.66% |
| BUY | retest1 | 2025-09-17 11:05:00 | 278.70 | 2025-09-17 11:20:00 | 279.96 | PARTIAL | 0.50 | 0.45% |
| BUY | retest1 | 2025-09-17 11:05:00 | 278.70 | 2025-09-17 15:20:00 | 281.55 | TARGET_HIT | 0.50 | 1.02% |
| BUY | retest1 | 2025-09-18 10:00:00 | 284.30 | 2025-09-18 10:25:00 | 285.92 | PARTIAL | 0.50 | 0.57% |
| BUY | retest1 | 2025-09-18 10:00:00 | 284.30 | 2025-09-18 12:40:00 | 287.35 | TARGET_HIT | 0.50 | 1.07% |
| BUY | retest1 | 2025-09-19 09:30:00 | 288.30 | 2025-09-19 10:50:00 | 290.07 | PARTIAL | 0.50 | 0.61% |
| BUY | retest1 | 2025-09-19 09:30:00 | 288.30 | 2025-09-19 14:00:00 | 289.40 | TARGET_HIT | 0.50 | 0.38% |
| SELL | retest1 | 2025-09-23 10:00:00 | 285.70 | 2025-09-23 11:30:00 | 283.97 | PARTIAL | 0.50 | 0.60% |
| SELL | retest1 | 2025-09-23 10:00:00 | 285.70 | 2025-09-23 15:20:00 | 282.70 | TARGET_HIT | 0.50 | 1.05% |
| BUY | retest1 | 2025-09-24 10:35:00 | 284.60 | 2025-09-24 10:45:00 | 283.68 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2025-10-03 10:00:00 | 270.10 | 2025-10-03 10:45:00 | 271.19 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest1 | 2025-11-10 10:45:00 | 312.80 | 2025-11-10 11:00:00 | 314.02 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest1 | 2025-11-11 11:00:00 | 312.00 | 2025-11-11 11:20:00 | 310.69 | PARTIAL | 0.50 | 0.42% |
| SELL | retest1 | 2025-11-11 11:00:00 | 312.00 | 2025-11-11 11:45:00 | 312.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-11-14 11:00:00 | 311.80 | 2025-11-14 11:35:00 | 311.18 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2025-11-20 10:25:00 | 337.75 | 2025-11-20 11:20:00 | 339.83 | PARTIAL | 0.50 | 0.62% |
| BUY | retest1 | 2025-11-20 10:25:00 | 337.75 | 2025-11-20 15:20:00 | 346.00 | TARGET_HIT | 0.50 | 2.44% |
| SELL | retest1 | 2025-11-26 10:40:00 | 355.75 | 2025-11-26 10:45:00 | 357.41 | STOP_HIT | 1.00 | -0.47% |
| BUY | retest1 | 2025-12-18 10:05:00 | 356.85 | 2025-12-18 10:25:00 | 359.26 | PARTIAL | 0.50 | 0.67% |
| BUY | retest1 | 2025-12-18 10:05:00 | 356.85 | 2025-12-18 15:20:00 | 364.45 | TARGET_HIT | 0.50 | 2.13% |
| BUY | retest1 | 2026-01-01 11:00:00 | 403.45 | 2026-01-01 11:25:00 | 402.04 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2026-01-14 09:45:00 | 354.90 | 2026-01-14 09:55:00 | 356.42 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest1 | 2026-01-23 10:15:00 | 363.25 | 2026-01-23 10:25:00 | 365.09 | PARTIAL | 0.50 | 0.51% |
| BUY | retest1 | 2026-01-23 10:15:00 | 363.25 | 2026-01-23 11:15:00 | 365.40 | TARGET_HIT | 0.50 | 0.59% |
| BUY | retest1 | 2026-01-28 09:30:00 | 368.55 | 2026-01-28 09:45:00 | 366.92 | STOP_HIT | 1.00 | -0.44% |
| BUY | retest1 | 2026-02-18 11:15:00 | 383.85 | 2026-02-18 11:50:00 | 382.87 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2026-02-20 10:55:00 | 375.70 | 2026-02-20 12:35:00 | 377.33 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest1 | 2026-02-25 09:55:00 | 375.20 | 2026-02-25 10:30:00 | 377.46 | PARTIAL | 0.50 | 0.60% |
| BUY | retest1 | 2026-02-25 09:55:00 | 375.20 | 2026-02-25 14:40:00 | 378.50 | TARGET_HIT | 0.50 | 0.88% |
