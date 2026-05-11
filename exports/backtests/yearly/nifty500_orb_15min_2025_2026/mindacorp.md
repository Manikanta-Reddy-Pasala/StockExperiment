# Minda Corporation Ltd. (MINDACORP)

## Backtest Summary

- **Window:** 2025-05-12 09:15:00 → 2026-05-08 15:25:00 (18463 bars)
- **Last close:** 537.80
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
| ENTRY1 | 85 |
| ENTRY2 | 0 |
| PARTIAL | 34 |
| TARGET_HIT | 17 |
| STOP_HIT | 68 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 119 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 51 / 68
- **Target hits / Stop hits / Partials:** 17 / 68 / 34
- **Avg / median % per leg:** 0.13% / 0.00%
- **Sum % (uncompounded):** 15.99%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 62 | 25 | 40.3% | 9 | 37 | 16 | 0.12% | 7.3% |
| BUY @ 2nd Alert (retest1) | 62 | 25 | 40.3% | 9 | 37 | 16 | 0.12% | 7.3% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 57 | 26 | 45.6% | 8 | 31 | 18 | 0.15% | 8.6% |
| SELL @ 2nd Alert (retest1) | 57 | 26 | 45.6% | 8 | 31 | 18 | 0.15% | 8.6% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 119 | 51 | 42.9% | 17 | 68 | 34 | 0.13% | 16.0% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-14 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-14 10:40:00 | 496.30 | 493.53 | 0.00 | ORB-long ORB[487.60,493.80] vol=1.7x ATR=1.30 |
| Stop hit — per-position SL triggered | 2025-05-14 10:55:00 | 495.00 | 493.88 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2025-05-16 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-16 09:30:00 | 509.80 | 508.04 | 0.00 | ORB-long ORB[504.30,509.50] vol=1.5x ATR=1.79 |
| Stop hit — per-position SL triggered | 2025-05-16 09:40:00 | 508.01 | 508.28 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2025-05-20 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-20 11:05:00 | 510.00 | 512.35 | 0.00 | ORB-short ORB[511.20,517.95] vol=2.5x ATR=1.69 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-20 12:05:00 | 507.47 | 511.40 | 0.00 | T1 1.5R @ 507.47 |
| Target hit | 2025-05-20 15:20:00 | 500.85 | 505.99 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 4 — BUY (started 2025-05-21 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-21 09:40:00 | 505.00 | 502.13 | 0.00 | ORB-long ORB[498.55,503.85] vol=1.8x ATR=1.69 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-21 09:45:00 | 507.54 | 502.52 | 0.00 | T1 1.5R @ 507.54 |
| Stop hit — per-position SL triggered | 2025-05-21 09:55:00 | 505.00 | 503.13 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2025-05-23 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-23 10:55:00 | 515.35 | 512.15 | 0.00 | ORB-long ORB[508.70,513.70] vol=2.5x ATR=1.52 |
| Stop hit — per-position SL triggered | 2025-05-23 11:00:00 | 513.83 | 512.37 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2025-05-30 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-30 10:50:00 | 526.30 | 529.55 | 0.00 | ORB-short ORB[530.95,535.85] vol=2.9x ATR=1.43 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-30 12:10:00 | 524.15 | 528.41 | 0.00 | T1 1.5R @ 524.15 |
| Target hit | 2025-05-30 15:20:00 | 521.65 | 526.21 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 7 — SELL (started 2025-06-04 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-04 09:50:00 | 511.60 | 512.68 | 0.00 | ORB-short ORB[512.00,515.90] vol=1.8x ATR=1.68 |
| Stop hit — per-position SL triggered | 2025-06-04 10:00:00 | 513.28 | 512.65 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2025-06-06 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-06 09:30:00 | 549.80 | 546.95 | 0.00 | ORB-long ORB[541.00,549.00] vol=4.8x ATR=2.62 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-06 09:40:00 | 553.74 | 548.52 | 0.00 | T1 1.5R @ 553.74 |
| Stop hit — per-position SL triggered | 2025-06-06 09:50:00 | 549.80 | 549.61 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2025-06-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-17 11:15:00 | 531.55 | 535.32 | 0.00 | ORB-short ORB[533.00,539.95] vol=4.2x ATR=1.19 |
| Stop hit — per-position SL triggered | 2025-06-17 11:30:00 | 532.74 | 534.88 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2025-06-23 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-23 11:00:00 | 505.00 | 501.57 | 0.00 | ORB-long ORB[498.00,504.00] vol=1.8x ATR=1.54 |
| Target hit | 2025-06-23 15:20:00 | 505.05 | 503.25 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 11 — SELL (started 2025-06-26 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-26 09:55:00 | 510.45 | 512.84 | 0.00 | ORB-short ORB[511.75,516.55] vol=2.0x ATR=1.66 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-26 10:50:00 | 507.95 | 511.06 | 0.00 | T1 1.5R @ 507.95 |
| Target hit | 2025-06-26 14:30:00 | 510.00 | 509.95 | 0.00 | Trail-exit close>VWAP |

### Cycle 12 — BUY (started 2025-06-27 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-27 09:35:00 | 519.55 | 516.38 | 0.00 | ORB-long ORB[513.05,516.80] vol=3.7x ATR=1.81 |
| Stop hit — per-position SL triggered | 2025-06-27 09:40:00 | 517.74 | 516.99 | 0.00 | SL hit |

### Cycle 13 — SELL (started 2025-07-01 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-01 09:30:00 | 515.20 | 518.70 | 0.00 | ORB-short ORB[516.45,523.50] vol=2.6x ATR=1.50 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-01 09:35:00 | 512.96 | 517.00 | 0.00 | T1 1.5R @ 512.96 |
| Stop hit — per-position SL triggered | 2025-07-01 09:45:00 | 515.20 | 515.84 | 0.00 | SL hit |

### Cycle 14 — SELL (started 2025-07-02 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-02 10:25:00 | 521.55 | 522.97 | 0.00 | ORB-short ORB[523.50,529.80] vol=4.3x ATR=2.36 |
| Stop hit — per-position SL triggered | 2025-07-02 10:35:00 | 523.91 | 522.97 | 0.00 | SL hit |

### Cycle 15 — BUY (started 2025-07-04 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-04 10:45:00 | 521.60 | 516.76 | 0.00 | ORB-long ORB[512.50,519.90] vol=3.9x ATR=1.71 |
| Stop hit — per-position SL triggered | 2025-07-04 10:50:00 | 519.89 | 516.97 | 0.00 | SL hit |

### Cycle 16 — SELL (started 2025-07-07 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-07 11:00:00 | 512.20 | 514.73 | 0.00 | ORB-short ORB[514.25,519.50] vol=4.0x ATR=1.02 |
| Stop hit — per-position SL triggered | 2025-07-07 11:15:00 | 513.22 | 514.62 | 0.00 | SL hit |

### Cycle 17 — SELL (started 2025-07-11 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-11 10:10:00 | 511.00 | 514.91 | 0.00 | ORB-short ORB[512.25,517.70] vol=3.6x ATR=1.28 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-11 10:30:00 | 509.08 | 514.17 | 0.00 | T1 1.5R @ 509.08 |
| Stop hit — per-position SL triggered | 2025-07-11 10:35:00 | 511.00 | 514.08 | 0.00 | SL hit |

### Cycle 18 — SELL (started 2025-07-16 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-16 11:05:00 | 516.25 | 517.82 | 0.00 | ORB-short ORB[516.85,522.00] vol=1.7x ATR=0.95 |
| Stop hit — per-position SL triggered | 2025-07-16 11:45:00 | 517.20 | 517.60 | 0.00 | SL hit |

### Cycle 19 — SELL (started 2025-07-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-18 10:15:00 | 530.90 | 535.47 | 0.00 | ORB-short ORB[534.60,540.25] vol=2.3x ATR=1.63 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-18 10:20:00 | 528.46 | 535.34 | 0.00 | T1 1.5R @ 528.46 |
| Stop hit — per-position SL triggered | 2025-07-18 10:45:00 | 530.90 | 533.57 | 0.00 | SL hit |

### Cycle 20 — SELL (started 2025-07-25 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-25 10:35:00 | 505.00 | 508.61 | 0.00 | ORB-short ORB[510.05,514.30] vol=6.0x ATR=1.30 |
| Stop hit — per-position SL triggered | 2025-07-25 11:05:00 | 506.30 | 506.78 | 0.00 | SL hit |

### Cycle 21 — SELL (started 2025-07-29 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-29 10:45:00 | 505.00 | 508.13 | 0.00 | ORB-short ORB[505.60,512.70] vol=1.8x ATR=1.56 |
| Stop hit — per-position SL triggered | 2025-07-29 12:10:00 | 506.56 | 507.35 | 0.00 | SL hit |

### Cycle 22 — SELL (started 2025-07-30 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-30 10:50:00 | 511.70 | 513.90 | 0.00 | ORB-short ORB[513.45,517.95] vol=1.7x ATR=1.20 |
| Stop hit — per-position SL triggered | 2025-07-30 11:25:00 | 512.90 | 513.72 | 0.00 | SL hit |

### Cycle 23 — SELL (started 2025-08-01 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-01 10:00:00 | 496.50 | 498.89 | 0.00 | ORB-short ORB[497.30,503.10] vol=2.8x ATR=1.55 |
| Stop hit — per-position SL triggered | 2025-08-01 10:30:00 | 498.05 | 498.67 | 0.00 | SL hit |

### Cycle 24 — SELL (started 2025-08-05 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-05 09:40:00 | 489.00 | 491.58 | 0.00 | ORB-short ORB[490.00,495.05] vol=1.5x ATR=1.97 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-05 10:25:00 | 486.04 | 489.53 | 0.00 | T1 1.5R @ 486.04 |
| Target hit | 2025-08-05 12:25:00 | 488.50 | 487.74 | 0.00 | Trail-exit close>VWAP |

### Cycle 25 — SELL (started 2025-08-11 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-11 09:40:00 | 452.65 | 456.46 | 0.00 | ORB-short ORB[455.35,460.50] vol=2.3x ATR=2.02 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-11 09:50:00 | 449.63 | 453.49 | 0.00 | T1 1.5R @ 449.63 |
| Stop hit — per-position SL triggered | 2025-08-11 14:05:00 | 452.65 | 449.64 | 0.00 | SL hit |

### Cycle 26 — SELL (started 2025-08-20 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-20 09:35:00 | 513.85 | 515.76 | 0.00 | ORB-short ORB[515.55,519.55] vol=1.9x ATR=1.33 |
| Stop hit — per-position SL triggered | 2025-08-20 09:40:00 | 515.18 | 515.67 | 0.00 | SL hit |

### Cycle 27 — SELL (started 2025-08-21 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-21 10:35:00 | 506.70 | 506.99 | 0.00 | ORB-short ORB[506.80,511.20] vol=2.3x ATR=1.16 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-21 12:00:00 | 504.96 | 506.51 | 0.00 | T1 1.5R @ 504.96 |
| Target hit | 2025-08-21 15:20:00 | 500.50 | 504.05 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 28 — SELL (started 2025-08-22 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-22 09:30:00 | 497.80 | 499.39 | 0.00 | ORB-short ORB[498.85,502.35] vol=2.0x ATR=1.28 |
| Stop hit — per-position SL triggered | 2025-08-22 09:40:00 | 499.08 | 498.84 | 0.00 | SL hit |

### Cycle 29 — SELL (started 2025-08-29 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-29 09:30:00 | 498.10 | 500.35 | 0.00 | ORB-short ORB[499.85,504.00] vol=1.8x ATR=1.46 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-29 09:35:00 | 495.91 | 499.49 | 0.00 | T1 1.5R @ 495.91 |
| Stop hit — per-position SL triggered | 2025-08-29 09:40:00 | 498.10 | 499.13 | 0.00 | SL hit |

### Cycle 30 — BUY (started 2025-09-08 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-08 09:35:00 | 509.80 | 508.15 | 0.00 | ORB-long ORB[504.80,509.75] vol=1.7x ATR=1.28 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-08 09:50:00 | 511.71 | 508.77 | 0.00 | T1 1.5R @ 511.71 |
| Target hit | 2025-09-08 10:30:00 | 513.65 | 514.37 | 0.00 | Trail-exit close<VWAP |

### Cycle 31 — BUY (started 2025-09-10 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-10 09:40:00 | 520.65 | 518.34 | 0.00 | ORB-long ORB[515.00,520.30] vol=2.2x ATR=1.77 |
| Stop hit — per-position SL triggered | 2025-09-10 09:50:00 | 518.88 | 518.44 | 0.00 | SL hit |

### Cycle 32 — SELL (started 2025-09-11 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-11 09:30:00 | 506.20 | 508.75 | 0.00 | ORB-short ORB[508.95,511.00] vol=2.7x ATR=1.24 |
| Stop hit — per-position SL triggered | 2025-09-11 09:35:00 | 507.44 | 508.68 | 0.00 | SL hit |

### Cycle 33 — BUY (started 2025-09-12 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-12 09:30:00 | 510.65 | 507.87 | 0.00 | ORB-long ORB[504.75,509.70] vol=2.2x ATR=1.38 |
| Stop hit — per-position SL triggered | 2025-09-12 09:40:00 | 509.27 | 508.38 | 0.00 | SL hit |

### Cycle 34 — BUY (started 2025-09-17 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-17 10:50:00 | 524.50 | 516.54 | 0.00 | ORB-long ORB[510.50,517.00] vol=2.6x ATR=1.98 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-17 12:35:00 | 527.47 | 520.70 | 0.00 | T1 1.5R @ 527.47 |
| Target hit | 2025-09-17 15:20:00 | 531.50 | 528.06 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 35 — BUY (started 2025-09-19 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-19 09:30:00 | 543.20 | 538.90 | 0.00 | ORB-long ORB[532.55,539.00] vol=5.1x ATR=1.89 |
| Stop hit — per-position SL triggered | 2025-09-19 09:35:00 | 541.31 | 539.98 | 0.00 | SL hit |

### Cycle 36 — BUY (started 2025-09-26 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-26 09:40:00 | 580.75 | 572.79 | 0.00 | ORB-long ORB[566.10,573.90] vol=2.7x ATR=3.49 |
| Stop hit — per-position SL triggered | 2025-09-26 09:55:00 | 577.26 | 575.09 | 0.00 | SL hit |

### Cycle 37 — BUY (started 2025-10-03 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-03 09:45:00 | 586.40 | 582.65 | 0.00 | ORB-long ORB[577.25,583.80] vol=2.6x ATR=2.86 |
| Stop hit — per-position SL triggered | 2025-10-03 09:50:00 | 583.54 | 583.23 | 0.00 | SL hit |

### Cycle 38 — BUY (started 2025-10-07 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-07 09:30:00 | 583.05 | 578.48 | 0.00 | ORB-long ORB[572.85,580.00] vol=3.4x ATR=2.23 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-07 10:25:00 | 586.40 | 581.13 | 0.00 | T1 1.5R @ 586.40 |
| Target hit | 2025-10-07 14:05:00 | 595.30 | 595.38 | 0.00 | Trail-exit close<VWAP |

### Cycle 39 — BUY (started 2025-10-08 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-08 09:40:00 | 601.45 | 597.96 | 0.00 | ORB-long ORB[591.35,599.75] vol=1.5x ATR=2.57 |
| Stop hit — per-position SL triggered | 2025-10-08 09:55:00 | 598.88 | 599.49 | 0.00 | SL hit |

### Cycle 40 — BUY (started 2025-10-09 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-09 09:40:00 | 587.55 | 582.97 | 0.00 | ORB-long ORB[577.90,585.80] vol=1.5x ATR=2.78 |
| Stop hit — per-position SL triggered | 2025-10-09 10:55:00 | 584.77 | 585.18 | 0.00 | SL hit |

### Cycle 41 — BUY (started 2025-10-10 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-10 09:30:00 | 588.00 | 583.15 | 0.00 | ORB-long ORB[578.75,585.00] vol=2.7x ATR=2.28 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-10 09:40:00 | 591.42 | 586.28 | 0.00 | T1 1.5R @ 591.42 |
| Stop hit — per-position SL triggered | 2025-10-10 09:45:00 | 588.00 | 586.63 | 0.00 | SL hit |

### Cycle 42 — SELL (started 2025-10-14 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-14 11:00:00 | 562.55 | 567.17 | 0.00 | ORB-short ORB[567.50,572.00] vol=2.4x ATR=2.03 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-14 11:20:00 | 559.51 | 566.52 | 0.00 | T1 1.5R @ 559.51 |
| Stop hit — per-position SL triggered | 2025-10-14 14:55:00 | 562.55 | 561.77 | 0.00 | SL hit |

### Cycle 43 — BUY (started 2025-10-15 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-15 10:50:00 | 566.95 | 563.40 | 0.00 | ORB-long ORB[559.40,564.60] vol=2.0x ATR=1.92 |
| Stop hit — per-position SL triggered | 2025-10-15 11:05:00 | 565.03 | 564.10 | 0.00 | SL hit |

### Cycle 44 — BUY (started 2025-10-16 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-16 09:30:00 | 577.95 | 574.39 | 0.00 | ORB-long ORB[569.90,573.75] vol=4.5x ATR=2.03 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-16 09:45:00 | 581.00 | 577.01 | 0.00 | T1 1.5R @ 581.00 |
| Target hit | 2025-10-16 13:00:00 | 582.35 | 582.80 | 0.00 | Trail-exit close<VWAP |

### Cycle 45 — SELL (started 2025-10-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-27 11:15:00 | 561.75 | 564.55 | 0.00 | ORB-short ORB[562.20,567.90] vol=1.6x ATR=1.11 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-27 11:25:00 | 560.08 | 564.33 | 0.00 | T1 1.5R @ 560.08 |
| Stop hit — per-position SL triggered | 2025-10-27 11:30:00 | 561.75 | 564.29 | 0.00 | SL hit |

### Cycle 46 — BUY (started 2025-10-28 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-28 09:40:00 | 569.50 | 567.89 | 0.00 | ORB-long ORB[564.50,569.10] vol=2.4x ATR=1.25 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-28 09:45:00 | 571.38 | 568.48 | 0.00 | T1 1.5R @ 571.38 |
| Stop hit — per-position SL triggered | 2025-10-28 10:10:00 | 569.50 | 569.40 | 0.00 | SL hit |

### Cycle 47 — SELL (started 2025-10-30 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-30 10:20:00 | 566.80 | 569.44 | 0.00 | ORB-short ORB[572.00,578.10] vol=1.9x ATR=1.50 |
| Stop hit — per-position SL triggered | 2025-10-30 10:50:00 | 568.30 | 568.23 | 0.00 | SL hit |

### Cycle 48 — BUY (started 2025-10-31 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-31 09:30:00 | 576.00 | 573.22 | 0.00 | ORB-long ORB[569.15,573.20] vol=2.9x ATR=1.65 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-31 09:55:00 | 578.47 | 575.36 | 0.00 | T1 1.5R @ 578.47 |
| Stop hit — per-position SL triggered | 2025-10-31 10:05:00 | 576.00 | 575.61 | 0.00 | SL hit |

### Cycle 49 — BUY (started 2025-11-03 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-03 09:30:00 | 571.70 | 570.67 | 0.00 | ORB-long ORB[563.70,569.90] vol=6.5x ATR=2.15 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-03 09:35:00 | 574.92 | 571.37 | 0.00 | T1 1.5R @ 574.92 |
| Target hit | 2025-11-03 10:35:00 | 574.15 | 575.44 | 0.00 | Trail-exit close<VWAP |

### Cycle 50 — BUY (started 2025-11-04 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-04 10:20:00 | 597.55 | 590.45 | 0.00 | ORB-long ORB[585.25,591.90] vol=3.4x ATR=2.65 |
| Stop hit — per-position SL triggered | 2025-11-04 10:30:00 | 594.90 | 591.81 | 0.00 | SL hit |

### Cycle 51 — BUY (started 2025-11-06 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-06 09:30:00 | 596.40 | 592.31 | 0.00 | ORB-long ORB[588.10,595.00] vol=1.6x ATR=2.70 |
| Stop hit — per-position SL triggered | 2025-11-06 09:35:00 | 593.70 | 592.46 | 0.00 | SL hit |

### Cycle 52 — BUY (started 2025-11-17 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-17 09:30:00 | 610.80 | 607.98 | 0.00 | ORB-long ORB[604.30,609.00] vol=3.1x ATR=2.64 |
| Stop hit — per-position SL triggered | 2025-11-17 10:15:00 | 608.16 | 610.13 | 0.00 | SL hit |

### Cycle 53 — BUY (started 2025-11-19 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-19 10:30:00 | 609.00 | 600.58 | 0.00 | ORB-long ORB[593.00,596.90] vol=4.0x ATR=2.12 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-19 10:40:00 | 612.18 | 606.33 | 0.00 | T1 1.5R @ 612.18 |
| Stop hit — per-position SL triggered | 2025-11-19 10:55:00 | 609.00 | 611.32 | 0.00 | SL hit |

### Cycle 54 — SELL (started 2025-11-21 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-21 09:50:00 | 594.00 | 596.42 | 0.00 | ORB-short ORB[595.10,601.65] vol=1.5x ATR=1.92 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-21 10:10:00 | 591.12 | 595.40 | 0.00 | T1 1.5R @ 591.12 |
| Stop hit — per-position SL triggered | 2025-11-21 11:20:00 | 594.00 | 592.99 | 0.00 | SL hit |

### Cycle 55 — SELL (started 2025-11-27 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-27 11:00:00 | 589.40 | 592.34 | 0.00 | ORB-short ORB[591.15,597.45] vol=3.9x ATR=1.69 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-27 12:35:00 | 586.87 | 590.89 | 0.00 | T1 1.5R @ 586.87 |
| Target hit | 2025-11-27 15:20:00 | 587.10 | 589.25 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 56 — SELL (started 2025-11-28 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-28 10:00:00 | 578.85 | 582.97 | 0.00 | ORB-short ORB[582.60,587.00] vol=3.6x ATR=1.65 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-28 10:15:00 | 576.37 | 582.00 | 0.00 | T1 1.5R @ 576.37 |
| Stop hit — per-position SL triggered | 2025-11-28 11:35:00 | 578.85 | 580.44 | 0.00 | SL hit |

### Cycle 57 — BUY (started 2025-12-02 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-02 10:05:00 | 584.25 | 580.99 | 0.00 | ORB-long ORB[575.80,584.00] vol=1.6x ATR=1.96 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-02 11:45:00 | 587.19 | 583.47 | 0.00 | T1 1.5R @ 587.19 |
| Target hit | 2025-12-02 15:20:00 | 597.30 | 593.67 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 58 — BUY (started 2025-12-10 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-10 09:40:00 | 594.90 | 590.54 | 0.00 | ORB-long ORB[582.95,590.90] vol=3.0x ATR=2.43 |
| Stop hit — per-position SL triggered | 2025-12-10 09:50:00 | 592.47 | 591.69 | 0.00 | SL hit |

### Cycle 59 — BUY (started 2025-12-11 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-11 09:30:00 | 584.60 | 581.94 | 0.00 | ORB-long ORB[578.40,583.05] vol=2.0x ATR=2.58 |
| Stop hit — per-position SL triggered | 2025-12-11 09:35:00 | 582.02 | 582.11 | 0.00 | SL hit |

### Cycle 60 — BUY (started 2025-12-12 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-12 09:40:00 | 592.25 | 589.34 | 0.00 | ORB-long ORB[584.30,588.30] vol=4.8x ATR=1.77 |
| Stop hit — per-position SL triggered | 2025-12-12 09:50:00 | 590.48 | 590.04 | 0.00 | SL hit |

### Cycle 61 — BUY (started 2025-12-16 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-16 09:30:00 | 602.40 | 598.91 | 0.00 | ORB-long ORB[594.00,601.70] vol=3.0x ATR=1.72 |
| Stop hit — per-position SL triggered | 2025-12-16 10:00:00 | 600.68 | 600.64 | 0.00 | SL hit |

### Cycle 62 — BUY (started 2025-12-17 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-17 09:30:00 | 610.80 | 606.68 | 0.00 | ORB-long ORB[597.45,604.20] vol=4.5x ATR=2.33 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-17 09:35:00 | 614.29 | 609.88 | 0.00 | T1 1.5R @ 614.29 |
| Stop hit — per-position SL triggered | 2025-12-17 10:00:00 | 610.80 | 611.42 | 0.00 | SL hit |

### Cycle 63 — BUY (started 2025-12-19 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-19 10:45:00 | 594.15 | 591.46 | 0.00 | ORB-long ORB[586.00,593.00] vol=3.0x ATR=2.01 |
| Stop hit — per-position SL triggered | 2025-12-19 12:25:00 | 592.14 | 592.18 | 0.00 | SL hit |

### Cycle 64 — BUY (started 2025-12-22 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-22 10:40:00 | 600.60 | 597.08 | 0.00 | ORB-long ORB[593.35,598.80] vol=1.7x ATR=1.65 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-22 11:55:00 | 603.07 | 599.05 | 0.00 | T1 1.5R @ 603.07 |
| Target hit | 2025-12-22 15:20:00 | 604.10 | 602.64 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 65 — SELL (started 2025-12-23 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-23 11:05:00 | 599.85 | 602.19 | 0.00 | ORB-short ORB[601.75,609.00] vol=1.5x ATR=1.34 |
| Stop hit — per-position SL triggered | 2025-12-23 11:25:00 | 601.19 | 601.96 | 0.00 | SL hit |

### Cycle 66 — SELL (started 2025-12-24 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-24 09:30:00 | 594.45 | 597.09 | 0.00 | ORB-short ORB[596.10,600.25] vol=2.8x ATR=1.65 |
| Stop hit — per-position SL triggered | 2025-12-24 09:40:00 | 596.10 | 596.73 | 0.00 | SL hit |

### Cycle 67 — SELL (started 2025-12-29 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-29 09:45:00 | 576.55 | 577.34 | 0.00 | ORB-short ORB[578.35,584.10] vol=8.2x ATR=2.44 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-29 10:05:00 | 572.88 | 576.91 | 0.00 | T1 1.5R @ 572.88 |
| Target hit | 2025-12-29 15:20:00 | 567.10 | 572.15 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 68 — BUY (started 2026-01-05 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-05 10:55:00 | 609.00 | 603.91 | 0.00 | ORB-long ORB[599.95,605.85] vol=3.9x ATR=2.07 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-05 11:00:00 | 612.11 | 605.77 | 0.00 | T1 1.5R @ 612.11 |
| Stop hit — per-position SL triggered | 2026-01-05 11:15:00 | 609.00 | 607.35 | 0.00 | SL hit |

### Cycle 69 — SELL (started 2026-01-13 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-13 10:55:00 | 576.65 | 579.73 | 0.00 | ORB-short ORB[579.50,585.20] vol=1.5x ATR=1.67 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-13 12:50:00 | 574.14 | 578.88 | 0.00 | T1 1.5R @ 574.14 |
| Target hit | 2026-01-13 15:20:00 | 575.55 | 577.30 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 70 — SELL (started 2026-01-14 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-14 10:25:00 | 569.85 | 572.51 | 0.00 | ORB-short ORB[572.80,576.85] vol=2.3x ATR=1.77 |
| Stop hit — per-position SL triggered | 2026-01-14 11:10:00 | 571.62 | 571.61 | 0.00 | SL hit |

### Cycle 71 — SELL (started 2026-01-16 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-16 10:10:00 | 567.60 | 569.24 | 0.00 | ORB-short ORB[568.05,572.80] vol=2.2x ATR=1.71 |
| Stop hit — per-position SL triggered | 2026-01-16 10:20:00 | 569.31 | 569.20 | 0.00 | SL hit |

### Cycle 72 — SELL (started 2026-01-28 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-28 10:20:00 | 531.60 | 538.38 | 0.00 | ORB-short ORB[541.15,544.90] vol=2.2x ATR=1.88 |
| Stop hit — per-position SL triggered | 2026-01-28 10:25:00 | 533.48 | 538.00 | 0.00 | SL hit |

### Cycle 73 — BUY (started 2026-01-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-29 11:15:00 | 547.25 | 544.63 | 0.00 | ORB-long ORB[541.90,546.15] vol=2.6x ATR=1.31 |
| Stop hit — per-position SL triggered | 2026-01-29 12:00:00 | 545.94 | 544.77 | 0.00 | SL hit |

### Cycle 74 — BUY (started 2026-02-02 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-02 09:30:00 | 550.90 | 548.31 | 0.00 | ORB-long ORB[545.25,550.30] vol=2.0x ATR=3.22 |
| Stop hit — per-position SL triggered | 2026-02-02 10:05:00 | 547.68 | 548.70 | 0.00 | SL hit |

### Cycle 75 — SELL (started 2026-02-05 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-05 09:40:00 | 578.05 | 581.12 | 0.00 | ORB-short ORB[580.50,584.75] vol=5.0x ATR=2.11 |
| Stop hit — per-position SL triggered | 2026-02-05 10:15:00 | 580.16 | 579.89 | 0.00 | SL hit |

### Cycle 76 — BUY (started 2026-02-20 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-20 10:50:00 | 578.35 | 573.30 | 0.00 | ORB-long ORB[565.60,574.00] vol=1.7x ATR=1.96 |
| Stop hit — per-position SL triggered | 2026-02-20 11:05:00 | 576.39 | 573.54 | 0.00 | SL hit |

### Cycle 77 — SELL (started 2026-02-24 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-24 11:05:00 | 572.40 | 574.33 | 0.00 | ORB-short ORB[572.70,581.00] vol=5.5x ATR=1.42 |
| Stop hit — per-position SL triggered | 2026-02-24 11:10:00 | 573.82 | 574.32 | 0.00 | SL hit |

### Cycle 78 — SELL (started 2026-03-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-05 11:15:00 | 528.00 | 529.49 | 0.00 | ORB-short ORB[529.15,534.10] vol=3.1x ATR=1.28 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-05 11:25:00 | 526.07 | 529.43 | 0.00 | T1 1.5R @ 526.07 |
| Stop hit — per-position SL triggered | 2026-03-05 12:15:00 | 528.00 | 528.97 | 0.00 | SL hit |

### Cycle 79 — BUY (started 2026-03-17 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-17 10:00:00 | 484.90 | 482.83 | 0.00 | ORB-long ORB[477.00,483.50] vol=3.3x ATR=1.56 |
| Stop hit — per-position SL triggered | 2026-03-17 10:30:00 | 483.34 | 483.26 | 0.00 | SL hit |

### Cycle 80 — BUY (started 2026-03-19 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-19 11:00:00 | 501.25 | 497.00 | 0.00 | ORB-long ORB[494.35,499.90] vol=2.2x ATR=1.75 |
| Stop hit — per-position SL triggered | 2026-03-19 14:35:00 | 499.50 | 498.93 | 0.00 | SL hit |

### Cycle 81 — SELL (started 2026-03-20 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-20 09:45:00 | 497.35 | 501.78 | 0.00 | ORB-short ORB[501.05,508.00] vol=1.5x ATR=1.97 |
| Stop hit — per-position SL triggered | 2026-03-20 09:50:00 | 499.32 | 500.41 | 0.00 | SL hit |

### Cycle 82 — BUY (started 2026-04-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-17 10:15:00 | 524.00 | 518.31 | 0.00 | ORB-long ORB[507.00,514.65] vol=1.7x ATR=2.10 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-17 10:35:00 | 527.15 | 520.39 | 0.00 | T1 1.5R @ 527.15 |
| Target hit | 2026-04-17 15:20:00 | 532.75 | 531.23 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 83 — BUY (started 2026-04-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-21 11:15:00 | 542.55 | 540.98 | 0.00 | ORB-long ORB[536.65,540.80] vol=1.8x ATR=1.33 |
| Stop hit — per-position SL triggered | 2026-04-21 12:00:00 | 541.22 | 541.27 | 0.00 | SL hit |

### Cycle 84 — BUY (started 2026-04-22 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-22 09:35:00 | 551.00 | 546.36 | 0.00 | ORB-long ORB[541.10,548.70] vol=3.5x ATR=2.35 |
| Stop hit — per-position SL triggered | 2026-04-22 09:40:00 | 548.65 | 546.63 | 0.00 | SL hit |

### Cycle 85 — BUY (started 2026-05-08 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-08 10:45:00 | 541.45 | 535.35 | 0.00 | ORB-long ORB[531.60,538.00] vol=2.0x ATR=1.82 |
| Stop hit — per-position SL triggered | 2026-05-08 11:00:00 | 539.63 | 536.60 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-05-14 10:40:00 | 496.30 | 2025-05-14 10:55:00 | 495.00 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2025-05-16 09:30:00 | 509.80 | 2025-05-16 09:40:00 | 508.01 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2025-05-20 11:05:00 | 510.00 | 2025-05-20 12:05:00 | 507.47 | PARTIAL | 0.50 | 0.50% |
| SELL | retest1 | 2025-05-20 11:05:00 | 510.00 | 2025-05-20 15:20:00 | 500.85 | TARGET_HIT | 0.50 | 1.79% |
| BUY | retest1 | 2025-05-21 09:40:00 | 505.00 | 2025-05-21 09:45:00 | 507.54 | PARTIAL | 0.50 | 0.50% |
| BUY | retest1 | 2025-05-21 09:40:00 | 505.00 | 2025-05-21 09:55:00 | 505.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-05-23 10:55:00 | 515.35 | 2025-05-23 11:00:00 | 513.83 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2025-05-30 10:50:00 | 526.30 | 2025-05-30 12:10:00 | 524.15 | PARTIAL | 0.50 | 0.41% |
| SELL | retest1 | 2025-05-30 10:50:00 | 526.30 | 2025-05-30 15:20:00 | 521.65 | TARGET_HIT | 0.50 | 0.88% |
| SELL | retest1 | 2025-06-04 09:50:00 | 511.60 | 2025-06-04 10:00:00 | 513.28 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2025-06-06 09:30:00 | 549.80 | 2025-06-06 09:40:00 | 553.74 | PARTIAL | 0.50 | 0.72% |
| BUY | retest1 | 2025-06-06 09:30:00 | 549.80 | 2025-06-06 09:50:00 | 549.80 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-06-17 11:15:00 | 531.55 | 2025-06-17 11:30:00 | 532.74 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2025-06-23 11:00:00 | 505.00 | 2025-06-23 15:20:00 | 505.05 | TARGET_HIT | 1.00 | 0.01% |
| SELL | retest1 | 2025-06-26 09:55:00 | 510.45 | 2025-06-26 10:50:00 | 507.95 | PARTIAL | 0.50 | 0.49% |
| SELL | retest1 | 2025-06-26 09:55:00 | 510.45 | 2025-06-26 14:30:00 | 510.00 | TARGET_HIT | 0.50 | 0.09% |
| BUY | retest1 | 2025-06-27 09:35:00 | 519.55 | 2025-06-27 09:40:00 | 517.74 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2025-07-01 09:30:00 | 515.20 | 2025-07-01 09:35:00 | 512.96 | PARTIAL | 0.50 | 0.44% |
| SELL | retest1 | 2025-07-01 09:30:00 | 515.20 | 2025-07-01 09:45:00 | 515.20 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-07-02 10:25:00 | 521.55 | 2025-07-02 10:35:00 | 523.91 | STOP_HIT | 1.00 | -0.45% |
| BUY | retest1 | 2025-07-04 10:45:00 | 521.60 | 2025-07-04 10:50:00 | 519.89 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2025-07-07 11:00:00 | 512.20 | 2025-07-07 11:15:00 | 513.22 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2025-07-11 10:10:00 | 511.00 | 2025-07-11 10:30:00 | 509.08 | PARTIAL | 0.50 | 0.38% |
| SELL | retest1 | 2025-07-11 10:10:00 | 511.00 | 2025-07-11 10:35:00 | 511.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-07-16 11:05:00 | 516.25 | 2025-07-16 11:45:00 | 517.20 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest1 | 2025-07-18 10:15:00 | 530.90 | 2025-07-18 10:20:00 | 528.46 | PARTIAL | 0.50 | 0.46% |
| SELL | retest1 | 2025-07-18 10:15:00 | 530.90 | 2025-07-18 10:45:00 | 530.90 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-07-25 10:35:00 | 505.00 | 2025-07-25 11:05:00 | 506.30 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2025-07-29 10:45:00 | 505.00 | 2025-07-29 12:10:00 | 506.56 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2025-07-30 10:50:00 | 511.70 | 2025-07-30 11:25:00 | 512.90 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2025-08-01 10:00:00 | 496.50 | 2025-08-01 10:30:00 | 498.05 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2025-08-05 09:40:00 | 489.00 | 2025-08-05 10:25:00 | 486.04 | PARTIAL | 0.50 | 0.61% |
| SELL | retest1 | 2025-08-05 09:40:00 | 489.00 | 2025-08-05 12:25:00 | 488.50 | TARGET_HIT | 0.50 | 0.10% |
| SELL | retest1 | 2025-08-11 09:40:00 | 452.65 | 2025-08-11 09:50:00 | 449.63 | PARTIAL | 0.50 | 0.67% |
| SELL | retest1 | 2025-08-11 09:40:00 | 452.65 | 2025-08-11 14:05:00 | 452.65 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-08-20 09:35:00 | 513.85 | 2025-08-20 09:40:00 | 515.18 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2025-08-21 10:35:00 | 506.70 | 2025-08-21 12:00:00 | 504.96 | PARTIAL | 0.50 | 0.34% |
| SELL | retest1 | 2025-08-21 10:35:00 | 506.70 | 2025-08-21 15:20:00 | 500.50 | TARGET_HIT | 0.50 | 1.22% |
| SELL | retest1 | 2025-08-22 09:30:00 | 497.80 | 2025-08-22 09:40:00 | 499.08 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2025-08-29 09:30:00 | 498.10 | 2025-08-29 09:35:00 | 495.91 | PARTIAL | 0.50 | 0.44% |
| SELL | retest1 | 2025-08-29 09:30:00 | 498.10 | 2025-08-29 09:40:00 | 498.10 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-09-08 09:35:00 | 509.80 | 2025-09-08 09:50:00 | 511.71 | PARTIAL | 0.50 | 0.38% |
| BUY | retest1 | 2025-09-08 09:35:00 | 509.80 | 2025-09-08 10:30:00 | 513.65 | TARGET_HIT | 0.50 | 0.76% |
| BUY | retest1 | 2025-09-10 09:40:00 | 520.65 | 2025-09-10 09:50:00 | 518.88 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2025-09-11 09:30:00 | 506.20 | 2025-09-11 09:35:00 | 507.44 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2025-09-12 09:30:00 | 510.65 | 2025-09-12 09:40:00 | 509.27 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2025-09-17 10:50:00 | 524.50 | 2025-09-17 12:35:00 | 527.47 | PARTIAL | 0.50 | 0.57% |
| BUY | retest1 | 2025-09-17 10:50:00 | 524.50 | 2025-09-17 15:20:00 | 531.50 | TARGET_HIT | 0.50 | 1.33% |
| BUY | retest1 | 2025-09-19 09:30:00 | 543.20 | 2025-09-19 09:35:00 | 541.31 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2025-09-26 09:40:00 | 580.75 | 2025-09-26 09:55:00 | 577.26 | STOP_HIT | 1.00 | -0.60% |
| BUY | retest1 | 2025-10-03 09:45:00 | 586.40 | 2025-10-03 09:50:00 | 583.54 | STOP_HIT | 1.00 | -0.49% |
| BUY | retest1 | 2025-10-07 09:30:00 | 583.05 | 2025-10-07 10:25:00 | 586.40 | PARTIAL | 0.50 | 0.57% |
| BUY | retest1 | 2025-10-07 09:30:00 | 583.05 | 2025-10-07 14:05:00 | 595.30 | TARGET_HIT | 0.50 | 2.10% |
| BUY | retest1 | 2025-10-08 09:40:00 | 601.45 | 2025-10-08 09:55:00 | 598.88 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest1 | 2025-10-09 09:40:00 | 587.55 | 2025-10-09 10:55:00 | 584.77 | STOP_HIT | 1.00 | -0.47% |
| BUY | retest1 | 2025-10-10 09:30:00 | 588.00 | 2025-10-10 09:40:00 | 591.42 | PARTIAL | 0.50 | 0.58% |
| BUY | retest1 | 2025-10-10 09:30:00 | 588.00 | 2025-10-10 09:45:00 | 588.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-10-14 11:00:00 | 562.55 | 2025-10-14 11:20:00 | 559.51 | PARTIAL | 0.50 | 0.54% |
| SELL | retest1 | 2025-10-14 11:00:00 | 562.55 | 2025-10-14 14:55:00 | 562.55 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-10-15 10:50:00 | 566.95 | 2025-10-15 11:05:00 | 565.03 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2025-10-16 09:30:00 | 577.95 | 2025-10-16 09:45:00 | 581.00 | PARTIAL | 0.50 | 0.53% |
| BUY | retest1 | 2025-10-16 09:30:00 | 577.95 | 2025-10-16 13:00:00 | 582.35 | TARGET_HIT | 0.50 | 0.76% |
| SELL | retest1 | 2025-10-27 11:15:00 | 561.75 | 2025-10-27 11:25:00 | 560.08 | PARTIAL | 0.50 | 0.30% |
| SELL | retest1 | 2025-10-27 11:15:00 | 561.75 | 2025-10-27 11:30:00 | 561.75 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-10-28 09:40:00 | 569.50 | 2025-10-28 09:45:00 | 571.38 | PARTIAL | 0.50 | 0.33% |
| BUY | retest1 | 2025-10-28 09:40:00 | 569.50 | 2025-10-28 10:10:00 | 569.50 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-10-30 10:20:00 | 566.80 | 2025-10-30 10:50:00 | 568.30 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2025-10-31 09:30:00 | 576.00 | 2025-10-31 09:55:00 | 578.47 | PARTIAL | 0.50 | 0.43% |
| BUY | retest1 | 2025-10-31 09:30:00 | 576.00 | 2025-10-31 10:05:00 | 576.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-11-03 09:30:00 | 571.70 | 2025-11-03 09:35:00 | 574.92 | PARTIAL | 0.50 | 0.56% |
| BUY | retest1 | 2025-11-03 09:30:00 | 571.70 | 2025-11-03 10:35:00 | 574.15 | TARGET_HIT | 0.50 | 0.43% |
| BUY | retest1 | 2025-11-04 10:20:00 | 597.55 | 2025-11-04 10:30:00 | 594.90 | STOP_HIT | 1.00 | -0.44% |
| BUY | retest1 | 2025-11-06 09:30:00 | 596.40 | 2025-11-06 09:35:00 | 593.70 | STOP_HIT | 1.00 | -0.45% |
| BUY | retest1 | 2025-11-17 09:30:00 | 610.80 | 2025-11-17 10:15:00 | 608.16 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest1 | 2025-11-19 10:30:00 | 609.00 | 2025-11-19 10:40:00 | 612.18 | PARTIAL | 0.50 | 0.52% |
| BUY | retest1 | 2025-11-19 10:30:00 | 609.00 | 2025-11-19 10:55:00 | 609.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-11-21 09:50:00 | 594.00 | 2025-11-21 10:10:00 | 591.12 | PARTIAL | 0.50 | 0.49% |
| SELL | retest1 | 2025-11-21 09:50:00 | 594.00 | 2025-11-21 11:20:00 | 594.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-11-27 11:00:00 | 589.40 | 2025-11-27 12:35:00 | 586.87 | PARTIAL | 0.50 | 0.43% |
| SELL | retest1 | 2025-11-27 11:00:00 | 589.40 | 2025-11-27 15:20:00 | 587.10 | TARGET_HIT | 0.50 | 0.39% |
| SELL | retest1 | 2025-11-28 10:00:00 | 578.85 | 2025-11-28 10:15:00 | 576.37 | PARTIAL | 0.50 | 0.43% |
| SELL | retest1 | 2025-11-28 10:00:00 | 578.85 | 2025-11-28 11:35:00 | 578.85 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-12-02 10:05:00 | 584.25 | 2025-12-02 11:45:00 | 587.19 | PARTIAL | 0.50 | 0.50% |
| BUY | retest1 | 2025-12-02 10:05:00 | 584.25 | 2025-12-02 15:20:00 | 597.30 | TARGET_HIT | 0.50 | 2.23% |
| BUY | retest1 | 2025-12-10 09:40:00 | 594.90 | 2025-12-10 09:50:00 | 592.47 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest1 | 2025-12-11 09:30:00 | 584.60 | 2025-12-11 09:35:00 | 582.02 | STOP_HIT | 1.00 | -0.44% |
| BUY | retest1 | 2025-12-12 09:40:00 | 592.25 | 2025-12-12 09:50:00 | 590.48 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2025-12-16 09:30:00 | 602.40 | 2025-12-16 10:00:00 | 600.68 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2025-12-17 09:30:00 | 610.80 | 2025-12-17 09:35:00 | 614.29 | PARTIAL | 0.50 | 0.57% |
| BUY | retest1 | 2025-12-17 09:30:00 | 610.80 | 2025-12-17 10:00:00 | 610.80 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-12-19 10:45:00 | 594.15 | 2025-12-19 12:25:00 | 592.14 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2025-12-22 10:40:00 | 600.60 | 2025-12-22 11:55:00 | 603.07 | PARTIAL | 0.50 | 0.41% |
| BUY | retest1 | 2025-12-22 10:40:00 | 600.60 | 2025-12-22 15:20:00 | 604.10 | TARGET_HIT | 0.50 | 0.58% |
| SELL | retest1 | 2025-12-23 11:05:00 | 599.85 | 2025-12-23 11:25:00 | 601.19 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2025-12-24 09:30:00 | 594.45 | 2025-12-24 09:40:00 | 596.10 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2025-12-29 09:45:00 | 576.55 | 2025-12-29 10:05:00 | 572.88 | PARTIAL | 0.50 | 0.64% |
| SELL | retest1 | 2025-12-29 09:45:00 | 576.55 | 2025-12-29 15:20:00 | 567.10 | TARGET_HIT | 0.50 | 1.64% |
| BUY | retest1 | 2026-01-05 10:55:00 | 609.00 | 2026-01-05 11:00:00 | 612.11 | PARTIAL | 0.50 | 0.51% |
| BUY | retest1 | 2026-01-05 10:55:00 | 609.00 | 2026-01-05 11:15:00 | 609.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-01-13 10:55:00 | 576.65 | 2026-01-13 12:50:00 | 574.14 | PARTIAL | 0.50 | 0.43% |
| SELL | retest1 | 2026-01-13 10:55:00 | 576.65 | 2026-01-13 15:20:00 | 575.55 | TARGET_HIT | 0.50 | 0.19% |
| SELL | retest1 | 2026-01-14 10:25:00 | 569.85 | 2026-01-14 11:10:00 | 571.62 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2026-01-16 10:10:00 | 567.60 | 2026-01-16 10:20:00 | 569.31 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2026-01-28 10:20:00 | 531.60 | 2026-01-28 10:25:00 | 533.48 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2026-01-29 11:15:00 | 547.25 | 2026-01-29 12:00:00 | 545.94 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2026-02-02 09:30:00 | 550.90 | 2026-02-02 10:05:00 | 547.68 | STOP_HIT | 1.00 | -0.58% |
| SELL | retest1 | 2026-02-05 09:40:00 | 578.05 | 2026-02-05 10:15:00 | 580.16 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2026-02-20 10:50:00 | 578.35 | 2026-02-20 11:05:00 | 576.39 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2026-02-24 11:05:00 | 572.40 | 2026-02-24 11:10:00 | 573.82 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2026-03-05 11:15:00 | 528.00 | 2026-03-05 11:25:00 | 526.07 | PARTIAL | 0.50 | 0.36% |
| SELL | retest1 | 2026-03-05 11:15:00 | 528.00 | 2026-03-05 12:15:00 | 528.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-03-17 10:00:00 | 484.90 | 2026-03-17 10:30:00 | 483.34 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2026-03-19 11:00:00 | 501.25 | 2026-03-19 14:35:00 | 499.50 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2026-03-20 09:45:00 | 497.35 | 2026-03-20 09:50:00 | 499.32 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest1 | 2026-04-17 10:15:00 | 524.00 | 2026-04-17 10:35:00 | 527.15 | PARTIAL | 0.50 | 0.60% |
| BUY | retest1 | 2026-04-17 10:15:00 | 524.00 | 2026-04-17 15:20:00 | 532.75 | TARGET_HIT | 0.50 | 1.67% |
| BUY | retest1 | 2026-04-21 11:15:00 | 542.55 | 2026-04-21 12:00:00 | 541.22 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2026-04-22 09:35:00 | 551.00 | 2026-04-22 09:40:00 | 548.65 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest1 | 2026-05-08 10:45:00 | 541.45 | 2026-05-08 11:00:00 | 539.63 | STOP_HIT | 1.00 | -0.34% |
