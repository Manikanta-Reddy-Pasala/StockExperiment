# Anant Raj Ltd. (ANANTRAJ)

## Backtest Summary

- **Window:** 2024-05-13 09:15:00 → 2026-05-08 15:25:00 (36871 bars)
- **Last close:** 561.75
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
| PARTIAL | 11 |
| TARGET_HIT | 3 |
| STOP_HIT | 28 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 42 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 14 / 28
- **Target hits / Stop hits / Partials:** 3 / 28 / 11
- **Avg / median % per leg:** 0.23% / 0.00%
- **Sum % (uncompounded):** 9.65%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 23 | 10 | 43.5% | 3 | 13 | 7 | 0.52% | 11.9% |
| BUY @ 2nd Alert (retest1) | 23 | 10 | 43.5% | 3 | 13 | 7 | 0.52% | 11.9% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 19 | 4 | 21.1% | 0 | 15 | 4 | -0.12% | -2.2% |
| SELL @ 2nd Alert (retest1) | 19 | 4 | 21.1% | 0 | 15 | 4 | -0.12% | -2.2% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 42 | 14 | 33.3% | 3 | 28 | 11 | 0.23% | 9.7% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-05-14 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-14 10:45:00 | 381.40 | 377.88 | 0.00 | ORB-long ORB[375.10,380.00] vol=2.6x ATR=1.77 |
| Stop hit — per-position SL triggered | 2024-05-14 10:50:00 | 379.63 | 378.01 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2024-05-15 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-15 09:30:00 | 379.00 | 376.67 | 0.00 | ORB-long ORB[373.05,378.00] vol=2.2x ATR=1.38 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-15 09:35:00 | 381.06 | 378.16 | 0.00 | T1 1.5R @ 381.06 |
| Stop hit — per-position SL triggered | 2024-05-15 09:45:00 | 379.00 | 378.43 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2024-05-16 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-16 11:10:00 | 376.05 | 379.48 | 0.00 | ORB-short ORB[379.80,384.70] vol=2.2x ATR=1.17 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-16 11:20:00 | 374.29 | 378.55 | 0.00 | T1 1.5R @ 374.29 |
| Stop hit — per-position SL triggered | 2024-05-16 12:15:00 | 376.05 | 377.92 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2024-05-17 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-17 09:45:00 | 378.85 | 376.69 | 0.00 | ORB-long ORB[374.25,378.00] vol=1.6x ATR=1.55 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-17 12:15:00 | 381.18 | 379.18 | 0.00 | T1 1.5R @ 381.18 |
| Target hit | 2024-05-17 15:20:00 | 385.50 | 380.85 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 5 — SELL (started 2024-05-23 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-23 09:55:00 | 384.95 | 386.80 | 0.00 | ORB-short ORB[387.05,390.00] vol=1.9x ATR=1.32 |
| Stop hit — per-position SL triggered | 2024-05-23 10:00:00 | 386.27 | 386.56 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2024-05-27 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-27 09:45:00 | 379.00 | 383.16 | 0.00 | ORB-short ORB[382.05,387.20] vol=2.8x ATR=1.91 |
| Stop hit — per-position SL triggered | 2024-05-27 09:50:00 | 380.91 | 382.92 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2024-05-28 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-28 09:35:00 | 376.60 | 377.39 | 0.00 | ORB-short ORB[377.50,382.00] vol=3.6x ATR=1.56 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-28 09:50:00 | 374.26 | 376.56 | 0.00 | T1 1.5R @ 374.26 |
| Stop hit — per-position SL triggered | 2024-05-28 10:00:00 | 376.60 | 376.21 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2024-05-31 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-31 09:30:00 | 375.55 | 374.31 | 0.00 | ORB-long ORB[372.10,375.45] vol=1.8x ATR=1.11 |
| Stop hit — per-position SL triggered | 2024-05-31 09:35:00 | 374.44 | 374.58 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2024-06-10 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-10 09:55:00 | 398.30 | 395.02 | 0.00 | ORB-long ORB[391.05,396.75] vol=1.9x ATR=2.20 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-10 10:05:00 | 401.60 | 396.35 | 0.00 | T1 1.5R @ 401.60 |
| Stop hit — per-position SL triggered | 2024-06-10 14:40:00 | 398.30 | 400.78 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2024-06-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-13 11:15:00 | 415.50 | 420.30 | 0.00 | ORB-short ORB[421.05,424.90] vol=3.1x ATR=1.26 |
| Stop hit — per-position SL triggered | 2024-06-13 11:30:00 | 416.76 | 419.84 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2024-06-25 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-25 10:35:00 | 444.25 | 444.86 | 0.00 | ORB-short ORB[445.85,450.00] vol=14.1x ATR=1.77 |
| Stop hit — per-position SL triggered | 2024-06-25 10:45:00 | 446.02 | 444.92 | 0.00 | SL hit |

### Cycle 12 — SELL (started 2024-06-27 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-27 10:40:00 | 439.10 | 442.53 | 0.00 | ORB-short ORB[440.55,446.00] vol=1.7x ATR=1.32 |
| Stop hit — per-position SL triggered | 2024-06-27 10:50:00 | 440.42 | 442.42 | 0.00 | SL hit |

### Cycle 13 — SELL (started 2024-06-28 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-28 09:30:00 | 435.00 | 437.64 | 0.00 | ORB-short ORB[437.45,440.20] vol=3.0x ATR=1.61 |
| Stop hit — per-position SL triggered | 2024-06-28 09:35:00 | 436.61 | 437.06 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2024-07-01 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-01 09:30:00 | 438.00 | 434.71 | 0.00 | ORB-long ORB[430.60,436.00] vol=2.3x ATR=1.55 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-01 09:40:00 | 440.33 | 437.04 | 0.00 | T1 1.5R @ 440.33 |
| Target hit | 2024-07-01 11:00:00 | 438.70 | 438.89 | 0.00 | Trail-exit close<VWAP |

### Cycle 15 — SELL (started 2024-07-12 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-12 09:50:00 | 498.80 | 503.31 | 0.00 | ORB-short ORB[501.75,508.35] vol=1.7x ATR=2.27 |
| Stop hit — per-position SL triggered | 2024-07-12 09:55:00 | 501.07 | 503.07 | 0.00 | SL hit |

### Cycle 16 — BUY (started 2024-07-16 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-16 09:30:00 | 493.15 | 489.86 | 0.00 | ORB-long ORB[485.45,492.05] vol=3.0x ATR=2.34 |
| Stop hit — per-position SL triggered | 2024-07-16 09:35:00 | 490.81 | 490.44 | 0.00 | SL hit |

### Cycle 17 — SELL (started 2024-08-13 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-13 09:35:00 | 512.10 | 514.71 | 0.00 | ORB-short ORB[513.00,519.00] vol=2.0x ATR=2.11 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-13 10:15:00 | 508.94 | 513.32 | 0.00 | T1 1.5R @ 508.94 |
| Stop hit — per-position SL triggered | 2024-08-13 10:40:00 | 512.10 | 512.53 | 0.00 | SL hit |

### Cycle 18 — BUY (started 2024-08-16 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-16 09:50:00 | 533.75 | 529.40 | 0.00 | ORB-long ORB[522.50,529.40] vol=1.9x ATR=2.55 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-16 12:40:00 | 537.58 | 534.23 | 0.00 | T1 1.5R @ 537.58 |
| Target hit | 2024-08-16 15:20:00 | 584.55 | 564.46 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 19 — SELL (started 2024-08-29 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-29 09:30:00 | 604.95 | 609.08 | 0.00 | ORB-short ORB[607.05,614.10] vol=1.7x ATR=2.35 |
| Stop hit — per-position SL triggered | 2024-08-29 09:40:00 | 607.30 | 608.26 | 0.00 | SL hit |

### Cycle 20 — BUY (started 2024-10-14 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-14 10:45:00 | 760.85 | 751.69 | 0.00 | ORB-long ORB[741.90,752.85] vol=2.7x ATR=2.93 |
| Stop hit — per-position SL triggered | 2024-10-14 11:00:00 | 757.92 | 752.43 | 0.00 | SL hit |

### Cycle 21 — SELL (started 2024-11-06 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-06 10:40:00 | 746.35 | 751.94 | 0.00 | ORB-short ORB[752.60,760.00] vol=2.1x ATR=4.21 |
| Stop hit — per-position SL triggered | 2024-11-06 11:00:00 | 750.56 | 751.72 | 0.00 | SL hit |

### Cycle 22 — SELL (started 2024-11-12 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-12 09:45:00 | 733.10 | 736.16 | 0.00 | ORB-short ORB[734.50,741.80] vol=1.6x ATR=3.20 |
| Stop hit — per-position SL triggered | 2024-11-12 11:20:00 | 736.30 | 734.83 | 0.00 | SL hit |

### Cycle 23 — SELL (started 2024-11-26 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-26 09:40:00 | 662.85 | 671.97 | 0.00 | ORB-short ORB[673.50,679.80] vol=1.7x ATR=2.95 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-26 09:50:00 | 658.42 | 668.81 | 0.00 | T1 1.5R @ 658.42 |
| Stop hit — per-position SL triggered | 2024-11-26 10:15:00 | 662.85 | 666.59 | 0.00 | SL hit |

### Cycle 24 — BUY (started 2024-11-27 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-27 10:35:00 | 672.65 | 663.63 | 0.00 | ORB-long ORB[657.50,667.00] vol=1.8x ATR=2.65 |
| Stop hit — per-position SL triggered | 2024-11-27 10:40:00 | 670.00 | 664.22 | 0.00 | SL hit |

### Cycle 25 — SELL (started 2024-11-29 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-29 10:25:00 | 662.90 | 667.38 | 0.00 | ORB-short ORB[666.20,673.95] vol=2.4x ATR=3.37 |
| Stop hit — per-position SL triggered | 2024-11-29 11:10:00 | 666.27 | 664.94 | 0.00 | SL hit |

### Cycle 26 — BUY (started 2024-12-05 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-05 10:10:00 | 731.70 | 725.47 | 0.00 | ORB-long ORB[717.20,725.95] vol=4.3x ATR=3.79 |
| Stop hit — per-position SL triggered | 2024-12-05 10:20:00 | 727.91 | 725.89 | 0.00 | SL hit |

### Cycle 27 — BUY (started 2024-12-06 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-06 11:00:00 | 734.25 | 730.44 | 0.00 | ORB-long ORB[724.00,733.00] vol=1.8x ATR=2.98 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-06 11:25:00 | 738.72 | 731.44 | 0.00 | T1 1.5R @ 738.72 |
| Stop hit — per-position SL triggered | 2024-12-06 11:45:00 | 734.25 | 732.04 | 0.00 | SL hit |

### Cycle 28 — BUY (started 2024-12-12 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-12 09:55:00 | 741.45 | 733.44 | 0.00 | ORB-long ORB[724.75,734.25] vol=4.5x ATR=2.77 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-12 10:10:00 | 745.61 | 737.87 | 0.00 | T1 1.5R @ 745.61 |
| Stop hit — per-position SL triggered | 2024-12-12 10:25:00 | 741.45 | 741.02 | 0.00 | SL hit |

### Cycle 29 — BUY (started 2024-12-27 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-27 09:40:00 | 830.40 | 823.44 | 0.00 | ORB-long ORB[817.05,827.50] vol=1.5x ATR=4.12 |
| Stop hit — per-position SL triggered | 2024-12-27 09:45:00 | 826.28 | 823.86 | 0.00 | SL hit |

### Cycle 30 — BUY (started 2025-01-20 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-20 09:40:00 | 914.60 | 905.97 | 0.00 | ORB-long ORB[899.50,908.55] vol=2.8x ATR=4.26 |
| Stop hit — per-position SL triggered | 2025-01-20 09:45:00 | 910.34 | 906.75 | 0.00 | SL hit |

### Cycle 31 — BUY (started 2025-03-18 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-18 09:35:00 | 499.00 | 495.76 | 0.00 | ORB-long ORB[491.00,497.80] vol=1.8x ATR=2.46 |
| Stop hit — per-position SL triggered | 2025-03-18 09:50:00 | 496.54 | 496.47 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-05-14 10:45:00 | 381.40 | 2024-05-14 10:50:00 | 379.63 | STOP_HIT | 1.00 | -0.47% |
| BUY | retest1 | 2024-05-15 09:30:00 | 379.00 | 2024-05-15 09:35:00 | 381.06 | PARTIAL | 0.50 | 0.54% |
| BUY | retest1 | 2024-05-15 09:30:00 | 379.00 | 2024-05-15 09:45:00 | 379.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-05-16 11:10:00 | 376.05 | 2024-05-16 11:20:00 | 374.29 | PARTIAL | 0.50 | 0.47% |
| SELL | retest1 | 2024-05-16 11:10:00 | 376.05 | 2024-05-16 12:15:00 | 376.05 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-05-17 09:45:00 | 378.85 | 2024-05-17 12:15:00 | 381.18 | PARTIAL | 0.50 | 0.61% |
| BUY | retest1 | 2024-05-17 09:45:00 | 378.85 | 2024-05-17 15:20:00 | 385.50 | TARGET_HIT | 0.50 | 1.76% |
| SELL | retest1 | 2024-05-23 09:55:00 | 384.95 | 2024-05-23 10:00:00 | 386.27 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2024-05-27 09:45:00 | 379.00 | 2024-05-27 09:50:00 | 380.91 | STOP_HIT | 1.00 | -0.50% |
| SELL | retest1 | 2024-05-28 09:35:00 | 376.60 | 2024-05-28 09:50:00 | 374.26 | PARTIAL | 0.50 | 0.62% |
| SELL | retest1 | 2024-05-28 09:35:00 | 376.60 | 2024-05-28 10:00:00 | 376.60 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-05-31 09:30:00 | 375.55 | 2024-05-31 09:35:00 | 374.44 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2024-06-10 09:55:00 | 398.30 | 2024-06-10 10:05:00 | 401.60 | PARTIAL | 0.50 | 0.83% |
| BUY | retest1 | 2024-06-10 09:55:00 | 398.30 | 2024-06-10 14:40:00 | 398.30 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-06-13 11:15:00 | 415.50 | 2024-06-13 11:30:00 | 416.76 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2024-06-25 10:35:00 | 444.25 | 2024-06-25 10:45:00 | 446.02 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest1 | 2024-06-27 10:40:00 | 439.10 | 2024-06-27 10:50:00 | 440.42 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2024-06-28 09:30:00 | 435.00 | 2024-06-28 09:35:00 | 436.61 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2024-07-01 09:30:00 | 438.00 | 2024-07-01 09:40:00 | 440.33 | PARTIAL | 0.50 | 0.53% |
| BUY | retest1 | 2024-07-01 09:30:00 | 438.00 | 2024-07-01 11:00:00 | 438.70 | TARGET_HIT | 0.50 | 0.16% |
| SELL | retest1 | 2024-07-12 09:50:00 | 498.80 | 2024-07-12 09:55:00 | 501.07 | STOP_HIT | 1.00 | -0.46% |
| BUY | retest1 | 2024-07-16 09:30:00 | 493.15 | 2024-07-16 09:35:00 | 490.81 | STOP_HIT | 1.00 | -0.47% |
| SELL | retest1 | 2024-08-13 09:35:00 | 512.10 | 2024-08-13 10:15:00 | 508.94 | PARTIAL | 0.50 | 0.62% |
| SELL | retest1 | 2024-08-13 09:35:00 | 512.10 | 2024-08-13 10:40:00 | 512.10 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-08-16 09:50:00 | 533.75 | 2024-08-16 12:40:00 | 537.58 | PARTIAL | 0.50 | 0.72% |
| BUY | retest1 | 2024-08-16 09:50:00 | 533.75 | 2024-08-16 15:20:00 | 584.55 | TARGET_HIT | 0.50 | 9.52% |
| SELL | retest1 | 2024-08-29 09:30:00 | 604.95 | 2024-08-29 09:40:00 | 607.30 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest1 | 2024-10-14 10:45:00 | 760.85 | 2024-10-14 11:00:00 | 757.92 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest1 | 2024-11-06 10:40:00 | 746.35 | 2024-11-06 11:00:00 | 750.56 | STOP_HIT | 1.00 | -0.56% |
| SELL | retest1 | 2024-11-12 09:45:00 | 733.10 | 2024-11-12 11:20:00 | 736.30 | STOP_HIT | 1.00 | -0.44% |
| SELL | retest1 | 2024-11-26 09:40:00 | 662.85 | 2024-11-26 09:50:00 | 658.42 | PARTIAL | 0.50 | 0.67% |
| SELL | retest1 | 2024-11-26 09:40:00 | 662.85 | 2024-11-26 10:15:00 | 662.85 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-11-27 10:35:00 | 672.65 | 2024-11-27 10:40:00 | 670.00 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest1 | 2024-11-29 10:25:00 | 662.90 | 2024-11-29 11:10:00 | 666.27 | STOP_HIT | 1.00 | -0.51% |
| BUY | retest1 | 2024-12-05 10:10:00 | 731.70 | 2024-12-05 10:20:00 | 727.91 | STOP_HIT | 1.00 | -0.52% |
| BUY | retest1 | 2024-12-06 11:00:00 | 734.25 | 2024-12-06 11:25:00 | 738.72 | PARTIAL | 0.50 | 0.61% |
| BUY | retest1 | 2024-12-06 11:00:00 | 734.25 | 2024-12-06 11:45:00 | 734.25 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-12-12 09:55:00 | 741.45 | 2024-12-12 10:10:00 | 745.61 | PARTIAL | 0.50 | 0.56% |
| BUY | retest1 | 2024-12-12 09:55:00 | 741.45 | 2024-12-12 10:25:00 | 741.45 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-12-27 09:40:00 | 830.40 | 2024-12-27 09:45:00 | 826.28 | STOP_HIT | 1.00 | -0.50% |
| BUY | retest1 | 2025-01-20 09:40:00 | 914.60 | 2025-01-20 09:45:00 | 910.34 | STOP_HIT | 1.00 | -0.47% |
| BUY | retest1 | 2025-03-18 09:35:00 | 499.00 | 2025-03-18 09:50:00 | 496.54 | STOP_HIT | 1.00 | -0.49% |
