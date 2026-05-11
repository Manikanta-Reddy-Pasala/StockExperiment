# Sona BLW Precision Forgings Ltd. (SONACOMS)

## Backtest Summary

- **Window:** 2025-05-12 09:15:00 → 2026-05-08 15:25:00 (18463 bars)
- **Last close:** 579.65
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
| ENTRY1 | 79 |
| ENTRY2 | 0 |
| PARTIAL | 34 |
| TARGET_HIT | 20 |
| STOP_HIT | 59 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 113 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 54 / 59
- **Target hits / Stop hits / Partials:** 20 / 59 / 34
- **Avg / median % per leg:** 0.14% / 0.00%
- **Sum % (uncompounded):** 15.89%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 50 | 22 | 44.0% | 7 | 28 | 15 | 0.09% | 4.6% |
| BUY @ 2nd Alert (retest1) | 50 | 22 | 44.0% | 7 | 28 | 15 | 0.09% | 4.6% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 63 | 32 | 50.8% | 13 | 31 | 19 | 0.18% | 11.3% |
| SELL @ 2nd Alert (retest1) | 63 | 32 | 50.8% | 13 | 31 | 19 | 0.18% | 11.3% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 113 | 54 | 47.8% | 20 | 59 | 34 | 0.14% | 15.9% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-16 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-16 09:55:00 | 537.95 | 533.95 | 0.00 | ORB-long ORB[528.50,535.65] vol=1.9x ATR=1.78 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-16 10:00:00 | 540.62 | 534.77 | 0.00 | T1 1.5R @ 540.62 |
| Target hit | 2025-05-16 11:35:00 | 539.35 | 539.54 | 0.00 | Trail-exit close<VWAP |

### Cycle 2 — BUY (started 2025-05-30 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-30 11:05:00 | 548.00 | 544.42 | 0.00 | ORB-long ORB[540.35,546.00] vol=2.6x ATR=1.43 |
| Stop hit — per-position SL triggered | 2025-05-30 12:00:00 | 546.57 | 545.64 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2025-06-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-02 11:15:00 | 545.10 | 541.53 | 0.00 | ORB-long ORB[534.60,540.75] vol=1.8x ATR=1.07 |
| Stop hit — per-position SL triggered | 2025-06-02 11:25:00 | 544.03 | 541.72 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2025-06-03 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-03 11:05:00 | 537.85 | 538.85 | 0.00 | ORB-short ORB[539.15,543.45] vol=2.8x ATR=1.11 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-03 11:20:00 | 536.18 | 538.36 | 0.00 | T1 1.5R @ 536.18 |
| Target hit | 2025-06-03 15:20:00 | 533.95 | 535.74 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 5 — BUY (started 2025-06-12 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-12 11:05:00 | 525.75 | 522.58 | 0.00 | ORB-long ORB[519.60,525.50] vol=1.9x ATR=1.36 |
| Stop hit — per-position SL triggered | 2025-06-12 11:10:00 | 524.39 | 522.78 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2025-06-19 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-19 10:30:00 | 477.25 | 480.23 | 0.00 | ORB-short ORB[478.80,483.00] vol=3.0x ATR=1.41 |
| Stop hit — per-position SL triggered | 2025-06-19 10:35:00 | 478.66 | 479.97 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2025-06-25 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-25 11:05:00 | 481.00 | 483.70 | 0.00 | ORB-short ORB[482.45,486.40] vol=3.4x ATR=1.41 |
| Stop hit — per-position SL triggered | 2025-06-25 11:20:00 | 482.41 | 483.39 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2025-06-26 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-26 09:50:00 | 478.85 | 480.75 | 0.00 | ORB-short ORB[479.25,483.90] vol=2.0x ATR=1.66 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-26 10:05:00 | 476.36 | 479.93 | 0.00 | T1 1.5R @ 476.36 |
| Stop hit — per-position SL triggered | 2025-06-26 10:10:00 | 478.85 | 479.48 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2025-06-27 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-27 11:05:00 | 482.45 | 483.22 | 0.00 | ORB-short ORB[483.25,485.80] vol=1.5x ATR=1.27 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-27 13:15:00 | 480.54 | 482.51 | 0.00 | T1 1.5R @ 480.54 |
| Target hit | 2025-06-27 15:20:00 | 479.50 | 481.30 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 10 — SELL (started 2025-07-02 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-02 10:55:00 | 472.95 | 476.99 | 0.00 | ORB-short ORB[476.85,481.35] vol=1.8x ATR=1.38 |
| Stop hit — per-position SL triggered | 2025-07-02 11:20:00 | 474.33 | 476.30 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2025-07-07 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-07 10:10:00 | 475.20 | 477.65 | 0.00 | ORB-short ORB[476.05,480.50] vol=2.2x ATR=1.25 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-07 10:45:00 | 473.33 | 476.74 | 0.00 | T1 1.5R @ 473.33 |
| Target hit | 2025-07-07 15:20:00 | 468.15 | 470.89 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 12 — SELL (started 2025-07-08 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-08 10:45:00 | 466.00 | 468.74 | 0.00 | ORB-short ORB[467.00,471.55] vol=1.5x ATR=1.49 |
| Stop hit — per-position SL triggered | 2025-07-08 11:25:00 | 467.49 | 468.00 | 0.00 | SL hit |

### Cycle 13 — SELL (started 2025-07-11 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-11 10:05:00 | 452.00 | 452.60 | 0.00 | ORB-short ORB[452.55,455.80] vol=2.8x ATR=1.05 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-11 10:30:00 | 450.43 | 452.25 | 0.00 | T1 1.5R @ 450.43 |
| Target hit | 2025-07-11 12:35:00 | 451.40 | 451.07 | 0.00 | Trail-exit close>VWAP |

### Cycle 14 — SELL (started 2025-07-15 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-15 09:40:00 | 452.25 | 453.21 | 0.00 | ORB-short ORB[453.10,455.85] vol=6.9x ATR=1.13 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-15 09:50:00 | 450.56 | 453.15 | 0.00 | T1 1.5R @ 450.56 |
| Stop hit — per-position SL triggered | 2025-07-15 10:00:00 | 452.25 | 452.99 | 0.00 | SL hit |

### Cycle 15 — SELL (started 2025-07-18 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-18 10:45:00 | 475.90 | 483.30 | 0.00 | ORB-short ORB[482.35,489.00] vol=3.5x ATR=2.66 |
| Stop hit — per-position SL triggered | 2025-07-18 11:10:00 | 478.56 | 482.47 | 0.00 | SL hit |

### Cycle 16 — SELL (started 2025-07-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-23 11:15:00 | 469.00 | 472.54 | 0.00 | ORB-short ORB[470.45,476.55] vol=2.3x ATR=1.49 |
| Stop hit — per-position SL triggered | 2025-07-23 11:20:00 | 470.49 | 472.49 | 0.00 | SL hit |

### Cycle 17 — SELL (started 2025-07-25 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-25 09:30:00 | 487.65 | 488.54 | 0.00 | ORB-short ORB[487.75,489.95] vol=1.5x ATR=1.08 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-25 09:35:00 | 486.03 | 488.28 | 0.00 | T1 1.5R @ 486.03 |
| Target hit | 2025-07-25 10:40:00 | 487.00 | 486.36 | 0.00 | Trail-exit close>VWAP |

### Cycle 18 — SELL (started 2025-07-30 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-30 09:55:00 | 464.00 | 465.57 | 0.00 | ORB-short ORB[466.45,471.75] vol=1.8x ATR=1.59 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-30 10:25:00 | 461.62 | 464.68 | 0.00 | T1 1.5R @ 461.62 |
| Target hit | 2025-07-30 14:25:00 | 459.75 | 459.09 | 0.00 | Trail-exit close>VWAP |

### Cycle 19 — SELL (started 2025-08-08 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-08 10:55:00 | 441.80 | 445.70 | 0.00 | ORB-short ORB[443.10,449.30] vol=2.2x ATR=1.42 |
| Stop hit — per-position SL triggered | 2025-08-08 11:15:00 | 443.22 | 445.28 | 0.00 | SL hit |

### Cycle 20 — BUY (started 2025-08-13 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-13 10:00:00 | 445.55 | 444.12 | 0.00 | ORB-long ORB[442.05,444.95] vol=1.9x ATR=1.26 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-13 11:05:00 | 447.44 | 445.13 | 0.00 | T1 1.5R @ 447.44 |
| Target hit | 2025-08-13 13:20:00 | 446.45 | 446.49 | 0.00 | Trail-exit close<VWAP |

### Cycle 21 — SELL (started 2025-08-14 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-14 10:55:00 | 446.00 | 447.35 | 0.00 | ORB-short ORB[447.20,451.95] vol=1.8x ATR=0.82 |
| Stop hit — per-position SL triggered | 2025-08-14 11:15:00 | 446.82 | 447.16 | 0.00 | SL hit |

### Cycle 22 — BUY (started 2025-08-19 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-19 10:00:00 | 456.30 | 450.40 | 0.00 | ORB-long ORB[444.15,449.00] vol=3.7x ATR=1.70 |
| Stop hit — per-position SL triggered | 2025-08-19 10:15:00 | 454.60 | 452.63 | 0.00 | SL hit |

### Cycle 23 — SELL (started 2025-08-20 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-20 09:30:00 | 448.95 | 452.89 | 0.00 | ORB-short ORB[451.30,457.75] vol=1.6x ATR=1.95 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-20 09:35:00 | 446.02 | 448.54 | 0.00 | T1 1.5R @ 446.02 |
| Target hit | 2025-08-20 10:00:00 | 448.30 | 448.08 | 0.00 | Trail-exit close>VWAP |

### Cycle 24 — SELL (started 2025-08-29 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-29 09:55:00 | 444.15 | 447.12 | 0.00 | ORB-short ORB[448.00,453.70] vol=1.5x ATR=1.85 |
| Stop hit — per-position SL triggered | 2025-08-29 10:20:00 | 446.00 | 446.63 | 0.00 | SL hit |

### Cycle 25 — BUY (started 2025-09-05 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-05 10:00:00 | 448.00 | 445.07 | 0.00 | ORB-long ORB[439.60,443.75] vol=2.0x ATR=1.58 |
| Stop hit — per-position SL triggered | 2025-09-05 10:10:00 | 446.42 | 445.46 | 0.00 | SL hit |

### Cycle 26 — BUY (started 2025-09-08 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-08 09:55:00 | 452.90 | 449.54 | 0.00 | ORB-long ORB[442.50,447.40] vol=1.7x ATR=1.75 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-08 10:05:00 | 455.52 | 450.54 | 0.00 | T1 1.5R @ 455.52 |
| Target hit | 2025-09-08 12:35:00 | 453.95 | 454.21 | 0.00 | Trail-exit close<VWAP |

### Cycle 27 — SELL (started 2025-09-09 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-09 11:00:00 | 450.35 | 452.25 | 0.00 | ORB-short ORB[451.55,456.00] vol=1.5x ATR=1.12 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-09 11:50:00 | 448.67 | 451.82 | 0.00 | T1 1.5R @ 448.67 |
| Target hit | 2025-09-09 15:20:00 | 446.05 | 448.69 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 28 — SELL (started 2025-09-11 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-11 10:20:00 | 440.75 | 442.92 | 0.00 | ORB-short ORB[445.10,451.40] vol=2.4x ATR=1.26 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-11 11:35:00 | 438.86 | 441.76 | 0.00 | T1 1.5R @ 438.86 |
| Target hit | 2025-09-11 15:20:00 | 440.35 | 440.51 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 29 — BUY (started 2025-09-15 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-15 10:30:00 | 436.40 | 432.58 | 0.00 | ORB-long ORB[430.20,434.10] vol=3.3x ATR=1.30 |
| Stop hit — per-position SL triggered | 2025-09-15 10:35:00 | 435.10 | 432.65 | 0.00 | SL hit |

### Cycle 30 — BUY (started 2025-09-16 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-16 09:30:00 | 436.15 | 435.00 | 0.00 | ORB-long ORB[432.35,435.85] vol=2.1x ATR=0.85 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-16 09:55:00 | 437.42 | 436.20 | 0.00 | T1 1.5R @ 437.42 |
| Stop hit — per-position SL triggered | 2025-09-16 10:10:00 | 436.15 | 436.37 | 0.00 | SL hit |

### Cycle 31 — BUY (started 2025-09-17 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-17 09:30:00 | 436.50 | 435.07 | 0.00 | ORB-long ORB[433.65,436.00] vol=3.0x ATR=0.92 |
| Stop hit — per-position SL triggered | 2025-09-17 09:45:00 | 435.58 | 435.26 | 0.00 | SL hit |

### Cycle 32 — SELL (started 2025-09-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-19 11:15:00 | 435.90 | 438.13 | 0.00 | ORB-short ORB[437.60,441.20] vol=1.9x ATR=0.81 |
| Stop hit — per-position SL triggered | 2025-09-19 11:25:00 | 436.71 | 438.08 | 0.00 | SL hit |

### Cycle 33 — SELL (started 2025-09-22 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-22 10:50:00 | 430.70 | 432.94 | 0.00 | ORB-short ORB[431.60,437.25] vol=4.2x ATR=0.89 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-22 11:10:00 | 429.36 | 432.21 | 0.00 | T1 1.5R @ 429.36 |
| Target hit | 2025-09-22 15:20:00 | 419.95 | 426.42 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 34 — BUY (started 2025-09-30 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-30 10:50:00 | 407.40 | 405.46 | 0.00 | ORB-long ORB[403.35,407.00] vol=5.5x ATR=0.90 |
| Stop hit — per-position SL triggered | 2025-09-30 11:15:00 | 406.50 | 405.77 | 0.00 | SL hit |

### Cycle 35 — BUY (started 2025-10-03 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-03 10:55:00 | 415.85 | 413.26 | 0.00 | ORB-long ORB[410.25,415.35] vol=3.3x ATR=1.06 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-03 11:10:00 | 417.44 | 413.65 | 0.00 | T1 1.5R @ 417.44 |
| Target hit | 2025-10-03 15:20:00 | 420.00 | 416.94 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 36 — SELL (started 2025-10-06 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-06 09:30:00 | 417.95 | 418.98 | 0.00 | ORB-short ORB[418.00,422.00] vol=2.0x ATR=1.00 |
| Stop hit — per-position SL triggered | 2025-10-06 09:40:00 | 418.95 | 418.86 | 0.00 | SL hit |

### Cycle 37 — BUY (started 2025-10-14 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-14 09:45:00 | 442.00 | 439.41 | 0.00 | ORB-long ORB[434.40,440.50] vol=2.8x ATR=1.52 |
| Stop hit — per-position SL triggered | 2025-10-14 10:00:00 | 440.48 | 440.01 | 0.00 | SL hit |

### Cycle 38 — SELL (started 2025-10-20 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-20 10:40:00 | 460.20 | 462.46 | 0.00 | ORB-short ORB[461.15,464.00] vol=1.6x ATR=1.25 |
| Stop hit — per-position SL triggered | 2025-10-20 11:30:00 | 461.45 | 461.94 | 0.00 | SL hit |

### Cycle 39 — BUY (started 2025-10-23 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-23 09:40:00 | 474.05 | 470.58 | 0.00 | ORB-long ORB[467.20,472.95] vol=2.2x ATR=2.19 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-23 09:55:00 | 477.34 | 472.30 | 0.00 | T1 1.5R @ 477.34 |
| Target hit | 2025-10-23 15:10:00 | 476.70 | 477.21 | 0.00 | Trail-exit close<VWAP |

### Cycle 40 — BUY (started 2025-10-24 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-24 11:10:00 | 478.25 | 477.61 | 0.00 | ORB-long ORB[471.80,478.05] vol=3.2x ATR=1.28 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-24 11:55:00 | 480.16 | 477.91 | 0.00 | T1 1.5R @ 480.16 |
| Stop hit — per-position SL triggered | 2025-10-24 12:55:00 | 478.25 | 478.18 | 0.00 | SL hit |

### Cycle 41 — BUY (started 2025-10-27 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-27 09:50:00 | 481.60 | 480.11 | 0.00 | ORB-long ORB[476.25,481.50] vol=1.7x ATR=1.90 |
| Stop hit — per-position SL triggered | 2025-10-27 10:15:00 | 479.70 | 480.19 | 0.00 | SL hit |

### Cycle 42 — SELL (started 2025-10-31 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-31 10:45:00 | 479.70 | 483.10 | 0.00 | ORB-short ORB[482.75,487.20] vol=1.6x ATR=1.15 |
| Stop hit — per-position SL triggered | 2025-10-31 11:00:00 | 480.85 | 482.86 | 0.00 | SL hit |

### Cycle 43 — BUY (started 2025-11-03 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-03 09:40:00 | 478.35 | 476.88 | 0.00 | ORB-long ORB[472.85,477.50] vol=1.9x ATR=1.33 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-03 09:45:00 | 480.34 | 478.05 | 0.00 | T1 1.5R @ 480.34 |
| Stop hit — per-position SL triggered | 2025-11-03 09:50:00 | 478.35 | 478.15 | 0.00 | SL hit |

### Cycle 44 — BUY (started 2025-11-04 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-04 10:10:00 | 484.30 | 481.56 | 0.00 | ORB-long ORB[478.30,482.55] vol=2.4x ATR=1.32 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-04 10:25:00 | 486.28 | 482.59 | 0.00 | T1 1.5R @ 486.28 |
| Stop hit — per-position SL triggered | 2025-11-04 11:05:00 | 484.30 | 483.33 | 0.00 | SL hit |

### Cycle 45 — SELL (started 2025-11-10 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-10 10:30:00 | 487.35 | 489.60 | 0.00 | ORB-short ORB[487.65,493.00] vol=1.6x ATR=1.83 |
| Stop hit — per-position SL triggered | 2025-11-10 10:55:00 | 489.18 | 488.75 | 0.00 | SL hit |

### Cycle 46 — SELL (started 2025-11-12 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-12 10:20:00 | 490.00 | 492.22 | 0.00 | ORB-short ORB[490.40,494.30] vol=1.7x ATR=1.12 |
| Stop hit — per-position SL triggered | 2025-11-12 10:30:00 | 491.12 | 492.11 | 0.00 | SL hit |

### Cycle 47 — SELL (started 2025-11-14 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-14 10:50:00 | 479.25 | 482.65 | 0.00 | ORB-short ORB[485.75,489.45] vol=2.0x ATR=1.44 |
| Stop hit — per-position SL triggered | 2025-11-14 11:05:00 | 480.69 | 482.42 | 0.00 | SL hit |

### Cycle 48 — BUY (started 2025-11-24 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-24 09:55:00 | 506.25 | 505.28 | 0.00 | ORB-long ORB[499.70,505.65] vol=4.7x ATR=1.73 |
| Stop hit — per-position SL triggered | 2025-11-24 10:05:00 | 504.52 | 505.31 | 0.00 | SL hit |

### Cycle 49 — SELL (started 2025-11-25 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-25 11:00:00 | 500.05 | 500.97 | 0.00 | ORB-short ORB[500.45,506.30] vol=2.1x ATR=1.65 |
| Stop hit — per-position SL triggered | 2025-11-25 11:05:00 | 501.70 | 500.97 | 0.00 | SL hit |

### Cycle 50 — BUY (started 2025-11-27 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-27 10:00:00 | 513.35 | 510.46 | 0.00 | ORB-long ORB[508.10,511.95] vol=2.2x ATR=1.53 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-27 10:10:00 | 515.65 | 512.49 | 0.00 | T1 1.5R @ 515.65 |
| Target hit | 2025-11-27 11:30:00 | 515.10 | 515.31 | 0.00 | Trail-exit close<VWAP |

### Cycle 51 — SELL (started 2025-12-03 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-03 10:50:00 | 496.00 | 500.32 | 0.00 | ORB-short ORB[503.20,508.60] vol=2.5x ATR=1.47 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-03 11:15:00 | 493.80 | 499.16 | 0.00 | T1 1.5R @ 493.80 |
| Stop hit — per-position SL triggered | 2025-12-03 11:30:00 | 496.00 | 498.63 | 0.00 | SL hit |

### Cycle 52 — SELL (started 2025-12-05 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-05 10:20:00 | 492.40 | 493.53 | 0.00 | ORB-short ORB[495.00,499.90] vol=2.3x ATR=1.51 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-05 10:25:00 | 490.13 | 493.34 | 0.00 | T1 1.5R @ 490.13 |
| Stop hit — per-position SL triggered | 2025-12-05 10:55:00 | 492.40 | 492.98 | 0.00 | SL hit |

### Cycle 53 — SELL (started 2025-12-10 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-10 11:00:00 | 476.60 | 479.81 | 0.00 | ORB-short ORB[477.70,481.15] vol=1.7x ATR=1.48 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-10 11:25:00 | 474.38 | 479.57 | 0.00 | T1 1.5R @ 474.38 |
| Target hit | 2025-12-10 15:20:00 | 471.55 | 475.81 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 54 — BUY (started 2025-12-12 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-12 09:40:00 | 489.10 | 488.48 | 0.00 | ORB-long ORB[485.20,488.45] vol=1.6x ATR=1.74 |
| Stop hit — per-position SL triggered | 2025-12-12 09:55:00 | 487.36 | 488.45 | 0.00 | SL hit |

### Cycle 55 — SELL (started 2025-12-15 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-15 10:25:00 | 484.35 | 489.29 | 0.00 | ORB-short ORB[489.40,496.00] vol=4.5x ATR=1.61 |
| Stop hit — per-position SL triggered | 2025-12-15 11:05:00 | 485.96 | 487.76 | 0.00 | SL hit |

### Cycle 56 — BUY (started 2025-12-17 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-17 09:30:00 | 490.45 | 485.68 | 0.00 | ORB-long ORB[480.80,486.75] vol=3.0x ATR=1.59 |
| Stop hit — per-position SL triggered | 2025-12-17 09:40:00 | 488.86 | 486.59 | 0.00 | SL hit |

### Cycle 57 — SELL (started 2025-12-18 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-18 09:55:00 | 483.60 | 486.66 | 0.00 | ORB-short ORB[486.10,491.15] vol=3.6x ATR=1.65 |
| Stop hit — per-position SL triggered | 2025-12-18 10:25:00 | 485.25 | 486.04 | 0.00 | SL hit |

### Cycle 58 — BUY (started 2026-01-02 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-02 10:10:00 | 480.95 | 478.55 | 0.00 | ORB-long ORB[474.20,478.50] vol=2.6x ATR=1.19 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-02 10:25:00 | 482.74 | 479.14 | 0.00 | T1 1.5R @ 482.74 |
| Target hit | 2026-01-02 15:00:00 | 486.30 | 487.50 | 0.00 | Trail-exit close<VWAP |

### Cycle 59 — SELL (started 2026-01-08 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-08 10:05:00 | 468.85 | 470.42 | 0.00 | ORB-short ORB[469.65,473.60] vol=5.8x ATR=1.55 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-08 10:20:00 | 466.53 | 469.46 | 0.00 | T1 1.5R @ 466.53 |
| Target hit | 2026-01-08 14:40:00 | 465.35 | 465.15 | 0.00 | Trail-exit close>VWAP |

### Cycle 60 — SELL (started 2026-01-13 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-13 09:30:00 | 456.35 | 458.36 | 0.00 | ORB-short ORB[456.85,459.90] vol=1.5x ATR=1.64 |
| Stop hit — per-position SL triggered | 2026-01-13 09:45:00 | 457.99 | 457.48 | 0.00 | SL hit |

### Cycle 61 — BUY (started 2026-01-14 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-14 10:40:00 | 459.80 | 456.44 | 0.00 | ORB-long ORB[454.55,456.95] vol=1.6x ATR=1.40 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-14 11:00:00 | 461.91 | 457.51 | 0.00 | T1 1.5R @ 461.91 |
| Stop hit — per-position SL triggered | 2026-01-14 14:20:00 | 459.80 | 460.99 | 0.00 | SL hit |

### Cycle 62 — SELL (started 2026-01-19 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-19 10:00:00 | 452.65 | 453.99 | 0.00 | ORB-short ORB[454.60,458.80] vol=7.4x ATR=1.40 |
| Stop hit — per-position SL triggered | 2026-01-19 10:10:00 | 454.05 | 453.93 | 0.00 | SL hit |

### Cycle 63 — SELL (started 2026-01-21 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-21 10:50:00 | 441.00 | 444.14 | 0.00 | ORB-short ORB[442.40,447.65] vol=2.2x ATR=1.76 |
| Stop hit — per-position SL triggered | 2026-01-21 10:55:00 | 442.76 | 443.96 | 0.00 | SL hit |

### Cycle 64 — BUY (started 2026-01-22 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-22 09:35:00 | 454.40 | 450.61 | 0.00 | ORB-long ORB[446.10,450.85] vol=3.0x ATR=1.93 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-22 10:05:00 | 457.29 | 453.36 | 0.00 | T1 1.5R @ 457.29 |
| Stop hit — per-position SL triggered | 2026-01-22 11:10:00 | 454.40 | 455.37 | 0.00 | SL hit |

### Cycle 65 — BUY (started 2026-02-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-01 11:15:00 | 499.50 | 496.93 | 0.00 | ORB-long ORB[490.50,496.35] vol=3.2x ATR=1.38 |
| Stop hit — per-position SL triggered | 2026-02-01 11:25:00 | 498.12 | 496.99 | 0.00 | SL hit |

### Cycle 66 — BUY (started 2026-02-10 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-10 11:00:00 | 532.15 | 530.88 | 0.00 | ORB-long ORB[526.20,531.90] vol=1.6x ATR=1.46 |
| Stop hit — per-position SL triggered | 2026-02-10 11:15:00 | 530.69 | 530.89 | 0.00 | SL hit |

### Cycle 67 — BUY (started 2026-02-11 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-11 10:55:00 | 535.35 | 533.16 | 0.00 | ORB-long ORB[531.10,535.00] vol=1.9x ATR=1.34 |
| Stop hit — per-position SL triggered | 2026-02-11 12:15:00 | 534.01 | 533.96 | 0.00 | SL hit |

### Cycle 68 — BUY (started 2026-02-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-20 11:15:00 | 532.20 | 531.40 | 0.00 | ORB-long ORB[524.20,529.70] vol=17.4x ATR=1.58 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-20 11:55:00 | 534.57 | 531.46 | 0.00 | T1 1.5R @ 534.57 |
| Stop hit — per-position SL triggered | 2026-02-20 14:05:00 | 532.20 | 531.83 | 0.00 | SL hit |

### Cycle 69 — BUY (started 2026-02-25 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-25 09:45:00 | 534.90 | 532.13 | 0.00 | ORB-long ORB[529.60,532.15] vol=2.8x ATR=1.52 |
| Stop hit — per-position SL triggered | 2026-02-25 09:50:00 | 533.38 | 532.29 | 0.00 | SL hit |

### Cycle 70 — SELL (started 2026-02-26 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-26 11:05:00 | 539.15 | 540.26 | 0.00 | ORB-short ORB[539.25,545.00] vol=1.6x ATR=1.45 |
| Stop hit — per-position SL triggered | 2026-02-26 11:35:00 | 540.60 | 540.16 | 0.00 | SL hit |

### Cycle 71 — SELL (started 2026-02-27 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-27 09:30:00 | 537.90 | 540.52 | 0.00 | ORB-short ORB[538.50,545.85] vol=1.7x ATR=1.49 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-27 09:50:00 | 535.66 | 539.02 | 0.00 | T1 1.5R @ 535.66 |
| Stop hit — per-position SL triggered | 2026-02-27 10:40:00 | 537.90 | 537.53 | 0.00 | SL hit |

### Cycle 72 — BUY (started 2026-03-10 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-10 10:30:00 | 516.10 | 511.22 | 0.00 | ORB-long ORB[505.70,513.20] vol=1.8x ATR=2.21 |
| Stop hit — per-position SL triggered | 2026-03-10 12:05:00 | 513.89 | 513.13 | 0.00 | SL hit |

### Cycle 73 — SELL (started 2026-03-13 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-13 10:40:00 | 499.90 | 504.40 | 0.00 | ORB-short ORB[504.65,511.50] vol=1.7x ATR=1.94 |
| Stop hit — per-position SL triggered | 2026-03-13 10:45:00 | 501.84 | 504.13 | 0.00 | SL hit |

### Cycle 74 — SELL (started 2026-03-16 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-16 10:55:00 | 478.20 | 482.75 | 0.00 | ORB-short ORB[480.65,485.55] vol=1.7x ATR=2.26 |
| Stop hit — per-position SL triggered | 2026-03-16 11:15:00 | 480.46 | 482.65 | 0.00 | SL hit |

### Cycle 75 — BUY (started 2026-04-10 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-10 10:05:00 | 536.55 | 533.59 | 0.00 | ORB-long ORB[526.15,533.65] vol=1.7x ATR=2.33 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-10 10:15:00 | 540.04 | 534.77 | 0.00 | T1 1.5R @ 540.04 |
| Stop hit — per-position SL triggered | 2026-04-10 10:35:00 | 536.55 | 535.35 | 0.00 | SL hit |

### Cycle 76 — SELL (started 2026-04-23 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-23 10:10:00 | 582.95 | 584.15 | 0.00 | ORB-short ORB[583.00,587.90] vol=2.7x ATR=1.96 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-23 10:20:00 | 580.01 | 583.57 | 0.00 | T1 1.5R @ 580.01 |
| Target hit | 2026-04-23 15:20:00 | 574.05 | 578.08 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 77 — BUY (started 2026-04-28 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-28 11:00:00 | 591.75 | 588.93 | 0.00 | ORB-long ORB[586.00,591.60] vol=2.6x ATR=1.53 |
| Stop hit — per-position SL triggered | 2026-04-28 11:05:00 | 590.22 | 589.16 | 0.00 | SL hit |

### Cycle 78 — BUY (started 2026-05-05 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-05 09:50:00 | 580.00 | 576.19 | 0.00 | ORB-long ORB[569.35,577.15] vol=2.1x ATR=2.89 |
| Stop hit — per-position SL triggered | 2026-05-05 09:55:00 | 577.11 | 576.27 | 0.00 | SL hit |

### Cycle 79 — SELL (started 2026-05-07 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-07 09:55:00 | 580.00 | 583.54 | 0.00 | ORB-short ORB[580.95,588.00] vol=2.4x ATR=2.23 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-07 10:10:00 | 576.65 | 582.79 | 0.00 | T1 1.5R @ 576.65 |
| Stop hit — per-position SL triggered | 2026-05-07 12:00:00 | 580.00 | 579.43 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-05-16 09:55:00 | 537.95 | 2025-05-16 10:00:00 | 540.62 | PARTIAL | 0.50 | 0.50% |
| BUY | retest1 | 2025-05-16 09:55:00 | 537.95 | 2025-05-16 11:35:00 | 539.35 | TARGET_HIT | 0.50 | 0.26% |
| BUY | retest1 | 2025-05-30 11:05:00 | 548.00 | 2025-05-30 12:00:00 | 546.57 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2025-06-02 11:15:00 | 545.10 | 2025-06-02 11:25:00 | 544.03 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2025-06-03 11:05:00 | 537.85 | 2025-06-03 11:20:00 | 536.18 | PARTIAL | 0.50 | 0.31% |
| SELL | retest1 | 2025-06-03 11:05:00 | 537.85 | 2025-06-03 15:20:00 | 533.95 | TARGET_HIT | 0.50 | 0.73% |
| BUY | retest1 | 2025-06-12 11:05:00 | 525.75 | 2025-06-12 11:10:00 | 524.39 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2025-06-19 10:30:00 | 477.25 | 2025-06-19 10:35:00 | 478.66 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2025-06-25 11:05:00 | 481.00 | 2025-06-25 11:20:00 | 482.41 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2025-06-26 09:50:00 | 478.85 | 2025-06-26 10:05:00 | 476.36 | PARTIAL | 0.50 | 0.52% |
| SELL | retest1 | 2025-06-26 09:50:00 | 478.85 | 2025-06-26 10:10:00 | 478.85 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-06-27 11:05:00 | 482.45 | 2025-06-27 13:15:00 | 480.54 | PARTIAL | 0.50 | 0.40% |
| SELL | retest1 | 2025-06-27 11:05:00 | 482.45 | 2025-06-27 15:20:00 | 479.50 | TARGET_HIT | 0.50 | 0.61% |
| SELL | retest1 | 2025-07-02 10:55:00 | 472.95 | 2025-07-02 11:20:00 | 474.33 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2025-07-07 10:10:00 | 475.20 | 2025-07-07 10:45:00 | 473.33 | PARTIAL | 0.50 | 0.39% |
| SELL | retest1 | 2025-07-07 10:10:00 | 475.20 | 2025-07-07 15:20:00 | 468.15 | TARGET_HIT | 0.50 | 1.48% |
| SELL | retest1 | 2025-07-08 10:45:00 | 466.00 | 2025-07-08 11:25:00 | 467.49 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2025-07-11 10:05:00 | 452.00 | 2025-07-11 10:30:00 | 450.43 | PARTIAL | 0.50 | 0.35% |
| SELL | retest1 | 2025-07-11 10:05:00 | 452.00 | 2025-07-11 12:35:00 | 451.40 | TARGET_HIT | 0.50 | 0.13% |
| SELL | retest1 | 2025-07-15 09:40:00 | 452.25 | 2025-07-15 09:50:00 | 450.56 | PARTIAL | 0.50 | 0.37% |
| SELL | retest1 | 2025-07-15 09:40:00 | 452.25 | 2025-07-15 10:00:00 | 452.25 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-07-18 10:45:00 | 475.90 | 2025-07-18 11:10:00 | 478.56 | STOP_HIT | 1.00 | -0.56% |
| SELL | retest1 | 2025-07-23 11:15:00 | 469.00 | 2025-07-23 11:20:00 | 470.49 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2025-07-25 09:30:00 | 487.65 | 2025-07-25 09:35:00 | 486.03 | PARTIAL | 0.50 | 0.33% |
| SELL | retest1 | 2025-07-25 09:30:00 | 487.65 | 2025-07-25 10:40:00 | 487.00 | TARGET_HIT | 0.50 | 0.13% |
| SELL | retest1 | 2025-07-30 09:55:00 | 464.00 | 2025-07-30 10:25:00 | 461.62 | PARTIAL | 0.50 | 0.51% |
| SELL | retest1 | 2025-07-30 09:55:00 | 464.00 | 2025-07-30 14:25:00 | 459.75 | TARGET_HIT | 0.50 | 0.92% |
| SELL | retest1 | 2025-08-08 10:55:00 | 441.80 | 2025-08-08 11:15:00 | 443.22 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2025-08-13 10:00:00 | 445.55 | 2025-08-13 11:05:00 | 447.44 | PARTIAL | 0.50 | 0.42% |
| BUY | retest1 | 2025-08-13 10:00:00 | 445.55 | 2025-08-13 13:20:00 | 446.45 | TARGET_HIT | 0.50 | 0.20% |
| SELL | retest1 | 2025-08-14 10:55:00 | 446.00 | 2025-08-14 11:15:00 | 446.82 | STOP_HIT | 1.00 | -0.18% |
| BUY | retest1 | 2025-08-19 10:00:00 | 456.30 | 2025-08-19 10:15:00 | 454.60 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest1 | 2025-08-20 09:30:00 | 448.95 | 2025-08-20 09:35:00 | 446.02 | PARTIAL | 0.50 | 0.65% |
| SELL | retest1 | 2025-08-20 09:30:00 | 448.95 | 2025-08-20 10:00:00 | 448.30 | TARGET_HIT | 0.50 | 0.14% |
| SELL | retest1 | 2025-08-29 09:55:00 | 444.15 | 2025-08-29 10:20:00 | 446.00 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest1 | 2025-09-05 10:00:00 | 448.00 | 2025-09-05 10:10:00 | 446.42 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2025-09-08 09:55:00 | 452.90 | 2025-09-08 10:05:00 | 455.52 | PARTIAL | 0.50 | 0.58% |
| BUY | retest1 | 2025-09-08 09:55:00 | 452.90 | 2025-09-08 12:35:00 | 453.95 | TARGET_HIT | 0.50 | 0.23% |
| SELL | retest1 | 2025-09-09 11:00:00 | 450.35 | 2025-09-09 11:50:00 | 448.67 | PARTIAL | 0.50 | 0.37% |
| SELL | retest1 | 2025-09-09 11:00:00 | 450.35 | 2025-09-09 15:20:00 | 446.05 | TARGET_HIT | 0.50 | 0.95% |
| SELL | retest1 | 2025-09-11 10:20:00 | 440.75 | 2025-09-11 11:35:00 | 438.86 | PARTIAL | 0.50 | 0.43% |
| SELL | retest1 | 2025-09-11 10:20:00 | 440.75 | 2025-09-11 15:20:00 | 440.35 | TARGET_HIT | 0.50 | 0.09% |
| BUY | retest1 | 2025-09-15 10:30:00 | 436.40 | 2025-09-15 10:35:00 | 435.10 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2025-09-16 09:30:00 | 436.15 | 2025-09-16 09:55:00 | 437.42 | PARTIAL | 0.50 | 0.29% |
| BUY | retest1 | 2025-09-16 09:30:00 | 436.15 | 2025-09-16 10:10:00 | 436.15 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-09-17 09:30:00 | 436.50 | 2025-09-17 09:45:00 | 435.58 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2025-09-19 11:15:00 | 435.90 | 2025-09-19 11:25:00 | 436.71 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest1 | 2025-09-22 10:50:00 | 430.70 | 2025-09-22 11:10:00 | 429.36 | PARTIAL | 0.50 | 0.31% |
| SELL | retest1 | 2025-09-22 10:50:00 | 430.70 | 2025-09-22 15:20:00 | 419.95 | TARGET_HIT | 0.50 | 2.50% |
| BUY | retest1 | 2025-09-30 10:50:00 | 407.40 | 2025-09-30 11:15:00 | 406.50 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2025-10-03 10:55:00 | 415.85 | 2025-10-03 11:10:00 | 417.44 | PARTIAL | 0.50 | 0.38% |
| BUY | retest1 | 2025-10-03 10:55:00 | 415.85 | 2025-10-03 15:20:00 | 420.00 | TARGET_HIT | 0.50 | 1.00% |
| SELL | retest1 | 2025-10-06 09:30:00 | 417.95 | 2025-10-06 09:40:00 | 418.95 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2025-10-14 09:45:00 | 442.00 | 2025-10-14 10:00:00 | 440.48 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2025-10-20 10:40:00 | 460.20 | 2025-10-20 11:30:00 | 461.45 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2025-10-23 09:40:00 | 474.05 | 2025-10-23 09:55:00 | 477.34 | PARTIAL | 0.50 | 0.69% |
| BUY | retest1 | 2025-10-23 09:40:00 | 474.05 | 2025-10-23 15:10:00 | 476.70 | TARGET_HIT | 0.50 | 0.56% |
| BUY | retest1 | 2025-10-24 11:10:00 | 478.25 | 2025-10-24 11:55:00 | 480.16 | PARTIAL | 0.50 | 0.40% |
| BUY | retest1 | 2025-10-24 11:10:00 | 478.25 | 2025-10-24 12:55:00 | 478.25 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-10-27 09:50:00 | 481.60 | 2025-10-27 10:15:00 | 479.70 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest1 | 2025-10-31 10:45:00 | 479.70 | 2025-10-31 11:00:00 | 480.85 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2025-11-03 09:40:00 | 478.35 | 2025-11-03 09:45:00 | 480.34 | PARTIAL | 0.50 | 0.42% |
| BUY | retest1 | 2025-11-03 09:40:00 | 478.35 | 2025-11-03 09:50:00 | 478.35 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-11-04 10:10:00 | 484.30 | 2025-11-04 10:25:00 | 486.28 | PARTIAL | 0.50 | 0.41% |
| BUY | retest1 | 2025-11-04 10:10:00 | 484.30 | 2025-11-04 11:05:00 | 484.30 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-11-10 10:30:00 | 487.35 | 2025-11-10 10:55:00 | 489.18 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest1 | 2025-11-12 10:20:00 | 490.00 | 2025-11-12 10:30:00 | 491.12 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2025-11-14 10:50:00 | 479.25 | 2025-11-14 11:05:00 | 480.69 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2025-11-24 09:55:00 | 506.25 | 2025-11-24 10:05:00 | 504.52 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2025-11-25 11:00:00 | 500.05 | 2025-11-25 11:05:00 | 501.70 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2025-11-27 10:00:00 | 513.35 | 2025-11-27 10:10:00 | 515.65 | PARTIAL | 0.50 | 0.45% |
| BUY | retest1 | 2025-11-27 10:00:00 | 513.35 | 2025-11-27 11:30:00 | 515.10 | TARGET_HIT | 0.50 | 0.34% |
| SELL | retest1 | 2025-12-03 10:50:00 | 496.00 | 2025-12-03 11:15:00 | 493.80 | PARTIAL | 0.50 | 0.44% |
| SELL | retest1 | 2025-12-03 10:50:00 | 496.00 | 2025-12-03 11:30:00 | 496.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-12-05 10:20:00 | 492.40 | 2025-12-05 10:25:00 | 490.13 | PARTIAL | 0.50 | 0.46% |
| SELL | retest1 | 2025-12-05 10:20:00 | 492.40 | 2025-12-05 10:55:00 | 492.40 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-12-10 11:00:00 | 476.60 | 2025-12-10 11:25:00 | 474.38 | PARTIAL | 0.50 | 0.46% |
| SELL | retest1 | 2025-12-10 11:00:00 | 476.60 | 2025-12-10 15:20:00 | 471.55 | TARGET_HIT | 0.50 | 1.06% |
| BUY | retest1 | 2025-12-12 09:40:00 | 489.10 | 2025-12-12 09:55:00 | 487.36 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2025-12-15 10:25:00 | 484.35 | 2025-12-15 11:05:00 | 485.96 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2025-12-17 09:30:00 | 490.45 | 2025-12-17 09:40:00 | 488.86 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2025-12-18 09:55:00 | 483.60 | 2025-12-18 10:25:00 | 485.25 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2026-01-02 10:10:00 | 480.95 | 2026-01-02 10:25:00 | 482.74 | PARTIAL | 0.50 | 0.37% |
| BUY | retest1 | 2026-01-02 10:10:00 | 480.95 | 2026-01-02 15:00:00 | 486.30 | TARGET_HIT | 0.50 | 1.11% |
| SELL | retest1 | 2026-01-08 10:05:00 | 468.85 | 2026-01-08 10:20:00 | 466.53 | PARTIAL | 0.50 | 0.50% |
| SELL | retest1 | 2026-01-08 10:05:00 | 468.85 | 2026-01-08 14:40:00 | 465.35 | TARGET_HIT | 0.50 | 0.75% |
| SELL | retest1 | 2026-01-13 09:30:00 | 456.35 | 2026-01-13 09:45:00 | 457.99 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2026-01-14 10:40:00 | 459.80 | 2026-01-14 11:00:00 | 461.91 | PARTIAL | 0.50 | 0.46% |
| BUY | retest1 | 2026-01-14 10:40:00 | 459.80 | 2026-01-14 14:20:00 | 459.80 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-01-19 10:00:00 | 452.65 | 2026-01-19 10:10:00 | 454.05 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2026-01-21 10:50:00 | 441.00 | 2026-01-21 10:55:00 | 442.76 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest1 | 2026-01-22 09:35:00 | 454.40 | 2026-01-22 10:05:00 | 457.29 | PARTIAL | 0.50 | 0.64% |
| BUY | retest1 | 2026-01-22 09:35:00 | 454.40 | 2026-01-22 11:10:00 | 454.40 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-01 11:15:00 | 499.50 | 2026-02-01 11:25:00 | 498.12 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2026-02-10 11:00:00 | 532.15 | 2026-02-10 11:15:00 | 530.69 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2026-02-11 10:55:00 | 535.35 | 2026-02-11 12:15:00 | 534.01 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2026-02-20 11:15:00 | 532.20 | 2026-02-20 11:55:00 | 534.57 | PARTIAL | 0.50 | 0.45% |
| BUY | retest1 | 2026-02-20 11:15:00 | 532.20 | 2026-02-20 14:05:00 | 532.20 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-25 09:45:00 | 534.90 | 2026-02-25 09:50:00 | 533.38 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2026-02-26 11:05:00 | 539.15 | 2026-02-26 11:35:00 | 540.60 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2026-02-27 09:30:00 | 537.90 | 2026-02-27 09:50:00 | 535.66 | PARTIAL | 0.50 | 0.42% |
| SELL | retest1 | 2026-02-27 09:30:00 | 537.90 | 2026-02-27 10:40:00 | 537.90 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-03-10 10:30:00 | 516.10 | 2026-03-10 12:05:00 | 513.89 | STOP_HIT | 1.00 | -0.43% |
| SELL | retest1 | 2026-03-13 10:40:00 | 499.90 | 2026-03-13 10:45:00 | 501.84 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest1 | 2026-03-16 10:55:00 | 478.20 | 2026-03-16 11:15:00 | 480.46 | STOP_HIT | 1.00 | -0.47% |
| BUY | retest1 | 2026-04-10 10:05:00 | 536.55 | 2026-04-10 10:15:00 | 540.04 | PARTIAL | 0.50 | 0.65% |
| BUY | retest1 | 2026-04-10 10:05:00 | 536.55 | 2026-04-10 10:35:00 | 536.55 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-04-23 10:10:00 | 582.95 | 2026-04-23 10:20:00 | 580.01 | PARTIAL | 0.50 | 0.50% |
| SELL | retest1 | 2026-04-23 10:10:00 | 582.95 | 2026-04-23 15:20:00 | 574.05 | TARGET_HIT | 0.50 | 1.53% |
| BUY | retest1 | 2026-04-28 11:00:00 | 591.75 | 2026-04-28 11:05:00 | 590.22 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2026-05-05 09:50:00 | 580.00 | 2026-05-05 09:55:00 | 577.11 | STOP_HIT | 1.00 | -0.50% |
| SELL | retest1 | 2026-05-07 09:55:00 | 580.00 | 2026-05-07 10:10:00 | 576.65 | PARTIAL | 0.50 | 0.58% |
| SELL | retest1 | 2026-05-07 09:55:00 | 580.00 | 2026-05-07 12:00:00 | 580.00 | STOP_HIT | 0.50 | 0.00% |
