# Anand Rathi Wealth Ltd. (ANANDRATHI)

## Backtest Summary

- **Window:** 2023-05-12 09:15:00 → 2026-05-08 15:25:00 (50532 bars)
- **Last close:** 3602.30
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
| ENTRY1 | 63 |
| ENTRY2 | 0 |
| PARTIAL | 23 |
| TARGET_HIT | 12 |
| STOP_HIT | 51 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 86 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 35 / 51
- **Target hits / Stop hits / Partials:** 12 / 51 / 23
- **Avg / median % per leg:** 0.22% / 0.00%
- **Sum % (uncompounded):** 18.85%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 30 | 12 | 40.0% | 4 | 18 | 8 | 0.33% | 9.9% |
| BUY @ 2nd Alert (retest1) | 30 | 12 | 40.0% | 4 | 18 | 8 | 0.33% | 9.9% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 56 | 23 | 41.1% | 8 | 33 | 15 | 0.16% | 9.0% |
| SELL @ 2nd Alert (retest1) | 56 | 23 | 41.1% | 8 | 33 | 15 | 0.16% | 9.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 86 | 35 | 40.7% | 12 | 51 | 23 | 0.22% | 18.9% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-05-12 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-05-12 10:55:00 | 436.78 | 438.07 | 0.00 | ORB-short ORB[437.63,442.50] vol=4.4x ATR=1.62 |
| Stop hit — per-position SL triggered | 2023-05-12 11:05:00 | 438.40 | 438.06 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2023-05-16 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-05-16 09:55:00 | 418.93 | 420.42 | 0.00 | ORB-short ORB[420.05,423.68] vol=3.1x ATR=1.35 |
| Stop hit — per-position SL triggered | 2023-05-16 10:45:00 | 420.28 | 419.97 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2023-05-18 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-05-18 09:35:00 | 424.33 | 425.91 | 0.00 | ORB-short ORB[425.50,429.70] vol=3.6x ATR=0.98 |
| Stop hit — per-position SL triggered | 2023-05-18 09:40:00 | 425.31 | 425.83 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2023-05-19 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-05-19 09:30:00 | 425.93 | 427.35 | 0.00 | ORB-short ORB[427.50,430.05] vol=5.8x ATR=1.25 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-05-19 10:05:00 | 424.06 | 426.47 | 0.00 | T1 1.5R @ 424.06 |
| Stop hit — per-position SL triggered | 2023-05-19 10:20:00 | 425.93 | 426.18 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2023-05-23 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-05-23 09:30:00 | 432.88 | 432.50 | 0.00 | ORB-long ORB[429.85,432.18] vol=3.6x ATR=1.17 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-05-23 09:40:00 | 434.64 | 433.26 | 0.00 | T1 1.5R @ 434.64 |
| Stop hit — per-position SL triggered | 2023-05-23 09:50:00 | 432.88 | 433.28 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2023-05-24 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-05-24 10:10:00 | 422.00 | 424.77 | 0.00 | ORB-short ORB[424.50,429.95] vol=2.3x ATR=1.94 |
| Stop hit — per-position SL triggered | 2023-05-24 10:20:00 | 423.94 | 424.61 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2023-05-25 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-05-25 09:35:00 | 425.00 | 425.30 | 0.00 | ORB-short ORB[425.03,426.68] vol=1.8x ATR=0.59 |
| Stop hit — per-position SL triggered | 2023-05-25 09:40:00 | 425.59 | 425.47 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2023-05-26 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-05-26 10:25:00 | 424.48 | 425.17 | 0.00 | ORB-short ORB[425.20,426.98] vol=1.6x ATR=0.66 |
| Stop hit — per-position SL triggered | 2023-05-26 10:35:00 | 425.14 | 425.16 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2023-05-29 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-05-29 09:35:00 | 424.95 | 423.78 | 0.00 | ORB-long ORB[422.53,424.40] vol=3.1x ATR=1.00 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-05-29 11:05:00 | 426.46 | 424.30 | 0.00 | T1 1.5R @ 426.46 |
| Target hit | 2023-05-29 15:20:00 | 430.83 | 426.83 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 10 — BUY (started 2023-05-31 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-05-31 10:05:00 | 430.28 | 429.23 | 0.00 | ORB-long ORB[427.10,430.00] vol=3.5x ATR=1.21 |
| Stop hit — per-position SL triggered | 2023-05-31 10:10:00 | 429.07 | 429.26 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2023-06-02 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-02 10:30:00 | 426.18 | 427.45 | 0.00 | ORB-short ORB[427.18,428.53] vol=5.0x ATR=1.42 |
| Stop hit — per-position SL triggered | 2023-06-02 10:35:00 | 427.60 | 427.35 | 0.00 | SL hit |

### Cycle 12 — SELL (started 2023-06-05 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-05 09:30:00 | 424.95 | 426.11 | 0.00 | ORB-short ORB[427.28,429.50] vol=6.2x ATR=1.08 |
| Stop hit — per-position SL triggered | 2023-06-05 09:45:00 | 426.03 | 425.63 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2023-06-06 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-06 09:30:00 | 426.93 | 425.69 | 0.00 | ORB-long ORB[422.55,426.88] vol=2.9x ATR=0.78 |
| Stop hit — per-position SL triggered | 2023-06-06 09:35:00 | 426.15 | 425.80 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2023-06-09 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-09 11:10:00 | 434.50 | 432.91 | 0.00 | ORB-long ORB[430.08,434.23] vol=14.1x ATR=0.91 |
| Stop hit — per-position SL triggered | 2023-06-09 11:25:00 | 433.59 | 433.49 | 0.00 | SL hit |

### Cycle 15 — BUY (started 2023-06-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-12 10:15:00 | 436.60 | 435.88 | 0.00 | ORB-long ORB[432.50,436.50] vol=3.0x ATR=0.90 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-06-12 10:40:00 | 437.94 | 436.68 | 0.00 | T1 1.5R @ 437.94 |
| Stop hit — per-position SL triggered | 2023-06-12 11:40:00 | 436.60 | 436.96 | 0.00 | SL hit |

### Cycle 16 — SELL (started 2023-06-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-13 10:15:00 | 437.08 | 438.79 | 0.00 | ORB-short ORB[438.68,440.00] vol=2.6x ATR=0.75 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-06-13 10:55:00 | 435.96 | 437.52 | 0.00 | T1 1.5R @ 435.96 |
| Stop hit — per-position SL triggered | 2023-06-13 11:45:00 | 437.08 | 437.13 | 0.00 | SL hit |

### Cycle 17 — SELL (started 2023-06-14 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-14 10:55:00 | 437.50 | 439.48 | 0.00 | ORB-short ORB[438.15,442.65] vol=5.8x ATR=1.44 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-06-14 14:10:00 | 435.34 | 438.39 | 0.00 | T1 1.5R @ 435.34 |
| Target hit | 2023-06-14 15:20:00 | 434.25 | 437.74 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 18 — SELL (started 2023-06-15 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-15 09:35:00 | 432.30 | 433.22 | 0.00 | ORB-short ORB[432.80,435.50] vol=3.3x ATR=1.01 |
| Stop hit — per-position SL triggered | 2023-06-15 09:40:00 | 433.31 | 433.22 | 0.00 | SL hit |

### Cycle 19 — SELL (started 2023-06-16 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-16 09:30:00 | 434.20 | 435.74 | 0.00 | ORB-short ORB[434.53,439.63] vol=3.5x ATR=1.09 |
| Stop hit — per-position SL triggered | 2023-06-16 09:40:00 | 435.29 | 435.65 | 0.00 | SL hit |

### Cycle 20 — SELL (started 2023-06-19 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-19 11:00:00 | 431.70 | 436.02 | 0.00 | ORB-short ORB[434.30,440.08] vol=4.4x ATR=1.54 |
| Stop hit — per-position SL triggered | 2023-06-19 11:40:00 | 433.24 | 435.27 | 0.00 | SL hit |

### Cycle 21 — SELL (started 2023-06-20 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-20 09:30:00 | 429.25 | 430.98 | 0.00 | ORB-short ORB[430.33,433.45] vol=3.5x ATR=1.22 |
| Stop hit — per-position SL triggered | 2023-06-20 09:35:00 | 430.47 | 430.93 | 0.00 | SL hit |

### Cycle 22 — BUY (started 2023-06-21 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-21 10:40:00 | 432.33 | 431.53 | 0.00 | ORB-long ORB[430.30,432.00] vol=2.5x ATR=0.64 |
| Stop hit — per-position SL triggered | 2023-06-21 10:45:00 | 431.69 | 431.53 | 0.00 | SL hit |

### Cycle 23 — BUY (started 2023-06-22 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-22 09:35:00 | 434.30 | 433.23 | 0.00 | ORB-long ORB[430.93,433.35] vol=5.7x ATR=0.92 |
| Stop hit — per-position SL triggered | 2023-06-22 09:40:00 | 433.38 | 433.24 | 0.00 | SL hit |

### Cycle 24 — SELL (started 2023-06-28 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-28 10:30:00 | 431.28 | 431.89 | 0.00 | ORB-short ORB[432.53,434.35] vol=2.1x ATR=1.04 |
| Stop hit — per-position SL triggered | 2023-06-28 10:40:00 | 432.32 | 431.89 | 0.00 | SL hit |

### Cycle 25 — BUY (started 2023-07-27 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-27 11:05:00 | 556.60 | 553.59 | 0.00 | ORB-long ORB[549.03,555.85] vol=2.4x ATR=1.52 |
| Stop hit — per-position SL triggered | 2023-07-27 11:25:00 | 555.08 | 554.21 | 0.00 | SL hit |

### Cycle 26 — SELL (started 2023-07-28 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-07-28 09:55:00 | 554.35 | 554.60 | 0.00 | ORB-short ORB[554.95,562.50] vol=1.9x ATR=2.24 |
| Stop hit — per-position SL triggered | 2023-07-28 10:15:00 | 556.59 | 554.86 | 0.00 | SL hit |

### Cycle 27 — BUY (started 2023-08-01 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-01 11:00:00 | 592.03 | 587.76 | 0.00 | ORB-long ORB[584.48,590.42] vol=2.4x ATR=1.51 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-01 11:20:00 | 594.30 | 589.95 | 0.00 | T1 1.5R @ 594.30 |
| Target hit | 2023-08-01 15:20:00 | 628.13 | 610.42 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 28 — SELL (started 2023-08-23 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-23 11:00:00 | 704.50 | 707.10 | 0.00 | ORB-short ORB[705.53,712.50] vol=1.9x ATR=2.39 |
| Stop hit — per-position SL triggered | 2023-08-23 11:15:00 | 706.89 | 706.96 | 0.00 | SL hit |

### Cycle 29 — SELL (started 2023-08-29 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-29 09:30:00 | 704.20 | 706.21 | 0.00 | ORB-short ORB[705.08,711.98] vol=1.7x ATR=1.84 |
| Stop hit — per-position SL triggered | 2023-08-29 10:40:00 | 706.04 | 705.33 | 0.00 | SL hit |

### Cycle 30 — SELL (started 2023-09-04 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-09-04 10:00:00 | 698.88 | 700.54 | 0.00 | ORB-short ORB[700.00,705.48] vol=4.5x ATR=2.98 |
| Stop hit — per-position SL triggered | 2023-09-04 12:55:00 | 701.86 | 699.63 | 0.00 | SL hit |

### Cycle 31 — SELL (started 2023-09-06 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-09-06 10:45:00 | 688.88 | 689.58 | 0.00 | ORB-short ORB[689.35,696.50] vol=5.0x ATR=1.81 |
| Stop hit — per-position SL triggered | 2023-09-06 11:30:00 | 690.69 | 689.58 | 0.00 | SL hit |

### Cycle 32 — BUY (started 2023-09-14 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-14 11:00:00 | 705.75 | 697.97 | 0.00 | ORB-long ORB[690.98,699.95] vol=3.3x ATR=2.28 |
| Stop hit — per-position SL triggered | 2023-09-14 12:40:00 | 703.47 | 699.84 | 0.00 | SL hit |

### Cycle 33 — BUY (started 2023-09-15 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-15 09:35:00 | 709.68 | 707.26 | 0.00 | ORB-long ORB[703.00,706.98] vol=3.2x ATR=2.48 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-15 09:55:00 | 713.40 | 708.84 | 0.00 | T1 1.5R @ 713.40 |
| Stop hit — per-position SL triggered | 2023-09-15 10:45:00 | 709.68 | 710.07 | 0.00 | SL hit |

### Cycle 34 — BUY (started 2023-09-18 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-18 10:05:00 | 712.80 | 709.71 | 0.00 | ORB-long ORB[705.88,711.98] vol=1.7x ATR=2.04 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-18 10:10:00 | 715.85 | 711.11 | 0.00 | T1 1.5R @ 715.85 |
| Target hit | 2023-09-18 15:20:00 | 732.98 | 731.97 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 35 — BUY (started 2023-09-25 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-25 09:35:00 | 796.13 | 789.44 | 0.00 | ORB-long ORB[781.03,790.98] vol=2.9x ATR=4.46 |
| Stop hit — per-position SL triggered | 2023-09-25 09:50:00 | 791.67 | 790.61 | 0.00 | SL hit |

### Cycle 36 — BUY (started 2023-09-26 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-26 09:50:00 | 825.30 | 819.28 | 0.00 | ORB-long ORB[808.65,817.50] vol=3.5x ATR=4.59 |
| Stop hit — per-position SL triggered | 2023-09-26 09:55:00 | 820.71 | 819.89 | 0.00 | SL hit |

### Cycle 37 — BUY (started 2023-09-28 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-28 09:30:00 | 852.85 | 849.84 | 0.00 | ORB-long ORB[845.13,852.50] vol=1.8x ATR=3.84 |
| Stop hit — per-position SL triggered | 2023-09-28 09:45:00 | 849.01 | 850.07 | 0.00 | SL hit |

### Cycle 38 — SELL (started 2023-11-08 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-11-08 11:10:00 | 1019.28 | 1022.59 | 0.00 | ORB-short ORB[1020.00,1029.38] vol=1.8x ATR=4.04 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-08 11:25:00 | 1013.22 | 1021.54 | 0.00 | T1 1.5R @ 1013.22 |
| Target hit | 2023-11-08 15:20:00 | 1005.23 | 1014.12 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 39 — SELL (started 2023-11-16 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-11-16 09:35:00 | 1056.95 | 1063.23 | 0.00 | ORB-short ORB[1064.25,1074.83] vol=3.3x ATR=3.83 |
| Stop hit — per-position SL triggered | 2023-11-16 09:40:00 | 1060.78 | 1062.73 | 0.00 | SL hit |

### Cycle 40 — BUY (started 2023-11-24 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-24 09:40:00 | 1260.72 | 1250.24 | 0.00 | ORB-long ORB[1236.05,1253.50] vol=1.9x ATR=6.19 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-24 09:45:00 | 1270.00 | 1254.60 | 0.00 | T1 1.5R @ 1270.00 |
| Stop hit — per-position SL triggered | 2023-11-24 10:00:00 | 1260.72 | 1257.48 | 0.00 | SL hit |

### Cycle 41 — SELL (started 2023-12-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-12-05 10:15:00 | 1329.80 | 1336.62 | 0.00 | ORB-short ORB[1330.00,1346.48] vol=1.7x ATR=5.35 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-05 11:35:00 | 1321.78 | 1332.10 | 0.00 | T1 1.5R @ 1321.78 |
| Target hit | 2023-12-05 14:45:00 | 1322.33 | 1321.69 | 0.00 | Trail-exit close>VWAP |

### Cycle 42 — SELL (started 2023-12-08 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-12-08 10:00:00 | 1273.93 | 1281.98 | 0.00 | ORB-short ORB[1275.00,1290.70] vol=3.4x ATR=4.17 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-08 10:05:00 | 1267.68 | 1280.24 | 0.00 | T1 1.5R @ 1267.68 |
| Stop hit — per-position SL triggered | 2023-12-08 10:30:00 | 1273.93 | 1277.93 | 0.00 | SL hit |

### Cycle 43 — SELL (started 2023-12-27 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-12-27 10:05:00 | 1291.83 | 1296.21 | 0.00 | ORB-short ORB[1292.78,1305.00] vol=2.0x ATR=3.67 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-27 12:25:00 | 1286.32 | 1293.61 | 0.00 | T1 1.5R @ 1286.32 |
| Target hit | 2023-12-27 15:20:00 | 1281.47 | 1285.73 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 44 — SELL (started 2024-01-01 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-01 09:50:00 | 1289.03 | 1304.84 | 0.00 | ORB-short ORB[1305.00,1319.03] vol=1.8x ATR=6.12 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-01 10:20:00 | 1279.85 | 1297.22 | 0.00 | T1 1.5R @ 1279.85 |
| Stop hit — per-position SL triggered | 2024-01-01 10:30:00 | 1289.03 | 1296.17 | 0.00 | SL hit |

### Cycle 45 — BUY (started 2024-01-04 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-04 09:35:00 | 1302.43 | 1300.19 | 0.00 | ORB-long ORB[1292.53,1302.00] vol=2.0x ATR=1.97 |
| Stop hit — per-position SL triggered | 2024-01-04 09:40:00 | 1300.46 | 1300.28 | 0.00 | SL hit |

### Cycle 46 — BUY (started 2024-01-12 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-12 10:50:00 | 1420.58 | 1407.46 | 0.00 | ORB-long ORB[1400.00,1414.40] vol=1.5x ATR=4.78 |
| Stop hit — per-position SL triggered | 2024-01-12 10:55:00 | 1415.80 | 1407.95 | 0.00 | SL hit |

### Cycle 47 — SELL (started 2024-01-20 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-20 10:30:00 | 1367.80 | 1373.20 | 0.00 | ORB-short ORB[1391.65,1403.28] vol=8.3x ATR=7.18 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-20 11:20:00 | 1357.03 | 1370.82 | 0.00 | T1 1.5R @ 1357.03 |
| Target hit | 2024-01-20 15:20:00 | 1350.33 | 1359.50 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 48 — SELL (started 2024-03-13 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-03-13 09:30:00 | 1915.50 | 1931.90 | 0.00 | ORB-short ORB[1927.15,1945.00] vol=1.9x ATR=8.40 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-13 09:45:00 | 1902.90 | 1925.42 | 0.00 | T1 1.5R @ 1902.90 |
| Target hit | 2024-03-13 10:30:00 | 1886.03 | 1885.67 | 0.00 | Trail-exit close>VWAP |

### Cycle 49 — SELL (started 2024-03-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-03-20 10:15:00 | 1797.88 | 1808.17 | 0.00 | ORB-short ORB[1816.33,1836.25] vol=1.7x ATR=6.30 |
| Stop hit — per-position SL triggered | 2024-03-20 10:25:00 | 1804.18 | 1807.49 | 0.00 | SL hit |

### Cycle 50 — SELL (started 2024-03-21 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-03-21 10:05:00 | 1794.58 | 1802.51 | 0.00 | ORB-short ORB[1801.43,1819.00] vol=6.1x ATR=7.12 |
| Stop hit — per-position SL triggered | 2024-03-21 10:15:00 | 1801.70 | 1802.40 | 0.00 | SL hit |

### Cycle 51 — SELL (started 2024-03-27 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-03-27 10:45:00 | 1776.60 | 1792.92 | 0.00 | ORB-short ORB[1789.50,1799.98] vol=1.9x ATR=3.94 |
| Stop hit — per-position SL triggered | 2024-03-27 10:50:00 | 1780.54 | 1792.50 | 0.00 | SL hit |

### Cycle 52 — BUY (started 2024-03-28 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-28 10:50:00 | 1834.80 | 1820.78 | 0.00 | ORB-long ORB[1808.35,1824.83] vol=2.5x ATR=7.16 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-28 11:10:00 | 1845.54 | 1826.61 | 0.00 | T1 1.5R @ 1845.54 |
| Target hit | 2024-03-28 13:55:00 | 1835.68 | 1841.92 | 0.00 | Trail-exit close<VWAP |

### Cycle 53 — SELL (started 2024-04-03 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-04-03 10:00:00 | 1723.55 | 1740.39 | 0.00 | ORB-short ORB[1724.50,1748.98] vol=2.4x ATR=10.03 |
| Stop hit — per-position SL triggered | 2024-04-03 10:05:00 | 1733.58 | 1739.58 | 0.00 | SL hit |

### Cycle 54 — SELL (started 2024-04-04 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-04-04 10:05:00 | 1756.03 | 1764.34 | 0.00 | ORB-short ORB[1767.50,1778.50] vol=1.8x ATR=6.23 |
| Stop hit — per-position SL triggered | 2024-04-04 13:50:00 | 1762.26 | 1758.62 | 0.00 | SL hit |

### Cycle 55 — SELL (started 2024-04-05 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-04-05 09:35:00 | 1747.50 | 1754.31 | 0.00 | ORB-short ORB[1755.00,1772.40] vol=2.7x ATR=6.64 |
| Stop hit — per-position SL triggered | 2024-04-05 09:40:00 | 1754.14 | 1754.13 | 0.00 | SL hit |

### Cycle 56 — SELL (started 2024-04-18 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-04-18 09:30:00 | 2043.50 | 2057.68 | 0.00 | ORB-short ORB[2045.00,2069.20] vol=1.8x ATR=9.77 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-18 09:45:00 | 2028.84 | 2049.76 | 0.00 | T1 1.5R @ 2028.84 |
| Stop hit — per-position SL triggered | 2024-04-18 09:55:00 | 2043.50 | 2048.29 | 0.00 | SL hit |

### Cycle 57 — SELL (started 2024-04-23 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-04-23 09:45:00 | 2006.00 | 2017.19 | 0.00 | ORB-short ORB[2007.53,2033.70] vol=2.1x ATR=6.68 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-23 09:50:00 | 1995.98 | 2013.00 | 0.00 | T1 1.5R @ 1995.98 |
| Target hit | 2024-04-23 15:20:00 | 1979.28 | 1992.10 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 58 — SELL (started 2024-04-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-04-29 11:15:00 | 1975.00 | 1984.12 | 0.00 | ORB-short ORB[1979.50,1999.75] vol=1.9x ATR=3.00 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-29 14:35:00 | 1970.50 | 1979.21 | 0.00 | T1 1.5R @ 1970.50 |
| Stop hit — per-position SL triggered | 2024-04-29 15:05:00 | 1975.00 | 1978.88 | 0.00 | SL hit |

### Cycle 59 — BUY (started 2024-04-30 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-30 09:35:00 | 1997.50 | 1990.35 | 0.00 | ORB-long ORB[1982.25,1992.55] vol=2.2x ATR=5.05 |
| Stop hit — per-position SL triggered | 2024-04-30 11:10:00 | 1992.45 | 1995.84 | 0.00 | SL hit |

### Cycle 60 — SELL (started 2024-05-03 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-03 10:25:00 | 1999.23 | 2002.86 | 0.00 | ORB-short ORB[2000.53,2014.40] vol=1.8x ATR=3.94 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-03 11:00:00 | 1993.31 | 2001.54 | 0.00 | T1 1.5R @ 1993.31 |
| Stop hit — per-position SL triggered | 2024-05-03 13:30:00 | 1999.23 | 1991.63 | 0.00 | SL hit |

### Cycle 61 — SELL (started 2024-05-07 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-07 10:30:00 | 1991.00 | 2004.45 | 0.00 | ORB-short ORB[2007.50,2037.00] vol=4.3x ATR=6.25 |
| Stop hit — per-position SL triggered | 2024-05-07 10:40:00 | 1997.25 | 2003.53 | 0.00 | SL hit |

### Cycle 62 — SELL (started 2024-05-09 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-09 09:35:00 | 1988.85 | 1998.08 | 0.00 | ORB-short ORB[1997.53,2016.60] vol=3.2x ATR=4.70 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-09 09:45:00 | 1981.79 | 1995.26 | 0.00 | T1 1.5R @ 1981.79 |
| Target hit | 2024-05-09 15:20:00 | 1947.98 | 1962.77 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 63 — BUY (started 2024-05-10 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-10 09:30:00 | 1965.95 | 1957.67 | 0.00 | ORB-long ORB[1946.18,1962.50] vol=1.7x ATR=4.93 |
| Stop hit — per-position SL triggered | 2024-05-10 09:35:00 | 1961.02 | 1957.37 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2023-05-12 10:55:00 | 436.78 | 2023-05-12 11:05:00 | 438.40 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest1 | 2023-05-16 09:55:00 | 418.93 | 2023-05-16 10:45:00 | 420.28 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2023-05-18 09:35:00 | 424.33 | 2023-05-18 09:40:00 | 425.31 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2023-05-19 09:30:00 | 425.93 | 2023-05-19 10:05:00 | 424.06 | PARTIAL | 0.50 | 0.44% |
| SELL | retest1 | 2023-05-19 09:30:00 | 425.93 | 2023-05-19 10:20:00 | 425.93 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-05-23 09:30:00 | 432.88 | 2023-05-23 09:40:00 | 434.64 | PARTIAL | 0.50 | 0.41% |
| BUY | retest1 | 2023-05-23 09:30:00 | 432.88 | 2023-05-23 09:50:00 | 432.88 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-05-24 10:10:00 | 422.00 | 2023-05-24 10:20:00 | 423.94 | STOP_HIT | 1.00 | -0.46% |
| SELL | retest1 | 2023-05-25 09:35:00 | 425.00 | 2023-05-25 09:40:00 | 425.59 | STOP_HIT | 1.00 | -0.14% |
| SELL | retest1 | 2023-05-26 10:25:00 | 424.48 | 2023-05-26 10:35:00 | 425.14 | STOP_HIT | 1.00 | -0.15% |
| BUY | retest1 | 2023-05-29 09:35:00 | 424.95 | 2023-05-29 11:05:00 | 426.46 | PARTIAL | 0.50 | 0.35% |
| BUY | retest1 | 2023-05-29 09:35:00 | 424.95 | 2023-05-29 15:20:00 | 430.83 | TARGET_HIT | 0.50 | 1.38% |
| BUY | retest1 | 2023-05-31 10:05:00 | 430.28 | 2023-05-31 10:10:00 | 429.07 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2023-06-02 10:30:00 | 426.18 | 2023-06-02 10:35:00 | 427.60 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2023-06-05 09:30:00 | 424.95 | 2023-06-05 09:45:00 | 426.03 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2023-06-06 09:30:00 | 426.93 | 2023-06-06 09:35:00 | 426.15 | STOP_HIT | 1.00 | -0.18% |
| BUY | retest1 | 2023-06-09 11:10:00 | 434.50 | 2023-06-09 11:25:00 | 433.59 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2023-06-12 10:15:00 | 436.60 | 2023-06-12 10:40:00 | 437.94 | PARTIAL | 0.50 | 0.31% |
| BUY | retest1 | 2023-06-12 10:15:00 | 436.60 | 2023-06-12 11:40:00 | 436.60 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-06-13 10:15:00 | 437.08 | 2023-06-13 10:55:00 | 435.96 | PARTIAL | 0.50 | 0.26% |
| SELL | retest1 | 2023-06-13 10:15:00 | 437.08 | 2023-06-13 11:45:00 | 437.08 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-06-14 10:55:00 | 437.50 | 2023-06-14 14:10:00 | 435.34 | PARTIAL | 0.50 | 0.49% |
| SELL | retest1 | 2023-06-14 10:55:00 | 437.50 | 2023-06-14 15:20:00 | 434.25 | TARGET_HIT | 0.50 | 0.74% |
| SELL | retest1 | 2023-06-15 09:35:00 | 432.30 | 2023-06-15 09:40:00 | 433.31 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2023-06-16 09:30:00 | 434.20 | 2023-06-16 09:40:00 | 435.29 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2023-06-19 11:00:00 | 431.70 | 2023-06-19 11:40:00 | 433.24 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2023-06-20 09:30:00 | 429.25 | 2023-06-20 09:35:00 | 430.47 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2023-06-21 10:40:00 | 432.33 | 2023-06-21 10:45:00 | 431.69 | STOP_HIT | 1.00 | -0.15% |
| BUY | retest1 | 2023-06-22 09:35:00 | 434.30 | 2023-06-22 09:40:00 | 433.38 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2023-06-28 10:30:00 | 431.28 | 2023-06-28 10:40:00 | 432.32 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2023-07-27 11:05:00 | 556.60 | 2023-07-27 11:25:00 | 555.08 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2023-07-28 09:55:00 | 554.35 | 2023-07-28 10:15:00 | 556.59 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest1 | 2023-08-01 11:00:00 | 592.03 | 2023-08-01 11:20:00 | 594.30 | PARTIAL | 0.50 | 0.38% |
| BUY | retest1 | 2023-08-01 11:00:00 | 592.03 | 2023-08-01 15:20:00 | 628.13 | TARGET_HIT | 0.50 | 6.10% |
| SELL | retest1 | 2023-08-23 11:00:00 | 704.50 | 2023-08-23 11:15:00 | 706.89 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2023-08-29 09:30:00 | 704.20 | 2023-08-29 10:40:00 | 706.04 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2023-09-04 10:00:00 | 698.88 | 2023-09-04 12:55:00 | 701.86 | STOP_HIT | 1.00 | -0.43% |
| SELL | retest1 | 2023-09-06 10:45:00 | 688.88 | 2023-09-06 11:30:00 | 690.69 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2023-09-14 11:00:00 | 705.75 | 2023-09-14 12:40:00 | 703.47 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2023-09-15 09:35:00 | 709.68 | 2023-09-15 09:55:00 | 713.40 | PARTIAL | 0.50 | 0.52% |
| BUY | retest1 | 2023-09-15 09:35:00 | 709.68 | 2023-09-15 10:45:00 | 709.68 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-09-18 10:05:00 | 712.80 | 2023-09-18 10:10:00 | 715.85 | PARTIAL | 0.50 | 0.43% |
| BUY | retest1 | 2023-09-18 10:05:00 | 712.80 | 2023-09-18 15:20:00 | 732.98 | TARGET_HIT | 0.50 | 2.83% |
| BUY | retest1 | 2023-09-25 09:35:00 | 796.13 | 2023-09-25 09:50:00 | 791.67 | STOP_HIT | 1.00 | -0.56% |
| BUY | retest1 | 2023-09-26 09:50:00 | 825.30 | 2023-09-26 09:55:00 | 820.71 | STOP_HIT | 1.00 | -0.56% |
| BUY | retest1 | 2023-09-28 09:30:00 | 852.85 | 2023-09-28 09:45:00 | 849.01 | STOP_HIT | 1.00 | -0.45% |
| SELL | retest1 | 2023-11-08 11:10:00 | 1019.28 | 2023-11-08 11:25:00 | 1013.22 | PARTIAL | 0.50 | 0.59% |
| SELL | retest1 | 2023-11-08 11:10:00 | 1019.28 | 2023-11-08 15:20:00 | 1005.23 | TARGET_HIT | 0.50 | 1.38% |
| SELL | retest1 | 2023-11-16 09:35:00 | 1056.95 | 2023-11-16 09:40:00 | 1060.78 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2023-11-24 09:40:00 | 1260.72 | 2023-11-24 09:45:00 | 1270.00 | PARTIAL | 0.50 | 0.74% |
| BUY | retest1 | 2023-11-24 09:40:00 | 1260.72 | 2023-11-24 10:00:00 | 1260.72 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-12-05 10:15:00 | 1329.80 | 2023-12-05 11:35:00 | 1321.78 | PARTIAL | 0.50 | 0.60% |
| SELL | retest1 | 2023-12-05 10:15:00 | 1329.80 | 2023-12-05 14:45:00 | 1322.33 | TARGET_HIT | 0.50 | 0.56% |
| SELL | retest1 | 2023-12-08 10:00:00 | 1273.93 | 2023-12-08 10:05:00 | 1267.68 | PARTIAL | 0.50 | 0.49% |
| SELL | retest1 | 2023-12-08 10:00:00 | 1273.93 | 2023-12-08 10:30:00 | 1273.93 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-12-27 10:05:00 | 1291.83 | 2023-12-27 12:25:00 | 1286.32 | PARTIAL | 0.50 | 0.43% |
| SELL | retest1 | 2023-12-27 10:05:00 | 1291.83 | 2023-12-27 15:20:00 | 1281.47 | TARGET_HIT | 0.50 | 0.80% |
| SELL | retest1 | 2024-01-01 09:50:00 | 1289.03 | 2024-01-01 10:20:00 | 1279.85 | PARTIAL | 0.50 | 0.71% |
| SELL | retest1 | 2024-01-01 09:50:00 | 1289.03 | 2024-01-01 10:30:00 | 1289.03 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-01-04 09:35:00 | 1302.43 | 2024-01-04 09:40:00 | 1300.46 | STOP_HIT | 1.00 | -0.15% |
| BUY | retest1 | 2024-01-12 10:50:00 | 1420.58 | 2024-01-12 10:55:00 | 1415.80 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2024-01-20 10:30:00 | 1367.80 | 2024-01-20 11:20:00 | 1357.03 | PARTIAL | 0.50 | 0.79% |
| SELL | retest1 | 2024-01-20 10:30:00 | 1367.80 | 2024-01-20 15:20:00 | 1350.33 | TARGET_HIT | 0.50 | 1.28% |
| SELL | retest1 | 2024-03-13 09:30:00 | 1915.50 | 2024-03-13 09:45:00 | 1902.90 | PARTIAL | 0.50 | 0.66% |
| SELL | retest1 | 2024-03-13 09:30:00 | 1915.50 | 2024-03-13 10:30:00 | 1886.03 | TARGET_HIT | 0.50 | 1.54% |
| SELL | retest1 | 2024-03-20 10:15:00 | 1797.88 | 2024-03-20 10:25:00 | 1804.18 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2024-03-21 10:05:00 | 1794.58 | 2024-03-21 10:15:00 | 1801.70 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest1 | 2024-03-27 10:45:00 | 1776.60 | 2024-03-27 10:50:00 | 1780.54 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2024-03-28 10:50:00 | 1834.80 | 2024-03-28 11:10:00 | 1845.54 | PARTIAL | 0.50 | 0.59% |
| BUY | retest1 | 2024-03-28 10:50:00 | 1834.80 | 2024-03-28 13:55:00 | 1835.68 | TARGET_HIT | 0.50 | 0.05% |
| SELL | retest1 | 2024-04-03 10:00:00 | 1723.55 | 2024-04-03 10:05:00 | 1733.58 | STOP_HIT | 1.00 | -0.58% |
| SELL | retest1 | 2024-04-04 10:05:00 | 1756.03 | 2024-04-04 13:50:00 | 1762.26 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2024-04-05 09:35:00 | 1747.50 | 2024-04-05 09:40:00 | 1754.14 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest1 | 2024-04-18 09:30:00 | 2043.50 | 2024-04-18 09:45:00 | 2028.84 | PARTIAL | 0.50 | 0.72% |
| SELL | retest1 | 2024-04-18 09:30:00 | 2043.50 | 2024-04-18 09:55:00 | 2043.50 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-04-23 09:45:00 | 2006.00 | 2024-04-23 09:50:00 | 1995.98 | PARTIAL | 0.50 | 0.50% |
| SELL | retest1 | 2024-04-23 09:45:00 | 2006.00 | 2024-04-23 15:20:00 | 1979.28 | TARGET_HIT | 0.50 | 1.33% |
| SELL | retest1 | 2024-04-29 11:15:00 | 1975.00 | 2024-04-29 14:35:00 | 1970.50 | PARTIAL | 0.50 | 0.23% |
| SELL | retest1 | 2024-04-29 11:15:00 | 1975.00 | 2024-04-29 15:05:00 | 1975.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-04-30 09:35:00 | 1997.50 | 2024-04-30 11:10:00 | 1992.45 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2024-05-03 10:25:00 | 1999.23 | 2024-05-03 11:00:00 | 1993.31 | PARTIAL | 0.50 | 0.30% |
| SELL | retest1 | 2024-05-03 10:25:00 | 1999.23 | 2024-05-03 13:30:00 | 1999.23 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-05-07 10:30:00 | 1991.00 | 2024-05-07 10:40:00 | 1997.25 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2024-05-09 09:35:00 | 1988.85 | 2024-05-09 09:45:00 | 1981.79 | PARTIAL | 0.50 | 0.35% |
| SELL | retest1 | 2024-05-09 09:35:00 | 1988.85 | 2024-05-09 15:20:00 | 1947.98 | TARGET_HIT | 0.50 | 2.05% |
| BUY | retest1 | 2024-05-10 09:30:00 | 1965.95 | 2024-05-10 09:35:00 | 1961.02 | STOP_HIT | 1.00 | -0.25% |
