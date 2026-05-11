# Motilal Oswal Financial Services Ltd. (MOTILALOFS)

## Backtest Summary

- **Window:** 2024-05-13 09:15:00 → 2026-05-08 15:25:00 (35371 bars)
- **Last close:** 882.20
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
| ENTRY1 | 25 |
| ENTRY2 | 0 |
| PARTIAL | 8 |
| TARGET_HIT | 3 |
| STOP_HIT | 22 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 33 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 11 / 22
- **Target hits / Stop hits / Partials:** 3 / 22 / 8
- **Avg / median % per leg:** 0.03% / -0.26%
- **Sum % (uncompounded):** 1.15%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 17 | 6 | 35.3% | 1 | 11 | 5 | 0.05% | 0.8% |
| BUY @ 2nd Alert (retest1) | 17 | 6 | 35.3% | 1 | 11 | 5 | 0.05% | 0.8% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 16 | 5 | 31.2% | 2 | 11 | 3 | 0.02% | 0.4% |
| SELL @ 2nd Alert (retest1) | 16 | 5 | 31.2% | 2 | 11 | 3 | 0.02% | 0.4% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 33 | 11 | 33.3% | 3 | 22 | 8 | 0.03% | 1.2% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-05-15 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-15 11:05:00 | 564.00 | 565.98 | 0.00 | ORB-short ORB[565.51,572.99] vol=1.6x ATR=1.84 |
| Stop hit — per-position SL triggered | 2024-05-15 12:00:00 | 565.84 | 565.80 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2024-05-17 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-17 10:05:00 | 575.26 | 581.52 | 0.00 | ORB-short ORB[581.25,587.23] vol=1.7x ATR=3.07 |
| Stop hit — per-position SL triggered | 2024-05-17 10:15:00 | 578.33 | 581.28 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2024-05-24 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-24 10:00:00 | 561.33 | 565.33 | 0.00 | ORB-short ORB[563.94,571.49] vol=4.0x ATR=2.94 |
| Stop hit — per-position SL triggered | 2024-05-24 10:30:00 | 564.27 | 564.71 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2024-05-28 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-28 10:10:00 | 558.25 | 563.34 | 0.00 | ORB-short ORB[566.67,570.00] vol=1.5x ATR=2.06 |
| Stop hit — per-position SL triggered | 2024-05-28 10:20:00 | 560.31 | 562.96 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2024-05-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-30 11:15:00 | 570.99 | 575.16 | 0.00 | ORB-short ORB[572.49,579.98] vol=2.4x ATR=1.51 |
| Stop hit — per-position SL triggered | 2024-05-30 11:35:00 | 572.50 | 575.00 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2024-05-31 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-31 10:20:00 | 562.14 | 570.18 | 0.00 | ORB-short ORB[568.75,575.00] vol=1.8x ATR=2.74 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-31 13:05:00 | 558.03 | 565.94 | 0.00 | T1 1.5R @ 558.03 |
| Target hit | 2024-05-31 15:20:00 | 555.00 | 561.00 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 7 — BUY (started 2024-06-12 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-12 09:30:00 | 670.45 | 666.26 | 0.00 | ORB-long ORB[661.00,669.00] vol=1.6x ATR=3.16 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-12 09:40:00 | 675.19 | 669.09 | 0.00 | T1 1.5R @ 675.19 |
| Stop hit — per-position SL triggered | 2024-06-12 10:30:00 | 670.45 | 670.14 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2024-06-25 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-25 09:30:00 | 645.80 | 648.77 | 0.00 | ORB-short ORB[646.00,654.00] vol=2.1x ATR=2.32 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-25 09:40:00 | 642.31 | 647.03 | 0.00 | T1 1.5R @ 642.31 |
| Target hit | 2024-06-25 15:20:00 | 633.25 | 636.61 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 9 — SELL (started 2024-07-05 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-05 09:45:00 | 554.10 | 559.78 | 0.00 | ORB-short ORB[560.10,568.45] vol=3.3x ATR=2.30 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-05 10:05:00 | 550.65 | 556.85 | 0.00 | T1 1.5R @ 550.65 |
| Stop hit — per-position SL triggered | 2024-07-05 11:40:00 | 554.10 | 554.43 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2024-07-09 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-09 09:45:00 | 556.40 | 554.46 | 0.00 | ORB-long ORB[550.45,556.00] vol=2.6x ATR=2.19 |
| Stop hit — per-position SL triggered | 2024-07-09 09:50:00 | 554.21 | 554.51 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2024-07-12 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-12 09:30:00 | 541.55 | 539.36 | 0.00 | ORB-long ORB[535.25,541.50] vol=2.2x ATR=1.49 |
| Stop hit — per-position SL triggered | 2024-07-12 09:45:00 | 540.06 | 540.01 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2024-07-30 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-30 09:40:00 | 617.00 | 612.42 | 0.00 | ORB-long ORB[606.75,614.00] vol=2.3x ATR=2.32 |
| Stop hit — per-position SL triggered | 2024-07-30 09:50:00 | 614.68 | 613.73 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2024-08-19 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-19 09:35:00 | 635.95 | 633.93 | 0.00 | ORB-long ORB[627.15,635.00] vol=2.0x ATR=2.39 |
| Stop hit — per-position SL triggered | 2024-08-19 10:00:00 | 633.56 | 634.37 | 0.00 | SL hit |

### Cycle 14 — SELL (started 2024-09-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-05 10:15:00 | 755.00 | 767.80 | 0.00 | ORB-short ORB[767.70,776.60] vol=1.7x ATR=4.08 |
| Stop hit — per-position SL triggered | 2024-09-05 11:00:00 | 759.08 | 765.17 | 0.00 | SL hit |

### Cycle 15 — BUY (started 2024-09-11 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-11 09:35:00 | 748.00 | 745.56 | 0.00 | ORB-long ORB[736.05,747.00] vol=2.3x ATR=3.52 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-11 09:40:00 | 753.27 | 746.44 | 0.00 | T1 1.5R @ 753.27 |
| Stop hit — per-position SL triggered | 2024-09-11 10:00:00 | 748.00 | 749.03 | 0.00 | SL hit |

### Cycle 16 — BUY (started 2024-10-11 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-11 09:40:00 | 782.00 | 774.99 | 0.00 | ORB-long ORB[770.15,778.80] vol=2.8x ATR=3.74 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-11 09:45:00 | 787.60 | 777.79 | 0.00 | T1 1.5R @ 787.60 |
| Stop hit — per-position SL triggered | 2024-10-11 10:00:00 | 782.00 | 781.00 | 0.00 | SL hit |

### Cycle 17 — BUY (started 2024-11-29 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-29 09:45:00 | 912.15 | 907.83 | 0.00 | ORB-long ORB[902.00,910.95] vol=1.6x ATR=3.96 |
| Stop hit — per-position SL triggered | 2024-11-29 10:05:00 | 908.19 | 908.95 | 0.00 | SL hit |

### Cycle 18 — SELL (started 2024-12-06 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-06 09:30:00 | 969.60 | 976.68 | 0.00 | ORB-short ORB[972.05,986.50] vol=1.6x ATR=4.87 |
| Stop hit — per-position SL triggered | 2024-12-06 10:00:00 | 974.47 | 974.18 | 0.00 | SL hit |

### Cycle 19 — SELL (started 2024-12-26 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-26 09:30:00 | 915.75 | 920.93 | 0.00 | ORB-short ORB[916.50,928.75] vol=2.0x ATR=3.50 |
| Stop hit — per-position SL triggered | 2024-12-26 10:00:00 | 919.25 | 919.25 | 0.00 | SL hit |

### Cycle 20 — BUY (started 2025-01-01 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-01 09:50:00 | 967.45 | 956.10 | 0.00 | ORB-long ORB[945.25,959.00] vol=1.5x ATR=5.52 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-01 10:00:00 | 975.74 | 962.53 | 0.00 | T1 1.5R @ 975.74 |
| Stop hit — per-position SL triggered | 2025-01-01 10:05:00 | 967.45 | 963.01 | 0.00 | SL hit |

### Cycle 21 — BUY (started 2025-01-02 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-02 09:30:00 | 982.60 | 975.63 | 0.00 | ORB-long ORB[967.10,978.00] vol=3.0x ATR=4.17 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-02 09:40:00 | 988.85 | 983.23 | 0.00 | T1 1.5R @ 988.85 |
| Target hit | 2025-01-02 10:05:00 | 983.35 | 984.63 | 0.00 | Trail-exit close<VWAP |

### Cycle 22 — SELL (started 2025-01-10 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-10 09:55:00 | 885.20 | 896.97 | 0.00 | ORB-short ORB[898.30,910.30] vol=2.8x ATR=6.07 |
| Stop hit — per-position SL triggered | 2025-01-10 10:50:00 | 891.27 | 893.49 | 0.00 | SL hit |

### Cycle 23 — BUY (started 2025-04-16 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-16 09:50:00 | 647.25 | 639.80 | 0.00 | ORB-long ORB[632.05,640.70] vol=2.4x ATR=3.64 |
| Stop hit — per-position SL triggered | 2025-04-16 10:05:00 | 643.61 | 642.13 | 0.00 | SL hit |

### Cycle 24 — SELL (started 2025-04-29 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-29 09:45:00 | 664.25 | 675.57 | 0.00 | ORB-short ORB[679.00,687.75] vol=3.8x ATR=4.11 |
| Stop hit — per-position SL triggered | 2025-04-29 09:55:00 | 668.36 | 674.14 | 0.00 | SL hit |

### Cycle 25 — BUY (started 2025-05-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-05 11:15:00 | 667.50 | 653.74 | 0.00 | ORB-long ORB[638.05,648.00] vol=2.3x ATR=3.26 |
| Stop hit — per-position SL triggered | 2025-05-05 11:45:00 | 664.24 | 656.66 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2024-05-15 11:05:00 | 564.00 | 2024-05-15 12:00:00 | 565.84 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2024-05-17 10:05:00 | 575.26 | 2024-05-17 10:15:00 | 578.33 | STOP_HIT | 1.00 | -0.53% |
| SELL | retest1 | 2024-05-24 10:00:00 | 561.33 | 2024-05-24 10:30:00 | 564.27 | STOP_HIT | 1.00 | -0.52% |
| SELL | retest1 | 2024-05-28 10:10:00 | 558.25 | 2024-05-28 10:20:00 | 560.31 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest1 | 2024-05-30 11:15:00 | 570.99 | 2024-05-30 11:35:00 | 572.50 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2024-05-31 10:20:00 | 562.14 | 2024-05-31 13:05:00 | 558.03 | PARTIAL | 0.50 | 0.73% |
| SELL | retest1 | 2024-05-31 10:20:00 | 562.14 | 2024-05-31 15:20:00 | 555.00 | TARGET_HIT | 0.50 | 1.27% |
| BUY | retest1 | 2024-06-12 09:30:00 | 670.45 | 2024-06-12 09:40:00 | 675.19 | PARTIAL | 0.50 | 0.71% |
| BUY | retest1 | 2024-06-12 09:30:00 | 670.45 | 2024-06-12 10:30:00 | 670.45 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-06-25 09:30:00 | 645.80 | 2024-06-25 09:40:00 | 642.31 | PARTIAL | 0.50 | 0.54% |
| SELL | retest1 | 2024-06-25 09:30:00 | 645.80 | 2024-06-25 15:20:00 | 633.25 | TARGET_HIT | 0.50 | 1.94% |
| SELL | retest1 | 2024-07-05 09:45:00 | 554.10 | 2024-07-05 10:05:00 | 550.65 | PARTIAL | 0.50 | 0.62% |
| SELL | retest1 | 2024-07-05 09:45:00 | 554.10 | 2024-07-05 11:40:00 | 554.10 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-07-09 09:45:00 | 556.40 | 2024-07-09 09:50:00 | 554.21 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest1 | 2024-07-12 09:30:00 | 541.55 | 2024-07-12 09:45:00 | 540.06 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2024-07-30 09:40:00 | 617.00 | 2024-07-30 09:50:00 | 614.68 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2024-08-19 09:35:00 | 635.95 | 2024-08-19 10:00:00 | 633.56 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest1 | 2024-09-05 10:15:00 | 755.00 | 2024-09-05 11:00:00 | 759.08 | STOP_HIT | 1.00 | -0.54% |
| BUY | retest1 | 2024-09-11 09:35:00 | 748.00 | 2024-09-11 09:40:00 | 753.27 | PARTIAL | 0.50 | 0.71% |
| BUY | retest1 | 2024-09-11 09:35:00 | 748.00 | 2024-09-11 10:00:00 | 748.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-10-11 09:40:00 | 782.00 | 2024-10-11 09:45:00 | 787.60 | PARTIAL | 0.50 | 0.72% |
| BUY | retest1 | 2024-10-11 09:40:00 | 782.00 | 2024-10-11 10:00:00 | 782.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-11-29 09:45:00 | 912.15 | 2024-11-29 10:05:00 | 908.19 | STOP_HIT | 1.00 | -0.43% |
| SELL | retest1 | 2024-12-06 09:30:00 | 969.60 | 2024-12-06 10:00:00 | 974.47 | STOP_HIT | 1.00 | -0.50% |
| SELL | retest1 | 2024-12-26 09:30:00 | 915.75 | 2024-12-26 10:00:00 | 919.25 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2025-01-01 09:50:00 | 967.45 | 2025-01-01 10:00:00 | 975.74 | PARTIAL | 0.50 | 0.86% |
| BUY | retest1 | 2025-01-01 09:50:00 | 967.45 | 2025-01-01 10:05:00 | 967.45 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-01-02 09:30:00 | 982.60 | 2025-01-02 09:40:00 | 988.85 | PARTIAL | 0.50 | 0.64% |
| BUY | retest1 | 2025-01-02 09:30:00 | 982.60 | 2025-01-02 10:05:00 | 983.35 | TARGET_HIT | 0.50 | 0.08% |
| SELL | retest1 | 2025-01-10 09:55:00 | 885.20 | 2025-01-10 10:50:00 | 891.27 | STOP_HIT | 1.00 | -0.69% |
| BUY | retest1 | 2025-04-16 09:50:00 | 647.25 | 2025-04-16 10:05:00 | 643.61 | STOP_HIT | 1.00 | -0.56% |
| SELL | retest1 | 2025-04-29 09:45:00 | 664.25 | 2025-04-29 09:55:00 | 668.36 | STOP_HIT | 1.00 | -0.62% |
| BUY | retest1 | 2025-05-05 11:15:00 | 667.50 | 2025-05-05 11:45:00 | 664.24 | STOP_HIT | 1.00 | -0.49% |
