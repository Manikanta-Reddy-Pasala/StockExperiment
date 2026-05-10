# E.I.D. Parry (India) Ltd. (EIDPARRY)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 834.95
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
| ENTRY1 | 19 |
| ENTRY2 | 0 |
| PARTIAL | 7 |
| TARGET_HIT | 3 |
| STOP_HIT | 16 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 26 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 10 / 16
- **Target hits / Stop hits / Partials:** 3 / 16 / 7
- **Avg / median % per leg:** 0.21% / 0.00%
- **Sum % (uncompounded):** 5.40%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 14 | 5 | 35.7% | 1 | 9 | 4 | 0.27% | 3.8% |
| BUY @ 2nd Alert (retest1) | 14 | 5 | 35.7% | 1 | 9 | 4 | 0.27% | 3.8% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 12 | 5 | 41.7% | 2 | 7 | 3 | 0.13% | 1.6% |
| SELL @ 2nd Alert (retest1) | 12 | 5 | 41.7% | 2 | 7 | 3 | 0.13% | 1.6% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 26 | 10 | 38.5% | 3 | 16 | 7 | 0.21% | 5.4% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-09 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-09 10:35:00 | 902.15 | 894.18 | 0.00 | ORB-long ORB[882.40,892.45] vol=5.0x ATR=5.54 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-09 10:50:00 | 910.46 | 896.02 | 0.00 | T1 1.5R @ 910.46 |
| Target hit | 2026-02-09 15:20:00 | 930.40 | 912.51 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 2 — BUY (started 2026-02-13 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-13 10:05:00 | 916.85 | 910.81 | 0.00 | ORB-long ORB[900.40,912.90] vol=8.5x ATR=4.62 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-13 10:40:00 | 923.78 | 912.02 | 0.00 | T1 1.5R @ 923.78 |
| Stop hit — per-position SL triggered | 2026-02-13 13:20:00 | 916.85 | 918.05 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2026-02-17 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-17 09:30:00 | 910.20 | 908.40 | 0.00 | ORB-long ORB[893.55,903.35] vol=10.7x ATR=3.14 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-17 10:00:00 | 914.90 | 909.36 | 0.00 | T1 1.5R @ 914.90 |
| Stop hit — per-position SL triggered | 2026-02-17 10:55:00 | 910.20 | 910.75 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2026-02-19 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-19 09:30:00 | 901.10 | 905.31 | 0.00 | ORB-short ORB[903.00,912.60] vol=1.7x ATR=2.59 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-19 09:40:00 | 897.21 | 903.80 | 0.00 | T1 1.5R @ 897.21 |
| Stop hit — per-position SL triggered | 2026-02-19 09:45:00 | 901.10 | 903.47 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2026-02-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-24 11:15:00 | 878.95 | 880.45 | 0.00 | ORB-short ORB[881.15,888.60] vol=1.7x ATR=2.08 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-24 11:45:00 | 875.82 | 879.48 | 0.00 | T1 1.5R @ 875.82 |
| Target hit | 2026-02-24 15:20:00 | 868.50 | 870.31 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 6 — SELL (started 2026-03-06 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-06 10:45:00 | 826.05 | 836.44 | 0.00 | ORB-short ORB[834.95,844.60] vol=3.8x ATR=2.63 |
| Stop hit — per-position SL triggered | 2026-03-06 10:50:00 | 828.68 | 835.91 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2026-03-17 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-17 09:35:00 | 800.20 | 792.88 | 0.00 | ORB-long ORB[786.15,795.90] vol=1.6x ATR=4.18 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-17 10:20:00 | 806.47 | 796.76 | 0.00 | T1 1.5R @ 806.47 |
| Stop hit — per-position SL triggered | 2026-03-17 11:05:00 | 800.20 | 798.43 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2026-04-09 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-09 09:30:00 | 838.90 | 843.35 | 0.00 | ORB-short ORB[839.75,849.80] vol=2.2x ATR=3.39 |
| Stop hit — per-position SL triggered | 2026-04-09 09:50:00 | 842.29 | 841.66 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2026-04-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-10 11:15:00 | 857.05 | 852.14 | 0.00 | ORB-long ORB[835.30,843.50] vol=9.4x ATR=2.81 |
| Stop hit — per-position SL triggered | 2026-04-10 11:45:00 | 854.24 | 852.50 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2026-04-13 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-13 11:05:00 | 864.00 | 849.24 | 0.00 | ORB-long ORB[833.20,845.20] vol=3.7x ATR=2.97 |
| Stop hit — per-position SL triggered | 2026-04-13 11:10:00 | 861.03 | 850.30 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2026-04-16 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-16 09:35:00 | 858.10 | 864.26 | 0.00 | ORB-short ORB[862.10,874.00] vol=1.6x ATR=2.97 |
| Stop hit — per-position SL triggered | 2026-04-16 09:40:00 | 861.07 | 864.04 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2026-04-17 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-17 10:00:00 | 860.30 | 856.73 | 0.00 | ORB-long ORB[850.00,859.15] vol=2.1x ATR=3.38 |
| Stop hit — per-position SL triggered | 2026-04-17 10:50:00 | 856.92 | 857.12 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2026-04-22 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-22 10:25:00 | 864.45 | 858.24 | 0.00 | ORB-long ORB[850.20,863.00] vol=4.1x ATR=3.47 |
| Stop hit — per-position SL triggered | 2026-04-22 11:00:00 | 860.98 | 860.68 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2026-04-27 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-27 10:45:00 | 854.95 | 850.07 | 0.00 | ORB-long ORB[842.75,853.70] vol=1.8x ATR=2.69 |
| Stop hit — per-position SL triggered | 2026-04-27 10:55:00 | 852.26 | 850.20 | 0.00 | SL hit |

### Cycle 15 — BUY (started 2026-04-30 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-30 09:55:00 | 851.00 | 843.30 | 0.00 | ORB-long ORB[832.25,842.95] vol=5.4x ATR=4.56 |
| Stop hit — per-position SL triggered | 2026-04-30 10:10:00 | 846.44 | 846.64 | 0.00 | SL hit |

### Cycle 16 — SELL (started 2026-05-05 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-05 09:45:00 | 842.50 | 845.82 | 0.00 | ORB-short ORB[844.20,855.45] vol=2.6x ATR=2.84 |
| Stop hit — per-position SL triggered | 2026-05-05 10:25:00 | 845.34 | 844.74 | 0.00 | SL hit |

### Cycle 17 — SELL (started 2026-05-06 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-06 10:50:00 | 848.10 | 852.02 | 0.00 | ORB-short ORB[849.10,857.30] vol=1.8x ATR=1.90 |
| Stop hit — per-position SL triggered | 2026-05-06 11:05:00 | 850.00 | 851.72 | 0.00 | SL hit |

### Cycle 18 — SELL (started 2026-05-07 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-07 09:50:00 | 851.65 | 857.05 | 0.00 | ORB-short ORB[858.15,865.50] vol=2.0x ATR=3.11 |
| Stop hit — per-position SL triggered | 2026-05-07 11:30:00 | 854.76 | 855.20 | 0.00 | SL hit |

### Cycle 19 — SELL (started 2026-05-08 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-08 09:40:00 | 843.70 | 847.47 | 0.00 | ORB-short ORB[846.90,852.85] vol=1.6x ATR=2.20 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-08 09:50:00 | 840.39 | 845.33 | 0.00 | T1 1.5R @ 840.39 |
| Target hit | 2026-05-08 15:20:00 | 833.30 | 838.83 | 0.00 | EOD square-off @ 15:20:00 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-02-09 10:35:00 | 902.15 | 2026-02-09 10:50:00 | 910.46 | PARTIAL | 0.50 | 0.92% |
| BUY | retest1 | 2026-02-09 10:35:00 | 902.15 | 2026-02-09 15:20:00 | 930.40 | TARGET_HIT | 0.50 | 3.13% |
| BUY | retest1 | 2026-02-13 10:05:00 | 916.85 | 2026-02-13 10:40:00 | 923.78 | PARTIAL | 0.50 | 0.76% |
| BUY | retest1 | 2026-02-13 10:05:00 | 916.85 | 2026-02-13 13:20:00 | 916.85 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-17 09:30:00 | 910.20 | 2026-02-17 10:00:00 | 914.90 | PARTIAL | 0.50 | 0.52% |
| BUY | retest1 | 2026-02-17 09:30:00 | 910.20 | 2026-02-17 10:55:00 | 910.20 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-02-19 09:30:00 | 901.10 | 2026-02-19 09:40:00 | 897.21 | PARTIAL | 0.50 | 0.43% |
| SELL | retest1 | 2026-02-19 09:30:00 | 901.10 | 2026-02-19 09:45:00 | 901.10 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-02-24 11:15:00 | 878.95 | 2026-02-24 11:45:00 | 875.82 | PARTIAL | 0.50 | 0.36% |
| SELL | retest1 | 2026-02-24 11:15:00 | 878.95 | 2026-02-24 15:20:00 | 868.50 | TARGET_HIT | 0.50 | 1.19% |
| SELL | retest1 | 2026-03-06 10:45:00 | 826.05 | 2026-03-06 10:50:00 | 828.68 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2026-03-17 09:35:00 | 800.20 | 2026-03-17 10:20:00 | 806.47 | PARTIAL | 0.50 | 0.78% |
| BUY | retest1 | 2026-03-17 09:35:00 | 800.20 | 2026-03-17 11:05:00 | 800.20 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-04-09 09:30:00 | 838.90 | 2026-04-09 09:50:00 | 842.29 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest1 | 2026-04-10 11:15:00 | 857.05 | 2026-04-10 11:45:00 | 854.24 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2026-04-13 11:05:00 | 864.00 | 2026-04-13 11:10:00 | 861.03 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2026-04-16 09:35:00 | 858.10 | 2026-04-16 09:40:00 | 861.07 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2026-04-17 10:00:00 | 860.30 | 2026-04-17 10:50:00 | 856.92 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest1 | 2026-04-22 10:25:00 | 864.45 | 2026-04-22 11:00:00 | 860.98 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest1 | 2026-04-27 10:45:00 | 854.95 | 2026-04-27 10:55:00 | 852.26 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2026-04-30 09:55:00 | 851.00 | 2026-04-30 10:10:00 | 846.44 | STOP_HIT | 1.00 | -0.54% |
| SELL | retest1 | 2026-05-05 09:45:00 | 842.50 | 2026-05-05 10:25:00 | 845.34 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2026-05-06 10:50:00 | 848.10 | 2026-05-06 11:05:00 | 850.00 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2026-05-07 09:50:00 | 851.65 | 2026-05-07 11:30:00 | 854.76 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2026-05-08 09:40:00 | 843.70 | 2026-05-08 09:50:00 | 840.39 | PARTIAL | 0.50 | 0.39% |
| SELL | retest1 | 2026-05-08 09:40:00 | 843.70 | 2026-05-08 15:20:00 | 833.30 | TARGET_HIT | 0.50 | 1.23% |
