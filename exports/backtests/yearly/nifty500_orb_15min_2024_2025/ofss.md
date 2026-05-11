# Oracle Financial Services Software Ltd. (OFSS)

## Backtest Summary

- **Window:** 2024-05-13 09:15:00 → 2026-05-08 15:25:00 (36871 bars)
- **Last close:** 9321.00
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
| ENTRY1 | 73 |
| ENTRY2 | 0 |
| PARTIAL | 26 |
| TARGET_HIT | 13 |
| STOP_HIT | 60 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 99 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 39 / 60
- **Target hits / Stop hits / Partials:** 13 / 60 / 26
- **Avg / median % per leg:** 0.26% / 0.00%
- **Sum % (uncompounded):** 25.25%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 53 | 17 | 32.1% | 5 | 36 | 12 | 0.23% | 12.3% |
| BUY @ 2nd Alert (retest1) | 53 | 17 | 32.1% | 5 | 36 | 12 | 0.23% | 12.3% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 46 | 22 | 47.8% | 8 | 24 | 14 | 0.28% | 12.9% |
| SELL @ 2nd Alert (retest1) | 46 | 22 | 47.8% | 8 | 24 | 14 | 0.28% | 12.9% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 99 | 39 | 39.4% | 13 | 60 | 26 | 0.26% | 25.3% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-05-14 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-14 11:00:00 | 7717.75 | 7736.48 | 0.00 | ORB-short ORB[7742.15,7804.95] vol=1.6x ATR=21.25 |
| Stop hit — per-position SL triggered | 2024-05-14 11:05:00 | 7739.00 | 7737.27 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2024-05-15 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-15 09:30:00 | 7941.00 | 7888.79 | 0.00 | ORB-long ORB[7829.30,7916.00] vol=1.6x ATR=27.16 |
| Stop hit — per-position SL triggered | 2024-05-15 09:40:00 | 7913.84 | 7908.06 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2024-05-17 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-17 10:50:00 | 7800.00 | 7840.95 | 0.00 | ORB-short ORB[7855.55,7913.70] vol=2.1x ATR=18.65 |
| Stop hit — per-position SL triggered | 2024-05-17 11:00:00 | 7818.65 | 7839.84 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2024-05-21 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-21 09:35:00 | 7745.20 | 7780.70 | 0.00 | ORB-short ORB[7757.55,7827.00] vol=2.4x ATR=22.15 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-21 09:55:00 | 7711.98 | 7761.06 | 0.00 | T1 1.5R @ 7711.98 |
| Target hit | 2024-05-21 11:05:00 | 7741.80 | 7733.44 | 0.00 | Trail-exit close>VWAP |

### Cycle 5 — SELL (started 2024-05-22 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-22 09:40:00 | 7626.95 | 7676.08 | 0.00 | ORB-short ORB[7669.35,7707.40] vol=1.7x ATR=18.47 |
| Stop hit — per-position SL triggered | 2024-05-22 09:50:00 | 7645.42 | 7664.25 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2024-05-23 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-23 09:30:00 | 7697.00 | 7659.00 | 0.00 | ORB-long ORB[7629.00,7688.15] vol=2.1x ATR=24.54 |
| Stop hit — per-position SL triggered | 2024-05-23 10:05:00 | 7672.46 | 7667.72 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2024-05-24 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-24 10:25:00 | 7572.05 | 7613.86 | 0.00 | ORB-short ORB[7620.00,7675.95] vol=1.9x ATR=19.95 |
| Stop hit — per-position SL triggered | 2024-05-24 11:10:00 | 7592.00 | 7598.03 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2024-05-27 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-27 09:55:00 | 7480.00 | 7524.91 | 0.00 | ORB-short ORB[7520.00,7600.00] vol=2.3x ATR=20.50 |
| Stop hit — per-position SL triggered | 2024-05-27 10:05:00 | 7500.50 | 7520.79 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2024-05-28 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-28 10:05:00 | 7548.80 | 7582.19 | 0.00 | ORB-short ORB[7574.85,7629.90] vol=1.8x ATR=24.00 |
| Stop hit — per-position SL triggered | 2024-05-28 10:15:00 | 7572.80 | 7580.93 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2024-05-29 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-29 10:45:00 | 7579.40 | 7547.38 | 0.00 | ORB-long ORB[7500.25,7560.25] vol=5.4x ATR=17.68 |
| Stop hit — per-position SL triggered | 2024-05-29 10:55:00 | 7561.72 | 7549.04 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2024-06-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-06 11:15:00 | 7920.20 | 7826.81 | 0.00 | ORB-long ORB[7742.45,7858.35] vol=5.8x ATR=28.13 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-06 11:20:00 | 7962.40 | 7851.23 | 0.00 | T1 1.5R @ 7962.40 |
| Stop hit — per-position SL triggered | 2024-06-06 11:35:00 | 7920.20 | 7859.69 | 0.00 | SL hit |

### Cycle 12 — SELL (started 2024-06-10 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-10 09:35:00 | 8280.00 | 8325.24 | 0.00 | ORB-short ORB[8310.10,8420.00] vol=4.0x ATR=33.52 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-10 09:55:00 | 8229.71 | 8308.87 | 0.00 | T1 1.5R @ 8229.71 |
| Target hit | 2024-06-10 11:00:00 | 8271.00 | 8264.38 | 0.00 | Trail-exit close>VWAP |

### Cycle 13 — BUY (started 2024-06-11 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-11 09:40:00 | 8538.55 | 8466.15 | 0.00 | ORB-long ORB[8367.55,8470.00] vol=4.2x ATR=35.78 |
| Stop hit — per-position SL triggered | 2024-06-11 09:45:00 | 8502.77 | 8476.56 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2024-06-13 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-13 09:30:00 | 8998.25 | 8908.79 | 0.00 | ORB-long ORB[8802.00,8925.90] vol=4.4x ATR=37.72 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-13 09:35:00 | 9054.84 | 8995.59 | 0.00 | T1 1.5R @ 9054.84 |
| Target hit | 2024-06-13 15:20:00 | 9690.00 | 9409.88 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 15 — BUY (started 2024-06-24 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-24 11:05:00 | 9787.40 | 9699.55 | 0.00 | ORB-long ORB[9608.00,9749.20] vol=1.8x ATR=35.06 |
| Stop hit — per-position SL triggered | 2024-06-24 11:40:00 | 9752.34 | 9709.94 | 0.00 | SL hit |

### Cycle 16 — SELL (started 2024-06-26 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-26 09:30:00 | 9660.05 | 9707.11 | 0.00 | ORB-short ORB[9675.00,9770.40] vol=1.7x ATR=21.88 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-26 09:40:00 | 9627.23 | 9690.56 | 0.00 | T1 1.5R @ 9627.23 |
| Target hit | 2024-06-26 15:20:00 | 9488.85 | 9556.28 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 17 — BUY (started 2024-06-27 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-27 09:40:00 | 9541.20 | 9494.85 | 0.00 | ORB-long ORB[9449.00,9511.20] vol=1.7x ATR=25.79 |
| Stop hit — per-position SL triggered | 2024-06-27 09:45:00 | 9515.41 | 9498.52 | 0.00 | SL hit |

### Cycle 18 — SELL (started 2024-06-28 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-28 11:05:00 | 9761.90 | 9763.91 | 0.00 | ORB-short ORB[9801.00,9925.00] vol=2.3x ATR=28.71 |
| Stop hit — per-position SL triggered | 2024-06-28 11:10:00 | 9790.61 | 9765.12 | 0.00 | SL hit |

### Cycle 19 — BUY (started 2024-07-03 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-03 09:40:00 | 10376.00 | 10328.26 | 0.00 | ORB-long ORB[10250.10,10353.20] vol=2.1x ATR=37.68 |
| Stop hit — per-position SL triggered | 2024-07-03 10:00:00 | 10338.32 | 10336.06 | 0.00 | SL hit |

### Cycle 20 — SELL (started 2024-07-05 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-05 09:50:00 | 10392.50 | 10429.67 | 0.00 | ORB-short ORB[10430.15,10488.00] vol=3.8x ATR=30.36 |
| Stop hit — per-position SL triggered | 2024-07-05 10:05:00 | 10422.86 | 10422.90 | 0.00 | SL hit |

### Cycle 21 — SELL (started 2024-07-09 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-09 09:30:00 | 10273.05 | 10320.95 | 0.00 | ORB-short ORB[10300.00,10398.00] vol=1.5x ATR=32.97 |
| Stop hit — per-position SL triggered | 2024-07-09 09:35:00 | 10306.02 | 10316.92 | 0.00 | SL hit |

### Cycle 22 — SELL (started 2024-07-10 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-10 10:20:00 | 10123.50 | 10259.19 | 0.00 | ORB-short ORB[10278.45,10373.55] vol=1.5x ATR=37.30 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-10 10:35:00 | 10067.55 | 10212.98 | 0.00 | T1 1.5R @ 10067.55 |
| Stop hit — per-position SL triggered | 2024-07-10 10:40:00 | 10123.50 | 10207.39 | 0.00 | SL hit |

### Cycle 23 — BUY (started 2024-07-24 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-24 09:30:00 | 11235.20 | 11176.85 | 0.00 | ORB-long ORB[11022.00,11182.25] vol=5.8x ATR=61.01 |
| Stop hit — per-position SL triggered | 2024-07-24 09:40:00 | 11174.19 | 11182.11 | 0.00 | SL hit |

### Cycle 24 — BUY (started 2024-07-26 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-26 09:40:00 | 11285.80 | 11186.81 | 0.00 | ORB-long ORB[11090.00,11209.00] vol=2.1x ATR=50.91 |
| Stop hit — per-position SL triggered | 2024-07-26 09:50:00 | 11234.89 | 11209.01 | 0.00 | SL hit |

### Cycle 25 — BUY (started 2024-07-29 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-29 10:00:00 | 11323.00 | 11205.64 | 0.00 | ORB-long ORB[11115.05,11247.00] vol=1.7x ATR=42.79 |
| Stop hit — per-position SL triggered | 2024-07-29 10:05:00 | 11280.21 | 11218.62 | 0.00 | SL hit |

### Cycle 26 — BUY (started 2024-07-31 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-31 10:10:00 | 11185.00 | 11131.88 | 0.00 | ORB-long ORB[11077.20,11168.65] vol=1.7x ATR=30.15 |
| Stop hit — per-position SL triggered | 2024-07-31 10:20:00 | 11154.85 | 11137.21 | 0.00 | SL hit |

### Cycle 27 — BUY (started 2024-08-12 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-12 11:10:00 | 10675.00 | 10577.64 | 0.00 | ORB-long ORB[10455.05,10605.20] vol=2.1x ATR=36.77 |
| Stop hit — per-position SL triggered | 2024-08-12 11:25:00 | 10638.23 | 10583.14 | 0.00 | SL hit |

### Cycle 28 — BUY (started 2024-08-19 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-19 10:55:00 | 11098.10 | 11052.93 | 0.00 | ORB-long ORB[10900.00,11049.00] vol=2.0x ATR=32.69 |
| Stop hit — per-position SL triggered | 2024-08-19 11:00:00 | 11065.41 | 11053.65 | 0.00 | SL hit |

### Cycle 29 — BUY (started 2024-08-20 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-20 09:45:00 | 11059.30 | 11019.17 | 0.00 | ORB-long ORB[10950.80,11039.00] vol=1.8x ATR=29.22 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-20 10:50:00 | 11103.13 | 11048.25 | 0.00 | T1 1.5R @ 11103.13 |
| Stop hit — per-position SL triggered | 2024-08-20 11:10:00 | 11059.30 | 11051.16 | 0.00 | SL hit |

### Cycle 30 — BUY (started 2024-08-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-21 11:15:00 | 11149.95 | 11075.14 | 0.00 | ORB-long ORB[11020.05,11098.00] vol=1.7x ATR=23.77 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-21 14:25:00 | 11185.61 | 11117.54 | 0.00 | T1 1.5R @ 11185.61 |
| Stop hit — per-position SL triggered | 2024-08-21 15:15:00 | 11149.95 | 11128.55 | 0.00 | SL hit |

### Cycle 31 — SELL (started 2024-08-22 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-22 10:50:00 | 11150.50 | 11244.65 | 0.00 | ORB-short ORB[11176.15,11263.00] vol=1.9x ATR=28.18 |
| Stop hit — per-position SL triggered | 2024-08-22 11:10:00 | 11178.68 | 11236.01 | 0.00 | SL hit |

### Cycle 32 — SELL (started 2024-08-23 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-23 09:30:00 | 10991.20 | 11053.49 | 0.00 | ORB-short ORB[11027.80,11161.60] vol=1.6x ATR=30.65 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-23 09:40:00 | 10945.23 | 11024.84 | 0.00 | T1 1.5R @ 10945.23 |
| Stop hit — per-position SL triggered | 2024-08-23 09:55:00 | 10991.20 | 11003.19 | 0.00 | SL hit |

### Cycle 33 — SELL (started 2024-08-27 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-27 10:55:00 | 10937.00 | 10954.15 | 0.00 | ORB-short ORB[10960.10,11048.95] vol=7.5x ATR=27.35 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-27 11:10:00 | 10895.97 | 10946.26 | 0.00 | T1 1.5R @ 10895.97 |
| Stop hit — per-position SL triggered | 2024-08-27 11:45:00 | 10937.00 | 10936.62 | 0.00 | SL hit |

### Cycle 34 — SELL (started 2024-08-28 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-28 09:30:00 | 10907.20 | 10980.04 | 0.00 | ORB-short ORB[10915.05,11056.75] vol=1.6x ATR=36.35 |
| Stop hit — per-position SL triggered | 2024-08-28 09:35:00 | 10943.55 | 10970.78 | 0.00 | SL hit |

### Cycle 35 — SELL (started 2024-08-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-29 10:15:00 | 10863.90 | 10934.49 | 0.00 | ORB-short ORB[10929.10,11009.00] vol=3.8x ATR=31.87 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-29 10:35:00 | 10816.09 | 10906.22 | 0.00 | T1 1.5R @ 10816.09 |
| Stop hit — per-position SL triggered | 2024-08-29 11:15:00 | 10863.90 | 10887.23 | 0.00 | SL hit |

### Cycle 36 — BUY (started 2024-09-03 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-03 09:45:00 | 11090.05 | 11012.78 | 0.00 | ORB-long ORB[10928.20,11005.00] vol=2.1x ATR=32.19 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-03 11:45:00 | 11138.34 | 11069.97 | 0.00 | T1 1.5R @ 11138.34 |
| Target hit | 2024-09-03 15:20:00 | 11456.50 | 11355.20 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 37 — SELL (started 2024-09-06 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-06 09:35:00 | 11238.25 | 11277.83 | 0.00 | ORB-short ORB[11250.05,11333.00] vol=2.2x ATR=34.34 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-06 09:45:00 | 11186.75 | 11259.46 | 0.00 | T1 1.5R @ 11186.75 |
| Target hit | 2024-09-06 15:20:00 | 10846.10 | 11031.76 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 38 — BUY (started 2024-09-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-13 10:15:00 | 11683.00 | 11638.94 | 0.00 | ORB-long ORB[11595.20,11677.00] vol=1.6x ATR=34.30 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-13 10:40:00 | 11734.45 | 11662.59 | 0.00 | T1 1.5R @ 11734.45 |
| Target hit | 2024-09-13 15:20:00 | 12238.60 | 12050.21 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 39 — BUY (started 2024-09-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-16 11:15:00 | 12310.00 | 12160.24 | 0.00 | ORB-long ORB[12103.55,12286.50] vol=2.6x ATR=49.54 |
| Stop hit — per-position SL triggered | 2024-09-16 11:25:00 | 12260.46 | 12179.20 | 0.00 | SL hit |

### Cycle 40 — BUY (started 2024-09-17 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-17 09:30:00 | 12426.10 | 12327.13 | 0.00 | ORB-long ORB[12219.25,12339.00] vol=4.3x ATR=42.16 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-17 09:35:00 | 12489.34 | 12404.15 | 0.00 | T1 1.5R @ 12489.34 |
| Target hit | 2024-09-17 10:30:00 | 12465.60 | 12466.55 | 0.00 | Trail-exit close<VWAP |

### Cycle 41 — SELL (started 2024-09-25 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-25 09:30:00 | 11438.90 | 11557.47 | 0.00 | ORB-short ORB[11505.00,11663.30] vol=1.7x ATR=42.26 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-25 09:45:00 | 11375.51 | 11519.34 | 0.00 | T1 1.5R @ 11375.51 |
| Target hit | 2024-09-25 15:20:00 | 11172.00 | 11298.75 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 42 — BUY (started 2024-09-26 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-26 09:35:00 | 11337.00 | 11299.56 | 0.00 | ORB-long ORB[11230.95,11319.00] vol=2.0x ATR=35.83 |
| Stop hit — per-position SL triggered | 2024-09-26 09:40:00 | 11301.17 | 11302.14 | 0.00 | SL hit |

### Cycle 43 — BUY (started 2024-10-03 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-03 09:35:00 | 11432.45 | 11349.10 | 0.00 | ORB-long ORB[11247.50,11407.90] vol=1.6x ATR=37.80 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-03 09:50:00 | 11489.15 | 11384.15 | 0.00 | T1 1.5R @ 11489.15 |
| Stop hit — per-position SL triggered | 2024-10-03 10:00:00 | 11432.45 | 11392.34 | 0.00 | SL hit |

### Cycle 44 — BUY (started 2024-10-04 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-04 10:40:00 | 11406.50 | 11264.33 | 0.00 | ORB-long ORB[11143.65,11281.50] vol=2.3x ATR=48.17 |
| Stop hit — per-position SL triggered | 2024-10-04 11:20:00 | 11358.33 | 11297.83 | 0.00 | SL hit |

### Cycle 45 — BUY (started 2024-10-09 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-09 09:50:00 | 11541.00 | 11436.43 | 0.00 | ORB-long ORB[11261.95,11425.00] vol=2.9x ATR=44.37 |
| Stop hit — per-position SL triggered | 2024-10-09 09:55:00 | 11496.63 | 11447.82 | 0.00 | SL hit |

### Cycle 46 — BUY (started 2024-10-10 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-10 10:25:00 | 11799.80 | 11770.44 | 0.00 | ORB-long ORB[11685.50,11799.00] vol=1.9x ATR=41.29 |
| Stop hit — per-position SL triggered | 2024-10-10 10:35:00 | 11758.51 | 11773.19 | 0.00 | SL hit |

### Cycle 47 — BUY (started 2024-10-23 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-23 10:10:00 | 11170.45 | 11019.59 | 0.00 | ORB-long ORB[10872.10,10991.20] vol=1.8x ATR=60.92 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-23 10:50:00 | 11261.83 | 11129.44 | 0.00 | T1 1.5R @ 11261.83 |
| Target hit | 2024-10-23 14:35:00 | 11223.20 | 11234.11 | 0.00 | Trail-exit close<VWAP |

### Cycle 48 — SELL (started 2024-10-24 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-24 09:35:00 | 11089.90 | 11175.91 | 0.00 | ORB-short ORB[11114.05,11278.00] vol=2.2x ATR=65.33 |
| Stop hit — per-position SL triggered | 2024-10-24 09:40:00 | 11155.23 | 11165.07 | 0.00 | SL hit |

### Cycle 49 — BUY (started 2024-11-22 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-22 09:30:00 | 11455.30 | 11357.53 | 0.00 | ORB-long ORB[11249.95,11410.00] vol=1.5x ATR=43.51 |
| Stop hit — per-position SL triggered | 2024-11-22 09:35:00 | 11411.79 | 11378.45 | 0.00 | SL hit |

### Cycle 50 — SELL (started 2024-11-28 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-28 11:10:00 | 11600.00 | 11729.48 | 0.00 | ORB-short ORB[11711.05,11860.00] vol=1.5x ATR=44.08 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-28 11:40:00 | 11533.88 | 11708.21 | 0.00 | T1 1.5R @ 11533.88 |
| Stop hit — per-position SL triggered | 2024-11-28 12:05:00 | 11600.00 | 11686.79 | 0.00 | SL hit |

### Cycle 51 — BUY (started 2024-11-29 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-29 10:55:00 | 11792.40 | 11748.93 | 0.00 | ORB-long ORB[11615.05,11740.00] vol=1.5x ATR=44.95 |
| Stop hit — per-position SL triggered | 2024-11-29 11:25:00 | 11747.45 | 11750.66 | 0.00 | SL hit |

### Cycle 52 — BUY (started 2024-12-03 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-03 11:10:00 | 12476.80 | 12407.37 | 0.00 | ORB-long ORB[12276.05,12364.00] vol=1.7x ATR=37.78 |
| Stop hit — per-position SL triggered | 2024-12-03 11:25:00 | 12439.02 | 12411.09 | 0.00 | SL hit |

### Cycle 53 — BUY (started 2024-12-06 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-06 10:30:00 | 12819.75 | 12672.77 | 0.00 | ORB-long ORB[12440.90,12574.50] vol=2.9x ATR=54.59 |
| Stop hit — per-position SL triggered | 2024-12-06 10:35:00 | 12765.16 | 12684.96 | 0.00 | SL hit |

### Cycle 54 — BUY (started 2024-12-27 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-27 10:50:00 | 12410.35 | 12360.54 | 0.00 | ORB-long ORB[12310.25,12385.00] vol=1.7x ATR=40.43 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-27 11:10:00 | 12471.00 | 12367.39 | 0.00 | T1 1.5R @ 12471.00 |
| Stop hit — per-position SL triggered | 2024-12-27 11:20:00 | 12410.35 | 12373.60 | 0.00 | SL hit |

### Cycle 55 — SELL (started 2025-01-01 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-01 10:55:00 | 12664.65 | 12721.82 | 0.00 | ORB-short ORB[12730.10,12879.95] vol=1.5x ATR=45.23 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-01 11:20:00 | 12596.81 | 12710.82 | 0.00 | T1 1.5R @ 12596.81 |
| Stop hit — per-position SL triggered | 2025-01-01 15:10:00 | 12664.65 | 12632.82 | 0.00 | SL hit |

### Cycle 56 — BUY (started 2025-01-02 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-02 09:30:00 | 12720.20 | 12669.71 | 0.00 | ORB-long ORB[12610.10,12690.00] vol=1.5x ATR=35.57 |
| Stop hit — per-position SL triggered | 2025-01-02 10:00:00 | 12684.63 | 12704.63 | 0.00 | SL hit |

### Cycle 57 — SELL (started 2025-01-03 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-03 10:55:00 | 12440.00 | 12623.84 | 0.00 | ORB-short ORB[12650.05,12798.85] vol=4.0x ATR=37.22 |
| Stop hit — per-position SL triggered | 2025-01-03 11:20:00 | 12477.22 | 12606.76 | 0.00 | SL hit |

### Cycle 58 — SELL (started 2025-01-06 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-06 10:50:00 | 12207.70 | 12421.33 | 0.00 | ORB-short ORB[12402.15,12550.00] vol=2.0x ATR=49.51 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-06 11:10:00 | 12133.44 | 12383.77 | 0.00 | T1 1.5R @ 12133.44 |
| Target hit | 2025-01-06 15:20:00 | 11970.00 | 12160.07 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 59 — BUY (started 2025-02-07 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-07 11:00:00 | 9374.80 | 9286.96 | 0.00 | ORB-long ORB[9222.60,9347.05] vol=1.8x ATR=37.58 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-07 11:10:00 | 9431.16 | 9322.94 | 0.00 | T1 1.5R @ 9431.16 |
| Stop hit — per-position SL triggered | 2025-02-07 11:55:00 | 9374.80 | 9358.04 | 0.00 | SL hit |

### Cycle 60 — SELL (started 2025-02-11 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-11 10:10:00 | 9197.25 | 9260.86 | 0.00 | ORB-short ORB[9242.80,9339.25] vol=1.6x ATR=30.83 |
| Stop hit — per-position SL triggered | 2025-02-11 10:20:00 | 9228.08 | 9251.63 | 0.00 | SL hit |

### Cycle 61 — SELL (started 2025-02-18 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-18 10:25:00 | 8691.55 | 8779.81 | 0.00 | ORB-short ORB[8764.10,8888.00] vol=1.8x ATR=38.59 |
| Stop hit — per-position SL triggered | 2025-02-18 10:45:00 | 8730.14 | 8763.03 | 0.00 | SL hit |

### Cycle 62 — BUY (started 2025-02-19 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-19 11:10:00 | 8844.45 | 8815.55 | 0.00 | ORB-long ORB[8700.00,8768.50] vol=1.7x ATR=32.81 |
| Stop hit — per-position SL triggered | 2025-02-19 12:50:00 | 8811.64 | 8823.22 | 0.00 | SL hit |

### Cycle 63 — SELL (started 2025-02-21 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-21 09:35:00 | 8589.00 | 8632.37 | 0.00 | ORB-short ORB[8612.25,8724.40] vol=2.3x ATR=28.11 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-21 09:40:00 | 8546.84 | 8619.68 | 0.00 | T1 1.5R @ 8546.84 |
| Target hit | 2025-02-21 14:10:00 | 8427.05 | 8425.80 | 0.00 | Trail-exit close>VWAP |

### Cycle 64 — BUY (started 2025-03-17 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-17 10:40:00 | 7446.00 | 7416.42 | 0.00 | ORB-long ORB[7340.50,7440.00] vol=1.7x ATR=26.66 |
| Stop hit — per-position SL triggered | 2025-03-17 11:00:00 | 7419.34 | 7419.73 | 0.00 | SL hit |

### Cycle 65 — BUY (started 2025-03-18 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-18 10:00:00 | 7595.00 | 7540.83 | 0.00 | ORB-long ORB[7470.25,7558.00] vol=1.7x ATR=23.79 |
| Stop hit — per-position SL triggered | 2025-03-18 10:35:00 | 7571.21 | 7551.39 | 0.00 | SL hit |

### Cycle 66 — BUY (started 2025-03-25 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-25 09:35:00 | 8104.00 | 8024.78 | 0.00 | ORB-long ORB[7970.00,8050.00] vol=2.3x ATR=35.71 |
| Stop hit — per-position SL triggered | 2025-03-25 09:55:00 | 8068.29 | 8060.54 | 0.00 | SL hit |

### Cycle 67 — SELL (started 2025-04-03 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-03 09:45:00 | 7544.80 | 7620.06 | 0.00 | ORB-short ORB[7600.00,7679.95] vol=2.2x ATR=35.85 |
| Stop hit — per-position SL triggered | 2025-04-03 10:00:00 | 7580.65 | 7606.70 | 0.00 | SL hit |

### Cycle 68 — BUY (started 2025-04-04 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-04 09:35:00 | 7633.45 | 7578.20 | 0.00 | ORB-long ORB[7508.20,7609.90] vol=1.8x ATR=32.41 |
| Stop hit — per-position SL triggered | 2025-04-04 09:40:00 | 7601.04 | 7585.25 | 0.00 | SL hit |

### Cycle 69 — SELL (started 2025-04-15 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-15 09:30:00 | 7632.50 | 7667.45 | 0.00 | ORB-short ORB[7635.00,7742.00] vol=1.5x ATR=29.28 |
| Stop hit — per-position SL triggered | 2025-04-15 09:35:00 | 7661.78 | 7665.28 | 0.00 | SL hit |

### Cycle 70 — BUY (started 2025-04-17 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-17 09:45:00 | 7955.00 | 7890.34 | 0.00 | ORB-long ORB[7817.00,7919.50] vol=1.7x ATR=32.29 |
| Stop hit — per-position SL triggered | 2025-04-17 10:00:00 | 7922.71 | 7897.17 | 0.00 | SL hit |

### Cycle 71 — BUY (started 2025-04-23 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-23 09:40:00 | 8413.50 | 8342.33 | 0.00 | ORB-long ORB[8266.50,8370.00] vol=2.9x ATR=36.02 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-23 09:50:00 | 8467.53 | 8376.74 | 0.00 | T1 1.5R @ 8467.53 |
| Stop hit — per-position SL triggered | 2025-04-23 10:05:00 | 8413.50 | 8389.27 | 0.00 | SL hit |

### Cycle 72 — SELL (started 2025-04-29 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-29 09:40:00 | 8790.00 | 8843.28 | 0.00 | ORB-short ORB[8809.50,8899.50] vol=1.9x ATR=36.50 |
| Stop hit — per-position SL triggered | 2025-04-29 09:45:00 | 8826.50 | 8841.87 | 0.00 | SL hit |

### Cycle 73 — SELL (started 2025-05-05 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-05 10:55:00 | 8707.00 | 8742.97 | 0.00 | ORB-short ORB[8712.50,8800.00] vol=1.7x ATR=22.18 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-05 11:35:00 | 8673.72 | 8731.24 | 0.00 | T1 1.5R @ 8673.72 |
| Target hit | 2025-05-05 15:20:00 | 8650.00 | 8663.36 | 0.00 | EOD square-off @ 15:20:00 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2024-05-14 11:00:00 | 7717.75 | 2024-05-14 11:05:00 | 7739.00 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2024-05-15 09:30:00 | 7941.00 | 2024-05-15 09:40:00 | 7913.84 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2024-05-17 10:50:00 | 7800.00 | 2024-05-17 11:00:00 | 7818.65 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2024-05-21 09:35:00 | 7745.20 | 2024-05-21 09:55:00 | 7711.98 | PARTIAL | 0.50 | 0.43% |
| SELL | retest1 | 2024-05-21 09:35:00 | 7745.20 | 2024-05-21 11:05:00 | 7741.80 | TARGET_HIT | 0.50 | 0.04% |
| SELL | retest1 | 2024-05-22 09:40:00 | 7626.95 | 2024-05-22 09:50:00 | 7645.42 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2024-05-23 09:30:00 | 7697.00 | 2024-05-23 10:05:00 | 7672.46 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2024-05-24 10:25:00 | 7572.05 | 2024-05-24 11:10:00 | 7592.00 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2024-05-27 09:55:00 | 7480.00 | 2024-05-27 10:05:00 | 7500.50 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2024-05-28 10:05:00 | 7548.80 | 2024-05-28 10:15:00 | 7572.80 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2024-05-29 10:45:00 | 7579.40 | 2024-05-29 10:55:00 | 7561.72 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2024-06-06 11:15:00 | 7920.20 | 2024-06-06 11:20:00 | 7962.40 | PARTIAL | 0.50 | 0.53% |
| BUY | retest1 | 2024-06-06 11:15:00 | 7920.20 | 2024-06-06 11:35:00 | 7920.20 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-06-10 09:35:00 | 8280.00 | 2024-06-10 09:55:00 | 8229.71 | PARTIAL | 0.50 | 0.61% |
| SELL | retest1 | 2024-06-10 09:35:00 | 8280.00 | 2024-06-10 11:00:00 | 8271.00 | TARGET_HIT | 0.50 | 0.11% |
| BUY | retest1 | 2024-06-11 09:40:00 | 8538.55 | 2024-06-11 09:45:00 | 8502.77 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest1 | 2024-06-13 09:30:00 | 8998.25 | 2024-06-13 09:35:00 | 9054.84 | PARTIAL | 0.50 | 0.63% |
| BUY | retest1 | 2024-06-13 09:30:00 | 8998.25 | 2024-06-13 15:20:00 | 9690.00 | TARGET_HIT | 0.50 | 7.69% |
| BUY | retest1 | 2024-06-24 11:05:00 | 9787.40 | 2024-06-24 11:40:00 | 9752.34 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2024-06-26 09:30:00 | 9660.05 | 2024-06-26 09:40:00 | 9627.23 | PARTIAL | 0.50 | 0.34% |
| SELL | retest1 | 2024-06-26 09:30:00 | 9660.05 | 2024-06-26 15:20:00 | 9488.85 | TARGET_HIT | 0.50 | 1.77% |
| BUY | retest1 | 2024-06-27 09:40:00 | 9541.20 | 2024-06-27 09:45:00 | 9515.41 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2024-06-28 11:05:00 | 9761.90 | 2024-06-28 11:10:00 | 9790.61 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2024-07-03 09:40:00 | 10376.00 | 2024-07-03 10:00:00 | 10338.32 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2024-07-05 09:50:00 | 10392.50 | 2024-07-05 10:05:00 | 10422.86 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2024-07-09 09:30:00 | 10273.05 | 2024-07-09 09:35:00 | 10306.02 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2024-07-10 10:20:00 | 10123.50 | 2024-07-10 10:35:00 | 10067.55 | PARTIAL | 0.50 | 0.55% |
| SELL | retest1 | 2024-07-10 10:20:00 | 10123.50 | 2024-07-10 10:40:00 | 10123.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-07-24 09:30:00 | 11235.20 | 2024-07-24 09:40:00 | 11174.19 | STOP_HIT | 1.00 | -0.54% |
| BUY | retest1 | 2024-07-26 09:40:00 | 11285.80 | 2024-07-26 09:50:00 | 11234.89 | STOP_HIT | 1.00 | -0.45% |
| BUY | retest1 | 2024-07-29 10:00:00 | 11323.00 | 2024-07-29 10:05:00 | 11280.21 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2024-07-31 10:10:00 | 11185.00 | 2024-07-31 10:20:00 | 11154.85 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2024-08-12 11:10:00 | 10675.00 | 2024-08-12 11:25:00 | 10638.23 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2024-08-19 10:55:00 | 11098.10 | 2024-08-19 11:00:00 | 11065.41 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2024-08-20 09:45:00 | 11059.30 | 2024-08-20 10:50:00 | 11103.13 | PARTIAL | 0.50 | 0.40% |
| BUY | retest1 | 2024-08-20 09:45:00 | 11059.30 | 2024-08-20 11:10:00 | 11059.30 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-08-21 11:15:00 | 11149.95 | 2024-08-21 14:25:00 | 11185.61 | PARTIAL | 0.50 | 0.32% |
| BUY | retest1 | 2024-08-21 11:15:00 | 11149.95 | 2024-08-21 15:15:00 | 11149.95 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-08-22 10:50:00 | 11150.50 | 2024-08-22 11:10:00 | 11178.68 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2024-08-23 09:30:00 | 10991.20 | 2024-08-23 09:40:00 | 10945.23 | PARTIAL | 0.50 | 0.42% |
| SELL | retest1 | 2024-08-23 09:30:00 | 10991.20 | 2024-08-23 09:55:00 | 10991.20 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-08-27 10:55:00 | 10937.00 | 2024-08-27 11:10:00 | 10895.97 | PARTIAL | 0.50 | 0.38% |
| SELL | retest1 | 2024-08-27 10:55:00 | 10937.00 | 2024-08-27 11:45:00 | 10937.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-08-28 09:30:00 | 10907.20 | 2024-08-28 09:35:00 | 10943.55 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2024-08-29 10:15:00 | 10863.90 | 2024-08-29 10:35:00 | 10816.09 | PARTIAL | 0.50 | 0.44% |
| SELL | retest1 | 2024-08-29 10:15:00 | 10863.90 | 2024-08-29 11:15:00 | 10863.90 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-09-03 09:45:00 | 11090.05 | 2024-09-03 11:45:00 | 11138.34 | PARTIAL | 0.50 | 0.44% |
| BUY | retest1 | 2024-09-03 09:45:00 | 11090.05 | 2024-09-03 15:20:00 | 11456.50 | TARGET_HIT | 0.50 | 3.30% |
| SELL | retest1 | 2024-09-06 09:35:00 | 11238.25 | 2024-09-06 09:45:00 | 11186.75 | PARTIAL | 0.50 | 0.46% |
| SELL | retest1 | 2024-09-06 09:35:00 | 11238.25 | 2024-09-06 15:20:00 | 10846.10 | TARGET_HIT | 0.50 | 3.49% |
| BUY | retest1 | 2024-09-13 10:15:00 | 11683.00 | 2024-09-13 10:40:00 | 11734.45 | PARTIAL | 0.50 | 0.44% |
| BUY | retest1 | 2024-09-13 10:15:00 | 11683.00 | 2024-09-13 15:20:00 | 12238.60 | TARGET_HIT | 0.50 | 4.76% |
| BUY | retest1 | 2024-09-16 11:15:00 | 12310.00 | 2024-09-16 11:25:00 | 12260.46 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest1 | 2024-09-17 09:30:00 | 12426.10 | 2024-09-17 09:35:00 | 12489.34 | PARTIAL | 0.50 | 0.51% |
| BUY | retest1 | 2024-09-17 09:30:00 | 12426.10 | 2024-09-17 10:30:00 | 12465.60 | TARGET_HIT | 0.50 | 0.32% |
| SELL | retest1 | 2024-09-25 09:30:00 | 11438.90 | 2024-09-25 09:45:00 | 11375.51 | PARTIAL | 0.50 | 0.55% |
| SELL | retest1 | 2024-09-25 09:30:00 | 11438.90 | 2024-09-25 15:20:00 | 11172.00 | TARGET_HIT | 0.50 | 2.33% |
| BUY | retest1 | 2024-09-26 09:35:00 | 11337.00 | 2024-09-26 09:40:00 | 11301.17 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2024-10-03 09:35:00 | 11432.45 | 2024-10-03 09:50:00 | 11489.15 | PARTIAL | 0.50 | 0.50% |
| BUY | retest1 | 2024-10-03 09:35:00 | 11432.45 | 2024-10-03 10:00:00 | 11432.45 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-10-04 10:40:00 | 11406.50 | 2024-10-04 11:20:00 | 11358.33 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest1 | 2024-10-09 09:50:00 | 11541.00 | 2024-10-09 09:55:00 | 11496.63 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2024-10-10 10:25:00 | 11799.80 | 2024-10-10 10:35:00 | 11758.51 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2024-10-23 10:10:00 | 11170.45 | 2024-10-23 10:50:00 | 11261.83 | PARTIAL | 0.50 | 0.82% |
| BUY | retest1 | 2024-10-23 10:10:00 | 11170.45 | 2024-10-23 14:35:00 | 11223.20 | TARGET_HIT | 0.50 | 0.47% |
| SELL | retest1 | 2024-10-24 09:35:00 | 11089.90 | 2024-10-24 09:40:00 | 11155.23 | STOP_HIT | 1.00 | -0.59% |
| BUY | retest1 | 2024-11-22 09:30:00 | 11455.30 | 2024-11-22 09:35:00 | 11411.79 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest1 | 2024-11-28 11:10:00 | 11600.00 | 2024-11-28 11:40:00 | 11533.88 | PARTIAL | 0.50 | 0.57% |
| SELL | retest1 | 2024-11-28 11:10:00 | 11600.00 | 2024-11-28 12:05:00 | 11600.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-11-29 10:55:00 | 11792.40 | 2024-11-29 11:25:00 | 11747.45 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2024-12-03 11:10:00 | 12476.80 | 2024-12-03 11:25:00 | 12439.02 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2024-12-06 10:30:00 | 12819.75 | 2024-12-06 10:35:00 | 12765.16 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest1 | 2024-12-27 10:50:00 | 12410.35 | 2024-12-27 11:10:00 | 12471.00 | PARTIAL | 0.50 | 0.49% |
| BUY | retest1 | 2024-12-27 10:50:00 | 12410.35 | 2024-12-27 11:20:00 | 12410.35 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-01-01 10:55:00 | 12664.65 | 2025-01-01 11:20:00 | 12596.81 | PARTIAL | 0.50 | 0.54% |
| SELL | retest1 | 2025-01-01 10:55:00 | 12664.65 | 2025-01-01 15:10:00 | 12664.65 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-01-02 09:30:00 | 12720.20 | 2025-01-02 10:00:00 | 12684.63 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2025-01-03 10:55:00 | 12440.00 | 2025-01-03 11:20:00 | 12477.22 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2025-01-06 10:50:00 | 12207.70 | 2025-01-06 11:10:00 | 12133.44 | PARTIAL | 0.50 | 0.61% |
| SELL | retest1 | 2025-01-06 10:50:00 | 12207.70 | 2025-01-06 15:20:00 | 11970.00 | TARGET_HIT | 0.50 | 1.95% |
| BUY | retest1 | 2025-02-07 11:00:00 | 9374.80 | 2025-02-07 11:10:00 | 9431.16 | PARTIAL | 0.50 | 0.60% |
| BUY | retest1 | 2025-02-07 11:00:00 | 9374.80 | 2025-02-07 11:55:00 | 9374.80 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-02-11 10:10:00 | 9197.25 | 2025-02-11 10:20:00 | 9228.08 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2025-02-18 10:25:00 | 8691.55 | 2025-02-18 10:45:00 | 8730.14 | STOP_HIT | 1.00 | -0.44% |
| BUY | retest1 | 2025-02-19 11:10:00 | 8844.45 | 2025-02-19 12:50:00 | 8811.64 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest1 | 2025-02-21 09:35:00 | 8589.00 | 2025-02-21 09:40:00 | 8546.84 | PARTIAL | 0.50 | 0.49% |
| SELL | retest1 | 2025-02-21 09:35:00 | 8589.00 | 2025-02-21 14:10:00 | 8427.05 | TARGET_HIT | 0.50 | 1.89% |
| BUY | retest1 | 2025-03-17 10:40:00 | 7446.00 | 2025-03-17 11:00:00 | 7419.34 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2025-03-18 10:00:00 | 7595.00 | 2025-03-18 10:35:00 | 7571.21 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2025-03-25 09:35:00 | 8104.00 | 2025-03-25 09:55:00 | 8068.29 | STOP_HIT | 1.00 | -0.44% |
| SELL | retest1 | 2025-04-03 09:45:00 | 7544.80 | 2025-04-03 10:00:00 | 7580.65 | STOP_HIT | 1.00 | -0.48% |
| BUY | retest1 | 2025-04-04 09:35:00 | 7633.45 | 2025-04-04 09:40:00 | 7601.04 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest1 | 2025-04-15 09:30:00 | 7632.50 | 2025-04-15 09:35:00 | 7661.78 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2025-04-17 09:45:00 | 7955.00 | 2025-04-17 10:00:00 | 7922.71 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest1 | 2025-04-23 09:40:00 | 8413.50 | 2025-04-23 09:50:00 | 8467.53 | PARTIAL | 0.50 | 0.64% |
| BUY | retest1 | 2025-04-23 09:40:00 | 8413.50 | 2025-04-23 10:05:00 | 8413.50 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-04-29 09:40:00 | 8790.00 | 2025-04-29 09:45:00 | 8826.50 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest1 | 2025-05-05 10:55:00 | 8707.00 | 2025-05-05 11:35:00 | 8673.72 | PARTIAL | 0.50 | 0.38% |
| SELL | retest1 | 2025-05-05 10:55:00 | 8707.00 | 2025-05-05 15:20:00 | 8650.00 | TARGET_HIT | 0.50 | 0.65% |
