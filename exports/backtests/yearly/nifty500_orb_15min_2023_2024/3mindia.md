# 3M India Ltd. (3MINDIA)

## Backtest Summary

- **Window:** 2023-05-12 09:15:00 → 2026-05-08 15:25:00 (55349 bars)
- **Last close:** 32070.00
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
| ENTRY1 | 117 |
| ENTRY2 | 0 |
| PARTIAL | 49 |
| TARGET_HIT | 29 |
| STOP_HIT | 88 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 166 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 78 / 88
- **Target hits / Stop hits / Partials:** 29 / 88 / 49
- **Avg / median % per leg:** 0.25% / 0.00%
- **Sum % (uncompounded):** 41.32%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 70 | 30 | 42.9% | 10 | 40 | 20 | 0.28% | 19.9% |
| BUY @ 2nd Alert (retest1) | 70 | 30 | 42.9% | 10 | 40 | 20 | 0.28% | 19.9% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 96 | 48 | 50.0% | 19 | 48 | 29 | 0.22% | 21.5% |
| SELL @ 2nd Alert (retest1) | 96 | 48 | 50.0% | 19 | 48 | 29 | 0.22% | 21.5% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 166 | 78 | 47.0% | 29 | 88 | 49 | 0.25% | 41.3% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-05-17 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-05-17 10:00:00 | 23415.60 | 23349.66 | 0.00 | ORB-long ORB[23207.00,23341.40] vol=3.0x ATR=31.87 |
| Stop hit — per-position SL triggered | 2023-05-17 10:45:00 | 23383.73 | 23370.68 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2023-05-19 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-05-19 09:30:00 | 23340.00 | 23391.20 | 0.00 | ORB-short ORB[23364.20,23447.40] vol=1.7x ATR=24.95 |
| Stop hit — per-position SL triggered | 2023-05-19 09:40:00 | 23364.95 | 23382.44 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2023-05-24 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-05-24 09:40:00 | 24314.70 | 24256.10 | 0.00 | ORB-long ORB[24044.40,24286.40] vol=2.5x ATR=73.75 |
| Stop hit — per-position SL triggered | 2023-05-24 15:20:00 | 24256.90 | 24290.90 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 4 — SELL (started 2023-05-25 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-05-25 11:05:00 | 24050.00 | 24194.43 | 0.00 | ORB-short ORB[24178.60,24400.00] vol=5.7x ATR=31.13 |
| Stop hit — per-position SL triggered | 2023-05-25 11:10:00 | 24081.13 | 24189.82 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2023-05-26 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-05-26 09:35:00 | 23910.30 | 24033.94 | 0.00 | ORB-short ORB[23952.70,24238.00] vol=2.9x ATR=58.71 |
| Stop hit — per-position SL triggered | 2023-05-26 11:00:00 | 23969.01 | 23962.27 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2023-05-30 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-05-30 10:40:00 | 24079.20 | 24105.67 | 0.00 | ORB-short ORB[24160.00,24300.00] vol=11.0x ATR=61.49 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-05-30 10:45:00 | 23986.96 | 24078.63 | 0.00 | T1 1.5R @ 23986.96 |
| Stop hit — per-position SL triggered | 2023-05-30 11:35:00 | 24079.20 | 24052.22 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2023-06-01 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-01 11:10:00 | 25543.30 | 25337.79 | 0.00 | ORB-long ORB[25100.00,25327.80] vol=2.5x ATR=71.25 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-06-01 11:30:00 | 25650.17 | 25453.90 | 0.00 | T1 1.5R @ 25650.17 |
| Target hit | 2023-06-01 15:20:00 | 26323.40 | 25873.92 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 8 — BUY (started 2023-06-05 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-05 10:05:00 | 26399.40 | 26308.77 | 0.00 | ORB-long ORB[26200.00,26350.00] vol=2.9x ATR=60.41 |
| Stop hit — per-position SL triggered | 2023-06-05 10:10:00 | 26338.99 | 26311.43 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2023-06-06 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-06 09:35:00 | 27027.10 | 26829.01 | 0.00 | ORB-long ORB[26576.00,26800.00] vol=2.4x ATR=110.06 |
| Stop hit — per-position SL triggered | 2023-06-06 09:45:00 | 26917.04 | 26844.55 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2023-06-09 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-09 11:10:00 | 26944.00 | 26805.45 | 0.00 | ORB-long ORB[26685.00,26900.00] vol=4.0x ATR=49.00 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-06-09 11:15:00 | 27017.50 | 26863.71 | 0.00 | T1 1.5R @ 27017.50 |
| Stop hit — per-position SL triggered | 2023-06-09 12:55:00 | 26944.00 | 26914.60 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2023-06-14 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-14 11:00:00 | 27050.20 | 27004.60 | 0.00 | ORB-long ORB[26888.00,27047.20] vol=1.9x ATR=42.13 |
| Stop hit — per-position SL triggered | 2023-06-14 11:30:00 | 27008.07 | 27005.53 | 0.00 | SL hit |

### Cycle 12 — SELL (started 2023-06-20 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-20 10:20:00 | 27721.80 | 27797.29 | 0.00 | ORB-short ORB[27760.00,27921.20] vol=3.0x ATR=37.71 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-06-20 10:25:00 | 27665.24 | 27788.94 | 0.00 | T1 1.5R @ 27665.24 |
| Stop hit — per-position SL triggered | 2023-06-20 12:35:00 | 27721.80 | 27721.06 | 0.00 | SL hit |

### Cycle 13 — SELL (started 2023-06-21 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-21 10:45:00 | 27701.00 | 27820.70 | 0.00 | ORB-short ORB[27732.10,27965.90] vol=1.6x ATR=47.20 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-06-21 10:50:00 | 27630.20 | 27815.29 | 0.00 | T1 1.5R @ 27630.20 |
| Target hit | 2023-06-21 15:20:00 | 27250.00 | 27322.62 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 14 — BUY (started 2023-06-23 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-23 09:35:00 | 26874.90 | 26728.61 | 0.00 | ORB-long ORB[26581.50,26788.80] vol=1.6x ATR=82.58 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-06-23 09:45:00 | 26998.77 | 26774.92 | 0.00 | T1 1.5R @ 26998.77 |
| Target hit | 2023-06-23 11:25:00 | 27100.00 | 27105.01 | 0.00 | Trail-exit close<VWAP |

### Cycle 15 — BUY (started 2023-06-26 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-26 11:10:00 | 27234.80 | 27157.09 | 0.00 | ORB-long ORB[26995.10,27234.40] vol=1.9x ATR=60.02 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-06-26 11:40:00 | 27324.83 | 27187.28 | 0.00 | T1 1.5R @ 27324.83 |
| Target hit | 2023-06-26 15:20:00 | 27463.30 | 27353.38 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 16 — SELL (started 2023-06-27 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-27 09:50:00 | 27446.20 | 27538.01 | 0.00 | ORB-short ORB[27527.20,27636.90] vol=2.9x ATR=69.07 |
| Stop hit — per-position SL triggered | 2023-06-27 10:00:00 | 27515.27 | 27527.53 | 0.00 | SL hit |

### Cycle 17 — BUY (started 2023-06-30 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-30 10:55:00 | 27534.50 | 27382.62 | 0.00 | ORB-long ORB[27274.60,27496.00] vol=4.2x ATR=57.53 |
| Stop hit — per-position SL triggered | 2023-06-30 11:00:00 | 27476.97 | 27386.53 | 0.00 | SL hit |

### Cycle 18 — SELL (started 2023-07-03 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-07-03 11:10:00 | 27075.70 | 27148.98 | 0.00 | ORB-short ORB[27143.60,27339.00] vol=6.5x ATR=54.20 |
| Stop hit — per-position SL triggered | 2023-07-03 11:25:00 | 27129.90 | 27140.13 | 0.00 | SL hit |

### Cycle 19 — SELL (started 2023-07-04 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-07-04 11:05:00 | 27174.40 | 27258.76 | 0.00 | ORB-short ORB[27212.20,27489.90] vol=6.5x ATR=63.40 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-04 11:50:00 | 27079.30 | 27213.95 | 0.00 | T1 1.5R @ 27079.30 |
| Stop hit — per-position SL triggered | 2023-07-04 15:00:00 | 27174.40 | 27184.89 | 0.00 | SL hit |

### Cycle 20 — SELL (started 2023-07-05 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-07-05 10:55:00 | 27119.10 | 27165.06 | 0.00 | ORB-short ORB[27130.40,27230.00] vol=1.9x ATR=32.85 |
| Stop hit — per-position SL triggered | 2023-07-05 11:25:00 | 27151.95 | 27160.62 | 0.00 | SL hit |

### Cycle 21 — SELL (started 2023-07-07 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-07-07 10:55:00 | 27100.00 | 27231.73 | 0.00 | ORB-short ORB[27200.00,27400.00] vol=2.6x ATR=54.68 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-07 11:05:00 | 27017.99 | 27209.94 | 0.00 | T1 1.5R @ 27017.99 |
| Stop hit — per-position SL triggered | 2023-07-07 14:45:00 | 27100.00 | 27104.94 | 0.00 | SL hit |

### Cycle 22 — BUY (started 2023-07-10 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-10 10:30:00 | 27225.00 | 27214.79 | 0.00 | ORB-long ORB[26754.10,27000.00] vol=17.7x ATR=54.98 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-10 10:40:00 | 27307.47 | 27216.33 | 0.00 | T1 1.5R @ 27307.47 |
| Target hit | 2023-07-10 15:20:00 | 27865.20 | 27494.44 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 23 — BUY (started 2023-07-11 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-11 10:40:00 | 28090.60 | 28009.04 | 0.00 | ORB-long ORB[27736.60,28000.00] vol=4.7x ATR=83.51 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-11 10:55:00 | 28215.87 | 28061.49 | 0.00 | T1 1.5R @ 28215.87 |
| Stop hit — per-position SL triggered | 2023-07-11 11:05:00 | 28090.60 | 28071.00 | 0.00 | SL hit |

### Cycle 24 — BUY (started 2023-07-12 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-12 09:40:00 | 28291.50 | 28119.73 | 0.00 | ORB-long ORB[27969.60,28200.00] vol=1.9x ATR=126.70 |
| Stop hit — per-position SL triggered | 2023-07-12 09:50:00 | 28164.80 | 28145.42 | 0.00 | SL hit |

### Cycle 25 — BUY (started 2023-07-13 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-13 09:30:00 | 28750.00 | 28606.38 | 0.00 | ORB-long ORB[28412.20,28690.00] vol=1.6x ATR=86.35 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-13 09:40:00 | 28879.52 | 28719.68 | 0.00 | T1 1.5R @ 28879.52 |
| Stop hit — per-position SL triggered | 2023-07-13 10:05:00 | 28750.00 | 28746.72 | 0.00 | SL hit |

### Cycle 26 — BUY (started 2023-07-14 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-14 10:50:00 | 29099.00 | 28950.48 | 0.00 | ORB-long ORB[28797.00,28999.00] vol=4.1x ATR=74.11 |
| Stop hit — per-position SL triggered | 2023-07-14 11:15:00 | 29024.89 | 28996.59 | 0.00 | SL hit |

### Cycle 27 — SELL (started 2023-07-18 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-07-18 11:00:00 | 28175.00 | 28239.72 | 0.00 | ORB-short ORB[28278.10,28434.60] vol=1.9x ATR=49.84 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-18 11:40:00 | 28100.24 | 28213.97 | 0.00 | T1 1.5R @ 28100.24 |
| Target hit | 2023-07-18 15:20:00 | 27959.70 | 28049.88 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 28 — BUY (started 2023-07-19 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-19 09:45:00 | 28368.00 | 28105.57 | 0.00 | ORB-long ORB[27929.50,28139.00] vol=3.5x ATR=86.79 |
| Stop hit — per-position SL triggered | 2023-07-19 09:50:00 | 28281.21 | 28126.09 | 0.00 | SL hit |

### Cycle 29 — BUY (started 2023-07-28 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-28 09:55:00 | 28038.90 | 27944.56 | 0.00 | ORB-long ORB[27734.40,27924.00] vol=3.1x ATR=81.73 |
| Stop hit — per-position SL triggered | 2023-07-28 10:05:00 | 27957.17 | 27946.59 | 0.00 | SL hit |

### Cycle 30 — BUY (started 2023-07-31 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-31 10:00:00 | 28220.90 | 27990.99 | 0.00 | ORB-long ORB[27776.50,28038.20] vol=3.7x ATR=90.00 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-31 10:05:00 | 28355.90 | 28075.79 | 0.00 | T1 1.5R @ 28355.90 |
| Target hit | 2023-07-31 15:20:00 | 28937.30 | 28634.13 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 31 — SELL (started 2023-08-01 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-01 10:40:00 | 28565.30 | 28603.83 | 0.00 | ORB-short ORB[28702.00,29097.90] vol=16.3x ATR=74.47 |
| Stop hit — per-position SL triggered | 2023-08-01 10:55:00 | 28639.77 | 28596.80 | 0.00 | SL hit |

### Cycle 32 — BUY (started 2023-08-02 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-02 09:45:00 | 28779.20 | 28721.35 | 0.00 | ORB-long ORB[28619.00,28750.00] vol=3.0x ATR=44.53 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-02 10:05:00 | 28846.00 | 28740.36 | 0.00 | T1 1.5R @ 28846.00 |
| Stop hit — per-position SL triggered | 2023-08-02 10:30:00 | 28779.20 | 28757.44 | 0.00 | SL hit |

### Cycle 33 — BUY (started 2023-08-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-08 10:15:00 | 27936.00 | 27725.18 | 0.00 | ORB-long ORB[27362.50,27721.80] vol=1.8x ATR=113.21 |
| Stop hit — per-position SL triggered | 2023-08-08 10:50:00 | 27822.79 | 27740.33 | 0.00 | SL hit |

### Cycle 34 — SELL (started 2023-08-09 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-09 10:25:00 | 27131.80 | 27311.20 | 0.00 | ORB-short ORB[27504.40,27696.10] vol=6.2x ATR=75.96 |
| Stop hit — per-position SL triggered | 2023-08-09 10:30:00 | 27207.76 | 27305.81 | 0.00 | SL hit |

### Cycle 35 — BUY (started 2023-08-11 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-11 09:50:00 | 28158.30 | 28062.85 | 0.00 | ORB-long ORB[27728.20,28067.70] vol=1.7x ATR=148.38 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-11 10:30:00 | 28380.87 | 28182.46 | 0.00 | T1 1.5R @ 28380.87 |
| Target hit | 2023-08-11 15:20:00 | 29444.70 | 28834.18 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 36 — BUY (started 2023-08-16 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-16 09:35:00 | 28737.40 | 28589.77 | 0.00 | ORB-long ORB[28420.10,28650.00] vol=2.2x ATR=99.67 |
| Stop hit — per-position SL triggered | 2023-08-16 09:40:00 | 28637.73 | 28601.76 | 0.00 | SL hit |

### Cycle 37 — BUY (started 2023-08-17 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-17 09:45:00 | 29020.00 | 28891.49 | 0.00 | ORB-long ORB[28692.40,28961.90] vol=2.1x ATR=124.77 |
| Stop hit — per-position SL triggered | 2023-08-17 12:05:00 | 28895.23 | 28952.57 | 0.00 | SL hit |

### Cycle 38 — BUY (started 2023-08-18 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-18 09:50:00 | 29279.30 | 29133.54 | 0.00 | ORB-long ORB[28989.60,29143.90] vol=1.9x ATR=87.41 |
| Stop hit — per-position SL triggered | 2023-08-18 10:35:00 | 29191.89 | 29208.93 | 0.00 | SL hit |

### Cycle 39 — BUY (started 2023-08-24 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-24 09:40:00 | 30560.00 | 30431.14 | 0.00 | ORB-long ORB[30281.10,30499.90] vol=1.5x ATR=86.04 |
| Stop hit — per-position SL triggered | 2023-08-24 09:55:00 | 30473.96 | 30456.71 | 0.00 | SL hit |

### Cycle 40 — BUY (started 2023-08-29 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-29 11:10:00 | 30694.10 | 30613.25 | 0.00 | ORB-long ORB[30387.10,30689.90] vol=3.0x ATR=47.90 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-29 11:30:00 | 30765.95 | 30664.22 | 0.00 | T1 1.5R @ 30765.95 |
| Stop hit — per-position SL triggered | 2023-08-29 12:00:00 | 30694.10 | 30672.80 | 0.00 | SL hit |

### Cycle 41 — SELL (started 2023-09-12 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-09-12 09:35:00 | 32331.80 | 32499.10 | 0.00 | ORB-short ORB[32404.70,32800.00] vol=2.4x ATR=92.60 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-12 09:40:00 | 32192.89 | 32392.89 | 0.00 | T1 1.5R @ 32192.89 |
| Target hit | 2023-09-12 10:20:00 | 32125.20 | 32110.84 | 0.00 | Trail-exit close>VWAP |

### Cycle 42 — SELL (started 2023-09-14 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-09-14 10:50:00 | 31014.10 | 31218.38 | 0.00 | ORB-short ORB[31149.00,31490.20] vol=2.3x ATR=73.66 |
| Stop hit — per-position SL triggered | 2023-09-14 12:20:00 | 31087.76 | 31143.03 | 0.00 | SL hit |

### Cycle 43 — SELL (started 2023-09-15 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-09-15 10:40:00 | 31071.00 | 31202.22 | 0.00 | ORB-short ORB[31150.10,31384.40] vol=2.3x ATR=60.67 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-15 10:50:00 | 30979.99 | 31110.37 | 0.00 | T1 1.5R @ 30979.99 |
| Target hit | 2023-09-15 15:20:00 | 30183.90 | 30635.27 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 44 — BUY (started 2023-09-25 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-25 11:05:00 | 31847.40 | 31670.49 | 0.00 | ORB-long ORB[31113.00,31446.70] vol=2.0x ATR=98.81 |
| Stop hit — per-position SL triggered | 2023-09-25 13:10:00 | 31748.59 | 31721.49 | 0.00 | SL hit |

### Cycle 45 — SELL (started 2023-09-26 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-09-26 11:10:00 | 31070.00 | 31458.24 | 0.00 | ORB-short ORB[31522.00,31936.10] vol=2.3x ATR=79.15 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-26 12:00:00 | 30951.27 | 31379.74 | 0.00 | T1 1.5R @ 30951.27 |
| Stop hit — per-position SL triggered | 2023-09-26 12:30:00 | 31070.00 | 31121.19 | 0.00 | SL hit |

### Cycle 46 — SELL (started 2023-09-28 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-09-28 10:35:00 | 31096.00 | 31182.77 | 0.00 | ORB-short ORB[31204.40,31317.40] vol=2.1x ATR=45.71 |
| Stop hit — per-position SL triggered | 2023-09-28 10:40:00 | 31141.71 | 31176.03 | 0.00 | SL hit |

### Cycle 47 — BUY (started 2023-09-29 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-29 10:35:00 | 31093.90 | 30957.94 | 0.00 | ORB-long ORB[30839.20,31059.30] vol=1.9x ATR=86.83 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-29 11:00:00 | 31224.14 | 31454.46 | 0.00 | T1 1.5R @ 31224.14 |
| Target hit | 2023-09-29 11:10:00 | 31960.70 | 32052.39 | 0.00 | Trail-exit close<VWAP |

### Cycle 48 — SELL (started 2023-10-03 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-10-03 09:35:00 | 31639.90 | 31763.30 | 0.00 | ORB-short ORB[31681.90,32064.80] vol=2.6x ATR=170.94 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-03 14:55:00 | 31383.49 | 31624.02 | 0.00 | T1 1.5R @ 31383.49 |
| Target hit | 2023-10-03 15:20:00 | 31323.10 | 31602.40 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 49 — SELL (started 2023-10-04 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-10-04 11:00:00 | 31112.20 | 31321.11 | 0.00 | ORB-short ORB[31301.00,31537.40] vol=2.9x ATR=63.00 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-04 11:15:00 | 31017.69 | 31294.65 | 0.00 | T1 1.5R @ 31017.69 |
| Target hit | 2023-10-04 14:20:00 | 31057.30 | 31033.61 | 0.00 | Trail-exit close>VWAP |

### Cycle 50 — BUY (started 2023-10-06 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-06 09:35:00 | 31264.10 | 31146.55 | 0.00 | ORB-long ORB[30950.00,31100.00] vol=2.2x ATR=70.61 |
| Stop hit — per-position SL triggered | 2023-10-06 09:45:00 | 31193.49 | 31165.77 | 0.00 | SL hit |

### Cycle 51 — BUY (started 2023-10-11 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-11 09:35:00 | 31262.10 | 31205.83 | 0.00 | ORB-long ORB[31000.00,31216.80] vol=2.0x ATR=73.10 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-11 09:45:00 | 31371.74 | 31262.23 | 0.00 | T1 1.5R @ 31371.74 |
| Stop hit — per-position SL triggered | 2023-10-11 10:20:00 | 31262.10 | 31279.68 | 0.00 | SL hit |

### Cycle 52 — SELL (started 2023-10-12 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-10-12 10:45:00 | 30900.10 | 31051.60 | 0.00 | ORB-short ORB[30937.10,31377.80] vol=4.5x ATR=91.14 |
| Stop hit — per-position SL triggered | 2023-10-12 12:30:00 | 30991.24 | 31015.33 | 0.00 | SL hit |

### Cycle 53 — SELL (started 2023-10-13 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-10-13 10:50:00 | 30823.20 | 30878.10 | 0.00 | ORB-short ORB[30885.60,31271.00] vol=3.1x ATR=51.16 |
| Stop hit — per-position SL triggered | 2023-10-13 10:55:00 | 30874.36 | 30877.90 | 0.00 | SL hit |

### Cycle 54 — SELL (started 2023-10-18 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-10-18 10:45:00 | 30275.80 | 30514.37 | 0.00 | ORB-short ORB[30438.20,30787.90] vol=2.5x ATR=69.94 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-18 11:05:00 | 30170.89 | 30458.24 | 0.00 | T1 1.5R @ 30170.89 |
| Stop hit — per-position SL triggered | 2023-10-18 12:55:00 | 30275.80 | 30367.02 | 0.00 | SL hit |

### Cycle 55 — BUY (started 2023-10-19 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-19 10:30:00 | 30443.30 | 30359.68 | 0.00 | ORB-long ORB[30258.40,30410.00] vol=2.9x ATR=65.99 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-19 10:35:00 | 30542.28 | 30370.34 | 0.00 | T1 1.5R @ 30542.28 |
| Stop hit — per-position SL triggered | 2023-10-19 12:20:00 | 30443.30 | 30455.68 | 0.00 | SL hit |

### Cycle 56 — SELL (started 2023-10-20 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-10-20 09:55:00 | 30408.20 | 30484.29 | 0.00 | ORB-short ORB[30439.60,30673.40] vol=1.8x ATR=74.35 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-20 10:40:00 | 30296.68 | 30419.00 | 0.00 | T1 1.5R @ 30296.68 |
| Target hit | 2023-10-20 15:20:00 | 30066.80 | 30279.76 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 57 — SELL (started 2023-10-23 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-10-23 10:10:00 | 29820.00 | 29966.77 | 0.00 | ORB-short ORB[29950.00,30175.90] vol=1.5x ATR=74.28 |
| Stop hit — per-position SL triggered | 2023-10-23 10:15:00 | 29894.28 | 29943.11 | 0.00 | SL hit |

### Cycle 58 — SELL (started 2023-10-25 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-10-25 09:30:00 | 29450.20 | 29651.88 | 0.00 | ORB-short ORB[29514.80,29899.60] vol=1.7x ATR=133.73 |
| Stop hit — per-position SL triggered | 2023-10-25 09:40:00 | 29583.93 | 29622.11 | 0.00 | SL hit |

### Cycle 59 — SELL (started 2023-10-26 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-10-26 09:55:00 | 28968.20 | 29107.83 | 0.00 | ORB-short ORB[29070.20,29382.30] vol=1.8x ATR=99.53 |
| Stop hit — per-position SL triggered | 2023-10-26 10:25:00 | 29067.73 | 29089.32 | 0.00 | SL hit |

### Cycle 60 — SELL (started 2023-10-31 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-10-31 09:55:00 | 29591.40 | 29707.01 | 0.00 | ORB-short ORB[29640.30,29869.90] vol=1.5x ATR=64.06 |
| Stop hit — per-position SL triggered | 2023-10-31 10:05:00 | 29655.46 | 29686.46 | 0.00 | SL hit |

### Cycle 61 — BUY (started 2023-11-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-01 11:15:00 | 29898.90 | 29739.45 | 0.00 | ORB-long ORB[29633.80,29844.40] vol=4.9x ATR=69.16 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-01 11:25:00 | 30002.63 | 29767.06 | 0.00 | T1 1.5R @ 30002.63 |
| Stop hit — per-position SL triggered | 2023-11-01 11:35:00 | 29898.90 | 29801.91 | 0.00 | SL hit |

### Cycle 62 — BUY (started 2023-11-02 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-02 10:50:00 | 30600.00 | 30478.86 | 0.00 | ORB-long ORB[30320.90,30497.40] vol=7.1x ATR=61.64 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-02 11:05:00 | 30692.47 | 30533.88 | 0.00 | T1 1.5R @ 30692.47 |
| Target hit | 2023-11-02 15:20:00 | 31398.90 | 30972.09 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 63 — BUY (started 2023-11-06 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-06 09:50:00 | 31661.90 | 31476.64 | 0.00 | ORB-long ORB[31310.00,31446.60] vol=3.5x ATR=82.80 |
| Stop hit — per-position SL triggered | 2023-11-06 10:00:00 | 31579.10 | 31494.93 | 0.00 | SL hit |

### Cycle 64 — SELL (started 2023-11-07 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-11-07 09:50:00 | 31207.10 | 31249.16 | 0.00 | ORB-short ORB[31277.20,31450.80] vol=1.7x ATR=61.98 |
| Stop hit — per-position SL triggered | 2023-11-07 10:05:00 | 31269.08 | 31244.43 | 0.00 | SL hit |

### Cycle 65 — BUY (started 2023-11-08 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-08 09:40:00 | 31511.80 | 31402.27 | 0.00 | ORB-long ORB[31239.20,31337.70] vol=3.0x ATR=94.96 |
| Stop hit — per-position SL triggered | 2023-11-08 09:50:00 | 31416.84 | 31403.43 | 0.00 | SL hit |

### Cycle 66 — SELL (started 2023-11-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-11-17 11:15:00 | 30749.90 | 30818.51 | 0.00 | ORB-short ORB[30793.60,30979.90] vol=5.7x ATR=44.31 |
| Stop hit — per-position SL triggered | 2023-11-17 11:25:00 | 30794.21 | 30818.36 | 0.00 | SL hit |

### Cycle 67 — BUY (started 2023-11-20 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-20 10:45:00 | 30784.70 | 30686.58 | 0.00 | ORB-long ORB[30462.20,30705.70] vol=3.5x ATR=84.72 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-20 11:15:00 | 30911.78 | 30740.52 | 0.00 | T1 1.5R @ 30911.78 |
| Stop hit — per-position SL triggered | 2023-11-20 11:45:00 | 30784.70 | 30754.68 | 0.00 | SL hit |

### Cycle 68 — SELL (started 2023-11-22 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-11-22 09:50:00 | 30814.90 | 30839.59 | 0.00 | ORB-short ORB[30856.20,30996.60] vol=3.7x ATR=89.35 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-22 11:25:00 | 30680.88 | 30771.46 | 0.00 | T1 1.5R @ 30680.88 |
| Stop hit — per-position SL triggered | 2023-11-22 13:05:00 | 30814.90 | 30750.69 | 0.00 | SL hit |

### Cycle 69 — SELL (started 2023-11-23 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-11-23 09:30:00 | 30703.10 | 30774.06 | 0.00 | ORB-short ORB[30778.30,30900.00] vol=2.6x ATR=94.78 |
| Stop hit — per-position SL triggered | 2023-11-23 10:00:00 | 30797.88 | 30743.92 | 0.00 | SL hit |

### Cycle 70 — BUY (started 2023-11-28 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-28 10:40:00 | 30486.60 | 30440.40 | 0.00 | ORB-long ORB[30306.00,30482.80] vol=6.0x ATR=53.71 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-28 11:30:00 | 30567.16 | 30448.23 | 0.00 | T1 1.5R @ 30567.16 |
| Stop hit — per-position SL triggered | 2023-11-28 14:10:00 | 30486.60 | 30475.29 | 0.00 | SL hit |

### Cycle 71 — BUY (started 2023-12-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-01 10:15:00 | 31444.90 | 31303.41 | 0.00 | ORB-long ORB[31025.10,31297.30] vol=1.7x ATR=93.93 |
| Stop hit — per-position SL triggered | 2023-12-01 10:20:00 | 31350.97 | 31318.06 | 0.00 | SL hit |

### Cycle 72 — SELL (started 2023-12-05 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-12-05 11:05:00 | 31166.40 | 31213.61 | 0.00 | ORB-short ORB[31224.70,31364.90] vol=1.7x ATR=51.57 |
| Stop hit — per-position SL triggered | 2023-12-05 11:45:00 | 31217.97 | 31205.82 | 0.00 | SL hit |

### Cycle 73 — BUY (started 2023-12-07 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-07 10:55:00 | 31188.40 | 31014.03 | 0.00 | ORB-long ORB[30903.20,31100.00] vol=3.0x ATR=57.69 |
| Stop hit — per-position SL triggered | 2023-12-07 11:00:00 | 31130.71 | 31018.56 | 0.00 | SL hit |

### Cycle 74 — SELL (started 2023-12-08 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-12-08 11:05:00 | 31274.20 | 31413.17 | 0.00 | ORB-short ORB[31366.60,31531.40] vol=3.3x ATR=59.16 |
| Stop hit — per-position SL triggered | 2023-12-08 11:25:00 | 31333.36 | 31398.37 | 0.00 | SL hit |

### Cycle 75 — SELL (started 2023-12-11 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-12-11 10:10:00 | 31324.50 | 31407.30 | 0.00 | ORB-short ORB[31485.10,31630.80] vol=2.2x ATR=59.99 |
| Stop hit — per-position SL triggered | 2023-12-11 11:00:00 | 31384.49 | 31377.65 | 0.00 | SL hit |

### Cycle 76 — SELL (started 2023-12-12 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-12-12 11:00:00 | 31499.90 | 31604.39 | 0.00 | ORB-short ORB[31625.90,31807.20] vol=1.8x ATR=58.56 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-12 11:25:00 | 31412.07 | 31557.62 | 0.00 | T1 1.5R @ 31412.07 |
| Target hit | 2023-12-12 15:20:00 | 31237.20 | 31389.98 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 77 — SELL (started 2023-12-14 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-12-14 09:55:00 | 31200.00 | 31425.81 | 0.00 | ORB-short ORB[31290.10,31644.00] vol=3.7x ATR=102.77 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-14 12:05:00 | 31045.84 | 31238.96 | 0.00 | T1 1.5R @ 31045.84 |
| Stop hit — per-position SL triggered | 2023-12-14 15:00:00 | 31200.00 | 31206.24 | 0.00 | SL hit |

### Cycle 78 — SELL (started 2023-12-15 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-12-15 09:35:00 | 31097.30 | 31244.31 | 0.00 | ORB-short ORB[31229.10,31400.00] vol=3.0x ATR=82.63 |
| Stop hit — per-position SL triggered | 2023-12-15 15:00:00 | 31179.93 | 31153.32 | 0.00 | SL hit |

### Cycle 79 — BUY (started 2023-12-19 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-19 09:40:00 | 31396.00 | 31276.61 | 0.00 | ORB-long ORB[31141.20,31307.20] vol=3.4x ATR=103.32 |
| Stop hit — per-position SL triggered | 2023-12-19 09:45:00 | 31292.68 | 31306.33 | 0.00 | SL hit |

### Cycle 80 — SELL (started 2023-12-20 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-12-20 10:30:00 | 31298.60 | 31399.40 | 0.00 | ORB-short ORB[31418.40,31526.10] vol=1.8x ATR=65.42 |
| Stop hit — per-position SL triggered | 2023-12-20 11:55:00 | 31364.02 | 31358.95 | 0.00 | SL hit |

### Cycle 81 — SELL (started 2023-12-22 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-12-22 10:00:00 | 31351.00 | 31410.70 | 0.00 | ORB-short ORB[31358.30,31563.60] vol=5.8x ATR=91.52 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-22 10:30:00 | 31213.72 | 31360.26 | 0.00 | T1 1.5R @ 31213.72 |
| Target hit | 2023-12-22 14:50:00 | 31233.70 | 31225.38 | 0.00 | Trail-exit close>VWAP |

### Cycle 82 — BUY (started 2023-12-26 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-26 09:40:00 | 31419.90 | 31285.56 | 0.00 | ORB-long ORB[31127.00,31323.90] vol=3.2x ATR=78.36 |
| Stop hit — per-position SL triggered | 2023-12-26 09:50:00 | 31341.54 | 31299.87 | 0.00 | SL hit |

### Cycle 83 — SELL (started 2024-01-02 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-02 09:30:00 | 36454.90 | 36723.57 | 0.00 | ORB-short ORB[36570.00,36934.20] vol=1.7x ATR=153.46 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-02 09:55:00 | 36224.71 | 36627.03 | 0.00 | T1 1.5R @ 36224.71 |
| Target hit | 2024-01-02 15:20:00 | 35604.20 | 35905.81 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 84 — BUY (started 2024-01-09 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-09 10:05:00 | 34748.00 | 34440.73 | 0.00 | ORB-long ORB[34180.50,34607.20] vol=1.6x ATR=163.54 |
| Stop hit — per-position SL triggered | 2024-01-09 10:15:00 | 34584.46 | 34470.62 | 0.00 | SL hit |

### Cycle 85 — SELL (started 2024-01-15 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-15 09:40:00 | 34000.00 | 34119.70 | 0.00 | ORB-short ORB[34092.00,34264.40] vol=3.2x ATR=75.86 |
| Stop hit — per-position SL triggered | 2024-01-15 09:50:00 | 34075.86 | 34110.78 | 0.00 | SL hit |

### Cycle 86 — BUY (started 2024-01-16 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-16 10:50:00 | 34588.20 | 34557.26 | 0.00 | ORB-long ORB[34387.10,34574.60] vol=2.0x ATR=78.47 |
| Stop hit — per-position SL triggered | 2024-01-16 11:40:00 | 34509.73 | 34556.52 | 0.00 | SL hit |

### Cycle 87 — SELL (started 2024-01-18 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-18 09:35:00 | 33942.50 | 34129.41 | 0.00 | ORB-short ORB[34091.80,34418.50] vol=3.7x ATR=119.67 |
| Stop hit — per-position SL triggered | 2024-01-18 09:40:00 | 34062.17 | 34112.87 | 0.00 | SL hit |

### Cycle 88 — SELL (started 2024-01-19 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-19 11:05:00 | 33980.00 | 34187.18 | 0.00 | ORB-short ORB[34141.40,34484.90] vol=4.0x ATR=92.43 |
| Stop hit — per-position SL triggered | 2024-01-19 12:45:00 | 34072.43 | 34110.73 | 0.00 | SL hit |

### Cycle 89 — SELL (started 2024-01-29 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-29 10:05:00 | 33478.10 | 33774.78 | 0.00 | ORB-short ORB[33777.00,34259.90] vol=1.8x ATR=115.86 |
| Stop hit — per-position SL triggered | 2024-01-29 10:15:00 | 33593.96 | 33760.45 | 0.00 | SL hit |

### Cycle 90 — BUY (started 2024-01-30 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-30 09:55:00 | 33972.80 | 33930.48 | 0.00 | ORB-long ORB[33810.00,33950.60] vol=3.2x ATR=76.02 |
| Stop hit — per-position SL triggered | 2024-01-30 10:30:00 | 33896.78 | 33950.81 | 0.00 | SL hit |

### Cycle 91 — SELL (started 2024-02-01 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-01 10:20:00 | 33910.00 | 34145.86 | 0.00 | ORB-short ORB[34068.40,34500.00] vol=3.3x ATR=127.62 |
| Target hit | 2024-02-01 15:20:00 | 33805.10 | 33949.21 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 92 — SELL (started 2024-02-02 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-02 09:30:00 | 33670.30 | 33837.25 | 0.00 | ORB-short ORB[33825.20,34039.00] vol=1.7x ATR=89.46 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-02 09:40:00 | 33536.12 | 33752.64 | 0.00 | T1 1.5R @ 33536.12 |
| Target hit | 2024-02-02 15:20:00 | 32890.00 | 33122.35 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 93 — BUY (started 2024-02-08 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-08 09:50:00 | 34299.90 | 34151.46 | 0.00 | ORB-long ORB[33985.00,34254.80] vol=1.6x ATR=74.09 |
| Stop hit — per-position SL triggered | 2024-02-08 10:05:00 | 34225.81 | 34175.74 | 0.00 | SL hit |

### Cycle 94 — SELL (started 2024-02-12 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-12 10:50:00 | 32137.80 | 32548.55 | 0.00 | ORB-short ORB[32465.80,32857.40] vol=1.8x ATR=116.19 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-12 11:50:00 | 31963.51 | 32357.01 | 0.00 | T1 1.5R @ 31963.51 |
| Target hit | 2024-02-12 15:20:00 | 31780.10 | 32053.59 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 95 — SELL (started 2024-02-14 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-14 10:55:00 | 31168.20 | 31343.42 | 0.00 | ORB-short ORB[31277.10,31599.90] vol=1.9x ATR=60.39 |
| Stop hit — per-position SL triggered | 2024-02-14 11:00:00 | 31228.59 | 31329.33 | 0.00 | SL hit |

### Cycle 96 — SELL (started 2024-02-15 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-15 11:10:00 | 30709.20 | 30891.68 | 0.00 | ORB-short ORB[30818.70,31174.40] vol=4.9x ATR=79.04 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-15 11:20:00 | 30590.63 | 30854.09 | 0.00 | T1 1.5R @ 30590.63 |
| Stop hit — per-position SL triggered | 2024-02-15 11:45:00 | 30709.20 | 30815.16 | 0.00 | SL hit |

### Cycle 97 — SELL (started 2024-02-21 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-21 10:50:00 | 31150.00 | 31363.65 | 0.00 | ORB-short ORB[31241.90,31520.00] vol=2.6x ATR=84.33 |
| Stop hit — per-position SL triggered | 2024-02-21 11:05:00 | 31234.33 | 31342.25 | 0.00 | SL hit |

### Cycle 98 — BUY (started 2024-02-26 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-26 11:10:00 | 30620.40 | 30519.24 | 0.00 | ORB-long ORB[30438.00,30587.90] vol=1.7x ATR=52.60 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-26 11:40:00 | 30699.29 | 30532.93 | 0.00 | T1 1.5R @ 30699.29 |
| Target hit | 2024-02-26 15:15:00 | 30664.75 | 30680.76 | 0.00 | Trail-exit close<VWAP |

### Cycle 99 — BUY (started 2024-02-27 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-27 10:50:00 | 31204.75 | 31112.59 | 0.00 | ORB-long ORB[30817.00,31125.00] vol=2.9x ATR=56.41 |
| Stop hit — per-position SL triggered | 2024-02-27 11:35:00 | 31148.34 | 31131.98 | 0.00 | SL hit |

### Cycle 100 — SELL (started 2024-03-01 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-03-01 10:35:00 | 31770.35 | 31805.67 | 0.00 | ORB-short ORB[31823.85,32275.00] vol=4.1x ATR=113.93 |
| Stop hit — per-position SL triggered | 2024-03-01 11:10:00 | 31884.28 | 31803.49 | 0.00 | SL hit |

### Cycle 101 — SELL (started 2024-03-05 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-03-05 09:45:00 | 30551.00 | 30662.06 | 0.00 | ORB-short ORB[30600.05,30885.80] vol=1.9x ATR=102.89 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-05 10:45:00 | 30396.67 | 30542.13 | 0.00 | T1 1.5R @ 30396.67 |
| Target hit | 2024-03-05 11:30:00 | 30521.45 | 30493.96 | 0.00 | Trail-exit close>VWAP |

### Cycle 102 — SELL (started 2024-03-15 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-03-15 10:10:00 | 29705.05 | 29884.58 | 0.00 | ORB-short ORB[30150.05,30334.25] vol=15.7x ATR=130.62 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-15 10:15:00 | 29509.12 | 29705.34 | 0.00 | T1 1.5R @ 29509.12 |
| Stop hit — per-position SL triggered | 2024-03-15 10:20:00 | 29705.05 | 29674.17 | 0.00 | SL hit |

### Cycle 103 — SELL (started 2024-03-26 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-03-26 10:25:00 | 30542.45 | 30736.05 | 0.00 | ORB-short ORB[30765.20,31000.00] vol=1.6x ATR=88.78 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-26 11:00:00 | 30409.28 | 30654.99 | 0.00 | T1 1.5R @ 30409.28 |
| Target hit | 2024-03-26 15:20:00 | 30484.55 | 30539.37 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 104 — SELL (started 2024-03-27 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-03-27 10:30:00 | 30374.35 | 30425.70 | 0.00 | ORB-short ORB[30400.05,30788.00] vol=6.5x ATR=72.77 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-27 10:50:00 | 30265.19 | 30399.09 | 0.00 | T1 1.5R @ 30265.19 |
| Stop hit — per-position SL triggered | 2024-03-27 10:55:00 | 30374.35 | 30394.70 | 0.00 | SL hit |

### Cycle 105 — SELL (started 2024-04-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-04-08 11:15:00 | 30290.00 | 30414.16 | 0.00 | ORB-short ORB[30348.45,30556.45] vol=1.5x ATR=50.12 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-08 11:45:00 | 30214.82 | 30368.00 | 0.00 | T1 1.5R @ 30214.82 |
| Target hit | 2024-04-08 15:20:00 | 29998.35 | 30055.38 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 106 — SELL (started 2024-04-10 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-04-10 10:25:00 | 29959.95 | 30080.59 | 0.00 | ORB-short ORB[30112.95,30348.95] vol=2.1x ATR=62.27 |
| Stop hit — per-position SL triggered | 2024-04-10 10:40:00 | 30022.22 | 30061.15 | 0.00 | SL hit |

### Cycle 107 — SELL (started 2024-04-12 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-04-12 10:40:00 | 29817.00 | 29887.96 | 0.00 | ORB-short ORB[29850.90,30150.00] vol=1.9x ATR=57.04 |
| Stop hit — per-position SL triggered | 2024-04-12 11:35:00 | 29874.04 | 29875.66 | 0.00 | SL hit |

### Cycle 108 — SELL (started 2024-04-18 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-04-18 09:40:00 | 29670.00 | 29721.06 | 0.00 | ORB-short ORB[29707.55,29999.95] vol=3.9x ATR=75.43 |
| Stop hit — per-position SL triggered | 2024-04-18 09:50:00 | 29745.43 | 29718.27 | 0.00 | SL hit |

### Cycle 109 — BUY (started 2024-04-22 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-22 11:10:00 | 30078.85 | 29882.50 | 0.00 | ORB-long ORB[29673.05,29900.00] vol=10.3x ATR=65.68 |
| Stop hit — per-position SL triggered | 2024-04-22 11:25:00 | 30013.17 | 29889.38 | 0.00 | SL hit |

### Cycle 110 — BUY (started 2024-04-24 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-24 09:40:00 | 30288.95 | 30092.07 | 0.00 | ORB-long ORB[29905.95,30138.30] vol=2.6x ATR=78.01 |
| Stop hit — per-position SL triggered | 2024-04-24 09:45:00 | 30210.94 | 30122.84 | 0.00 | SL hit |

### Cycle 111 — SELL (started 2024-04-25 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-04-25 10:40:00 | 30000.05 | 30093.46 | 0.00 | ORB-short ORB[30050.10,30300.05] vol=11.2x ATR=77.02 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-25 14:35:00 | 29884.53 | 30065.97 | 0.00 | T1 1.5R @ 29884.53 |
| Target hit | 2024-04-25 15:20:00 | 29986.95 | 30043.81 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 112 — SELL (started 2024-04-26 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-04-26 10:00:00 | 29972.20 | 30020.30 | 0.00 | ORB-short ORB[29981.40,30239.65] vol=1.6x ATR=70.56 |
| Stop hit — per-position SL triggered | 2024-04-26 11:40:00 | 30042.76 | 29984.49 | 0.00 | SL hit |

### Cycle 113 — SELL (started 2024-04-29 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-04-29 10:25:00 | 29862.75 | 29996.69 | 0.00 | ORB-short ORB[29888.00,30127.40] vol=1.9x ATR=89.98 |
| Stop hit — per-position SL triggered | 2024-04-29 10:40:00 | 29952.73 | 29945.23 | 0.00 | SL hit |

### Cycle 114 — BUY (started 2024-04-30 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-30 09:40:00 | 30251.25 | 30125.28 | 0.00 | ORB-long ORB[29936.15,30130.40] vol=1.6x ATR=133.91 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-30 10:10:00 | 30452.11 | 30291.99 | 0.00 | T1 1.5R @ 30452.11 |
| Target hit | 2024-04-30 15:20:00 | 30480.15 | 30414.94 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 115 — SELL (started 2024-05-07 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-07 10:35:00 | 29345.00 | 29480.41 | 0.00 | ORB-short ORB[29488.45,29693.95] vol=2.5x ATR=70.63 |
| Stop hit — per-position SL triggered | 2024-05-07 11:00:00 | 29415.63 | 29434.24 | 0.00 | SL hit |

### Cycle 116 — SELL (started 2024-05-08 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-08 09:30:00 | 29389.00 | 29478.05 | 0.00 | ORB-short ORB[29406.85,29748.20] vol=2.2x ATR=137.33 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-08 10:25:00 | 29183.00 | 29327.78 | 0.00 | T1 1.5R @ 29183.00 |
| Target hit | 2024-05-08 15:20:00 | 29190.00 | 29213.36 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 117 — SELL (started 2024-05-09 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-09 09:55:00 | 29040.95 | 29163.15 | 0.00 | ORB-short ORB[29139.35,29351.90] vol=1.5x ATR=62.89 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-09 10:20:00 | 28946.62 | 29110.80 | 0.00 | T1 1.5R @ 28946.62 |
| Target hit | 2024-05-09 15:20:00 | 28769.85 | 28913.73 | 0.00 | EOD square-off @ 15:20:00 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-05-17 10:00:00 | 23415.60 | 2023-05-17 10:45:00 | 23383.73 | STOP_HIT | 1.00 | -0.14% |
| SELL | retest1 | 2023-05-19 09:30:00 | 23340.00 | 2023-05-19 09:40:00 | 23364.95 | STOP_HIT | 1.00 | -0.11% |
| BUY | retest1 | 2023-05-24 09:40:00 | 24314.70 | 2023-05-24 15:20:00 | 24256.90 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2023-05-25 11:05:00 | 24050.00 | 2023-05-25 11:10:00 | 24081.13 | STOP_HIT | 1.00 | -0.13% |
| SELL | retest1 | 2023-05-26 09:35:00 | 23910.30 | 2023-05-26 11:00:00 | 23969.01 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2023-05-30 10:40:00 | 24079.20 | 2023-05-30 10:45:00 | 23986.96 | PARTIAL | 0.50 | 0.38% |
| SELL | retest1 | 2023-05-30 10:40:00 | 24079.20 | 2023-05-30 11:35:00 | 24079.20 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-06-01 11:10:00 | 25543.30 | 2023-06-01 11:30:00 | 25650.17 | PARTIAL | 0.50 | 0.42% |
| BUY | retest1 | 2023-06-01 11:10:00 | 25543.30 | 2023-06-01 15:20:00 | 26323.40 | TARGET_HIT | 0.50 | 3.05% |
| BUY | retest1 | 2023-06-05 10:05:00 | 26399.40 | 2023-06-05 10:10:00 | 26338.99 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2023-06-06 09:35:00 | 27027.10 | 2023-06-06 09:45:00 | 26917.04 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest1 | 2023-06-09 11:10:00 | 26944.00 | 2023-06-09 11:15:00 | 27017.50 | PARTIAL | 0.50 | 0.27% |
| BUY | retest1 | 2023-06-09 11:10:00 | 26944.00 | 2023-06-09 12:55:00 | 26944.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-06-14 11:00:00 | 27050.20 | 2023-06-14 11:30:00 | 27008.07 | STOP_HIT | 1.00 | -0.16% |
| SELL | retest1 | 2023-06-20 10:20:00 | 27721.80 | 2023-06-20 10:25:00 | 27665.24 | PARTIAL | 0.50 | 0.20% |
| SELL | retest1 | 2023-06-20 10:20:00 | 27721.80 | 2023-06-20 12:35:00 | 27721.80 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-06-21 10:45:00 | 27701.00 | 2023-06-21 10:50:00 | 27630.20 | PARTIAL | 0.50 | 0.26% |
| SELL | retest1 | 2023-06-21 10:45:00 | 27701.00 | 2023-06-21 15:20:00 | 27250.00 | TARGET_HIT | 0.50 | 1.63% |
| BUY | retest1 | 2023-06-23 09:35:00 | 26874.90 | 2023-06-23 09:45:00 | 26998.77 | PARTIAL | 0.50 | 0.46% |
| BUY | retest1 | 2023-06-23 09:35:00 | 26874.90 | 2023-06-23 11:25:00 | 27100.00 | TARGET_HIT | 0.50 | 0.84% |
| BUY | retest1 | 2023-06-26 11:10:00 | 27234.80 | 2023-06-26 11:40:00 | 27324.83 | PARTIAL | 0.50 | 0.33% |
| BUY | retest1 | 2023-06-26 11:10:00 | 27234.80 | 2023-06-26 15:20:00 | 27463.30 | TARGET_HIT | 0.50 | 0.84% |
| SELL | retest1 | 2023-06-27 09:50:00 | 27446.20 | 2023-06-27 10:00:00 | 27515.27 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2023-06-30 10:55:00 | 27534.50 | 2023-06-30 11:00:00 | 27476.97 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2023-07-03 11:10:00 | 27075.70 | 2023-07-03 11:25:00 | 27129.90 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2023-07-04 11:05:00 | 27174.40 | 2023-07-04 11:50:00 | 27079.30 | PARTIAL | 0.50 | 0.35% |
| SELL | retest1 | 2023-07-04 11:05:00 | 27174.40 | 2023-07-04 15:00:00 | 27174.40 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-07-05 10:55:00 | 27119.10 | 2023-07-05 11:25:00 | 27151.95 | STOP_HIT | 1.00 | -0.12% |
| SELL | retest1 | 2023-07-07 10:55:00 | 27100.00 | 2023-07-07 11:05:00 | 27017.99 | PARTIAL | 0.50 | 0.30% |
| SELL | retest1 | 2023-07-07 10:55:00 | 27100.00 | 2023-07-07 14:45:00 | 27100.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-07-10 10:30:00 | 27225.00 | 2023-07-10 10:40:00 | 27307.47 | PARTIAL | 0.50 | 0.30% |
| BUY | retest1 | 2023-07-10 10:30:00 | 27225.00 | 2023-07-10 15:20:00 | 27865.20 | TARGET_HIT | 0.50 | 2.35% |
| BUY | retest1 | 2023-07-11 10:40:00 | 28090.60 | 2023-07-11 10:55:00 | 28215.87 | PARTIAL | 0.50 | 0.45% |
| BUY | retest1 | 2023-07-11 10:40:00 | 28090.60 | 2023-07-11 11:05:00 | 28090.60 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-07-12 09:40:00 | 28291.50 | 2023-07-12 09:50:00 | 28164.80 | STOP_HIT | 1.00 | -0.45% |
| BUY | retest1 | 2023-07-13 09:30:00 | 28750.00 | 2023-07-13 09:40:00 | 28879.52 | PARTIAL | 0.50 | 0.45% |
| BUY | retest1 | 2023-07-13 09:30:00 | 28750.00 | 2023-07-13 10:05:00 | 28750.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-07-14 10:50:00 | 29099.00 | 2023-07-14 11:15:00 | 29024.89 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2023-07-18 11:00:00 | 28175.00 | 2023-07-18 11:40:00 | 28100.24 | PARTIAL | 0.50 | 0.27% |
| SELL | retest1 | 2023-07-18 11:00:00 | 28175.00 | 2023-07-18 15:20:00 | 27959.70 | TARGET_HIT | 0.50 | 0.76% |
| BUY | retest1 | 2023-07-19 09:45:00 | 28368.00 | 2023-07-19 09:50:00 | 28281.21 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2023-07-28 09:55:00 | 28038.90 | 2023-07-28 10:05:00 | 27957.17 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2023-07-31 10:00:00 | 28220.90 | 2023-07-31 10:05:00 | 28355.90 | PARTIAL | 0.50 | 0.48% |
| BUY | retest1 | 2023-07-31 10:00:00 | 28220.90 | 2023-07-31 15:20:00 | 28937.30 | TARGET_HIT | 0.50 | 2.54% |
| SELL | retest1 | 2023-08-01 10:40:00 | 28565.30 | 2023-08-01 10:55:00 | 28639.77 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2023-08-02 09:45:00 | 28779.20 | 2023-08-02 10:05:00 | 28846.00 | PARTIAL | 0.50 | 0.23% |
| BUY | retest1 | 2023-08-02 09:45:00 | 28779.20 | 2023-08-02 10:30:00 | 28779.20 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-08-08 10:15:00 | 27936.00 | 2023-08-08 10:50:00 | 27822.79 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest1 | 2023-08-09 10:25:00 | 27131.80 | 2023-08-09 10:30:00 | 27207.76 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2023-08-11 09:50:00 | 28158.30 | 2023-08-11 10:30:00 | 28380.87 | PARTIAL | 0.50 | 0.79% |
| BUY | retest1 | 2023-08-11 09:50:00 | 28158.30 | 2023-08-11 15:20:00 | 29444.70 | TARGET_HIT | 0.50 | 4.57% |
| BUY | retest1 | 2023-08-16 09:35:00 | 28737.40 | 2023-08-16 09:40:00 | 28637.73 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2023-08-17 09:45:00 | 29020.00 | 2023-08-17 12:05:00 | 28895.23 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest1 | 2023-08-18 09:50:00 | 29279.30 | 2023-08-18 10:35:00 | 29191.89 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2023-08-24 09:40:00 | 30560.00 | 2023-08-24 09:55:00 | 30473.96 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2023-08-29 11:10:00 | 30694.10 | 2023-08-29 11:30:00 | 30765.95 | PARTIAL | 0.50 | 0.23% |
| BUY | retest1 | 2023-08-29 11:10:00 | 30694.10 | 2023-08-29 12:00:00 | 30694.10 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-09-12 09:35:00 | 32331.80 | 2023-09-12 09:40:00 | 32192.89 | PARTIAL | 0.50 | 0.43% |
| SELL | retest1 | 2023-09-12 09:35:00 | 32331.80 | 2023-09-12 10:20:00 | 32125.20 | TARGET_HIT | 0.50 | 0.64% |
| SELL | retest1 | 2023-09-14 10:50:00 | 31014.10 | 2023-09-14 12:20:00 | 31087.76 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2023-09-15 10:40:00 | 31071.00 | 2023-09-15 10:50:00 | 30979.99 | PARTIAL | 0.50 | 0.29% |
| SELL | retest1 | 2023-09-15 10:40:00 | 31071.00 | 2023-09-15 15:20:00 | 30183.90 | TARGET_HIT | 0.50 | 2.86% |
| BUY | retest1 | 2023-09-25 11:05:00 | 31847.40 | 2023-09-25 13:10:00 | 31748.59 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2023-09-26 11:10:00 | 31070.00 | 2023-09-26 12:00:00 | 30951.27 | PARTIAL | 0.50 | 0.38% |
| SELL | retest1 | 2023-09-26 11:10:00 | 31070.00 | 2023-09-26 12:30:00 | 31070.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-09-28 10:35:00 | 31096.00 | 2023-09-28 10:40:00 | 31141.71 | STOP_HIT | 1.00 | -0.15% |
| BUY | retest1 | 2023-09-29 10:35:00 | 31093.90 | 2023-09-29 11:00:00 | 31224.14 | PARTIAL | 0.50 | 0.42% |
| BUY | retest1 | 2023-09-29 10:35:00 | 31093.90 | 2023-09-29 11:10:00 | 31960.70 | TARGET_HIT | 0.50 | 2.79% |
| SELL | retest1 | 2023-10-03 09:35:00 | 31639.90 | 2023-10-03 14:55:00 | 31383.49 | PARTIAL | 0.50 | 0.81% |
| SELL | retest1 | 2023-10-03 09:35:00 | 31639.90 | 2023-10-03 15:20:00 | 31323.10 | TARGET_HIT | 0.50 | 1.00% |
| SELL | retest1 | 2023-10-04 11:00:00 | 31112.20 | 2023-10-04 11:15:00 | 31017.69 | PARTIAL | 0.50 | 0.30% |
| SELL | retest1 | 2023-10-04 11:00:00 | 31112.20 | 2023-10-04 14:20:00 | 31057.30 | TARGET_HIT | 0.50 | 0.18% |
| BUY | retest1 | 2023-10-06 09:35:00 | 31264.10 | 2023-10-06 09:45:00 | 31193.49 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2023-10-11 09:35:00 | 31262.10 | 2023-10-11 09:45:00 | 31371.74 | PARTIAL | 0.50 | 0.35% |
| BUY | retest1 | 2023-10-11 09:35:00 | 31262.10 | 2023-10-11 10:20:00 | 31262.10 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-10-12 10:45:00 | 30900.10 | 2023-10-12 12:30:00 | 30991.24 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2023-10-13 10:50:00 | 30823.20 | 2023-10-13 10:55:00 | 30874.36 | STOP_HIT | 1.00 | -0.17% |
| SELL | retest1 | 2023-10-18 10:45:00 | 30275.80 | 2023-10-18 11:05:00 | 30170.89 | PARTIAL | 0.50 | 0.35% |
| SELL | retest1 | 2023-10-18 10:45:00 | 30275.80 | 2023-10-18 12:55:00 | 30275.80 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-10-19 10:30:00 | 30443.30 | 2023-10-19 10:35:00 | 30542.28 | PARTIAL | 0.50 | 0.33% |
| BUY | retest1 | 2023-10-19 10:30:00 | 30443.30 | 2023-10-19 12:20:00 | 30443.30 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-10-20 09:55:00 | 30408.20 | 2023-10-20 10:40:00 | 30296.68 | PARTIAL | 0.50 | 0.37% |
| SELL | retest1 | 2023-10-20 09:55:00 | 30408.20 | 2023-10-20 15:20:00 | 30066.80 | TARGET_HIT | 0.50 | 1.12% |
| SELL | retest1 | 2023-10-23 10:10:00 | 29820.00 | 2023-10-23 10:15:00 | 29894.28 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2023-10-25 09:30:00 | 29450.20 | 2023-10-25 09:40:00 | 29583.93 | STOP_HIT | 1.00 | -0.45% |
| SELL | retest1 | 2023-10-26 09:55:00 | 28968.20 | 2023-10-26 10:25:00 | 29067.73 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2023-10-31 09:55:00 | 29591.40 | 2023-10-31 10:05:00 | 29655.46 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2023-11-01 11:15:00 | 29898.90 | 2023-11-01 11:25:00 | 30002.63 | PARTIAL | 0.50 | 0.35% |
| BUY | retest1 | 2023-11-01 11:15:00 | 29898.90 | 2023-11-01 11:35:00 | 29898.90 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-11-02 10:50:00 | 30600.00 | 2023-11-02 11:05:00 | 30692.47 | PARTIAL | 0.50 | 0.30% |
| BUY | retest1 | 2023-11-02 10:50:00 | 30600.00 | 2023-11-02 15:20:00 | 31398.90 | TARGET_HIT | 0.50 | 2.61% |
| BUY | retest1 | 2023-11-06 09:50:00 | 31661.90 | 2023-11-06 10:00:00 | 31579.10 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2023-11-07 09:50:00 | 31207.10 | 2023-11-07 10:05:00 | 31269.08 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2023-11-08 09:40:00 | 31511.80 | 2023-11-08 09:50:00 | 31416.84 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2023-11-17 11:15:00 | 30749.90 | 2023-11-17 11:25:00 | 30794.21 | STOP_HIT | 1.00 | -0.14% |
| BUY | retest1 | 2023-11-20 10:45:00 | 30784.70 | 2023-11-20 11:15:00 | 30911.78 | PARTIAL | 0.50 | 0.41% |
| BUY | retest1 | 2023-11-20 10:45:00 | 30784.70 | 2023-11-20 11:45:00 | 30784.70 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-11-22 09:50:00 | 30814.90 | 2023-11-22 11:25:00 | 30680.88 | PARTIAL | 0.50 | 0.43% |
| SELL | retest1 | 2023-11-22 09:50:00 | 30814.90 | 2023-11-22 13:05:00 | 30814.90 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-11-23 09:30:00 | 30703.10 | 2023-11-23 10:00:00 | 30797.88 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2023-11-28 10:40:00 | 30486.60 | 2023-11-28 11:30:00 | 30567.16 | PARTIAL | 0.50 | 0.26% |
| BUY | retest1 | 2023-11-28 10:40:00 | 30486.60 | 2023-11-28 14:10:00 | 30486.60 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-12-01 10:15:00 | 31444.90 | 2023-12-01 10:20:00 | 31350.97 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2023-12-05 11:05:00 | 31166.40 | 2023-12-05 11:45:00 | 31217.97 | STOP_HIT | 1.00 | -0.17% |
| BUY | retest1 | 2023-12-07 10:55:00 | 31188.40 | 2023-12-07 11:00:00 | 31130.71 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest1 | 2023-12-08 11:05:00 | 31274.20 | 2023-12-08 11:25:00 | 31333.36 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest1 | 2023-12-11 10:10:00 | 31324.50 | 2023-12-11 11:00:00 | 31384.49 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest1 | 2023-12-12 11:00:00 | 31499.90 | 2023-12-12 11:25:00 | 31412.07 | PARTIAL | 0.50 | 0.28% |
| SELL | retest1 | 2023-12-12 11:00:00 | 31499.90 | 2023-12-12 15:20:00 | 31237.20 | TARGET_HIT | 0.50 | 0.83% |
| SELL | retest1 | 2023-12-14 09:55:00 | 31200.00 | 2023-12-14 12:05:00 | 31045.84 | PARTIAL | 0.50 | 0.49% |
| SELL | retest1 | 2023-12-14 09:55:00 | 31200.00 | 2023-12-14 15:00:00 | 31200.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-12-15 09:35:00 | 31097.30 | 2023-12-15 15:00:00 | 31179.93 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2023-12-19 09:40:00 | 31396.00 | 2023-12-19 09:45:00 | 31292.68 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2023-12-20 10:30:00 | 31298.60 | 2023-12-20 11:55:00 | 31364.02 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2023-12-22 10:00:00 | 31351.00 | 2023-12-22 10:30:00 | 31213.72 | PARTIAL | 0.50 | 0.44% |
| SELL | retest1 | 2023-12-22 10:00:00 | 31351.00 | 2023-12-22 14:50:00 | 31233.70 | TARGET_HIT | 0.50 | 0.37% |
| BUY | retest1 | 2023-12-26 09:40:00 | 31419.90 | 2023-12-26 09:50:00 | 31341.54 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2024-01-02 09:30:00 | 36454.90 | 2024-01-02 09:55:00 | 36224.71 | PARTIAL | 0.50 | 0.63% |
| SELL | retest1 | 2024-01-02 09:30:00 | 36454.90 | 2024-01-02 15:20:00 | 35604.20 | TARGET_HIT | 0.50 | 2.33% |
| BUY | retest1 | 2024-01-09 10:05:00 | 34748.00 | 2024-01-09 10:15:00 | 34584.46 | STOP_HIT | 1.00 | -0.47% |
| SELL | retest1 | 2024-01-15 09:40:00 | 34000.00 | 2024-01-15 09:50:00 | 34075.86 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2024-01-16 10:50:00 | 34588.20 | 2024-01-16 11:40:00 | 34509.73 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2024-01-18 09:35:00 | 33942.50 | 2024-01-18 09:40:00 | 34062.17 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2024-01-19 11:05:00 | 33980.00 | 2024-01-19 12:45:00 | 34072.43 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2024-01-29 10:05:00 | 33478.10 | 2024-01-29 10:15:00 | 33593.96 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2024-01-30 09:55:00 | 33972.80 | 2024-01-30 10:30:00 | 33896.78 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2024-02-01 10:20:00 | 33910.00 | 2024-02-01 15:20:00 | 33805.10 | TARGET_HIT | 1.00 | 0.31% |
| SELL | retest1 | 2024-02-02 09:30:00 | 33670.30 | 2024-02-02 09:40:00 | 33536.12 | PARTIAL | 0.50 | 0.40% |
| SELL | retest1 | 2024-02-02 09:30:00 | 33670.30 | 2024-02-02 15:20:00 | 32890.00 | TARGET_HIT | 0.50 | 2.32% |
| BUY | retest1 | 2024-02-08 09:50:00 | 34299.90 | 2024-02-08 10:05:00 | 34225.81 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2024-02-12 10:50:00 | 32137.80 | 2024-02-12 11:50:00 | 31963.51 | PARTIAL | 0.50 | 0.54% |
| SELL | retest1 | 2024-02-12 10:50:00 | 32137.80 | 2024-02-12 15:20:00 | 31780.10 | TARGET_HIT | 0.50 | 1.11% |
| SELL | retest1 | 2024-02-14 10:55:00 | 31168.20 | 2024-02-14 11:00:00 | 31228.59 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest1 | 2024-02-15 11:10:00 | 30709.20 | 2024-02-15 11:20:00 | 30590.63 | PARTIAL | 0.50 | 0.39% |
| SELL | retest1 | 2024-02-15 11:10:00 | 30709.20 | 2024-02-15 11:45:00 | 30709.20 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-02-21 10:50:00 | 31150.00 | 2024-02-21 11:05:00 | 31234.33 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2024-02-26 11:10:00 | 30620.40 | 2024-02-26 11:40:00 | 30699.29 | PARTIAL | 0.50 | 0.26% |
| BUY | retest1 | 2024-02-26 11:10:00 | 30620.40 | 2024-02-26 15:15:00 | 30664.75 | TARGET_HIT | 0.50 | 0.14% |
| BUY | retest1 | 2024-02-27 10:50:00 | 31204.75 | 2024-02-27 11:35:00 | 31148.34 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest1 | 2024-03-01 10:35:00 | 31770.35 | 2024-03-01 11:10:00 | 31884.28 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2024-03-05 09:45:00 | 30551.00 | 2024-03-05 10:45:00 | 30396.67 | PARTIAL | 0.50 | 0.51% |
| SELL | retest1 | 2024-03-05 09:45:00 | 30551.00 | 2024-03-05 11:30:00 | 30521.45 | TARGET_HIT | 0.50 | 0.10% |
| SELL | retest1 | 2024-03-15 10:10:00 | 29705.05 | 2024-03-15 10:15:00 | 29509.12 | PARTIAL | 0.50 | 0.66% |
| SELL | retest1 | 2024-03-15 10:10:00 | 29705.05 | 2024-03-15 10:20:00 | 29705.05 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-03-26 10:25:00 | 30542.45 | 2024-03-26 11:00:00 | 30409.28 | PARTIAL | 0.50 | 0.44% |
| SELL | retest1 | 2024-03-26 10:25:00 | 30542.45 | 2024-03-26 15:20:00 | 30484.55 | TARGET_HIT | 0.50 | 0.19% |
| SELL | retest1 | 2024-03-27 10:30:00 | 30374.35 | 2024-03-27 10:50:00 | 30265.19 | PARTIAL | 0.50 | 0.36% |
| SELL | retest1 | 2024-03-27 10:30:00 | 30374.35 | 2024-03-27 10:55:00 | 30374.35 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-04-08 11:15:00 | 30290.00 | 2024-04-08 11:45:00 | 30214.82 | PARTIAL | 0.50 | 0.25% |
| SELL | retest1 | 2024-04-08 11:15:00 | 30290.00 | 2024-04-08 15:20:00 | 29998.35 | TARGET_HIT | 0.50 | 0.96% |
| SELL | retest1 | 2024-04-10 10:25:00 | 29959.95 | 2024-04-10 10:40:00 | 30022.22 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2024-04-12 10:40:00 | 29817.00 | 2024-04-12 11:35:00 | 29874.04 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest1 | 2024-04-18 09:40:00 | 29670.00 | 2024-04-18 09:50:00 | 29745.43 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2024-04-22 11:10:00 | 30078.85 | 2024-04-22 11:25:00 | 30013.17 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2024-04-24 09:40:00 | 30288.95 | 2024-04-24 09:45:00 | 30210.94 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2024-04-25 10:40:00 | 30000.05 | 2024-04-25 14:35:00 | 29884.53 | PARTIAL | 0.50 | 0.39% |
| SELL | retest1 | 2024-04-25 10:40:00 | 30000.05 | 2024-04-25 15:20:00 | 29986.95 | TARGET_HIT | 0.50 | 0.04% |
| SELL | retest1 | 2024-04-26 10:00:00 | 29972.20 | 2024-04-26 11:40:00 | 30042.76 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2024-04-29 10:25:00 | 29862.75 | 2024-04-29 10:40:00 | 29952.73 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2024-04-30 09:40:00 | 30251.25 | 2024-04-30 10:10:00 | 30452.11 | PARTIAL | 0.50 | 0.66% |
| BUY | retest1 | 2024-04-30 09:40:00 | 30251.25 | 2024-04-30 15:20:00 | 30480.15 | TARGET_HIT | 0.50 | 0.76% |
| SELL | retest1 | 2024-05-07 10:35:00 | 29345.00 | 2024-05-07 11:00:00 | 29415.63 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2024-05-08 09:30:00 | 29389.00 | 2024-05-08 10:25:00 | 29183.00 | PARTIAL | 0.50 | 0.70% |
| SELL | retest1 | 2024-05-08 09:30:00 | 29389.00 | 2024-05-08 15:20:00 | 29190.00 | TARGET_HIT | 0.50 | 0.68% |
| SELL | retest1 | 2024-05-09 09:55:00 | 29040.95 | 2024-05-09 10:20:00 | 28946.62 | PARTIAL | 0.50 | 0.32% |
| SELL | retest1 | 2024-05-09 09:55:00 | 29040.95 | 2024-05-09 15:20:00 | 28769.85 | TARGET_HIT | 0.50 | 0.93% |
