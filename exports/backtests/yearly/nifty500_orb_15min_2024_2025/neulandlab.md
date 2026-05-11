# Neuland Laboratories Ltd. (NEULANDLAB)

## Backtest Summary

- **Window:** 2024-05-13 09:15:00 → 2026-05-08 15:25:00 (36871 bars)
- **Last close:** 17713.00
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
| ENTRY1 | 34 |
| ENTRY2 | 0 |
| PARTIAL | 12 |
| TARGET_HIT | 7 |
| STOP_HIT | 27 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 46 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 19 / 27
- **Target hits / Stop hits / Partials:** 7 / 27 / 12
- **Avg / median % per leg:** 0.21% / 0.00%
- **Sum % (uncompounded):** 9.81%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 25 | 12 | 48.0% | 4 | 13 | 8 | 0.30% | 7.4% |
| BUY @ 2nd Alert (retest1) | 25 | 12 | 48.0% | 4 | 13 | 8 | 0.30% | 7.4% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 21 | 7 | 33.3% | 3 | 14 | 4 | 0.12% | 2.4% |
| SELL @ 2nd Alert (retest1) | 21 | 7 | 33.3% | 3 | 14 | 4 | 0.12% | 2.4% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 46 | 19 | 41.3% | 7 | 27 | 12 | 0.21% | 9.8% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-05-16 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-16 09:55:00 | 6228.40 | 6259.86 | 0.00 | ORB-short ORB[6245.00,6297.90] vol=1.8x ATR=20.95 |
| Stop hit — per-position SL triggered | 2024-05-16 12:40:00 | 6249.35 | 6237.56 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2024-05-22 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-22 09:40:00 | 6198.00 | 6214.84 | 0.00 | ORB-short ORB[6200.00,6279.90] vol=2.5x ATR=19.59 |
| Stop hit — per-position SL triggered | 2024-05-22 09:55:00 | 6217.59 | 6212.80 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2024-05-23 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-23 10:35:00 | 6209.55 | 6225.81 | 0.00 | ORB-short ORB[6215.00,6299.90] vol=1.8x ATR=13.22 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-23 11:05:00 | 6189.72 | 6214.73 | 0.00 | T1 1.5R @ 6189.72 |
| Stop hit — per-position SL triggered | 2024-05-23 12:30:00 | 6209.55 | 6207.23 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2024-05-27 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-27 09:35:00 | 6420.00 | 6392.85 | 0.00 | ORB-long ORB[6318.70,6407.65] vol=1.6x ATR=32.88 |
| Stop hit — per-position SL triggered | 2024-05-27 09:45:00 | 6387.12 | 6392.99 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2024-05-30 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-30 09:45:00 | 5979.05 | 6038.06 | 0.00 | ORB-short ORB[6063.00,6099.90] vol=2.8x ATR=26.99 |
| Stop hit — per-position SL triggered | 2024-05-30 09:50:00 | 6006.04 | 6029.57 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2024-05-31 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-31 10:35:00 | 6020.00 | 6096.65 | 0.00 | ORB-short ORB[6110.05,6182.05] vol=2.2x ATR=26.81 |
| Stop hit — per-position SL triggered | 2024-05-31 10:45:00 | 6046.81 | 6091.92 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2024-06-06 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-06 10:05:00 | 6422.00 | 6367.11 | 0.00 | ORB-long ORB[6291.05,6380.00] vol=1.7x ATR=35.53 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-06 10:35:00 | 6475.29 | 6387.86 | 0.00 | T1 1.5R @ 6475.29 |
| Stop hit — per-position SL triggered | 2024-06-06 10:40:00 | 6422.00 | 6388.79 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2024-06-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-11 10:15:00 | 6350.40 | 6314.43 | 0.00 | ORB-long ORB[6280.00,6316.85] vol=2.3x ATR=15.51 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-11 11:40:00 | 6373.67 | 6336.19 | 0.00 | T1 1.5R @ 6373.67 |
| Stop hit — per-position SL triggered | 2024-06-11 11:50:00 | 6350.40 | 6338.92 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2024-06-14 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-14 10:45:00 | 6510.05 | 6529.73 | 0.00 | ORB-short ORB[6525.00,6618.30] vol=9.5x ATR=18.67 |
| Stop hit — per-position SL triggered | 2024-06-14 11:20:00 | 6528.72 | 6529.62 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2024-06-21 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-21 09:35:00 | 6516.90 | 6544.72 | 0.00 | ORB-short ORB[6540.00,6585.85] vol=2.2x ATR=23.86 |
| Stop hit — per-position SL triggered | 2024-06-21 09:45:00 | 6540.76 | 6533.25 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2024-06-24 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-24 09:35:00 | 6711.00 | 6639.00 | 0.00 | ORB-long ORB[6551.00,6636.00] vol=4.1x ATR=39.56 |
| Stop hit — per-position SL triggered | 2024-06-24 09:45:00 | 6671.44 | 6667.22 | 0.00 | SL hit |

### Cycle 12 — SELL (started 2024-07-09 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-09 09:30:00 | 7754.10 | 7785.37 | 0.00 | ORB-short ORB[7764.85,7824.55] vol=1.7x ATR=30.74 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-09 09:50:00 | 7708.00 | 7768.18 | 0.00 | T1 1.5R @ 7708.00 |
| Target hit | 2024-07-09 13:25:00 | 7725.00 | 7712.17 | 0.00 | Trail-exit close>VWAP |

### Cycle 13 — SELL (started 2024-07-12 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-12 10:55:00 | 8050.80 | 8088.79 | 0.00 | ORB-short ORB[8065.00,8152.75] vol=3.1x ATR=16.71 |
| Stop hit — per-position SL triggered | 2024-07-12 11:10:00 | 8067.51 | 8082.58 | 0.00 | SL hit |

### Cycle 14 — SELL (started 2024-08-13 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-13 09:55:00 | 11523.65 | 11583.02 | 0.00 | ORB-short ORB[11571.00,11699.00] vol=2.6x ATR=46.79 |
| Stop hit — per-position SL triggered | 2024-08-13 10:05:00 | 11570.44 | 11581.93 | 0.00 | SL hit |

### Cycle 15 — BUY (started 2024-08-26 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-26 09:30:00 | 12429.00 | 12364.94 | 0.00 | ORB-long ORB[12249.00,12417.25] vol=1.7x ATR=71.35 |
| Stop hit — per-position SL triggered | 2024-08-26 10:10:00 | 12357.65 | 12403.48 | 0.00 | SL hit |

### Cycle 16 — BUY (started 2024-09-03 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-03 09:45:00 | 12749.95 | 12651.00 | 0.00 | ORB-long ORB[12476.20,12659.95] vol=5.4x ATR=58.47 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-03 10:20:00 | 12837.66 | 12710.94 | 0.00 | T1 1.5R @ 12837.66 |
| Target hit | 2024-09-03 10:55:00 | 12764.60 | 12765.27 | 0.00 | Trail-exit close<VWAP |

### Cycle 17 — BUY (started 2024-09-04 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-04 09:50:00 | 12859.25 | 12800.87 | 0.00 | ORB-long ORB[12679.20,12848.40] vol=2.9x ATR=41.38 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-04 10:15:00 | 12921.32 | 12821.89 | 0.00 | T1 1.5R @ 12921.32 |
| Target hit | 2024-09-04 10:55:00 | 12881.00 | 12908.10 | 0.00 | Trail-exit close<VWAP |

### Cycle 18 — BUY (started 2024-09-12 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-12 10:35:00 | 12800.00 | 12703.81 | 0.00 | ORB-long ORB[12627.25,12749.00] vol=3.0x ATR=36.00 |
| Stop hit — per-position SL triggered | 2024-09-12 10:45:00 | 12764.00 | 12707.87 | 0.00 | SL hit |

### Cycle 19 — SELL (started 2024-09-13 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-13 10:55:00 | 12587.35 | 12681.44 | 0.00 | ORB-short ORB[12701.05,12828.85] vol=5.1x ATR=35.95 |
| Stop hit — per-position SL triggered | 2024-09-13 11:05:00 | 12623.30 | 12677.29 | 0.00 | SL hit |

### Cycle 20 — BUY (started 2024-09-20 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-20 10:05:00 | 12749.95 | 12665.52 | 0.00 | ORB-long ORB[12565.05,12693.40] vol=2.6x ATR=59.80 |
| Stop hit — per-position SL triggered | 2024-09-20 11:00:00 | 12690.15 | 12684.10 | 0.00 | SL hit |

### Cycle 21 — SELL (started 2024-09-23 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-23 10:10:00 | 12700.00 | 12751.72 | 0.00 | ORB-short ORB[12767.60,12849.00] vol=2.1x ATR=45.10 |
| Stop hit — per-position SL triggered | 2024-09-23 11:25:00 | 12745.10 | 12740.03 | 0.00 | SL hit |

### Cycle 22 — BUY (started 2024-10-31 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-31 10:25:00 | 14072.55 | 13901.68 | 0.00 | ORB-long ORB[13752.90,13931.95] vol=1.7x ATR=60.83 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-31 11:05:00 | 14163.80 | 13973.28 | 0.00 | T1 1.5R @ 14163.80 |
| Target hit | 2024-10-31 15:20:00 | 14769.10 | 14417.75 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 23 — BUY (started 2024-11-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-27 10:15:00 | 15150.00 | 15060.54 | 0.00 | ORB-long ORB[14890.25,15100.00] vol=7.0x ATR=76.43 |
| Stop hit — per-position SL triggered | 2024-11-27 10:20:00 | 15073.57 | 15063.95 | 0.00 | SL hit |

### Cycle 24 — BUY (started 2024-11-28 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-28 09:50:00 | 16403.35 | 16301.85 | 0.00 | ORB-long ORB[15940.00,16186.55] vol=1.6x ATR=140.51 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-28 10:30:00 | 16614.11 | 16390.75 | 0.00 | T1 1.5R @ 16614.11 |
| Stop hit — per-position SL triggered | 2024-11-28 10:35:00 | 16403.35 | 16385.68 | 0.00 | SL hit |

### Cycle 25 — BUY (started 2024-12-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-02 10:15:00 | 17034.65 | 16809.54 | 0.00 | ORB-long ORB[16501.05,16754.00] vol=3.9x ATR=108.81 |
| Stop hit — per-position SL triggered | 2024-12-02 11:20:00 | 16925.84 | 16893.61 | 0.00 | SL hit |

### Cycle 26 — SELL (started 2024-12-26 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-26 10:40:00 | 13669.00 | 13789.52 | 0.00 | ORB-short ORB[13850.25,13989.95] vol=1.7x ATR=50.99 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-26 12:15:00 | 13592.51 | 13738.52 | 0.00 | T1 1.5R @ 13592.51 |
| Target hit | 2024-12-26 15:20:00 | 13450.00 | 13553.06 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 27 — BUY (started 2024-12-30 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-30 10:50:00 | 13824.65 | 13781.77 | 0.00 | ORB-long ORB[13700.00,13799.50] vol=1.6x ATR=58.18 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-30 12:15:00 | 13911.92 | 13809.63 | 0.00 | T1 1.5R @ 13911.92 |
| Stop hit — per-position SL triggered | 2024-12-30 13:20:00 | 13824.65 | 13851.55 | 0.00 | SL hit |

### Cycle 28 — SELL (started 2025-01-02 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-02 10:50:00 | 14031.75 | 14213.95 | 0.00 | ORB-short ORB[14261.90,14437.00] vol=2.7x ATR=52.30 |
| Stop hit — per-position SL triggered | 2025-01-02 11:00:00 | 14084.05 | 14203.09 | 0.00 | SL hit |

### Cycle 29 — SELL (started 2025-01-03 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-03 09:30:00 | 14230.00 | 14303.10 | 0.00 | ORB-short ORB[14267.70,14398.00] vol=1.6x ATR=49.34 |
| Stop hit — per-position SL triggered | 2025-01-03 09:40:00 | 14279.34 | 14289.48 | 0.00 | SL hit |

### Cycle 30 — BUY (started 2025-01-29 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-29 10:35:00 | 13582.40 | 13438.89 | 0.00 | ORB-long ORB[13272.00,13440.00] vol=2.5x ATR=51.00 |
| Stop hit — per-position SL triggered | 2025-01-29 11:15:00 | 13531.40 | 13468.78 | 0.00 | SL hit |

### Cycle 31 — BUY (started 2025-01-31 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-31 09:40:00 | 14016.60 | 13911.44 | 0.00 | ORB-long ORB[13711.25,13885.00] vol=1.8x ATR=81.24 |
| Stop hit — per-position SL triggered | 2025-01-31 10:05:00 | 13935.36 | 13958.94 | 0.00 | SL hit |

### Cycle 32 — SELL (started 2025-02-01 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-01 11:00:00 | 13992.50 | 14111.20 | 0.00 | ORB-short ORB[14124.95,14217.00] vol=1.8x ATR=28.08 |
| Stop hit — per-position SL triggered | 2025-02-01 11:05:00 | 14020.58 | 14107.29 | 0.00 | SL hit |

### Cycle 33 — BUY (started 2025-04-21 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-21 09:35:00 | 13013.00 | 12878.99 | 0.00 | ORB-long ORB[12794.00,12940.00] vol=2.3x ATR=67.20 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-21 11:30:00 | 13113.80 | 12952.79 | 0.00 | T1 1.5R @ 13113.80 |
| Target hit | 2025-04-21 15:20:00 | 13140.00 | 13095.87 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 34 — SELL (started 2025-04-29 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-29 10:25:00 | 12600.00 | 12698.98 | 0.00 | ORB-short ORB[12668.00,12810.00] vol=1.8x ATR=54.21 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-29 12:00:00 | 12518.69 | 12669.55 | 0.00 | T1 1.5R @ 12518.69 |
| Target hit | 2025-04-29 15:20:00 | 12260.00 | 12462.54 | 0.00 | EOD square-off @ 15:20:00 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2024-05-16 09:55:00 | 6228.40 | 2024-05-16 12:40:00 | 6249.35 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2024-05-22 09:40:00 | 6198.00 | 2024-05-22 09:55:00 | 6217.59 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2024-05-23 10:35:00 | 6209.55 | 2024-05-23 11:05:00 | 6189.72 | PARTIAL | 0.50 | 0.32% |
| SELL | retest1 | 2024-05-23 10:35:00 | 6209.55 | 2024-05-23 12:30:00 | 6209.55 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-05-27 09:35:00 | 6420.00 | 2024-05-27 09:45:00 | 6387.12 | STOP_HIT | 1.00 | -0.51% |
| SELL | retest1 | 2024-05-30 09:45:00 | 5979.05 | 2024-05-30 09:50:00 | 6006.04 | STOP_HIT | 1.00 | -0.45% |
| SELL | retest1 | 2024-05-31 10:35:00 | 6020.00 | 2024-05-31 10:45:00 | 6046.81 | STOP_HIT | 1.00 | -0.45% |
| BUY | retest1 | 2024-06-06 10:05:00 | 6422.00 | 2024-06-06 10:35:00 | 6475.29 | PARTIAL | 0.50 | 0.83% |
| BUY | retest1 | 2024-06-06 10:05:00 | 6422.00 | 2024-06-06 10:40:00 | 6422.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-06-11 10:15:00 | 6350.40 | 2024-06-11 11:40:00 | 6373.67 | PARTIAL | 0.50 | 0.37% |
| BUY | retest1 | 2024-06-11 10:15:00 | 6350.40 | 2024-06-11 11:50:00 | 6350.40 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-06-14 10:45:00 | 6510.05 | 2024-06-14 11:20:00 | 6528.72 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2024-06-21 09:35:00 | 6516.90 | 2024-06-21 09:45:00 | 6540.76 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2024-06-24 09:35:00 | 6711.00 | 2024-06-24 09:45:00 | 6671.44 | STOP_HIT | 1.00 | -0.59% |
| SELL | retest1 | 2024-07-09 09:30:00 | 7754.10 | 2024-07-09 09:50:00 | 7708.00 | PARTIAL | 0.50 | 0.59% |
| SELL | retest1 | 2024-07-09 09:30:00 | 7754.10 | 2024-07-09 13:25:00 | 7725.00 | TARGET_HIT | 0.50 | 0.38% |
| SELL | retest1 | 2024-07-12 10:55:00 | 8050.80 | 2024-07-12 11:10:00 | 8067.51 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2024-08-13 09:55:00 | 11523.65 | 2024-08-13 10:05:00 | 11570.44 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest1 | 2024-08-26 09:30:00 | 12429.00 | 2024-08-26 10:10:00 | 12357.65 | STOP_HIT | 1.00 | -0.57% |
| BUY | retest1 | 2024-09-03 09:45:00 | 12749.95 | 2024-09-03 10:20:00 | 12837.66 | PARTIAL | 0.50 | 0.69% |
| BUY | retest1 | 2024-09-03 09:45:00 | 12749.95 | 2024-09-03 10:55:00 | 12764.60 | TARGET_HIT | 0.50 | 0.11% |
| BUY | retest1 | 2024-09-04 09:50:00 | 12859.25 | 2024-09-04 10:15:00 | 12921.32 | PARTIAL | 0.50 | 0.48% |
| BUY | retest1 | 2024-09-04 09:50:00 | 12859.25 | 2024-09-04 10:55:00 | 12881.00 | TARGET_HIT | 0.50 | 0.17% |
| BUY | retest1 | 2024-09-12 10:35:00 | 12800.00 | 2024-09-12 10:45:00 | 12764.00 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2024-09-13 10:55:00 | 12587.35 | 2024-09-13 11:05:00 | 12623.30 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2024-09-20 10:05:00 | 12749.95 | 2024-09-20 11:00:00 | 12690.15 | STOP_HIT | 1.00 | -0.47% |
| SELL | retest1 | 2024-09-23 10:10:00 | 12700.00 | 2024-09-23 11:25:00 | 12745.10 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2024-10-31 10:25:00 | 14072.55 | 2024-10-31 11:05:00 | 14163.80 | PARTIAL | 0.50 | 0.65% |
| BUY | retest1 | 2024-10-31 10:25:00 | 14072.55 | 2024-10-31 15:20:00 | 14769.10 | TARGET_HIT | 0.50 | 4.95% |
| BUY | retest1 | 2024-11-27 10:15:00 | 15150.00 | 2024-11-27 10:20:00 | 15073.57 | STOP_HIT | 1.00 | -0.50% |
| BUY | retest1 | 2024-11-28 09:50:00 | 16403.35 | 2024-11-28 10:30:00 | 16614.11 | PARTIAL | 0.50 | 1.28% |
| BUY | retest1 | 2024-11-28 09:50:00 | 16403.35 | 2024-11-28 10:35:00 | 16403.35 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-12-02 10:15:00 | 17034.65 | 2024-12-02 11:20:00 | 16925.84 | STOP_HIT | 1.00 | -0.64% |
| SELL | retest1 | 2024-12-26 10:40:00 | 13669.00 | 2024-12-26 12:15:00 | 13592.51 | PARTIAL | 0.50 | 0.56% |
| SELL | retest1 | 2024-12-26 10:40:00 | 13669.00 | 2024-12-26 15:20:00 | 13450.00 | TARGET_HIT | 0.50 | 1.60% |
| BUY | retest1 | 2024-12-30 10:50:00 | 13824.65 | 2024-12-30 12:15:00 | 13911.92 | PARTIAL | 0.50 | 0.63% |
| BUY | retest1 | 2024-12-30 10:50:00 | 13824.65 | 2024-12-30 13:20:00 | 13824.65 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-01-02 10:50:00 | 14031.75 | 2025-01-02 11:00:00 | 14084.05 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest1 | 2025-01-03 09:30:00 | 14230.00 | 2025-01-03 09:40:00 | 14279.34 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2025-01-29 10:35:00 | 13582.40 | 2025-01-29 11:15:00 | 13531.40 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2025-01-31 09:40:00 | 14016.60 | 2025-01-31 10:05:00 | 13935.36 | STOP_HIT | 1.00 | -0.58% |
| SELL | retest1 | 2025-02-01 11:00:00 | 13992.50 | 2025-02-01 11:05:00 | 14020.58 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2025-04-21 09:35:00 | 13013.00 | 2025-04-21 11:30:00 | 13113.80 | PARTIAL | 0.50 | 0.77% |
| BUY | retest1 | 2025-04-21 09:35:00 | 13013.00 | 2025-04-21 15:20:00 | 13140.00 | TARGET_HIT | 0.50 | 0.98% |
| SELL | retest1 | 2025-04-29 10:25:00 | 12600.00 | 2025-04-29 12:00:00 | 12518.69 | PARTIAL | 0.50 | 0.65% |
| SELL | retest1 | 2025-04-29 10:25:00 | 12600.00 | 2025-04-29 15:20:00 | 12260.00 | TARGET_HIT | 0.50 | 2.70% |
