# Abbott India Ltd. (ABBOTINDIA)

## Backtest Summary

- **Window:** 2024-05-13 09:15:00 → 2026-05-08 15:25:00 (36871 bars)
- **Last close:** 26850.00
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
| ENTRY1 | 123 |
| ENTRY2 | 0 |
| PARTIAL | 47 |
| TARGET_HIT | 21 |
| STOP_HIT | 102 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 170 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 68 / 102
- **Target hits / Stop hits / Partials:** 21 / 102 / 47
- **Avg / median % per leg:** 0.10% / 0.00%
- **Sum % (uncompounded):** 16.32%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 83 | 28 | 33.7% | 7 | 55 | 21 | 0.04% | 3.7% |
| BUY @ 2nd Alert (retest1) | 83 | 28 | 33.7% | 7 | 55 | 21 | 0.04% | 3.7% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 87 | 40 | 46.0% | 14 | 47 | 26 | 0.15% | 12.7% |
| SELL @ 2nd Alert (retest1) | 87 | 40 | 46.0% | 14 | 47 | 26 | 0.15% | 12.7% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 170 | 68 | 40.0% | 21 | 102 | 47 | 0.10% | 16.3% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-05-14 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-14 10:20:00 | 27156.85 | 27023.47 | 0.00 | ORB-long ORB[26860.00,27149.20] vol=1.5x ATR=77.55 |
| Stop hit — per-position SL triggered | 2024-05-14 10:30:00 | 27079.30 | 27028.48 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2024-05-15 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-15 10:35:00 | 26584.00 | 26679.22 | 0.00 | ORB-short ORB[26596.50,26862.55] vol=1.8x ATR=53.77 |
| Stop hit — per-position SL triggered | 2024-05-15 10:40:00 | 26637.77 | 26678.45 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2024-05-16 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-16 10:30:00 | 26372.00 | 26413.61 | 0.00 | ORB-short ORB[26412.85,26624.90] vol=3.3x ATR=50.61 |
| Stop hit — per-position SL triggered | 2024-05-16 11:50:00 | 26422.61 | 26401.23 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2024-05-21 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-21 09:30:00 | 26417.15 | 26512.89 | 0.00 | ORB-short ORB[26425.90,26755.00] vol=3.2x ATR=79.81 |
| Stop hit — per-position SL triggered | 2024-05-21 15:20:00 | 26534.95 | 26433.76 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 5 — SELL (started 2024-05-22 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-22 09:40:00 | 26286.90 | 26338.02 | 0.00 | ORB-short ORB[26300.00,26450.00] vol=5.1x ATR=60.43 |
| Stop hit — per-position SL triggered | 2024-05-22 09:50:00 | 26347.33 | 26338.18 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2024-05-23 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-23 09:35:00 | 25980.00 | 26072.48 | 0.00 | ORB-short ORB[26052.70,26278.55] vol=2.7x ATR=51.83 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-23 09:55:00 | 25902.25 | 26021.00 | 0.00 | T1 1.5R @ 25902.25 |
| Target hit | 2024-05-23 10:50:00 | 25940.00 | 25912.27 | 0.00 | Trail-exit close>VWAP |

### Cycle 7 — BUY (started 2024-05-24 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-24 09:35:00 | 26286.05 | 26138.15 | 0.00 | ORB-long ORB[26000.00,26163.10] vol=4.8x ATR=65.90 |
| Stop hit — per-position SL triggered | 2024-05-24 09:50:00 | 26220.15 | 26176.11 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2024-05-29 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-29 10:00:00 | 26005.00 | 25852.33 | 0.00 | ORB-long ORB[25717.90,25984.00] vol=1.6x ATR=63.93 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-29 10:15:00 | 26100.89 | 25909.57 | 0.00 | T1 1.5R @ 26100.89 |
| Target hit | 2024-05-29 15:20:00 | 26163.85 | 26166.58 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 9 — SELL (started 2024-05-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-30 11:15:00 | 25906.80 | 25965.55 | 0.00 | ORB-short ORB[25921.40,26200.10] vol=1.6x ATR=48.87 |
| Stop hit — per-position SL triggered | 2024-05-30 11:40:00 | 25955.67 | 25964.03 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2024-05-31 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-31 11:00:00 | 25785.75 | 25873.56 | 0.00 | ORB-short ORB[25823.30,26025.00] vol=2.1x ATR=50.82 |
| Stop hit — per-position SL triggered | 2024-05-31 11:30:00 | 25836.57 | 25861.66 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2024-06-04 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-04 10:35:00 | 26285.80 | 26012.93 | 0.00 | ORB-long ORB[25771.00,25987.05] vol=3.7x ATR=100.31 |
| Stop hit — per-position SL triggered | 2024-06-04 10:40:00 | 26185.49 | 26022.24 | 0.00 | SL hit |

### Cycle 12 — SELL (started 2024-06-14 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-14 09:55:00 | 27650.00 | 27715.71 | 0.00 | ORB-short ORB[27661.75,27754.80] vol=3.1x ATR=49.38 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-14 10:10:00 | 27575.93 | 27694.49 | 0.00 | T1 1.5R @ 27575.93 |
| Target hit | 2024-06-14 15:20:00 | 27460.15 | 27537.50 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 13 — SELL (started 2024-06-18 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-18 10:50:00 | 27150.00 | 27266.07 | 0.00 | ORB-short ORB[27250.10,27454.75] vol=5.7x ATR=46.54 |
| Stop hit — per-position SL triggered | 2024-06-18 10:55:00 | 27196.54 | 27253.00 | 0.00 | SL hit |

### Cycle 14 — SELL (started 2024-06-19 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-19 10:10:00 | 27116.10 | 27167.05 | 0.00 | ORB-short ORB[27155.75,27274.95] vol=3.6x ATR=43.01 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-19 10:15:00 | 27051.58 | 27159.65 | 0.00 | T1 1.5R @ 27051.58 |
| Stop hit — per-position SL triggered | 2024-06-19 10:50:00 | 27116.10 | 27108.99 | 0.00 | SL hit |

### Cycle 15 — SELL (started 2024-06-20 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-20 09:30:00 | 26828.70 | 26875.19 | 0.00 | ORB-short ORB[26850.00,26950.00] vol=1.9x ATR=61.07 |
| Stop hit — per-position SL triggered | 2024-06-20 09:40:00 | 26889.77 | 26872.91 | 0.00 | SL hit |

### Cycle 16 — BUY (started 2024-06-21 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-21 09:30:00 | 27211.00 | 27150.78 | 0.00 | ORB-long ORB[26960.00,27199.95] vol=2.7x ATR=53.19 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-21 09:35:00 | 27290.78 | 27209.56 | 0.00 | T1 1.5R @ 27290.78 |
| Stop hit — per-position SL triggered | 2024-06-21 09:40:00 | 27211.00 | 27212.71 | 0.00 | SL hit |

### Cycle 17 — SELL (started 2024-06-25 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-25 09:40:00 | 26823.90 | 26878.57 | 0.00 | ORB-short ORB[26860.00,27011.65] vol=1.8x ATR=62.35 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-25 10:10:00 | 26730.38 | 26844.07 | 0.00 | T1 1.5R @ 26730.38 |
| Stop hit — per-position SL triggered | 2024-06-25 10:30:00 | 26823.90 | 26816.06 | 0.00 | SL hit |

### Cycle 18 — BUY (started 2024-06-26 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-26 10:55:00 | 27567.45 | 27222.60 | 0.00 | ORB-long ORB[26855.65,27248.00] vol=3.0x ATR=78.22 |
| Stop hit — per-position SL triggered | 2024-06-26 11:00:00 | 27489.23 | 27234.32 | 0.00 | SL hit |

### Cycle 19 — BUY (started 2024-06-27 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-27 09:45:00 | 27866.00 | 27714.23 | 0.00 | ORB-long ORB[27577.10,27774.70] vol=1.9x ATR=76.73 |
| Stop hit — per-position SL triggered | 2024-06-27 09:55:00 | 27789.27 | 27728.86 | 0.00 | SL hit |

### Cycle 20 — BUY (started 2024-07-03 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-03 09:35:00 | 27752.25 | 27707.85 | 0.00 | ORB-long ORB[27623.20,27734.65] vol=1.7x ATR=59.02 |
| Stop hit — per-position SL triggered | 2024-07-03 09:50:00 | 27693.23 | 27710.65 | 0.00 | SL hit |

### Cycle 21 — SELL (started 2024-07-04 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-04 09:50:00 | 27565.00 | 27664.10 | 0.00 | ORB-short ORB[27612.05,27836.70] vol=2.0x ATR=67.94 |
| Stop hit — per-position SL triggered | 2024-07-04 09:55:00 | 27632.94 | 27660.29 | 0.00 | SL hit |

### Cycle 22 — SELL (started 2024-07-05 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-05 11:05:00 | 27793.10 | 27842.41 | 0.00 | ORB-short ORB[27820.80,27997.15] vol=4.8x ATR=50.78 |
| Stop hit — per-position SL triggered | 2024-07-05 11:10:00 | 27843.88 | 27843.58 | 0.00 | SL hit |

### Cycle 23 — SELL (started 2024-07-08 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-08 10:50:00 | 27808.10 | 27872.78 | 0.00 | ORB-short ORB[27865.00,28099.50] vol=3.1x ATR=61.85 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-08 11:20:00 | 27715.33 | 27850.33 | 0.00 | T1 1.5R @ 27715.33 |
| Target hit | 2024-07-08 12:35:00 | 27800.00 | 27779.14 | 0.00 | Trail-exit close>VWAP |

### Cycle 24 — BUY (started 2024-07-09 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-09 10:25:00 | 28247.95 | 28051.89 | 0.00 | ORB-long ORB[27860.60,28037.10] vol=8.2x ATR=70.93 |
| Stop hit — per-position SL triggered | 2024-07-09 10:30:00 | 28177.02 | 28105.14 | 0.00 | SL hit |

### Cycle 25 — SELL (started 2024-07-10 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-10 10:05:00 | 28216.70 | 28375.86 | 0.00 | ORB-short ORB[28337.00,28545.90] vol=2.2x ATR=69.45 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-10 10:10:00 | 28112.52 | 28347.75 | 0.00 | T1 1.5R @ 28112.52 |
| Stop hit — per-position SL triggered | 2024-07-10 10:15:00 | 28216.70 | 28344.11 | 0.00 | SL hit |

### Cycle 26 — SELL (started 2024-07-11 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-11 11:05:00 | 27929.50 | 28017.67 | 0.00 | ORB-short ORB[27960.10,28199.95] vol=3.2x ATR=51.28 |
| Stop hit — per-position SL triggered | 2024-07-11 11:30:00 | 27980.78 | 28009.19 | 0.00 | SL hit |

### Cycle 27 — SELL (started 2024-07-12 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-12 10:05:00 | 27432.95 | 27664.58 | 0.00 | ORB-short ORB[27700.10,27899.80] vol=4.2x ATR=78.94 |
| Stop hit — per-position SL triggered | 2024-07-12 10:15:00 | 27511.89 | 27632.40 | 0.00 | SL hit |

### Cycle 28 — BUY (started 2024-07-22 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-22 10:05:00 | 27531.90 | 27291.52 | 0.00 | ORB-long ORB[27119.60,27399.95] vol=2.0x ATR=89.84 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-22 10:20:00 | 27666.66 | 27453.87 | 0.00 | T1 1.5R @ 27666.66 |
| Target hit | 2024-07-22 14:10:00 | 27811.65 | 27812.81 | 0.00 | Trail-exit close<VWAP |

### Cycle 29 — SELL (started 2024-07-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-25 10:15:00 | 27487.35 | 27599.31 | 0.00 | ORB-short ORB[27519.30,27724.85] vol=1.7x ATR=63.03 |
| Stop hit — per-position SL triggered | 2024-07-25 10:30:00 | 27550.38 | 27573.75 | 0.00 | SL hit |

### Cycle 30 — SELL (started 2024-07-29 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-29 10:55:00 | 28183.90 | 28338.32 | 0.00 | ORB-short ORB[28449.50,28698.80] vol=2.4x ATR=52.82 |
| Stop hit — per-position SL triggered | 2024-07-29 11:20:00 | 28236.72 | 28318.00 | 0.00 | SL hit |

### Cycle 31 — BUY (started 2024-07-31 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-31 09:30:00 | 28370.25 | 28330.96 | 0.00 | ORB-long ORB[28212.00,28345.00] vol=4.7x ATR=71.64 |
| Stop hit — per-position SL triggered | 2024-07-31 09:35:00 | 28298.61 | 28330.54 | 0.00 | SL hit |

### Cycle 32 — BUY (started 2024-08-05 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-05 10:30:00 | 28498.55 | 28188.93 | 0.00 | ORB-long ORB[27900.50,28105.30] vol=1.9x ATR=86.61 |
| Stop hit — per-position SL triggered | 2024-08-05 10:40:00 | 28411.94 | 28218.46 | 0.00 | SL hit |

### Cycle 33 — BUY (started 2024-08-06 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-06 10:05:00 | 28336.05 | 28232.33 | 0.00 | ORB-long ORB[27915.15,28204.00] vol=1.6x ATR=70.58 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-06 10:10:00 | 28441.92 | 28308.66 | 0.00 | T1 1.5R @ 28441.92 |
| Target hit | 2024-08-06 10:25:00 | 28345.00 | 28350.68 | 0.00 | Trail-exit close<VWAP |

### Cycle 34 — SELL (started 2024-08-09 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-09 09:35:00 | 27213.20 | 27315.72 | 0.00 | ORB-short ORB[27250.00,27627.30] vol=1.9x ATR=108.47 |
| Stop hit — per-position SL triggered | 2024-08-09 09:45:00 | 27321.67 | 27311.97 | 0.00 | SL hit |

### Cycle 35 — SELL (started 2024-08-12 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-12 09:30:00 | 26990.00 | 27089.78 | 0.00 | ORB-short ORB[27037.05,27327.95] vol=2.8x ATR=92.56 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-12 09:50:00 | 26851.16 | 27036.00 | 0.00 | T1 1.5R @ 26851.16 |
| Stop hit — per-position SL triggered | 2024-08-12 10:10:00 | 26990.00 | 27003.86 | 0.00 | SL hit |

### Cycle 36 — BUY (started 2024-08-13 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-13 10:55:00 | 27335.95 | 27126.81 | 0.00 | ORB-long ORB[26918.85,27172.00] vol=1.8x ATR=67.63 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-13 12:10:00 | 27437.40 | 27211.10 | 0.00 | T1 1.5R @ 27437.40 |
| Stop hit — per-position SL triggered | 2024-08-13 12:25:00 | 27335.95 | 27218.40 | 0.00 | SL hit |

### Cycle 37 — BUY (started 2024-08-16 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-16 10:45:00 | 27296.05 | 27239.49 | 0.00 | ORB-long ORB[27110.50,27242.60] vol=4.4x ATR=49.73 |
| Stop hit — per-position SL triggered | 2024-08-16 10:55:00 | 27246.32 | 27244.73 | 0.00 | SL hit |

### Cycle 38 — SELL (started 2024-08-19 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-19 11:00:00 | 27667.80 | 27814.71 | 0.00 | ORB-short ORB[27893.45,27994.70] vol=2.8x ATR=56.87 |
| Stop hit — per-position SL triggered | 2024-08-19 11:05:00 | 27724.67 | 27881.60 | 0.00 | SL hit |

### Cycle 39 — SELL (started 2024-08-20 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-20 10:45:00 | 27926.30 | 28010.39 | 0.00 | ORB-short ORB[27984.45,28250.00] vol=2.9x ATR=59.82 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-20 10:50:00 | 27836.57 | 27993.29 | 0.00 | T1 1.5R @ 27836.57 |
| Target hit | 2024-08-20 15:20:00 | 27830.00 | 27834.45 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 40 — BUY (started 2024-08-21 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-21 10:00:00 | 28167.95 | 28047.20 | 0.00 | ORB-long ORB[27706.95,28039.95] vol=2.2x ATR=60.96 |
| Stop hit — per-position SL triggered | 2024-08-21 10:15:00 | 28106.99 | 28086.39 | 0.00 | SL hit |

### Cycle 41 — BUY (started 2024-08-22 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-22 09:45:00 | 29118.75 | 28893.29 | 0.00 | ORB-long ORB[28501.90,28818.45] vol=2.6x ATR=98.15 |
| Stop hit — per-position SL triggered | 2024-08-22 09:50:00 | 29020.60 | 28934.00 | 0.00 | SL hit |

### Cycle 42 — SELL (started 2024-08-23 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-23 10:00:00 | 28938.55 | 29066.71 | 0.00 | ORB-short ORB[29000.00,29287.30] vol=1.6x ATR=84.49 |
| Stop hit — per-position SL triggered | 2024-08-23 10:05:00 | 29023.04 | 29058.84 | 0.00 | SL hit |

### Cycle 43 — BUY (started 2024-08-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-27 11:15:00 | 29207.95 | 29052.30 | 0.00 | ORB-long ORB[28900.00,29191.95] vol=2.8x ATR=81.26 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-27 11:20:00 | 29329.84 | 29061.97 | 0.00 | T1 1.5R @ 29329.84 |
| Stop hit — per-position SL triggered | 2024-08-27 11:25:00 | 29207.95 | 29068.98 | 0.00 | SL hit |

### Cycle 44 — BUY (started 2024-08-28 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-28 09:55:00 | 29960.00 | 29827.29 | 0.00 | ORB-long ORB[29630.00,29920.00] vol=2.6x ATR=94.95 |
| Stop hit — per-position SL triggered | 2024-08-28 10:05:00 | 29865.05 | 29834.71 | 0.00 | SL hit |

### Cycle 45 — BUY (started 2024-08-30 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-30 10:55:00 | 30310.00 | 30107.17 | 0.00 | ORB-long ORB[29902.00,30185.95] vol=5.3x ATR=81.11 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-30 11:15:00 | 30431.66 | 30172.85 | 0.00 | T1 1.5R @ 30431.66 |
| Stop hit — per-position SL triggered | 2024-08-30 13:05:00 | 30310.00 | 30285.27 | 0.00 | SL hit |

### Cycle 46 — SELL (started 2024-09-02 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-02 09:50:00 | 29950.00 | 30086.50 | 0.00 | ORB-short ORB[30044.40,30325.00] vol=1.6x ATR=82.22 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-02 11:25:00 | 29826.67 | 30001.24 | 0.00 | T1 1.5R @ 29826.67 |
| Target hit | 2024-09-02 14:00:00 | 29935.00 | 29927.15 | 0.00 | Trail-exit close>VWAP |

### Cycle 47 — BUY (started 2024-09-03 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-03 09:35:00 | 30329.00 | 30214.82 | 0.00 | ORB-long ORB[29982.35,30151.15] vol=3.1x ATR=82.30 |
| Stop hit — per-position SL triggered | 2024-09-03 09:40:00 | 30246.70 | 30231.09 | 0.00 | SL hit |

### Cycle 48 — BUY (started 2024-09-05 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-05 10:10:00 | 29999.80 | 29939.52 | 0.00 | ORB-long ORB[29832.40,29949.55] vol=3.7x ATR=52.62 |
| Stop hit — per-position SL triggered | 2024-09-05 10:15:00 | 29947.18 | 29939.89 | 0.00 | SL hit |

### Cycle 49 — BUY (started 2024-09-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-06 11:15:00 | 29885.10 | 29826.41 | 0.00 | ORB-long ORB[29594.75,29878.00] vol=2.3x ATR=47.10 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-06 11:40:00 | 29955.74 | 29846.01 | 0.00 | T1 1.5R @ 29955.74 |
| Stop hit — per-position SL triggered | 2024-09-06 14:15:00 | 29885.10 | 29891.56 | 0.00 | SL hit |

### Cycle 50 — SELL (started 2024-09-10 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-10 09:40:00 | 29920.00 | 30034.02 | 0.00 | ORB-short ORB[29970.00,30267.20] vol=1.6x ATR=73.72 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-10 09:50:00 | 29809.42 | 30025.13 | 0.00 | T1 1.5R @ 29809.42 |
| Stop hit — per-position SL triggered | 2024-09-10 11:35:00 | 29920.00 | 29917.68 | 0.00 | SL hit |

### Cycle 51 — SELL (started 2024-09-12 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-12 10:10:00 | 29413.25 | 29573.05 | 0.00 | ORB-short ORB[29541.35,29767.90] vol=1.9x ATR=70.18 |
| Stop hit — per-position SL triggered | 2024-09-12 10:50:00 | 29483.43 | 29519.45 | 0.00 | SL hit |

### Cycle 52 — SELL (started 2024-09-16 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-16 11:00:00 | 29424.05 | 29509.58 | 0.00 | ORB-short ORB[29450.00,29770.10] vol=2.0x ATR=47.33 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-16 12:05:00 | 29353.05 | 29471.33 | 0.00 | T1 1.5R @ 29353.05 |
| Target hit | 2024-09-16 15:20:00 | 29141.40 | 29309.74 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 53 — SELL (started 2024-09-18 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-18 09:30:00 | 28534.55 | 28647.57 | 0.00 | ORB-short ORB[28576.00,28999.95] vol=2.6x ATR=81.94 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-18 09:55:00 | 28411.64 | 28523.43 | 0.00 | T1 1.5R @ 28411.64 |
| Target hit | 2024-09-18 15:20:00 | 27824.85 | 27911.41 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 54 — SELL (started 2024-09-20 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-20 10:00:00 | 27860.85 | 27959.04 | 0.00 | ORB-short ORB[27931.60,28200.00] vol=1.7x ATR=78.30 |
| Stop hit — per-position SL triggered | 2024-09-20 12:00:00 | 27939.15 | 27906.12 | 0.00 | SL hit |

### Cycle 55 — SELL (started 2024-09-24 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-24 09:55:00 | 28266.60 | 28375.52 | 0.00 | ORB-short ORB[28270.10,28519.95] vol=2.4x ATR=59.99 |
| Stop hit — per-position SL triggered | 2024-09-24 10:00:00 | 28326.59 | 28364.90 | 0.00 | SL hit |

### Cycle 56 — BUY (started 2024-10-03 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-03 09:35:00 | 29007.35 | 28885.33 | 0.00 | ORB-long ORB[28640.10,28950.00] vol=4.1x ATR=95.18 |
| Stop hit — per-position SL triggered | 2024-10-03 09:45:00 | 28912.17 | 28892.35 | 0.00 | SL hit |

### Cycle 57 — BUY (started 2024-10-08 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-08 09:40:00 | 28154.00 | 28068.00 | 0.00 | ORB-long ORB[27958.75,28069.00] vol=2.3x ATR=75.12 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-08 09:55:00 | 28266.69 | 28111.00 | 0.00 | T1 1.5R @ 28266.69 |
| Target hit | 2024-10-08 12:20:00 | 28425.30 | 28425.82 | 0.00 | Trail-exit close<VWAP |

### Cycle 58 — BUY (started 2024-10-10 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-10 09:55:00 | 28800.00 | 28676.95 | 0.00 | ORB-long ORB[28587.35,28716.90] vol=2.0x ATR=64.93 |
| Stop hit — per-position SL triggered | 2024-10-10 10:00:00 | 28735.07 | 28679.51 | 0.00 | SL hit |

### Cycle 59 — BUY (started 2024-10-11 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-11 10:30:00 | 28746.95 | 28624.12 | 0.00 | ORB-long ORB[28340.60,28654.65] vol=3.0x ATR=70.83 |
| Stop hit — per-position SL triggered | 2024-10-11 10:55:00 | 28676.12 | 28646.16 | 0.00 | SL hit |

### Cycle 60 — BUY (started 2024-10-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-14 11:15:00 | 28947.95 | 28826.40 | 0.00 | ORB-long ORB[28690.55,28941.50] vol=3.5x ATR=54.82 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-14 11:45:00 | 29030.19 | 28858.55 | 0.00 | T1 1.5R @ 29030.19 |
| Stop hit — per-position SL triggered | 2024-10-14 14:45:00 | 28947.95 | 28952.06 | 0.00 | SL hit |

### Cycle 61 — BUY (started 2024-10-18 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-18 10:10:00 | 28912.60 | 28699.04 | 0.00 | ORB-long ORB[28290.00,28592.00] vol=5.3x ATR=105.93 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-18 10:40:00 | 29071.49 | 28785.19 | 0.00 | T1 1.5R @ 29071.49 |
| Stop hit — per-position SL triggered | 2024-10-18 11:40:00 | 28912.60 | 28870.00 | 0.00 | SL hit |

### Cycle 62 — SELL (started 2024-10-29 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-29 09:30:00 | 28138.10 | 28217.07 | 0.00 | ORB-short ORB[28250.35,28400.00] vol=2.9x ATR=78.86 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-29 09:35:00 | 28019.82 | 28149.53 | 0.00 | T1 1.5R @ 28019.82 |
| Target hit | 2024-10-29 10:40:00 | 27840.15 | 27774.01 | 0.00 | Trail-exit close>VWAP |

### Cycle 63 — BUY (started 2024-10-30 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-30 11:10:00 | 28536.85 | 28377.84 | 0.00 | ORB-long ORB[28270.35,28500.05] vol=3.6x ATR=70.92 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-30 11:45:00 | 28643.24 | 28434.81 | 0.00 | T1 1.5R @ 28643.24 |
| Stop hit — per-position SL triggered | 2024-10-30 13:15:00 | 28536.85 | 28491.36 | 0.00 | SL hit |

### Cycle 64 — BUY (started 2024-10-31 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-31 09:40:00 | 28564.20 | 28452.08 | 0.00 | ORB-long ORB[28285.00,28498.00] vol=1.8x ATR=69.57 |
| Stop hit — per-position SL triggered | 2024-10-31 09:45:00 | 28494.63 | 28454.27 | 0.00 | SL hit |

### Cycle 65 — BUY (started 2024-11-04 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-04 09:40:00 | 29430.00 | 29285.94 | 0.00 | ORB-long ORB[29123.25,29400.00] vol=1.9x ATR=104.25 |
| Stop hit — per-position SL triggered | 2024-11-04 09:50:00 | 29325.75 | 29301.09 | 0.00 | SL hit |

### Cycle 66 — SELL (started 2024-11-05 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-05 10:35:00 | 29161.00 | 29411.57 | 0.00 | ORB-short ORB[29460.45,29809.20] vol=1.6x ATR=94.40 |
| Stop hit — per-position SL triggered | 2024-11-05 11:35:00 | 29255.40 | 29339.50 | 0.00 | SL hit |

### Cycle 67 — SELL (started 2024-11-06 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-06 11:05:00 | 28959.05 | 29202.84 | 0.00 | ORB-short ORB[29162.80,29445.65] vol=3.3x ATR=80.25 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-06 11:10:00 | 28838.68 | 29155.76 | 0.00 | T1 1.5R @ 28838.68 |
| Stop hit — per-position SL triggered | 2024-11-06 11:35:00 | 28959.05 | 29061.50 | 0.00 | SL hit |

### Cycle 68 — SELL (started 2024-11-07 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-07 09:40:00 | 28954.20 | 29145.73 | 0.00 | ORB-short ORB[29137.15,29500.00] vol=1.5x ATR=87.75 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-07 09:55:00 | 28822.58 | 29100.75 | 0.00 | T1 1.5R @ 28822.58 |
| Stop hit — per-position SL triggered | 2024-11-07 10:05:00 | 28954.20 | 29030.69 | 0.00 | SL hit |

### Cycle 69 — BUY (started 2024-11-11 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-11 10:55:00 | 28730.85 | 28562.06 | 0.00 | ORB-long ORB[28356.05,28651.15] vol=2.0x ATR=62.64 |
| Stop hit — per-position SL triggered | 2024-11-11 12:00:00 | 28668.21 | 28609.10 | 0.00 | SL hit |

### Cycle 70 — BUY (started 2024-11-12 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-12 09:40:00 | 28964.25 | 28890.17 | 0.00 | ORB-long ORB[28749.15,28888.15] vol=1.7x ATR=59.70 |
| Stop hit — per-position SL triggered | 2024-11-12 09:50:00 | 28904.55 | 28900.05 | 0.00 | SL hit |

### Cycle 71 — SELL (started 2024-11-13 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-13 09:45:00 | 28294.85 | 28363.35 | 0.00 | ORB-short ORB[28379.60,28593.50] vol=2.6x ATR=87.50 |
| Stop hit — per-position SL triggered | 2024-11-13 09:50:00 | 28382.35 | 28362.86 | 0.00 | SL hit |

### Cycle 72 — BUY (started 2024-11-22 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-22 10:45:00 | 27370.00 | 27323.84 | 0.00 | ORB-long ORB[27114.70,27332.95] vol=3.6x ATR=56.87 |
| Stop hit — per-position SL triggered | 2024-11-22 10:55:00 | 27313.13 | 27325.16 | 0.00 | SL hit |

### Cycle 73 — SELL (started 2024-11-25 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-25 09:45:00 | 27640.25 | 27736.39 | 0.00 | ORB-short ORB[27646.20,27978.70] vol=1.5x ATR=73.84 |
| Stop hit — per-position SL triggered | 2024-11-25 09:55:00 | 27714.09 | 27727.85 | 0.00 | SL hit |

### Cycle 74 — SELL (started 2024-11-26 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-26 09:35:00 | 27538.00 | 27663.68 | 0.00 | ORB-short ORB[27587.90,27861.65] vol=1.6x ATR=65.64 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-26 10:05:00 | 27439.54 | 27577.31 | 0.00 | T1 1.5R @ 27439.54 |
| Stop hit — per-position SL triggered | 2024-11-26 10:20:00 | 27538.00 | 27558.62 | 0.00 | SL hit |

### Cycle 75 — SELL (started 2024-11-27 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-27 10:40:00 | 27370.95 | 27502.86 | 0.00 | ORB-short ORB[27501.20,27653.15] vol=3.4x ATR=58.46 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-27 11:20:00 | 27283.26 | 27448.71 | 0.00 | T1 1.5R @ 27283.26 |
| Target hit | 2024-11-27 12:40:00 | 27369.95 | 27345.56 | 0.00 | Trail-exit close>VWAP |

### Cycle 76 — SELL (started 2024-11-28 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-28 11:10:00 | 27443.90 | 27501.06 | 0.00 | ORB-short ORB[27505.30,27642.90] vol=2.3x ATR=66.99 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-28 13:05:00 | 27343.42 | 27473.87 | 0.00 | T1 1.5R @ 27343.42 |
| Stop hit — per-position SL triggered | 2024-11-28 13:55:00 | 27443.90 | 27456.07 | 0.00 | SL hit |

### Cycle 77 — BUY (started 2024-11-29 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-29 10:50:00 | 27771.80 | 27594.70 | 0.00 | ORB-long ORB[27378.30,27695.95] vol=2.5x ATR=76.85 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-29 11:15:00 | 27887.07 | 27614.52 | 0.00 | T1 1.5R @ 27887.07 |
| Stop hit — per-position SL triggered | 2024-11-29 11:45:00 | 27771.80 | 27630.96 | 0.00 | SL hit |

### Cycle 78 — BUY (started 2024-12-02 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-02 09:50:00 | 28033.30 | 27962.28 | 0.00 | ORB-long ORB[27800.00,27995.00] vol=2.5x ATR=75.09 |
| Stop hit — per-position SL triggered | 2024-12-02 10:05:00 | 27958.21 | 27969.04 | 0.00 | SL hit |

### Cycle 79 — BUY (started 2024-12-04 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-04 10:55:00 | 28739.70 | 28571.72 | 0.00 | ORB-long ORB[28391.55,28629.95] vol=2.6x ATR=67.11 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-04 11:55:00 | 28840.36 | 28636.36 | 0.00 | T1 1.5R @ 28840.36 |
| Target hit | 2024-12-04 15:20:00 | 28825.70 | 28719.22 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 80 — BUY (started 2024-12-05 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-05 09:40:00 | 29035.95 | 28912.99 | 0.00 | ORB-long ORB[28709.00,28949.75] vol=6.6x ATR=69.96 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-05 09:45:00 | 29140.89 | 28987.05 | 0.00 | T1 1.5R @ 29140.89 |
| Stop hit — per-position SL triggered | 2024-12-05 09:50:00 | 29035.95 | 29005.33 | 0.00 | SL hit |

### Cycle 81 — BUY (started 2024-12-06 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-06 09:35:00 | 29196.45 | 29136.56 | 0.00 | ORB-long ORB[29040.05,29183.15] vol=3.7x ATR=68.09 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-06 09:55:00 | 29298.58 | 29183.94 | 0.00 | T1 1.5R @ 29298.58 |
| Stop hit — per-position SL triggered | 2024-12-06 10:00:00 | 29196.45 | 29186.29 | 0.00 | SL hit |

### Cycle 82 — BUY (started 2024-12-11 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-11 10:30:00 | 29257.30 | 29137.02 | 0.00 | ORB-long ORB[29016.10,29199.95] vol=1.8x ATR=46.07 |
| Stop hit — per-position SL triggered | 2024-12-11 10:35:00 | 29211.23 | 29148.18 | 0.00 | SL hit |

### Cycle 83 — SELL (started 2024-12-13 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-13 11:00:00 | 28500.00 | 28664.59 | 0.00 | ORB-short ORB[28505.00,28811.00] vol=1.5x ATR=64.61 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-13 12:35:00 | 28403.09 | 28597.30 | 0.00 | T1 1.5R @ 28403.09 |
| Stop hit — per-position SL triggered | 2024-12-13 13:50:00 | 28500.00 | 28550.08 | 0.00 | SL hit |

### Cycle 84 — SELL (started 2024-12-16 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-16 10:55:00 | 28455.70 | 28595.98 | 0.00 | ORB-short ORB[28532.35,28790.00] vol=1.9x ATR=64.11 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-16 11:55:00 | 28359.54 | 28518.50 | 0.00 | T1 1.5R @ 28359.54 |
| Target hit | 2024-12-16 15:20:00 | 28260.00 | 28389.24 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 85 — SELL (started 2024-12-17 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-17 10:55:00 | 28160.00 | 28160.08 | 0.00 | ORB-short ORB[28185.95,28368.50] vol=1.7x ATR=49.57 |
| Stop hit — per-position SL triggered | 2024-12-17 12:25:00 | 28209.57 | 28191.22 | 0.00 | SL hit |

### Cycle 86 — BUY (started 2024-12-18 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-18 09:50:00 | 28331.30 | 28199.07 | 0.00 | ORB-long ORB[27893.75,28250.00] vol=2.2x ATR=79.30 |
| Stop hit — per-position SL triggered | 2024-12-18 10:00:00 | 28252.00 | 28234.31 | 0.00 | SL hit |

### Cycle 87 — BUY (started 2024-12-19 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-19 10:25:00 | 28199.95 | 28085.46 | 0.00 | ORB-long ORB[27940.05,28196.70] vol=2.6x ATR=75.29 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-19 10:40:00 | 28312.88 | 28117.52 | 0.00 | T1 1.5R @ 28312.88 |
| Target hit | 2024-12-19 15:20:00 | 28970.20 | 28850.09 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 88 — BUY (started 2024-12-20 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-20 10:40:00 | 29114.05 | 28975.78 | 0.00 | ORB-long ORB[28927.00,29086.95] vol=1.9x ATR=91.56 |
| Stop hit — per-position SL triggered | 2024-12-20 10:50:00 | 29022.49 | 29022.79 | 0.00 | SL hit |

### Cycle 89 — BUY (started 2024-12-23 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-23 10:40:00 | 28721.05 | 28611.78 | 0.00 | ORB-long ORB[28462.20,28671.95] vol=1.9x ATR=81.56 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-23 11:00:00 | 28843.39 | 28639.85 | 0.00 | T1 1.5R @ 28843.39 |
| Stop hit — per-position SL triggered | 2024-12-23 11:35:00 | 28721.05 | 28672.46 | 0.00 | SL hit |

### Cycle 90 — BUY (started 2024-12-31 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-31 11:05:00 | 29975.00 | 29821.86 | 0.00 | ORB-long ORB[29670.15,29900.00] vol=2.3x ATR=79.24 |
| Stop hit — per-position SL triggered | 2024-12-31 12:35:00 | 29895.76 | 29888.13 | 0.00 | SL hit |

### Cycle 91 — SELL (started 2025-01-02 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-02 09:50:00 | 29793.65 | 29913.39 | 0.00 | ORB-short ORB[29901.00,30113.90] vol=1.5x ATR=58.43 |
| Stop hit — per-position SL triggered | 2025-01-02 09:55:00 | 29852.08 | 29907.12 | 0.00 | SL hit |

### Cycle 92 — SELL (started 2025-01-03 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-03 09:30:00 | 29694.55 | 29827.30 | 0.00 | ORB-short ORB[29719.45,30122.35] vol=3.1x ATR=73.71 |
| Stop hit — per-position SL triggered | 2025-01-03 09:40:00 | 29768.26 | 29801.65 | 0.00 | SL hit |

### Cycle 93 — SELL (started 2025-01-08 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-08 09:40:00 | 29809.15 | 30000.35 | 0.00 | ORB-short ORB[30001.05,30200.00] vol=3.0x ATR=79.72 |
| Stop hit — per-position SL triggered | 2025-01-08 09:50:00 | 29888.87 | 29971.78 | 0.00 | SL hit |

### Cycle 94 — BUY (started 2025-01-09 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-09 11:05:00 | 29756.40 | 29614.03 | 0.00 | ORB-long ORB[29449.95,29705.95] vol=1.8x ATR=67.43 |
| Stop hit — per-position SL triggered | 2025-01-09 12:15:00 | 29688.97 | 29671.22 | 0.00 | SL hit |

### Cycle 95 — SELL (started 2025-01-10 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-10 09:55:00 | 29475.15 | 29630.23 | 0.00 | ORB-short ORB[29576.10,29821.95] vol=1.6x ATR=76.43 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-10 10:30:00 | 29360.51 | 29557.42 | 0.00 | T1 1.5R @ 29360.51 |
| Target hit | 2025-01-10 15:20:00 | 28817.60 | 29148.81 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 96 — SELL (started 2025-01-13 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-13 10:50:00 | 28303.55 | 28456.96 | 0.00 | ORB-short ORB[28400.00,28656.25] vol=1.7x ATR=75.93 |
| Stop hit — per-position SL triggered | 2025-01-13 11:05:00 | 28379.48 | 28433.24 | 0.00 | SL hit |

### Cycle 97 — SELL (started 2025-01-16 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-16 10:35:00 | 27500.90 | 27636.86 | 0.00 | ORB-short ORB[27636.30,27950.00] vol=2.1x ATR=72.21 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-16 11:20:00 | 27392.58 | 27582.10 | 0.00 | T1 1.5R @ 27392.58 |
| Target hit | 2025-01-16 15:20:00 | 27261.55 | 27393.90 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 98 — BUY (started 2025-01-17 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-17 09:45:00 | 27550.00 | 27489.90 | 0.00 | ORB-long ORB[27298.10,27440.00] vol=4.5x ATR=66.92 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-17 10:00:00 | 27650.38 | 27526.40 | 0.00 | T1 1.5R @ 27650.38 |
| Target hit | 2025-01-17 15:20:00 | 27939.10 | 27819.47 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 99 — SELL (started 2025-01-22 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-22 09:50:00 | 27516.25 | 27634.26 | 0.00 | ORB-short ORB[27577.25,27839.95] vol=2.2x ATR=84.03 |
| Stop hit — per-position SL triggered | 2025-01-22 10:40:00 | 27600.28 | 27608.79 | 0.00 | SL hit |

### Cycle 100 — SELL (started 2025-01-27 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-27 09:45:00 | 26985.05 | 27159.35 | 0.00 | ORB-short ORB[27051.10,27357.50] vol=2.4x ATR=83.97 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-27 10:10:00 | 26859.09 | 27059.98 | 0.00 | T1 1.5R @ 26859.09 |
| Target hit | 2025-01-27 15:20:00 | 26513.80 | 26758.56 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 101 — BUY (started 2025-01-29 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-29 09:35:00 | 25625.00 | 25489.93 | 0.00 | ORB-long ORB[25325.00,25584.95] vol=1.7x ATR=109.20 |
| Stop hit — per-position SL triggered | 2025-01-29 09:45:00 | 25515.80 | 25515.97 | 0.00 | SL hit |

### Cycle 102 — BUY (started 2025-02-05 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-05 09:35:00 | 26498.00 | 26274.21 | 0.00 | ORB-long ORB[26026.05,26396.80] vol=2.0x ATR=90.94 |
| Stop hit — per-position SL triggered | 2025-02-05 09:40:00 | 26407.06 | 26291.76 | 0.00 | SL hit |

### Cycle 103 — SELL (started 2025-02-11 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-11 10:50:00 | 29328.75 | 29401.89 | 0.00 | ORB-short ORB[29360.55,29700.00] vol=1.9x ATR=68.49 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-11 11:00:00 | 29226.02 | 29387.37 | 0.00 | T1 1.5R @ 29226.02 |
| Stop hit — per-position SL triggered | 2025-02-11 11:05:00 | 29328.75 | 29384.76 | 0.00 | SL hit |

### Cycle 104 — BUY (started 2025-02-20 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-20 11:00:00 | 29712.10 | 29654.46 | 0.00 | ORB-long ORB[29500.00,29679.00] vol=2.4x ATR=59.86 |
| Stop hit — per-position SL triggered | 2025-02-20 11:20:00 | 29652.24 | 29655.93 | 0.00 | SL hit |

### Cycle 105 — SELL (started 2025-02-24 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-24 10:20:00 | 28742.60 | 28875.50 | 0.00 | ORB-short ORB[28750.00,29149.95] vol=1.5x ATR=117.47 |
| Stop hit — per-position SL triggered | 2025-02-24 10:25:00 | 28860.07 | 28873.56 | 0.00 | SL hit |

### Cycle 106 — BUY (started 2025-02-27 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-27 10:05:00 | 30358.05 | 30152.17 | 0.00 | ORB-long ORB[29910.05,30150.00] vol=3.9x ATR=121.83 |
| Stop hit — per-position SL triggered | 2025-02-27 10:10:00 | 30236.22 | 30160.94 | 0.00 | SL hit |

### Cycle 107 — BUY (started 2025-03-11 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-11 10:20:00 | 30957.40 | 30749.66 | 0.00 | ORB-long ORB[30394.30,30643.25] vol=2.0x ATR=152.49 |
| Stop hit — per-position SL triggered | 2025-03-11 10:35:00 | 30804.91 | 30767.56 | 0.00 | SL hit |

### Cycle 108 — SELL (started 2025-03-12 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-12 09:30:00 | 30270.00 | 30346.89 | 0.00 | ORB-short ORB[30301.20,30679.95] vol=2.9x ATR=87.96 |
| Stop hit — per-position SL triggered | 2025-03-12 09:35:00 | 30357.96 | 30344.96 | 0.00 | SL hit |

### Cycle 109 — BUY (started 2025-03-13 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-13 10:50:00 | 30018.50 | 29942.88 | 0.00 | ORB-long ORB[29635.20,29967.00] vol=2.4x ATR=99.34 |
| Stop hit — per-position SL triggered | 2025-03-13 11:10:00 | 29919.16 | 29942.14 | 0.00 | SL hit |

### Cycle 110 — SELL (started 2025-03-17 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-17 09:35:00 | 29601.85 | 29716.50 | 0.00 | ORB-short ORB[29632.15,29849.90] vol=1.6x ATR=87.84 |
| Stop hit — per-position SL triggered | 2025-03-17 09:40:00 | 29689.69 | 29713.97 | 0.00 | SL hit |

### Cycle 111 — BUY (started 2025-03-19 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-19 10:30:00 | 30232.45 | 29957.26 | 0.00 | ORB-long ORB[29712.10,30038.50] vol=1.6x ATR=100.86 |
| Stop hit — per-position SL triggered | 2025-03-19 11:00:00 | 30131.59 | 30005.97 | 0.00 | SL hit |

### Cycle 112 — BUY (started 2025-03-20 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-20 10:10:00 | 30510.00 | 30358.80 | 0.00 | ORB-long ORB[30156.55,30407.30] vol=2.8x ATR=103.27 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-20 10:30:00 | 30664.91 | 30438.20 | 0.00 | T1 1.5R @ 30664.91 |
| Stop hit — per-position SL triggered | 2025-03-20 10:35:00 | 30510.00 | 30449.18 | 0.00 | SL hit |

### Cycle 113 — BUY (started 2025-03-24 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-24 10:45:00 | 30765.05 | 30555.31 | 0.00 | ORB-long ORB[30300.05,30690.00] vol=4.1x ATR=109.39 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-24 10:50:00 | 30929.14 | 30590.88 | 0.00 | T1 1.5R @ 30929.14 |
| Stop hit — per-position SL triggered | 2025-03-24 10:55:00 | 30765.05 | 30605.39 | 0.00 | SL hit |

### Cycle 114 — BUY (started 2025-03-26 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-26 10:05:00 | 31155.00 | 30931.91 | 0.00 | ORB-long ORB[30753.65,31100.00] vol=3.2x ATR=117.17 |
| Stop hit — per-position SL triggered | 2025-03-26 10:50:00 | 31037.83 | 31018.54 | 0.00 | SL hit |

### Cycle 115 — BUY (started 2025-04-01 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-01 11:05:00 | 30780.00 | 30594.10 | 0.00 | ORB-long ORB[30352.05,30720.60] vol=3.1x ATR=73.88 |
| Stop hit — per-position SL triggered | 2025-04-01 11:10:00 | 30706.12 | 30597.94 | 0.00 | SL hit |

### Cycle 116 — BUY (started 2025-04-09 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-09 10:55:00 | 29544.30 | 29224.89 | 0.00 | ORB-long ORB[28975.55,29249.95] vol=8.1x ATR=91.32 |
| Stop hit — per-position SL triggered | 2025-04-09 11:00:00 | 29452.98 | 29258.04 | 0.00 | SL hit |

### Cycle 117 — BUY (started 2025-04-21 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-21 10:45:00 | 31150.00 | 30934.52 | 0.00 | ORB-long ORB[30705.00,31010.00] vol=2.3x ATR=116.53 |
| Stop hit — per-position SL triggered | 2025-04-21 15:00:00 | 31033.47 | 31066.68 | 0.00 | SL hit |

### Cycle 118 — SELL (started 2025-04-23 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-23 10:35:00 | 30935.00 | 31025.42 | 0.00 | ORB-short ORB[31000.00,31400.00] vol=3.5x ATR=109.03 |
| Stop hit — per-position SL triggered | 2025-04-23 11:25:00 | 31044.03 | 31023.25 | 0.00 | SL hit |

### Cycle 119 — SELL (started 2025-04-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-24 10:15:00 | 30305.00 | 30402.52 | 0.00 | ORB-short ORB[30400.00,30680.00] vol=3.0x ATR=71.68 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-24 11:00:00 | 30197.47 | 30358.49 | 0.00 | T1 1.5R @ 30197.47 |
| Target hit | 2025-04-24 15:20:00 | 30070.00 | 30201.78 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 120 — SELL (started 2025-04-25 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-25 09:35:00 | 29715.00 | 29988.42 | 0.00 | ORB-short ORB[30060.00,30295.00] vol=2.2x ATR=88.86 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-25 09:40:00 | 29581.71 | 29901.29 | 0.00 | T1 1.5R @ 29581.71 |
| Stop hit — per-position SL triggered | 2025-04-25 09:50:00 | 29715.00 | 29848.85 | 0.00 | SL hit |

### Cycle 121 — SELL (started 2025-05-02 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-02 11:05:00 | 29760.00 | 29920.80 | 0.00 | ORB-short ORB[29800.00,30155.00] vol=2.1x ATR=64.15 |
| Stop hit — per-position SL triggered | 2025-05-02 11:15:00 | 29824.15 | 29894.60 | 0.00 | SL hit |

### Cycle 122 — SELL (started 2025-05-06 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-06 10:50:00 | 30325.00 | 30492.18 | 0.00 | ORB-short ORB[30400.00,30695.00] vol=2.1x ATR=81.68 |
| Stop hit — per-position SL triggered | 2025-05-06 11:05:00 | 30406.68 | 30490.90 | 0.00 | SL hit |

### Cycle 123 — BUY (started 2025-05-09 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-09 10:50:00 | 29950.00 | 29714.64 | 0.00 | ORB-long ORB[29600.00,29820.00] vol=4.8x ATR=88.33 |
| Stop hit — per-position SL triggered | 2025-05-09 10:55:00 | 29861.67 | 29721.16 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-05-14 10:20:00 | 27156.85 | 2024-05-14 10:30:00 | 27079.30 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2024-05-15 10:35:00 | 26584.00 | 2024-05-15 10:40:00 | 26637.77 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2024-05-16 10:30:00 | 26372.00 | 2024-05-16 11:50:00 | 26422.61 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest1 | 2024-05-21 09:30:00 | 26417.15 | 2024-05-21 15:20:00 | 26534.95 | STOP_HIT | 1.00 | -0.45% |
| SELL | retest1 | 2024-05-22 09:40:00 | 26286.90 | 2024-05-22 09:50:00 | 26347.33 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2024-05-23 09:35:00 | 25980.00 | 2024-05-23 09:55:00 | 25902.25 | PARTIAL | 0.50 | 0.30% |
| SELL | retest1 | 2024-05-23 09:35:00 | 25980.00 | 2024-05-23 10:50:00 | 25940.00 | TARGET_HIT | 0.50 | 0.15% |
| BUY | retest1 | 2024-05-24 09:35:00 | 26286.05 | 2024-05-24 09:50:00 | 26220.15 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2024-05-29 10:00:00 | 26005.00 | 2024-05-29 10:15:00 | 26100.89 | PARTIAL | 0.50 | 0.37% |
| BUY | retest1 | 2024-05-29 10:00:00 | 26005.00 | 2024-05-29 15:20:00 | 26163.85 | TARGET_HIT | 0.50 | 0.61% |
| SELL | retest1 | 2024-05-30 11:15:00 | 25906.80 | 2024-05-30 11:40:00 | 25955.67 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest1 | 2024-05-31 11:00:00 | 25785.75 | 2024-05-31 11:30:00 | 25836.57 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2024-06-04 10:35:00 | 26285.80 | 2024-06-04 10:40:00 | 26185.49 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest1 | 2024-06-14 09:55:00 | 27650.00 | 2024-06-14 10:10:00 | 27575.93 | PARTIAL | 0.50 | 0.27% |
| SELL | retest1 | 2024-06-14 09:55:00 | 27650.00 | 2024-06-14 15:20:00 | 27460.15 | TARGET_HIT | 0.50 | 0.69% |
| SELL | retest1 | 2024-06-18 10:50:00 | 27150.00 | 2024-06-18 10:55:00 | 27196.54 | STOP_HIT | 1.00 | -0.17% |
| SELL | retest1 | 2024-06-19 10:10:00 | 27116.10 | 2024-06-19 10:15:00 | 27051.58 | PARTIAL | 0.50 | 0.24% |
| SELL | retest1 | 2024-06-19 10:10:00 | 27116.10 | 2024-06-19 10:50:00 | 27116.10 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-06-20 09:30:00 | 26828.70 | 2024-06-20 09:40:00 | 26889.77 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2024-06-21 09:30:00 | 27211.00 | 2024-06-21 09:35:00 | 27290.78 | PARTIAL | 0.50 | 0.29% |
| BUY | retest1 | 2024-06-21 09:30:00 | 27211.00 | 2024-06-21 09:40:00 | 27211.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-06-25 09:40:00 | 26823.90 | 2024-06-25 10:10:00 | 26730.38 | PARTIAL | 0.50 | 0.35% |
| SELL | retest1 | 2024-06-25 09:40:00 | 26823.90 | 2024-06-25 10:30:00 | 26823.90 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-06-26 10:55:00 | 27567.45 | 2024-06-26 11:00:00 | 27489.23 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2024-06-27 09:45:00 | 27866.00 | 2024-06-27 09:55:00 | 27789.27 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2024-07-03 09:35:00 | 27752.25 | 2024-07-03 09:50:00 | 27693.23 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2024-07-04 09:50:00 | 27565.00 | 2024-07-04 09:55:00 | 27632.94 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2024-07-05 11:05:00 | 27793.10 | 2024-07-05 11:10:00 | 27843.88 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest1 | 2024-07-08 10:50:00 | 27808.10 | 2024-07-08 11:20:00 | 27715.33 | PARTIAL | 0.50 | 0.33% |
| SELL | retest1 | 2024-07-08 10:50:00 | 27808.10 | 2024-07-08 12:35:00 | 27800.00 | TARGET_HIT | 0.50 | 0.03% |
| BUY | retest1 | 2024-07-09 10:25:00 | 28247.95 | 2024-07-09 10:30:00 | 28177.02 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2024-07-10 10:05:00 | 28216.70 | 2024-07-10 10:10:00 | 28112.52 | PARTIAL | 0.50 | 0.37% |
| SELL | retest1 | 2024-07-10 10:05:00 | 28216.70 | 2024-07-10 10:15:00 | 28216.70 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-07-11 11:05:00 | 27929.50 | 2024-07-11 11:30:00 | 27980.78 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest1 | 2024-07-12 10:05:00 | 27432.95 | 2024-07-12 10:15:00 | 27511.89 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2024-07-22 10:05:00 | 27531.90 | 2024-07-22 10:20:00 | 27666.66 | PARTIAL | 0.50 | 0.49% |
| BUY | retest1 | 2024-07-22 10:05:00 | 27531.90 | 2024-07-22 14:10:00 | 27811.65 | TARGET_HIT | 0.50 | 1.02% |
| SELL | retest1 | 2024-07-25 10:15:00 | 27487.35 | 2024-07-25 10:30:00 | 27550.38 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2024-07-29 10:55:00 | 28183.90 | 2024-07-29 11:20:00 | 28236.72 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest1 | 2024-07-31 09:30:00 | 28370.25 | 2024-07-31 09:35:00 | 28298.61 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2024-08-05 10:30:00 | 28498.55 | 2024-08-05 10:40:00 | 28411.94 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2024-08-06 10:05:00 | 28336.05 | 2024-08-06 10:10:00 | 28441.92 | PARTIAL | 0.50 | 0.37% |
| BUY | retest1 | 2024-08-06 10:05:00 | 28336.05 | 2024-08-06 10:25:00 | 28345.00 | TARGET_HIT | 0.50 | 0.03% |
| SELL | retest1 | 2024-08-09 09:35:00 | 27213.20 | 2024-08-09 09:45:00 | 27321.67 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest1 | 2024-08-12 09:30:00 | 26990.00 | 2024-08-12 09:50:00 | 26851.16 | PARTIAL | 0.50 | 0.51% |
| SELL | retest1 | 2024-08-12 09:30:00 | 26990.00 | 2024-08-12 10:10:00 | 26990.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-08-13 10:55:00 | 27335.95 | 2024-08-13 12:10:00 | 27437.40 | PARTIAL | 0.50 | 0.37% |
| BUY | retest1 | 2024-08-13 10:55:00 | 27335.95 | 2024-08-13 12:25:00 | 27335.95 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-08-16 10:45:00 | 27296.05 | 2024-08-16 10:55:00 | 27246.32 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest1 | 2024-08-19 11:00:00 | 27667.80 | 2024-08-19 11:05:00 | 27724.67 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2024-08-20 10:45:00 | 27926.30 | 2024-08-20 10:50:00 | 27836.57 | PARTIAL | 0.50 | 0.32% |
| SELL | retest1 | 2024-08-20 10:45:00 | 27926.30 | 2024-08-20 15:20:00 | 27830.00 | TARGET_HIT | 0.50 | 0.34% |
| BUY | retest1 | 2024-08-21 10:00:00 | 28167.95 | 2024-08-21 10:15:00 | 28106.99 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2024-08-22 09:45:00 | 29118.75 | 2024-08-22 09:50:00 | 29020.60 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2024-08-23 10:00:00 | 28938.55 | 2024-08-23 10:05:00 | 29023.04 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2024-08-27 11:15:00 | 29207.95 | 2024-08-27 11:20:00 | 29329.84 | PARTIAL | 0.50 | 0.42% |
| BUY | retest1 | 2024-08-27 11:15:00 | 29207.95 | 2024-08-27 11:25:00 | 29207.95 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-08-28 09:55:00 | 29960.00 | 2024-08-28 10:05:00 | 29865.05 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2024-08-30 10:55:00 | 30310.00 | 2024-08-30 11:15:00 | 30431.66 | PARTIAL | 0.50 | 0.40% |
| BUY | retest1 | 2024-08-30 10:55:00 | 30310.00 | 2024-08-30 13:05:00 | 30310.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-09-02 09:50:00 | 29950.00 | 2024-09-02 11:25:00 | 29826.67 | PARTIAL | 0.50 | 0.41% |
| SELL | retest1 | 2024-09-02 09:50:00 | 29950.00 | 2024-09-02 14:00:00 | 29935.00 | TARGET_HIT | 0.50 | 0.05% |
| BUY | retest1 | 2024-09-03 09:35:00 | 30329.00 | 2024-09-03 09:40:00 | 30246.70 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2024-09-05 10:10:00 | 29999.80 | 2024-09-05 10:15:00 | 29947.18 | STOP_HIT | 1.00 | -0.18% |
| BUY | retest1 | 2024-09-06 11:15:00 | 29885.10 | 2024-09-06 11:40:00 | 29955.74 | PARTIAL | 0.50 | 0.24% |
| BUY | retest1 | 2024-09-06 11:15:00 | 29885.10 | 2024-09-06 14:15:00 | 29885.10 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-09-10 09:40:00 | 29920.00 | 2024-09-10 09:50:00 | 29809.42 | PARTIAL | 0.50 | 0.37% |
| SELL | retest1 | 2024-09-10 09:40:00 | 29920.00 | 2024-09-10 11:35:00 | 29920.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-09-12 10:10:00 | 29413.25 | 2024-09-12 10:50:00 | 29483.43 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2024-09-16 11:00:00 | 29424.05 | 2024-09-16 12:05:00 | 29353.05 | PARTIAL | 0.50 | 0.24% |
| SELL | retest1 | 2024-09-16 11:00:00 | 29424.05 | 2024-09-16 15:20:00 | 29141.40 | TARGET_HIT | 0.50 | 0.96% |
| SELL | retest1 | 2024-09-18 09:30:00 | 28534.55 | 2024-09-18 09:55:00 | 28411.64 | PARTIAL | 0.50 | 0.43% |
| SELL | retest1 | 2024-09-18 09:30:00 | 28534.55 | 2024-09-18 15:20:00 | 27824.85 | TARGET_HIT | 0.50 | 2.49% |
| SELL | retest1 | 2024-09-20 10:00:00 | 27860.85 | 2024-09-20 12:00:00 | 27939.15 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2024-09-24 09:55:00 | 28266.60 | 2024-09-24 10:00:00 | 28326.59 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2024-10-03 09:35:00 | 29007.35 | 2024-10-03 09:45:00 | 28912.17 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2024-10-08 09:40:00 | 28154.00 | 2024-10-08 09:55:00 | 28266.69 | PARTIAL | 0.50 | 0.40% |
| BUY | retest1 | 2024-10-08 09:40:00 | 28154.00 | 2024-10-08 12:20:00 | 28425.30 | TARGET_HIT | 0.50 | 0.96% |
| BUY | retest1 | 2024-10-10 09:55:00 | 28800.00 | 2024-10-10 10:00:00 | 28735.07 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2024-10-11 10:30:00 | 28746.95 | 2024-10-11 10:55:00 | 28676.12 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2024-10-14 11:15:00 | 28947.95 | 2024-10-14 11:45:00 | 29030.19 | PARTIAL | 0.50 | 0.28% |
| BUY | retest1 | 2024-10-14 11:15:00 | 28947.95 | 2024-10-14 14:45:00 | 28947.95 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-10-18 10:10:00 | 28912.60 | 2024-10-18 10:40:00 | 29071.49 | PARTIAL | 0.50 | 0.55% |
| BUY | retest1 | 2024-10-18 10:10:00 | 28912.60 | 2024-10-18 11:40:00 | 28912.60 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-10-29 09:30:00 | 28138.10 | 2024-10-29 09:35:00 | 28019.82 | PARTIAL | 0.50 | 0.42% |
| SELL | retest1 | 2024-10-29 09:30:00 | 28138.10 | 2024-10-29 10:40:00 | 27840.15 | TARGET_HIT | 0.50 | 1.06% |
| BUY | retest1 | 2024-10-30 11:10:00 | 28536.85 | 2024-10-30 11:45:00 | 28643.24 | PARTIAL | 0.50 | 0.37% |
| BUY | retest1 | 2024-10-30 11:10:00 | 28536.85 | 2024-10-30 13:15:00 | 28536.85 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-10-31 09:40:00 | 28564.20 | 2024-10-31 09:45:00 | 28494.63 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2024-11-04 09:40:00 | 29430.00 | 2024-11-04 09:50:00 | 29325.75 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2024-11-05 10:35:00 | 29161.00 | 2024-11-05 11:35:00 | 29255.40 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2024-11-06 11:05:00 | 28959.05 | 2024-11-06 11:10:00 | 28838.68 | PARTIAL | 0.50 | 0.42% |
| SELL | retest1 | 2024-11-06 11:05:00 | 28959.05 | 2024-11-06 11:35:00 | 28959.05 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-11-07 09:40:00 | 28954.20 | 2024-11-07 09:55:00 | 28822.58 | PARTIAL | 0.50 | 0.45% |
| SELL | retest1 | 2024-11-07 09:40:00 | 28954.20 | 2024-11-07 10:05:00 | 28954.20 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-11-11 10:55:00 | 28730.85 | 2024-11-11 12:00:00 | 28668.21 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2024-11-12 09:40:00 | 28964.25 | 2024-11-12 09:50:00 | 28904.55 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2024-11-13 09:45:00 | 28294.85 | 2024-11-13 09:50:00 | 28382.35 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2024-11-22 10:45:00 | 27370.00 | 2024-11-22 10:55:00 | 27313.13 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2024-11-25 09:45:00 | 27640.25 | 2024-11-25 09:55:00 | 27714.09 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2024-11-26 09:35:00 | 27538.00 | 2024-11-26 10:05:00 | 27439.54 | PARTIAL | 0.50 | 0.36% |
| SELL | retest1 | 2024-11-26 09:35:00 | 27538.00 | 2024-11-26 10:20:00 | 27538.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-11-27 10:40:00 | 27370.95 | 2024-11-27 11:20:00 | 27283.26 | PARTIAL | 0.50 | 0.32% |
| SELL | retest1 | 2024-11-27 10:40:00 | 27370.95 | 2024-11-27 12:40:00 | 27369.95 | TARGET_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-11-28 11:10:00 | 27443.90 | 2024-11-28 13:05:00 | 27343.42 | PARTIAL | 0.50 | 0.37% |
| SELL | retest1 | 2024-11-28 11:10:00 | 27443.90 | 2024-11-28 13:55:00 | 27443.90 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-11-29 10:50:00 | 27771.80 | 2024-11-29 11:15:00 | 27887.07 | PARTIAL | 0.50 | 0.42% |
| BUY | retest1 | 2024-11-29 10:50:00 | 27771.80 | 2024-11-29 11:45:00 | 27771.80 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-12-02 09:50:00 | 28033.30 | 2024-12-02 10:05:00 | 27958.21 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2024-12-04 10:55:00 | 28739.70 | 2024-12-04 11:55:00 | 28840.36 | PARTIAL | 0.50 | 0.35% |
| BUY | retest1 | 2024-12-04 10:55:00 | 28739.70 | 2024-12-04 15:20:00 | 28825.70 | TARGET_HIT | 0.50 | 0.30% |
| BUY | retest1 | 2024-12-05 09:40:00 | 29035.95 | 2024-12-05 09:45:00 | 29140.89 | PARTIAL | 0.50 | 0.36% |
| BUY | retest1 | 2024-12-05 09:40:00 | 29035.95 | 2024-12-05 09:50:00 | 29035.95 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-12-06 09:35:00 | 29196.45 | 2024-12-06 09:55:00 | 29298.58 | PARTIAL | 0.50 | 0.35% |
| BUY | retest1 | 2024-12-06 09:35:00 | 29196.45 | 2024-12-06 10:00:00 | 29196.45 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-12-11 10:30:00 | 29257.30 | 2024-12-11 10:35:00 | 29211.23 | STOP_HIT | 1.00 | -0.16% |
| SELL | retest1 | 2024-12-13 11:00:00 | 28500.00 | 2024-12-13 12:35:00 | 28403.09 | PARTIAL | 0.50 | 0.34% |
| SELL | retest1 | 2024-12-13 11:00:00 | 28500.00 | 2024-12-13 13:50:00 | 28500.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-12-16 10:55:00 | 28455.70 | 2024-12-16 11:55:00 | 28359.54 | PARTIAL | 0.50 | 0.34% |
| SELL | retest1 | 2024-12-16 10:55:00 | 28455.70 | 2024-12-16 15:20:00 | 28260.00 | TARGET_HIT | 0.50 | 0.69% |
| SELL | retest1 | 2024-12-17 10:55:00 | 28160.00 | 2024-12-17 12:25:00 | 28209.57 | STOP_HIT | 1.00 | -0.18% |
| BUY | retest1 | 2024-12-18 09:50:00 | 28331.30 | 2024-12-18 10:00:00 | 28252.00 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2024-12-19 10:25:00 | 28199.95 | 2024-12-19 10:40:00 | 28312.88 | PARTIAL | 0.50 | 0.40% |
| BUY | retest1 | 2024-12-19 10:25:00 | 28199.95 | 2024-12-19 15:20:00 | 28970.20 | TARGET_HIT | 0.50 | 2.73% |
| BUY | retest1 | 2024-12-20 10:40:00 | 29114.05 | 2024-12-20 10:50:00 | 29022.49 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2024-12-23 10:40:00 | 28721.05 | 2024-12-23 11:00:00 | 28843.39 | PARTIAL | 0.50 | 0.43% |
| BUY | retest1 | 2024-12-23 10:40:00 | 28721.05 | 2024-12-23 11:35:00 | 28721.05 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-12-31 11:05:00 | 29975.00 | 2024-12-31 12:35:00 | 29895.76 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2025-01-02 09:50:00 | 29793.65 | 2025-01-02 09:55:00 | 29852.08 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2025-01-03 09:30:00 | 29694.55 | 2025-01-03 09:40:00 | 29768.26 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2025-01-08 09:40:00 | 29809.15 | 2025-01-08 09:50:00 | 29888.87 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2025-01-09 11:05:00 | 29756.40 | 2025-01-09 12:15:00 | 29688.97 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2025-01-10 09:55:00 | 29475.15 | 2025-01-10 10:30:00 | 29360.51 | PARTIAL | 0.50 | 0.39% |
| SELL | retest1 | 2025-01-10 09:55:00 | 29475.15 | 2025-01-10 15:20:00 | 28817.60 | TARGET_HIT | 0.50 | 2.23% |
| SELL | retest1 | 2025-01-13 10:50:00 | 28303.55 | 2025-01-13 11:05:00 | 28379.48 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2025-01-16 10:35:00 | 27500.90 | 2025-01-16 11:20:00 | 27392.58 | PARTIAL | 0.50 | 0.39% |
| SELL | retest1 | 2025-01-16 10:35:00 | 27500.90 | 2025-01-16 15:20:00 | 27261.55 | TARGET_HIT | 0.50 | 0.87% |
| BUY | retest1 | 2025-01-17 09:45:00 | 27550.00 | 2025-01-17 10:00:00 | 27650.38 | PARTIAL | 0.50 | 0.36% |
| BUY | retest1 | 2025-01-17 09:45:00 | 27550.00 | 2025-01-17 15:20:00 | 27939.10 | TARGET_HIT | 0.50 | 1.41% |
| SELL | retest1 | 2025-01-22 09:50:00 | 27516.25 | 2025-01-22 10:40:00 | 27600.28 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2025-01-27 09:45:00 | 26985.05 | 2025-01-27 10:10:00 | 26859.09 | PARTIAL | 0.50 | 0.47% |
| SELL | retest1 | 2025-01-27 09:45:00 | 26985.05 | 2025-01-27 15:20:00 | 26513.80 | TARGET_HIT | 0.50 | 1.75% |
| BUY | retest1 | 2025-01-29 09:35:00 | 25625.00 | 2025-01-29 09:45:00 | 25515.80 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest1 | 2025-02-05 09:35:00 | 26498.00 | 2025-02-05 09:40:00 | 26407.06 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2025-02-11 10:50:00 | 29328.75 | 2025-02-11 11:00:00 | 29226.02 | PARTIAL | 0.50 | 0.35% |
| SELL | retest1 | 2025-02-11 10:50:00 | 29328.75 | 2025-02-11 11:05:00 | 29328.75 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-02-20 11:00:00 | 29712.10 | 2025-02-20 11:20:00 | 29652.24 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2025-02-24 10:20:00 | 28742.60 | 2025-02-24 10:25:00 | 28860.07 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest1 | 2025-02-27 10:05:00 | 30358.05 | 2025-02-27 10:10:00 | 30236.22 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest1 | 2025-03-11 10:20:00 | 30957.40 | 2025-03-11 10:35:00 | 30804.91 | STOP_HIT | 1.00 | -0.49% |
| SELL | retest1 | 2025-03-12 09:30:00 | 30270.00 | 2025-03-12 09:35:00 | 30357.96 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2025-03-13 10:50:00 | 30018.50 | 2025-03-13 11:10:00 | 29919.16 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2025-03-17 09:35:00 | 29601.85 | 2025-03-17 09:40:00 | 29689.69 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2025-03-19 10:30:00 | 30232.45 | 2025-03-19 11:00:00 | 30131.59 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2025-03-20 10:10:00 | 30510.00 | 2025-03-20 10:30:00 | 30664.91 | PARTIAL | 0.50 | 0.51% |
| BUY | retest1 | 2025-03-20 10:10:00 | 30510.00 | 2025-03-20 10:35:00 | 30510.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-03-24 10:45:00 | 30765.05 | 2025-03-24 10:50:00 | 30929.14 | PARTIAL | 0.50 | 0.53% |
| BUY | retest1 | 2025-03-24 10:45:00 | 30765.05 | 2025-03-24 10:55:00 | 30765.05 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-03-26 10:05:00 | 31155.00 | 2025-03-26 10:50:00 | 31037.83 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2025-04-01 11:05:00 | 30780.00 | 2025-04-01 11:10:00 | 30706.12 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2025-04-09 10:55:00 | 29544.30 | 2025-04-09 11:00:00 | 29452.98 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2025-04-21 10:45:00 | 31150.00 | 2025-04-21 15:00:00 | 31033.47 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest1 | 2025-04-23 10:35:00 | 30935.00 | 2025-04-23 11:25:00 | 31044.03 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2025-04-24 10:15:00 | 30305.00 | 2025-04-24 11:00:00 | 30197.47 | PARTIAL | 0.50 | 0.35% |
| SELL | retest1 | 2025-04-24 10:15:00 | 30305.00 | 2025-04-24 15:20:00 | 30070.00 | TARGET_HIT | 0.50 | 0.78% |
| SELL | retest1 | 2025-04-25 09:35:00 | 29715.00 | 2025-04-25 09:40:00 | 29581.71 | PARTIAL | 0.50 | 0.45% |
| SELL | retest1 | 2025-04-25 09:35:00 | 29715.00 | 2025-04-25 09:50:00 | 29715.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-05-02 11:05:00 | 29760.00 | 2025-05-02 11:15:00 | 29824.15 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2025-05-06 10:50:00 | 30325.00 | 2025-05-06 11:05:00 | 30406.68 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2025-05-09 10:50:00 | 29950.00 | 2025-05-09 10:55:00 | 29861.67 | STOP_HIT | 1.00 | -0.29% |
