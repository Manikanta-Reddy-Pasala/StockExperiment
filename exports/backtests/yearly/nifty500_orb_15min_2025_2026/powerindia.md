# Hitachi Energy India Ltd. (POWERINDIA)

## Backtest Summary

- **Window:** 2025-05-12 09:15:00 → 2026-05-08 15:25:00 (18463 bars)
- **Last close:** 33960.00
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
| ENTRY1 | 46 |
| ENTRY2 | 0 |
| PARTIAL | 19 |
| TARGET_HIT | 11 |
| STOP_HIT | 35 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 65 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 29 / 36
- **Target hits / Stop hits / Partials:** 11 / 35 / 19
- **Avg / median % per leg:** 0.20% / 0.00%
- **Sum % (uncompounded):** 13.19%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 40 | 15 | 37.5% | 6 | 24 | 10 | 0.09% | 3.5% |
| BUY @ 2nd Alert (retest1) | 40 | 15 | 37.5% | 6 | 24 | 10 | 0.09% | 3.5% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 25 | 14 | 56.0% | 5 | 11 | 9 | 0.39% | 9.6% |
| SELL @ 2nd Alert (retest1) | 25 | 14 | 56.0% | 5 | 11 | 9 | 0.39% | 9.6% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 65 | 29 | 44.6% | 11 | 35 | 19 | 0.20% | 13.2% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-23 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-23 09:50:00 | 17650.00 | 17464.06 | 0.00 | ORB-long ORB[17250.00,17500.00] vol=1.6x ATR=71.01 |
| Stop hit — per-position SL triggered | 2025-05-23 10:05:00 | 17578.99 | 17512.19 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2025-06-17 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-17 09:40:00 | 17973.00 | 17841.97 | 0.00 | ORB-long ORB[17646.00,17903.00] vol=2.1x ATR=66.07 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-17 09:55:00 | 18072.10 | 17894.59 | 0.00 | T1 1.5R @ 18072.10 |
| Stop hit — per-position SL triggered | 2025-06-17 10:00:00 | 17973.00 | 17901.16 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2025-07-04 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-04 09:30:00 | 20180.00 | 20037.20 | 0.00 | ORB-long ORB[19855.00,20145.00] vol=2.7x ATR=77.55 |
| Stop hit — per-position SL triggered | 2025-07-04 09:45:00 | 20102.45 | 20070.86 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2025-07-07 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-07 09:35:00 | 20075.00 | 19910.66 | 0.00 | ORB-long ORB[19720.00,19995.00] vol=1.9x ATR=74.03 |
| Stop hit — per-position SL triggered | 2025-07-07 09:40:00 | 20000.97 | 19920.19 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2025-07-16 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-16 09:35:00 | 18405.00 | 18491.00 | 0.00 | ORB-short ORB[18410.00,18585.00] vol=1.6x ATR=46.48 |
| Stop hit — per-position SL triggered | 2025-07-16 09:45:00 | 18451.48 | 18481.28 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2025-07-17 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-17 09:45:00 | 18940.00 | 18776.09 | 0.00 | ORB-long ORB[18605.00,18800.00] vol=3.4x ATR=77.52 |
| Stop hit — per-position SL triggered | 2025-07-17 09:55:00 | 18862.48 | 18798.19 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2025-07-24 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-24 11:10:00 | 20010.00 | 19967.69 | 0.00 | ORB-long ORB[19695.00,19995.00] vol=2.4x ATR=72.73 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-24 11:15:00 | 20119.09 | 19977.66 | 0.00 | T1 1.5R @ 20119.09 |
| Stop hit — per-position SL triggered | 2025-07-24 11:20:00 | 20010.00 | 19979.12 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2025-07-25 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-25 09:30:00 | 19620.00 | 19744.48 | 0.00 | ORB-short ORB[19625.00,19850.00] vol=1.8x ATR=74.80 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-25 09:40:00 | 19507.80 | 19675.75 | 0.00 | T1 1.5R @ 19507.80 |
| Target hit | 2025-07-25 15:20:00 | 19335.00 | 19404.16 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 9 — SELL (started 2025-08-14 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-14 10:05:00 | 20960.00 | 21204.56 | 0.00 | ORB-short ORB[21100.00,21400.00] vol=1.6x ATR=110.83 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-14 11:55:00 | 20793.76 | 21108.59 | 0.00 | T1 1.5R @ 20793.76 |
| Target hit | 2025-08-14 15:20:00 | 20350.00 | 20768.07 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 10 — SELL (started 2025-08-19 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-19 10:45:00 | 19545.00 | 19708.80 | 0.00 | ORB-short ORB[19720.00,19950.00] vol=1.8x ATR=117.79 |
| Stop hit — per-position SL triggered | 2025-08-19 10:55:00 | 19662.79 | 19700.86 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2025-09-03 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-03 11:00:00 | 19214.00 | 19098.73 | 0.00 | ORB-long ORB[18940.00,19200.00] vol=3.7x ATR=67.91 |
| Stop hit — per-position SL triggered | 2025-09-03 11:35:00 | 19146.09 | 19126.82 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2025-09-23 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-23 09:35:00 | 19452.00 | 19365.48 | 0.00 | ORB-long ORB[19206.00,19400.00] vol=1.6x ATR=100.17 |
| Stop hit — per-position SL triggered | 2025-09-23 09:45:00 | 19351.83 | 19360.22 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2025-09-26 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-26 09:55:00 | 19510.00 | 19361.42 | 0.00 | ORB-long ORB[19175.00,19445.00] vol=1.5x ATR=73.45 |
| Stop hit — per-position SL triggered | 2025-09-26 10:15:00 | 19436.55 | 19401.45 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2025-10-07 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-07 10:55:00 | 18355.00 | 18242.64 | 0.00 | ORB-long ORB[18123.00,18341.00] vol=3.2x ATR=58.28 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-07 11:00:00 | 18442.42 | 18266.48 | 0.00 | T1 1.5R @ 18442.42 |
| Stop hit — per-position SL triggered | 2025-10-07 11:10:00 | 18355.00 | 18273.29 | 0.00 | SL hit |

### Cycle 15 — SELL (started 2025-10-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-14 11:15:00 | 17505.00 | 17767.08 | 0.00 | ORB-short ORB[17815.00,17974.00] vol=2.3x ATR=55.57 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-14 11:30:00 | 17421.65 | 17733.68 | 0.00 | T1 1.5R @ 17421.65 |
| Stop hit — per-position SL triggered | 2025-10-14 11:45:00 | 17505.00 | 17714.20 | 0.00 | SL hit |

### Cycle 16 — SELL (started 2025-10-16 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-16 10:30:00 | 17423.00 | 17492.75 | 0.00 | ORB-short ORB[17442.00,17580.00] vol=1.8x ATR=43.52 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-16 11:25:00 | 17357.72 | 17465.91 | 0.00 | T1 1.5R @ 17357.72 |
| Stop hit — per-position SL triggered | 2025-10-16 12:25:00 | 17423.00 | 17451.61 | 0.00 | SL hit |

### Cycle 17 — SELL (started 2025-10-17 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-17 11:05:00 | 17640.00 | 17677.39 | 0.00 | ORB-short ORB[17661.00,17785.00] vol=3.2x ATR=45.56 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-17 11:20:00 | 17571.65 | 17665.32 | 0.00 | T1 1.5R @ 17571.65 |
| Target hit | 2025-10-17 15:20:00 | 17413.00 | 17546.80 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 18 — SELL (started 2025-10-27 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-27 11:10:00 | 16656.00 | 16819.37 | 0.00 | ORB-short ORB[16767.00,16913.00] vol=2.7x ATR=39.03 |
| Stop hit — per-position SL triggered | 2025-10-27 11:15:00 | 16695.03 | 16810.20 | 0.00 | SL hit |

### Cycle 19 — BUY (started 2025-10-28 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-28 09:30:00 | 16920.00 | 16812.53 | 0.00 | ORB-long ORB[16614.00,16772.00] vol=3.5x ATR=63.86 |
| Stop hit — per-position SL triggered | 2025-10-28 09:35:00 | 16856.14 | 16815.05 | 0.00 | SL hit |

### Cycle 20 — BUY (started 2025-10-29 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-29 09:45:00 | 17200.00 | 17071.42 | 0.00 | ORB-long ORB[16860.00,17080.00] vol=4.0x ATR=65.12 |
| Stop hit — per-position SL triggered | 2025-10-29 11:25:00 | 17134.88 | 17140.64 | 0.00 | SL hit |

### Cycle 21 — BUY (started 2025-10-30 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-30 09:35:00 | 17689.00 | 17559.45 | 0.00 | ORB-long ORB[17350.00,17582.00] vol=2.0x ATR=61.02 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-30 09:55:00 | 17780.53 | 17641.99 | 0.00 | T1 1.5R @ 17780.53 |
| Target hit | 2025-10-30 15:20:00 | 17914.00 | 17882.12 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 22 — BUY (started 2025-10-31 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-31 10:35:00 | 18091.00 | 18045.92 | 0.00 | ORB-long ORB[17830.00,18080.00] vol=3.2x ATR=47.45 |
| Stop hit — per-position SL triggered | 2025-10-31 10:45:00 | 18043.55 | 18051.65 | 0.00 | SL hit |

### Cycle 23 — BUY (started 2025-11-11 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-11 09:55:00 | 21657.00 | 21459.92 | 0.00 | ORB-long ORB[21255.00,21550.00] vol=1.6x ATR=80.65 |
| Stop hit — per-position SL triggered | 2025-11-11 10:10:00 | 21576.35 | 21503.36 | 0.00 | SL hit |

### Cycle 24 — BUY (started 2025-11-12 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-12 10:25:00 | 22025.00 | 21792.58 | 0.00 | ORB-long ORB[21671.00,21970.00] vol=1.8x ATR=87.58 |
| Stop hit — per-position SL triggered | 2025-11-12 10:55:00 | 21937.42 | 21828.08 | 0.00 | SL hit |

### Cycle 25 — SELL (started 2025-11-13 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-13 10:10:00 | 21515.00 | 21753.54 | 0.00 | ORB-short ORB[21701.00,21992.00] vol=1.6x ATR=73.78 |
| Stop hit — per-position SL triggered | 2025-11-13 10:15:00 | 21588.78 | 21740.07 | 0.00 | SL hit |

### Cycle 26 — BUY (started 2025-11-17 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-17 09:30:00 | 21698.00 | 21603.79 | 0.00 | ORB-long ORB[21416.00,21687.00] vol=1.7x ATR=55.65 |
| Stop hit — per-position SL triggered | 2025-11-17 09:40:00 | 21642.35 | 21624.09 | 0.00 | SL hit |

### Cycle 27 — BUY (started 2025-11-20 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-20 09:30:00 | 21979.00 | 21874.50 | 0.00 | ORB-long ORB[21690.00,21930.00] vol=2.4x ATR=71.95 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-20 09:40:00 | 22086.93 | 21959.06 | 0.00 | T1 1.5R @ 22086.93 |
| Target hit | 2025-11-20 15:20:00 | 22415.00 | 22269.18 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 28 — SELL (started 2025-12-10 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-10 10:55:00 | 19140.00 | 19380.05 | 0.00 | ORB-short ORB[19380.00,19655.00] vol=1.9x ATR=72.12 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-10 13:45:00 | 19031.82 | 19294.53 | 0.00 | T1 1.5R @ 19031.82 |
| Target hit | 2025-12-10 15:20:00 | 18960.00 | 19175.40 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 29 — BUY (started 2025-12-11 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-11 10:20:00 | 19070.00 | 18901.81 | 0.00 | ORB-long ORB[18785.00,18975.00] vol=2.2x ATR=89.35 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-11 10:45:00 | 19204.02 | 18979.38 | 0.00 | T1 1.5R @ 19204.02 |
| Stop hit — per-position SL triggered | 2025-12-11 10:50:00 | 19070.00 | 18984.94 | 0.00 | SL hit |

### Cycle 30 — BUY (started 2025-12-12 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-12 09:30:00 | 19460.00 | 19391.47 | 0.00 | ORB-long ORB[19285.00,19420.00] vol=2.2x ATR=70.17 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-12 10:20:00 | 19565.25 | 19425.76 | 0.00 | T1 1.5R @ 19565.25 |
| Target hit | 2025-12-12 10:20:00 | 19405.00 | 19425.76 | 0.00 | Trail-exit close<VWAP |

### Cycle 31 — SELL (started 2025-12-16 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-16 11:10:00 | 19335.00 | 19433.39 | 0.00 | ORB-short ORB[19375.00,19535.00] vol=1.7x ATR=44.36 |
| Stop hit — per-position SL triggered | 2025-12-16 12:20:00 | 19379.36 | 19413.62 | 0.00 | SL hit |

### Cycle 32 — SELL (started 2025-12-30 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-30 10:40:00 | 18155.00 | 18249.38 | 0.00 | ORB-short ORB[18235.00,18385.00] vol=1.5x ATR=47.82 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-30 10:50:00 | 18083.27 | 18228.86 | 0.00 | T1 1.5R @ 18083.27 |
| Target hit | 2025-12-30 13:50:00 | 18010.00 | 17997.64 | 0.00 | Trail-exit close>VWAP |

### Cycle 33 — BUY (started 2026-01-02 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-02 09:30:00 | 18734.00 | 18625.79 | 0.00 | ORB-long ORB[18541.00,18658.00] vol=1.8x ATR=57.07 |
| Stop hit — per-position SL triggered | 2026-01-02 09:35:00 | 18676.93 | 18633.44 | 0.00 | SL hit |

### Cycle 34 — SELL (started 2026-01-06 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-06 10:55:00 | 18851.00 | 18957.27 | 0.00 | ORB-short ORB[18868.00,19100.00] vol=2.3x ATR=51.41 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-06 11:15:00 | 18773.88 | 18939.69 | 0.00 | T1 1.5R @ 18773.88 |
| Stop hit — per-position SL triggered | 2026-01-06 11:45:00 | 18851.00 | 18923.43 | 0.00 | SL hit |

### Cycle 35 — SELL (started 2026-01-16 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-16 09:30:00 | 16686.00 | 16804.44 | 0.00 | ORB-short ORB[16737.00,16953.00] vol=2.3x ATR=73.24 |
| Stop hit — per-position SL triggered | 2026-01-16 09:55:00 | 16759.24 | 16748.27 | 0.00 | SL hit |

### Cycle 36 — BUY (started 2026-02-12 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-12 09:30:00 | 22908.00 | 22762.07 | 0.00 | ORB-long ORB[22600.00,22823.00] vol=1.5x ATR=75.21 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-12 09:40:00 | 23020.82 | 22867.18 | 0.00 | T1 1.5R @ 23020.82 |
| Target hit | 2026-02-12 10:05:00 | 22930.00 | 22938.15 | 0.00 | Trail-exit close<VWAP |

### Cycle 37 — BUY (started 2026-02-16 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-16 09:30:00 | 22749.00 | 22598.08 | 0.00 | ORB-long ORB[22350.00,22670.00] vol=1.9x ATR=81.30 |
| Stop hit — per-position SL triggered | 2026-02-16 09:40:00 | 22667.70 | 22616.95 | 0.00 | SL hit |

### Cycle 38 — BUY (started 2026-02-23 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-23 09:40:00 | 24259.00 | 24116.49 | 0.00 | ORB-long ORB[23827.00,24170.00] vol=1.8x ATR=96.29 |
| Stop hit — per-position SL triggered | 2026-02-23 10:00:00 | 24162.71 | 24170.72 | 0.00 | SL hit |

### Cycle 39 — BUY (started 2026-02-24 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-24 10:30:00 | 24472.00 | 24222.93 | 0.00 | ORB-long ORB[24000.00,24245.00] vol=2.3x ATR=73.39 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-24 11:30:00 | 24582.08 | 24315.00 | 0.00 | T1 1.5R @ 24582.08 |
| Target hit | 2026-02-24 15:20:00 | 24951.00 | 24604.00 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 40 — BUY (started 2026-02-25 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-25 09:35:00 | 25240.00 | 25056.47 | 0.00 | ORB-long ORB[24881.00,25179.00] vol=1.6x ATR=101.31 |
| Stop hit — per-position SL triggered | 2026-02-25 09:50:00 | 25138.69 | 25088.71 | 0.00 | SL hit |

### Cycle 41 — SELL (started 2026-04-08 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-08 09:35:00 | 25550.00 | 25723.27 | 0.00 | ORB-short ORB[25600.00,25940.00] vol=1.7x ATR=128.68 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-08 09:50:00 | 25356.98 | 25659.12 | 0.00 | T1 1.5R @ 25356.98 |
| Stop hit — per-position SL triggered | 2026-04-08 10:15:00 | 25550.00 | 25569.92 | 0.00 | SL hit |

### Cycle 42 — BUY (started 2026-04-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-17 11:15:00 | 28710.00 | 28559.54 | 0.00 | ORB-long ORB[28180.00,28500.00] vol=1.6x ATR=81.23 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-17 14:30:00 | 28831.84 | 28613.62 | 0.00 | T1 1.5R @ 28831.84 |
| Target hit | 2026-04-17 15:20:00 | 28930.00 | 28683.51 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 43 — SELL (started 2026-04-27 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-27 10:40:00 | 31965.00 | 32243.67 | 0.00 | ORB-short ORB[32085.00,32550.00] vol=3.2x ATR=117.65 |
| Stop hit — per-position SL triggered | 2026-04-27 11:40:00 | 32082.65 | 32163.97 | 0.00 | SL hit |

### Cycle 44 — BUY (started 2026-04-29 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-29 10:25:00 | 32990.00 | 32789.67 | 0.00 | ORB-long ORB[32655.00,32925.00] vol=3.4x ATR=117.25 |
| Stop hit — per-position SL triggered | 2026-04-29 10:30:00 | 32872.75 | 32805.11 | 0.00 | SL hit |

### Cycle 45 — BUY (started 2026-04-30 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-30 10:00:00 | 33210.00 | 32908.86 | 0.00 | ORB-long ORB[32530.00,33025.00] vol=1.5x ATR=134.74 |
| Stop hit — per-position SL triggered | 2026-04-30 10:10:00 | 33075.26 | 32943.36 | 0.00 | SL hit |

### Cycle 46 — BUY (started 2026-05-05 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-05 09:30:00 | 34170.00 | 34005.00 | 0.00 | ORB-long ORB[33720.00,34090.00] vol=1.9x ATR=125.23 |
| Stop hit — per-position SL triggered | 2026-05-05 09:50:00 | 34044.77 | 34047.62 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-05-23 09:50:00 | 17650.00 | 2025-05-23 10:05:00 | 17578.99 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest1 | 2025-06-17 09:40:00 | 17973.00 | 2025-06-17 09:55:00 | 18072.10 | PARTIAL | 0.50 | 0.55% |
| BUY | retest1 | 2025-06-17 09:40:00 | 17973.00 | 2025-06-17 10:00:00 | 17973.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-07-04 09:30:00 | 20180.00 | 2025-07-04 09:45:00 | 20102.45 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2025-07-07 09:35:00 | 20075.00 | 2025-07-07 09:40:00 | 20000.97 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest1 | 2025-07-16 09:35:00 | 18405.00 | 2025-07-16 09:45:00 | 18451.48 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2025-07-17 09:45:00 | 18940.00 | 2025-07-17 09:55:00 | 18862.48 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest1 | 2025-07-24 11:10:00 | 20010.00 | 2025-07-24 11:15:00 | 20119.09 | PARTIAL | 0.50 | 0.55% |
| BUY | retest1 | 2025-07-24 11:10:00 | 20010.00 | 2025-07-24 11:20:00 | 20010.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-07-25 09:30:00 | 19620.00 | 2025-07-25 09:40:00 | 19507.80 | PARTIAL | 0.50 | 0.57% |
| SELL | retest1 | 2025-07-25 09:30:00 | 19620.00 | 2025-07-25 15:20:00 | 19335.00 | TARGET_HIT | 0.50 | 1.45% |
| SELL | retest1 | 2025-08-14 10:05:00 | 20960.00 | 2025-08-14 11:55:00 | 20793.76 | PARTIAL | 0.50 | 0.79% |
| SELL | retest1 | 2025-08-14 10:05:00 | 20960.00 | 2025-08-14 15:20:00 | 20350.00 | TARGET_HIT | 0.50 | 2.91% |
| SELL | retest1 | 2025-08-19 10:45:00 | 19545.00 | 2025-08-19 10:55:00 | 19662.79 | STOP_HIT | 1.00 | -0.60% |
| BUY | retest1 | 2025-09-03 11:00:00 | 19214.00 | 2025-09-03 11:35:00 | 19146.09 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2025-09-23 09:35:00 | 19452.00 | 2025-09-23 09:45:00 | 19351.83 | STOP_HIT | 1.00 | -0.51% |
| BUY | retest1 | 2025-09-26 09:55:00 | 19510.00 | 2025-09-26 10:15:00 | 19436.55 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2025-10-07 10:55:00 | 18355.00 | 2025-10-07 11:00:00 | 18442.42 | PARTIAL | 0.50 | 0.48% |
| BUY | retest1 | 2025-10-07 10:55:00 | 18355.00 | 2025-10-07 11:10:00 | 18355.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-10-14 11:15:00 | 17505.00 | 2025-10-14 11:30:00 | 17421.65 | PARTIAL | 0.50 | 0.48% |
| SELL | retest1 | 2025-10-14 11:15:00 | 17505.00 | 2025-10-14 11:45:00 | 17505.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-10-16 10:30:00 | 17423.00 | 2025-10-16 11:25:00 | 17357.72 | PARTIAL | 0.50 | 0.37% |
| SELL | retest1 | 2025-10-16 10:30:00 | 17423.00 | 2025-10-16 12:25:00 | 17423.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-10-17 11:05:00 | 17640.00 | 2025-10-17 11:20:00 | 17571.65 | PARTIAL | 0.50 | 0.39% |
| SELL | retest1 | 2025-10-17 11:05:00 | 17640.00 | 2025-10-17 15:20:00 | 17413.00 | TARGET_HIT | 0.50 | 1.29% |
| SELL | retest1 | 2025-10-27 11:10:00 | 16656.00 | 2025-10-27 11:15:00 | 16695.03 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2025-10-28 09:30:00 | 16920.00 | 2025-10-28 09:35:00 | 16856.14 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2025-10-29 09:45:00 | 17200.00 | 2025-10-29 11:25:00 | 17134.88 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2025-10-30 09:35:00 | 17689.00 | 2025-10-30 09:55:00 | 17780.53 | PARTIAL | 0.50 | 0.52% |
| BUY | retest1 | 2025-10-30 09:35:00 | 17689.00 | 2025-10-30 15:20:00 | 17914.00 | TARGET_HIT | 0.50 | 1.27% |
| BUY | retest1 | 2025-10-31 10:35:00 | 18091.00 | 2025-10-31 10:45:00 | 18043.55 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2025-11-11 09:55:00 | 21657.00 | 2025-11-11 10:10:00 | 21576.35 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2025-11-12 10:25:00 | 22025.00 | 2025-11-12 10:55:00 | 21937.42 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest1 | 2025-11-13 10:10:00 | 21515.00 | 2025-11-13 10:15:00 | 21588.78 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2025-11-17 09:30:00 | 21698.00 | 2025-11-17 09:40:00 | 21642.35 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2025-11-20 09:30:00 | 21979.00 | 2025-11-20 09:40:00 | 22086.93 | PARTIAL | 0.50 | 0.49% |
| BUY | retest1 | 2025-11-20 09:30:00 | 21979.00 | 2025-11-20 15:20:00 | 22415.00 | TARGET_HIT | 0.50 | 1.98% |
| SELL | retest1 | 2025-12-10 10:55:00 | 19140.00 | 2025-12-10 13:45:00 | 19031.82 | PARTIAL | 0.50 | 0.57% |
| SELL | retest1 | 2025-12-10 10:55:00 | 19140.00 | 2025-12-10 15:20:00 | 18960.00 | TARGET_HIT | 0.50 | 0.94% |
| BUY | retest1 | 2025-12-11 10:20:00 | 19070.00 | 2025-12-11 10:45:00 | 19204.02 | PARTIAL | 0.50 | 0.70% |
| BUY | retest1 | 2025-12-11 10:20:00 | 19070.00 | 2025-12-11 10:50:00 | 19070.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-12-12 09:30:00 | 19460.00 | 2025-12-12 10:20:00 | 19565.25 | PARTIAL | 0.50 | 0.54% |
| BUY | retest1 | 2025-12-12 09:30:00 | 19460.00 | 2025-12-12 10:20:00 | 19405.00 | TARGET_HIT | 0.50 | -0.28% |
| SELL | retest1 | 2025-12-16 11:10:00 | 19335.00 | 2025-12-16 12:20:00 | 19379.36 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2025-12-30 10:40:00 | 18155.00 | 2025-12-30 10:50:00 | 18083.27 | PARTIAL | 0.50 | 0.40% |
| SELL | retest1 | 2025-12-30 10:40:00 | 18155.00 | 2025-12-30 13:50:00 | 18010.00 | TARGET_HIT | 0.50 | 0.80% |
| BUY | retest1 | 2026-01-02 09:30:00 | 18734.00 | 2026-01-02 09:35:00 | 18676.93 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2026-01-06 10:55:00 | 18851.00 | 2026-01-06 11:15:00 | 18773.88 | PARTIAL | 0.50 | 0.41% |
| SELL | retest1 | 2026-01-06 10:55:00 | 18851.00 | 2026-01-06 11:45:00 | 18851.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-01-16 09:30:00 | 16686.00 | 2026-01-16 09:55:00 | 16759.24 | STOP_HIT | 1.00 | -0.44% |
| BUY | retest1 | 2026-02-12 09:30:00 | 22908.00 | 2026-02-12 09:40:00 | 23020.82 | PARTIAL | 0.50 | 0.49% |
| BUY | retest1 | 2026-02-12 09:30:00 | 22908.00 | 2026-02-12 10:05:00 | 22930.00 | TARGET_HIT | 0.50 | 0.10% |
| BUY | retest1 | 2026-02-16 09:30:00 | 22749.00 | 2026-02-16 09:40:00 | 22667.70 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2026-02-23 09:40:00 | 24259.00 | 2026-02-23 10:00:00 | 24162.71 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest1 | 2026-02-24 10:30:00 | 24472.00 | 2026-02-24 11:30:00 | 24582.08 | PARTIAL | 0.50 | 0.45% |
| BUY | retest1 | 2026-02-24 10:30:00 | 24472.00 | 2026-02-24 15:20:00 | 24951.00 | TARGET_HIT | 0.50 | 1.96% |
| BUY | retest1 | 2026-02-25 09:35:00 | 25240.00 | 2026-02-25 09:50:00 | 25138.69 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest1 | 2026-04-08 09:35:00 | 25550.00 | 2026-04-08 09:50:00 | 25356.98 | PARTIAL | 0.50 | 0.76% |
| SELL | retest1 | 2026-04-08 09:35:00 | 25550.00 | 2026-04-08 10:15:00 | 25550.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-17 11:15:00 | 28710.00 | 2026-04-17 14:30:00 | 28831.84 | PARTIAL | 0.50 | 0.42% |
| BUY | retest1 | 2026-04-17 11:15:00 | 28710.00 | 2026-04-17 15:20:00 | 28930.00 | TARGET_HIT | 0.50 | 0.77% |
| SELL | retest1 | 2026-04-27 10:40:00 | 31965.00 | 2026-04-27 11:40:00 | 32082.65 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2026-04-29 10:25:00 | 32990.00 | 2026-04-29 10:30:00 | 32872.75 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2026-04-30 10:00:00 | 33210.00 | 2026-04-30 10:10:00 | 33075.26 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest1 | 2026-05-05 09:30:00 | 34170.00 | 2026-05-05 09:50:00 | 34044.77 | STOP_HIT | 1.00 | -0.37% |
