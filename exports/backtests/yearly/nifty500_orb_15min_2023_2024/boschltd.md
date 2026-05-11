# Bosch Ltd. (BOSCHLTD)

## Backtest Summary

- **Window:** 2023-08-10 09:15:00 → 2026-05-08 15:25:00 (49117 bars)
- **Last close:** 38050.00
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
| ENTRY1 | 72 |
| ENTRY2 | 0 |
| PARTIAL | 22 |
| TARGET_HIT | 12 |
| STOP_HIT | 60 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 94 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 34 / 60
- **Target hits / Stop hits / Partials:** 12 / 60 / 22
- **Avg / median % per leg:** 0.07% / -0.16%
- **Sum % (uncompounded):** 6.77%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 59 | 24 | 40.7% | 8 | 35 | 16 | 0.11% | 6.6% |
| BUY @ 2nd Alert (retest1) | 59 | 24 | 40.7% | 8 | 35 | 16 | 0.11% | 6.6% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 35 | 10 | 28.6% | 4 | 25 | 6 | 0.00% | 0.1% |
| SELL @ 2nd Alert (retest1) | 35 | 10 | 28.6% | 4 | 25 | 6 | 0.00% | 0.1% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 94 | 34 | 36.2% | 12 | 60 | 22 | 0.07% | 6.8% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-08-11 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-11 10:35:00 | 18284.80 | 18190.23 | 0.00 | ORB-long ORB[18140.00,18229.90] vol=1.7x ATR=38.30 |
| Stop hit — per-position SL triggered | 2023-08-11 10:40:00 | 18246.50 | 18191.89 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2023-08-17 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-17 10:35:00 | 18260.00 | 18214.08 | 0.00 | ORB-long ORB[18162.20,18244.90] vol=2.5x ATR=27.63 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-17 11:10:00 | 18301.45 | 18244.19 | 0.00 | T1 1.5R @ 18301.45 |
| Stop hit — per-position SL triggered | 2023-08-17 11:20:00 | 18260.00 | 18248.56 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2023-08-24 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-24 09:35:00 | 18349.00 | 18386.26 | 0.00 | ORB-short ORB[18372.00,18470.00] vol=1.9x ATR=30.39 |
| Stop hit — per-position SL triggered | 2023-08-24 09:45:00 | 18379.39 | 18383.72 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2023-08-29 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-29 11:00:00 | 18500.00 | 18525.45 | 0.00 | ORB-short ORB[18510.00,18569.90] vol=4.1x ATR=15.65 |
| Stop hit — per-position SL triggered | 2023-08-29 11:25:00 | 18515.65 | 18520.69 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2023-09-04 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-04 10:00:00 | 19124.90 | 19069.54 | 0.00 | ORB-long ORB[18950.00,19079.00] vol=2.1x ATR=38.14 |
| Stop hit — per-position SL triggered | 2023-09-04 10:10:00 | 19086.76 | 19073.16 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2023-09-05 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-05 10:30:00 | 19173.20 | 19118.83 | 0.00 | ORB-long ORB[18999.40,19143.30] vol=3.9x ATR=31.17 |
| Stop hit — per-position SL triggered | 2023-09-05 10:35:00 | 19142.03 | 19120.07 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2023-09-07 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-07 09:50:00 | 19318.00 | 19243.97 | 0.00 | ORB-long ORB[19191.30,19299.90] vol=1.6x ATR=35.76 |
| Stop hit — per-position SL triggered | 2023-09-07 10:00:00 | 19282.24 | 19250.21 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2023-09-11 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-09-11 10:10:00 | 19285.20 | 19408.29 | 0.00 | ORB-short ORB[19404.10,19499.90] vol=1.7x ATR=30.79 |
| Stop hit — per-position SL triggered | 2023-09-11 10:20:00 | 19315.99 | 19405.07 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2023-09-14 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-14 09:45:00 | 19600.00 | 19516.30 | 0.00 | ORB-long ORB[19393.60,19550.00] vol=2.2x ATR=39.41 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-14 09:50:00 | 19659.12 | 19575.90 | 0.00 | T1 1.5R @ 19659.12 |
| Stop hit — per-position SL triggered | 2023-09-14 10:05:00 | 19600.00 | 19586.31 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2023-09-20 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-09-20 10:50:00 | 19401.50 | 19516.08 | 0.00 | ORB-short ORB[19450.90,19624.90] vol=1.7x ATR=39.07 |
| Stop hit — per-position SL triggered | 2023-09-20 11:15:00 | 19440.57 | 19492.06 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2023-09-26 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-09-26 09:40:00 | 19038.60 | 19088.82 | 0.00 | ORB-short ORB[19074.80,19170.00] vol=1.7x ATR=34.90 |
| Stop hit — per-position SL triggered | 2023-09-26 09:45:00 | 19073.50 | 19086.28 | 0.00 | SL hit |

### Cycle 12 — SELL (started 2023-10-03 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-10-03 10:00:00 | 18692.50 | 18816.55 | 0.00 | ORB-short ORB[18789.00,19060.00] vol=1.7x ATR=60.37 |
| Stop hit — per-position SL triggered | 2023-10-03 10:55:00 | 18752.87 | 18778.39 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2023-10-04 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-04 10:35:00 | 18890.00 | 18740.62 | 0.00 | ORB-long ORB[18650.00,18762.80] vol=2.0x ATR=54.16 |
| Stop hit — per-position SL triggered | 2023-10-04 10:40:00 | 18835.84 | 18748.15 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2023-10-06 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-06 09:30:00 | 19010.00 | 18958.26 | 0.00 | ORB-long ORB[18900.00,18970.00] vol=2.8x ATR=31.89 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-06 09:45:00 | 19057.84 | 18989.13 | 0.00 | T1 1.5R @ 19057.84 |
| Stop hit — per-position SL triggered | 2023-10-06 09:50:00 | 19010.00 | 19014.71 | 0.00 | SL hit |

### Cycle 15 — BUY (started 2023-10-09 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-09 09:45:00 | 19273.60 | 19170.95 | 0.00 | ORB-long ORB[19051.30,19225.80] vol=1.9x ATR=51.28 |
| Stop hit — per-position SL triggered | 2023-10-09 10:00:00 | 19222.32 | 19179.07 | 0.00 | SL hit |

### Cycle 16 — BUY (started 2023-10-10 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-10 11:05:00 | 19414.90 | 19363.67 | 0.00 | ORB-long ORB[19205.20,19406.60] vol=2.2x ATR=37.01 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-10 11:20:00 | 19470.42 | 19405.04 | 0.00 | T1 1.5R @ 19470.42 |
| Target hit | 2023-10-10 15:20:00 | 19700.20 | 19541.55 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 17 — BUY (started 2023-10-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-11 11:15:00 | 19888.80 | 19809.09 | 0.00 | ORB-long ORB[19651.10,19871.80] vol=1.5x ATR=39.38 |
| Stop hit — per-position SL triggered | 2023-10-11 11:20:00 | 19849.42 | 19815.95 | 0.00 | SL hit |

### Cycle 18 — BUY (started 2023-10-12 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-12 09:40:00 | 19900.80 | 19807.10 | 0.00 | ORB-long ORB[19601.20,19797.90] vol=3.1x ATR=39.34 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-12 09:45:00 | 19959.81 | 19873.30 | 0.00 | T1 1.5R @ 19959.81 |
| Target hit | 2023-10-12 15:20:00 | 20448.60 | 20285.94 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 19 — BUY (started 2023-10-13 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-13 11:00:00 | 20518.60 | 20378.72 | 0.00 | ORB-long ORB[20215.10,20488.00] vol=4.4x ATR=54.37 |
| Stop hit — per-position SL triggered | 2023-10-13 11:05:00 | 20464.23 | 20382.57 | 0.00 | SL hit |

### Cycle 20 — SELL (started 2023-10-18 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-10-18 10:45:00 | 20345.20 | 20402.80 | 0.00 | ORB-short ORB[20351.10,20499.90] vol=2.2x ATR=37.63 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-18 11:05:00 | 20288.76 | 20388.06 | 0.00 | T1 1.5R @ 20288.76 |
| Target hit | 2023-10-18 15:20:00 | 20104.90 | 20209.61 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 21 — BUY (started 2023-10-19 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-19 10:50:00 | 20160.00 | 20041.74 | 0.00 | ORB-long ORB[19960.60,20055.10] vol=5.7x ATR=38.79 |
| Stop hit — per-position SL triggered | 2023-10-19 11:05:00 | 20121.21 | 20053.59 | 0.00 | SL hit |

### Cycle 22 — BUY (started 2023-10-27 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-27 09:30:00 | 19941.20 | 19888.75 | 0.00 | ORB-long ORB[19739.00,19924.90] vol=2.1x ATR=82.31 |
| Stop hit — per-position SL triggered | 2023-10-27 09:45:00 | 19858.89 | 19893.57 | 0.00 | SL hit |

### Cycle 23 — SELL (started 2023-11-03 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-11-03 09:55:00 | 19463.90 | 19515.56 | 0.00 | ORB-short ORB[19505.10,19576.80] vol=1.8x ATR=38.02 |
| Stop hit — per-position SL triggered | 2023-11-03 10:00:00 | 19501.92 | 19513.88 | 0.00 | SL hit |

### Cycle 24 — SELL (started 2023-12-08 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-12-08 11:05:00 | 21570.00 | 21620.95 | 0.00 | ORB-short ORB[21588.20,21746.70] vol=2.0x ATR=76.31 |
| Stop hit — per-position SL triggered | 2023-12-08 14:40:00 | 21646.31 | 21586.89 | 0.00 | SL hit |

### Cycle 25 — BUY (started 2023-12-11 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-11 09:45:00 | 21722.50 | 21692.24 | 0.00 | ORB-long ORB[21585.90,21717.20] vol=1.6x ATR=50.69 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-11 09:55:00 | 21798.53 | 21718.56 | 0.00 | T1 1.5R @ 21798.53 |
| Target hit | 2023-12-11 10:45:00 | 21750.00 | 21750.38 | 0.00 | Trail-exit close<VWAP |

### Cycle 26 — BUY (started 2023-12-12 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-12 10:40:00 | 22056.20 | 21997.15 | 0.00 | ORB-long ORB[21901.00,22046.30] vol=2.5x ATR=46.54 |
| Stop hit — per-position SL triggered | 2023-12-12 11:00:00 | 22009.66 | 22003.80 | 0.00 | SL hit |

### Cycle 27 — SELL (started 2023-12-15 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-12-15 10:50:00 | 21952.90 | 22082.95 | 0.00 | ORB-short ORB[22029.40,22151.30] vol=1.6x ATR=42.82 |
| Stop hit — per-position SL triggered | 2023-12-15 11:05:00 | 21995.72 | 22076.05 | 0.00 | SL hit |

### Cycle 28 — BUY (started 2023-12-18 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-18 11:05:00 | 21989.00 | 21894.24 | 0.00 | ORB-long ORB[21715.00,21890.00] vol=2.3x ATR=40.53 |
| Stop hit — per-position SL triggered | 2023-12-18 11:15:00 | 21948.47 | 21897.11 | 0.00 | SL hit |

### Cycle 29 — SELL (started 2023-12-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-12-19 11:15:00 | 21970.10 | 22032.84 | 0.00 | ORB-short ORB[21977.40,22127.90] vol=1.6x ATR=44.12 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-19 13:55:00 | 21903.92 | 22000.91 | 0.00 | T1 1.5R @ 21903.92 |
| Target hit | 2023-12-19 15:20:00 | 21949.50 | 21965.49 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 30 — SELL (started 2023-12-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-12-20 11:15:00 | 21886.60 | 21971.67 | 0.00 | ORB-short ORB[21970.00,22077.70] vol=9.4x ATR=29.39 |
| Stop hit — per-position SL triggered | 2023-12-20 11:30:00 | 21915.99 | 21965.56 | 0.00 | SL hit |

### Cycle 31 — BUY (started 2023-12-21 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-21 11:00:00 | 21630.00 | 21474.45 | 0.00 | ORB-long ORB[21348.30,21581.90] vol=3.0x ATR=70.97 |
| Stop hit — per-position SL triggered | 2023-12-21 11:05:00 | 21559.03 | 21474.98 | 0.00 | SL hit |

### Cycle 32 — BUY (started 2023-12-22 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-22 09:45:00 | 21828.10 | 21756.61 | 0.00 | ORB-long ORB[21630.00,21794.80] vol=1.8x ATR=55.56 |
| Stop hit — per-position SL triggered | 2023-12-22 10:30:00 | 21772.54 | 21790.62 | 0.00 | SL hit |

### Cycle 33 — SELL (started 2023-12-28 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-12-28 10:55:00 | 21980.00 | 22009.31 | 0.00 | ORB-short ORB[22008.20,22150.00] vol=2.0x ATR=34.81 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-28 11:40:00 | 21927.79 | 22000.28 | 0.00 | T1 1.5R @ 21927.79 |
| Stop hit — per-position SL triggered | 2023-12-28 15:00:00 | 21980.00 | 21986.18 | 0.00 | SL hit |

### Cycle 34 — BUY (started 2024-01-01 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-01 09:40:00 | 22473.00 | 22364.48 | 0.00 | ORB-long ORB[22209.20,22373.80] vol=1.6x ATR=62.63 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-01 09:45:00 | 22566.95 | 22409.24 | 0.00 | T1 1.5R @ 22566.95 |
| Stop hit — per-position SL triggered | 2024-01-01 10:00:00 | 22473.00 | 22426.35 | 0.00 | SL hit |

### Cycle 35 — SELL (started 2024-01-02 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-02 10:50:00 | 22150.00 | 22298.61 | 0.00 | ORB-short ORB[22312.70,22469.50] vol=1.8x ATR=55.45 |
| Stop hit — per-position SL triggered | 2024-01-02 12:30:00 | 22205.45 | 22254.28 | 0.00 | SL hit |

### Cycle 36 — SELL (started 2024-01-04 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-04 11:05:00 | 22225.60 | 22389.37 | 0.00 | ORB-short ORB[22289.20,22550.00] vol=1.7x ATR=47.44 |
| Stop hit — per-position SL triggered | 2024-01-04 11:40:00 | 22273.04 | 22350.57 | 0.00 | SL hit |

### Cycle 37 — BUY (started 2024-01-05 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-05 10:25:00 | 22650.10 | 22564.44 | 0.00 | ORB-long ORB[22400.10,22590.00] vol=3.9x ATR=51.94 |
| Stop hit — per-position SL triggered | 2024-01-05 10:35:00 | 22598.16 | 22572.23 | 0.00 | SL hit |

### Cycle 38 — SELL (started 2024-01-08 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-08 11:05:00 | 22500.80 | 22604.89 | 0.00 | ORB-short ORB[22603.10,22810.00] vol=1.9x ATR=49.12 |
| Stop hit — per-position SL triggered | 2024-01-08 11:15:00 | 22549.92 | 22600.28 | 0.00 | SL hit |

### Cycle 39 — BUY (started 2024-01-09 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-09 11:05:00 | 22750.00 | 22636.60 | 0.00 | ORB-long ORB[22517.20,22741.30] vol=3.7x ATR=42.72 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-09 11:25:00 | 22814.07 | 22670.08 | 0.00 | T1 1.5R @ 22814.07 |
| Stop hit — per-position SL triggered | 2024-01-09 14:25:00 | 22750.00 | 22781.44 | 0.00 | SL hit |

### Cycle 40 — BUY (started 2024-01-11 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-11 09:35:00 | 22870.00 | 22785.04 | 0.00 | ORB-long ORB[22615.20,22811.20] vol=2.5x ATR=51.79 |
| Stop hit — per-position SL triggered | 2024-01-11 09:40:00 | 22818.21 | 22790.77 | 0.00 | SL hit |

### Cycle 41 — SELL (started 2024-01-16 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-16 11:05:00 | 23331.10 | 23453.05 | 0.00 | ORB-short ORB[23348.20,23477.00] vol=1.6x ATR=44.02 |
| Stop hit — per-position SL triggered | 2024-01-16 11:10:00 | 23375.12 | 23447.72 | 0.00 | SL hit |

### Cycle 42 — SELL (started 2024-01-17 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-17 10:20:00 | 22866.10 | 23009.00 | 0.00 | ORB-short ORB[22893.70,23180.00] vol=3.1x ATR=70.59 |
| Stop hit — per-position SL triggered | 2024-01-17 10:25:00 | 22936.69 | 22993.48 | 0.00 | SL hit |

### Cycle 43 — SELL (started 2024-01-18 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-18 09:50:00 | 22620.00 | 22750.70 | 0.00 | ORB-short ORB[22711.00,23033.20] vol=4.2x ATR=84.11 |
| Stop hit — per-position SL triggered | 2024-01-18 10:05:00 | 22704.11 | 22733.56 | 0.00 | SL hit |

### Cycle 44 — SELL (started 2024-01-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-20 10:15:00 | 23068.00 | 23164.18 | 0.00 | ORB-short ORB[23101.10,23255.80] vol=1.6x ATR=86.01 |
| Stop hit — per-position SL triggered | 2024-01-20 10:20:00 | 23154.01 | 23157.62 | 0.00 | SL hit |

### Cycle 45 — SELL (started 2024-01-23 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-23 10:00:00 | 22932.00 | 23024.62 | 0.00 | ORB-short ORB[22977.60,23112.20] vol=4.1x ATR=81.23 |
| Stop hit — per-position SL triggered | 2024-01-23 10:05:00 | 23013.23 | 23007.86 | 0.00 | SL hit |

### Cycle 46 — BUY (started 2024-01-29 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-29 09:40:00 | 22970.70 | 22886.78 | 0.00 | ORB-long ORB[22775.10,22961.90] vol=4.4x ATR=67.96 |
| Stop hit — per-position SL triggered | 2024-01-29 09:45:00 | 22902.74 | 22891.54 | 0.00 | SL hit |

### Cycle 47 — BUY (started 2024-01-30 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-30 09:45:00 | 23619.10 | 23498.76 | 0.00 | ORB-long ORB[23199.80,23549.00] vol=1.6x ATR=76.58 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-30 10:05:00 | 23733.97 | 23570.34 | 0.00 | T1 1.5R @ 23733.97 |
| Target hit | 2024-01-30 15:05:00 | 23825.10 | 23860.64 | 0.00 | Trail-exit close<VWAP |

### Cycle 48 — BUY (started 2024-01-31 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-31 09:45:00 | 23984.90 | 23888.56 | 0.00 | ORB-long ORB[23714.30,23924.70] vol=1.7x ATR=72.29 |
| Stop hit — per-position SL triggered | 2024-01-31 10:30:00 | 23912.61 | 23937.30 | 0.00 | SL hit |

### Cycle 49 — BUY (started 2024-02-02 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-02 09:50:00 | 23956.20 | 23901.72 | 0.00 | ORB-long ORB[23705.30,23948.00] vol=2.0x ATR=78.23 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-02 10:30:00 | 24073.54 | 23956.81 | 0.00 | T1 1.5R @ 24073.54 |
| Target hit | 2024-02-02 12:20:00 | 24039.10 | 24068.46 | 0.00 | Trail-exit close<VWAP |

### Cycle 50 — SELL (started 2024-02-09 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-09 10:00:00 | 25062.10 | 25307.78 | 0.00 | ORB-short ORB[25149.00,25519.00] vol=1.7x ATR=97.93 |
| Stop hit — per-position SL triggered | 2024-02-09 10:05:00 | 25160.03 | 25293.45 | 0.00 | SL hit |

### Cycle 51 — BUY (started 2024-02-13 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-13 10:05:00 | 25399.90 | 25273.05 | 0.00 | ORB-long ORB[25118.30,25300.00] vol=1.7x ATR=84.63 |
| Stop hit — per-position SL triggered | 2024-02-13 12:25:00 | 25315.27 | 25335.57 | 0.00 | SL hit |

### Cycle 52 — BUY (started 2024-02-15 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-15 10:50:00 | 28090.00 | 27595.99 | 0.00 | ORB-long ORB[27350.30,27661.20] vol=2.3x ATR=122.47 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-15 10:55:00 | 28273.70 | 27665.06 | 0.00 | T1 1.5R @ 28273.70 |
| Stop hit — per-position SL triggered | 2024-02-15 11:05:00 | 28090.00 | 27725.78 | 0.00 | SL hit |

### Cycle 53 — BUY (started 2024-02-16 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-16 09:30:00 | 28105.70 | 27985.13 | 0.00 | ORB-long ORB[27800.00,28000.00] vol=2.5x ATR=97.60 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-16 09:55:00 | 28252.10 | 28047.96 | 0.00 | T1 1.5R @ 28252.10 |
| Stop hit — per-position SL triggered | 2024-02-16 10:45:00 | 28105.70 | 28102.23 | 0.00 | SL hit |

### Cycle 54 — SELL (started 2024-02-21 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-21 10:50:00 | 28674.40 | 28945.55 | 0.00 | ORB-short ORB[28953.10,29199.95] vol=1.5x ATR=70.77 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-21 11:00:00 | 28568.25 | 28913.56 | 0.00 | T1 1.5R @ 28568.25 |
| Target hit | 2024-02-21 15:20:00 | 28107.90 | 28344.60 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 55 — BUY (started 2024-02-27 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-27 10:10:00 | 29188.95 | 29060.08 | 0.00 | ORB-long ORB[28772.75,29138.40] vol=3.4x ATR=86.03 |
| Stop hit — per-position SL triggered | 2024-02-27 11:15:00 | 29102.92 | 29121.07 | 0.00 | SL hit |

### Cycle 56 — SELL (started 2024-02-28 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-28 11:10:00 | 28633.10 | 28928.47 | 0.00 | ORB-short ORB[28920.00,29150.00] vol=1.9x ATR=76.23 |
| Stop hit — per-position SL triggered | 2024-02-28 11:15:00 | 28709.33 | 28911.69 | 0.00 | SL hit |

### Cycle 57 — BUY (started 2024-03-05 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-05 11:10:00 | 29599.00 | 29439.55 | 0.00 | ORB-long ORB[29320.00,29500.00] vol=2.5x ATR=76.88 |
| Stop hit — per-position SL triggered | 2024-03-05 11:15:00 | 29522.12 | 29443.98 | 0.00 | SL hit |

### Cycle 58 — BUY (started 2024-03-19 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-19 09:30:00 | 29914.10 | 29864.12 | 0.00 | ORB-long ORB[29725.00,29880.35] vol=2.9x ATR=60.27 |
| Stop hit — per-position SL triggered | 2024-03-19 09:40:00 | 29853.83 | 29870.01 | 0.00 | SL hit |

### Cycle 59 — BUY (started 2024-03-22 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-22 10:55:00 | 30193.10 | 30095.03 | 0.00 | ORB-long ORB[29899.25,30133.90] vol=1.5x ATR=59.95 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-22 11:30:00 | 30283.02 | 30135.48 | 0.00 | T1 1.5R @ 30283.02 |
| Target hit | 2024-03-22 12:20:00 | 30206.10 | 30215.74 | 0.00 | Trail-exit close<VWAP |

### Cycle 60 — BUY (started 2024-03-26 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-26 11:05:00 | 30490.20 | 30376.05 | 0.00 | ORB-long ORB[30180.10,30450.00] vol=2.5x ATR=55.80 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-26 14:05:00 | 30573.90 | 30453.00 | 0.00 | T1 1.5R @ 30573.90 |
| Target hit | 2024-03-26 15:20:00 | 30686.85 | 30519.51 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 61 — BUY (started 2024-04-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-02 10:15:00 | 30473.00 | 30392.27 | 0.00 | ORB-long ORB[30307.50,30449.60] vol=1.8x ATR=53.55 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-02 10:30:00 | 30553.32 | 30423.59 | 0.00 | T1 1.5R @ 30553.32 |
| Target hit | 2024-04-02 15:20:00 | 30894.80 | 30702.38 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 62 — BUY (started 2024-04-08 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-08 10:40:00 | 31272.25 | 31054.51 | 0.00 | ORB-long ORB[30805.00,31125.00] vol=3.7x ATR=72.75 |
| Stop hit — per-position SL triggered | 2024-04-08 10:45:00 | 31199.50 | 31065.71 | 0.00 | SL hit |

### Cycle 63 — SELL (started 2024-04-10 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-04-10 10:40:00 | 29883.30 | 30036.25 | 0.00 | ORB-short ORB[30068.25,30345.30] vol=2.4x ATR=75.88 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-10 11:05:00 | 29769.48 | 29998.73 | 0.00 | T1 1.5R @ 29769.48 |
| Target hit | 2024-04-10 12:15:00 | 29745.10 | 29735.26 | 0.00 | Trail-exit close>VWAP |

### Cycle 64 — BUY (started 2024-04-15 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-15 10:40:00 | 30149.95 | 29792.03 | 0.00 | ORB-long ORB[29500.00,29848.55] vol=2.9x ATR=87.03 |
| Stop hit — per-position SL triggered | 2024-04-15 11:00:00 | 30062.92 | 29850.76 | 0.00 | SL hit |

### Cycle 65 — BUY (started 2024-04-16 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-16 09:30:00 | 30100.80 | 29972.90 | 0.00 | ORB-long ORB[29651.15,30075.00] vol=2.0x ATR=73.24 |
| Stop hit — per-position SL triggered | 2024-04-16 09:40:00 | 30027.56 | 29990.57 | 0.00 | SL hit |

### Cycle 66 — SELL (started 2024-04-24 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-04-24 10:45:00 | 29050.30 | 29246.39 | 0.00 | ORB-short ORB[29302.20,29485.60] vol=1.7x ATR=67.77 |
| Stop hit — per-position SL triggered | 2024-04-24 11:40:00 | 29118.07 | 29176.72 | 0.00 | SL hit |

### Cycle 67 — SELL (started 2024-04-25 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-04-25 10:50:00 | 28550.00 | 28872.93 | 0.00 | ORB-short ORB[28906.80,29171.65] vol=2.6x ATR=59.94 |
| Stop hit — per-position SL triggered | 2024-04-25 10:55:00 | 28609.94 | 28849.29 | 0.00 | SL hit |

### Cycle 68 — BUY (started 2024-04-30 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-30 09:30:00 | 29606.85 | 29513.09 | 0.00 | ORB-long ORB[29304.05,29589.45] vol=1.5x ATR=77.35 |
| Stop hit — per-position SL triggered | 2024-04-30 10:05:00 | 29529.50 | 29532.82 | 0.00 | SL hit |

### Cycle 69 — BUY (started 2024-05-02 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-02 09:45:00 | 29800.30 | 29637.59 | 0.00 | ORB-long ORB[29356.05,29628.80] vol=2.2x ATR=87.87 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-02 10:00:00 | 29932.10 | 29740.02 | 0.00 | T1 1.5R @ 29932.10 |
| Stop hit — per-position SL triggered | 2024-05-02 10:35:00 | 29800.30 | 29809.72 | 0.00 | SL hit |

### Cycle 70 — SELL (started 2024-05-06 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-06 09:35:00 | 29832.10 | 30062.40 | 0.00 | ORB-short ORB[29950.05,30310.65] vol=1.7x ATR=100.16 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-06 09:40:00 | 29681.85 | 30000.58 | 0.00 | T1 1.5R @ 29681.85 |
| Stop hit — per-position SL triggered | 2024-05-06 09:45:00 | 29832.10 | 29976.53 | 0.00 | SL hit |

### Cycle 71 — SELL (started 2024-05-07 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-07 10:50:00 | 29477.15 | 29791.71 | 0.00 | ORB-short ORB[29821.10,30186.00] vol=1.6x ATR=83.59 |
| Stop hit — per-position SL triggered | 2024-05-07 10:55:00 | 29560.74 | 29778.72 | 0.00 | SL hit |

### Cycle 72 — BUY (started 2024-05-09 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-09 09:30:00 | 30340.20 | 30242.04 | 0.00 | ORB-long ORB[29900.05,30292.50] vol=3.0x ATR=95.70 |
| Stop hit — per-position SL triggered | 2024-05-09 09:50:00 | 30244.50 | 30269.04 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-08-11 10:35:00 | 18284.80 | 2023-08-11 10:40:00 | 18246.50 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2023-08-17 10:35:00 | 18260.00 | 2023-08-17 11:10:00 | 18301.45 | PARTIAL | 0.50 | 0.23% |
| BUY | retest1 | 2023-08-17 10:35:00 | 18260.00 | 2023-08-17 11:20:00 | 18260.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-08-24 09:35:00 | 18349.00 | 2023-08-24 09:45:00 | 18379.39 | STOP_HIT | 1.00 | -0.17% |
| SELL | retest1 | 2023-08-29 11:00:00 | 18500.00 | 2023-08-29 11:25:00 | 18515.65 | STOP_HIT | 1.00 | -0.08% |
| BUY | retest1 | 2023-09-04 10:00:00 | 19124.90 | 2023-09-04 10:10:00 | 19086.76 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2023-09-05 10:30:00 | 19173.20 | 2023-09-05 10:35:00 | 19142.03 | STOP_HIT | 1.00 | -0.16% |
| BUY | retest1 | 2023-09-07 09:50:00 | 19318.00 | 2023-09-07 10:00:00 | 19282.24 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest1 | 2023-09-11 10:10:00 | 19285.20 | 2023-09-11 10:20:00 | 19315.99 | STOP_HIT | 1.00 | -0.16% |
| BUY | retest1 | 2023-09-14 09:45:00 | 19600.00 | 2023-09-14 09:50:00 | 19659.12 | PARTIAL | 0.50 | 0.30% |
| BUY | retest1 | 2023-09-14 09:45:00 | 19600.00 | 2023-09-14 10:05:00 | 19600.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-09-20 10:50:00 | 19401.50 | 2023-09-20 11:15:00 | 19440.57 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2023-09-26 09:40:00 | 19038.60 | 2023-09-26 09:45:00 | 19073.50 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest1 | 2023-10-03 10:00:00 | 18692.50 | 2023-10-03 10:55:00 | 18752.87 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2023-10-04 10:35:00 | 18890.00 | 2023-10-04 10:40:00 | 18835.84 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2023-10-06 09:30:00 | 19010.00 | 2023-10-06 09:45:00 | 19057.84 | PARTIAL | 0.50 | 0.25% |
| BUY | retest1 | 2023-10-06 09:30:00 | 19010.00 | 2023-10-06 09:50:00 | 19010.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-10-09 09:45:00 | 19273.60 | 2023-10-09 10:00:00 | 19222.32 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2023-10-10 11:05:00 | 19414.90 | 2023-10-10 11:20:00 | 19470.42 | PARTIAL | 0.50 | 0.29% |
| BUY | retest1 | 2023-10-10 11:05:00 | 19414.90 | 2023-10-10 15:20:00 | 19700.20 | TARGET_HIT | 0.50 | 1.47% |
| BUY | retest1 | 2023-10-11 11:15:00 | 19888.80 | 2023-10-11 11:20:00 | 19849.42 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2023-10-12 09:40:00 | 19900.80 | 2023-10-12 09:45:00 | 19959.81 | PARTIAL | 0.50 | 0.30% |
| BUY | retest1 | 2023-10-12 09:40:00 | 19900.80 | 2023-10-12 15:20:00 | 20448.60 | TARGET_HIT | 0.50 | 2.75% |
| BUY | retest1 | 2023-10-13 11:00:00 | 20518.60 | 2023-10-13 11:05:00 | 20464.23 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2023-10-18 10:45:00 | 20345.20 | 2023-10-18 11:05:00 | 20288.76 | PARTIAL | 0.50 | 0.28% |
| SELL | retest1 | 2023-10-18 10:45:00 | 20345.20 | 2023-10-18 15:20:00 | 20104.90 | TARGET_HIT | 0.50 | 1.18% |
| BUY | retest1 | 2023-10-19 10:50:00 | 20160.00 | 2023-10-19 11:05:00 | 20121.21 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest1 | 2023-10-27 09:30:00 | 19941.20 | 2023-10-27 09:45:00 | 19858.89 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest1 | 2023-11-03 09:55:00 | 19463.90 | 2023-11-03 10:00:00 | 19501.92 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2023-12-08 11:05:00 | 21570.00 | 2023-12-08 14:40:00 | 21646.31 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2023-12-11 09:45:00 | 21722.50 | 2023-12-11 09:55:00 | 21798.53 | PARTIAL | 0.50 | 0.35% |
| BUY | retest1 | 2023-12-11 09:45:00 | 21722.50 | 2023-12-11 10:45:00 | 21750.00 | TARGET_HIT | 0.50 | 0.13% |
| BUY | retest1 | 2023-12-12 10:40:00 | 22056.20 | 2023-12-12 11:00:00 | 22009.66 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2023-12-15 10:50:00 | 21952.90 | 2023-12-15 11:05:00 | 21995.72 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2023-12-18 11:05:00 | 21989.00 | 2023-12-18 11:15:00 | 21948.47 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest1 | 2023-12-19 11:15:00 | 21970.10 | 2023-12-19 13:55:00 | 21903.92 | PARTIAL | 0.50 | 0.30% |
| SELL | retest1 | 2023-12-19 11:15:00 | 21970.10 | 2023-12-19 15:20:00 | 21949.50 | TARGET_HIT | 0.50 | 0.09% |
| SELL | retest1 | 2023-12-20 11:15:00 | 21886.60 | 2023-12-20 11:30:00 | 21915.99 | STOP_HIT | 1.00 | -0.13% |
| BUY | retest1 | 2023-12-21 11:00:00 | 21630.00 | 2023-12-21 11:05:00 | 21559.03 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2023-12-22 09:45:00 | 21828.10 | 2023-12-22 10:30:00 | 21772.54 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2023-12-28 10:55:00 | 21980.00 | 2023-12-28 11:40:00 | 21927.79 | PARTIAL | 0.50 | 0.24% |
| SELL | retest1 | 2023-12-28 10:55:00 | 21980.00 | 2023-12-28 15:00:00 | 21980.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-01-01 09:40:00 | 22473.00 | 2024-01-01 09:45:00 | 22566.95 | PARTIAL | 0.50 | 0.42% |
| BUY | retest1 | 2024-01-01 09:40:00 | 22473.00 | 2024-01-01 10:00:00 | 22473.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-01-02 10:50:00 | 22150.00 | 2024-01-02 12:30:00 | 22205.45 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2024-01-04 11:05:00 | 22225.60 | 2024-01-04 11:40:00 | 22273.04 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2024-01-05 10:25:00 | 22650.10 | 2024-01-05 10:35:00 | 22598.16 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2024-01-08 11:05:00 | 22500.80 | 2024-01-08 11:15:00 | 22549.92 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2024-01-09 11:05:00 | 22750.00 | 2024-01-09 11:25:00 | 22814.07 | PARTIAL | 0.50 | 0.28% |
| BUY | retest1 | 2024-01-09 11:05:00 | 22750.00 | 2024-01-09 14:25:00 | 22750.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-01-11 09:35:00 | 22870.00 | 2024-01-11 09:40:00 | 22818.21 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2024-01-16 11:05:00 | 23331.10 | 2024-01-16 11:10:00 | 23375.12 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest1 | 2024-01-17 10:20:00 | 22866.10 | 2024-01-17 10:25:00 | 22936.69 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2024-01-18 09:50:00 | 22620.00 | 2024-01-18 10:05:00 | 22704.11 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest1 | 2024-01-20 10:15:00 | 23068.00 | 2024-01-20 10:20:00 | 23154.01 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest1 | 2024-01-23 10:00:00 | 22932.00 | 2024-01-23 10:05:00 | 23013.23 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2024-01-29 09:40:00 | 22970.70 | 2024-01-29 09:45:00 | 22902.74 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2024-01-30 09:45:00 | 23619.10 | 2024-01-30 10:05:00 | 23733.97 | PARTIAL | 0.50 | 0.49% |
| BUY | retest1 | 2024-01-30 09:45:00 | 23619.10 | 2024-01-30 15:05:00 | 23825.10 | TARGET_HIT | 0.50 | 0.87% |
| BUY | retest1 | 2024-01-31 09:45:00 | 23984.90 | 2024-01-31 10:30:00 | 23912.61 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2024-02-02 09:50:00 | 23956.20 | 2024-02-02 10:30:00 | 24073.54 | PARTIAL | 0.50 | 0.49% |
| BUY | retest1 | 2024-02-02 09:50:00 | 23956.20 | 2024-02-02 12:20:00 | 24039.10 | TARGET_HIT | 0.50 | 0.35% |
| SELL | retest1 | 2024-02-09 10:00:00 | 25062.10 | 2024-02-09 10:05:00 | 25160.03 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest1 | 2024-02-13 10:05:00 | 25399.90 | 2024-02-13 12:25:00 | 25315.27 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2024-02-15 10:50:00 | 28090.00 | 2024-02-15 10:55:00 | 28273.70 | PARTIAL | 0.50 | 0.65% |
| BUY | retest1 | 2024-02-15 10:50:00 | 28090.00 | 2024-02-15 11:05:00 | 28090.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-02-16 09:30:00 | 28105.70 | 2024-02-16 09:55:00 | 28252.10 | PARTIAL | 0.50 | 0.52% |
| BUY | retest1 | 2024-02-16 09:30:00 | 28105.70 | 2024-02-16 10:45:00 | 28105.70 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-02-21 10:50:00 | 28674.40 | 2024-02-21 11:00:00 | 28568.25 | PARTIAL | 0.50 | 0.37% |
| SELL | retest1 | 2024-02-21 10:50:00 | 28674.40 | 2024-02-21 15:20:00 | 28107.90 | TARGET_HIT | 0.50 | 1.98% |
| BUY | retest1 | 2024-02-27 10:10:00 | 29188.95 | 2024-02-27 11:15:00 | 29102.92 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2024-02-28 11:10:00 | 28633.10 | 2024-02-28 11:15:00 | 28709.33 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2024-03-05 11:10:00 | 29599.00 | 2024-03-05 11:15:00 | 29522.12 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2024-03-19 09:30:00 | 29914.10 | 2024-03-19 09:40:00 | 29853.83 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2024-03-22 10:55:00 | 30193.10 | 2024-03-22 11:30:00 | 30283.02 | PARTIAL | 0.50 | 0.30% |
| BUY | retest1 | 2024-03-22 10:55:00 | 30193.10 | 2024-03-22 12:20:00 | 30206.10 | TARGET_HIT | 0.50 | 0.04% |
| BUY | retest1 | 2024-03-26 11:05:00 | 30490.20 | 2024-03-26 14:05:00 | 30573.90 | PARTIAL | 0.50 | 0.27% |
| BUY | retest1 | 2024-03-26 11:05:00 | 30490.20 | 2024-03-26 15:20:00 | 30686.85 | TARGET_HIT | 0.50 | 0.64% |
| BUY | retest1 | 2024-04-02 10:15:00 | 30473.00 | 2024-04-02 10:30:00 | 30553.32 | PARTIAL | 0.50 | 0.26% |
| BUY | retest1 | 2024-04-02 10:15:00 | 30473.00 | 2024-04-02 15:20:00 | 30894.80 | TARGET_HIT | 0.50 | 1.38% |
| BUY | retest1 | 2024-04-08 10:40:00 | 31272.25 | 2024-04-08 10:45:00 | 31199.50 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2024-04-10 10:40:00 | 29883.30 | 2024-04-10 11:05:00 | 29769.48 | PARTIAL | 0.50 | 0.38% |
| SELL | retest1 | 2024-04-10 10:40:00 | 29883.30 | 2024-04-10 12:15:00 | 29745.10 | TARGET_HIT | 0.50 | 0.46% |
| BUY | retest1 | 2024-04-15 10:40:00 | 30149.95 | 2024-04-15 11:00:00 | 30062.92 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2024-04-16 09:30:00 | 30100.80 | 2024-04-16 09:40:00 | 30027.56 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2024-04-24 10:45:00 | 29050.30 | 2024-04-24 11:40:00 | 29118.07 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2024-04-25 10:50:00 | 28550.00 | 2024-04-25 10:55:00 | 28609.94 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2024-04-30 09:30:00 | 29606.85 | 2024-04-30 10:05:00 | 29529.50 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2024-05-02 09:45:00 | 29800.30 | 2024-05-02 10:00:00 | 29932.10 | PARTIAL | 0.50 | 0.44% |
| BUY | retest1 | 2024-05-02 09:45:00 | 29800.30 | 2024-05-02 10:35:00 | 29800.30 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-05-06 09:35:00 | 29832.10 | 2024-05-06 09:40:00 | 29681.85 | PARTIAL | 0.50 | 0.50% |
| SELL | retest1 | 2024-05-06 09:35:00 | 29832.10 | 2024-05-06 09:45:00 | 29832.10 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-05-07 10:50:00 | 29477.15 | 2024-05-07 10:55:00 | 29560.74 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2024-05-09 09:30:00 | 30340.20 | 2024-05-09 09:50:00 | 30244.50 | STOP_HIT | 1.00 | -0.32% |
