# 3M India Ltd. (3MINDIA)

## Backtest Summary

- **Window:** 2024-05-13 09:15:00 → 2026-05-08 15:25:00 (35146 bars)
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
| ENTRY1 | 73 |
| ENTRY2 | 0 |
| PARTIAL | 32 |
| TARGET_HIT | 18 |
| STOP_HIT | 55 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 105 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 50 / 55
- **Target hits / Stop hits / Partials:** 18 / 55 / 32
- **Avg / median % per leg:** 0.23% / 0.00%
- **Sum % (uncompounded):** 23.87%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 42 | 18 | 42.9% | 7 | 24 | 11 | 0.22% | 9.2% |
| BUY @ 2nd Alert (retest1) | 42 | 18 | 42.9% | 7 | 24 | 11 | 0.22% | 9.2% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 63 | 32 | 50.8% | 11 | 31 | 21 | 0.23% | 14.7% |
| SELL @ 2nd Alert (retest1) | 63 | 32 | 50.8% | 11 | 31 | 21 | 0.23% | 14.7% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 105 | 50 | 47.6% | 18 | 55 | 32 | 0.23% | 23.9% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-05-13 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-13 10:50:00 | 28572.15 | 28639.05 | 0.00 | ORB-short ORB[28625.00,28845.45] vol=2.2x ATR=90.22 |
| Stop hit — per-position SL triggered | 2024-05-13 11:35:00 | 28662.37 | 28747.47 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2024-05-15 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-15 10:10:00 | 28852.35 | 29008.61 | 0.00 | ORB-short ORB[28942.65,29199.00] vol=2.6x ATR=57.46 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-15 10:40:00 | 28766.16 | 28957.13 | 0.00 | T1 1.5R @ 28766.16 |
| Target hit | 2024-05-15 15:20:00 | 28598.50 | 28719.45 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 3 — BUY (started 2024-05-16 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-16 09:30:00 | 29115.05 | 28983.22 | 0.00 | ORB-long ORB[28686.25,29078.60] vol=1.6x ATR=99.54 |
| Stop hit — per-position SL triggered | 2024-05-16 10:20:00 | 29015.51 | 29067.76 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2024-05-23 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-23 10:40:00 | 31059.95 | 31132.00 | 0.00 | ORB-short ORB[31060.05,31256.45] vol=5.3x ATR=76.87 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-23 13:15:00 | 30944.65 | 31076.85 | 0.00 | T1 1.5R @ 30944.65 |
| Target hit | 2024-05-23 15:20:00 | 30807.00 | 31002.37 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 5 — BUY (started 2024-05-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-27 10:15:00 | 30831.75 | 30705.41 | 0.00 | ORB-long ORB[30586.10,30811.45] vol=4.0x ATR=85.32 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-27 10:20:00 | 30959.74 | 30972.94 | 0.00 | T1 1.5R @ 30959.74 |
| Target hit | 2024-05-27 10:30:00 | 31007.85 | 31045.80 | 0.00 | Trail-exit close<VWAP |

### Cycle 6 — SELL (started 2024-05-28 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-28 09:30:00 | 30900.85 | 31046.35 | 0.00 | ORB-short ORB[31000.00,31300.00] vol=1.9x ATR=84.20 |
| Stop hit — per-position SL triggered | 2024-05-28 10:10:00 | 30985.05 | 30999.40 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2024-06-10 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-10 09:55:00 | 34299.35 | 34079.39 | 0.00 | ORB-long ORB[33754.45,34200.00] vol=1.7x ATR=177.95 |
| Stop hit — per-position SL triggered | 2024-06-10 14:35:00 | 34121.40 | 34228.37 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2024-06-13 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-13 09:30:00 | 36257.70 | 36707.33 | 0.00 | ORB-short ORB[36600.00,37100.00] vol=2.1x ATR=177.35 |
| Stop hit — per-position SL triggered | 2024-06-13 09:35:00 | 36435.05 | 36676.77 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2024-06-18 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-18 09:40:00 | 37010.00 | 36848.39 | 0.00 | ORB-long ORB[36540.00,36969.80] vol=1.6x ATR=150.16 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-18 10:35:00 | 37235.25 | 36935.71 | 0.00 | T1 1.5R @ 37235.25 |
| Stop hit — per-position SL triggered | 2024-06-18 11:30:00 | 37010.00 | 36961.76 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2024-06-19 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-19 10:50:00 | 36943.05 | 37112.33 | 0.00 | ORB-short ORB[37014.00,37500.00] vol=2.4x ATR=98.84 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-19 12:15:00 | 36794.79 | 36989.64 | 0.00 | T1 1.5R @ 36794.79 |
| Stop hit — per-position SL triggered | 2024-06-19 15:05:00 | 36943.05 | 36964.54 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2024-06-20 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-20 09:35:00 | 37140.05 | 36836.06 | 0.00 | ORB-long ORB[36520.10,37000.00] vol=1.8x ATR=138.90 |
| Stop hit — per-position SL triggered | 2024-06-20 09:55:00 | 37001.15 | 36875.17 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2024-06-25 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-25 09:40:00 | 38470.30 | 38360.72 | 0.00 | ORB-long ORB[38143.00,38458.05] vol=2.4x ATR=137.41 |
| Stop hit — per-position SL triggered | 2024-06-25 09:55:00 | 38332.89 | 38363.47 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2024-06-26 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-26 10:30:00 | 38200.05 | 37998.67 | 0.00 | ORB-long ORB[37697.05,38199.00] vol=2.1x ATR=123.56 |
| Stop hit — per-position SL triggered | 2024-06-26 10:40:00 | 38076.49 | 38017.57 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2024-06-27 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-27 09:35:00 | 37794.25 | 37589.69 | 0.00 | ORB-long ORB[37331.75,37699.95] vol=2.0x ATR=165.66 |
| Stop hit — per-position SL triggered | 2024-06-27 09:45:00 | 37628.59 | 37596.86 | 0.00 | SL hit |

### Cycle 15 — BUY (started 2024-07-04 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-04 11:00:00 | 39874.35 | 39502.71 | 0.00 | ORB-long ORB[39202.60,39647.95] vol=2.4x ATR=163.12 |
| Stop hit — per-position SL triggered | 2024-07-04 12:40:00 | 39711.23 | 39618.06 | 0.00 | SL hit |

### Cycle 16 — SELL (started 2024-07-08 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-08 09:55:00 | 38392.10 | 38570.28 | 0.00 | ORB-short ORB[38456.05,38873.70] vol=1.7x ATR=125.61 |
| Stop hit — per-position SL triggered | 2024-07-08 10:00:00 | 38517.71 | 38568.30 | 0.00 | SL hit |

### Cycle 17 — BUY (started 2024-07-09 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-09 09:45:00 | 39559.90 | 39417.92 | 0.00 | ORB-long ORB[39115.80,39550.00] vol=3.0x ATR=113.50 |
| Stop hit — per-position SL triggered | 2024-07-09 09:55:00 | 39446.40 | 39431.19 | 0.00 | SL hit |

### Cycle 18 — SELL (started 2024-07-10 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-10 10:35:00 | 39150.00 | 39733.11 | 0.00 | ORB-short ORB[39701.05,40120.00] vol=3.5x ATR=156.55 |
| Stop hit — per-position SL triggered | 2024-07-10 10:40:00 | 39306.55 | 39680.74 | 0.00 | SL hit |

### Cycle 19 — SELL (started 2024-07-18 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-18 10:10:00 | 38456.00 | 38642.91 | 0.00 | ORB-short ORB[38559.40,39099.95] vol=2.3x ATR=125.69 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-18 11:15:00 | 38267.46 | 38555.05 | 0.00 | T1 1.5R @ 38267.46 |
| Target hit | 2024-07-18 15:20:00 | 38020.90 | 38325.85 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 20 — SELL (started 2024-07-19 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-19 10:05:00 | 37609.00 | 37983.51 | 0.00 | ORB-short ORB[37926.05,38381.70] vol=1.8x ATR=119.09 |
| Stop hit — per-position SL triggered | 2024-07-19 10:15:00 | 37728.09 | 37939.08 | 0.00 | SL hit |

### Cycle 21 — BUY (started 2024-07-22 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-22 09:35:00 | 37717.85 | 37337.63 | 0.00 | ORB-long ORB[37018.90,37499.80] vol=1.6x ATR=163.79 |
| Stop hit — per-position SL triggered | 2024-07-22 09:40:00 | 37554.06 | 37432.96 | 0.00 | SL hit |

### Cycle 22 — BUY (started 2024-07-23 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-23 10:30:00 | 39043.15 | 38888.29 | 0.00 | ORB-long ORB[38500.00,39000.00] vol=3.3x ATR=114.80 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-23 11:05:00 | 39215.35 | 38944.32 | 0.00 | T1 1.5R @ 39215.35 |
| Stop hit — per-position SL triggered | 2024-07-23 11:15:00 | 39043.15 | 38956.59 | 0.00 | SL hit |

### Cycle 23 — SELL (started 2024-07-25 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-25 09:35:00 | 38200.00 | 38392.55 | 0.00 | ORB-short ORB[38259.05,38574.65] vol=1.9x ATR=101.59 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-25 10:20:00 | 38047.62 | 38184.55 | 0.00 | T1 1.5R @ 38047.62 |
| Target hit | 2024-07-25 15:20:00 | 37578.00 | 37855.27 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 24 — BUY (started 2024-07-26 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-26 10:35:00 | 38593.15 | 38284.65 | 0.00 | ORB-long ORB[37900.00,38089.90] vol=1.9x ATR=106.05 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-26 10:40:00 | 38752.23 | 38343.96 | 0.00 | T1 1.5R @ 38752.23 |
| Stop hit — per-position SL triggered | 2024-07-26 10:50:00 | 38593.15 | 38361.86 | 0.00 | SL hit |

### Cycle 25 — BUY (started 2024-08-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-01 11:15:00 | 39850.00 | 39693.45 | 0.00 | ORB-long ORB[39400.00,39816.70] vol=8.1x ATR=100.09 |
| Stop hit — per-position SL triggered | 2024-08-01 11:40:00 | 39749.91 | 39696.92 | 0.00 | SL hit |

### Cycle 26 — BUY (started 2024-08-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-02 10:15:00 | 39466.55 | 39341.53 | 0.00 | ORB-long ORB[39093.55,39324.90] vol=2.2x ATR=119.58 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-02 10:20:00 | 39645.92 | 39396.37 | 0.00 | T1 1.5R @ 39645.92 |
| Target hit | 2024-08-02 13:00:00 | 39518.85 | 39555.23 | 0.00 | Trail-exit close<VWAP |

### Cycle 27 — SELL (started 2024-08-09 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-09 11:10:00 | 36963.00 | 37181.04 | 0.00 | ORB-short ORB[37000.05,37499.00] vol=1.6x ATR=64.79 |
| Stop hit — per-position SL triggered | 2024-08-09 11:15:00 | 37027.79 | 37179.52 | 0.00 | SL hit |

### Cycle 28 — BUY (started 2024-08-12 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-12 11:05:00 | 37839.00 | 37625.22 | 0.00 | ORB-long ORB[37440.00,37799.10] vol=2.8x ATR=91.12 |
| Stop hit — per-position SL triggered | 2024-08-12 11:10:00 | 37747.88 | 37631.57 | 0.00 | SL hit |

### Cycle 29 — SELL (started 2024-08-13 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-13 09:50:00 | 37500.00 | 37572.40 | 0.00 | ORB-short ORB[37515.00,37918.60] vol=3.0x ATR=95.06 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-13 11:55:00 | 37357.41 | 37550.61 | 0.00 | T1 1.5R @ 37357.41 |
| Target hit | 2024-08-13 15:20:00 | 36964.40 | 37199.86 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 30 — BUY (started 2024-08-20 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-20 11:10:00 | 35941.05 | 35711.53 | 0.00 | ORB-long ORB[35407.65,35899.95] vol=2.9x ATR=88.41 |
| Stop hit — per-position SL triggered | 2024-08-20 11:50:00 | 35852.64 | 35776.30 | 0.00 | SL hit |

### Cycle 31 — SELL (started 2024-08-29 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-29 10:30:00 | 35285.00 | 35438.67 | 0.00 | ORB-short ORB[35304.00,35578.00] vol=1.9x ATR=74.13 |
| Stop hit — per-position SL triggered | 2024-08-29 11:45:00 | 35359.13 | 35366.17 | 0.00 | SL hit |

### Cycle 32 — BUY (started 2024-09-05 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-05 09:50:00 | 35905.00 | 35700.34 | 0.00 | ORB-long ORB[35317.95,35659.40] vol=2.0x ATR=94.33 |
| Stop hit — per-position SL triggered | 2024-09-05 11:05:00 | 35810.67 | 35789.22 | 0.00 | SL hit |

### Cycle 33 — SELL (started 2024-09-06 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-06 10:30:00 | 34820.00 | 35318.96 | 0.00 | ORB-short ORB[35228.20,35528.20] vol=2.8x ATR=118.86 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-06 10:35:00 | 34641.72 | 35157.73 | 0.00 | T1 1.5R @ 34641.72 |
| Stop hit — per-position SL triggered | 2024-09-06 10:40:00 | 34820.00 | 35116.79 | 0.00 | SL hit |

### Cycle 34 — SELL (started 2024-09-17 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-17 10:30:00 | 35732.15 | 35863.75 | 0.00 | ORB-short ORB[35735.35,36128.00] vol=1.7x ATR=91.04 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-17 10:55:00 | 35595.59 | 35835.42 | 0.00 | T1 1.5R @ 35595.59 |
| Stop hit — per-position SL triggered | 2024-09-17 11:20:00 | 35732.15 | 35818.94 | 0.00 | SL hit |

### Cycle 35 — BUY (started 2024-09-19 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-19 09:30:00 | 35597.00 | 35537.10 | 0.00 | ORB-long ORB[35300.05,35549.95] vol=5.2x ATR=116.06 |
| Stop hit — per-position SL triggered | 2024-09-19 09:45:00 | 35480.94 | 35550.25 | 0.00 | SL hit |

### Cycle 36 — SELL (started 2024-09-20 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-20 10:05:00 | 35060.00 | 35240.38 | 0.00 | ORB-short ORB[35209.00,35698.80] vol=1.8x ATR=111.54 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-20 13:40:00 | 34892.68 | 35076.79 | 0.00 | T1 1.5R @ 34892.68 |
| Target hit | 2024-09-20 15:20:00 | 34877.00 | 34927.80 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 37 — SELL (started 2024-09-24 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-24 11:10:00 | 35412.55 | 35570.10 | 0.00 | ORB-short ORB[35488.20,35854.85] vol=2.3x ATR=72.16 |
| Stop hit — per-position SL triggered | 2024-09-24 11:25:00 | 35484.71 | 35568.93 | 0.00 | SL hit |

### Cycle 38 — SELL (started 2024-09-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-27 10:15:00 | 35415.00 | 35725.85 | 0.00 | ORB-short ORB[35647.90,35900.00] vol=2.2x ATR=86.66 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-27 10:25:00 | 35285.01 | 35615.28 | 0.00 | T1 1.5R @ 35285.01 |
| Target hit | 2024-09-27 15:20:00 | 34499.00 | 34946.44 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 39 — SELL (started 2024-10-11 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-11 10:10:00 | 34380.00 | 34521.92 | 0.00 | ORB-short ORB[34512.40,34711.80] vol=2.0x ATR=92.24 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-11 11:25:00 | 34241.64 | 34457.58 | 0.00 | T1 1.5R @ 34241.64 |
| Stop hit — per-position SL triggered | 2024-10-11 11:50:00 | 34380.00 | 34442.71 | 0.00 | SL hit |

### Cycle 40 — SELL (started 2024-10-16 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-16 10:00:00 | 33900.00 | 34193.78 | 0.00 | ORB-short ORB[34062.50,34410.20] vol=1.5x ATR=123.30 |
| Stop hit — per-position SL triggered | 2024-10-16 10:05:00 | 34023.30 | 34191.07 | 0.00 | SL hit |

### Cycle 41 — BUY (started 2024-10-18 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-18 11:10:00 | 33794.75 | 33595.30 | 0.00 | ORB-long ORB[33325.80,33788.00] vol=2.0x ATR=97.40 |
| Stop hit — per-position SL triggered | 2024-10-18 12:50:00 | 33697.35 | 33642.96 | 0.00 | SL hit |

### Cycle 42 — SELL (started 2024-10-22 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-22 10:35:00 | 33725.65 | 33807.99 | 0.00 | ORB-short ORB[33803.50,34159.95] vol=3.0x ATR=97.02 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-22 10:40:00 | 33580.12 | 33806.53 | 0.00 | T1 1.5R @ 33580.12 |
| Target hit | 2024-10-22 14:15:00 | 33650.00 | 33598.90 | 0.00 | Trail-exit close>VWAP |

### Cycle 43 — SELL (started 2024-10-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-25 11:15:00 | 33605.20 | 33786.94 | 0.00 | ORB-short ORB[33725.00,34119.00] vol=1.7x ATR=102.39 |
| Stop hit — per-position SL triggered | 2024-10-25 12:10:00 | 33707.59 | 33776.40 | 0.00 | SL hit |

### Cycle 44 — SELL (started 2024-10-28 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-28 09:35:00 | 33450.00 | 33557.31 | 0.00 | ORB-short ORB[33510.00,33939.85] vol=4.3x ATR=142.17 |
| Stop hit — per-position SL triggered | 2024-10-28 11:25:00 | 33592.17 | 33513.54 | 0.00 | SL hit |

### Cycle 45 — BUY (started 2024-11-25 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-25 11:00:00 | 31311.60 | 30996.00 | 0.00 | ORB-long ORB[30690.00,31147.00] vol=1.7x ATR=84.95 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-25 11:25:00 | 31439.02 | 31018.17 | 0.00 | T1 1.5R @ 31439.02 |
| Target hit | 2024-11-25 15:20:00 | 31903.45 | 31318.06 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 46 — SELL (started 2024-12-05 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-05 10:55:00 | 31051.25 | 31225.63 | 0.00 | ORB-short ORB[31152.25,31449.95] vol=2.5x ATR=62.70 |
| Stop hit — per-position SL triggered | 2024-12-05 11:00:00 | 31113.95 | 31223.03 | 0.00 | SL hit |

### Cycle 47 — SELL (started 2024-12-11 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-11 09:30:00 | 31250.05 | 31346.33 | 0.00 | ORB-short ORB[31277.00,31559.95] vol=2.5x ATR=97.63 |
| Stop hit — per-position SL triggered | 2024-12-11 09:35:00 | 31347.68 | 31343.46 | 0.00 | SL hit |

### Cycle 48 — BUY (started 2024-12-19 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-19 10:30:00 | 30947.85 | 30811.68 | 0.00 | ORB-long ORB[30500.00,30849.95] vol=1.9x ATR=86.75 |
| Stop hit — per-position SL triggered | 2024-12-19 12:05:00 | 30861.10 | 30865.89 | 0.00 | SL hit |

### Cycle 49 — BUY (started 2024-12-24 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-24 11:10:00 | 30962.60 | 30813.10 | 0.00 | ORB-long ORB[30510.00,30912.90] vol=1.8x ATR=48.35 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-24 11:35:00 | 31035.13 | 30828.48 | 0.00 | T1 1.5R @ 31035.13 |
| Stop hit — per-position SL triggered | 2024-12-24 12:10:00 | 30962.60 | 30981.80 | 0.00 | SL hit |

### Cycle 50 — SELL (started 2024-12-26 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-26 10:45:00 | 30678.00 | 30872.40 | 0.00 | ORB-short ORB[31031.95,31248.00] vol=3.2x ATR=68.93 |
| Stop hit — per-position SL triggered | 2024-12-26 11:10:00 | 30746.93 | 30863.74 | 0.00 | SL hit |

### Cycle 51 — SELL (started 2024-12-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-27 11:15:00 | 30609.75 | 30765.01 | 0.00 | ORB-short ORB[30978.30,31224.95] vol=18.0x ATR=69.14 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-27 11:55:00 | 30506.04 | 30722.18 | 0.00 | T1 1.5R @ 30506.04 |
| Stop hit — per-position SL triggered | 2024-12-27 13:40:00 | 30609.75 | 30669.98 | 0.00 | SL hit |

### Cycle 52 — SELL (started 2024-12-30 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-30 09:30:00 | 30244.80 | 30351.29 | 0.00 | ORB-short ORB[30250.00,30649.95] vol=1.9x ATR=88.41 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-30 10:10:00 | 30112.18 | 30260.44 | 0.00 | T1 1.5R @ 30112.18 |
| Stop hit — per-position SL triggered | 2024-12-30 14:45:00 | 30244.80 | 30203.53 | 0.00 | SL hit |

### Cycle 53 — BUY (started 2025-01-02 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-02 10:55:00 | 30903.00 | 30720.48 | 0.00 | ORB-long ORB[30150.10,30322.00] vol=1.6x ATR=87.35 |
| Stop hit — per-position SL triggered | 2025-01-02 11:15:00 | 30815.65 | 30730.24 | 0.00 | SL hit |

### Cycle 54 — SELL (started 2025-02-05 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-05 09:30:00 | 28759.95 | 28909.31 | 0.00 | ORB-short ORB[28820.00,29196.30] vol=2.2x ATR=172.59 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-05 10:50:00 | 28501.06 | 28722.18 | 0.00 | T1 1.5R @ 28501.06 |
| Target hit | 2025-02-05 15:20:00 | 28467.55 | 28573.11 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 55 — SELL (started 2025-02-14 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-14 10:40:00 | 26866.45 | 26978.66 | 0.00 | ORB-short ORB[26947.95,27290.95] vol=3.8x ATR=79.23 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-14 11:05:00 | 26747.61 | 26956.53 | 0.00 | T1 1.5R @ 26747.61 |
| Stop hit — per-position SL triggered | 2025-02-14 11:30:00 | 26866.45 | 26950.67 | 0.00 | SL hit |

### Cycle 56 — BUY (started 2025-02-20 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-20 11:10:00 | 27142.25 | 26925.99 | 0.00 | ORB-long ORB[26677.75,27054.40] vol=4.0x ATR=58.65 |
| Stop hit — per-position SL triggered | 2025-02-20 11:15:00 | 27083.60 | 26931.12 | 0.00 | SL hit |

### Cycle 57 — SELL (started 2025-02-21 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-21 09:40:00 | 27344.80 | 27543.53 | 0.00 | ORB-short ORB[27400.00,27750.00] vol=2.1x ATR=120.34 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-21 09:55:00 | 27164.29 | 27433.99 | 0.00 | T1 1.5R @ 27164.29 |
| Stop hit — per-position SL triggered | 2025-02-21 10:40:00 | 27344.80 | 27285.29 | 0.00 | SL hit |

### Cycle 58 — SELL (started 2025-02-25 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-25 11:00:00 | 26704.70 | 26851.82 | 0.00 | ORB-short ORB[26707.55,26961.85] vol=2.2x ATR=58.47 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-25 12:15:00 | 26617.00 | 26785.38 | 0.00 | T1 1.5R @ 26617.00 |
| Target hit | 2025-02-25 15:20:00 | 26497.35 | 26659.72 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 59 — BUY (started 2025-03-05 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-05 10:35:00 | 26350.00 | 26129.53 | 0.00 | ORB-long ORB[25769.50,26118.40] vol=3.3x ATR=87.97 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-05 10:40:00 | 26481.95 | 26231.82 | 0.00 | T1 1.5R @ 26481.95 |
| Target hit | 2025-03-05 15:20:00 | 27695.30 | 27261.76 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 60 — SELL (started 2025-03-10 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-10 09:30:00 | 27452.35 | 27644.12 | 0.00 | ORB-short ORB[27555.00,27900.05] vol=1.6x ATR=118.53 |
| Stop hit — per-position SL triggered | 2025-03-10 09:40:00 | 27570.88 | 27611.86 | 0.00 | SL hit |

### Cycle 61 — SELL (started 2025-03-12 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-12 10:45:00 | 27708.75 | 27759.98 | 0.00 | ORB-short ORB[27712.00,27999.00] vol=3.2x ATR=78.68 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-12 11:00:00 | 27590.74 | 27697.22 | 0.00 | T1 1.5R @ 27590.74 |
| Target hit | 2025-03-12 14:05:00 | 27480.00 | 27445.20 | 0.00 | Trail-exit close>VWAP |

### Cycle 62 — BUY (started 2025-03-18 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-18 10:40:00 | 27580.65 | 27390.55 | 0.00 | ORB-long ORB[27207.30,27550.00] vol=1.8x ATR=71.82 |
| Stop hit — per-position SL triggered | 2025-03-18 11:15:00 | 27508.83 | 27432.21 | 0.00 | SL hit |

### Cycle 63 — BUY (started 2025-03-19 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-19 10:10:00 | 27900.00 | 27729.26 | 0.00 | ORB-long ORB[27459.70,27772.00] vol=2.1x ATR=89.70 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-19 10:50:00 | 28034.55 | 27844.40 | 0.00 | T1 1.5R @ 28034.55 |
| Target hit | 2025-03-19 15:20:00 | 28285.45 | 28163.07 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 64 — SELL (started 2025-03-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-20 10:15:00 | 27985.15 | 28136.51 | 0.00 | ORB-short ORB[28342.25,28700.05] vol=2.8x ATR=118.11 |
| Stop hit — per-position SL triggered | 2025-03-20 13:40:00 | 28103.26 | 28071.90 | 0.00 | SL hit |

### Cycle 65 — SELL (started 2025-03-26 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-26 10:30:00 | 28345.15 | 28494.96 | 0.00 | ORB-short ORB[28410.95,28715.95] vol=2.7x ATR=74.57 |
| Stop hit — per-position SL triggered | 2025-03-26 11:25:00 | 28419.72 | 28441.41 | 0.00 | SL hit |

### Cycle 66 — SELL (started 2025-04-01 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-01 11:05:00 | 28245.00 | 28470.73 | 0.00 | ORB-short ORB[28416.50,28760.05] vol=2.0x ATR=87.97 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-01 11:35:00 | 28113.05 | 28439.94 | 0.00 | T1 1.5R @ 28113.05 |
| Stop hit — per-position SL triggered | 2025-04-01 12:00:00 | 28245.00 | 28388.94 | 0.00 | SL hit |

### Cycle 67 — BUY (started 2025-04-02 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-02 10:45:00 | 28410.05 | 28342.45 | 0.00 | ORB-long ORB[28071.70,28394.20] vol=2.0x ATR=61.12 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-02 12:00:00 | 28501.72 | 28369.39 | 0.00 | T1 1.5R @ 28501.72 |
| Target hit | 2025-04-02 15:20:00 | 28749.00 | 28601.29 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 68 — SELL (started 2025-04-21 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-21 09:40:00 | 30175.00 | 30314.50 | 0.00 | ORB-short ORB[30250.00,30465.00] vol=2.9x ATR=87.74 |
| Stop hit — per-position SL triggered | 2025-04-21 13:10:00 | 30262.74 | 30219.87 | 0.00 | SL hit |

### Cycle 69 — SELL (started 2025-04-23 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-23 10:55:00 | 29860.00 | 30038.99 | 0.00 | ORB-short ORB[30005.00,30360.00] vol=4.7x ATR=64.99 |
| Stop hit — per-position SL triggered | 2025-04-23 11:00:00 | 29924.99 | 30037.71 | 0.00 | SL hit |

### Cycle 70 — SELL (started 2025-04-25 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-25 09:50:00 | 29305.00 | 29547.75 | 0.00 | ORB-short ORB[29780.00,30000.00] vol=3.3x ATR=84.60 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-25 10:30:00 | 29178.09 | 29448.72 | 0.00 | T1 1.5R @ 29178.09 |
| Stop hit — per-position SL triggered | 2025-04-25 12:15:00 | 29305.00 | 29389.35 | 0.00 | SL hit |

### Cycle 71 — SELL (started 2025-04-28 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-28 10:40:00 | 29610.00 | 29820.30 | 0.00 | ORB-short ORB[29695.00,30050.00] vol=2.9x ATR=93.88 |
| Stop hit — per-position SL triggered | 2025-04-28 10:55:00 | 29703.88 | 29809.23 | 0.00 | SL hit |

### Cycle 72 — BUY (started 2025-04-29 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-29 11:00:00 | 29855.00 | 29718.80 | 0.00 | ORB-long ORB[29600.00,29845.00] vol=2.0x ATR=79.70 |
| Stop hit — per-position SL triggered | 2025-04-29 11:30:00 | 29775.30 | 29754.40 | 0.00 | SL hit |

### Cycle 73 — BUY (started 2025-05-05 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-05 11:05:00 | 29960.00 | 29803.27 | 0.00 | ORB-long ORB[29610.00,29925.00] vol=5.5x ATR=66.01 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-05 11:20:00 | 30059.02 | 29855.94 | 0.00 | T1 1.5R @ 30059.02 |
| Target hit | 2025-05-05 15:20:00 | 30170.00 | 30039.50 | 0.00 | EOD square-off @ 15:20:00 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2024-05-13 10:50:00 | 28572.15 | 2024-05-13 11:35:00 | 28662.37 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2024-05-15 10:10:00 | 28852.35 | 2024-05-15 10:40:00 | 28766.16 | PARTIAL | 0.50 | 0.30% |
| SELL | retest1 | 2024-05-15 10:10:00 | 28852.35 | 2024-05-15 15:20:00 | 28598.50 | TARGET_HIT | 0.50 | 0.88% |
| BUY | retest1 | 2024-05-16 09:30:00 | 29115.05 | 2024-05-16 10:20:00 | 29015.51 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2024-05-23 10:40:00 | 31059.95 | 2024-05-23 13:15:00 | 30944.65 | PARTIAL | 0.50 | 0.37% |
| SELL | retest1 | 2024-05-23 10:40:00 | 31059.95 | 2024-05-23 15:20:00 | 30807.00 | TARGET_HIT | 0.50 | 0.81% |
| BUY | retest1 | 2024-05-27 10:15:00 | 30831.75 | 2024-05-27 10:20:00 | 30959.74 | PARTIAL | 0.50 | 0.42% |
| BUY | retest1 | 2024-05-27 10:15:00 | 30831.75 | 2024-05-27 10:30:00 | 31007.85 | TARGET_HIT | 0.50 | 0.57% |
| SELL | retest1 | 2024-05-28 09:30:00 | 30900.85 | 2024-05-28 10:10:00 | 30985.05 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2024-06-10 09:55:00 | 34299.35 | 2024-06-10 14:35:00 | 34121.40 | STOP_HIT | 1.00 | -0.52% |
| SELL | retest1 | 2024-06-13 09:30:00 | 36257.70 | 2024-06-13 09:35:00 | 36435.05 | STOP_HIT | 1.00 | -0.49% |
| BUY | retest1 | 2024-06-18 09:40:00 | 37010.00 | 2024-06-18 10:35:00 | 37235.25 | PARTIAL | 0.50 | 0.61% |
| BUY | retest1 | 2024-06-18 09:40:00 | 37010.00 | 2024-06-18 11:30:00 | 37010.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-06-19 10:50:00 | 36943.05 | 2024-06-19 12:15:00 | 36794.79 | PARTIAL | 0.50 | 0.40% |
| SELL | retest1 | 2024-06-19 10:50:00 | 36943.05 | 2024-06-19 15:05:00 | 36943.05 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-06-20 09:35:00 | 37140.05 | 2024-06-20 09:55:00 | 37001.15 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2024-06-25 09:40:00 | 38470.30 | 2024-06-25 09:55:00 | 38332.89 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2024-06-26 10:30:00 | 38200.05 | 2024-06-26 10:40:00 | 38076.49 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2024-06-27 09:35:00 | 37794.25 | 2024-06-27 09:45:00 | 37628.59 | STOP_HIT | 1.00 | -0.44% |
| BUY | retest1 | 2024-07-04 11:00:00 | 39874.35 | 2024-07-04 12:40:00 | 39711.23 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest1 | 2024-07-08 09:55:00 | 38392.10 | 2024-07-08 10:00:00 | 38517.71 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2024-07-09 09:45:00 | 39559.90 | 2024-07-09 09:55:00 | 39446.40 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2024-07-10 10:35:00 | 39150.00 | 2024-07-10 10:40:00 | 39306.55 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest1 | 2024-07-18 10:10:00 | 38456.00 | 2024-07-18 11:15:00 | 38267.46 | PARTIAL | 0.50 | 0.49% |
| SELL | retest1 | 2024-07-18 10:10:00 | 38456.00 | 2024-07-18 15:20:00 | 38020.90 | TARGET_HIT | 0.50 | 1.13% |
| SELL | retest1 | 2024-07-19 10:05:00 | 37609.00 | 2024-07-19 10:15:00 | 37728.09 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2024-07-22 09:35:00 | 37717.85 | 2024-07-22 09:40:00 | 37554.06 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest1 | 2024-07-23 10:30:00 | 39043.15 | 2024-07-23 11:05:00 | 39215.35 | PARTIAL | 0.50 | 0.44% |
| BUY | retest1 | 2024-07-23 10:30:00 | 39043.15 | 2024-07-23 11:15:00 | 39043.15 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-07-25 09:35:00 | 38200.00 | 2024-07-25 10:20:00 | 38047.62 | PARTIAL | 0.50 | 0.40% |
| SELL | retest1 | 2024-07-25 09:35:00 | 38200.00 | 2024-07-25 15:20:00 | 37578.00 | TARGET_HIT | 0.50 | 1.63% |
| BUY | retest1 | 2024-07-26 10:35:00 | 38593.15 | 2024-07-26 10:40:00 | 38752.23 | PARTIAL | 0.50 | 0.41% |
| BUY | retest1 | 2024-07-26 10:35:00 | 38593.15 | 2024-07-26 10:50:00 | 38593.15 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-08-01 11:15:00 | 39850.00 | 2024-08-01 11:40:00 | 39749.91 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2024-08-02 10:15:00 | 39466.55 | 2024-08-02 10:20:00 | 39645.92 | PARTIAL | 0.50 | 0.45% |
| BUY | retest1 | 2024-08-02 10:15:00 | 39466.55 | 2024-08-02 13:00:00 | 39518.85 | TARGET_HIT | 0.50 | 0.13% |
| SELL | retest1 | 2024-08-09 11:10:00 | 36963.00 | 2024-08-09 11:15:00 | 37027.79 | STOP_HIT | 1.00 | -0.18% |
| BUY | retest1 | 2024-08-12 11:05:00 | 37839.00 | 2024-08-12 11:10:00 | 37747.88 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2024-08-13 09:50:00 | 37500.00 | 2024-08-13 11:55:00 | 37357.41 | PARTIAL | 0.50 | 0.38% |
| SELL | retest1 | 2024-08-13 09:50:00 | 37500.00 | 2024-08-13 15:20:00 | 36964.40 | TARGET_HIT | 0.50 | 1.43% |
| BUY | retest1 | 2024-08-20 11:10:00 | 35941.05 | 2024-08-20 11:50:00 | 35852.64 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2024-08-29 10:30:00 | 35285.00 | 2024-08-29 11:45:00 | 35359.13 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2024-09-05 09:50:00 | 35905.00 | 2024-09-05 11:05:00 | 35810.67 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2024-09-06 10:30:00 | 34820.00 | 2024-09-06 10:35:00 | 34641.72 | PARTIAL | 0.50 | 0.51% |
| SELL | retest1 | 2024-09-06 10:30:00 | 34820.00 | 2024-09-06 10:40:00 | 34820.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-09-17 10:30:00 | 35732.15 | 2024-09-17 10:55:00 | 35595.59 | PARTIAL | 0.50 | 0.38% |
| SELL | retest1 | 2024-09-17 10:30:00 | 35732.15 | 2024-09-17 11:20:00 | 35732.15 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-09-19 09:30:00 | 35597.00 | 2024-09-19 09:45:00 | 35480.94 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2024-09-20 10:05:00 | 35060.00 | 2024-09-20 13:40:00 | 34892.68 | PARTIAL | 0.50 | 0.48% |
| SELL | retest1 | 2024-09-20 10:05:00 | 35060.00 | 2024-09-20 15:20:00 | 34877.00 | TARGET_HIT | 0.50 | 0.52% |
| SELL | retest1 | 2024-09-24 11:10:00 | 35412.55 | 2024-09-24 11:25:00 | 35484.71 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2024-09-27 10:15:00 | 35415.00 | 2024-09-27 10:25:00 | 35285.01 | PARTIAL | 0.50 | 0.37% |
| SELL | retest1 | 2024-09-27 10:15:00 | 35415.00 | 2024-09-27 15:20:00 | 34499.00 | TARGET_HIT | 0.50 | 2.59% |
| SELL | retest1 | 2024-10-11 10:10:00 | 34380.00 | 2024-10-11 11:25:00 | 34241.64 | PARTIAL | 0.50 | 0.40% |
| SELL | retest1 | 2024-10-11 10:10:00 | 34380.00 | 2024-10-11 11:50:00 | 34380.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-10-16 10:00:00 | 33900.00 | 2024-10-16 10:05:00 | 34023.30 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2024-10-18 11:10:00 | 33794.75 | 2024-10-18 12:50:00 | 33697.35 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2024-10-22 10:35:00 | 33725.65 | 2024-10-22 10:40:00 | 33580.12 | PARTIAL | 0.50 | 0.43% |
| SELL | retest1 | 2024-10-22 10:35:00 | 33725.65 | 2024-10-22 14:15:00 | 33650.00 | TARGET_HIT | 0.50 | 0.22% |
| SELL | retest1 | 2024-10-25 11:15:00 | 33605.20 | 2024-10-25 12:10:00 | 33707.59 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2024-10-28 09:35:00 | 33450.00 | 2024-10-28 11:25:00 | 33592.17 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest1 | 2024-11-25 11:00:00 | 31311.60 | 2024-11-25 11:25:00 | 31439.02 | PARTIAL | 0.50 | 0.41% |
| BUY | retest1 | 2024-11-25 11:00:00 | 31311.60 | 2024-11-25 15:20:00 | 31903.45 | TARGET_HIT | 0.50 | 1.89% |
| SELL | retest1 | 2024-12-05 10:55:00 | 31051.25 | 2024-12-05 11:00:00 | 31113.95 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2024-12-11 09:30:00 | 31250.05 | 2024-12-11 09:35:00 | 31347.68 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2024-12-19 10:30:00 | 30947.85 | 2024-12-19 12:05:00 | 30861.10 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2024-12-24 11:10:00 | 30962.60 | 2024-12-24 11:35:00 | 31035.13 | PARTIAL | 0.50 | 0.23% |
| BUY | retest1 | 2024-12-24 11:10:00 | 30962.60 | 2024-12-24 12:10:00 | 30962.60 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-12-26 10:45:00 | 30678.00 | 2024-12-26 11:10:00 | 30746.93 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2024-12-27 11:15:00 | 30609.75 | 2024-12-27 11:55:00 | 30506.04 | PARTIAL | 0.50 | 0.34% |
| SELL | retest1 | 2024-12-27 11:15:00 | 30609.75 | 2024-12-27 13:40:00 | 30609.75 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-12-30 09:30:00 | 30244.80 | 2024-12-30 10:10:00 | 30112.18 | PARTIAL | 0.50 | 0.44% |
| SELL | retest1 | 2024-12-30 09:30:00 | 30244.80 | 2024-12-30 14:45:00 | 30244.80 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-01-02 10:55:00 | 30903.00 | 2025-01-02 11:15:00 | 30815.65 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2025-02-05 09:30:00 | 28759.95 | 2025-02-05 10:50:00 | 28501.06 | PARTIAL | 0.50 | 0.90% |
| SELL | retest1 | 2025-02-05 09:30:00 | 28759.95 | 2025-02-05 15:20:00 | 28467.55 | TARGET_HIT | 0.50 | 1.02% |
| SELL | retest1 | 2025-02-14 10:40:00 | 26866.45 | 2025-02-14 11:05:00 | 26747.61 | PARTIAL | 0.50 | 0.44% |
| SELL | retest1 | 2025-02-14 10:40:00 | 26866.45 | 2025-02-14 11:30:00 | 26866.45 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-02-20 11:10:00 | 27142.25 | 2025-02-20 11:15:00 | 27083.60 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2025-02-21 09:40:00 | 27344.80 | 2025-02-21 09:55:00 | 27164.29 | PARTIAL | 0.50 | 0.66% |
| SELL | retest1 | 2025-02-21 09:40:00 | 27344.80 | 2025-02-21 10:40:00 | 27344.80 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-02-25 11:00:00 | 26704.70 | 2025-02-25 12:15:00 | 26617.00 | PARTIAL | 0.50 | 0.33% |
| SELL | retest1 | 2025-02-25 11:00:00 | 26704.70 | 2025-02-25 15:20:00 | 26497.35 | TARGET_HIT | 0.50 | 0.78% |
| BUY | retest1 | 2025-03-05 10:35:00 | 26350.00 | 2025-03-05 10:40:00 | 26481.95 | PARTIAL | 0.50 | 0.50% |
| BUY | retest1 | 2025-03-05 10:35:00 | 26350.00 | 2025-03-05 15:20:00 | 27695.30 | TARGET_HIT | 0.50 | 5.11% |
| SELL | retest1 | 2025-03-10 09:30:00 | 27452.35 | 2025-03-10 09:40:00 | 27570.88 | STOP_HIT | 1.00 | -0.43% |
| SELL | retest1 | 2025-03-12 10:45:00 | 27708.75 | 2025-03-12 11:00:00 | 27590.74 | PARTIAL | 0.50 | 0.43% |
| SELL | retest1 | 2025-03-12 10:45:00 | 27708.75 | 2025-03-12 14:05:00 | 27480.00 | TARGET_HIT | 0.50 | 0.83% |
| BUY | retest1 | 2025-03-18 10:40:00 | 27580.65 | 2025-03-18 11:15:00 | 27508.83 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2025-03-19 10:10:00 | 27900.00 | 2025-03-19 10:50:00 | 28034.55 | PARTIAL | 0.50 | 0.48% |
| BUY | retest1 | 2025-03-19 10:10:00 | 27900.00 | 2025-03-19 15:20:00 | 28285.45 | TARGET_HIT | 0.50 | 1.38% |
| SELL | retest1 | 2025-03-20 10:15:00 | 27985.15 | 2025-03-20 13:40:00 | 28103.26 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest1 | 2025-03-26 10:30:00 | 28345.15 | 2025-03-26 11:25:00 | 28419.72 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2025-04-01 11:05:00 | 28245.00 | 2025-04-01 11:35:00 | 28113.05 | PARTIAL | 0.50 | 0.47% |
| SELL | retest1 | 2025-04-01 11:05:00 | 28245.00 | 2025-04-01 12:00:00 | 28245.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-04-02 10:45:00 | 28410.05 | 2025-04-02 12:00:00 | 28501.72 | PARTIAL | 0.50 | 0.32% |
| BUY | retest1 | 2025-04-02 10:45:00 | 28410.05 | 2025-04-02 15:20:00 | 28749.00 | TARGET_HIT | 0.50 | 1.19% |
| SELL | retest1 | 2025-04-21 09:40:00 | 30175.00 | 2025-04-21 13:10:00 | 30262.74 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2025-04-23 10:55:00 | 29860.00 | 2025-04-23 11:00:00 | 29924.99 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2025-04-25 09:50:00 | 29305.00 | 2025-04-25 10:30:00 | 29178.09 | PARTIAL | 0.50 | 0.43% |
| SELL | retest1 | 2025-04-25 09:50:00 | 29305.00 | 2025-04-25 12:15:00 | 29305.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-04-28 10:40:00 | 29610.00 | 2025-04-28 10:55:00 | 29703.88 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2025-04-29 11:00:00 | 29855.00 | 2025-04-29 11:30:00 | 29775.30 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2025-05-05 11:05:00 | 29960.00 | 2025-05-05 11:20:00 | 30059.02 | PARTIAL | 0.50 | 0.33% |
| BUY | retest1 | 2025-05-05 11:05:00 | 29960.00 | 2025-05-05 15:20:00 | 30170.00 | TARGET_HIT | 0.50 | 0.70% |
