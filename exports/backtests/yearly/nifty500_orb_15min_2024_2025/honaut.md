# Honeywell Automation India Ltd. (HONAUT)

## Backtest Summary

- **Window:** 2024-05-13 09:15:00 → 2026-05-08 15:25:00 (36871 bars)
- **Last close:** 30210.00
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
| ENTRY1 | 91 |
| ENTRY2 | 0 |
| PARTIAL | 36 |
| TARGET_HIT | 17 |
| STOP_HIT | 74 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 127 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 53 / 74
- **Target hits / Stop hits / Partials:** 17 / 74 / 36
- **Avg / median % per leg:** 0.12% / 0.00%
- **Sum % (uncompounded):** 15.02%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 47 | 18 | 38.3% | 6 | 29 | 12 | 0.06% | 3.0% |
| BUY @ 2nd Alert (retest1) | 47 | 18 | 38.3% | 6 | 29 | 12 | 0.06% | 3.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 80 | 35 | 43.8% | 11 | 45 | 24 | 0.15% | 12.0% |
| SELL @ 2nd Alert (retest1) | 80 | 35 | 43.8% | 11 | 45 | 24 | 0.15% | 12.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 127 | 53 | 41.7% | 17 | 74 | 36 | 0.12% | 15.0% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-05-23 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-23 09:40:00 | 53888.40 | 53509.27 | 0.00 | ORB-long ORB[53000.00,53510.00] vol=2.7x ATR=230.74 |
| Stop hit — per-position SL triggered | 2024-05-23 09:45:00 | 53657.66 | 53523.61 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2024-05-30 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-30 10:30:00 | 52497.45 | 52340.79 | 0.00 | ORB-long ORB[52019.45,52446.30] vol=5.1x ATR=247.92 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-30 11:10:00 | 52869.33 | 52406.34 | 0.00 | T1 1.5R @ 52869.33 |
| Target hit | 2024-05-30 14:05:00 | 52742.55 | 52797.15 | 0.00 | Trail-exit close<VWAP |

### Cycle 3 — SELL (started 2024-05-31 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-31 10:10:00 | 52078.00 | 52357.78 | 0.00 | ORB-short ORB[52200.00,52747.30] vol=1.7x ATR=176.53 |
| Stop hit — per-position SL triggered | 2024-05-31 10:20:00 | 52254.53 | 52304.31 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2024-06-06 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-06 10:05:00 | 52447.75 | 51844.90 | 0.00 | ORB-long ORB[51024.70,51800.00] vol=3.7x ATR=262.23 |
| Stop hit — per-position SL triggered | 2024-06-06 10:10:00 | 52185.52 | 51887.24 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2024-06-10 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-10 10:20:00 | 53400.00 | 52778.68 | 0.00 | ORB-long ORB[52339.15,52950.00] vol=3.5x ATR=267.59 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-10 10:30:00 | 53801.38 | 53276.65 | 0.00 | T1 1.5R @ 53801.38 |
| Target hit | 2024-06-10 11:10:00 | 53750.00 | 53786.04 | 0.00 | Trail-exit close<VWAP |

### Cycle 6 — BUY (started 2024-06-14 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-14 09:50:00 | 55108.25 | 54933.40 | 0.00 | ORB-long ORB[54639.05,55059.10] vol=2.8x ATR=142.08 |
| Stop hit — per-position SL triggered | 2024-06-14 10:05:00 | 54966.17 | 54968.88 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2024-06-20 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-20 10:25:00 | 56006.45 | 55676.82 | 0.00 | ORB-long ORB[55277.35,55948.00] vol=1.7x ATR=204.58 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-20 10:30:00 | 56313.33 | 55767.32 | 0.00 | T1 1.5R @ 56313.33 |
| Stop hit — per-position SL triggered | 2024-06-20 11:45:00 | 56006.45 | 56056.79 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2024-06-25 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-25 09:35:00 | 57635.90 | 57334.88 | 0.00 | ORB-long ORB[56900.00,57449.85] vol=2.1x ATR=217.89 |
| Stop hit — per-position SL triggered | 2024-06-25 09:40:00 | 57418.01 | 57355.06 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2024-06-26 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-26 09:30:00 | 57841.60 | 58085.40 | 0.00 | ORB-short ORB[57985.25,58624.95] vol=1.6x ATR=202.13 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-26 09:35:00 | 57538.40 | 57987.41 | 0.00 | T1 1.5R @ 57538.40 |
| Stop hit — per-position SL triggered | 2024-06-26 09:55:00 | 57841.60 | 57794.60 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2024-07-02 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-02 09:30:00 | 56778.40 | 56967.56 | 0.00 | ORB-short ORB[56800.00,57200.00] vol=1.8x ATR=212.10 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-02 10:00:00 | 56460.25 | 56764.91 | 0.00 | T1 1.5R @ 56460.25 |
| Stop hit — per-position SL triggered | 2024-07-02 10:30:00 | 56778.40 | 56700.34 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2024-07-04 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-04 10:35:00 | 57133.90 | 56770.73 | 0.00 | ORB-long ORB[56600.00,56999.95] vol=4.0x ATR=195.89 |
| Stop hit — per-position SL triggered | 2024-07-04 10:40:00 | 56938.01 | 56805.67 | 0.00 | SL hit |

### Cycle 12 — SELL (started 2024-07-08 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-08 10:50:00 | 56900.00 | 57615.71 | 0.00 | ORB-short ORB[57660.00,58500.00] vol=3.6x ATR=204.32 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-08 11:00:00 | 56593.52 | 57360.37 | 0.00 | T1 1.5R @ 56593.52 |
| Stop hit — per-position SL triggered | 2024-07-08 11:25:00 | 56900.00 | 57209.11 | 0.00 | SL hit |

### Cycle 13 — SELL (started 2024-07-12 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-12 10:25:00 | 56678.10 | 56897.42 | 0.00 | ORB-short ORB[56900.05,57300.00] vol=1.8x ATR=113.54 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-12 10:45:00 | 56507.79 | 56766.85 | 0.00 | T1 1.5R @ 56507.79 |
| Stop hit — per-position SL triggered | 2024-07-12 11:05:00 | 56678.10 | 56745.51 | 0.00 | SL hit |

### Cycle 14 — SELL (started 2024-07-15 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-15 09:55:00 | 56000.00 | 56323.91 | 0.00 | ORB-short ORB[56400.00,56999.00] vol=1.6x ATR=203.40 |
| Stop hit — per-position SL triggered | 2024-07-15 10:05:00 | 56203.40 | 56302.15 | 0.00 | SL hit |

### Cycle 15 — SELL (started 2024-07-16 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-16 09:40:00 | 56252.55 | 56357.66 | 0.00 | ORB-short ORB[56300.00,56661.40] vol=3.2x ATR=150.64 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-16 10:35:00 | 56026.59 | 56252.66 | 0.00 | T1 1.5R @ 56026.59 |
| Stop hit — per-position SL triggered | 2024-07-16 11:10:00 | 56252.55 | 56242.72 | 0.00 | SL hit |

### Cycle 16 — SELL (started 2024-07-18 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-18 09:30:00 | 55890.30 | 56168.60 | 0.00 | ORB-short ORB[56080.30,56650.00] vol=3.0x ATR=182.36 |
| Stop hit — per-position SL triggered | 2024-07-18 09:40:00 | 56072.66 | 56114.04 | 0.00 | SL hit |

### Cycle 17 — BUY (started 2024-07-24 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-24 10:35:00 | 54500.00 | 54053.94 | 0.00 | ORB-long ORB[53502.05,54281.55] vol=2.8x ATR=212.92 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-24 10:55:00 | 54819.38 | 54134.97 | 0.00 | T1 1.5R @ 54819.38 |
| Target hit | 2024-07-24 14:15:00 | 54556.10 | 54793.30 | 0.00 | Trail-exit close<VWAP |

### Cycle 18 — SELL (started 2024-07-30 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-30 09:55:00 | 54200.00 | 54456.94 | 0.00 | ORB-short ORB[54300.00,54757.20] vol=2.3x ATR=193.19 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-30 12:55:00 | 53910.21 | 54320.15 | 0.00 | T1 1.5R @ 53910.21 |
| Target hit | 2024-07-30 15:20:00 | 53990.60 | 53953.39 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 19 — SELL (started 2024-08-01 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-01 09:35:00 | 54416.80 | 54681.15 | 0.00 | ORB-short ORB[54587.65,55193.60] vol=1.5x ATR=201.14 |
| Stop hit — per-position SL triggered | 2024-08-01 09:40:00 | 54617.94 | 54672.73 | 0.00 | SL hit |

### Cycle 20 — SELL (started 2024-08-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-09 11:15:00 | 51140.05 | 51441.45 | 0.00 | ORB-short ORB[51500.00,52224.95] vol=4.3x ATR=93.41 |
| Stop hit — per-position SL triggered | 2024-08-09 11:20:00 | 51233.46 | 51423.27 | 0.00 | SL hit |

### Cycle 21 — BUY (started 2024-08-13 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-13 09:40:00 | 51559.30 | 51345.15 | 0.00 | ORB-long ORB[51016.05,51450.05] vol=2.1x ATR=150.74 |
| Stop hit — per-position SL triggered | 2024-08-13 09:50:00 | 51408.56 | 51394.26 | 0.00 | SL hit |

### Cycle 22 — SELL (started 2024-08-14 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-14 09:30:00 | 50978.45 | 51184.92 | 0.00 | ORB-short ORB[51089.65,51698.95] vol=2.8x ATR=180.88 |
| Stop hit — per-position SL triggered | 2024-08-14 10:20:00 | 51159.33 | 51071.51 | 0.00 | SL hit |

### Cycle 23 — BUY (started 2024-08-19 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-19 09:35:00 | 52800.00 | 52311.84 | 0.00 | ORB-long ORB[51616.55,52300.00] vol=2.4x ATR=241.13 |
| Stop hit — per-position SL triggered | 2024-08-19 09:40:00 | 52558.87 | 52384.60 | 0.00 | SL hit |

### Cycle 24 — SELL (started 2024-08-20 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-20 09:50:00 | 51918.00 | 52155.47 | 0.00 | ORB-short ORB[52044.25,52500.00] vol=1.5x ATR=161.75 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-20 09:55:00 | 51675.38 | 52046.00 | 0.00 | T1 1.5R @ 51675.38 |
| Stop hit — per-position SL triggered | 2024-08-20 10:00:00 | 51918.00 | 52044.88 | 0.00 | SL hit |

### Cycle 25 — SELL (started 2024-08-22 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-22 10:45:00 | 52346.65 | 52534.90 | 0.00 | ORB-short ORB[52398.20,52919.95] vol=2.1x ATR=123.97 |
| Stop hit — per-position SL triggered | 2024-08-22 11:55:00 | 52470.62 | 52505.62 | 0.00 | SL hit |

### Cycle 26 — SELL (started 2024-08-23 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-23 10:40:00 | 52300.00 | 52445.98 | 0.00 | ORB-short ORB[52472.05,52910.90] vol=2.9x ATR=151.32 |
| Stop hit — per-position SL triggered | 2024-08-23 10:45:00 | 52451.32 | 52473.21 | 0.00 | SL hit |

### Cycle 27 — SELL (started 2024-08-26 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-26 09:30:00 | 52332.55 | 52478.77 | 0.00 | ORB-short ORB[52368.55,52825.80] vol=2.3x ATR=189.30 |
| Stop hit — per-position SL triggered | 2024-08-26 09:40:00 | 52521.85 | 52477.45 | 0.00 | SL hit |

### Cycle 28 — BUY (started 2024-08-27 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-27 10:45:00 | 52650.00 | 52388.49 | 0.00 | ORB-long ORB[52107.05,52590.00] vol=3.8x ATR=146.31 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-27 10:50:00 | 52869.46 | 52504.37 | 0.00 | T1 1.5R @ 52869.46 |
| Stop hit — per-position SL triggered | 2024-08-27 11:00:00 | 52650.00 | 52510.65 | 0.00 | SL hit |

### Cycle 29 — SELL (started 2024-08-28 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-28 09:30:00 | 52021.00 | 52336.67 | 0.00 | ORB-short ORB[52344.25,52750.00] vol=1.5x ATR=153.85 |
| Stop hit — per-position SL triggered | 2024-08-28 10:00:00 | 52174.85 | 52203.43 | 0.00 | SL hit |

### Cycle 30 — BUY (started 2024-09-03 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-03 09:30:00 | 51000.00 | 50845.01 | 0.00 | ORB-long ORB[50298.10,50840.95] vol=1.7x ATR=202.55 |
| Stop hit — per-position SL triggered | 2024-09-03 09:40:00 | 50797.45 | 50841.81 | 0.00 | SL hit |

### Cycle 31 — SELL (started 2024-09-06 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-06 11:05:00 | 49600.00 | 49814.64 | 0.00 | ORB-short ORB[49818.05,50299.00] vol=2.7x ATR=138.16 |
| Stop hit — per-position SL triggered | 2024-09-06 11:55:00 | 49738.16 | 49777.22 | 0.00 | SL hit |

### Cycle 32 — BUY (started 2024-09-10 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-10 09:55:00 | 50700.00 | 50450.36 | 0.00 | ORB-long ORB[49885.55,50520.00] vol=2.1x ATR=176.02 |
| Stop hit — per-position SL triggered | 2024-09-10 10:05:00 | 50523.98 | 50458.30 | 0.00 | SL hit |

### Cycle 33 — BUY (started 2024-09-13 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-13 10:10:00 | 51381.70 | 51211.32 | 0.00 | ORB-long ORB[51014.35,51301.05] vol=2.2x ATR=151.38 |
| Stop hit — per-position SL triggered | 2024-09-13 10:20:00 | 51230.32 | 51262.69 | 0.00 | SL hit |

### Cycle 34 — SELL (started 2024-09-16 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-16 10:10:00 | 49893.10 | 50326.87 | 0.00 | ORB-short ORB[50288.45,50900.00] vol=2.5x ATR=144.72 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-16 11:45:00 | 49676.03 | 50056.14 | 0.00 | T1 1.5R @ 49676.03 |
| Target hit | 2024-09-16 14:25:00 | 49714.00 | 49694.50 | 0.00 | Trail-exit close>VWAP |

### Cycle 35 — SELL (started 2024-09-18 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-18 10:50:00 | 49640.00 | 49853.46 | 0.00 | ORB-short ORB[49854.05,50084.25] vol=2.2x ATR=87.55 |
| Stop hit — per-position SL triggered | 2024-09-18 10:55:00 | 49727.55 | 49851.83 | 0.00 | SL hit |

### Cycle 36 — SELL (started 2024-09-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-19 10:15:00 | 49139.25 | 49481.68 | 0.00 | ORB-short ORB[49696.55,49999.80] vol=4.1x ATR=130.03 |
| Stop hit — per-position SL triggered | 2024-09-19 10:20:00 | 49269.28 | 49480.37 | 0.00 | SL hit |

### Cycle 37 — SELL (started 2024-09-20 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-20 10:30:00 | 49215.00 | 49500.92 | 0.00 | ORB-short ORB[49600.00,50124.95] vol=1.5x ATR=129.16 |
| Stop hit — per-position SL triggered | 2024-09-20 10:55:00 | 49344.16 | 49423.51 | 0.00 | SL hit |

### Cycle 38 — SELL (started 2024-09-25 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-25 09:35:00 | 49700.30 | 49929.15 | 0.00 | ORB-short ORB[49798.20,50300.00] vol=1.5x ATR=119.75 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-25 09:55:00 | 49520.68 | 49879.82 | 0.00 | T1 1.5R @ 49520.68 |
| Stop hit — per-position SL triggered | 2024-09-25 10:05:00 | 49700.30 | 49825.45 | 0.00 | SL hit |

### Cycle 39 — BUY (started 2024-09-26 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-26 10:40:00 | 49899.90 | 49702.53 | 0.00 | ORB-long ORB[49575.00,49884.30] vol=2.2x ATR=98.96 |
| Stop hit — per-position SL triggered | 2024-09-26 11:15:00 | 49800.94 | 49765.51 | 0.00 | SL hit |

### Cycle 40 — SELL (started 2024-10-07 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-07 09:50:00 | 47792.50 | 47994.46 | 0.00 | ORB-short ORB[47900.00,48567.45] vol=1.5x ATR=159.63 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-07 10:30:00 | 47553.06 | 47852.67 | 0.00 | T1 1.5R @ 47553.06 |
| Target hit | 2024-10-07 11:25:00 | 47684.95 | 47557.74 | 0.00 | Trail-exit close>VWAP |

### Cycle 41 — BUY (started 2024-10-09 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-09 10:00:00 | 48869.00 | 48698.71 | 0.00 | ORB-long ORB[48273.65,48749.00] vol=14.0x ATR=126.15 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-09 10:05:00 | 49058.22 | 48753.48 | 0.00 | T1 1.5R @ 49058.22 |
| Stop hit — per-position SL triggered | 2024-10-09 10:10:00 | 48869.00 | 48760.82 | 0.00 | SL hit |

### Cycle 42 — SELL (started 2024-10-11 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-11 11:10:00 | 50109.80 | 50121.45 | 0.00 | ORB-short ORB[50150.00,50507.35] vol=1.6x ATR=84.51 |
| Stop hit — per-position SL triggered | 2024-10-11 11:15:00 | 50194.31 | 50160.94 | 0.00 | SL hit |

### Cycle 43 — SELL (started 2024-10-14 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-14 11:05:00 | 50098.50 | 50381.17 | 0.00 | ORB-short ORB[50305.60,50554.90] vol=2.9x ATR=89.84 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-14 11:10:00 | 49963.74 | 50349.42 | 0.00 | T1 1.5R @ 49963.74 |
| Stop hit — per-position SL triggered | 2024-10-14 15:00:00 | 50098.50 | 50153.42 | 0.00 | SL hit |

### Cycle 44 — SELL (started 2024-10-17 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-17 09:30:00 | 49507.75 | 49674.89 | 0.00 | ORB-short ORB[49550.20,49850.00] vol=1.6x ATR=121.64 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-17 09:55:00 | 49325.29 | 49514.62 | 0.00 | T1 1.5R @ 49325.29 |
| Target hit | 2024-10-17 10:30:00 | 49480.15 | 49422.38 | 0.00 | Trail-exit close>VWAP |

### Cycle 45 — SELL (started 2024-10-22 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-22 10:30:00 | 51115.50 | 51529.20 | 0.00 | ORB-short ORB[51260.00,51999.00] vol=2.1x ATR=135.02 |
| Stop hit — per-position SL triggered | 2024-10-22 10:40:00 | 51250.52 | 51496.66 | 0.00 | SL hit |

### Cycle 46 — SELL (started 2024-10-23 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-23 09:45:00 | 50374.80 | 50706.96 | 0.00 | ORB-short ORB[50640.00,51324.00] vol=1.5x ATR=176.05 |
| Stop hit — per-position SL triggered | 2024-10-23 09:50:00 | 50550.85 | 50726.51 | 0.00 | SL hit |

### Cycle 47 — SELL (started 2024-10-25 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-25 10:05:00 | 50060.50 | 50357.74 | 0.00 | ORB-short ORB[50251.00,50899.80] vol=5.6x ATR=171.03 |
| Stop hit — per-position SL triggered | 2024-10-25 10:20:00 | 50231.53 | 50305.34 | 0.00 | SL hit |

### Cycle 48 — SELL (started 2024-10-29 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-29 11:10:00 | 48248.70 | 48590.98 | 0.00 | ORB-short ORB[48840.10,49098.95] vol=5.6x ATR=123.93 |
| Stop hit — per-position SL triggered | 2024-10-29 13:25:00 | 48372.63 | 48494.35 | 0.00 | SL hit |

### Cycle 49 — SELL (started 2024-11-05 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-05 11:10:00 | 44996.70 | 45240.06 | 0.00 | ORB-short ORB[45177.55,45692.45] vol=3.9x ATR=91.04 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-05 11:25:00 | 44860.14 | 45221.19 | 0.00 | T1 1.5R @ 44860.14 |
| Stop hit — per-position SL triggered | 2024-11-05 11:35:00 | 44996.70 | 45177.44 | 0.00 | SL hit |

### Cycle 50 — SELL (started 2024-11-13 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-13 09:30:00 | 42990.00 | 43252.29 | 0.00 | ORB-short ORB[43109.20,43605.85] vol=1.7x ATR=124.61 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-13 09:40:00 | 42803.08 | 43150.06 | 0.00 | T1 1.5R @ 42803.08 |
| Target hit | 2024-11-13 15:20:00 | 41990.45 | 42217.63 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 51 — SELL (started 2024-11-21 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-21 09:45:00 | 41547.70 | 41796.22 | 0.00 | ORB-short ORB[41755.00,42326.00] vol=1.5x ATR=216.43 |
| Stop hit — per-position SL triggered | 2024-11-21 09:50:00 | 41764.13 | 41669.26 | 0.00 | SL hit |

### Cycle 52 — SELL (started 2024-12-04 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-04 11:10:00 | 40928.45 | 41453.53 | 0.00 | ORB-short ORB[41240.00,41849.75] vol=8.2x ATR=126.54 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-04 11:15:00 | 40738.63 | 41362.39 | 0.00 | T1 1.5R @ 40738.63 |
| Stop hit — per-position SL triggered | 2024-12-04 15:05:00 | 40928.45 | 41101.90 | 0.00 | SL hit |

### Cycle 53 — SELL (started 2024-12-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-06 11:15:00 | 40898.65 | 40941.48 | 0.00 | ORB-short ORB[40992.80,41257.65] vol=2.0x ATR=88.18 |
| Stop hit — per-position SL triggered | 2024-12-06 13:40:00 | 40986.83 | 40925.93 | 0.00 | SL hit |

### Cycle 54 — SELL (started 2024-12-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-09 11:15:00 | 40588.00 | 40718.07 | 0.00 | ORB-short ORB[40750.00,41150.00] vol=1.9x ATR=93.48 |
| Stop hit — per-position SL triggered | 2024-12-09 11:25:00 | 40681.48 | 40717.15 | 0.00 | SL hit |

### Cycle 55 — SELL (started 2024-12-12 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-12 09:45:00 | 41181.00 | 41435.96 | 0.00 | ORB-short ORB[41514.40,41798.90] vol=2.9x ATR=125.62 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-12 11:00:00 | 40992.58 | 41267.70 | 0.00 | T1 1.5R @ 40992.58 |
| Target hit | 2024-12-12 15:20:00 | 40811.00 | 41035.84 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 56 — SELL (started 2024-12-13 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-13 09:35:00 | 40550.00 | 40782.02 | 0.00 | ORB-short ORB[40725.00,41086.30] vol=1.5x ATR=108.16 |
| Stop hit — per-position SL triggered | 2024-12-13 10:50:00 | 40658.16 | 40615.50 | 0.00 | SL hit |

### Cycle 57 — SELL (started 2024-12-16 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-16 09:35:00 | 40595.25 | 40913.06 | 0.00 | ORB-short ORB[40760.40,41148.00] vol=2.2x ATR=135.72 |
| Stop hit — per-position SL triggered | 2024-12-16 09:40:00 | 40730.97 | 40856.82 | 0.00 | SL hit |

### Cycle 58 — SELL (started 2024-12-17 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-17 09:35:00 | 40881.45 | 41139.37 | 0.00 | ORB-short ORB[40976.55,41361.65] vol=1.9x ATR=117.83 |
| Stop hit — per-position SL triggered | 2024-12-17 09:40:00 | 40999.28 | 41124.76 | 0.00 | SL hit |

### Cycle 59 — SELL (started 2024-12-18 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-18 09:30:00 | 40174.65 | 40329.93 | 0.00 | ORB-short ORB[40272.00,40683.00] vol=1.7x ATR=161.22 |
| Stop hit — per-position SL triggered | 2024-12-18 09:40:00 | 40335.87 | 40321.78 | 0.00 | SL hit |

### Cycle 60 — SELL (started 2024-12-27 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-27 09:40:00 | 41943.45 | 42204.41 | 0.00 | ORB-short ORB[42020.00,42612.10] vol=2.8x ATR=165.31 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-27 09:55:00 | 41695.49 | 42139.83 | 0.00 | T1 1.5R @ 41695.49 |
| Target hit | 2024-12-27 15:20:00 | 41153.05 | 41632.49 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 61 — BUY (started 2025-01-02 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-02 11:05:00 | 42537.50 | 42369.81 | 0.00 | ORB-long ORB[42086.25,42450.95] vol=7.8x ATR=73.54 |
| Stop hit — per-position SL triggered | 2025-01-02 11:10:00 | 42463.96 | 42415.86 | 0.00 | SL hit |

### Cycle 62 — BUY (started 2025-01-03 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-03 09:30:00 | 43312.25 | 42977.93 | 0.00 | ORB-long ORB[42430.00,43050.45] vol=4.5x ATR=148.05 |
| Stop hit — per-position SL triggered | 2025-01-03 09:35:00 | 43164.20 | 43038.52 | 0.00 | SL hit |

### Cycle 63 — BUY (started 2025-01-08 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-08 09:40:00 | 43009.90 | 42869.60 | 0.00 | ORB-long ORB[42567.00,42998.90] vol=1.7x ATR=161.44 |
| Stop hit — per-position SL triggered | 2025-01-08 10:10:00 | 42848.46 | 42913.20 | 0.00 | SL hit |

### Cycle 64 — SELL (started 2025-01-10 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-10 09:40:00 | 41807.90 | 42034.38 | 0.00 | ORB-short ORB[42000.00,42573.30] vol=1.5x ATR=164.76 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-10 09:50:00 | 41560.76 | 41965.68 | 0.00 | T1 1.5R @ 41560.76 |
| Target hit | 2025-01-10 11:15:00 | 41781.30 | 41750.05 | 0.00 | Trail-exit close>VWAP |

### Cycle 65 — SELL (started 2025-01-13 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-13 09:35:00 | 40661.45 | 41204.94 | 0.00 | ORB-short ORB[41075.05,41560.90] vol=1.8x ATR=182.17 |
| Stop hit — per-position SL triggered | 2025-01-13 09:40:00 | 40843.62 | 41147.88 | 0.00 | SL hit |

### Cycle 66 — SELL (started 2025-01-16 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-16 11:10:00 | 40857.70 | 41017.45 | 0.00 | ORB-short ORB[40886.05,41285.45] vol=1.6x ATR=67.19 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-16 15:05:00 | 40756.91 | 40957.00 | 0.00 | T1 1.5R @ 40756.91 |
| Target hit | 2025-01-16 15:20:00 | 40750.00 | 40883.86 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 67 — SELL (started 2025-01-21 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-21 11:10:00 | 40390.25 | 40565.72 | 0.00 | ORB-short ORB[40598.00,40900.95] vol=2.0x ATR=92.01 |
| Stop hit — per-position SL triggered | 2025-01-21 11:45:00 | 40482.26 | 40549.96 | 0.00 | SL hit |

### Cycle 68 — BUY (started 2025-01-23 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-23 09:55:00 | 40860.90 | 40590.88 | 0.00 | ORB-long ORB[40351.30,40555.75] vol=1.7x ATR=122.72 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-23 10:10:00 | 41044.98 | 40775.18 | 0.00 | T1 1.5R @ 41044.98 |
| Target hit | 2025-01-23 15:20:00 | 41600.00 | 41394.34 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 69 — SELL (started 2025-02-04 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-04 10:35:00 | 38025.65 | 38265.22 | 0.00 | ORB-short ORB[38275.05,38594.25] vol=1.7x ATR=121.24 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-04 10:45:00 | 37843.80 | 38211.89 | 0.00 | T1 1.5R @ 37843.80 |
| Stop hit — per-position SL triggered | 2025-02-04 11:50:00 | 38025.65 | 38043.05 | 0.00 | SL hit |

### Cycle 70 — SELL (started 2025-02-12 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-12 09:30:00 | 36432.60 | 36637.02 | 0.00 | ORB-short ORB[36469.00,36852.00] vol=2.1x ATR=154.86 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-12 09:35:00 | 36200.31 | 36563.93 | 0.00 | T1 1.5R @ 36200.31 |
| Target hit | 2025-02-12 15:20:00 | 34915.45 | 35413.88 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 71 — BUY (started 2025-02-14 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-14 09:55:00 | 35690.40 | 35485.72 | 0.00 | ORB-long ORB[35237.65,35669.95] vol=1.5x ATR=134.53 |
| Stop hit — per-position SL triggered | 2025-02-14 10:00:00 | 35555.87 | 35497.78 | 0.00 | SL hit |

### Cycle 72 — BUY (started 2025-02-19 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-19 09:30:00 | 33880.00 | 33698.15 | 0.00 | ORB-long ORB[33390.00,33858.85] vol=1.8x ATR=120.28 |
| Stop hit — per-position SL triggered | 2025-02-19 09:40:00 | 33759.72 | 33724.18 | 0.00 | SL hit |

### Cycle 73 — BUY (started 2025-03-04 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-04 09:30:00 | 34048.00 | 33699.97 | 0.00 | ORB-long ORB[33409.15,33830.00] vol=1.8x ATR=138.30 |
| Stop hit — per-position SL triggered | 2025-03-04 09:35:00 | 33909.70 | 33710.12 | 0.00 | SL hit |

### Cycle 74 — BUY (started 2025-03-05 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-05 09:45:00 | 34090.25 | 33985.21 | 0.00 | ORB-long ORB[33612.10,34000.00] vol=3.3x ATR=85.18 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-05 09:55:00 | 34218.03 | 34057.22 | 0.00 | T1 1.5R @ 34218.03 |
| Stop hit — per-position SL triggered | 2025-03-05 10:40:00 | 34090.25 | 34091.70 | 0.00 | SL hit |

### Cycle 75 — BUY (started 2025-03-07 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-07 09:30:00 | 35679.90 | 35524.19 | 0.00 | ORB-long ORB[35140.85,35625.00] vol=1.8x ATR=118.68 |
| Stop hit — per-position SL triggered | 2025-03-07 09:40:00 | 35561.22 | 35545.60 | 0.00 | SL hit |

### Cycle 76 — SELL (started 2025-03-12 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-12 10:30:00 | 34062.85 | 34191.41 | 0.00 | ORB-short ORB[34131.40,34500.10] vol=1.7x ATR=91.32 |
| Stop hit — per-position SL triggered | 2025-03-12 10:45:00 | 34154.17 | 34183.54 | 0.00 | SL hit |

### Cycle 77 — SELL (started 2025-03-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-17 10:15:00 | 33748.45 | 33923.68 | 0.00 | ORB-short ORB[33885.10,34200.00] vol=2.0x ATR=121.68 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-17 10:25:00 | 33565.93 | 33878.61 | 0.00 | T1 1.5R @ 33565.93 |
| Stop hit — per-position SL triggered | 2025-03-17 11:00:00 | 33748.45 | 33819.14 | 0.00 | SL hit |

### Cycle 78 — SELL (started 2025-03-21 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-21 11:10:00 | 33885.70 | 34123.53 | 0.00 | ORB-short ORB[34000.00,34275.00] vol=2.2x ATR=100.69 |
| Stop hit — per-position SL triggered | 2025-03-21 12:20:00 | 33986.39 | 34067.56 | 0.00 | SL hit |

### Cycle 79 — SELL (started 2025-03-25 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-25 11:05:00 | 33648.05 | 33911.02 | 0.00 | ORB-short ORB[33751.00,34164.45] vol=1.6x ATR=127.72 |
| Stop hit — per-position SL triggered | 2025-03-25 11:10:00 | 33775.77 | 33908.82 | 0.00 | SL hit |

### Cycle 80 — BUY (started 2025-04-01 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-01 10:30:00 | 33883.10 | 33843.43 | 0.00 | ORB-long ORB[33400.05,33778.50] vol=1.6x ATR=126.13 |
| Stop hit — per-position SL triggered | 2025-04-01 10:45:00 | 33756.97 | 33840.89 | 0.00 | SL hit |

### Cycle 81 — BUY (started 2025-04-09 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-09 10:25:00 | 33629.70 | 33459.08 | 0.00 | ORB-long ORB[33250.40,33510.85] vol=3.0x ATR=105.84 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-09 11:45:00 | 33788.45 | 33534.95 | 0.00 | T1 1.5R @ 33788.45 |
| Target hit | 2025-04-09 15:20:00 | 33989.80 | 33683.37 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 82 — SELL (started 2025-04-11 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-11 09:45:00 | 34001.95 | 34088.81 | 0.00 | ORB-short ORB[34010.00,34249.95] vol=2.4x ATR=100.52 |
| Stop hit — per-position SL triggered | 2025-04-11 09:50:00 | 34102.47 | 34085.28 | 0.00 | SL hit |

### Cycle 83 — BUY (started 2025-04-17 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-17 10:40:00 | 34800.00 | 34653.22 | 0.00 | ORB-long ORB[34030.00,34450.00] vol=18.7x ATR=92.97 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-17 11:05:00 | 34939.46 | 34668.90 | 0.00 | T1 1.5R @ 34939.46 |
| Stop hit — per-position SL triggered | 2025-04-17 11:45:00 | 34800.00 | 34687.97 | 0.00 | SL hit |

### Cycle 84 — BUY (started 2025-04-22 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-22 09:35:00 | 34800.00 | 34587.64 | 0.00 | ORB-long ORB[34405.00,34795.00] vol=2.2x ATR=90.04 |
| Stop hit — per-position SL triggered | 2025-04-22 10:05:00 | 34709.96 | 34650.19 | 0.00 | SL hit |

### Cycle 85 — SELL (started 2025-04-23 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-23 10:40:00 | 34330.00 | 34437.89 | 0.00 | ORB-short ORB[34400.00,34880.00] vol=4.3x ATR=102.49 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-23 10:55:00 | 34176.26 | 34367.82 | 0.00 | T1 1.5R @ 34176.26 |
| Target hit | 2025-04-23 14:45:00 | 34320.00 | 34298.99 | 0.00 | Trail-exit close>VWAP |

### Cycle 86 — BUY (started 2025-04-24 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-24 10:55:00 | 34580.00 | 34427.32 | 0.00 | ORB-long ORB[34155.00,34445.00] vol=4.8x ATR=85.01 |
| Stop hit — per-position SL triggered | 2025-04-24 11:15:00 | 34494.99 | 34511.20 | 0.00 | SL hit |

### Cycle 87 — SELL (started 2025-04-25 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-25 09:55:00 | 33840.00 | 34231.66 | 0.00 | ORB-short ORB[34400.00,34770.00] vol=1.8x ATR=132.21 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-25 10:30:00 | 33641.68 | 34047.99 | 0.00 | T1 1.5R @ 33641.68 |
| Stop hit — per-position SL triggered | 2025-04-25 12:40:00 | 33840.00 | 33841.32 | 0.00 | SL hit |

### Cycle 88 — BUY (started 2025-04-29 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-29 10:10:00 | 34485.00 | 34344.96 | 0.00 | ORB-long ORB[33930.00,34350.00] vol=5.5x ATR=150.80 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-29 10:20:00 | 34711.20 | 34444.48 | 0.00 | T1 1.5R @ 34711.20 |
| Stop hit — per-position SL triggered | 2025-04-29 10:40:00 | 34485.00 | 34452.25 | 0.00 | SL hit |

### Cycle 89 — BUY (started 2025-04-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-30 11:15:00 | 34485.00 | 34194.08 | 0.00 | ORB-long ORB[33880.00,34315.00] vol=2.0x ATR=95.02 |
| Stop hit — per-position SL triggered | 2025-04-30 15:20:00 | 34450.00 | 34397.93 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 90 — BUY (started 2025-05-05 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-05 09:35:00 | 35165.00 | 35027.25 | 0.00 | ORB-long ORB[34675.00,35130.00] vol=2.1x ATR=113.78 |
| Stop hit — per-position SL triggered | 2025-05-05 10:05:00 | 35051.22 | 35048.70 | 0.00 | SL hit |

### Cycle 91 — BUY (started 2025-05-08 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-08 09:30:00 | 35500.00 | 35328.44 | 0.00 | ORB-long ORB[34955.00,35445.00] vol=2.7x ATR=141.23 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-08 10:05:00 | 35711.84 | 35568.01 | 0.00 | T1 1.5R @ 35711.84 |
| Target hit | 2025-05-08 10:20:00 | 35530.00 | 35613.38 | 0.00 | Trail-exit close<VWAP |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-05-23 09:40:00 | 53888.40 | 2024-05-23 09:45:00 | 53657.66 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest1 | 2024-05-30 10:30:00 | 52497.45 | 2024-05-30 11:10:00 | 52869.33 | PARTIAL | 0.50 | 0.71% |
| BUY | retest1 | 2024-05-30 10:30:00 | 52497.45 | 2024-05-30 14:05:00 | 52742.55 | TARGET_HIT | 0.50 | 0.47% |
| SELL | retest1 | 2024-05-31 10:10:00 | 52078.00 | 2024-05-31 10:20:00 | 52254.53 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2024-06-06 10:05:00 | 52447.75 | 2024-06-06 10:10:00 | 52185.52 | STOP_HIT | 1.00 | -0.50% |
| BUY | retest1 | 2024-06-10 10:20:00 | 53400.00 | 2024-06-10 10:30:00 | 53801.38 | PARTIAL | 0.50 | 0.75% |
| BUY | retest1 | 2024-06-10 10:20:00 | 53400.00 | 2024-06-10 11:10:00 | 53750.00 | TARGET_HIT | 0.50 | 0.66% |
| BUY | retest1 | 2024-06-14 09:50:00 | 55108.25 | 2024-06-14 10:05:00 | 54966.17 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2024-06-20 10:25:00 | 56006.45 | 2024-06-20 10:30:00 | 56313.33 | PARTIAL | 0.50 | 0.55% |
| BUY | retest1 | 2024-06-20 10:25:00 | 56006.45 | 2024-06-20 11:45:00 | 56006.45 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-06-25 09:35:00 | 57635.90 | 2024-06-25 09:40:00 | 57418.01 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest1 | 2024-06-26 09:30:00 | 57841.60 | 2024-06-26 09:35:00 | 57538.40 | PARTIAL | 0.50 | 0.52% |
| SELL | retest1 | 2024-06-26 09:30:00 | 57841.60 | 2024-06-26 09:55:00 | 57841.60 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-07-02 09:30:00 | 56778.40 | 2024-07-02 10:00:00 | 56460.25 | PARTIAL | 0.50 | 0.56% |
| SELL | retest1 | 2024-07-02 09:30:00 | 56778.40 | 2024-07-02 10:30:00 | 56778.40 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-07-04 10:35:00 | 57133.90 | 2024-07-04 10:40:00 | 56938.01 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2024-07-08 10:50:00 | 56900.00 | 2024-07-08 11:00:00 | 56593.52 | PARTIAL | 0.50 | 0.54% |
| SELL | retest1 | 2024-07-08 10:50:00 | 56900.00 | 2024-07-08 11:25:00 | 56900.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-07-12 10:25:00 | 56678.10 | 2024-07-12 10:45:00 | 56507.79 | PARTIAL | 0.50 | 0.30% |
| SELL | retest1 | 2024-07-12 10:25:00 | 56678.10 | 2024-07-12 11:05:00 | 56678.10 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-07-15 09:55:00 | 56000.00 | 2024-07-15 10:05:00 | 56203.40 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2024-07-16 09:40:00 | 56252.55 | 2024-07-16 10:35:00 | 56026.59 | PARTIAL | 0.50 | 0.40% |
| SELL | retest1 | 2024-07-16 09:40:00 | 56252.55 | 2024-07-16 11:10:00 | 56252.55 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-07-18 09:30:00 | 55890.30 | 2024-07-18 09:40:00 | 56072.66 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2024-07-24 10:35:00 | 54500.00 | 2024-07-24 10:55:00 | 54819.38 | PARTIAL | 0.50 | 0.59% |
| BUY | retest1 | 2024-07-24 10:35:00 | 54500.00 | 2024-07-24 14:15:00 | 54556.10 | TARGET_HIT | 0.50 | 0.10% |
| SELL | retest1 | 2024-07-30 09:55:00 | 54200.00 | 2024-07-30 12:55:00 | 53910.21 | PARTIAL | 0.50 | 0.53% |
| SELL | retest1 | 2024-07-30 09:55:00 | 54200.00 | 2024-07-30 15:20:00 | 53990.60 | TARGET_HIT | 0.50 | 0.39% |
| SELL | retest1 | 2024-08-01 09:35:00 | 54416.80 | 2024-08-01 09:40:00 | 54617.94 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest1 | 2024-08-09 11:15:00 | 51140.05 | 2024-08-09 11:20:00 | 51233.46 | STOP_HIT | 1.00 | -0.18% |
| BUY | retest1 | 2024-08-13 09:40:00 | 51559.30 | 2024-08-13 09:50:00 | 51408.56 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2024-08-14 09:30:00 | 50978.45 | 2024-08-14 10:20:00 | 51159.33 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2024-08-19 09:35:00 | 52800.00 | 2024-08-19 09:40:00 | 52558.87 | STOP_HIT | 1.00 | -0.46% |
| SELL | retest1 | 2024-08-20 09:50:00 | 51918.00 | 2024-08-20 09:55:00 | 51675.38 | PARTIAL | 0.50 | 0.47% |
| SELL | retest1 | 2024-08-20 09:50:00 | 51918.00 | 2024-08-20 10:00:00 | 51918.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-08-22 10:45:00 | 52346.65 | 2024-08-22 11:55:00 | 52470.62 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2024-08-23 10:40:00 | 52300.00 | 2024-08-23 10:45:00 | 52451.32 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2024-08-26 09:30:00 | 52332.55 | 2024-08-26 09:40:00 | 52521.85 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2024-08-27 10:45:00 | 52650.00 | 2024-08-27 10:50:00 | 52869.46 | PARTIAL | 0.50 | 0.42% |
| BUY | retest1 | 2024-08-27 10:45:00 | 52650.00 | 2024-08-27 11:00:00 | 52650.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-08-28 09:30:00 | 52021.00 | 2024-08-28 10:00:00 | 52174.85 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2024-09-03 09:30:00 | 51000.00 | 2024-09-03 09:40:00 | 50797.45 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest1 | 2024-09-06 11:05:00 | 49600.00 | 2024-09-06 11:55:00 | 49738.16 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2024-09-10 09:55:00 | 50700.00 | 2024-09-10 10:05:00 | 50523.98 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2024-09-13 10:10:00 | 51381.70 | 2024-09-13 10:20:00 | 51230.32 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2024-09-16 10:10:00 | 49893.10 | 2024-09-16 11:45:00 | 49676.03 | PARTIAL | 0.50 | 0.44% |
| SELL | retest1 | 2024-09-16 10:10:00 | 49893.10 | 2024-09-16 14:25:00 | 49714.00 | TARGET_HIT | 0.50 | 0.36% |
| SELL | retest1 | 2024-09-18 10:50:00 | 49640.00 | 2024-09-18 10:55:00 | 49727.55 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest1 | 2024-09-19 10:15:00 | 49139.25 | 2024-09-19 10:20:00 | 49269.28 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2024-09-20 10:30:00 | 49215.00 | 2024-09-20 10:55:00 | 49344.16 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2024-09-25 09:35:00 | 49700.30 | 2024-09-25 09:55:00 | 49520.68 | PARTIAL | 0.50 | 0.36% |
| SELL | retest1 | 2024-09-25 09:35:00 | 49700.30 | 2024-09-25 10:05:00 | 49700.30 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-09-26 10:40:00 | 49899.90 | 2024-09-26 11:15:00 | 49800.94 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2024-10-07 09:50:00 | 47792.50 | 2024-10-07 10:30:00 | 47553.06 | PARTIAL | 0.50 | 0.50% |
| SELL | retest1 | 2024-10-07 09:50:00 | 47792.50 | 2024-10-07 11:25:00 | 47684.95 | TARGET_HIT | 0.50 | 0.23% |
| BUY | retest1 | 2024-10-09 10:00:00 | 48869.00 | 2024-10-09 10:05:00 | 49058.22 | PARTIAL | 0.50 | 0.39% |
| BUY | retest1 | 2024-10-09 10:00:00 | 48869.00 | 2024-10-09 10:10:00 | 48869.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-10-11 11:10:00 | 50109.80 | 2024-10-11 11:15:00 | 50194.31 | STOP_HIT | 1.00 | -0.17% |
| SELL | retest1 | 2024-10-14 11:05:00 | 50098.50 | 2024-10-14 11:10:00 | 49963.74 | PARTIAL | 0.50 | 0.27% |
| SELL | retest1 | 2024-10-14 11:05:00 | 50098.50 | 2024-10-14 15:00:00 | 50098.50 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-10-17 09:30:00 | 49507.75 | 2024-10-17 09:55:00 | 49325.29 | PARTIAL | 0.50 | 0.37% |
| SELL | retest1 | 2024-10-17 09:30:00 | 49507.75 | 2024-10-17 10:30:00 | 49480.15 | TARGET_HIT | 0.50 | 0.06% |
| SELL | retest1 | 2024-10-22 10:30:00 | 51115.50 | 2024-10-22 10:40:00 | 51250.52 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2024-10-23 09:45:00 | 50374.80 | 2024-10-23 09:50:00 | 50550.85 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2024-10-25 10:05:00 | 50060.50 | 2024-10-25 10:20:00 | 50231.53 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2024-10-29 11:10:00 | 48248.70 | 2024-10-29 13:25:00 | 48372.63 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2024-11-05 11:10:00 | 44996.70 | 2024-11-05 11:25:00 | 44860.14 | PARTIAL | 0.50 | 0.30% |
| SELL | retest1 | 2024-11-05 11:10:00 | 44996.70 | 2024-11-05 11:35:00 | 44996.70 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-11-13 09:30:00 | 42990.00 | 2024-11-13 09:40:00 | 42803.08 | PARTIAL | 0.50 | 0.43% |
| SELL | retest1 | 2024-11-13 09:30:00 | 42990.00 | 2024-11-13 15:20:00 | 41990.45 | TARGET_HIT | 0.50 | 2.33% |
| SELL | retest1 | 2024-11-21 09:45:00 | 41547.70 | 2024-11-21 09:50:00 | 41764.13 | STOP_HIT | 1.00 | -0.52% |
| SELL | retest1 | 2024-12-04 11:10:00 | 40928.45 | 2024-12-04 11:15:00 | 40738.63 | PARTIAL | 0.50 | 0.46% |
| SELL | retest1 | 2024-12-04 11:10:00 | 40928.45 | 2024-12-04 15:05:00 | 40928.45 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-12-06 11:15:00 | 40898.65 | 2024-12-06 13:40:00 | 40986.83 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2024-12-09 11:15:00 | 40588.00 | 2024-12-09 11:25:00 | 40681.48 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2024-12-12 09:45:00 | 41181.00 | 2024-12-12 11:00:00 | 40992.58 | PARTIAL | 0.50 | 0.46% |
| SELL | retest1 | 2024-12-12 09:45:00 | 41181.00 | 2024-12-12 15:20:00 | 40811.00 | TARGET_HIT | 0.50 | 0.90% |
| SELL | retest1 | 2024-12-13 09:35:00 | 40550.00 | 2024-12-13 10:50:00 | 40658.16 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2024-12-16 09:35:00 | 40595.25 | 2024-12-16 09:40:00 | 40730.97 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2024-12-17 09:35:00 | 40881.45 | 2024-12-17 09:40:00 | 40999.28 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2024-12-18 09:30:00 | 40174.65 | 2024-12-18 09:40:00 | 40335.87 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest1 | 2024-12-27 09:40:00 | 41943.45 | 2024-12-27 09:55:00 | 41695.49 | PARTIAL | 0.50 | 0.59% |
| SELL | retest1 | 2024-12-27 09:40:00 | 41943.45 | 2024-12-27 15:20:00 | 41153.05 | TARGET_HIT | 0.50 | 1.88% |
| BUY | retest1 | 2025-01-02 11:05:00 | 42537.50 | 2025-01-02 11:10:00 | 42463.96 | STOP_HIT | 1.00 | -0.17% |
| BUY | retest1 | 2025-01-03 09:30:00 | 43312.25 | 2025-01-03 09:35:00 | 43164.20 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2025-01-08 09:40:00 | 43009.90 | 2025-01-08 10:10:00 | 42848.46 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest1 | 2025-01-10 09:40:00 | 41807.90 | 2025-01-10 09:50:00 | 41560.76 | PARTIAL | 0.50 | 0.59% |
| SELL | retest1 | 2025-01-10 09:40:00 | 41807.90 | 2025-01-10 11:15:00 | 41781.30 | TARGET_HIT | 0.50 | 0.06% |
| SELL | retest1 | 2025-01-13 09:35:00 | 40661.45 | 2025-01-13 09:40:00 | 40843.62 | STOP_HIT | 1.00 | -0.45% |
| SELL | retest1 | 2025-01-16 11:10:00 | 40857.70 | 2025-01-16 15:05:00 | 40756.91 | PARTIAL | 0.50 | 0.25% |
| SELL | retest1 | 2025-01-16 11:10:00 | 40857.70 | 2025-01-16 15:20:00 | 40750.00 | TARGET_HIT | 0.50 | 0.26% |
| SELL | retest1 | 2025-01-21 11:10:00 | 40390.25 | 2025-01-21 11:45:00 | 40482.26 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2025-01-23 09:55:00 | 40860.90 | 2025-01-23 10:10:00 | 41044.98 | PARTIAL | 0.50 | 0.45% |
| BUY | retest1 | 2025-01-23 09:55:00 | 40860.90 | 2025-01-23 15:20:00 | 41600.00 | TARGET_HIT | 0.50 | 1.81% |
| SELL | retest1 | 2025-02-04 10:35:00 | 38025.65 | 2025-02-04 10:45:00 | 37843.80 | PARTIAL | 0.50 | 0.48% |
| SELL | retest1 | 2025-02-04 10:35:00 | 38025.65 | 2025-02-04 11:50:00 | 38025.65 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-02-12 09:30:00 | 36432.60 | 2025-02-12 09:35:00 | 36200.31 | PARTIAL | 0.50 | 0.64% |
| SELL | retest1 | 2025-02-12 09:30:00 | 36432.60 | 2025-02-12 15:20:00 | 34915.45 | TARGET_HIT | 0.50 | 4.16% |
| BUY | retest1 | 2025-02-14 09:55:00 | 35690.40 | 2025-02-14 10:00:00 | 35555.87 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2025-02-19 09:30:00 | 33880.00 | 2025-02-19 09:40:00 | 33759.72 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2025-03-04 09:30:00 | 34048.00 | 2025-03-04 09:35:00 | 33909.70 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest1 | 2025-03-05 09:45:00 | 34090.25 | 2025-03-05 09:55:00 | 34218.03 | PARTIAL | 0.50 | 0.37% |
| BUY | retest1 | 2025-03-05 09:45:00 | 34090.25 | 2025-03-05 10:40:00 | 34090.25 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-03-07 09:30:00 | 35679.90 | 2025-03-07 09:40:00 | 35561.22 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2025-03-12 10:30:00 | 34062.85 | 2025-03-12 10:45:00 | 34154.17 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2025-03-17 10:15:00 | 33748.45 | 2025-03-17 10:25:00 | 33565.93 | PARTIAL | 0.50 | 0.54% |
| SELL | retest1 | 2025-03-17 10:15:00 | 33748.45 | 2025-03-17 11:00:00 | 33748.45 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-03-21 11:10:00 | 33885.70 | 2025-03-21 12:20:00 | 33986.39 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2025-03-25 11:05:00 | 33648.05 | 2025-03-25 11:10:00 | 33775.77 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2025-04-01 10:30:00 | 33883.10 | 2025-04-01 10:45:00 | 33756.97 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2025-04-09 10:25:00 | 33629.70 | 2025-04-09 11:45:00 | 33788.45 | PARTIAL | 0.50 | 0.47% |
| BUY | retest1 | 2025-04-09 10:25:00 | 33629.70 | 2025-04-09 15:20:00 | 33989.80 | TARGET_HIT | 0.50 | 1.07% |
| SELL | retest1 | 2025-04-11 09:45:00 | 34001.95 | 2025-04-11 09:50:00 | 34102.47 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2025-04-17 10:40:00 | 34800.00 | 2025-04-17 11:05:00 | 34939.46 | PARTIAL | 0.50 | 0.40% |
| BUY | retest1 | 2025-04-17 10:40:00 | 34800.00 | 2025-04-17 11:45:00 | 34800.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-04-22 09:35:00 | 34800.00 | 2025-04-22 10:05:00 | 34709.96 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2025-04-23 10:40:00 | 34330.00 | 2025-04-23 10:55:00 | 34176.26 | PARTIAL | 0.50 | 0.45% |
| SELL | retest1 | 2025-04-23 10:40:00 | 34330.00 | 2025-04-23 14:45:00 | 34320.00 | TARGET_HIT | 0.50 | 0.03% |
| BUY | retest1 | 2025-04-24 10:55:00 | 34580.00 | 2025-04-24 11:15:00 | 34494.99 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2025-04-25 09:55:00 | 33840.00 | 2025-04-25 10:30:00 | 33641.68 | PARTIAL | 0.50 | 0.59% |
| SELL | retest1 | 2025-04-25 09:55:00 | 33840.00 | 2025-04-25 12:40:00 | 33840.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-04-29 10:10:00 | 34485.00 | 2025-04-29 10:20:00 | 34711.20 | PARTIAL | 0.50 | 0.66% |
| BUY | retest1 | 2025-04-29 10:10:00 | 34485.00 | 2025-04-29 10:40:00 | 34485.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-04-30 11:15:00 | 34485.00 | 2025-04-30 15:20:00 | 34450.00 | STOP_HIT | 1.00 | -0.10% |
| BUY | retest1 | 2025-05-05 09:35:00 | 35165.00 | 2025-05-05 10:05:00 | 35051.22 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2025-05-08 09:30:00 | 35500.00 | 2025-05-08 10:05:00 | 35711.84 | PARTIAL | 0.50 | 0.60% |
| BUY | retest1 | 2025-05-08 09:30:00 | 35500.00 | 2025-05-08 10:20:00 | 35530.00 | TARGET_HIT | 0.50 | 0.08% |
