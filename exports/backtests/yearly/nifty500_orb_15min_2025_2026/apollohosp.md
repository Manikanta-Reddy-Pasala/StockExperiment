# Apollo Hospitals Enterprise Ltd. (APOLLOHOSP)

## Backtest Summary

- **Window:** 2025-05-12 09:15:00 → 2026-05-08 15:25:00 (18463 bars)
- **Last close:** 8100.00
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
| PARTIAL | 40 |
| TARGET_HIT | 14 |
| STOP_HIT | 77 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 131 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 54 / 77
- **Target hits / Stop hits / Partials:** 14 / 77 / 40
- **Avg / median % per leg:** 0.08% / 0.00%
- **Sum % (uncompounded):** 10.48%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 72 | 26 | 36.1% | 6 | 46 | 20 | 0.04% | 2.6% |
| BUY @ 2nd Alert (retest1) | 72 | 26 | 36.1% | 6 | 46 | 20 | 0.04% | 2.6% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 59 | 28 | 47.5% | 8 | 31 | 20 | 0.13% | 7.9% |
| SELL @ 2nd Alert (retest1) | 59 | 28 | 47.5% | 8 | 31 | 20 | 0.13% | 7.9% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 131 | 54 | 41.2% | 14 | 77 | 40 | 0.08% | 10.5% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-13 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-13 09:50:00 | 7007.00 | 6977.83 | 0.00 | ORB-long ORB[6907.50,6992.00] vol=1.6x ATR=15.78 |
| Stop hit — per-position SL triggered | 2025-05-13 10:05:00 | 6991.22 | 6983.05 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2025-05-14 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-14 10:40:00 | 6940.00 | 6937.70 | 0.00 | ORB-long ORB[6862.50,6938.00] vol=1.9x ATR=15.86 |
| Stop hit — per-position SL triggered | 2025-05-14 11:00:00 | 6924.14 | 6937.43 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2025-05-19 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-19 10:40:00 | 7050.00 | 7032.23 | 0.00 | ORB-long ORB[6980.00,7031.50] vol=2.6x ATR=11.47 |
| Stop hit — per-position SL triggered | 2025-05-19 10:45:00 | 7038.53 | 7032.15 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2025-05-21 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-21 09:30:00 | 7010.00 | 6986.37 | 0.00 | ORB-long ORB[6928.50,7008.50] vol=1.7x ATR=19.13 |
| Stop hit — per-position SL triggered | 2025-05-21 09:35:00 | 6990.87 | 6988.12 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2025-05-23 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-23 10:40:00 | 7032.00 | 7009.38 | 0.00 | ORB-long ORB[6930.00,6999.00] vol=2.3x ATR=14.30 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-23 10:55:00 | 7053.45 | 7027.39 | 0.00 | T1 1.5R @ 7053.45 |
| Stop hit — per-position SL triggered | 2025-05-23 11:15:00 | 7032.00 | 7029.65 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2025-05-27 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-27 09:35:00 | 7073.00 | 7084.32 | 0.00 | ORB-short ORB[7078.50,7124.50] vol=5.2x ATR=14.13 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-27 09:50:00 | 7051.81 | 7081.63 | 0.00 | T1 1.5R @ 7051.81 |
| Stop hit — per-position SL triggered | 2025-05-27 11:10:00 | 7073.00 | 7072.87 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2025-06-04 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-04 10:50:00 | 6837.00 | 6813.23 | 0.00 | ORB-long ORB[6772.00,6832.50] vol=2.9x ATR=11.86 |
| Stop hit — per-position SL triggered | 2025-06-04 10:55:00 | 6825.14 | 6815.02 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2025-06-05 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-05 09:30:00 | 6907.50 | 6881.36 | 0.00 | ORB-long ORB[6850.50,6895.00] vol=1.7x ATR=16.74 |
| Stop hit — per-position SL triggered | 2025-06-05 10:45:00 | 6890.76 | 6898.36 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2025-06-12 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-12 09:30:00 | 6999.50 | 6975.42 | 0.00 | ORB-long ORB[6930.00,6984.00] vol=2.2x ATR=11.99 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-12 09:35:00 | 7017.48 | 6986.47 | 0.00 | T1 1.5R @ 7017.48 |
| Stop hit — per-position SL triggered | 2025-06-12 09:40:00 | 6999.50 | 6988.89 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2025-06-16 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-16 10:30:00 | 7048.00 | 7014.26 | 0.00 | ORB-long ORB[7003.00,7034.50] vol=2.2x ATR=12.49 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-16 10:35:00 | 7066.73 | 7020.79 | 0.00 | T1 1.5R @ 7066.73 |
| Stop hit — per-position SL triggered | 2025-06-16 11:00:00 | 7048.00 | 7030.28 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2025-06-18 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-18 10:35:00 | 6993.00 | 7005.74 | 0.00 | ORB-short ORB[6995.50,7030.00] vol=4.2x ATR=12.43 |
| Stop hit — per-position SL triggered | 2025-06-18 10:45:00 | 7005.43 | 7004.50 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2025-06-19 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-19 10:35:00 | 7000.00 | 6985.22 | 0.00 | ORB-long ORB[6912.00,6983.00] vol=3.0x ATR=12.67 |
| Stop hit — per-position SL triggered | 2025-06-19 11:25:00 | 6987.33 | 6986.92 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2025-06-20 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-20 10:40:00 | 7010.00 | 7006.77 | 0.00 | ORB-long ORB[6974.50,7008.50] vol=2.7x ATR=12.58 |
| Stop hit — per-position SL triggered | 2025-06-20 11:35:00 | 6997.42 | 7007.26 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2025-06-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-23 11:15:00 | 7043.50 | 7016.68 | 0.00 | ORB-long ORB[7000.00,7035.50] vol=3.4x ATR=9.79 |
| Stop hit — per-position SL triggered | 2025-06-23 11:50:00 | 7033.71 | 7022.12 | 0.00 | SL hit |

### Cycle 15 — BUY (started 2025-06-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-24 11:15:00 | 7061.50 | 7040.69 | 0.00 | ORB-long ORB[7017.00,7057.50] vol=2.4x ATR=10.23 |
| Stop hit — per-position SL triggered | 2025-06-24 12:10:00 | 7051.27 | 7047.31 | 0.00 | SL hit |

### Cycle 16 — SELL (started 2025-07-09 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-09 10:25:00 | 7515.50 | 7536.13 | 0.00 | ORB-short ORB[7524.00,7568.50] vol=1.5x ATR=12.73 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-09 12:10:00 | 7496.41 | 7526.19 | 0.00 | T1 1.5R @ 7496.41 |
| Target hit | 2025-07-09 15:20:00 | 7465.50 | 7508.50 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 17 — SELL (started 2025-07-10 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-10 09:40:00 | 7400.00 | 7413.74 | 0.00 | ORB-short ORB[7404.00,7455.00] vol=1.5x ATR=11.96 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-10 09:50:00 | 7382.06 | 7409.31 | 0.00 | T1 1.5R @ 7382.06 |
| Stop hit — per-position SL triggered | 2025-07-10 10:20:00 | 7400.00 | 7397.13 | 0.00 | SL hit |

### Cycle 18 — SELL (started 2025-07-11 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-11 10:00:00 | 7261.00 | 7297.71 | 0.00 | ORB-short ORB[7277.50,7384.00] vol=1.6x ATR=17.67 |
| Stop hit — per-position SL triggered | 2025-07-11 10:05:00 | 7278.67 | 7295.93 | 0.00 | SL hit |

### Cycle 19 — BUY (started 2025-07-14 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-14 10:05:00 | 7236.00 | 7226.23 | 0.00 | ORB-long ORB[7173.00,7218.00] vol=1.5x ATR=15.97 |
| Stop hit — per-position SL triggered | 2025-07-14 10:15:00 | 7220.03 | 7228.29 | 0.00 | SL hit |

### Cycle 20 — BUY (started 2025-07-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-16 10:15:00 | 7389.00 | 7379.51 | 0.00 | ORB-long ORB[7322.50,7387.00] vol=3.7x ATR=14.07 |
| Stop hit — per-position SL triggered | 2025-07-16 10:40:00 | 7374.93 | 7381.29 | 0.00 | SL hit |

### Cycle 21 — SELL (started 2025-07-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-18 10:15:00 | 7261.50 | 7304.85 | 0.00 | ORB-short ORB[7323.00,7384.50] vol=2.1x ATR=13.82 |
| Stop hit — per-position SL triggered | 2025-07-18 10:40:00 | 7275.32 | 7296.83 | 0.00 | SL hit |

### Cycle 22 — SELL (started 2025-07-22 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-22 10:30:00 | 7239.00 | 7259.91 | 0.00 | ORB-short ORB[7264.00,7291.50] vol=3.6x ATR=9.59 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-22 10:40:00 | 7224.61 | 7258.13 | 0.00 | T1 1.5R @ 7224.61 |
| Stop hit — per-position SL triggered | 2025-07-22 11:00:00 | 7239.00 | 7253.93 | 0.00 | SL hit |

### Cycle 23 — BUY (started 2025-07-23 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-23 09:50:00 | 7308.00 | 7284.77 | 0.00 | ORB-long ORB[7244.00,7266.50] vol=1.7x ATR=10.12 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-23 09:55:00 | 7323.17 | 7291.91 | 0.00 | T1 1.5R @ 7323.17 |
| Stop hit — per-position SL triggered | 2025-07-23 10:05:00 | 7308.00 | 7296.25 | 0.00 | SL hit |

### Cycle 24 — BUY (started 2025-07-24 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-24 10:00:00 | 7454.50 | 7424.65 | 0.00 | ORB-long ORB[7360.00,7434.00] vol=1.6x ATR=15.02 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-24 10:30:00 | 7477.03 | 7437.65 | 0.00 | T1 1.5R @ 7477.03 |
| Stop hit — per-position SL triggered | 2025-07-24 10:50:00 | 7454.50 | 7441.87 | 0.00 | SL hit |

### Cycle 25 — SELL (started 2025-07-28 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-28 10:45:00 | 7444.00 | 7459.27 | 0.00 | ORB-short ORB[7447.50,7502.00] vol=2.6x ATR=15.53 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-28 11:25:00 | 7420.71 | 7454.46 | 0.00 | T1 1.5R @ 7420.71 |
| Target hit | 2025-07-28 15:20:00 | 7356.00 | 7390.19 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 26 — SELL (started 2025-08-06 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-06 10:00:00 | 7240.50 | 7257.20 | 0.00 | ORB-short ORB[7245.00,7279.00] vol=2.7x ATR=12.56 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-06 10:35:00 | 7221.65 | 7251.16 | 0.00 | T1 1.5R @ 7221.65 |
| Target hit | 2025-08-06 15:20:00 | 7176.50 | 7214.03 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 27 — SELL (started 2025-08-12 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-12 11:10:00 | 7221.50 | 7247.84 | 0.00 | ORB-short ORB[7229.50,7268.00] vol=2.7x ATR=12.44 |
| Stop hit — per-position SL triggered | 2025-08-12 11:20:00 | 7233.94 | 7246.70 | 0.00 | SL hit |

### Cycle 28 — BUY (started 2025-08-18 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-18 09:30:00 | 7878.50 | 7866.70 | 0.00 | ORB-long ORB[7846.00,7878.00] vol=1.5x ATR=13.86 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-18 09:50:00 | 7899.28 | 7875.54 | 0.00 | T1 1.5R @ 7899.28 |
| Target hit | 2025-08-18 10:45:00 | 7883.50 | 7883.99 | 0.00 | Trail-exit close<VWAP |

### Cycle 29 — BUY (started 2025-08-20 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-20 10:30:00 | 7901.50 | 7859.25 | 0.00 | ORB-long ORB[7812.50,7860.00] vol=2.1x ATR=12.06 |
| Stop hit — per-position SL triggered | 2025-08-20 12:30:00 | 7889.44 | 7878.23 | 0.00 | SL hit |

### Cycle 30 — BUY (started 2025-08-22 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-22 09:40:00 | 7910.50 | 7886.77 | 0.00 | ORB-long ORB[7852.00,7910.00] vol=1.7x ATR=15.38 |
| Stop hit — per-position SL triggered | 2025-08-22 09:50:00 | 7895.12 | 7892.37 | 0.00 | SL hit |

### Cycle 31 — SELL (started 2025-08-25 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-25 10:50:00 | 7850.00 | 7882.16 | 0.00 | ORB-short ORB[7881.50,7950.00] vol=1.6x ATR=12.60 |
| Stop hit — per-position SL triggered | 2025-08-25 11:55:00 | 7862.60 | 7871.63 | 0.00 | SL hit |

### Cycle 32 — SELL (started 2025-08-26 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-26 10:40:00 | 7762.50 | 7796.86 | 0.00 | ORB-short ORB[7788.50,7834.00] vol=1.5x ATR=14.14 |
| Stop hit — per-position SL triggered | 2025-08-26 11:30:00 | 7776.64 | 7786.65 | 0.00 | SL hit |

### Cycle 33 — BUY (started 2025-08-28 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-28 11:00:00 | 7816.00 | 7792.05 | 0.00 | ORB-long ORB[7740.00,7790.00] vol=1.8x ATR=13.37 |
| Stop hit — per-position SL triggered | 2025-08-28 11:35:00 | 7802.63 | 7796.52 | 0.00 | SL hit |

### Cycle 34 — BUY (started 2025-09-03 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-03 09:30:00 | 7724.50 | 7693.77 | 0.00 | ORB-long ORB[7656.00,7708.50] vol=1.8x ATR=14.42 |
| Stop hit — per-position SL triggered | 2025-09-03 09:35:00 | 7710.08 | 7697.38 | 0.00 | SL hit |

### Cycle 35 — BUY (started 2025-09-04 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-04 10:50:00 | 7873.00 | 7824.53 | 0.00 | ORB-long ORB[7751.00,7790.00] vol=1.8x ATR=16.07 |
| Stop hit — per-position SL triggered | 2025-09-04 11:20:00 | 7856.93 | 7828.95 | 0.00 | SL hit |

### Cycle 36 — SELL (started 2025-09-08 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-08 11:05:00 | 7762.00 | 7790.72 | 0.00 | ORB-short ORB[7764.50,7849.50] vol=1.6x ATR=13.48 |
| Stop hit — per-position SL triggered | 2025-09-08 11:25:00 | 7775.48 | 7788.67 | 0.00 | SL hit |

### Cycle 37 — SELL (started 2025-09-12 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-12 11:00:00 | 7869.00 | 7879.96 | 0.00 | ORB-short ORB[7873.50,7915.00] vol=1.7x ATR=8.33 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-12 11:05:00 | 7856.50 | 7879.28 | 0.00 | T1 1.5R @ 7856.50 |
| Stop hit — per-position SL triggered | 2025-09-12 11:35:00 | 7869.00 | 7875.29 | 0.00 | SL hit |

### Cycle 38 — BUY (started 2025-09-17 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-17 10:35:00 | 7899.00 | 7891.22 | 0.00 | ORB-long ORB[7860.00,7888.00] vol=2.1x ATR=10.32 |
| Stop hit — per-position SL triggered | 2025-09-17 10:40:00 | 7888.68 | 7890.87 | 0.00 | SL hit |

### Cycle 39 — SELL (started 2025-09-23 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-23 10:25:00 | 7673.50 | 7676.86 | 0.00 | ORB-short ORB[7696.00,7749.00] vol=12.9x ATR=9.68 |
| Stop hit — per-position SL triggered | 2025-09-23 10:35:00 | 7683.18 | 7676.81 | 0.00 | SL hit |

### Cycle 40 — BUY (started 2025-09-24 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-24 10:00:00 | 7709.50 | 7686.26 | 0.00 | ORB-long ORB[7657.00,7701.00] vol=2.1x ATR=12.20 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-24 11:50:00 | 7727.80 | 7698.46 | 0.00 | T1 1.5R @ 7727.80 |
| Stop hit — per-position SL triggered | 2025-09-24 12:20:00 | 7709.50 | 7700.05 | 0.00 | SL hit |

### Cycle 41 — BUY (started 2025-10-01 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-01 09:35:00 | 7462.50 | 7430.02 | 0.00 | ORB-long ORB[7374.50,7448.00] vol=2.0x ATR=19.29 |
| Stop hit — per-position SL triggered | 2025-10-01 09:45:00 | 7443.21 | 7434.95 | 0.00 | SL hit |

### Cycle 42 — BUY (started 2025-10-09 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-09 09:45:00 | 7703.00 | 7677.78 | 0.00 | ORB-long ORB[7620.00,7691.00] vol=4.5x ATR=18.21 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-09 11:10:00 | 7730.32 | 7689.30 | 0.00 | T1 1.5R @ 7730.32 |
| Stop hit — per-position SL triggered | 2025-10-09 14:10:00 | 7703.00 | 7703.58 | 0.00 | SL hit |

### Cycle 43 — SELL (started 2025-10-10 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-10 11:00:00 | 7679.50 | 7701.57 | 0.00 | ORB-short ORB[7680.50,7719.50] vol=2.0x ATR=11.78 |
| Stop hit — per-position SL triggered | 2025-10-10 11:20:00 | 7691.28 | 7697.49 | 0.00 | SL hit |

### Cycle 44 — BUY (started 2025-10-17 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-17 09:40:00 | 7889.00 | 7877.34 | 0.00 | ORB-long ORB[7840.50,7880.00] vol=6.9x ATR=12.15 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-17 09:45:00 | 7907.23 | 7879.65 | 0.00 | T1 1.5R @ 7907.23 |
| Stop hit — per-position SL triggered | 2025-10-17 09:50:00 | 7889.00 | 7879.90 | 0.00 | SL hit |

### Cycle 45 — BUY (started 2025-10-20 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-20 09:30:00 | 8038.00 | 8010.74 | 0.00 | ORB-long ORB[7921.50,8028.00] vol=2.2x ATR=26.48 |
| Stop hit — per-position SL triggered | 2025-10-20 09:35:00 | 8011.52 | 8011.65 | 0.00 | SL hit |

### Cycle 46 — SELL (started 2025-10-24 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-24 11:05:00 | 7880.50 | 7913.83 | 0.00 | ORB-short ORB[7920.50,7982.00] vol=2.5x ATR=12.22 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-24 11:15:00 | 7862.17 | 7910.80 | 0.00 | T1 1.5R @ 7862.17 |
| Stop hit — per-position SL triggered | 2025-10-24 11:50:00 | 7880.50 | 7901.34 | 0.00 | SL hit |

### Cycle 47 — BUY (started 2025-10-27 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-27 09:40:00 | 7894.50 | 7869.28 | 0.00 | ORB-long ORB[7837.50,7880.00] vol=1.5x ATR=14.67 |
| Stop hit — per-position SL triggered | 2025-10-27 10:30:00 | 7879.83 | 7880.01 | 0.00 | SL hit |

### Cycle 48 — BUY (started 2025-11-03 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-03 10:50:00 | 7768.00 | 7727.13 | 0.00 | ORB-long ORB[7661.50,7716.50] vol=2.0x ATR=13.46 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-03 11:00:00 | 7788.19 | 7741.03 | 0.00 | T1 1.5R @ 7788.19 |
| Stop hit — per-position SL triggered | 2025-11-03 11:10:00 | 7768.00 | 7742.12 | 0.00 | SL hit |

### Cycle 49 — BUY (started 2025-11-04 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-04 09:30:00 | 7863.00 | 7832.92 | 0.00 | ORB-long ORB[7795.00,7849.00] vol=1.8x ATR=14.27 |
| Stop hit — per-position SL triggered | 2025-11-04 09:35:00 | 7848.73 | 7836.79 | 0.00 | SL hit |

### Cycle 50 — BUY (started 2025-11-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-06 11:15:00 | 7817.00 | 7755.33 | 0.00 | ORB-long ORB[7700.00,7796.50] vol=2.1x ATR=16.11 |
| Stop hit — per-position SL triggered | 2025-11-06 11:20:00 | 7800.89 | 7756.45 | 0.00 | SL hit |

### Cycle 51 — SELL (started 2025-11-11 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-11 10:50:00 | 7469.00 | 7499.50 | 0.00 | ORB-short ORB[7484.00,7569.00] vol=2.1x ATR=14.30 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-11 11:10:00 | 7447.54 | 7493.09 | 0.00 | T1 1.5R @ 7447.54 |
| Stop hit — per-position SL triggered | 2025-11-11 13:45:00 | 7469.00 | 7476.44 | 0.00 | SL hit |

### Cycle 52 — BUY (started 2025-11-13 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-13 11:00:00 | 7507.50 | 7464.53 | 0.00 | ORB-long ORB[7425.00,7501.00] vol=1.6x ATR=11.55 |
| Stop hit — per-position SL triggered | 2025-11-13 11:10:00 | 7495.95 | 7467.10 | 0.00 | SL hit |

### Cycle 53 — SELL (started 2025-11-18 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-18 09:50:00 | 7417.00 | 7445.56 | 0.00 | ORB-short ORB[7442.50,7474.00] vol=1.5x ATR=10.83 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-18 10:30:00 | 7400.76 | 7433.14 | 0.00 | T1 1.5R @ 7400.76 |
| Stop hit — per-position SL triggered | 2025-11-18 11:15:00 | 7417.00 | 7424.68 | 0.00 | SL hit |

### Cycle 54 — SELL (started 2025-11-27 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-27 09:40:00 | 7353.50 | 7370.98 | 0.00 | ORB-short ORB[7362.50,7396.00] vol=1.9x ATR=11.14 |
| Stop hit — per-position SL triggered | 2025-11-27 09:45:00 | 7364.64 | 7370.32 | 0.00 | SL hit |

### Cycle 55 — SELL (started 2025-12-02 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-02 09:30:00 | 7266.00 | 7278.63 | 0.00 | ORB-short ORB[7270.00,7299.00] vol=1.7x ATR=8.78 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-02 09:45:00 | 7252.84 | 7270.17 | 0.00 | T1 1.5R @ 7252.84 |
| Stop hit — per-position SL triggered | 2025-12-02 10:25:00 | 7266.00 | 7260.91 | 0.00 | SL hit |

### Cycle 56 — SELL (started 2025-12-03 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-03 10:00:00 | 7182.50 | 7211.72 | 0.00 | ORB-short ORB[7200.00,7257.00] vol=1.9x ATR=12.15 |
| Stop hit — per-position SL triggered | 2025-12-03 10:10:00 | 7194.65 | 7209.25 | 0.00 | SL hit |

### Cycle 57 — BUY (started 2025-12-04 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-04 10:05:00 | 7202.00 | 7191.91 | 0.00 | ORB-long ORB[7158.00,7198.00] vol=5.1x ATR=14.58 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-04 10:25:00 | 7223.87 | 7193.93 | 0.00 | T1 1.5R @ 7223.87 |
| Stop hit — per-position SL triggered | 2025-12-04 11:00:00 | 7202.00 | 7198.23 | 0.00 | SL hit |

### Cycle 58 — SELL (started 2025-12-08 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-08 10:50:00 | 7152.00 | 7180.97 | 0.00 | ORB-short ORB[7157.00,7219.00] vol=2.0x ATR=9.96 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-08 11:10:00 | 7137.06 | 7178.33 | 0.00 | T1 1.5R @ 7137.06 |
| Target hit | 2025-12-08 15:20:00 | 7097.00 | 7123.87 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 59 — SELL (started 2025-12-10 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-10 10:50:00 | 7065.50 | 7090.86 | 0.00 | ORB-short ORB[7082.00,7116.00] vol=3.5x ATR=11.21 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-10 11:30:00 | 7048.68 | 7080.19 | 0.00 | T1 1.5R @ 7048.68 |
| Target hit | 2025-12-10 13:15:00 | 7054.00 | 7053.78 | 0.00 | Trail-exit close>VWAP |

### Cycle 60 — BUY (started 2025-12-12 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-12 10:45:00 | 7067.00 | 7044.02 | 0.00 | ORB-long ORB[6996.00,7039.50] vol=1.6x ATR=11.38 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-12 11:15:00 | 7084.07 | 7052.73 | 0.00 | T1 1.5R @ 7084.07 |
| Stop hit — per-position SL triggered | 2025-12-12 11:25:00 | 7067.00 | 7056.20 | 0.00 | SL hit |

### Cycle 61 — SELL (started 2025-12-17 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-17 10:55:00 | 6987.00 | 7036.25 | 0.00 | ORB-short ORB[7052.00,7109.00] vol=1.8x ATR=11.85 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-17 12:10:00 | 6969.22 | 7016.77 | 0.00 | T1 1.5R @ 6969.22 |
| Target hit | 2025-12-17 15:20:00 | 6916.50 | 6957.33 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 62 — BUY (started 2025-12-18 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-18 11:05:00 | 6937.50 | 6916.39 | 0.00 | ORB-long ORB[6892.50,6934.00] vol=3.8x ATR=11.93 |
| Stop hit — per-position SL triggered | 2025-12-18 13:10:00 | 6925.57 | 6925.87 | 0.00 | SL hit |

### Cycle 63 — BUY (started 2025-12-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-23 11:15:00 | 7074.00 | 7050.17 | 0.00 | ORB-long ORB[7020.00,7069.50] vol=1.6x ATR=8.54 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-23 11:30:00 | 7086.82 | 7054.48 | 0.00 | T1 1.5R @ 7086.82 |
| Stop hit — per-position SL triggered | 2025-12-23 11:45:00 | 7074.00 | 7055.86 | 0.00 | SL hit |

### Cycle 64 — BUY (started 2025-12-24 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-24 09:50:00 | 7120.50 | 7098.91 | 0.00 | ORB-long ORB[7044.50,7090.50] vol=2.2x ATR=11.17 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-24 10:15:00 | 7137.25 | 7108.18 | 0.00 | T1 1.5R @ 7137.25 |
| Target hit | 2025-12-24 14:35:00 | 7168.50 | 7169.65 | 0.00 | Trail-exit close<VWAP |

### Cycle 65 — SELL (started 2025-12-29 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-29 10:00:00 | 7131.00 | 7149.57 | 0.00 | ORB-short ORB[7132.50,7180.00] vol=1.6x ATR=12.74 |
| Stop hit — per-position SL triggered | 2025-12-29 10:20:00 | 7143.74 | 7145.56 | 0.00 | SL hit |

### Cycle 66 — BUY (started 2026-01-01 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-01 10:50:00 | 7090.00 | 7064.39 | 0.00 | ORB-long ORB[7050.50,7078.00] vol=2.7x ATR=11.45 |
| Stop hit — per-position SL triggered | 2026-01-01 11:00:00 | 7078.55 | 7065.71 | 0.00 | SL hit |

### Cycle 67 — SELL (started 2026-01-08 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-08 10:10:00 | 7380.00 | 7395.73 | 0.00 | ORB-short ORB[7385.00,7443.00] vol=7.6x ATR=16.22 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-08 10:50:00 | 7355.66 | 7392.77 | 0.00 | T1 1.5R @ 7355.66 |
| Stop hit — per-position SL triggered | 2026-01-08 11:55:00 | 7380.00 | 7385.83 | 0.00 | SL hit |

### Cycle 68 — SELL (started 2026-01-19 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-19 11:10:00 | 7188.00 | 7198.72 | 0.00 | ORB-short ORB[7205.00,7260.00] vol=2.5x ATR=12.50 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-19 11:20:00 | 7169.26 | 7195.63 | 0.00 | T1 1.5R @ 7169.26 |
| Stop hit — per-position SL triggered | 2026-01-19 11:40:00 | 7188.00 | 7194.26 | 0.00 | SL hit |

### Cycle 69 — SELL (started 2026-01-20 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-20 11:00:00 | 7000.00 | 7032.48 | 0.00 | ORB-short ORB[7040.00,7129.00] vol=2.4x ATR=13.31 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-20 12:15:00 | 6980.04 | 7020.28 | 0.00 | T1 1.5R @ 6980.04 |
| Target hit | 2026-01-20 15:20:00 | 6892.00 | 6988.54 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 70 — SELL (started 2026-01-21 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-21 10:55:00 | 6768.00 | 6835.24 | 0.00 | ORB-short ORB[6841.00,6911.50] vol=2.1x ATR=21.18 |
| Stop hit — per-position SL triggered | 2026-01-21 11:10:00 | 6789.18 | 6825.16 | 0.00 | SL hit |

### Cycle 71 — BUY (started 2026-01-28 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-28 10:45:00 | 6875.00 | 6847.16 | 0.00 | ORB-long ORB[6784.50,6867.50] vol=2.7x ATR=14.84 |
| Stop hit — per-position SL triggered | 2026-01-28 11:55:00 | 6860.16 | 6856.11 | 0.00 | SL hit |

### Cycle 72 — BUY (started 2026-02-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-03 11:15:00 | 7118.00 | 7067.50 | 0.00 | ORB-long ORB[7002.50,7100.00] vol=1.6x ATR=22.47 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-03 12:10:00 | 7151.71 | 7084.78 | 0.00 | T1 1.5R @ 7151.71 |
| Stop hit — per-position SL triggered | 2026-02-03 12:55:00 | 7118.00 | 7092.31 | 0.00 | SL hit |

### Cycle 73 — SELL (started 2026-02-06 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-06 10:05:00 | 7071.00 | 7099.05 | 0.00 | ORB-short ORB[7080.00,7137.50] vol=2.0x ATR=16.26 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-06 10:10:00 | 7046.60 | 7085.07 | 0.00 | T1 1.5R @ 7046.60 |
| Target hit | 2026-02-06 11:30:00 | 7056.00 | 7052.02 | 0.00 | Trail-exit close>VWAP |

### Cycle 74 — SELL (started 2026-02-09 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-09 10:55:00 | 7173.50 | 7198.75 | 0.00 | ORB-short ORB[7184.00,7250.00] vol=3.0x ATR=17.17 |
| Stop hit — per-position SL triggered | 2026-02-09 12:05:00 | 7190.67 | 7194.66 | 0.00 | SL hit |

### Cycle 75 — SELL (started 2026-02-10 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-10 09:40:00 | 7208.50 | 7245.05 | 0.00 | ORB-short ORB[7218.00,7272.00] vol=1.6x ATR=23.20 |
| Stop hit — per-position SL triggered | 2026-02-10 11:00:00 | 7231.70 | 7225.57 | 0.00 | SL hit |

### Cycle 76 — BUY (started 2026-02-13 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-13 09:45:00 | 7557.50 | 7530.69 | 0.00 | ORB-long ORB[7485.00,7537.00] vol=2.1x ATR=18.31 |
| Stop hit — per-position SL triggered | 2026-02-13 10:15:00 | 7539.19 | 7545.01 | 0.00 | SL hit |

### Cycle 77 — BUY (started 2026-02-16 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-16 10:10:00 | 7544.50 | 7520.86 | 0.00 | ORB-long ORB[7478.00,7532.50] vol=1.8x ATR=18.70 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-16 12:15:00 | 7572.55 | 7541.26 | 0.00 | T1 1.5R @ 7572.55 |
| Target hit | 2026-02-16 15:20:00 | 7606.00 | 7575.52 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 78 — BUY (started 2026-02-18 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-18 09:35:00 | 7651.50 | 7637.37 | 0.00 | ORB-long ORB[7590.00,7638.50] vol=8.0x ATR=16.09 |
| Stop hit — per-position SL triggered | 2026-02-18 09:55:00 | 7635.41 | 7639.80 | 0.00 | SL hit |

### Cycle 79 — SELL (started 2026-03-05 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-05 11:00:00 | 7670.50 | 7718.67 | 0.00 | ORB-short ORB[7680.00,7746.00] vol=1.7x ATR=19.01 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-05 11:55:00 | 7641.98 | 7706.74 | 0.00 | T1 1.5R @ 7641.98 |
| Stop hit — per-position SL triggered | 2026-03-05 12:25:00 | 7670.50 | 7695.11 | 0.00 | SL hit |

### Cycle 80 — SELL (started 2026-03-06 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-06 10:45:00 | 7723.00 | 7747.97 | 0.00 | ORB-short ORB[7732.50,7781.00] vol=1.9x ATR=15.85 |
| Stop hit — per-position SL triggered | 2026-03-06 11:00:00 | 7738.85 | 7746.29 | 0.00 | SL hit |

### Cycle 81 — SELL (started 2026-03-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-10 11:15:00 | 7785.00 | 7816.81 | 0.00 | ORB-short ORB[7801.50,7870.00] vol=4.5x ATR=15.85 |
| Stop hit — per-position SL triggered | 2026-03-10 11:20:00 | 7800.85 | 7816.42 | 0.00 | SL hit |

### Cycle 82 — SELL (started 2026-03-11 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-11 11:10:00 | 7712.50 | 7740.97 | 0.00 | ORB-short ORB[7732.00,7798.50] vol=3.4x ATR=14.94 |
| Stop hit — per-position SL triggered | 2026-03-11 12:10:00 | 7727.44 | 7735.14 | 0.00 | SL hit |

### Cycle 83 — BUY (started 2026-03-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-25 11:15:00 | 7558.50 | 7535.50 | 0.00 | ORB-long ORB[7437.50,7525.00] vol=2.6x ATR=16.43 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-25 12:00:00 | 7583.14 | 7544.20 | 0.00 | T1 1.5R @ 7583.14 |
| Target hit | 2026-03-25 15:20:00 | 7586.00 | 7561.26 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 84 — BUY (started 2026-04-10 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-10 11:05:00 | 7554.50 | 7517.68 | 0.00 | ORB-long ORB[7461.50,7515.00] vol=2.9x ATR=14.74 |
| Stop hit — per-position SL triggered | 2026-04-10 11:30:00 | 7539.76 | 7528.41 | 0.00 | SL hit |

### Cycle 85 — BUY (started 2026-04-13 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-13 10:45:00 | 7497.00 | 7467.19 | 0.00 | ORB-long ORB[7403.50,7485.00] vol=2.0x ATR=17.96 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-13 10:50:00 | 7523.93 | 7472.80 | 0.00 | T1 1.5R @ 7523.93 |
| Target hit | 2026-04-13 15:20:00 | 7510.50 | 7506.26 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 86 — BUY (started 2026-04-17 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-17 10:05:00 | 7627.00 | 7580.70 | 0.00 | ORB-long ORB[7515.00,7591.00] vol=2.8x ATR=20.46 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-17 11:35:00 | 7657.68 | 7595.71 | 0.00 | T1 1.5R @ 7657.68 |
| Target hit | 2026-04-17 15:20:00 | 7703.50 | 7655.79 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 87 — BUY (started 2026-04-21 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-21 09:50:00 | 7715.00 | 7692.23 | 0.00 | ORB-long ORB[7667.50,7710.00] vol=1.6x ATR=14.06 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-21 10:05:00 | 7736.09 | 7705.33 | 0.00 | T1 1.5R @ 7736.09 |
| Stop hit — per-position SL triggered | 2026-04-21 11:35:00 | 7715.00 | 7717.18 | 0.00 | SL hit |

### Cycle 88 — BUY (started 2026-04-22 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-22 10:30:00 | 7757.00 | 7734.54 | 0.00 | ORB-long ORB[7697.00,7752.00] vol=2.8x ATR=15.23 |
| Stop hit — per-position SL triggered | 2026-04-22 10:35:00 | 7741.77 | 7735.28 | 0.00 | SL hit |

### Cycle 89 — SELL (started 2026-04-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-24 11:15:00 | 7718.00 | 7765.45 | 0.00 | ORB-short ORB[7751.00,7834.50] vol=7.8x ATR=18.30 |
| Stop hit — per-position SL triggered | 2026-04-24 11:20:00 | 7736.30 | 7765.03 | 0.00 | SL hit |

### Cycle 90 — BUY (started 2026-04-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-27 11:15:00 | 7792.50 | 7772.87 | 0.00 | ORB-long ORB[7735.00,7786.50] vol=2.1x ATR=16.77 |
| Stop hit — per-position SL triggered | 2026-04-27 11:30:00 | 7775.73 | 7773.82 | 0.00 | SL hit |

### Cycle 91 — SELL (started 2026-04-28 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-28 10:55:00 | 7771.50 | 7807.69 | 0.00 | ORB-short ORB[7794.00,7878.50] vol=3.3x ATR=14.50 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-28 11:55:00 | 7749.75 | 7797.52 | 0.00 | T1 1.5R @ 7749.75 |
| Stop hit — per-position SL triggered | 2026-04-28 14:30:00 | 7771.50 | 7771.30 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-05-13 09:50:00 | 7007.00 | 2025-05-13 10:05:00 | 6991.22 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2025-05-14 10:40:00 | 6940.00 | 2025-05-14 11:00:00 | 6924.14 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2025-05-19 10:40:00 | 7050.00 | 2025-05-19 10:45:00 | 7038.53 | STOP_HIT | 1.00 | -0.16% |
| BUY | retest1 | 2025-05-21 09:30:00 | 7010.00 | 2025-05-21 09:35:00 | 6990.87 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2025-05-23 10:40:00 | 7032.00 | 2025-05-23 10:55:00 | 7053.45 | PARTIAL | 0.50 | 0.30% |
| BUY | retest1 | 2025-05-23 10:40:00 | 7032.00 | 2025-05-23 11:15:00 | 7032.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-05-27 09:35:00 | 7073.00 | 2025-05-27 09:50:00 | 7051.81 | PARTIAL | 0.50 | 0.30% |
| SELL | retest1 | 2025-05-27 09:35:00 | 7073.00 | 2025-05-27 11:10:00 | 7073.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-06-04 10:50:00 | 6837.00 | 2025-06-04 10:55:00 | 6825.14 | STOP_HIT | 1.00 | -0.17% |
| BUY | retest1 | 2025-06-05 09:30:00 | 6907.50 | 2025-06-05 10:45:00 | 6890.76 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2025-06-12 09:30:00 | 6999.50 | 2025-06-12 09:35:00 | 7017.48 | PARTIAL | 0.50 | 0.26% |
| BUY | retest1 | 2025-06-12 09:30:00 | 6999.50 | 2025-06-12 09:40:00 | 6999.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-06-16 10:30:00 | 7048.00 | 2025-06-16 10:35:00 | 7066.73 | PARTIAL | 0.50 | 0.27% |
| BUY | retest1 | 2025-06-16 10:30:00 | 7048.00 | 2025-06-16 11:00:00 | 7048.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-06-18 10:35:00 | 6993.00 | 2025-06-18 10:45:00 | 7005.43 | STOP_HIT | 1.00 | -0.18% |
| BUY | retest1 | 2025-06-19 10:35:00 | 7000.00 | 2025-06-19 11:25:00 | 6987.33 | STOP_HIT | 1.00 | -0.18% |
| BUY | retest1 | 2025-06-20 10:40:00 | 7010.00 | 2025-06-20 11:35:00 | 6997.42 | STOP_HIT | 1.00 | -0.18% |
| BUY | retest1 | 2025-06-23 11:15:00 | 7043.50 | 2025-06-23 11:50:00 | 7033.71 | STOP_HIT | 1.00 | -0.14% |
| BUY | retest1 | 2025-06-24 11:15:00 | 7061.50 | 2025-06-24 12:10:00 | 7051.27 | STOP_HIT | 1.00 | -0.14% |
| SELL | retest1 | 2025-07-09 10:25:00 | 7515.50 | 2025-07-09 12:10:00 | 7496.41 | PARTIAL | 0.50 | 0.25% |
| SELL | retest1 | 2025-07-09 10:25:00 | 7515.50 | 2025-07-09 15:20:00 | 7465.50 | TARGET_HIT | 0.50 | 0.67% |
| SELL | retest1 | 2025-07-10 09:40:00 | 7400.00 | 2025-07-10 09:50:00 | 7382.06 | PARTIAL | 0.50 | 0.24% |
| SELL | retest1 | 2025-07-10 09:40:00 | 7400.00 | 2025-07-10 10:20:00 | 7400.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-07-11 10:00:00 | 7261.00 | 2025-07-11 10:05:00 | 7278.67 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2025-07-14 10:05:00 | 7236.00 | 2025-07-14 10:15:00 | 7220.03 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2025-07-16 10:15:00 | 7389.00 | 2025-07-16 10:40:00 | 7374.93 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest1 | 2025-07-18 10:15:00 | 7261.50 | 2025-07-18 10:40:00 | 7275.32 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest1 | 2025-07-22 10:30:00 | 7239.00 | 2025-07-22 10:40:00 | 7224.61 | PARTIAL | 0.50 | 0.20% |
| SELL | retest1 | 2025-07-22 10:30:00 | 7239.00 | 2025-07-22 11:00:00 | 7239.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-07-23 09:50:00 | 7308.00 | 2025-07-23 09:55:00 | 7323.17 | PARTIAL | 0.50 | 0.21% |
| BUY | retest1 | 2025-07-23 09:50:00 | 7308.00 | 2025-07-23 10:05:00 | 7308.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-07-24 10:00:00 | 7454.50 | 2025-07-24 10:30:00 | 7477.03 | PARTIAL | 0.50 | 0.30% |
| BUY | retest1 | 2025-07-24 10:00:00 | 7454.50 | 2025-07-24 10:50:00 | 7454.50 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-07-28 10:45:00 | 7444.00 | 2025-07-28 11:25:00 | 7420.71 | PARTIAL | 0.50 | 0.31% |
| SELL | retest1 | 2025-07-28 10:45:00 | 7444.00 | 2025-07-28 15:20:00 | 7356.00 | TARGET_HIT | 0.50 | 1.18% |
| SELL | retest1 | 2025-08-06 10:00:00 | 7240.50 | 2025-08-06 10:35:00 | 7221.65 | PARTIAL | 0.50 | 0.26% |
| SELL | retest1 | 2025-08-06 10:00:00 | 7240.50 | 2025-08-06 15:20:00 | 7176.50 | TARGET_HIT | 0.50 | 0.88% |
| SELL | retest1 | 2025-08-12 11:10:00 | 7221.50 | 2025-08-12 11:20:00 | 7233.94 | STOP_HIT | 1.00 | -0.17% |
| BUY | retest1 | 2025-08-18 09:30:00 | 7878.50 | 2025-08-18 09:50:00 | 7899.28 | PARTIAL | 0.50 | 0.26% |
| BUY | retest1 | 2025-08-18 09:30:00 | 7878.50 | 2025-08-18 10:45:00 | 7883.50 | TARGET_HIT | 0.50 | 0.06% |
| BUY | retest1 | 2025-08-20 10:30:00 | 7901.50 | 2025-08-20 12:30:00 | 7889.44 | STOP_HIT | 1.00 | -0.15% |
| BUY | retest1 | 2025-08-22 09:40:00 | 7910.50 | 2025-08-22 09:50:00 | 7895.12 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest1 | 2025-08-25 10:50:00 | 7850.00 | 2025-08-25 11:55:00 | 7862.60 | STOP_HIT | 1.00 | -0.16% |
| SELL | retest1 | 2025-08-26 10:40:00 | 7762.50 | 2025-08-26 11:30:00 | 7776.64 | STOP_HIT | 1.00 | -0.18% |
| BUY | retest1 | 2025-08-28 11:00:00 | 7816.00 | 2025-08-28 11:35:00 | 7802.63 | STOP_HIT | 1.00 | -0.17% |
| BUY | retest1 | 2025-09-03 09:30:00 | 7724.50 | 2025-09-03 09:35:00 | 7710.08 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest1 | 2025-09-04 10:50:00 | 7873.00 | 2025-09-04 11:20:00 | 7856.93 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2025-09-08 11:05:00 | 7762.00 | 2025-09-08 11:25:00 | 7775.48 | STOP_HIT | 1.00 | -0.17% |
| SELL | retest1 | 2025-09-12 11:00:00 | 7869.00 | 2025-09-12 11:05:00 | 7856.50 | PARTIAL | 0.50 | 0.16% |
| SELL | retest1 | 2025-09-12 11:00:00 | 7869.00 | 2025-09-12 11:35:00 | 7869.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-09-17 10:35:00 | 7899.00 | 2025-09-17 10:40:00 | 7888.68 | STOP_HIT | 1.00 | -0.13% |
| SELL | retest1 | 2025-09-23 10:25:00 | 7673.50 | 2025-09-23 10:35:00 | 7683.18 | STOP_HIT | 1.00 | -0.13% |
| BUY | retest1 | 2025-09-24 10:00:00 | 7709.50 | 2025-09-24 11:50:00 | 7727.80 | PARTIAL | 0.50 | 0.24% |
| BUY | retest1 | 2025-09-24 10:00:00 | 7709.50 | 2025-09-24 12:20:00 | 7709.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-10-01 09:35:00 | 7462.50 | 2025-10-01 09:45:00 | 7443.21 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2025-10-09 09:45:00 | 7703.00 | 2025-10-09 11:10:00 | 7730.32 | PARTIAL | 0.50 | 0.35% |
| BUY | retest1 | 2025-10-09 09:45:00 | 7703.00 | 2025-10-09 14:10:00 | 7703.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-10-10 11:00:00 | 7679.50 | 2025-10-10 11:20:00 | 7691.28 | STOP_HIT | 1.00 | -0.15% |
| BUY | retest1 | 2025-10-17 09:40:00 | 7889.00 | 2025-10-17 09:45:00 | 7907.23 | PARTIAL | 0.50 | 0.23% |
| BUY | retest1 | 2025-10-17 09:40:00 | 7889.00 | 2025-10-17 09:50:00 | 7889.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-10-20 09:30:00 | 8038.00 | 2025-10-20 09:35:00 | 8011.52 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2025-10-24 11:05:00 | 7880.50 | 2025-10-24 11:15:00 | 7862.17 | PARTIAL | 0.50 | 0.23% |
| SELL | retest1 | 2025-10-24 11:05:00 | 7880.50 | 2025-10-24 11:50:00 | 7880.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-10-27 09:40:00 | 7894.50 | 2025-10-27 10:30:00 | 7879.83 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest1 | 2025-11-03 10:50:00 | 7768.00 | 2025-11-03 11:00:00 | 7788.19 | PARTIAL | 0.50 | 0.26% |
| BUY | retest1 | 2025-11-03 10:50:00 | 7768.00 | 2025-11-03 11:10:00 | 7768.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-11-04 09:30:00 | 7863.00 | 2025-11-04 09:35:00 | 7848.73 | STOP_HIT | 1.00 | -0.18% |
| BUY | retest1 | 2025-11-06 11:15:00 | 7817.00 | 2025-11-06 11:20:00 | 7800.89 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2025-11-11 10:50:00 | 7469.00 | 2025-11-11 11:10:00 | 7447.54 | PARTIAL | 0.50 | 0.29% |
| SELL | retest1 | 2025-11-11 10:50:00 | 7469.00 | 2025-11-11 13:45:00 | 7469.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-11-13 11:00:00 | 7507.50 | 2025-11-13 11:10:00 | 7495.95 | STOP_HIT | 1.00 | -0.15% |
| SELL | retest1 | 2025-11-18 09:50:00 | 7417.00 | 2025-11-18 10:30:00 | 7400.76 | PARTIAL | 0.50 | 0.22% |
| SELL | retest1 | 2025-11-18 09:50:00 | 7417.00 | 2025-11-18 11:15:00 | 7417.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-11-27 09:40:00 | 7353.50 | 2025-11-27 09:45:00 | 7364.64 | STOP_HIT | 1.00 | -0.15% |
| SELL | retest1 | 2025-12-02 09:30:00 | 7266.00 | 2025-12-02 09:45:00 | 7252.84 | PARTIAL | 0.50 | 0.18% |
| SELL | retest1 | 2025-12-02 09:30:00 | 7266.00 | 2025-12-02 10:25:00 | 7266.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-12-03 10:00:00 | 7182.50 | 2025-12-03 10:10:00 | 7194.65 | STOP_HIT | 1.00 | -0.17% |
| BUY | retest1 | 2025-12-04 10:05:00 | 7202.00 | 2025-12-04 10:25:00 | 7223.87 | PARTIAL | 0.50 | 0.30% |
| BUY | retest1 | 2025-12-04 10:05:00 | 7202.00 | 2025-12-04 11:00:00 | 7202.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-12-08 10:50:00 | 7152.00 | 2025-12-08 11:10:00 | 7137.06 | PARTIAL | 0.50 | 0.21% |
| SELL | retest1 | 2025-12-08 10:50:00 | 7152.00 | 2025-12-08 15:20:00 | 7097.00 | TARGET_HIT | 0.50 | 0.77% |
| SELL | retest1 | 2025-12-10 10:50:00 | 7065.50 | 2025-12-10 11:30:00 | 7048.68 | PARTIAL | 0.50 | 0.24% |
| SELL | retest1 | 2025-12-10 10:50:00 | 7065.50 | 2025-12-10 13:15:00 | 7054.00 | TARGET_HIT | 0.50 | 0.16% |
| BUY | retest1 | 2025-12-12 10:45:00 | 7067.00 | 2025-12-12 11:15:00 | 7084.07 | PARTIAL | 0.50 | 0.24% |
| BUY | retest1 | 2025-12-12 10:45:00 | 7067.00 | 2025-12-12 11:25:00 | 7067.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-12-17 10:55:00 | 6987.00 | 2025-12-17 12:10:00 | 6969.22 | PARTIAL | 0.50 | 0.25% |
| SELL | retest1 | 2025-12-17 10:55:00 | 6987.00 | 2025-12-17 15:20:00 | 6916.50 | TARGET_HIT | 0.50 | 1.01% |
| BUY | retest1 | 2025-12-18 11:05:00 | 6937.50 | 2025-12-18 13:10:00 | 6925.57 | STOP_HIT | 1.00 | -0.17% |
| BUY | retest1 | 2025-12-23 11:15:00 | 7074.00 | 2025-12-23 11:30:00 | 7086.82 | PARTIAL | 0.50 | 0.18% |
| BUY | retest1 | 2025-12-23 11:15:00 | 7074.00 | 2025-12-23 11:45:00 | 7074.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-12-24 09:50:00 | 7120.50 | 2025-12-24 10:15:00 | 7137.25 | PARTIAL | 0.50 | 0.24% |
| BUY | retest1 | 2025-12-24 09:50:00 | 7120.50 | 2025-12-24 14:35:00 | 7168.50 | TARGET_HIT | 0.50 | 0.67% |
| SELL | retest1 | 2025-12-29 10:00:00 | 7131.00 | 2025-12-29 10:20:00 | 7143.74 | STOP_HIT | 1.00 | -0.18% |
| BUY | retest1 | 2026-01-01 10:50:00 | 7090.00 | 2026-01-01 11:00:00 | 7078.55 | STOP_HIT | 1.00 | -0.16% |
| SELL | retest1 | 2026-01-08 10:10:00 | 7380.00 | 2026-01-08 10:50:00 | 7355.66 | PARTIAL | 0.50 | 0.33% |
| SELL | retest1 | 2026-01-08 10:10:00 | 7380.00 | 2026-01-08 11:55:00 | 7380.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-01-19 11:10:00 | 7188.00 | 2026-01-19 11:20:00 | 7169.26 | PARTIAL | 0.50 | 0.26% |
| SELL | retest1 | 2026-01-19 11:10:00 | 7188.00 | 2026-01-19 11:40:00 | 7188.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-01-20 11:00:00 | 7000.00 | 2026-01-20 12:15:00 | 6980.04 | PARTIAL | 0.50 | 0.29% |
| SELL | retest1 | 2026-01-20 11:00:00 | 7000.00 | 2026-01-20 15:20:00 | 6892.00 | TARGET_HIT | 0.50 | 1.54% |
| SELL | retest1 | 2026-01-21 10:55:00 | 6768.00 | 2026-01-21 11:10:00 | 6789.18 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2026-01-28 10:45:00 | 6875.00 | 2026-01-28 11:55:00 | 6860.16 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2026-02-03 11:15:00 | 7118.00 | 2026-02-03 12:10:00 | 7151.71 | PARTIAL | 0.50 | 0.47% |
| BUY | retest1 | 2026-02-03 11:15:00 | 7118.00 | 2026-02-03 12:55:00 | 7118.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-02-06 10:05:00 | 7071.00 | 2026-02-06 10:10:00 | 7046.60 | PARTIAL | 0.50 | 0.35% |
| SELL | retest1 | 2026-02-06 10:05:00 | 7071.00 | 2026-02-06 11:30:00 | 7056.00 | TARGET_HIT | 0.50 | 0.21% |
| SELL | retest1 | 2026-02-09 10:55:00 | 7173.50 | 2026-02-09 12:05:00 | 7190.67 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2026-02-10 09:40:00 | 7208.50 | 2026-02-10 11:00:00 | 7231.70 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2026-02-13 09:45:00 | 7557.50 | 2026-02-13 10:15:00 | 7539.19 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2026-02-16 10:10:00 | 7544.50 | 2026-02-16 12:15:00 | 7572.55 | PARTIAL | 0.50 | 0.37% |
| BUY | retest1 | 2026-02-16 10:10:00 | 7544.50 | 2026-02-16 15:20:00 | 7606.00 | TARGET_HIT | 0.50 | 0.82% |
| BUY | retest1 | 2026-02-18 09:35:00 | 7651.50 | 2026-02-18 09:55:00 | 7635.41 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2026-03-05 11:00:00 | 7670.50 | 2026-03-05 11:55:00 | 7641.98 | PARTIAL | 0.50 | 0.37% |
| SELL | retest1 | 2026-03-05 11:00:00 | 7670.50 | 2026-03-05 12:25:00 | 7670.50 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-03-06 10:45:00 | 7723.00 | 2026-03-06 11:00:00 | 7738.85 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2026-03-10 11:15:00 | 7785.00 | 2026-03-10 11:20:00 | 7800.85 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2026-03-11 11:10:00 | 7712.50 | 2026-03-11 12:10:00 | 7727.44 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest1 | 2026-03-25 11:15:00 | 7558.50 | 2026-03-25 12:00:00 | 7583.14 | PARTIAL | 0.50 | 0.33% |
| BUY | retest1 | 2026-03-25 11:15:00 | 7558.50 | 2026-03-25 15:20:00 | 7586.00 | TARGET_HIT | 0.50 | 0.36% |
| BUY | retest1 | 2026-04-10 11:05:00 | 7554.50 | 2026-04-10 11:30:00 | 7539.76 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2026-04-13 10:45:00 | 7497.00 | 2026-04-13 10:50:00 | 7523.93 | PARTIAL | 0.50 | 0.36% |
| BUY | retest1 | 2026-04-13 10:45:00 | 7497.00 | 2026-04-13 15:20:00 | 7510.50 | TARGET_HIT | 0.50 | 0.18% |
| BUY | retest1 | 2026-04-17 10:05:00 | 7627.00 | 2026-04-17 11:35:00 | 7657.68 | PARTIAL | 0.50 | 0.40% |
| BUY | retest1 | 2026-04-17 10:05:00 | 7627.00 | 2026-04-17 15:20:00 | 7703.50 | TARGET_HIT | 0.50 | 1.00% |
| BUY | retest1 | 2026-04-21 09:50:00 | 7715.00 | 2026-04-21 10:05:00 | 7736.09 | PARTIAL | 0.50 | 0.27% |
| BUY | retest1 | 2026-04-21 09:50:00 | 7715.00 | 2026-04-21 11:35:00 | 7715.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-22 10:30:00 | 7757.00 | 2026-04-22 10:35:00 | 7741.77 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2026-04-24 11:15:00 | 7718.00 | 2026-04-24 11:20:00 | 7736.30 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2026-04-27 11:15:00 | 7792.50 | 2026-04-27 11:30:00 | 7775.73 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2026-04-28 10:55:00 | 7771.50 | 2026-04-28 11:55:00 | 7749.75 | PARTIAL | 0.50 | 0.28% |
| SELL | retest1 | 2026-04-28 10:55:00 | 7771.50 | 2026-04-28 14:30:00 | 7771.50 | STOP_HIT | 0.50 | 0.00% |
