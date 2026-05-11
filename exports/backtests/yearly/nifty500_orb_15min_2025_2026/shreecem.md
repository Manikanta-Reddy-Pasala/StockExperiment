# Shree Cement Ltd. (SHREECEM)

## Backtest Summary

- **Window:** 2025-05-12 09:15:00 → 2026-01-05 15:25:00 (12313 bars)
- **Last close:** 27680.00
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
| ENTRY1 | 74 |
| ENTRY2 | 0 |
| PARTIAL | 33 |
| TARGET_HIT | 15 |
| STOP_HIT | 59 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 107 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 47 / 60
- **Target hits / Stop hits / Partials:** 15 / 59 / 33
- **Avg / median % per leg:** 0.09% / 0.00%
- **Sum % (uncompounded):** 9.98%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 44 | 17 | 38.6% | 5 | 27 | 12 | 0.06% | 2.7% |
| BUY @ 2nd Alert (retest1) | 44 | 17 | 38.6% | 5 | 27 | 12 | 0.06% | 2.7% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 63 | 30 | 47.6% | 10 | 32 | 21 | 0.11% | 7.2% |
| SELL @ 2nd Alert (retest1) | 63 | 30 | 47.6% | 10 | 32 | 21 | 0.11% | 7.2% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 107 | 47 | 43.9% | 15 | 59 | 33 | 0.09% | 10.0% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-13 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-13 10:40:00 | 30445.00 | 30288.45 | 0.00 | ORB-long ORB[30145.00,30370.00] vol=1.8x ATR=72.25 |
| Stop hit — per-position SL triggered | 2025-05-13 10:45:00 | 30372.75 | 30291.99 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2025-05-15 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-15 09:30:00 | 31550.00 | 31259.50 | 0.00 | ORB-long ORB[30895.00,31360.00] vol=2.5x ATR=171.89 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-15 09:50:00 | 31807.83 | 31351.03 | 0.00 | T1 1.5R @ 31807.83 |
| Stop hit — per-position SL triggered | 2025-05-15 10:15:00 | 31550.00 | 31407.03 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2025-05-22 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-22 10:50:00 | 31530.00 | 31489.53 | 0.00 | ORB-long ORB[31305.00,31490.00] vol=4.2x ATR=70.69 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-22 11:55:00 | 31636.03 | 31509.75 | 0.00 | T1 1.5R @ 31636.03 |
| Stop hit — per-position SL triggered | 2025-05-22 12:05:00 | 31530.00 | 31511.55 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2025-05-26 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-26 11:00:00 | 31085.00 | 31326.88 | 0.00 | ORB-short ORB[31425.00,31695.00] vol=2.2x ATR=70.37 |
| Stop hit — per-position SL triggered | 2025-05-26 11:10:00 | 31155.37 | 31315.98 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2025-05-28 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-28 10:45:00 | 30430.00 | 30683.81 | 0.00 | ORB-short ORB[30825.00,31230.00] vol=2.1x ATR=79.48 |
| Stop hit — per-position SL triggered | 2025-05-28 11:05:00 | 30509.48 | 30629.25 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2025-05-29 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-29 11:00:00 | 29630.00 | 29819.68 | 0.00 | ORB-short ORB[29750.00,30020.00] vol=1.6x ATR=64.43 |
| Stop hit — per-position SL triggered | 2025-05-29 11:05:00 | 29694.43 | 29812.32 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2025-05-30 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-30 10:55:00 | 29805.00 | 29819.00 | 0.00 | ORB-short ORB[29815.00,30075.00] vol=1.9x ATR=61.14 |
| Stop hit — per-position SL triggered | 2025-05-30 11:25:00 | 29866.14 | 29814.68 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2025-06-04 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-04 11:10:00 | 29330.00 | 29436.53 | 0.00 | ORB-short ORB[29350.00,29700.00] vol=1.7x ATR=50.16 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-04 11:25:00 | 29254.76 | 29423.53 | 0.00 | T1 1.5R @ 29254.76 |
| Stop hit — per-position SL triggered | 2025-06-04 11:35:00 | 29330.00 | 29400.56 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2025-06-06 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-06 11:05:00 | 29715.00 | 29699.33 | 0.00 | ORB-long ORB[29515.00,29695.00] vol=10.2x ATR=51.31 |
| Stop hit — per-position SL triggered | 2025-06-06 11:45:00 | 29663.69 | 29698.85 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2025-06-10 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-10 10:10:00 | 30035.00 | 29919.75 | 0.00 | ORB-long ORB[29735.00,30000.00] vol=3.3x ATR=60.16 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-10 10:15:00 | 30125.24 | 29979.35 | 0.00 | T1 1.5R @ 30125.24 |
| Stop hit — per-position SL triggered | 2025-06-10 10:45:00 | 30035.00 | 30013.49 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2025-06-12 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-12 11:05:00 | 29850.00 | 29891.84 | 0.00 | ORB-short ORB[29870.00,30045.00] vol=12.8x ATR=57.43 |
| Stop hit — per-position SL triggered | 2025-06-12 11:20:00 | 29907.43 | 29887.90 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2025-06-17 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-17 11:05:00 | 29900.00 | 29838.52 | 0.00 | ORB-long ORB[29750.00,29880.00] vol=2.0x ATR=43.77 |
| Stop hit — per-position SL triggered | 2025-06-17 11:15:00 | 29856.23 | 29840.18 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2025-06-20 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-20 10:35:00 | 29455.00 | 29373.53 | 0.00 | ORB-long ORB[29200.00,29450.00] vol=1.6x ATR=67.01 |
| Stop hit — per-position SL triggered | 2025-06-20 11:40:00 | 29387.99 | 29411.24 | 0.00 | SL hit |

### Cycle 14 — SELL (started 2025-06-24 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-24 10:55:00 | 28495.00 | 28628.69 | 0.00 | ORB-short ORB[28575.00,28770.00] vol=1.9x ATR=52.47 |
| Stop hit — per-position SL triggered | 2025-06-24 11:00:00 | 28547.47 | 28619.72 | 0.00 | SL hit |

### Cycle 15 — BUY (started 2025-06-26 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-26 09:50:00 | 29730.00 | 29603.35 | 0.00 | ORB-long ORB[29130.00,29555.00] vol=4.3x ATR=89.53 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-26 10:00:00 | 29864.30 | 29628.25 | 0.00 | T1 1.5R @ 29864.30 |
| Stop hit — per-position SL triggered | 2025-06-26 10:35:00 | 29730.00 | 29760.72 | 0.00 | SL hit |

### Cycle 16 — BUY (started 2025-06-27 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-27 10:55:00 | 30770.00 | 30496.56 | 0.00 | ORB-long ORB[30000.00,30460.00] vol=3.3x ATR=78.84 |
| Stop hit — per-position SL triggered | 2025-06-27 11:00:00 | 30691.16 | 30504.54 | 0.00 | SL hit |

### Cycle 17 — SELL (started 2025-07-02 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-02 10:50:00 | 31145.00 | 31204.81 | 0.00 | ORB-short ORB[31225.00,31435.00] vol=2.0x ATR=64.60 |
| Stop hit — per-position SL triggered | 2025-07-02 10:55:00 | 31209.60 | 31205.91 | 0.00 | SL hit |

### Cycle 18 — SELL (started 2025-07-04 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-04 11:10:00 | 31325.00 | 31400.64 | 0.00 | ORB-short ORB[31345.00,31575.00] vol=2.5x ATR=48.54 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-04 11:35:00 | 31252.20 | 31384.43 | 0.00 | T1 1.5R @ 31252.20 |
| Stop hit — per-position SL triggered | 2025-07-04 12:35:00 | 31325.00 | 31353.65 | 0.00 | SL hit |

### Cycle 19 — SELL (started 2025-07-07 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-07 10:05:00 | 31080.00 | 31216.19 | 0.00 | ORB-short ORB[31245.00,31385.00] vol=1.7x ATR=61.65 |
| Stop hit — per-position SL triggered | 2025-07-07 10:40:00 | 31141.65 | 31186.69 | 0.00 | SL hit |

### Cycle 20 — SELL (started 2025-07-08 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-08 11:05:00 | 31065.00 | 31107.93 | 0.00 | ORB-short ORB[31075.00,31270.00] vol=2.2x ATR=41.38 |
| Stop hit — per-position SL triggered | 2025-07-08 11:45:00 | 31106.38 | 31113.13 | 0.00 | SL hit |

### Cycle 21 — BUY (started 2025-07-09 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-09 10:50:00 | 31445.00 | 31390.65 | 0.00 | ORB-long ORB[31150.00,31295.00] vol=4.6x ATR=57.96 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-09 11:25:00 | 31531.94 | 31412.97 | 0.00 | T1 1.5R @ 31531.94 |
| Target hit | 2025-07-09 14:15:00 | 31525.00 | 31537.93 | 0.00 | Trail-exit close<VWAP |

### Cycle 22 — BUY (started 2025-07-11 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-11 09:40:00 | 31570.00 | 31367.87 | 0.00 | ORB-long ORB[31170.00,31330.00] vol=1.7x ATR=69.02 |
| Stop hit — per-position SL triggered | 2025-07-11 09:45:00 | 31500.98 | 31392.38 | 0.00 | SL hit |

### Cycle 23 — BUY (started 2025-07-14 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-14 11:00:00 | 31430.00 | 31358.85 | 0.00 | ORB-long ORB[31140.00,31360.00] vol=2.5x ATR=50.09 |
| Stop hit — per-position SL triggered | 2025-07-14 11:05:00 | 31379.91 | 31360.83 | 0.00 | SL hit |

### Cycle 24 — SELL (started 2025-07-15 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-15 11:10:00 | 30950.00 | 31024.96 | 0.00 | ORB-short ORB[31035.00,31195.00] vol=1.5x ATR=53.19 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-15 12:20:00 | 30870.22 | 30996.63 | 0.00 | T1 1.5R @ 30870.22 |
| Stop hit — per-position SL triggered | 2025-07-15 12:30:00 | 30950.00 | 30995.05 | 0.00 | SL hit |

### Cycle 25 — SELL (started 2025-07-16 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-16 10:40:00 | 30800.00 | 30909.60 | 0.00 | ORB-short ORB[30910.00,31190.00] vol=1.5x ATR=42.20 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-16 11:40:00 | 30736.69 | 30871.15 | 0.00 | T1 1.5R @ 30736.69 |
| Stop hit — per-position SL triggered | 2025-07-16 13:20:00 | 30800.00 | 30824.63 | 0.00 | SL hit |

### Cycle 26 — BUY (started 2025-07-21 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-21 09:45:00 | 31190.00 | 31091.28 | 0.00 | ORB-long ORB[30880.00,31150.00] vol=2.1x ATR=71.80 |
| Stop hit — per-position SL triggered | 2025-07-21 10:05:00 | 31118.20 | 31129.69 | 0.00 | SL hit |

### Cycle 27 — BUY (started 2025-07-22 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-22 10:10:00 | 32010.00 | 31804.91 | 0.00 | ORB-long ORB[31500.00,31740.00] vol=2.0x ATR=98.38 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-22 10:35:00 | 32157.58 | 31904.11 | 0.00 | T1 1.5R @ 32157.58 |
| Target hit | 2025-07-22 15:20:00 | 32425.00 | 32161.16 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 28 — SELL (started 2025-07-25 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-25 10:40:00 | 31375.00 | 31506.66 | 0.00 | ORB-short ORB[31460.00,31745.00] vol=3.0x ATR=66.70 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-25 10:45:00 | 31274.95 | 31488.22 | 0.00 | T1 1.5R @ 31274.95 |
| Target hit | 2025-07-25 15:20:00 | 30900.00 | 31106.46 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 29 — BUY (started 2025-07-28 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-28 09:30:00 | 31415.00 | 31245.56 | 0.00 | ORB-long ORB[30945.00,31290.00] vol=2.8x ATR=86.18 |
| Stop hit — per-position SL triggered | 2025-07-28 09:45:00 | 31328.82 | 31306.62 | 0.00 | SL hit |

### Cycle 30 — SELL (started 2025-07-30 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-30 10:55:00 | 30305.00 | 30449.77 | 0.00 | ORB-short ORB[30345.00,30595.00] vol=2.1x ATR=73.70 |
| Stop hit — per-position SL triggered | 2025-07-30 11:20:00 | 30378.70 | 30408.21 | 0.00 | SL hit |

### Cycle 31 — BUY (started 2025-07-31 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-31 11:05:00 | 30750.00 | 30657.81 | 0.00 | ORB-long ORB[30390.00,30600.00] vol=1.5x ATR=54.26 |
| Stop hit — per-position SL triggered | 2025-07-31 11:10:00 | 30695.74 | 30658.76 | 0.00 | SL hit |

### Cycle 32 — SELL (started 2025-08-06 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-06 11:10:00 | 30135.00 | 30253.45 | 0.00 | ORB-short ORB[30250.00,30420.00] vol=8.8x ATR=81.20 |
| Stop hit — per-position SL triggered | 2025-08-06 11:25:00 | 30216.20 | 30249.08 | 0.00 | SL hit |

### Cycle 33 — BUY (started 2025-08-11 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-11 10:50:00 | 30625.00 | 30494.47 | 0.00 | ORB-long ORB[30380.00,30530.00] vol=2.4x ATR=52.26 |
| Stop hit — per-position SL triggered | 2025-08-11 11:10:00 | 30572.74 | 30501.17 | 0.00 | SL hit |

### Cycle 34 — SELL (started 2025-08-13 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-13 11:10:00 | 30395.00 | 30433.30 | 0.00 | ORB-short ORB[30450.00,30735.00] vol=3.1x ATR=57.72 |
| Stop hit — per-position SL triggered | 2025-08-13 11:15:00 | 30452.72 | 30428.38 | 0.00 | SL hit |

### Cycle 35 — SELL (started 2025-08-14 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-14 11:05:00 | 30345.00 | 30477.59 | 0.00 | ORB-short ORB[30450.00,30610.00] vol=1.9x ATR=46.66 |
| Stop hit — per-position SL triggered | 2025-08-14 13:25:00 | 30391.66 | 30391.59 | 0.00 | SL hit |

### Cycle 36 — SELL (started 2025-08-19 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-19 10:05:00 | 31155.00 | 31340.08 | 0.00 | ORB-short ORB[31310.00,31600.00] vol=2.7x ATR=81.31 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-19 10:35:00 | 31033.04 | 31251.17 | 0.00 | T1 1.5R @ 31033.04 |
| Target hit | 2025-08-19 15:20:00 | 30900.00 | 30965.16 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 37 — SELL (started 2025-08-22 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-22 10:05:00 | 30370.00 | 30477.55 | 0.00 | ORB-short ORB[30420.00,30800.00] vol=5.3x ATR=73.67 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-22 10:35:00 | 30259.49 | 30433.20 | 0.00 | T1 1.5R @ 30259.49 |
| Target hit | 2025-08-22 15:20:00 | 30000.00 | 30180.05 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 38 — BUY (started 2025-08-25 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-25 10:40:00 | 30345.00 | 30254.83 | 0.00 | ORB-long ORB[30000.00,30325.00] vol=2.6x ATR=72.25 |
| Stop hit — per-position SL triggered | 2025-08-25 10:55:00 | 30272.75 | 30265.61 | 0.00 | SL hit |

### Cycle 39 — SELL (started 2025-08-29 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-29 09:35:00 | 29635.00 | 29698.82 | 0.00 | ORB-short ORB[29705.00,29930.00] vol=3.8x ATR=74.73 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-29 09:45:00 | 29522.90 | 29685.86 | 0.00 | T1 1.5R @ 29522.90 |
| Stop hit — per-position SL triggered | 2025-08-29 09:55:00 | 29635.00 | 29671.97 | 0.00 | SL hit |

### Cycle 40 — BUY (started 2025-09-02 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-02 10:45:00 | 29995.00 | 29880.28 | 0.00 | ORB-long ORB[29745.00,29970.00] vol=2.0x ATR=43.92 |
| Stop hit — per-position SL triggered | 2025-09-02 10:50:00 | 29951.08 | 29901.03 | 0.00 | SL hit |

### Cycle 41 — BUY (started 2025-09-03 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-03 09:35:00 | 30180.00 | 30096.04 | 0.00 | ORB-long ORB[29945.00,30130.00] vol=1.9x ATR=70.17 |
| Stop hit — per-position SL triggered | 2025-09-03 09:55:00 | 30109.83 | 30129.76 | 0.00 | SL hit |

### Cycle 42 — SELL (started 2025-09-05 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-05 10:20:00 | 29875.00 | 29887.24 | 0.00 | ORB-short ORB[29935.00,30100.00] vol=2.1x ATR=65.66 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-05 11:10:00 | 29776.51 | 29870.26 | 0.00 | T1 1.5R @ 29776.51 |
| Target hit | 2025-09-05 13:15:00 | 29860.00 | 29829.30 | 0.00 | Trail-exit close>VWAP |

### Cycle 43 — SELL (started 2025-09-10 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-10 10:50:00 | 29970.00 | 30157.10 | 0.00 | ORB-short ORB[30300.00,30520.00] vol=1.7x ATR=45.67 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-10 11:40:00 | 29901.50 | 30113.26 | 0.00 | T1 1.5R @ 29901.50 |
| Target hit | 2025-09-10 15:20:00 | 29855.00 | 29933.35 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 44 — SELL (started 2025-09-11 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-11 10:55:00 | 29795.00 | 29935.66 | 0.00 | ORB-short ORB[29920.00,30100.00] vol=1.7x ATR=43.34 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-11 11:30:00 | 29729.99 | 29901.44 | 0.00 | T1 1.5R @ 29729.99 |
| Stop hit — per-position SL triggered | 2025-09-11 12:55:00 | 29795.00 | 29817.88 | 0.00 | SL hit |

### Cycle 45 — SELL (started 2025-09-12 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-12 10:55:00 | 29740.00 | 29784.94 | 0.00 | ORB-short ORB[29785.00,29935.00] vol=2.0x ATR=43.39 |
| Stop hit — per-position SL triggered | 2025-09-12 11:25:00 | 29783.39 | 29778.11 | 0.00 | SL hit |

### Cycle 46 — SELL (started 2025-09-17 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-17 10:55:00 | 29725.00 | 29919.91 | 0.00 | ORB-short ORB[29755.00,30145.00] vol=3.7x ATR=79.11 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-17 12:25:00 | 29606.34 | 29840.63 | 0.00 | T1 1.5R @ 29606.34 |
| Stop hit — per-position SL triggered | 2025-09-17 13:05:00 | 29725.00 | 29818.95 | 0.00 | SL hit |

### Cycle 47 — BUY (started 2025-09-18 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-18 11:10:00 | 29915.00 | 29835.48 | 0.00 | ORB-long ORB[29650.00,29880.00] vol=1.5x ATR=39.82 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-18 11:45:00 | 29974.73 | 29854.68 | 0.00 | T1 1.5R @ 29974.73 |
| Stop hit — per-position SL triggered | 2025-09-18 12:40:00 | 29915.00 | 29901.30 | 0.00 | SL hit |

### Cycle 48 — SELL (started 2025-09-19 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-19 10:20:00 | 29790.00 | 29805.76 | 0.00 | ORB-short ORB[29860.00,30145.00] vol=12.7x ATR=69.54 |
| Stop hit — per-position SL triggered | 2025-09-19 10:25:00 | 29859.54 | 29806.55 | 0.00 | SL hit |

### Cycle 49 — BUY (started 2025-09-23 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-23 10:05:00 | 30080.00 | 29895.80 | 0.00 | ORB-long ORB[29840.00,29995.00] vol=1.6x ATR=72.24 |
| Stop hit — per-position SL triggered | 2025-09-23 10:15:00 | 30007.76 | 29910.74 | 0.00 | SL hit |

### Cycle 50 — SELL (started 2025-09-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-26 11:15:00 | 29065.00 | 29221.18 | 0.00 | ORB-short ORB[29320.00,29635.00] vol=1.9x ATR=57.64 |
| Stop hit — per-position SL triggered | 2025-09-26 11:20:00 | 29122.64 | 29214.61 | 0.00 | SL hit |

### Cycle 51 — BUY (started 2025-09-29 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-29 11:10:00 | 29015.00 | 28862.83 | 0.00 | ORB-long ORB[28755.00,28990.00] vol=1.9x ATR=52.51 |
| Stop hit — per-position SL triggered | 2025-09-29 11:15:00 | 28962.49 | 28873.09 | 0.00 | SL hit |

### Cycle 52 — SELL (started 2025-10-08 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-08 11:00:00 | 29285.00 | 29451.57 | 0.00 | ORB-short ORB[29425.00,29690.00] vol=2.0x ATR=48.28 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-08 11:15:00 | 29212.58 | 29431.69 | 0.00 | T1 1.5R @ 29212.58 |
| Stop hit — per-position SL triggered | 2025-10-08 11:50:00 | 29285.00 | 29391.86 | 0.00 | SL hit |

### Cycle 53 — BUY (started 2025-10-09 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-09 10:50:00 | 29645.00 | 29565.88 | 0.00 | ORB-long ORB[29295.00,29485.00] vol=3.6x ATR=54.22 |
| Stop hit — per-position SL triggered | 2025-10-09 12:35:00 | 29590.78 | 29615.54 | 0.00 | SL hit |

### Cycle 54 — SELL (started 2025-10-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-14 11:15:00 | 29540.00 | 29575.42 | 0.00 | ORB-short ORB[29640.00,29745.00] vol=2.9x ATR=54.60 |
| Stop hit — per-position SL triggered | 2025-10-14 11:30:00 | 29594.60 | 29579.98 | 0.00 | SL hit |

### Cycle 55 — BUY (started 2025-10-15 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-15 10:05:00 | 29755.00 | 29662.13 | 0.00 | ORB-long ORB[29610.00,29710.00] vol=1.5x ATR=59.70 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-15 10:50:00 | 29844.54 | 29709.04 | 0.00 | T1 1.5R @ 29844.54 |
| Stop hit — per-position SL triggered | 2025-10-15 11:15:00 | 29755.00 | 29721.83 | 0.00 | SL hit |

### Cycle 56 — SELL (started 2025-10-17 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-17 09:30:00 | 29785.00 | 29853.39 | 0.00 | ORB-short ORB[29810.00,30040.00] vol=1.7x ATR=67.05 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-17 09:40:00 | 29684.43 | 29815.98 | 0.00 | T1 1.5R @ 29684.43 |
| Target hit | 2025-10-17 10:55:00 | 29735.00 | 29699.00 | 0.00 | Trail-exit close>VWAP |

### Cycle 57 — SELL (started 2025-11-03 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-03 11:00:00 | 28060.00 | 28223.69 | 0.00 | ORB-short ORB[28150.00,28330.00] vol=2.3x ATR=49.01 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-03 13:05:00 | 27986.48 | 28147.81 | 0.00 | T1 1.5R @ 27986.48 |
| Target hit | 2025-11-03 15:20:00 | 27985.00 | 28067.04 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 58 — SELL (started 2025-11-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-04 11:15:00 | 27800.00 | 27933.32 | 0.00 | ORB-short ORB[27880.00,28095.00] vol=2.0x ATR=33.95 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-04 11:35:00 | 27749.07 | 27911.49 | 0.00 | T1 1.5R @ 27749.07 |
| Target hit | 2025-11-04 15:20:00 | 27625.00 | 27714.81 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 59 — BUY (started 2025-11-07 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-07 10:40:00 | 27525.00 | 27393.64 | 0.00 | ORB-long ORB[27210.00,27410.00] vol=4.8x ATR=68.69 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-07 11:00:00 | 27628.03 | 27417.54 | 0.00 | T1 1.5R @ 27628.03 |
| Stop hit — per-position SL triggered | 2025-11-07 13:00:00 | 27525.00 | 27546.46 | 0.00 | SL hit |

### Cycle 60 — SELL (started 2025-11-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-11 11:15:00 | 27010.00 | 27040.55 | 0.00 | ORB-short ORB[27075.00,27250.00] vol=2.0x ATR=50.59 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-11 11:40:00 | 26934.11 | 27033.03 | 0.00 | T1 1.5R @ 26934.11 |
| Stop hit — per-position SL triggered | 2025-11-11 11:45:00 | 27010.00 | 27031.82 | 0.00 | SL hit |

### Cycle 61 — SELL (started 2025-11-12 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-12 11:10:00 | 27005.00 | 27032.97 | 0.00 | ORB-short ORB[27060.00,27245.00] vol=1.6x ATR=40.26 |
| Stop hit — per-position SL triggered | 2025-11-12 11:25:00 | 27045.26 | 27033.93 | 0.00 | SL hit |

### Cycle 62 — SELL (started 2025-11-18 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-18 09:45:00 | 26430.00 | 26545.38 | 0.00 | ORB-short ORB[26515.00,26760.00] vol=1.6x ATR=48.00 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-18 09:50:00 | 26357.99 | 26524.35 | 0.00 | T1 1.5R @ 26357.99 |
| Stop hit — per-position SL triggered | 2025-11-18 10:25:00 | 26430.00 | 26461.15 | 0.00 | SL hit |

### Cycle 63 — BUY (started 2025-11-21 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-21 10:50:00 | 26570.00 | 26532.68 | 0.00 | ORB-long ORB[26440.00,26530.00] vol=2.3x ATR=40.29 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-21 11:15:00 | 26630.44 | 26560.77 | 0.00 | T1 1.5R @ 26630.44 |
| Target hit | 2025-11-21 13:40:00 | 26635.00 | 26642.70 | 0.00 | Trail-exit close<VWAP |

### Cycle 64 — SELL (started 2025-11-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-27 11:15:00 | 26695.00 | 26712.04 | 0.00 | ORB-short ORB[26700.00,26945.00] vol=3.1x ATR=38.64 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-27 11:20:00 | 26637.04 | 26702.43 | 0.00 | T1 1.5R @ 26637.04 |
| Target hit | 2025-11-27 11:20:00 | 26705.00 | 26702.43 | 0.00 | Trail-exit close>VWAP |

### Cycle 65 — SELL (started 2025-11-28 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-28 10:50:00 | 26545.00 | 26643.64 | 0.00 | ORB-short ORB[26645.00,26865.00] vol=1.8x ATR=34.48 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-28 11:00:00 | 26493.28 | 26570.16 | 0.00 | T1 1.5R @ 26493.28 |
| Target hit | 2025-11-28 15:20:00 | 26400.00 | 26444.20 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 66 — SELL (started 2025-12-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-03 10:15:00 | 26205.00 | 26277.84 | 0.00 | ORB-short ORB[26315.00,26685.00] vol=1.7x ATR=58.09 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-03 11:00:00 | 26117.86 | 26249.76 | 0.00 | T1 1.5R @ 26117.86 |
| Stop hit — per-position SL triggered | 2025-12-03 11:05:00 | 26205.00 | 26247.75 | 0.00 | SL hit |

### Cycle 67 — SELL (started 2025-12-05 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-05 11:10:00 | 26180.00 | 26388.41 | 0.00 | ORB-short ORB[26295.00,26455.00] vol=2.4x ATR=42.91 |
| Stop hit — per-position SL triggered | 2025-12-05 11:50:00 | 26222.91 | 26380.26 | 0.00 | SL hit |

### Cycle 68 — SELL (started 2025-12-09 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-09 09:45:00 | 25840.00 | 25989.36 | 0.00 | ORB-short ORB[25930.00,26255.00] vol=1.6x ATR=82.11 |
| Stop hit — per-position SL triggered | 2025-12-09 11:05:00 | 25922.11 | 25913.53 | 0.00 | SL hit |

### Cycle 69 — BUY (started 2025-12-11 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-11 11:10:00 | 26165.00 | 25996.83 | 0.00 | ORB-long ORB[25740.00,26100.00] vol=5.7x ATR=48.11 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-11 12:40:00 | 26237.16 | 26206.19 | 0.00 | T1 1.5R @ 26237.16 |
| Target hit | 2025-12-11 13:00:00 | 26200.00 | 26207.97 | 0.00 | Trail-exit close<VWAP |

### Cycle 70 — BUY (started 2025-12-23 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-23 09:30:00 | 25770.00 | 25687.44 | 0.00 | ORB-long ORB[25600.00,25735.00] vol=2.6x ATR=45.29 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-23 09:35:00 | 25837.94 | 25771.34 | 0.00 | T1 1.5R @ 25837.94 |
| Target hit | 2025-12-23 14:40:00 | 25905.00 | 25978.01 | 0.00 | Trail-exit close<VWAP |

### Cycle 71 — BUY (started 2025-12-26 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-26 10:30:00 | 26350.00 | 26256.52 | 0.00 | ORB-long ORB[26070.00,26260.00] vol=5.7x ATR=48.91 |
| Stop hit — per-position SL triggered | 2025-12-26 11:20:00 | 26301.09 | 26286.28 | 0.00 | SL hit |

### Cycle 72 — SELL (started 2025-12-30 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-30 10:45:00 | 26105.00 | 26203.23 | 0.00 | ORB-short ORB[26125.00,26275.00] vol=3.1x ATR=41.68 |
| Stop hit — per-position SL triggered | 2025-12-30 10:55:00 | 26146.68 | 26194.94 | 0.00 | SL hit |

### Cycle 73 — BUY (started 2025-12-31 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-31 11:05:00 | 26485.00 | 26380.49 | 0.00 | ORB-long ORB[26350.00,26450.00] vol=3.1x ATR=48.30 |
| Stop hit — per-position SL triggered | 2025-12-31 11:15:00 | 26436.70 | 26390.56 | 0.00 | SL hit |

### Cycle 74 — BUY (started 2026-01-01 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-01 11:10:00 | 26650.00 | 26602.22 | 0.00 | ORB-long ORB[26440.00,26600.00] vol=3.5x ATR=36.17 |
| Stop hit — per-position SL triggered | 2026-01-01 12:25:00 | 26613.83 | 26624.36 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-05-13 10:40:00 | 30445.00 | 2025-05-13 10:45:00 | 30372.75 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2025-05-15 09:30:00 | 31550.00 | 2025-05-15 09:50:00 | 31807.83 | PARTIAL | 0.50 | 0.82% |
| BUY | retest1 | 2025-05-15 09:30:00 | 31550.00 | 2025-05-15 10:15:00 | 31550.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-05-22 10:50:00 | 31530.00 | 2025-05-22 11:55:00 | 31636.03 | PARTIAL | 0.50 | 0.34% |
| BUY | retest1 | 2025-05-22 10:50:00 | 31530.00 | 2025-05-22 12:05:00 | 31530.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-05-26 11:00:00 | 31085.00 | 2025-05-26 11:10:00 | 31155.37 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2025-05-28 10:45:00 | 30430.00 | 2025-05-28 11:05:00 | 30509.48 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2025-05-29 11:00:00 | 29630.00 | 2025-05-29 11:05:00 | 29694.43 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2025-05-30 10:55:00 | 29805.00 | 2025-05-30 11:25:00 | 29866.14 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2025-06-04 11:10:00 | 29330.00 | 2025-06-04 11:25:00 | 29254.76 | PARTIAL | 0.50 | 0.26% |
| SELL | retest1 | 2025-06-04 11:10:00 | 29330.00 | 2025-06-04 11:35:00 | 29330.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-06-06 11:05:00 | 29715.00 | 2025-06-06 11:45:00 | 29663.69 | STOP_HIT | 1.00 | -0.17% |
| BUY | retest1 | 2025-06-10 10:10:00 | 30035.00 | 2025-06-10 10:15:00 | 30125.24 | PARTIAL | 0.50 | 0.30% |
| BUY | retest1 | 2025-06-10 10:10:00 | 30035.00 | 2025-06-10 10:45:00 | 30035.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-06-12 11:05:00 | 29850.00 | 2025-06-12 11:20:00 | 29907.43 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest1 | 2025-06-17 11:05:00 | 29900.00 | 2025-06-17 11:15:00 | 29856.23 | STOP_HIT | 1.00 | -0.15% |
| BUY | retest1 | 2025-06-20 10:35:00 | 29455.00 | 2025-06-20 11:40:00 | 29387.99 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2025-06-24 10:55:00 | 28495.00 | 2025-06-24 11:00:00 | 28547.47 | STOP_HIT | 1.00 | -0.18% |
| BUY | retest1 | 2025-06-26 09:50:00 | 29730.00 | 2025-06-26 10:00:00 | 29864.30 | PARTIAL | 0.50 | 0.45% |
| BUY | retest1 | 2025-06-26 09:50:00 | 29730.00 | 2025-06-26 10:35:00 | 29730.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-06-27 10:55:00 | 30770.00 | 2025-06-27 11:00:00 | 30691.16 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2025-07-02 10:50:00 | 31145.00 | 2025-07-02 10:55:00 | 31209.60 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2025-07-04 11:10:00 | 31325.00 | 2025-07-04 11:35:00 | 31252.20 | PARTIAL | 0.50 | 0.23% |
| SELL | retest1 | 2025-07-04 11:10:00 | 31325.00 | 2025-07-04 12:35:00 | 31325.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-07-07 10:05:00 | 31080.00 | 2025-07-07 10:40:00 | 31141.65 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2025-07-08 11:05:00 | 31065.00 | 2025-07-08 11:45:00 | 31106.38 | STOP_HIT | 1.00 | -0.13% |
| BUY | retest1 | 2025-07-09 10:50:00 | 31445.00 | 2025-07-09 11:25:00 | 31531.94 | PARTIAL | 0.50 | 0.28% |
| BUY | retest1 | 2025-07-09 10:50:00 | 31445.00 | 2025-07-09 14:15:00 | 31525.00 | TARGET_HIT | 0.50 | 0.25% |
| BUY | retest1 | 2025-07-11 09:40:00 | 31570.00 | 2025-07-11 09:45:00 | 31500.98 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2025-07-14 11:00:00 | 31430.00 | 2025-07-14 11:05:00 | 31379.91 | STOP_HIT | 1.00 | -0.16% |
| SELL | retest1 | 2025-07-15 11:10:00 | 30950.00 | 2025-07-15 12:20:00 | 30870.22 | PARTIAL | 0.50 | 0.26% |
| SELL | retest1 | 2025-07-15 11:10:00 | 30950.00 | 2025-07-15 12:30:00 | 30950.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-07-16 10:40:00 | 30800.00 | 2025-07-16 11:40:00 | 30736.69 | PARTIAL | 0.50 | 0.21% |
| SELL | retest1 | 2025-07-16 10:40:00 | 30800.00 | 2025-07-16 13:20:00 | 30800.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-07-21 09:45:00 | 31190.00 | 2025-07-21 10:05:00 | 31118.20 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2025-07-22 10:10:00 | 32010.00 | 2025-07-22 10:35:00 | 32157.58 | PARTIAL | 0.50 | 0.46% |
| BUY | retest1 | 2025-07-22 10:10:00 | 32010.00 | 2025-07-22 15:20:00 | 32425.00 | TARGET_HIT | 0.50 | 1.30% |
| SELL | retest1 | 2025-07-25 10:40:00 | 31375.00 | 2025-07-25 10:45:00 | 31274.95 | PARTIAL | 0.50 | 0.32% |
| SELL | retest1 | 2025-07-25 10:40:00 | 31375.00 | 2025-07-25 15:20:00 | 30900.00 | TARGET_HIT | 0.50 | 1.51% |
| BUY | retest1 | 2025-07-28 09:30:00 | 31415.00 | 2025-07-28 09:45:00 | 31328.82 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2025-07-30 10:55:00 | 30305.00 | 2025-07-30 11:20:00 | 30378.70 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2025-07-31 11:05:00 | 30750.00 | 2025-07-31 11:10:00 | 30695.74 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest1 | 2025-08-06 11:10:00 | 30135.00 | 2025-08-06 11:25:00 | 30216.20 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2025-08-11 10:50:00 | 30625.00 | 2025-08-11 11:10:00 | 30572.74 | STOP_HIT | 1.00 | -0.17% |
| SELL | retest1 | 2025-08-13 11:10:00 | 30395.00 | 2025-08-13 11:15:00 | 30452.72 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest1 | 2025-08-14 11:05:00 | 30345.00 | 2025-08-14 13:25:00 | 30391.66 | STOP_HIT | 1.00 | -0.15% |
| SELL | retest1 | 2025-08-19 10:05:00 | 31155.00 | 2025-08-19 10:35:00 | 31033.04 | PARTIAL | 0.50 | 0.39% |
| SELL | retest1 | 2025-08-19 10:05:00 | 31155.00 | 2025-08-19 15:20:00 | 30900.00 | TARGET_HIT | 0.50 | 0.82% |
| SELL | retest1 | 2025-08-22 10:05:00 | 30370.00 | 2025-08-22 10:35:00 | 30259.49 | PARTIAL | 0.50 | 0.36% |
| SELL | retest1 | 2025-08-22 10:05:00 | 30370.00 | 2025-08-22 15:20:00 | 30000.00 | TARGET_HIT | 0.50 | 1.22% |
| BUY | retest1 | 2025-08-25 10:40:00 | 30345.00 | 2025-08-25 10:55:00 | 30272.75 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2025-08-29 09:35:00 | 29635.00 | 2025-08-29 09:45:00 | 29522.90 | PARTIAL | 0.50 | 0.38% |
| SELL | retest1 | 2025-08-29 09:35:00 | 29635.00 | 2025-08-29 09:55:00 | 29635.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-09-02 10:45:00 | 29995.00 | 2025-09-02 10:50:00 | 29951.08 | STOP_HIT | 1.00 | -0.15% |
| BUY | retest1 | 2025-09-03 09:35:00 | 30180.00 | 2025-09-03 09:55:00 | 30109.83 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2025-09-05 10:20:00 | 29875.00 | 2025-09-05 11:10:00 | 29776.51 | PARTIAL | 0.50 | 0.33% |
| SELL | retest1 | 2025-09-05 10:20:00 | 29875.00 | 2025-09-05 13:15:00 | 29860.00 | TARGET_HIT | 0.50 | 0.05% |
| SELL | retest1 | 2025-09-10 10:50:00 | 29970.00 | 2025-09-10 11:40:00 | 29901.50 | PARTIAL | 0.50 | 0.23% |
| SELL | retest1 | 2025-09-10 10:50:00 | 29970.00 | 2025-09-10 15:20:00 | 29855.00 | TARGET_HIT | 0.50 | 0.38% |
| SELL | retest1 | 2025-09-11 10:55:00 | 29795.00 | 2025-09-11 11:30:00 | 29729.99 | PARTIAL | 0.50 | 0.22% |
| SELL | retest1 | 2025-09-11 10:55:00 | 29795.00 | 2025-09-11 12:55:00 | 29795.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-09-12 10:55:00 | 29740.00 | 2025-09-12 11:25:00 | 29783.39 | STOP_HIT | 1.00 | -0.15% |
| SELL | retest1 | 2025-09-17 10:55:00 | 29725.00 | 2025-09-17 12:25:00 | 29606.34 | PARTIAL | 0.50 | 0.40% |
| SELL | retest1 | 2025-09-17 10:55:00 | 29725.00 | 2025-09-17 13:05:00 | 29725.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-09-18 11:10:00 | 29915.00 | 2025-09-18 11:45:00 | 29974.73 | PARTIAL | 0.50 | 0.20% |
| BUY | retest1 | 2025-09-18 11:10:00 | 29915.00 | 2025-09-18 12:40:00 | 29915.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-09-19 10:20:00 | 29790.00 | 2025-09-19 10:25:00 | 29859.54 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2025-09-23 10:05:00 | 30080.00 | 2025-09-23 10:15:00 | 30007.76 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2025-09-26 11:15:00 | 29065.00 | 2025-09-26 11:20:00 | 29122.64 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2025-09-29 11:10:00 | 29015.00 | 2025-09-29 11:15:00 | 28962.49 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest1 | 2025-10-08 11:00:00 | 29285.00 | 2025-10-08 11:15:00 | 29212.58 | PARTIAL | 0.50 | 0.25% |
| SELL | retest1 | 2025-10-08 11:00:00 | 29285.00 | 2025-10-08 11:50:00 | 29285.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-10-09 10:50:00 | 29645.00 | 2025-10-09 12:35:00 | 29590.78 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest1 | 2025-10-14 11:15:00 | 29540.00 | 2025-10-14 11:30:00 | 29594.60 | STOP_HIT | 1.00 | -0.18% |
| BUY | retest1 | 2025-10-15 10:05:00 | 29755.00 | 2025-10-15 10:50:00 | 29844.54 | PARTIAL | 0.50 | 0.30% |
| BUY | retest1 | 2025-10-15 10:05:00 | 29755.00 | 2025-10-15 11:15:00 | 29755.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-10-17 09:30:00 | 29785.00 | 2025-10-17 09:40:00 | 29684.43 | PARTIAL | 0.50 | 0.34% |
| SELL | retest1 | 2025-10-17 09:30:00 | 29785.00 | 2025-10-17 10:55:00 | 29735.00 | TARGET_HIT | 0.50 | 0.17% |
| SELL | retest1 | 2025-11-03 11:00:00 | 28060.00 | 2025-11-03 13:05:00 | 27986.48 | PARTIAL | 0.50 | 0.26% |
| SELL | retest1 | 2025-11-03 11:00:00 | 28060.00 | 2025-11-03 15:20:00 | 27985.00 | TARGET_HIT | 0.50 | 0.27% |
| SELL | retest1 | 2025-11-04 11:15:00 | 27800.00 | 2025-11-04 11:35:00 | 27749.07 | PARTIAL | 0.50 | 0.18% |
| SELL | retest1 | 2025-11-04 11:15:00 | 27800.00 | 2025-11-04 15:20:00 | 27625.00 | TARGET_HIT | 0.50 | 0.63% |
| BUY | retest1 | 2025-11-07 10:40:00 | 27525.00 | 2025-11-07 11:00:00 | 27628.03 | PARTIAL | 0.50 | 0.37% |
| BUY | retest1 | 2025-11-07 10:40:00 | 27525.00 | 2025-11-07 13:00:00 | 27525.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-11-11 11:15:00 | 27010.00 | 2025-11-11 11:40:00 | 26934.11 | PARTIAL | 0.50 | 0.28% |
| SELL | retest1 | 2025-11-11 11:15:00 | 27010.00 | 2025-11-11 11:45:00 | 27010.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-11-12 11:10:00 | 27005.00 | 2025-11-12 11:25:00 | 27045.26 | STOP_HIT | 1.00 | -0.15% |
| SELL | retest1 | 2025-11-18 09:45:00 | 26430.00 | 2025-11-18 09:50:00 | 26357.99 | PARTIAL | 0.50 | 0.27% |
| SELL | retest1 | 2025-11-18 09:45:00 | 26430.00 | 2025-11-18 10:25:00 | 26430.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-11-21 10:50:00 | 26570.00 | 2025-11-21 11:15:00 | 26630.44 | PARTIAL | 0.50 | 0.23% |
| BUY | retest1 | 2025-11-21 10:50:00 | 26570.00 | 2025-11-21 13:40:00 | 26635.00 | TARGET_HIT | 0.50 | 0.24% |
| SELL | retest1 | 2025-11-27 11:15:00 | 26695.00 | 2025-11-27 11:20:00 | 26637.04 | PARTIAL | 0.50 | 0.22% |
| SELL | retest1 | 2025-11-27 11:15:00 | 26695.00 | 2025-11-27 11:20:00 | 26705.00 | TARGET_HIT | 0.50 | -0.04% |
| SELL | retest1 | 2025-11-28 10:50:00 | 26545.00 | 2025-11-28 11:00:00 | 26493.28 | PARTIAL | 0.50 | 0.19% |
| SELL | retest1 | 2025-11-28 10:50:00 | 26545.00 | 2025-11-28 15:20:00 | 26400.00 | TARGET_HIT | 0.50 | 0.55% |
| SELL | retest1 | 2025-12-03 10:15:00 | 26205.00 | 2025-12-03 11:00:00 | 26117.86 | PARTIAL | 0.50 | 0.33% |
| SELL | retest1 | 2025-12-03 10:15:00 | 26205.00 | 2025-12-03 11:05:00 | 26205.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-12-05 11:10:00 | 26180.00 | 2025-12-05 11:50:00 | 26222.91 | STOP_HIT | 1.00 | -0.16% |
| SELL | retest1 | 2025-12-09 09:45:00 | 25840.00 | 2025-12-09 11:05:00 | 25922.11 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2025-12-11 11:10:00 | 26165.00 | 2025-12-11 12:40:00 | 26237.16 | PARTIAL | 0.50 | 0.28% |
| BUY | retest1 | 2025-12-11 11:10:00 | 26165.00 | 2025-12-11 13:00:00 | 26200.00 | TARGET_HIT | 0.50 | 0.13% |
| BUY | retest1 | 2025-12-23 09:30:00 | 25770.00 | 2025-12-23 09:35:00 | 25837.94 | PARTIAL | 0.50 | 0.26% |
| BUY | retest1 | 2025-12-23 09:30:00 | 25770.00 | 2025-12-23 14:40:00 | 25905.00 | TARGET_HIT | 0.50 | 0.52% |
| BUY | retest1 | 2025-12-26 10:30:00 | 26350.00 | 2025-12-26 11:20:00 | 26301.09 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest1 | 2025-12-30 10:45:00 | 26105.00 | 2025-12-30 10:55:00 | 26146.68 | STOP_HIT | 1.00 | -0.16% |
| BUY | retest1 | 2025-12-31 11:05:00 | 26485.00 | 2025-12-31 11:15:00 | 26436.70 | STOP_HIT | 1.00 | -0.18% |
| BUY | retest1 | 2026-01-01 11:10:00 | 26650.00 | 2026-01-01 12:25:00 | 26613.83 | STOP_HIT | 1.00 | -0.14% |
