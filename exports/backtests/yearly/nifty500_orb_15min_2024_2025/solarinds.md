# Solar Industries India Ltd. (SOLARINDS)

## Backtest Summary

- **Window:** 2024-05-13 09:15:00 → 2025-05-05 15:25:00 (18108 bars)
- **Last close:** 13176.00
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
| ENTRY1 | 48 |
| ENTRY2 | 0 |
| PARTIAL | 19 |
| TARGET_HIT | 7 |
| STOP_HIT | 41 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 67 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 26 / 41
- **Target hits / Stop hits / Partials:** 7 / 41 / 19
- **Avg / median % per leg:** 0.17% / 0.00%
- **Sum % (uncompounded):** 11.11%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 34 | 14 | 41.2% | 4 | 20 | 10 | 0.22% | 7.5% |
| BUY @ 2nd Alert (retest1) | 34 | 14 | 41.2% | 4 | 20 | 10 | 0.22% | 7.5% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 33 | 12 | 36.4% | 3 | 21 | 9 | 0.11% | 3.6% |
| SELL @ 2nd Alert (retest1) | 33 | 12 | 36.4% | 3 | 21 | 9 | 0.11% | 3.6% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 67 | 26 | 38.8% | 7 | 41 | 19 | 0.17% | 11.1% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-05-15 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-15 10:05:00 | 8520.50 | 8587.07 | 0.00 | ORB-short ORB[8550.15,8635.70] vol=1.8x ATR=28.30 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-15 10:15:00 | 8478.05 | 8571.31 | 0.00 | T1 1.5R @ 8478.05 |
| Target hit | 2024-05-15 15:20:00 | 8329.75 | 8432.58 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 2 — SELL (started 2024-06-21 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-21 10:45:00 | 9771.30 | 9819.76 | 0.00 | ORB-short ORB[9776.15,9889.15] vol=2.8x ATR=26.82 |
| Stop hit — per-position SL triggered | 2024-06-21 11:50:00 | 9798.12 | 9810.25 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2024-06-28 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-28 09:35:00 | 9810.10 | 9867.90 | 0.00 | ORB-short ORB[9833.60,9940.00] vol=2.0x ATR=36.44 |
| Stop hit — per-position SL triggered | 2024-06-28 09:50:00 | 9846.54 | 9864.17 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2024-07-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-09 10:15:00 | 12034.40 | 12200.48 | 0.00 | ORB-short ORB[12193.95,12340.00] vol=2.2x ATR=62.75 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-09 10:25:00 | 11940.27 | 12157.08 | 0.00 | T1 1.5R @ 11940.27 |
| Stop hit — per-position SL triggered | 2024-07-09 10:30:00 | 12034.40 | 12151.87 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2024-07-25 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-25 10:25:00 | 10376.80 | 10468.55 | 0.00 | ORB-short ORB[10436.85,10566.25] vol=2.2x ATR=51.85 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-25 10:30:00 | 10299.03 | 10409.95 | 0.00 | T1 1.5R @ 10299.03 |
| Stop hit — per-position SL triggered | 2024-07-25 15:20:00 | 10376.90 | 10206.44 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 6 — BUY (started 2024-08-06 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-06 09:35:00 | 10630.20 | 10536.48 | 0.00 | ORB-long ORB[10400.00,10530.00] vol=4.5x ATR=49.26 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-06 09:55:00 | 10704.09 | 10591.33 | 0.00 | T1 1.5R @ 10704.09 |
| Stop hit — per-position SL triggered | 2024-08-06 10:40:00 | 10630.20 | 10637.54 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2024-08-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-14 11:15:00 | 10037.85 | 10048.45 | 0.00 | ORB-short ORB[10050.00,10178.85] vol=2.5x ATR=30.05 |
| Stop hit — per-position SL triggered | 2024-08-14 11:50:00 | 10067.90 | 10048.38 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2024-08-16 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-16 11:00:00 | 10150.15 | 10204.41 | 0.00 | ORB-short ORB[10176.05,10259.50] vol=2.9x ATR=24.29 |
| Stop hit — per-position SL triggered | 2024-08-16 11:20:00 | 10174.44 | 10200.94 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2024-08-20 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-20 11:05:00 | 10313.35 | 10361.92 | 0.00 | ORB-short ORB[10334.05,10428.75] vol=2.1x ATR=28.76 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-20 11:30:00 | 10270.21 | 10351.51 | 0.00 | T1 1.5R @ 10270.21 |
| Stop hit — per-position SL triggered | 2024-08-20 11:40:00 | 10313.35 | 10346.02 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2024-08-29 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-29 09:30:00 | 10325.00 | 10377.57 | 0.00 | ORB-short ORB[10345.60,10450.00] vol=1.8x ATR=52.34 |
| Stop hit — per-position SL triggered | 2024-08-29 09:35:00 | 10377.34 | 10376.59 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2024-08-30 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-30 09:40:00 | 10498.75 | 10462.55 | 0.00 | ORB-long ORB[10407.65,10490.00] vol=1.8x ATR=55.97 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-30 10:25:00 | 10582.70 | 10493.01 | 0.00 | T1 1.5R @ 10582.70 |
| Target hit | 2024-08-30 15:10:00 | 10778.95 | 10779.18 | 0.00 | Trail-exit close<VWAP |

### Cycle 12 — SELL (started 2024-09-05 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-05 10:55:00 | 10998.95 | 11047.77 | 0.00 | ORB-short ORB[11040.45,11164.35] vol=2.0x ATR=41.89 |
| Stop hit — per-position SL triggered | 2024-09-05 11:10:00 | 11040.84 | 11045.38 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2024-09-11 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-11 10:05:00 | 11025.00 | 10955.47 | 0.00 | ORB-long ORB[10763.55,10850.05] vol=6.2x ATR=52.73 |
| Stop hit — per-position SL triggered | 2024-09-11 11:50:00 | 10972.27 | 10999.40 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2024-09-12 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-12 09:35:00 | 11298.85 | 11170.11 | 0.00 | ORB-long ORB[11033.15,11147.00] vol=2.3x ATR=59.39 |
| Stop hit — per-position SL triggered | 2024-09-12 09:40:00 | 11239.46 | 11188.16 | 0.00 | SL hit |

### Cycle 15 — SELL (started 2024-09-16 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-16 11:10:00 | 10868.35 | 11009.65 | 0.00 | ORB-short ORB[11036.45,11195.40] vol=4.6x ATR=33.10 |
| Stop hit — per-position SL triggered | 2024-09-16 11:15:00 | 10901.45 | 11004.89 | 0.00 | SL hit |

### Cycle 16 — SELL (started 2024-09-30 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-30 10:55:00 | 11545.05 | 11676.86 | 0.00 | ORB-short ORB[11690.25,11830.50] vol=2.3x ATR=54.72 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-30 11:10:00 | 11462.97 | 11640.86 | 0.00 | T1 1.5R @ 11462.97 |
| Stop hit — per-position SL triggered | 2024-09-30 13:15:00 | 11545.05 | 11576.91 | 0.00 | SL hit |

### Cycle 17 — SELL (started 2024-10-07 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-07 10:05:00 | 10600.00 | 10769.52 | 0.00 | ORB-short ORB[10806.25,10949.95] vol=1.7x ATR=56.37 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-07 10:20:00 | 10515.44 | 10725.87 | 0.00 | T1 1.5R @ 10515.44 |
| Stop hit — per-position SL triggered | 2024-10-07 10:25:00 | 10600.00 | 10723.51 | 0.00 | SL hit |

### Cycle 18 — BUY (started 2024-10-16 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-16 10:35:00 | 11499.35 | 11410.63 | 0.00 | ORB-long ORB[11280.90,11443.15] vol=1.6x ATR=47.30 |
| Stop hit — per-position SL triggered | 2024-10-16 10:45:00 | 11452.05 | 11415.55 | 0.00 | SL hit |

### Cycle 19 — BUY (started 2024-11-11 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-11 11:00:00 | 10290.40 | 10172.30 | 0.00 | ORB-long ORB[10075.40,10208.15] vol=4.5x ATR=37.95 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-11 11:15:00 | 10347.33 | 10191.95 | 0.00 | T1 1.5R @ 10347.33 |
| Stop hit — per-position SL triggered | 2024-11-11 11:40:00 | 10290.40 | 10216.68 | 0.00 | SL hit |

### Cycle 20 — SELL (started 2024-12-12 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-12 09:50:00 | 10752.15 | 10796.00 | 0.00 | ORB-short ORB[10796.50,10865.00] vol=1.6x ATR=31.53 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-12 14:35:00 | 10704.86 | 10748.22 | 0.00 | T1 1.5R @ 10704.86 |
| Target hit | 2024-12-12 15:20:00 | 10647.20 | 10725.01 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 21 — SELL (started 2024-12-17 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-17 09:40:00 | 10470.00 | 10549.00 | 0.00 | ORB-short ORB[10503.75,10635.80] vol=2.3x ATR=39.00 |
| Stop hit — per-position SL triggered | 2024-12-17 10:05:00 | 10509.00 | 10537.96 | 0.00 | SL hit |

### Cycle 22 — BUY (started 2025-01-01 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-01 10:50:00 | 9910.55 | 9865.09 | 0.00 | ORB-long ORB[9750.00,9898.20] vol=2.6x ATR=31.97 |
| Stop hit — per-position SL triggered | 2025-01-01 11:10:00 | 9878.58 | 9868.67 | 0.00 | SL hit |

### Cycle 23 — SELL (started 2025-01-02 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-02 11:05:00 | 9755.00 | 9811.08 | 0.00 | ORB-short ORB[9830.00,9908.90] vol=1.6x ATR=25.23 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-02 11:20:00 | 9717.16 | 9800.19 | 0.00 | T1 1.5R @ 9717.16 |
| Target hit | 2025-01-02 14:55:00 | 9725.00 | 9707.82 | 0.00 | Trail-exit close>VWAP |

### Cycle 24 — BUY (started 2025-01-03 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-03 09:30:00 | 9797.55 | 9757.16 | 0.00 | ORB-long ORB[9680.20,9770.00] vol=1.8x ATR=25.89 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-03 09:55:00 | 9836.38 | 9781.24 | 0.00 | T1 1.5R @ 9836.38 |
| Stop hit — per-position SL triggered | 2025-01-03 10:30:00 | 9797.55 | 9788.59 | 0.00 | SL hit |

### Cycle 25 — SELL (started 2025-01-06 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-06 09:30:00 | 9620.00 | 9668.08 | 0.00 | ORB-short ORB[9651.00,9754.20] vol=2.3x ATR=29.86 |
| Stop hit — per-position SL triggered | 2025-01-06 09:35:00 | 9649.86 | 9662.00 | 0.00 | SL hit |

### Cycle 26 — SELL (started 2025-01-08 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-08 09:30:00 | 9700.00 | 9776.55 | 0.00 | ORB-short ORB[9735.80,9833.00] vol=1.7x ATR=37.94 |
| Stop hit — per-position SL triggered | 2025-01-08 09:35:00 | 9737.94 | 9772.57 | 0.00 | SL hit |

### Cycle 27 — SELL (started 2025-01-10 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-10 09:35:00 | 9429.30 | 9533.07 | 0.00 | ORB-short ORB[9515.00,9650.00] vol=2.1x ATR=42.33 |
| Stop hit — per-position SL triggered | 2025-01-10 09:40:00 | 9471.63 | 9520.85 | 0.00 | SL hit |

### Cycle 28 — BUY (started 2025-01-15 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-15 10:50:00 | 9465.00 | 9348.63 | 0.00 | ORB-long ORB[9243.45,9353.15] vol=1.6x ATR=41.39 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-15 11:30:00 | 9527.08 | 9386.41 | 0.00 | T1 1.5R @ 9527.08 |
| Target hit | 2025-01-15 15:20:00 | 9539.55 | 9485.77 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 29 — SELL (started 2025-01-16 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-16 11:05:00 | 9529.95 | 9598.34 | 0.00 | ORB-short ORB[9584.15,9700.00] vol=1.7x ATR=25.86 |
| Stop hit — per-position SL triggered | 2025-01-16 11:10:00 | 9555.81 | 9597.00 | 0.00 | SL hit |

### Cycle 30 — BUY (started 2025-01-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-23 10:15:00 | 9509.50 | 9417.33 | 0.00 | ORB-long ORB[9242.70,9313.95] vol=2.4x ATR=46.20 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-23 13:35:00 | 9578.80 | 9460.23 | 0.00 | T1 1.5R @ 9578.80 |
| Stop hit — per-position SL triggered | 2025-01-23 14:50:00 | 9509.50 | 9492.57 | 0.00 | SL hit |

### Cycle 31 — BUY (started 2025-01-29 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-29 11:10:00 | 9450.30 | 9425.33 | 0.00 | ORB-long ORB[9255.00,9335.65] vol=2.4x ATR=31.98 |
| Stop hit — per-position SL triggered | 2025-01-29 11:20:00 | 9418.32 | 9425.79 | 0.00 | SL hit |

### Cycle 32 — BUY (started 2025-02-01 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-01 10:50:00 | 10437.75 | 10320.09 | 0.00 | ORB-long ORB[10235.90,10300.05] vol=2.1x ATR=30.23 |
| Stop hit — per-position SL triggered | 2025-02-01 11:00:00 | 10407.52 | 10327.41 | 0.00 | SL hit |

### Cycle 33 — BUY (started 2025-02-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-05 10:15:00 | 9873.75 | 9850.18 | 0.00 | ORB-long ORB[9768.90,9870.00] vol=2.3x ATR=41.62 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-05 10:30:00 | 9936.18 | 9888.20 | 0.00 | T1 1.5R @ 9936.18 |
| Target hit | 2025-02-05 14:25:00 | 9960.10 | 9969.74 | 0.00 | Trail-exit close<VWAP |

### Cycle 34 — SELL (started 2025-02-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-10 11:15:00 | 9104.50 | 9153.37 | 0.00 | ORB-short ORB[9105.05,9230.00] vol=1.7x ATR=25.84 |
| Stop hit — per-position SL triggered | 2025-02-10 11:20:00 | 9130.34 | 9152.96 | 0.00 | SL hit |

### Cycle 35 — SELL (started 2025-02-11 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-11 10:20:00 | 8930.40 | 9032.10 | 0.00 | ORB-short ORB[9021.60,9150.00] vol=1.5x ATR=35.52 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-11 10:30:00 | 8877.13 | 9006.33 | 0.00 | T1 1.5R @ 8877.13 |
| Stop hit — per-position SL triggered | 2025-02-11 10:35:00 | 8930.40 | 9003.71 | 0.00 | SL hit |

### Cycle 36 — BUY (started 2025-02-13 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-13 10:30:00 | 9054.80 | 9030.27 | 0.00 | ORB-long ORB[8882.90,9004.40] vol=11.3x ATR=35.60 |
| Stop hit — per-position SL triggered | 2025-02-13 11:25:00 | 9019.20 | 9031.71 | 0.00 | SL hit |

### Cycle 37 — BUY (started 2025-02-24 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-24 09:55:00 | 8824.55 | 8785.33 | 0.00 | ORB-long ORB[8709.60,8822.00] vol=1.6x ATR=37.14 |
| Stop hit — per-position SL triggered | 2025-02-24 10:40:00 | 8787.41 | 8797.83 | 0.00 | SL hit |

### Cycle 38 — BUY (started 2025-03-05 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-05 09:40:00 | 9225.00 | 9152.82 | 0.00 | ORB-long ORB[9060.00,9150.90] vol=1.6x ATR=35.56 |
| Stop hit — per-position SL triggered | 2025-03-05 10:20:00 | 9189.44 | 9182.70 | 0.00 | SL hit |

### Cycle 39 — SELL (started 2025-03-12 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-12 10:05:00 | 9764.90 | 9843.62 | 0.00 | ORB-short ORB[9778.05,9886.15] vol=1.6x ATR=37.90 |
| Stop hit — per-position SL triggered | 2025-03-12 10:40:00 | 9802.80 | 9833.69 | 0.00 | SL hit |

### Cycle 40 — BUY (started 2025-03-13 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-13 09:55:00 | 9880.00 | 9830.22 | 0.00 | ORB-long ORB[9781.10,9859.35] vol=1.7x ATR=30.20 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-13 10:35:00 | 9925.31 | 9864.17 | 0.00 | T1 1.5R @ 9925.31 |
| Target hit | 2025-03-13 14:05:00 | 10124.25 | 10128.88 | 0.00 | Trail-exit close<VWAP |

### Cycle 41 — BUY (started 2025-03-19 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-19 09:30:00 | 10187.80 | 10146.55 | 0.00 | ORB-long ORB[10061.60,10158.55] vol=2.0x ATR=35.13 |
| Stop hit — per-position SL triggered | 2025-03-19 09:35:00 | 10152.67 | 10145.46 | 0.00 | SL hit |

### Cycle 42 — BUY (started 2025-03-21 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-21 09:40:00 | 10549.75 | 10491.49 | 0.00 | ORB-long ORB[10372.00,10508.85] vol=1.6x ATR=34.47 |
| Stop hit — per-position SL triggered | 2025-03-21 09:50:00 | 10515.28 | 10495.67 | 0.00 | SL hit |

### Cycle 43 — BUY (started 2025-03-24 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-24 10:40:00 | 10762.90 | 10713.66 | 0.00 | ORB-long ORB[10652.00,10749.60] vol=1.9x ATR=35.54 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-24 10:55:00 | 10816.20 | 10732.12 | 0.00 | T1 1.5R @ 10816.20 |
| Stop hit — per-position SL triggered | 2025-03-24 11:10:00 | 10762.90 | 10735.20 | 0.00 | SL hit |

### Cycle 44 — BUY (started 2025-03-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-28 11:15:00 | 11124.45 | 11069.10 | 0.00 | ORB-long ORB[10949.95,11099.80] vol=1.7x ATR=42.48 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-28 13:35:00 | 11188.16 | 11111.10 | 0.00 | T1 1.5R @ 11188.16 |
| Stop hit — per-position SL triggered | 2025-03-28 13:50:00 | 11124.45 | 11116.05 | 0.00 | SL hit |

### Cycle 45 — BUY (started 2025-04-15 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-15 09:40:00 | 11662.00 | 11595.93 | 0.00 | ORB-long ORB[11469.00,11610.00] vol=2.8x ATR=45.88 |
| Stop hit — per-position SL triggered | 2025-04-15 09:45:00 | 11616.12 | 11600.77 | 0.00 | SL hit |

### Cycle 46 — BUY (started 2025-04-16 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-16 09:55:00 | 12128.00 | 12020.98 | 0.00 | ORB-long ORB[11934.00,12045.00] vol=1.7x ATR=40.38 |
| Stop hit — per-position SL triggered | 2025-04-16 10:20:00 | 12087.62 | 12056.02 | 0.00 | SL hit |

### Cycle 47 — BUY (started 2025-04-22 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-22 09:30:00 | 12675.00 | 12601.60 | 0.00 | ORB-long ORB[12529.00,12640.00] vol=1.5x ATR=40.52 |
| Stop hit — per-position SL triggered | 2025-04-22 10:30:00 | 12634.48 | 12645.17 | 0.00 | SL hit |

### Cycle 48 — SELL (started 2025-04-23 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-23 09:30:00 | 12953.00 | 13045.39 | 0.00 | ORB-short ORB[12972.00,13155.00] vol=2.3x ATR=59.69 |
| Stop hit — per-position SL triggered | 2025-04-23 09:40:00 | 13012.69 | 13038.36 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2024-05-15 10:05:00 | 8520.50 | 2024-05-15 10:15:00 | 8478.05 | PARTIAL | 0.50 | 0.50% |
| SELL | retest1 | 2024-05-15 10:05:00 | 8520.50 | 2024-05-15 15:20:00 | 8329.75 | TARGET_HIT | 0.50 | 2.24% |
| SELL | retest1 | 2024-06-21 10:45:00 | 9771.30 | 2024-06-21 11:50:00 | 9798.12 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2024-06-28 09:35:00 | 9810.10 | 2024-06-28 09:50:00 | 9846.54 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest1 | 2024-07-09 10:15:00 | 12034.40 | 2024-07-09 10:25:00 | 11940.27 | PARTIAL | 0.50 | 0.78% |
| SELL | retest1 | 2024-07-09 10:15:00 | 12034.40 | 2024-07-09 10:30:00 | 12034.40 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-07-25 10:25:00 | 10376.80 | 2024-07-25 10:30:00 | 10299.03 | PARTIAL | 0.50 | 0.75% |
| SELL | retest1 | 2024-07-25 10:25:00 | 10376.80 | 2024-07-25 15:20:00 | 10376.90 | STOP_HIT | 0.50 | -0.00% |
| BUY | retest1 | 2024-08-06 09:35:00 | 10630.20 | 2024-08-06 09:55:00 | 10704.09 | PARTIAL | 0.50 | 0.70% |
| BUY | retest1 | 2024-08-06 09:35:00 | 10630.20 | 2024-08-06 10:40:00 | 10630.20 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-08-14 11:15:00 | 10037.85 | 2024-08-14 11:50:00 | 10067.90 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2024-08-16 11:00:00 | 10150.15 | 2024-08-16 11:20:00 | 10174.44 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2024-08-20 11:05:00 | 10313.35 | 2024-08-20 11:30:00 | 10270.21 | PARTIAL | 0.50 | 0.42% |
| SELL | retest1 | 2024-08-20 11:05:00 | 10313.35 | 2024-08-20 11:40:00 | 10313.35 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-08-29 09:30:00 | 10325.00 | 2024-08-29 09:35:00 | 10377.34 | STOP_HIT | 1.00 | -0.51% |
| BUY | retest1 | 2024-08-30 09:40:00 | 10498.75 | 2024-08-30 10:25:00 | 10582.70 | PARTIAL | 0.50 | 0.80% |
| BUY | retest1 | 2024-08-30 09:40:00 | 10498.75 | 2024-08-30 15:10:00 | 10778.95 | TARGET_HIT | 0.50 | 2.67% |
| SELL | retest1 | 2024-09-05 10:55:00 | 10998.95 | 2024-09-05 11:10:00 | 11040.84 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2024-09-11 10:05:00 | 11025.00 | 2024-09-11 11:50:00 | 10972.27 | STOP_HIT | 1.00 | -0.48% |
| BUY | retest1 | 2024-09-12 09:35:00 | 11298.85 | 2024-09-12 09:40:00 | 11239.46 | STOP_HIT | 1.00 | -0.53% |
| SELL | retest1 | 2024-09-16 11:10:00 | 10868.35 | 2024-09-16 11:15:00 | 10901.45 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2024-09-30 10:55:00 | 11545.05 | 2024-09-30 11:10:00 | 11462.97 | PARTIAL | 0.50 | 0.71% |
| SELL | retest1 | 2024-09-30 10:55:00 | 11545.05 | 2024-09-30 13:15:00 | 11545.05 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-10-07 10:05:00 | 10600.00 | 2024-10-07 10:20:00 | 10515.44 | PARTIAL | 0.50 | 0.80% |
| SELL | retest1 | 2024-10-07 10:05:00 | 10600.00 | 2024-10-07 10:25:00 | 10600.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-10-16 10:35:00 | 11499.35 | 2024-10-16 10:45:00 | 11452.05 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest1 | 2024-11-11 11:00:00 | 10290.40 | 2024-11-11 11:15:00 | 10347.33 | PARTIAL | 0.50 | 0.55% |
| BUY | retest1 | 2024-11-11 11:00:00 | 10290.40 | 2024-11-11 11:40:00 | 10290.40 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-12-12 09:50:00 | 10752.15 | 2024-12-12 14:35:00 | 10704.86 | PARTIAL | 0.50 | 0.44% |
| SELL | retest1 | 2024-12-12 09:50:00 | 10752.15 | 2024-12-12 15:20:00 | 10647.20 | TARGET_HIT | 0.50 | 0.98% |
| SELL | retest1 | 2024-12-17 09:40:00 | 10470.00 | 2024-12-17 10:05:00 | 10509.00 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2025-01-01 10:50:00 | 9910.55 | 2025-01-01 11:10:00 | 9878.58 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2025-01-02 11:05:00 | 9755.00 | 2025-01-02 11:20:00 | 9717.16 | PARTIAL | 0.50 | 0.39% |
| SELL | retest1 | 2025-01-02 11:05:00 | 9755.00 | 2025-01-02 14:55:00 | 9725.00 | TARGET_HIT | 0.50 | 0.31% |
| BUY | retest1 | 2025-01-03 09:30:00 | 9797.55 | 2025-01-03 09:55:00 | 9836.38 | PARTIAL | 0.50 | 0.40% |
| BUY | retest1 | 2025-01-03 09:30:00 | 9797.55 | 2025-01-03 10:30:00 | 9797.55 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-01-06 09:30:00 | 9620.00 | 2025-01-06 09:35:00 | 9649.86 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2025-01-08 09:30:00 | 9700.00 | 2025-01-08 09:35:00 | 9737.94 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest1 | 2025-01-10 09:35:00 | 9429.30 | 2025-01-10 09:40:00 | 9471.63 | STOP_HIT | 1.00 | -0.45% |
| BUY | retest1 | 2025-01-15 10:50:00 | 9465.00 | 2025-01-15 11:30:00 | 9527.08 | PARTIAL | 0.50 | 0.66% |
| BUY | retest1 | 2025-01-15 10:50:00 | 9465.00 | 2025-01-15 15:20:00 | 9539.55 | TARGET_HIT | 0.50 | 0.79% |
| SELL | retest1 | 2025-01-16 11:05:00 | 9529.95 | 2025-01-16 11:10:00 | 9555.81 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2025-01-23 10:15:00 | 9509.50 | 2025-01-23 13:35:00 | 9578.80 | PARTIAL | 0.50 | 0.73% |
| BUY | retest1 | 2025-01-23 10:15:00 | 9509.50 | 2025-01-23 14:50:00 | 9509.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-01-29 11:10:00 | 9450.30 | 2025-01-29 11:20:00 | 9418.32 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2025-02-01 10:50:00 | 10437.75 | 2025-02-01 11:00:00 | 10407.52 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2025-02-05 10:15:00 | 9873.75 | 2025-02-05 10:30:00 | 9936.18 | PARTIAL | 0.50 | 0.63% |
| BUY | retest1 | 2025-02-05 10:15:00 | 9873.75 | 2025-02-05 14:25:00 | 9960.10 | TARGET_HIT | 0.50 | 0.87% |
| SELL | retest1 | 2025-02-10 11:15:00 | 9104.50 | 2025-02-10 11:20:00 | 9130.34 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2025-02-11 10:20:00 | 8930.40 | 2025-02-11 10:30:00 | 8877.13 | PARTIAL | 0.50 | 0.60% |
| SELL | retest1 | 2025-02-11 10:20:00 | 8930.40 | 2025-02-11 10:35:00 | 8930.40 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-02-13 10:30:00 | 9054.80 | 2025-02-13 11:25:00 | 9019.20 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest1 | 2025-02-24 09:55:00 | 8824.55 | 2025-02-24 10:40:00 | 8787.41 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest1 | 2025-03-05 09:40:00 | 9225.00 | 2025-03-05 10:20:00 | 9189.44 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest1 | 2025-03-12 10:05:00 | 9764.90 | 2025-03-12 10:40:00 | 9802.80 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest1 | 2025-03-13 09:55:00 | 9880.00 | 2025-03-13 10:35:00 | 9925.31 | PARTIAL | 0.50 | 0.46% |
| BUY | retest1 | 2025-03-13 09:55:00 | 9880.00 | 2025-03-13 14:05:00 | 10124.25 | TARGET_HIT | 0.50 | 2.47% |
| BUY | retest1 | 2025-03-19 09:30:00 | 10187.80 | 2025-03-19 09:35:00 | 10152.67 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2025-03-21 09:40:00 | 10549.75 | 2025-03-21 09:50:00 | 10515.28 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2025-03-24 10:40:00 | 10762.90 | 2025-03-24 10:55:00 | 10816.20 | PARTIAL | 0.50 | 0.50% |
| BUY | retest1 | 2025-03-24 10:40:00 | 10762.90 | 2025-03-24 11:10:00 | 10762.90 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-03-28 11:15:00 | 11124.45 | 2025-03-28 13:35:00 | 11188.16 | PARTIAL | 0.50 | 0.57% |
| BUY | retest1 | 2025-03-28 11:15:00 | 11124.45 | 2025-03-28 13:50:00 | 11124.45 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-04-15 09:40:00 | 11662.00 | 2025-04-15 09:45:00 | 11616.12 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest1 | 2025-04-16 09:55:00 | 12128.00 | 2025-04-16 10:20:00 | 12087.62 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2025-04-22 09:30:00 | 12675.00 | 2025-04-22 10:30:00 | 12634.48 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2025-04-23 09:30:00 | 12953.00 | 2025-04-23 09:40:00 | 13012.69 | STOP_HIT | 1.00 | -0.46% |
