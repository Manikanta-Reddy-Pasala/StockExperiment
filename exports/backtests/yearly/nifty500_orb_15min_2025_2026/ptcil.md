# PTC Industries Ltd. (PTCIL)

## Backtest Summary

- **Window:** 2025-05-12 09:15:00 → 2026-05-08 15:25:00 (18463 bars)
- **Last close:** 16790.00
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
| PARTIAL | 34 |
| TARGET_HIT | 12 |
| STOP_HIT | 62 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 108 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 46 / 62
- **Target hits / Stop hits / Partials:** 12 / 62 / 34
- **Avg / median % per leg:** 0.19% / 0.00%
- **Sum % (uncompounded):** 20.42%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 44 | 14 | 31.8% | 3 | 30 | 11 | 0.03% | 1.5% |
| BUY @ 2nd Alert (retest1) | 44 | 14 | 31.8% | 3 | 30 | 11 | 0.03% | 1.5% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 64 | 32 | 50.0% | 9 | 32 | 23 | 0.30% | 18.9% |
| SELL @ 2nd Alert (retest1) | 64 | 32 | 50.0% | 9 | 32 | 23 | 0.30% | 18.9% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 108 | 46 | 42.6% | 12 | 62 | 34 | 0.19% | 20.4% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-13 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-13 10:00:00 | 13048.00 | 12923.29 | 0.00 | ORB-long ORB[12800.00,12991.00] vol=2.9x ATR=84.69 |
| Stop hit — per-position SL triggered | 2025-05-13 10:55:00 | 12963.31 | 12981.74 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2025-05-29 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-29 10:00:00 | 15455.00 | 15592.11 | 0.00 | ORB-short ORB[15566.00,15750.00] vol=1.9x ATR=62.47 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-29 11:05:00 | 15361.30 | 15534.24 | 0.00 | T1 1.5R @ 15361.30 |
| Target hit | 2025-05-29 15:20:00 | 15255.00 | 15410.93 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 3 — BUY (started 2025-05-30 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-30 10:00:00 | 15623.00 | 15370.06 | 0.00 | ORB-long ORB[15284.00,15495.00] vol=1.8x ATR=81.93 |
| Stop hit — per-position SL triggered | 2025-05-30 10:05:00 | 15541.07 | 15384.99 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2025-06-04 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-04 09:35:00 | 15457.00 | 15355.76 | 0.00 | ORB-long ORB[15206.00,15343.00] vol=4.2x ATR=66.79 |
| Stop hit — per-position SL triggered | 2025-06-04 09:40:00 | 15390.21 | 15366.14 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2025-06-06 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-06 09:35:00 | 14753.00 | 14839.86 | 0.00 | ORB-short ORB[14791.00,14933.00] vol=2.2x ATR=48.21 |
| Stop hit — per-position SL triggered | 2025-06-06 09:40:00 | 14801.21 | 14824.06 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2025-06-09 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-09 10:45:00 | 14354.00 | 14464.26 | 0.00 | ORB-short ORB[14440.00,14624.00] vol=2.1x ATR=49.96 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-09 13:00:00 | 14279.07 | 14419.69 | 0.00 | T1 1.5R @ 14279.07 |
| Stop hit — per-position SL triggered | 2025-06-09 14:45:00 | 14354.00 | 14390.92 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2025-06-19 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-19 09:50:00 | 14957.00 | 14831.78 | 0.00 | ORB-long ORB[14658.00,14773.00] vol=3.0x ATR=51.38 |
| Stop hit — per-position SL triggered | 2025-06-19 09:55:00 | 14905.62 | 14836.59 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2025-06-20 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-20 11:10:00 | 14200.00 | 14250.25 | 0.00 | ORB-short ORB[14235.00,14370.00] vol=8.0x ATR=53.28 |
| Stop hit — per-position SL triggered | 2025-06-20 11:15:00 | 14253.28 | 14269.27 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2025-06-27 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-27 10:50:00 | 15072.00 | 14913.67 | 0.00 | ORB-long ORB[14764.00,14903.00] vol=6.9x ATR=59.62 |
| Stop hit — per-position SL triggered | 2025-06-27 10:55:00 | 15012.38 | 14934.07 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2025-06-30 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-30 10:05:00 | 15288.00 | 15179.85 | 0.00 | ORB-long ORB[15080.00,15280.00] vol=1.7x ATR=80.20 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-30 10:10:00 | 15408.31 | 15222.05 | 0.00 | T1 1.5R @ 15408.31 |
| Stop hit — per-position SL triggered | 2025-06-30 11:20:00 | 15288.00 | 15269.41 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2025-07-01 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-01 09:45:00 | 15587.00 | 15683.01 | 0.00 | ORB-short ORB[15607.00,15818.00] vol=2.9x ATR=81.73 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-01 10:15:00 | 15464.41 | 15644.10 | 0.00 | T1 1.5R @ 15464.41 |
| Target hit | 2025-07-01 15:20:00 | 14900.00 | 15248.55 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 12 — BUY (started 2025-07-03 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-03 09:35:00 | 15188.00 | 15058.60 | 0.00 | ORB-long ORB[14955.00,15078.00] vol=1.6x ATR=61.05 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-03 09:55:00 | 15279.58 | 15138.04 | 0.00 | T1 1.5R @ 15279.58 |
| Stop hit — per-position SL triggered | 2025-07-03 10:30:00 | 15188.00 | 15175.73 | 0.00 | SL hit |

### Cycle 13 — SELL (started 2025-07-10 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-10 10:45:00 | 14472.00 | 14560.47 | 0.00 | ORB-short ORB[14561.00,14650.00] vol=1.9x ATR=32.75 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-10 11:20:00 | 14422.87 | 14536.14 | 0.00 | T1 1.5R @ 14422.87 |
| Target hit | 2025-07-10 15:20:00 | 14270.00 | 14417.77 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 14 — SELL (started 2025-07-11 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-11 11:00:00 | 14000.00 | 14137.86 | 0.00 | ORB-short ORB[14207.00,14332.00] vol=2.5x ATR=37.51 |
| Stop hit — per-position SL triggered | 2025-07-11 11:05:00 | 14037.51 | 14135.10 | 0.00 | SL hit |

### Cycle 15 — SELL (started 2025-07-17 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-17 10:55:00 | 14519.00 | 14588.88 | 0.00 | ORB-short ORB[14549.00,14639.00] vol=2.3x ATR=37.91 |
| Stop hit — per-position SL triggered | 2025-07-17 11:30:00 | 14556.91 | 14585.49 | 0.00 | SL hit |

### Cycle 16 — SELL (started 2025-07-18 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-18 10:10:00 | 14415.00 | 14497.39 | 0.00 | ORB-short ORB[14456.00,14571.00] vol=1.9x ATR=35.91 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-18 10:15:00 | 14361.13 | 14462.32 | 0.00 | T1 1.5R @ 14361.13 |
| Stop hit — per-position SL triggered | 2025-07-18 10:50:00 | 14415.00 | 14438.04 | 0.00 | SL hit |

### Cycle 17 — SELL (started 2025-07-23 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-23 10:45:00 | 14459.00 | 14467.66 | 0.00 | ORB-short ORB[14460.00,14547.00] vol=6.6x ATR=29.55 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-23 10:50:00 | 14414.68 | 14441.49 | 0.00 | T1 1.5R @ 14414.68 |
| Target hit | 2025-07-23 14:50:00 | 14317.00 | 14316.79 | 0.00 | Trail-exit close>VWAP |

### Cycle 18 — SELL (started 2025-07-25 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-25 10:50:00 | 14410.00 | 14445.08 | 0.00 | ORB-short ORB[14414.00,14570.00] vol=1.8x ATR=33.15 |
| Stop hit — per-position SL triggered | 2025-07-25 11:30:00 | 14443.15 | 14426.17 | 0.00 | SL hit |

### Cycle 19 — BUY (started 2025-07-28 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-28 09:30:00 | 14703.00 | 14625.21 | 0.00 | ORB-long ORB[14451.00,14600.00] vol=4.4x ATR=76.94 |
| Stop hit — per-position SL triggered | 2025-07-28 09:35:00 | 14626.06 | 14632.64 | 0.00 | SL hit |

### Cycle 20 — SELL (started 2025-08-06 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-06 10:00:00 | 14756.00 | 14930.92 | 0.00 | ORB-short ORB[14951.00,15080.00] vol=1.6x ATR=53.50 |
| Stop hit — per-position SL triggered | 2025-08-06 10:20:00 | 14809.50 | 14907.91 | 0.00 | SL hit |

### Cycle 21 — SELL (started 2025-08-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-07 11:15:00 | 14820.00 | 14892.79 | 0.00 | ORB-short ORB[14874.00,15025.00] vol=2.7x ATR=28.92 |
| Stop hit — per-position SL triggered | 2025-08-07 11:50:00 | 14848.92 | 14890.12 | 0.00 | SL hit |

### Cycle 22 — SELL (started 2025-08-21 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-21 09:30:00 | 13550.00 | 13621.18 | 0.00 | ORB-short ORB[13568.00,13747.00] vol=1.8x ATR=53.53 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-21 11:25:00 | 13469.70 | 13559.20 | 0.00 | T1 1.5R @ 13469.70 |
| Stop hit — per-position SL triggered | 2025-08-21 13:05:00 | 13550.00 | 13510.73 | 0.00 | SL hit |

### Cycle 23 — SELL (started 2025-08-29 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-29 11:10:00 | 13844.00 | 13858.39 | 0.00 | ORB-short ORB[13856.00,13939.00] vol=1.6x ATR=30.98 |
| Stop hit — per-position SL triggered | 2025-08-29 12:05:00 | 13874.98 | 13855.51 | 0.00 | SL hit |

### Cycle 24 — BUY (started 2025-09-02 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-02 10:35:00 | 13940.00 | 13812.32 | 0.00 | ORB-long ORB[13676.00,13840.00] vol=2.3x ATR=30.77 |
| Stop hit — per-position SL triggered | 2025-09-02 10:55:00 | 13909.23 | 13830.44 | 0.00 | SL hit |

### Cycle 25 — SELL (started 2025-09-03 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-03 10:50:00 | 13809.00 | 13845.46 | 0.00 | ORB-short ORB[13831.00,13931.00] vol=4.7x ATR=27.63 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-03 11:40:00 | 13767.56 | 13826.77 | 0.00 | T1 1.5R @ 13767.56 |
| Stop hit — per-position SL triggered | 2025-09-03 14:35:00 | 13809.00 | 13793.43 | 0.00 | SL hit |

### Cycle 26 — BUY (started 2025-09-23 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-23 10:00:00 | 15347.00 | 15244.40 | 0.00 | ORB-long ORB[15156.00,15300.00] vol=1.8x ATR=53.80 |
| Stop hit — per-position SL triggered | 2025-09-23 10:20:00 | 15293.20 | 15261.52 | 0.00 | SL hit |

### Cycle 27 — BUY (started 2025-09-26 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-26 10:45:00 | 15550.00 | 15470.16 | 0.00 | ORB-long ORB[15333.00,15529.00] vol=2.3x ATR=52.26 |
| Stop hit — per-position SL triggered | 2025-09-26 10:55:00 | 15497.74 | 15471.02 | 0.00 | SL hit |

### Cycle 28 — SELL (started 2025-10-13 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-13 10:25:00 | 16029.00 | 16117.87 | 0.00 | ORB-short ORB[16076.00,16293.00] vol=1.6x ATR=70.29 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-13 11:55:00 | 15923.57 | 16054.93 | 0.00 | T1 1.5R @ 15923.57 |
| Stop hit — per-position SL triggered | 2025-10-13 14:25:00 | 16029.00 | 16022.58 | 0.00 | SL hit |

### Cycle 29 — SELL (started 2025-10-14 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-14 11:00:00 | 16290.00 | 16405.04 | 0.00 | ORB-short ORB[16424.00,16595.00] vol=2.8x ATR=51.62 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-14 12:50:00 | 16212.57 | 16381.30 | 0.00 | T1 1.5R @ 16212.57 |
| Stop hit — per-position SL triggered | 2025-10-14 13:35:00 | 16290.00 | 16374.14 | 0.00 | SL hit |

### Cycle 30 — BUY (started 2025-10-17 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-17 09:40:00 | 16729.00 | 16629.34 | 0.00 | ORB-long ORB[16500.00,16629.00] vol=1.5x ATR=49.07 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-17 09:55:00 | 16802.60 | 16702.88 | 0.00 | T1 1.5R @ 16802.60 |
| Stop hit — per-position SL triggered | 2025-10-17 10:25:00 | 16729.00 | 16742.95 | 0.00 | SL hit |

### Cycle 31 — BUY (started 2025-10-24 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-24 09:55:00 | 17085.00 | 16923.66 | 0.00 | ORB-long ORB[16821.00,16970.00] vol=1.7x ATR=62.06 |
| Stop hit — per-position SL triggered | 2025-10-24 10:05:00 | 17022.94 | 16942.68 | 0.00 | SL hit |

### Cycle 32 — SELL (started 2025-10-28 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-28 10:50:00 | 17345.00 | 17410.66 | 0.00 | ORB-short ORB[17362.00,17570.00] vol=1.9x ATR=50.89 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-28 11:40:00 | 17268.66 | 17387.89 | 0.00 | T1 1.5R @ 17268.66 |
| Target hit | 2025-10-28 15:20:00 | 17124.00 | 17294.56 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 33 — BUY (started 2025-10-30 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-30 11:00:00 | 17193.00 | 17083.34 | 0.00 | ORB-long ORB[16977.00,17162.00] vol=6.1x ATR=42.97 |
| Stop hit — per-position SL triggered | 2025-10-30 11:10:00 | 17150.03 | 17093.47 | 0.00 | SL hit |

### Cycle 34 — BUY (started 2025-10-31 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-31 09:35:00 | 17169.00 | 17070.97 | 0.00 | ORB-long ORB[16911.00,17121.00] vol=3.2x ATR=58.41 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-31 09:55:00 | 17256.61 | 17133.77 | 0.00 | T1 1.5R @ 17256.61 |
| Stop hit — per-position SL triggered | 2025-10-31 10:20:00 | 17169.00 | 17137.90 | 0.00 | SL hit |

### Cycle 35 — BUY (started 2025-11-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-07 11:15:00 | 17544.00 | 17400.56 | 0.00 | ORB-long ORB[17322.00,17500.00] vol=3.3x ATR=47.28 |
| Stop hit — per-position SL triggered | 2025-11-07 11:25:00 | 17496.72 | 17409.82 | 0.00 | SL hit |

### Cycle 36 — SELL (started 2025-11-13 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-13 10:45:00 | 17499.00 | 17570.26 | 0.00 | ORB-short ORB[17526.00,17669.00] vol=1.9x ATR=37.82 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-13 10:55:00 | 17442.27 | 17542.34 | 0.00 | T1 1.5R @ 17442.27 |
| Target hit | 2025-11-13 14:55:00 | 17448.00 | 17422.56 | 0.00 | Trail-exit close>VWAP |

### Cycle 37 — BUY (started 2025-11-17 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-17 10:55:00 | 17181.00 | 17166.26 | 0.00 | ORB-long ORB[16950.00,17149.00] vol=1.5x ATR=42.20 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-17 11:35:00 | 17244.30 | 17173.59 | 0.00 | T1 1.5R @ 17244.30 |
| Stop hit — per-position SL triggered | 2025-11-17 14:00:00 | 17181.00 | 17187.68 | 0.00 | SL hit |

### Cycle 38 — SELL (started 2025-11-18 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-18 10:20:00 | 17197.00 | 17351.49 | 0.00 | ORB-short ORB[17340.00,17495.00] vol=2.0x ATR=55.00 |
| Stop hit — per-position SL triggered | 2025-11-18 11:45:00 | 17252.00 | 17318.98 | 0.00 | SL hit |

### Cycle 39 — SELL (started 2025-11-19 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-19 10:50:00 | 17145.00 | 17233.02 | 0.00 | ORB-short ORB[17200.00,17320.00] vol=1.9x ATR=44.98 |
| Stop hit — per-position SL triggered | 2025-11-19 11:05:00 | 17189.98 | 17231.90 | 0.00 | SL hit |

### Cycle 40 — SELL (started 2025-11-21 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-21 10:25:00 | 17017.00 | 17119.63 | 0.00 | ORB-short ORB[17080.00,17167.00] vol=2.4x ATR=45.70 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-21 11:50:00 | 16948.45 | 17066.53 | 0.00 | T1 1.5R @ 16948.45 |
| Stop hit — per-position SL triggered | 2025-11-21 12:15:00 | 17017.00 | 17057.29 | 0.00 | SL hit |

### Cycle 41 — SELL (started 2025-11-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-27 10:15:00 | 18017.00 | 18111.27 | 0.00 | ORB-short ORB[18062.00,18261.00] vol=1.6x ATR=57.25 |
| Stop hit — per-position SL triggered | 2025-11-27 14:05:00 | 18074.25 | 18073.67 | 0.00 | SL hit |

### Cycle 42 — BUY (started 2025-11-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-28 10:15:00 | 18104.00 | 18049.14 | 0.00 | ORB-long ORB[17952.00,18100.00] vol=6.4x ATR=42.12 |
| Stop hit — per-position SL triggered | 2025-11-28 10:20:00 | 18061.88 | 18049.99 | 0.00 | SL hit |

### Cycle 43 — SELL (started 2025-12-03 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-03 09:35:00 | 18201.00 | 18300.44 | 0.00 | ORB-short ORB[18299.00,18426.00] vol=1.7x ATR=41.64 |
| Stop hit — per-position SL triggered | 2025-12-03 09:45:00 | 18242.64 | 18290.84 | 0.00 | SL hit |

### Cycle 44 — BUY (started 2025-12-05 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-05 10:10:00 | 18434.00 | 18356.56 | 0.00 | ORB-long ORB[18180.00,18399.00] vol=3.4x ATR=62.51 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-05 11:15:00 | 18527.76 | 18395.04 | 0.00 | T1 1.5R @ 18527.76 |
| Target hit | 2025-12-05 15:20:00 | 18892.00 | 18735.58 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 45 — BUY (started 2025-12-12 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-12 10:30:00 | 18538.00 | 18465.92 | 0.00 | ORB-long ORB[18227.00,18488.00] vol=1.6x ATR=52.11 |
| Stop hit — per-position SL triggered | 2025-12-12 11:40:00 | 18485.89 | 18484.80 | 0.00 | SL hit |

### Cycle 46 — BUY (started 2025-12-19 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-19 09:30:00 | 17350.00 | 17102.84 | 0.00 | ORB-long ORB[16808.00,17049.00] vol=1.6x ATR=117.25 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-19 09:40:00 | 17525.88 | 17281.72 | 0.00 | T1 1.5R @ 17525.88 |
| Target hit | 2025-12-19 10:10:00 | 17415.00 | 17460.01 | 0.00 | Trail-exit close<VWAP |

### Cycle 47 — SELL (started 2025-12-26 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-26 11:10:00 | 18082.00 | 18117.83 | 0.00 | ORB-short ORB[18170.00,18301.00] vol=6.2x ATR=30.32 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-26 12:20:00 | 18036.53 | 18105.71 | 0.00 | T1 1.5R @ 18036.53 |
| Stop hit — per-position SL triggered | 2025-12-26 14:50:00 | 18082.00 | 18117.13 | 0.00 | SL hit |

### Cycle 48 — SELL (started 2025-12-30 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-30 10:50:00 | 18655.00 | 18875.85 | 0.00 | ORB-short ORB[18925.00,19176.00] vol=4.8x ATR=65.01 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-30 11:00:00 | 18557.49 | 18844.91 | 0.00 | T1 1.5R @ 18557.49 |
| Stop hit — per-position SL triggered | 2025-12-30 15:00:00 | 18655.00 | 18358.62 | 0.00 | SL hit |

### Cycle 49 — SELL (started 2026-01-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-05 11:15:00 | 18046.00 | 18095.03 | 0.00 | ORB-short ORB[18060.00,18172.00] vol=2.2x ATR=46.22 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-05 11:40:00 | 17976.66 | 18084.93 | 0.00 | T1 1.5R @ 17976.66 |
| Stop hit — per-position SL triggered | 2026-01-05 12:30:00 | 18046.00 | 18047.49 | 0.00 | SL hit |

### Cycle 50 — SELL (started 2026-01-07 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-07 09:40:00 | 17705.00 | 17783.91 | 0.00 | ORB-short ORB[17740.00,17899.00] vol=1.7x ATR=46.43 |
| Stop hit — per-position SL triggered | 2026-01-07 09:55:00 | 17751.43 | 17752.14 | 0.00 | SL hit |

### Cycle 51 — BUY (started 2026-01-08 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-08 09:40:00 | 17954.00 | 17870.08 | 0.00 | ORB-long ORB[17696.00,17880.00] vol=1.8x ATR=61.32 |
| Stop hit — per-position SL triggered | 2026-01-08 10:00:00 | 17892.68 | 17881.62 | 0.00 | SL hit |

### Cycle 52 — BUY (started 2026-01-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-09 11:15:00 | 17662.00 | 17560.12 | 0.00 | ORB-long ORB[17428.00,17567.00] vol=1.6x ATR=44.70 |
| Stop hit — per-position SL triggered | 2026-01-09 11:45:00 | 17617.30 | 17568.95 | 0.00 | SL hit |

### Cycle 53 — BUY (started 2026-01-13 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-13 10:35:00 | 18236.00 | 18127.97 | 0.00 | ORB-long ORB[17978.00,18200.00] vol=1.7x ATR=56.43 |
| Stop hit — per-position SL triggered | 2026-01-13 10:45:00 | 18179.57 | 18131.57 | 0.00 | SL hit |

### Cycle 54 — SELL (started 2026-01-14 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-14 10:05:00 | 17894.00 | 17967.63 | 0.00 | ORB-short ORB[17900.00,18044.00] vol=1.7x ATR=48.95 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-14 10:45:00 | 17820.57 | 17944.10 | 0.00 | T1 1.5R @ 17820.57 |
| Stop hit — per-position SL triggered | 2026-01-14 13:10:00 | 17894.00 | 17895.03 | 0.00 | SL hit |

### Cycle 55 — SELL (started 2026-01-22 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-22 11:10:00 | 17729.00 | 17875.95 | 0.00 | ORB-short ORB[17850.00,18048.00] vol=1.9x ATR=38.35 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-22 11:25:00 | 17671.48 | 17838.51 | 0.00 | T1 1.5R @ 17671.48 |
| Stop hit — per-position SL triggered | 2026-01-22 12:00:00 | 17729.00 | 17824.23 | 0.00 | SL hit |

### Cycle 56 — SELL (started 2026-01-23 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-23 11:05:00 | 17699.00 | 17772.89 | 0.00 | ORB-short ORB[17750.00,17950.00] vol=1.6x ATR=40.21 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-23 11:10:00 | 17638.69 | 17698.44 | 0.00 | T1 1.5R @ 17638.69 |
| Target hit | 2026-01-23 15:20:00 | 17452.00 | 17553.75 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 57 — BUY (started 2026-01-30 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-30 10:45:00 | 18474.00 | 18239.10 | 0.00 | ORB-long ORB[18008.00,18200.00] vol=2.7x ATR=65.44 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-30 11:00:00 | 18572.16 | 18291.38 | 0.00 | T1 1.5R @ 18572.16 |
| Stop hit — per-position SL triggered | 2026-01-30 11:15:00 | 18474.00 | 18316.96 | 0.00 | SL hit |

### Cycle 58 — SELL (started 2026-02-04 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-04 10:10:00 | 17801.00 | 17869.67 | 0.00 | ORB-short ORB[17806.00,17969.00] vol=2.1x ATR=49.39 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-04 10:35:00 | 17726.92 | 17840.53 | 0.00 | T1 1.5R @ 17726.92 |
| Stop hit — per-position SL triggered | 2026-02-04 11:40:00 | 17801.00 | 17801.78 | 0.00 | SL hit |

### Cycle 59 — BUY (started 2026-02-12 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-12 09:35:00 | 18425.00 | 18336.02 | 0.00 | ORB-long ORB[18234.00,18342.00] vol=2.6x ATR=51.20 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-12 10:05:00 | 18501.80 | 18387.17 | 0.00 | T1 1.5R @ 18501.80 |
| Stop hit — per-position SL triggered | 2026-02-12 10:25:00 | 18425.00 | 18391.08 | 0.00 | SL hit |

### Cycle 60 — BUY (started 2026-02-17 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-17 11:05:00 | 18326.00 | 18191.16 | 0.00 | ORB-long ORB[17881.00,18142.00] vol=1.7x ATR=56.31 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-17 11:20:00 | 18410.47 | 18211.93 | 0.00 | T1 1.5R @ 18410.47 |
| Target hit | 2026-02-17 12:20:00 | 18355.00 | 18385.11 | 0.00 | Trail-exit close<VWAP |

### Cycle 61 — SELL (started 2026-02-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-18 11:15:00 | 18230.00 | 18331.00 | 0.00 | ORB-short ORB[18287.00,18510.00] vol=1.6x ATR=38.12 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-18 11:30:00 | 18172.82 | 18292.47 | 0.00 | T1 1.5R @ 18172.82 |
| Stop hit — per-position SL triggered | 2026-02-18 12:25:00 | 18230.00 | 18234.67 | 0.00 | SL hit |

### Cycle 62 — BUY (started 2026-02-25 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-25 09:55:00 | 17995.00 | 17953.84 | 0.00 | ORB-long ORB[17838.00,17940.00] vol=2.0x ATR=46.78 |
| Stop hit — per-position SL triggered | 2026-02-25 10:20:00 | 17948.22 | 17961.42 | 0.00 | SL hit |

### Cycle 63 — BUY (started 2026-02-26 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-26 10:05:00 | 18232.00 | 18105.18 | 0.00 | ORB-long ORB[17950.00,18047.00] vol=1.6x ATR=46.32 |
| Stop hit — per-position SL triggered | 2026-02-26 10:10:00 | 18185.68 | 18114.65 | 0.00 | SL hit |

### Cycle 64 — SELL (started 2026-03-06 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-06 10:55:00 | 17797.00 | 18065.17 | 0.00 | ORB-short ORB[17990.00,18250.00] vol=1.7x ATR=63.41 |
| Stop hit — per-position SL triggered | 2026-03-06 14:40:00 | 17860.41 | 17964.68 | 0.00 | SL hit |

### Cycle 65 — SELL (started 2026-03-13 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-13 11:05:00 | 17529.00 | 17721.95 | 0.00 | ORB-short ORB[17755.00,17912.00] vol=1.9x ATR=49.60 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-13 11:50:00 | 17454.60 | 17671.33 | 0.00 | T1 1.5R @ 17454.60 |
| Target hit | 2026-03-13 15:20:00 | 17006.00 | 17195.74 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 66 — BUY (started 2026-03-17 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-17 10:30:00 | 17270.00 | 17097.19 | 0.00 | ORB-long ORB[16869.00,17098.00] vol=1.7x ATR=85.23 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-17 12:00:00 | 17397.85 | 17178.21 | 0.00 | T1 1.5R @ 17397.85 |
| Stop hit — per-position SL triggered | 2026-03-17 14:00:00 | 17270.00 | 17315.82 | 0.00 | SL hit |

### Cycle 67 — SELL (started 2026-03-20 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-20 11:05:00 | 17240.00 | 17354.69 | 0.00 | ORB-short ORB[17360.00,17561.00] vol=2.0x ATR=44.33 |
| Stop hit — per-position SL triggered | 2026-03-20 11:45:00 | 17284.33 | 17328.80 | 0.00 | SL hit |

### Cycle 68 — SELL (started 2026-04-17 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-17 10:40:00 | 15691.00 | 15877.06 | 0.00 | ORB-short ORB[15863.00,15963.00] vol=4.3x ATR=52.14 |
| Stop hit — per-position SL triggered | 2026-04-17 11:25:00 | 15743.14 | 15814.39 | 0.00 | SL hit |

### Cycle 69 — BUY (started 2026-04-21 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-21 09:45:00 | 16335.00 | 16252.66 | 0.00 | ORB-long ORB[16073.00,16226.00] vol=1.8x ATR=58.61 |
| Stop hit — per-position SL triggered | 2026-04-21 09:50:00 | 16276.39 | 16254.65 | 0.00 | SL hit |

### Cycle 70 — SELL (started 2026-04-22 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-22 10:30:00 | 15886.00 | 16000.90 | 0.00 | ORB-short ORB[15964.00,16126.00] vol=1.8x ATR=47.07 |
| Stop hit — per-position SL triggered | 2026-04-22 11:05:00 | 15933.07 | 15976.96 | 0.00 | SL hit |

### Cycle 71 — SELL (started 2026-04-24 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-24 10:00:00 | 15947.00 | 15996.89 | 0.00 | ORB-short ORB[15999.00,16195.00] vol=3.7x ATR=58.02 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-24 10:05:00 | 15859.96 | 15982.94 | 0.00 | T1 1.5R @ 15859.96 |
| Target hit | 2026-04-24 14:40:00 | 15940.00 | 15812.32 | 0.00 | Trail-exit close>VWAP |

### Cycle 72 — BUY (started 2026-05-04 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-04 09:50:00 | 16416.00 | 16263.86 | 0.00 | ORB-long ORB[16068.00,16307.00] vol=2.1x ATR=85.40 |
| Stop hit — per-position SL triggered | 2026-05-04 10:10:00 | 16330.60 | 16289.96 | 0.00 | SL hit |

### Cycle 73 — SELL (started 2026-05-07 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-07 10:10:00 | 17021.00 | 17088.79 | 0.00 | ORB-short ORB[17068.00,17241.00] vol=1.8x ATR=61.15 |
| Stop hit — per-position SL triggered | 2026-05-07 10:50:00 | 17082.15 | 17076.28 | 0.00 | SL hit |

### Cycle 74 — BUY (started 2026-05-08 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-08 09:40:00 | 17028.00 | 16924.35 | 0.00 | ORB-long ORB[16715.00,16950.00] vol=2.5x ATR=68.55 |
| Stop hit — per-position SL triggered | 2026-05-08 10:10:00 | 16959.45 | 16965.47 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-05-13 10:00:00 | 13048.00 | 2025-05-13 10:55:00 | 12963.31 | STOP_HIT | 1.00 | -0.65% |
| SELL | retest1 | 2025-05-29 10:00:00 | 15455.00 | 2025-05-29 11:05:00 | 15361.30 | PARTIAL | 0.50 | 0.61% |
| SELL | retest1 | 2025-05-29 10:00:00 | 15455.00 | 2025-05-29 15:20:00 | 15255.00 | TARGET_HIT | 0.50 | 1.29% |
| BUY | retest1 | 2025-05-30 10:00:00 | 15623.00 | 2025-05-30 10:05:00 | 15541.07 | STOP_HIT | 1.00 | -0.52% |
| BUY | retest1 | 2025-06-04 09:35:00 | 15457.00 | 2025-06-04 09:40:00 | 15390.21 | STOP_HIT | 1.00 | -0.43% |
| SELL | retest1 | 2025-06-06 09:35:00 | 14753.00 | 2025-06-06 09:40:00 | 14801.21 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2025-06-09 10:45:00 | 14354.00 | 2025-06-09 13:00:00 | 14279.07 | PARTIAL | 0.50 | 0.52% |
| SELL | retest1 | 2025-06-09 10:45:00 | 14354.00 | 2025-06-09 14:45:00 | 14354.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-06-19 09:50:00 | 14957.00 | 2025-06-19 09:55:00 | 14905.62 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2025-06-20 11:10:00 | 14200.00 | 2025-06-20 11:15:00 | 14253.28 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2025-06-27 10:50:00 | 15072.00 | 2025-06-27 10:55:00 | 15012.38 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest1 | 2025-06-30 10:05:00 | 15288.00 | 2025-06-30 10:10:00 | 15408.31 | PARTIAL | 0.50 | 0.79% |
| BUY | retest1 | 2025-06-30 10:05:00 | 15288.00 | 2025-06-30 11:20:00 | 15288.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-07-01 09:45:00 | 15587.00 | 2025-07-01 10:15:00 | 15464.41 | PARTIAL | 0.50 | 0.79% |
| SELL | retest1 | 2025-07-01 09:45:00 | 15587.00 | 2025-07-01 15:20:00 | 14900.00 | TARGET_HIT | 0.50 | 4.41% |
| BUY | retest1 | 2025-07-03 09:35:00 | 15188.00 | 2025-07-03 09:55:00 | 15279.58 | PARTIAL | 0.50 | 0.60% |
| BUY | retest1 | 2025-07-03 09:35:00 | 15188.00 | 2025-07-03 10:30:00 | 15188.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-07-10 10:45:00 | 14472.00 | 2025-07-10 11:20:00 | 14422.87 | PARTIAL | 0.50 | 0.34% |
| SELL | retest1 | 2025-07-10 10:45:00 | 14472.00 | 2025-07-10 15:20:00 | 14270.00 | TARGET_HIT | 0.50 | 1.40% |
| SELL | retest1 | 2025-07-11 11:00:00 | 14000.00 | 2025-07-11 11:05:00 | 14037.51 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2025-07-17 10:55:00 | 14519.00 | 2025-07-17 11:30:00 | 14556.91 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2025-07-18 10:10:00 | 14415.00 | 2025-07-18 10:15:00 | 14361.13 | PARTIAL | 0.50 | 0.37% |
| SELL | retest1 | 2025-07-18 10:10:00 | 14415.00 | 2025-07-18 10:50:00 | 14415.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-07-23 10:45:00 | 14459.00 | 2025-07-23 10:50:00 | 14414.68 | PARTIAL | 0.50 | 0.31% |
| SELL | retest1 | 2025-07-23 10:45:00 | 14459.00 | 2025-07-23 14:50:00 | 14317.00 | TARGET_HIT | 0.50 | 0.98% |
| SELL | retest1 | 2025-07-25 10:50:00 | 14410.00 | 2025-07-25 11:30:00 | 14443.15 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2025-07-28 09:30:00 | 14703.00 | 2025-07-28 09:35:00 | 14626.06 | STOP_HIT | 1.00 | -0.52% |
| SELL | retest1 | 2025-08-06 10:00:00 | 14756.00 | 2025-08-06 10:20:00 | 14809.50 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2025-08-07 11:15:00 | 14820.00 | 2025-08-07 11:50:00 | 14848.92 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2025-08-21 09:30:00 | 13550.00 | 2025-08-21 11:25:00 | 13469.70 | PARTIAL | 0.50 | 0.59% |
| SELL | retest1 | 2025-08-21 09:30:00 | 13550.00 | 2025-08-21 13:05:00 | 13550.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-08-29 11:10:00 | 13844.00 | 2025-08-29 12:05:00 | 13874.98 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2025-09-02 10:35:00 | 13940.00 | 2025-09-02 10:55:00 | 13909.23 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2025-09-03 10:50:00 | 13809.00 | 2025-09-03 11:40:00 | 13767.56 | PARTIAL | 0.50 | 0.30% |
| SELL | retest1 | 2025-09-03 10:50:00 | 13809.00 | 2025-09-03 14:35:00 | 13809.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-09-23 10:00:00 | 15347.00 | 2025-09-23 10:20:00 | 15293.20 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2025-09-26 10:45:00 | 15550.00 | 2025-09-26 10:55:00 | 15497.74 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2025-10-13 10:25:00 | 16029.00 | 2025-10-13 11:55:00 | 15923.57 | PARTIAL | 0.50 | 0.66% |
| SELL | retest1 | 2025-10-13 10:25:00 | 16029.00 | 2025-10-13 14:25:00 | 16029.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-10-14 11:00:00 | 16290.00 | 2025-10-14 12:50:00 | 16212.57 | PARTIAL | 0.50 | 0.48% |
| SELL | retest1 | 2025-10-14 11:00:00 | 16290.00 | 2025-10-14 13:35:00 | 16290.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-10-17 09:40:00 | 16729.00 | 2025-10-17 09:55:00 | 16802.60 | PARTIAL | 0.50 | 0.44% |
| BUY | retest1 | 2025-10-17 09:40:00 | 16729.00 | 2025-10-17 10:25:00 | 16729.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-10-24 09:55:00 | 17085.00 | 2025-10-24 10:05:00 | 17022.94 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2025-10-28 10:50:00 | 17345.00 | 2025-10-28 11:40:00 | 17268.66 | PARTIAL | 0.50 | 0.44% |
| SELL | retest1 | 2025-10-28 10:50:00 | 17345.00 | 2025-10-28 15:20:00 | 17124.00 | TARGET_HIT | 0.50 | 1.27% |
| BUY | retest1 | 2025-10-30 11:00:00 | 17193.00 | 2025-10-30 11:10:00 | 17150.03 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2025-10-31 09:35:00 | 17169.00 | 2025-10-31 09:55:00 | 17256.61 | PARTIAL | 0.50 | 0.51% |
| BUY | retest1 | 2025-10-31 09:35:00 | 17169.00 | 2025-10-31 10:20:00 | 17169.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-11-07 11:15:00 | 17544.00 | 2025-11-07 11:25:00 | 17496.72 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2025-11-13 10:45:00 | 17499.00 | 2025-11-13 10:55:00 | 17442.27 | PARTIAL | 0.50 | 0.32% |
| SELL | retest1 | 2025-11-13 10:45:00 | 17499.00 | 2025-11-13 14:55:00 | 17448.00 | TARGET_HIT | 0.50 | 0.29% |
| BUY | retest1 | 2025-11-17 10:55:00 | 17181.00 | 2025-11-17 11:35:00 | 17244.30 | PARTIAL | 0.50 | 0.37% |
| BUY | retest1 | 2025-11-17 10:55:00 | 17181.00 | 2025-11-17 14:00:00 | 17181.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-11-18 10:20:00 | 17197.00 | 2025-11-18 11:45:00 | 17252.00 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2025-11-19 10:50:00 | 17145.00 | 2025-11-19 11:05:00 | 17189.98 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2025-11-21 10:25:00 | 17017.00 | 2025-11-21 11:50:00 | 16948.45 | PARTIAL | 0.50 | 0.40% |
| SELL | retest1 | 2025-11-21 10:25:00 | 17017.00 | 2025-11-21 12:15:00 | 17017.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-11-27 10:15:00 | 18017.00 | 2025-11-27 14:05:00 | 18074.25 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2025-11-28 10:15:00 | 18104.00 | 2025-11-28 10:20:00 | 18061.88 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2025-12-03 09:35:00 | 18201.00 | 2025-12-03 09:45:00 | 18242.64 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2025-12-05 10:10:00 | 18434.00 | 2025-12-05 11:15:00 | 18527.76 | PARTIAL | 0.50 | 0.51% |
| BUY | retest1 | 2025-12-05 10:10:00 | 18434.00 | 2025-12-05 15:20:00 | 18892.00 | TARGET_HIT | 0.50 | 2.48% |
| BUY | retest1 | 2025-12-12 10:30:00 | 18538.00 | 2025-12-12 11:40:00 | 18485.89 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2025-12-19 09:30:00 | 17350.00 | 2025-12-19 09:40:00 | 17525.88 | PARTIAL | 0.50 | 1.01% |
| BUY | retest1 | 2025-12-19 09:30:00 | 17350.00 | 2025-12-19 10:10:00 | 17415.00 | TARGET_HIT | 0.50 | 0.37% |
| SELL | retest1 | 2025-12-26 11:10:00 | 18082.00 | 2025-12-26 12:20:00 | 18036.53 | PARTIAL | 0.50 | 0.25% |
| SELL | retest1 | 2025-12-26 11:10:00 | 18082.00 | 2025-12-26 14:50:00 | 18082.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-12-30 10:50:00 | 18655.00 | 2025-12-30 11:00:00 | 18557.49 | PARTIAL | 0.50 | 0.52% |
| SELL | retest1 | 2025-12-30 10:50:00 | 18655.00 | 2025-12-30 15:00:00 | 18655.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-01-05 11:15:00 | 18046.00 | 2026-01-05 11:40:00 | 17976.66 | PARTIAL | 0.50 | 0.38% |
| SELL | retest1 | 2026-01-05 11:15:00 | 18046.00 | 2026-01-05 12:30:00 | 18046.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-01-07 09:40:00 | 17705.00 | 2026-01-07 09:55:00 | 17751.43 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2026-01-08 09:40:00 | 17954.00 | 2026-01-08 10:00:00 | 17892.68 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2026-01-09 11:15:00 | 17662.00 | 2026-01-09 11:45:00 | 17617.30 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2026-01-13 10:35:00 | 18236.00 | 2026-01-13 10:45:00 | 18179.57 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2026-01-14 10:05:00 | 17894.00 | 2026-01-14 10:45:00 | 17820.57 | PARTIAL | 0.50 | 0.41% |
| SELL | retest1 | 2026-01-14 10:05:00 | 17894.00 | 2026-01-14 13:10:00 | 17894.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-01-22 11:10:00 | 17729.00 | 2026-01-22 11:25:00 | 17671.48 | PARTIAL | 0.50 | 0.32% |
| SELL | retest1 | 2026-01-22 11:10:00 | 17729.00 | 2026-01-22 12:00:00 | 17729.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-01-23 11:05:00 | 17699.00 | 2026-01-23 11:10:00 | 17638.69 | PARTIAL | 0.50 | 0.34% |
| SELL | retest1 | 2026-01-23 11:05:00 | 17699.00 | 2026-01-23 15:20:00 | 17452.00 | TARGET_HIT | 0.50 | 1.40% |
| BUY | retest1 | 2026-01-30 10:45:00 | 18474.00 | 2026-01-30 11:00:00 | 18572.16 | PARTIAL | 0.50 | 0.53% |
| BUY | retest1 | 2026-01-30 10:45:00 | 18474.00 | 2026-01-30 11:15:00 | 18474.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-02-04 10:10:00 | 17801.00 | 2026-02-04 10:35:00 | 17726.92 | PARTIAL | 0.50 | 0.42% |
| SELL | retest1 | 2026-02-04 10:10:00 | 17801.00 | 2026-02-04 11:40:00 | 17801.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-12 09:35:00 | 18425.00 | 2026-02-12 10:05:00 | 18501.80 | PARTIAL | 0.50 | 0.42% |
| BUY | retest1 | 2026-02-12 09:35:00 | 18425.00 | 2026-02-12 10:25:00 | 18425.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-17 11:05:00 | 18326.00 | 2026-02-17 11:20:00 | 18410.47 | PARTIAL | 0.50 | 0.46% |
| BUY | retest1 | 2026-02-17 11:05:00 | 18326.00 | 2026-02-17 12:20:00 | 18355.00 | TARGET_HIT | 0.50 | 0.16% |
| SELL | retest1 | 2026-02-18 11:15:00 | 18230.00 | 2026-02-18 11:30:00 | 18172.82 | PARTIAL | 0.50 | 0.31% |
| SELL | retest1 | 2026-02-18 11:15:00 | 18230.00 | 2026-02-18 12:25:00 | 18230.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-25 09:55:00 | 17995.00 | 2026-02-25 10:20:00 | 17948.22 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2026-02-26 10:05:00 | 18232.00 | 2026-02-26 10:10:00 | 18185.68 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2026-03-06 10:55:00 | 17797.00 | 2026-03-06 14:40:00 | 17860.41 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2026-03-13 11:05:00 | 17529.00 | 2026-03-13 11:50:00 | 17454.60 | PARTIAL | 0.50 | 0.42% |
| SELL | retest1 | 2026-03-13 11:05:00 | 17529.00 | 2026-03-13 15:20:00 | 17006.00 | TARGET_HIT | 0.50 | 2.98% |
| BUY | retest1 | 2026-03-17 10:30:00 | 17270.00 | 2026-03-17 12:00:00 | 17397.85 | PARTIAL | 0.50 | 0.74% |
| BUY | retest1 | 2026-03-17 10:30:00 | 17270.00 | 2026-03-17 14:00:00 | 17270.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-03-20 11:05:00 | 17240.00 | 2026-03-20 11:45:00 | 17284.33 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2026-04-17 10:40:00 | 15691.00 | 2026-04-17 11:25:00 | 15743.14 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2026-04-21 09:45:00 | 16335.00 | 2026-04-21 09:50:00 | 16276.39 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2026-04-22 10:30:00 | 15886.00 | 2026-04-22 11:05:00 | 15933.07 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2026-04-24 10:00:00 | 15947.00 | 2026-04-24 10:05:00 | 15859.96 | PARTIAL | 0.50 | 0.55% |
| SELL | retest1 | 2026-04-24 10:00:00 | 15947.00 | 2026-04-24 14:40:00 | 15940.00 | TARGET_HIT | 0.50 | 0.04% |
| BUY | retest1 | 2026-05-04 09:50:00 | 16416.00 | 2026-05-04 10:10:00 | 16330.60 | STOP_HIT | 1.00 | -0.52% |
| SELL | retest1 | 2026-05-07 10:10:00 | 17021.00 | 2026-05-07 10:50:00 | 17082.15 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2026-05-08 09:40:00 | 17028.00 | 2026-05-08 10:10:00 | 16959.45 | STOP_HIT | 1.00 | -0.40% |
