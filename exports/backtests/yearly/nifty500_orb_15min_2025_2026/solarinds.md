# Solar Industries India Ltd. (SOLARINDS)

## Backtest Summary

- **Window:** 2025-05-12 09:15:00 → 2026-05-08 15:25:00 (18463 bars)
- **Last close:** 16101.00
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
| ENTRY1 | 70 |
| ENTRY2 | 0 |
| PARTIAL | 28 |
| TARGET_HIT | 8 |
| STOP_HIT | 62 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 98 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 36 / 62
- **Target hits / Stop hits / Partials:** 8 / 62 / 28
- **Avg / median % per leg:** 0.04% / 0.00%
- **Sum % (uncompounded):** 4.08%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 44 | 13 | 29.5% | 3 | 31 | 10 | -0.04% | -1.7% |
| BUY @ 2nd Alert (retest1) | 44 | 13 | 29.5% | 3 | 31 | 10 | -0.04% | -1.7% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 54 | 23 | 42.6% | 5 | 31 | 18 | 0.11% | 5.8% |
| SELL @ 2nd Alert (retest1) | 54 | 23 | 42.6% | 5 | 31 | 18 | 0.11% | 5.8% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 98 | 36 | 36.7% | 8 | 62 | 28 | 0.04% | 4.1% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-28 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-28 10:45:00 | 16080.00 | 15990.63 | 0.00 | ORB-long ORB[15854.00,16010.00] vol=1.7x ATR=44.62 |
| Stop hit — per-position SL triggered | 2025-05-28 10:55:00 | 16035.38 | 15996.86 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2025-06-03 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-03 09:30:00 | 16595.00 | 16476.00 | 0.00 | ORB-long ORB[16312.00,16547.00] vol=2.7x ATR=44.61 |
| Stop hit — per-position SL triggered | 2025-06-03 09:40:00 | 16550.39 | 16497.06 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2025-06-09 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-09 10:45:00 | 16875.00 | 16736.93 | 0.00 | ORB-long ORB[16546.00,16715.00] vol=2.8x ATR=47.52 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-09 10:55:00 | 16946.28 | 16750.95 | 0.00 | T1 1.5R @ 16946.28 |
| Stop hit — per-position SL triggered | 2025-06-09 12:30:00 | 16875.00 | 16814.36 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2025-06-10 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-10 10:20:00 | 16985.00 | 16885.49 | 0.00 | ORB-long ORB[16781.00,16949.00] vol=1.7x ATR=48.02 |
| Stop hit — per-position SL triggered | 2025-06-10 10:30:00 | 16936.98 | 16894.57 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2025-06-11 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-11 10:00:00 | 17000.00 | 16966.81 | 0.00 | ORB-long ORB[16851.00,16990.00] vol=1.8x ATR=40.86 |
| Stop hit — per-position SL triggered | 2025-06-11 10:05:00 | 16959.14 | 16965.60 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2025-06-17 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-17 09:30:00 | 17119.00 | 17065.32 | 0.00 | ORB-long ORB[16975.00,17099.00] vol=2.7x ATR=36.99 |
| Stop hit — per-position SL triggered | 2025-06-17 09:40:00 | 17082.01 | 17075.95 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2025-06-25 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-25 10:10:00 | 16863.00 | 17072.93 | 0.00 | ORB-short ORB[17013.00,17215.00] vol=1.7x ATR=49.11 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-25 10:30:00 | 16789.33 | 17006.44 | 0.00 | T1 1.5R @ 16789.33 |
| Stop hit — per-position SL triggered | 2025-06-25 10:50:00 | 16863.00 | 16990.29 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2025-06-27 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-27 10:45:00 | 17392.00 | 17289.96 | 0.00 | ORB-long ORB[17230.00,17351.00] vol=3.5x ATR=43.81 |
| Stop hit — per-position SL triggered | 2025-06-27 11:00:00 | 17348.19 | 17325.99 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2025-07-02 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-02 10:40:00 | 16992.00 | 17051.45 | 0.00 | ORB-short ORB[17075.00,17283.00] vol=1.8x ATR=42.16 |
| Stop hit — per-position SL triggered | 2025-07-02 10:45:00 | 17034.16 | 17047.34 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2025-07-07 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-07 09:45:00 | 16665.00 | 16824.29 | 0.00 | ORB-short ORB[16840.00,16944.00] vol=2.6x ATR=49.23 |
| Stop hit — per-position SL triggered | 2025-07-07 09:50:00 | 16714.23 | 16808.10 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2025-07-08 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-08 09:35:00 | 16624.00 | 16701.76 | 0.00 | ORB-short ORB[16650.00,16837.00] vol=2.4x ATR=45.84 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-08 09:55:00 | 16555.24 | 16663.51 | 0.00 | T1 1.5R @ 16555.24 |
| Target hit | 2025-07-08 13:15:00 | 16502.00 | 16492.06 | 0.00 | Trail-exit close>VWAP |

### Cycle 12 — SELL (started 2025-07-18 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-18 10:20:00 | 14974.00 | 15035.71 | 0.00 | ORB-short ORB[15067.00,15221.00] vol=1.6x ATR=35.44 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-18 10:25:00 | 14920.84 | 15022.77 | 0.00 | T1 1.5R @ 14920.84 |
| Stop hit — per-position SL triggered | 2025-07-18 11:15:00 | 14974.00 | 14986.00 | 0.00 | SL hit |

### Cycle 13 — SELL (started 2025-07-23 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-23 09:35:00 | 14843.00 | 14901.55 | 0.00 | ORB-short ORB[14865.00,14993.00] vol=1.5x ATR=42.16 |
| Stop hit — per-position SL triggered | 2025-07-23 09:55:00 | 14885.16 | 14882.72 | 0.00 | SL hit |

### Cycle 14 — SELL (started 2025-07-24 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-24 11:05:00 | 14700.00 | 14770.83 | 0.00 | ORB-short ORB[14801.00,14878.00] vol=3.4x ATR=35.36 |
| Stop hit — per-position SL triggered | 2025-07-24 11:40:00 | 14735.36 | 14763.76 | 0.00 | SL hit |

### Cycle 15 — SELL (started 2025-07-25 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-25 10:35:00 | 14596.00 | 14703.63 | 0.00 | ORB-short ORB[14720.00,14798.00] vol=2.7x ATR=35.57 |
| Stop hit — per-position SL triggered | 2025-07-25 10:40:00 | 14631.57 | 14699.51 | 0.00 | SL hit |

### Cycle 16 — BUY (started 2025-08-05 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-05 11:05:00 | 14606.00 | 14442.22 | 0.00 | ORB-long ORB[14364.00,14485.00] vol=2.8x ATR=44.51 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-05 11:10:00 | 14672.77 | 14456.86 | 0.00 | T1 1.5R @ 14672.77 |
| Stop hit — per-position SL triggered | 2025-08-05 11:30:00 | 14606.00 | 14491.74 | 0.00 | SL hit |

### Cycle 17 — SELL (started 2025-08-07 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-07 11:05:00 | 14737.00 | 14870.00 | 0.00 | ORB-short ORB[14863.00,15059.00] vol=1.5x ATR=50.53 |
| Stop hit — per-position SL triggered | 2025-08-07 12:10:00 | 14787.53 | 14831.16 | 0.00 | SL hit |

### Cycle 18 — SELL (started 2025-08-19 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-19 10:55:00 | 14836.00 | 14847.91 | 0.00 | ORB-short ORB[14881.00,15080.00] vol=2.5x ATR=35.29 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-19 11:05:00 | 14783.07 | 14845.96 | 0.00 | T1 1.5R @ 14783.07 |
| Target hit | 2025-08-19 12:05:00 | 14834.00 | 14833.09 | 0.00 | Trail-exit close>VWAP |

### Cycle 19 — SELL (started 2025-08-20 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-20 09:50:00 | 14585.00 | 14648.65 | 0.00 | ORB-short ORB[14620.00,14719.00] vol=1.6x ATR=37.17 |
| Stop hit — per-position SL triggered | 2025-08-20 10:15:00 | 14622.17 | 14636.14 | 0.00 | SL hit |

### Cycle 20 — BUY (started 2025-08-21 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-21 09:55:00 | 14624.00 | 14517.79 | 0.00 | ORB-long ORB[14432.00,14562.00] vol=1.5x ATR=47.12 |
| Stop hit — per-position SL triggered | 2025-08-21 10:00:00 | 14576.88 | 14538.15 | 0.00 | SL hit |

### Cycle 21 — BUY (started 2025-09-01 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-01 10:40:00 | 13940.00 | 13913.27 | 0.00 | ORB-long ORB[13825.00,13925.00] vol=4.6x ATR=35.57 |
| Stop hit — per-position SL triggered | 2025-09-01 10:45:00 | 13904.43 | 13914.45 | 0.00 | SL hit |

### Cycle 22 — SELL (started 2025-09-05 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-05 10:05:00 | 13994.00 | 14078.37 | 0.00 | ORB-short ORB[14079.00,14189.00] vol=1.7x ATR=37.53 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-05 10:15:00 | 13937.71 | 14058.34 | 0.00 | T1 1.5R @ 13937.71 |
| Stop hit — per-position SL triggered | 2025-09-05 13:45:00 | 13994.00 | 13971.86 | 0.00 | SL hit |

### Cycle 23 — SELL (started 2025-09-08 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-08 09:50:00 | 13828.00 | 13925.19 | 0.00 | ORB-short ORB[13882.00,13994.00] vol=2.0x ATR=38.92 |
| Stop hit — per-position SL triggered | 2025-09-08 10:00:00 | 13866.92 | 13912.00 | 0.00 | SL hit |

### Cycle 24 — SELL (started 2025-09-10 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-10 11:00:00 | 13917.00 | 13973.44 | 0.00 | ORB-short ORB[13935.00,14138.00] vol=1.8x ATR=30.33 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-10 11:15:00 | 13871.51 | 13966.56 | 0.00 | T1 1.5R @ 13871.51 |
| Stop hit — per-position SL triggered | 2025-09-10 12:00:00 | 13917.00 | 13943.94 | 0.00 | SL hit |

### Cycle 25 — BUY (started 2025-09-12 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-12 09:50:00 | 14180.00 | 14102.99 | 0.00 | ORB-long ORB[14002.00,14100.00] vol=2.2x ATR=35.00 |
| Stop hit — per-position SL triggered | 2025-09-12 09:55:00 | 14145.00 | 14105.79 | 0.00 | SL hit |

### Cycle 26 — BUY (started 2025-09-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-16 11:15:00 | 14641.00 | 14576.75 | 0.00 | ORB-long ORB[14403.00,14619.00] vol=1.6x ATR=31.16 |
| Stop hit — per-position SL triggered | 2025-09-16 11:30:00 | 14609.84 | 14578.52 | 0.00 | SL hit |

### Cycle 27 — BUY (started 2025-09-17 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-17 09:35:00 | 14802.00 | 14731.45 | 0.00 | ORB-long ORB[14629.00,14760.00] vol=2.2x ATR=42.50 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-17 09:55:00 | 14865.75 | 14776.63 | 0.00 | T1 1.5R @ 14865.75 |
| Target hit | 2025-09-17 10:25:00 | 14811.00 | 14811.50 | 0.00 | Trail-exit close<VWAP |

### Cycle 28 — SELL (started 2025-09-18 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-18 10:10:00 | 14584.00 | 14705.51 | 0.00 | ORB-short ORB[14688.00,14790.00] vol=1.7x ATR=33.96 |
| Stop hit — per-position SL triggered | 2025-09-18 10:30:00 | 14617.96 | 14676.01 | 0.00 | SL hit |

### Cycle 29 — SELL (started 2025-09-22 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-22 11:05:00 | 14565.00 | 14647.22 | 0.00 | ORB-short ORB[14580.00,14703.00] vol=1.9x ATR=31.73 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-22 11:15:00 | 14517.40 | 14636.36 | 0.00 | T1 1.5R @ 14517.40 |
| Stop hit — per-position SL triggered | 2025-09-22 11:20:00 | 14565.00 | 14626.72 | 0.00 | SL hit |

### Cycle 30 — SELL (started 2025-09-23 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-23 09:30:00 | 14327.00 | 14416.93 | 0.00 | ORB-short ORB[14360.00,14527.00] vol=2.4x ATR=40.18 |
| Stop hit — per-position SL triggered | 2025-09-23 09:35:00 | 14367.18 | 14408.93 | 0.00 | SL hit |

### Cycle 31 — SELL (started 2025-09-26 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-26 10:55:00 | 13940.00 | 14021.37 | 0.00 | ORB-short ORB[14001.00,14190.00] vol=2.4x ATR=38.27 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-26 11:30:00 | 13882.59 | 13990.81 | 0.00 | T1 1.5R @ 13882.59 |
| Stop hit — per-position SL triggered | 2025-09-26 12:10:00 | 13940.00 | 13950.57 | 0.00 | SL hit |

### Cycle 32 — BUY (started 2025-10-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-06 10:15:00 | 14143.00 | 13953.51 | 0.00 | ORB-long ORB[13830.00,14012.00] vol=1.9x ATR=56.14 |
| Stop hit — per-position SL triggered | 2025-10-06 10:50:00 | 14086.86 | 14018.04 | 0.00 | SL hit |

### Cycle 33 — BUY (started 2025-10-07 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-07 09:35:00 | 14190.00 | 14116.15 | 0.00 | ORB-long ORB[14020.00,14180.00] vol=1.5x ATR=39.94 |
| Stop hit — per-position SL triggered | 2025-10-07 09:40:00 | 14150.06 | 14122.83 | 0.00 | SL hit |

### Cycle 34 — BUY (started 2025-10-09 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-09 10:00:00 | 14118.00 | 14034.16 | 0.00 | ORB-long ORB[13957.00,14099.00] vol=1.5x ATR=50.05 |
| Stop hit — per-position SL triggered | 2025-10-09 10:10:00 | 14067.95 | 14041.77 | 0.00 | SL hit |

### Cycle 35 — SELL (started 2025-10-13 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-13 09:30:00 | 13976.00 | 14028.38 | 0.00 | ORB-short ORB[14001.00,14125.00] vol=1.9x ATR=39.61 |
| Stop hit — per-position SL triggered | 2025-10-13 09:35:00 | 14015.61 | 14025.78 | 0.00 | SL hit |

### Cycle 36 — BUY (started 2025-10-15 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-15 10:30:00 | 14035.00 | 13999.74 | 0.00 | ORB-long ORB[13951.00,14025.00] vol=1.5x ATR=24.79 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-15 10:40:00 | 14072.18 | 14009.75 | 0.00 | T1 1.5R @ 14072.18 |
| Stop hit — per-position SL triggered | 2025-10-15 10:50:00 | 14035.00 | 14018.30 | 0.00 | SL hit |

### Cycle 37 — SELL (started 2025-10-16 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-16 10:40:00 | 13994.00 | 14040.22 | 0.00 | ORB-short ORB[14052.00,14124.00] vol=2.5x ATR=24.15 |
| Stop hit — per-position SL triggered | 2025-10-16 11:10:00 | 14018.15 | 14031.26 | 0.00 | SL hit |

### Cycle 38 — BUY (started 2025-10-28 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-28 09:55:00 | 14066.00 | 14043.14 | 0.00 | ORB-long ORB[13957.00,14036.00] vol=15.8x ATR=26.53 |
| Stop hit — per-position SL triggered | 2025-10-28 10:00:00 | 14039.47 | 14042.99 | 0.00 | SL hit |

### Cycle 39 — SELL (started 2025-10-30 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-30 09:45:00 | 13878.00 | 13910.18 | 0.00 | ORB-short ORB[13922.00,13995.00] vol=2.9x ATR=30.13 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-30 10:00:00 | 13832.80 | 13892.55 | 0.00 | T1 1.5R @ 13832.80 |
| Stop hit — per-position SL triggered | 2025-10-30 11:05:00 | 13878.00 | 13878.28 | 0.00 | SL hit |

### Cycle 40 — BUY (started 2025-10-31 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-31 09:55:00 | 13970.00 | 13906.94 | 0.00 | ORB-long ORB[13835.00,13921.00] vol=1.6x ATR=25.86 |
| Stop hit — per-position SL triggered | 2025-10-31 10:00:00 | 13944.14 | 13909.55 | 0.00 | SL hit |

### Cycle 41 — BUY (started 2025-11-03 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-03 11:00:00 | 14020.00 | 13929.37 | 0.00 | ORB-long ORB[13826.00,13936.00] vol=2.6x ATR=30.79 |
| Stop hit — per-position SL triggered | 2025-11-03 11:05:00 | 13989.21 | 13933.42 | 0.00 | SL hit |

### Cycle 42 — SELL (started 2025-11-04 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-04 09:35:00 | 14055.00 | 14091.68 | 0.00 | ORB-short ORB[14068.00,14172.00] vol=1.5x ATR=32.50 |
| Stop hit — per-position SL triggered | 2025-11-04 09:50:00 | 14087.50 | 14080.36 | 0.00 | SL hit |

### Cycle 43 — BUY (started 2025-11-17 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-17 10:45:00 | 14054.00 | 13916.55 | 0.00 | ORB-long ORB[13822.00,13980.00] vol=2.4x ATR=42.91 |
| Stop hit — per-position SL triggered | 2025-11-17 10:50:00 | 14011.09 | 13937.60 | 0.00 | SL hit |

### Cycle 44 — SELL (started 2025-11-27 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-27 10:45:00 | 13436.00 | 13517.35 | 0.00 | ORB-short ORB[13491.00,13577.00] vol=1.7x ATR=31.88 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-27 11:25:00 | 13388.18 | 13490.85 | 0.00 | T1 1.5R @ 13388.18 |
| Target hit | 2025-11-27 15:20:00 | 13350.00 | 13411.45 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 45 — SELL (started 2025-11-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-28 10:15:00 | 13242.00 | 13291.27 | 0.00 | ORB-short ORB[13260.00,13379.00] vol=1.6x ATR=31.89 |
| Stop hit — per-position SL triggered | 2025-11-28 10:40:00 | 13273.89 | 13275.49 | 0.00 | SL hit |

### Cycle 46 — SELL (started 2025-12-02 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-02 10:45:00 | 13176.00 | 13211.07 | 0.00 | ORB-short ORB[13179.00,13285.00] vol=3.6x ATR=28.82 |
| Stop hit — per-position SL triggered | 2025-12-02 10:50:00 | 13204.82 | 13210.54 | 0.00 | SL hit |

### Cycle 47 — SELL (started 2025-12-12 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-12 10:35:00 | 12325.00 | 12400.54 | 0.00 | ORB-short ORB[12431.00,12518.00] vol=1.8x ATR=27.23 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-12 10:40:00 | 12284.16 | 12391.16 | 0.00 | T1 1.5R @ 12284.16 |
| Stop hit — per-position SL triggered | 2025-12-12 11:00:00 | 12325.00 | 12363.51 | 0.00 | SL hit |

### Cycle 48 — SELL (started 2025-12-16 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-16 10:50:00 | 12040.00 | 12081.53 | 0.00 | ORB-short ORB[12050.00,12200.00] vol=1.6x ATR=24.13 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-16 11:00:00 | 12003.81 | 12078.09 | 0.00 | T1 1.5R @ 12003.81 |
| Stop hit — per-position SL triggered | 2025-12-16 12:05:00 | 12040.00 | 12059.87 | 0.00 | SL hit |

### Cycle 49 — SELL (started 2025-12-18 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-18 09:40:00 | 11703.00 | 11734.35 | 0.00 | ORB-short ORB[11720.00,11827.00] vol=1.8x ATR=26.87 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-18 09:55:00 | 11662.70 | 11702.31 | 0.00 | T1 1.5R @ 11662.70 |
| Stop hit — per-position SL triggered | 2025-12-18 10:10:00 | 11703.00 | 11698.12 | 0.00 | SL hit |

### Cycle 50 — BUY (started 2025-12-24 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-24 10:55:00 | 12623.00 | 12587.04 | 0.00 | ORB-long ORB[12500.00,12620.00] vol=3.5x ATR=29.18 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-24 11:55:00 | 12666.77 | 12603.52 | 0.00 | T1 1.5R @ 12666.77 |
| Stop hit — per-position SL triggered | 2025-12-24 13:20:00 | 12623.00 | 12626.42 | 0.00 | SL hit |

### Cycle 51 — SELL (started 2025-12-30 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-30 10:30:00 | 12171.00 | 12232.75 | 0.00 | ORB-short ORB[12235.00,12360.00] vol=2.0x ATR=39.79 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-30 10:45:00 | 12111.31 | 12206.53 | 0.00 | T1 1.5R @ 12111.31 |
| Target hit | 2025-12-30 14:15:00 | 12004.00 | 11988.84 | 0.00 | Trail-exit close>VWAP |

### Cycle 52 — BUY (started 2026-01-05 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-05 09:40:00 | 12644.00 | 12500.81 | 0.00 | ORB-long ORB[12250.00,12434.00] vol=3.6x ATR=43.95 |
| Stop hit — per-position SL triggered | 2026-01-05 09:45:00 | 12600.05 | 12522.33 | 0.00 | SL hit |

### Cycle 53 — SELL (started 2026-01-06 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-06 11:00:00 | 12603.00 | 12687.72 | 0.00 | ORB-short ORB[12670.00,12788.00] vol=2.0x ATR=33.13 |
| Stop hit — per-position SL triggered | 2026-01-06 11:20:00 | 12636.13 | 12680.82 | 0.00 | SL hit |

### Cycle 54 — BUY (started 2026-01-07 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-07 10:10:00 | 12974.00 | 12909.42 | 0.00 | ORB-long ORB[12820.00,12950.00] vol=4.2x ATR=38.94 |
| Stop hit — per-position SL triggered | 2026-01-07 10:25:00 | 12935.06 | 12923.96 | 0.00 | SL hit |

### Cycle 55 — BUY (started 2026-01-08 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-08 09:35:00 | 13570.00 | 13497.13 | 0.00 | ORB-long ORB[13353.00,13545.00] vol=1.6x ATR=67.09 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-08 10:05:00 | 13670.63 | 13553.96 | 0.00 | T1 1.5R @ 13670.63 |
| Target hit | 2026-01-08 10:40:00 | 13588.00 | 13592.88 | 0.00 | Trail-exit close<VWAP |

### Cycle 56 — BUY (started 2026-01-14 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-14 09:30:00 | 12787.00 | 12718.86 | 0.00 | ORB-long ORB[12600.00,12763.00] vol=1.7x ATR=55.94 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-14 10:25:00 | 12870.91 | 12787.74 | 0.00 | T1 1.5R @ 12870.91 |
| Stop hit — per-position SL triggered | 2026-01-14 10:30:00 | 12787.00 | 12787.20 | 0.00 | SL hit |

### Cycle 57 — SELL (started 2026-01-20 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-20 09:35:00 | 12800.00 | 12859.62 | 0.00 | ORB-short ORB[12836.00,12969.00] vol=1.8x ATR=46.97 |
| Stop hit — per-position SL triggered | 2026-01-20 09:50:00 | 12846.97 | 12835.60 | 0.00 | SL hit |

### Cycle 58 — SELL (started 2026-02-06 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-06 09:55:00 | 13125.00 | 13192.63 | 0.00 | ORB-short ORB[13206.00,13370.00] vol=1.5x ATR=48.07 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-06 10:20:00 | 13052.90 | 13168.46 | 0.00 | T1 1.5R @ 13052.90 |
| Target hit | 2026-02-06 15:20:00 | 13009.00 | 13017.77 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 59 — SELL (started 2026-02-12 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-12 11:10:00 | 13271.00 | 13289.95 | 0.00 | ORB-short ORB[13280.00,13439.00] vol=1.7x ATR=27.30 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-12 11:30:00 | 13230.05 | 13286.73 | 0.00 | T1 1.5R @ 13230.05 |
| Stop hit — per-position SL triggered | 2026-02-12 12:55:00 | 13271.00 | 13279.61 | 0.00 | SL hit |

### Cycle 60 — BUY (started 2026-02-17 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-17 09:30:00 | 13208.00 | 13163.70 | 0.00 | ORB-long ORB[13025.00,13199.00] vol=1.5x ATR=37.15 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-17 09:45:00 | 13263.73 | 13192.81 | 0.00 | T1 1.5R @ 13263.73 |
| Target hit | 2026-02-17 12:25:00 | 13220.00 | 13224.55 | 0.00 | Trail-exit close<VWAP |

### Cycle 61 — SELL (started 2026-02-23 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-23 10:55:00 | 13242.00 | 13421.03 | 0.00 | ORB-short ORB[13401.00,13590.00] vol=2.4x ATR=46.51 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-23 11:15:00 | 13172.23 | 13400.05 | 0.00 | T1 1.5R @ 13172.23 |
| Stop hit — per-position SL triggered | 2026-02-23 12:10:00 | 13242.00 | 13368.69 | 0.00 | SL hit |

### Cycle 62 — BUY (started 2026-02-25 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-25 11:00:00 | 13462.00 | 13406.55 | 0.00 | ORB-long ORB[13276.00,13415.00] vol=1.6x ATR=30.13 |
| Stop hit — per-position SL triggered | 2026-02-25 11:25:00 | 13431.87 | 13414.86 | 0.00 | SL hit |

### Cycle 63 — BUY (started 2026-02-26 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-26 09:30:00 | 13541.00 | 13510.46 | 0.00 | ORB-long ORB[13405.00,13540.00] vol=1.7x ATR=33.86 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-26 09:50:00 | 13591.79 | 13550.01 | 0.00 | T1 1.5R @ 13591.79 |
| Stop hit — per-position SL triggered | 2026-02-26 10:00:00 | 13541.00 | 13555.07 | 0.00 | SL hit |

### Cycle 64 — BUY (started 2026-03-18 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-18 09:35:00 | 14097.00 | 14008.85 | 0.00 | ORB-long ORB[13880.00,14088.00] vol=2.1x ATR=54.43 |
| Stop hit — per-position SL triggered | 2026-03-18 09:55:00 | 14042.57 | 14034.49 | 0.00 | SL hit |

### Cycle 65 — SELL (started 2026-03-20 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-20 10:55:00 | 13204.00 | 13325.30 | 0.00 | ORB-short ORB[13251.00,13445.00] vol=1.5x ATR=49.27 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-20 12:05:00 | 13130.10 | 13283.89 | 0.00 | T1 1.5R @ 13130.10 |
| Stop hit — per-position SL triggered | 2026-03-20 13:15:00 | 13204.00 | 13249.86 | 0.00 | SL hit |

### Cycle 66 — BUY (started 2026-04-21 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-21 09:35:00 | 15051.00 | 14971.33 | 0.00 | ORB-long ORB[14851.00,15035.00] vol=1.9x ATR=48.76 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-21 10:10:00 | 15124.14 | 15025.39 | 0.00 | T1 1.5R @ 15124.14 |
| Stop hit — per-position SL triggered | 2026-04-21 11:55:00 | 15051.00 | 15056.79 | 0.00 | SL hit |

### Cycle 67 — BUY (started 2026-04-22 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-22 09:30:00 | 15045.00 | 14990.98 | 0.00 | ORB-long ORB[14925.00,15036.00] vol=1.8x ATR=40.83 |
| Stop hit — per-position SL triggered | 2026-04-22 09:40:00 | 15004.17 | 14998.16 | 0.00 | SL hit |

### Cycle 68 — SELL (started 2026-04-24 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-24 09:30:00 | 15684.00 | 15852.31 | 0.00 | ORB-short ORB[15811.00,15995.00] vol=2.0x ATR=75.11 |
| Stop hit — per-position SL triggered | 2026-04-24 09:35:00 | 15759.11 | 15840.49 | 0.00 | SL hit |

### Cycle 69 — BUY (started 2026-04-29 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-29 11:10:00 | 15541.00 | 15452.26 | 0.00 | ORB-long ORB[15407.00,15516.00] vol=1.5x ATR=40.99 |
| Stop hit — per-position SL triggered | 2026-04-29 11:45:00 | 15500.01 | 15467.97 | 0.00 | SL hit |

### Cycle 70 — BUY (started 2026-05-05 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-05 09:30:00 | 15544.00 | 15498.07 | 0.00 | ORB-long ORB[15407.00,15524.00] vol=2.1x ATR=49.01 |
| Stop hit — per-position SL triggered | 2026-05-05 09:45:00 | 15494.99 | 15515.43 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-05-28 10:45:00 | 16080.00 | 2025-05-28 10:55:00 | 16035.38 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2025-06-03 09:30:00 | 16595.00 | 2025-06-03 09:40:00 | 16550.39 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2025-06-09 10:45:00 | 16875.00 | 2025-06-09 10:55:00 | 16946.28 | PARTIAL | 0.50 | 0.42% |
| BUY | retest1 | 2025-06-09 10:45:00 | 16875.00 | 2025-06-09 12:30:00 | 16875.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-06-10 10:20:00 | 16985.00 | 2025-06-10 10:30:00 | 16936.98 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2025-06-11 10:00:00 | 17000.00 | 2025-06-11 10:05:00 | 16959.14 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2025-06-17 09:30:00 | 17119.00 | 2025-06-17 09:40:00 | 17082.01 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2025-06-25 10:10:00 | 16863.00 | 2025-06-25 10:30:00 | 16789.33 | PARTIAL | 0.50 | 0.44% |
| SELL | retest1 | 2025-06-25 10:10:00 | 16863.00 | 2025-06-25 10:50:00 | 16863.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-06-27 10:45:00 | 17392.00 | 2025-06-27 11:00:00 | 17348.19 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2025-07-02 10:40:00 | 16992.00 | 2025-07-02 10:45:00 | 17034.16 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2025-07-07 09:45:00 | 16665.00 | 2025-07-07 09:50:00 | 16714.23 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2025-07-08 09:35:00 | 16624.00 | 2025-07-08 09:55:00 | 16555.24 | PARTIAL | 0.50 | 0.41% |
| SELL | retest1 | 2025-07-08 09:35:00 | 16624.00 | 2025-07-08 13:15:00 | 16502.00 | TARGET_HIT | 0.50 | 0.73% |
| SELL | retest1 | 2025-07-18 10:20:00 | 14974.00 | 2025-07-18 10:25:00 | 14920.84 | PARTIAL | 0.50 | 0.36% |
| SELL | retest1 | 2025-07-18 10:20:00 | 14974.00 | 2025-07-18 11:15:00 | 14974.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-07-23 09:35:00 | 14843.00 | 2025-07-23 09:55:00 | 14885.16 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2025-07-24 11:05:00 | 14700.00 | 2025-07-24 11:40:00 | 14735.36 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2025-07-25 10:35:00 | 14596.00 | 2025-07-25 10:40:00 | 14631.57 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2025-08-05 11:05:00 | 14606.00 | 2025-08-05 11:10:00 | 14672.77 | PARTIAL | 0.50 | 0.46% |
| BUY | retest1 | 2025-08-05 11:05:00 | 14606.00 | 2025-08-05 11:30:00 | 14606.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-08-07 11:05:00 | 14737.00 | 2025-08-07 12:10:00 | 14787.53 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2025-08-19 10:55:00 | 14836.00 | 2025-08-19 11:05:00 | 14783.07 | PARTIAL | 0.50 | 0.36% |
| SELL | retest1 | 2025-08-19 10:55:00 | 14836.00 | 2025-08-19 12:05:00 | 14834.00 | TARGET_HIT | 0.50 | 0.01% |
| SELL | retest1 | 2025-08-20 09:50:00 | 14585.00 | 2025-08-20 10:15:00 | 14622.17 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2025-08-21 09:55:00 | 14624.00 | 2025-08-21 10:00:00 | 14576.88 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2025-09-01 10:40:00 | 13940.00 | 2025-09-01 10:45:00 | 13904.43 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2025-09-05 10:05:00 | 13994.00 | 2025-09-05 10:15:00 | 13937.71 | PARTIAL | 0.50 | 0.40% |
| SELL | retest1 | 2025-09-05 10:05:00 | 13994.00 | 2025-09-05 13:45:00 | 13994.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-09-08 09:50:00 | 13828.00 | 2025-09-08 10:00:00 | 13866.92 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2025-09-10 11:00:00 | 13917.00 | 2025-09-10 11:15:00 | 13871.51 | PARTIAL | 0.50 | 0.33% |
| SELL | retest1 | 2025-09-10 11:00:00 | 13917.00 | 2025-09-10 12:00:00 | 13917.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-09-12 09:50:00 | 14180.00 | 2025-09-12 09:55:00 | 14145.00 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2025-09-16 11:15:00 | 14641.00 | 2025-09-16 11:30:00 | 14609.84 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2025-09-17 09:35:00 | 14802.00 | 2025-09-17 09:55:00 | 14865.75 | PARTIAL | 0.50 | 0.43% |
| BUY | retest1 | 2025-09-17 09:35:00 | 14802.00 | 2025-09-17 10:25:00 | 14811.00 | TARGET_HIT | 0.50 | 0.06% |
| SELL | retest1 | 2025-09-18 10:10:00 | 14584.00 | 2025-09-18 10:30:00 | 14617.96 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2025-09-22 11:05:00 | 14565.00 | 2025-09-22 11:15:00 | 14517.40 | PARTIAL | 0.50 | 0.33% |
| SELL | retest1 | 2025-09-22 11:05:00 | 14565.00 | 2025-09-22 11:20:00 | 14565.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-09-23 09:30:00 | 14327.00 | 2025-09-23 09:35:00 | 14367.18 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2025-09-26 10:55:00 | 13940.00 | 2025-09-26 11:30:00 | 13882.59 | PARTIAL | 0.50 | 0.41% |
| SELL | retest1 | 2025-09-26 10:55:00 | 13940.00 | 2025-09-26 12:10:00 | 13940.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-10-06 10:15:00 | 14143.00 | 2025-10-06 10:50:00 | 14086.86 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest1 | 2025-10-07 09:35:00 | 14190.00 | 2025-10-07 09:40:00 | 14150.06 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2025-10-09 10:00:00 | 14118.00 | 2025-10-09 10:10:00 | 14067.95 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2025-10-13 09:30:00 | 13976.00 | 2025-10-13 09:35:00 | 14015.61 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2025-10-15 10:30:00 | 14035.00 | 2025-10-15 10:40:00 | 14072.18 | PARTIAL | 0.50 | 0.26% |
| BUY | retest1 | 2025-10-15 10:30:00 | 14035.00 | 2025-10-15 10:50:00 | 14035.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-10-16 10:40:00 | 13994.00 | 2025-10-16 11:10:00 | 14018.15 | STOP_HIT | 1.00 | -0.17% |
| BUY | retest1 | 2025-10-28 09:55:00 | 14066.00 | 2025-10-28 10:00:00 | 14039.47 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest1 | 2025-10-30 09:45:00 | 13878.00 | 2025-10-30 10:00:00 | 13832.80 | PARTIAL | 0.50 | 0.33% |
| SELL | retest1 | 2025-10-30 09:45:00 | 13878.00 | 2025-10-30 11:05:00 | 13878.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-10-31 09:55:00 | 13970.00 | 2025-10-31 10:00:00 | 13944.14 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest1 | 2025-11-03 11:00:00 | 14020.00 | 2025-11-03 11:05:00 | 13989.21 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2025-11-04 09:35:00 | 14055.00 | 2025-11-04 09:50:00 | 14087.50 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2025-11-17 10:45:00 | 14054.00 | 2025-11-17 10:50:00 | 14011.09 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2025-11-27 10:45:00 | 13436.00 | 2025-11-27 11:25:00 | 13388.18 | PARTIAL | 0.50 | 0.36% |
| SELL | retest1 | 2025-11-27 10:45:00 | 13436.00 | 2025-11-27 15:20:00 | 13350.00 | TARGET_HIT | 0.50 | 0.64% |
| SELL | retest1 | 2025-11-28 10:15:00 | 13242.00 | 2025-11-28 10:40:00 | 13273.89 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2025-12-02 10:45:00 | 13176.00 | 2025-12-02 10:50:00 | 13204.82 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2025-12-12 10:35:00 | 12325.00 | 2025-12-12 10:40:00 | 12284.16 | PARTIAL | 0.50 | 0.33% |
| SELL | retest1 | 2025-12-12 10:35:00 | 12325.00 | 2025-12-12 11:00:00 | 12325.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-12-16 10:50:00 | 12040.00 | 2025-12-16 11:00:00 | 12003.81 | PARTIAL | 0.50 | 0.30% |
| SELL | retest1 | 2025-12-16 10:50:00 | 12040.00 | 2025-12-16 12:05:00 | 12040.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-12-18 09:40:00 | 11703.00 | 2025-12-18 09:55:00 | 11662.70 | PARTIAL | 0.50 | 0.34% |
| SELL | retest1 | 2025-12-18 09:40:00 | 11703.00 | 2025-12-18 10:10:00 | 11703.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-12-24 10:55:00 | 12623.00 | 2025-12-24 11:55:00 | 12666.77 | PARTIAL | 0.50 | 0.35% |
| BUY | retest1 | 2025-12-24 10:55:00 | 12623.00 | 2025-12-24 13:20:00 | 12623.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-12-30 10:30:00 | 12171.00 | 2025-12-30 10:45:00 | 12111.31 | PARTIAL | 0.50 | 0.49% |
| SELL | retest1 | 2025-12-30 10:30:00 | 12171.00 | 2025-12-30 14:15:00 | 12004.00 | TARGET_HIT | 0.50 | 1.37% |
| BUY | retest1 | 2026-01-05 09:40:00 | 12644.00 | 2026-01-05 09:45:00 | 12600.05 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2026-01-06 11:00:00 | 12603.00 | 2026-01-06 11:20:00 | 12636.13 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2026-01-07 10:10:00 | 12974.00 | 2026-01-07 10:25:00 | 12935.06 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2026-01-08 09:35:00 | 13570.00 | 2026-01-08 10:05:00 | 13670.63 | PARTIAL | 0.50 | 0.74% |
| BUY | retest1 | 2026-01-08 09:35:00 | 13570.00 | 2026-01-08 10:40:00 | 13588.00 | TARGET_HIT | 0.50 | 0.13% |
| BUY | retest1 | 2026-01-14 09:30:00 | 12787.00 | 2026-01-14 10:25:00 | 12870.91 | PARTIAL | 0.50 | 0.66% |
| BUY | retest1 | 2026-01-14 09:30:00 | 12787.00 | 2026-01-14 10:30:00 | 12787.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-01-20 09:35:00 | 12800.00 | 2026-01-20 09:50:00 | 12846.97 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest1 | 2026-02-06 09:55:00 | 13125.00 | 2026-02-06 10:20:00 | 13052.90 | PARTIAL | 0.50 | 0.55% |
| SELL | retest1 | 2026-02-06 09:55:00 | 13125.00 | 2026-02-06 15:20:00 | 13009.00 | TARGET_HIT | 0.50 | 0.88% |
| SELL | retest1 | 2026-02-12 11:10:00 | 13271.00 | 2026-02-12 11:30:00 | 13230.05 | PARTIAL | 0.50 | 0.31% |
| SELL | retest1 | 2026-02-12 11:10:00 | 13271.00 | 2026-02-12 12:55:00 | 13271.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-17 09:30:00 | 13208.00 | 2026-02-17 09:45:00 | 13263.73 | PARTIAL | 0.50 | 0.42% |
| BUY | retest1 | 2026-02-17 09:30:00 | 13208.00 | 2026-02-17 12:25:00 | 13220.00 | TARGET_HIT | 0.50 | 0.09% |
| SELL | retest1 | 2026-02-23 10:55:00 | 13242.00 | 2026-02-23 11:15:00 | 13172.23 | PARTIAL | 0.50 | 0.53% |
| SELL | retest1 | 2026-02-23 10:55:00 | 13242.00 | 2026-02-23 12:10:00 | 13242.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-25 11:00:00 | 13462.00 | 2026-02-25 11:25:00 | 13431.87 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2026-02-26 09:30:00 | 13541.00 | 2026-02-26 09:50:00 | 13591.79 | PARTIAL | 0.50 | 0.38% |
| BUY | retest1 | 2026-02-26 09:30:00 | 13541.00 | 2026-02-26 10:00:00 | 13541.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-03-18 09:35:00 | 14097.00 | 2026-03-18 09:55:00 | 14042.57 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest1 | 2026-03-20 10:55:00 | 13204.00 | 2026-03-20 12:05:00 | 13130.10 | PARTIAL | 0.50 | 0.56% |
| SELL | retest1 | 2026-03-20 10:55:00 | 13204.00 | 2026-03-20 13:15:00 | 13204.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-21 09:35:00 | 15051.00 | 2026-04-21 10:10:00 | 15124.14 | PARTIAL | 0.50 | 0.49% |
| BUY | retest1 | 2026-04-21 09:35:00 | 15051.00 | 2026-04-21 11:55:00 | 15051.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-22 09:30:00 | 15045.00 | 2026-04-22 09:40:00 | 15004.17 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2026-04-24 09:30:00 | 15684.00 | 2026-04-24 09:35:00 | 15759.11 | STOP_HIT | 1.00 | -0.48% |
| BUY | retest1 | 2026-04-29 11:10:00 | 15541.00 | 2026-04-29 11:45:00 | 15500.01 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2026-05-05 09:30:00 | 15544.00 | 2026-05-05 09:45:00 | 15494.99 | STOP_HIT | 1.00 | -0.32% |
