# Carborundum Universal Ltd. (CARBORUNIV)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 1020.20
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
| ENTRY1 | 19 |
| ENTRY2 | 0 |
| PARTIAL | 5 |
| TARGET_HIT | 1 |
| STOP_HIT | 18 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 24 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 6 / 18
- **Target hits / Stop hits / Partials:** 1 / 18 / 5
- **Avg / median % per leg:** -0.07% / -0.26%
- **Sum % (uncompounded):** -1.59%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 17 | 4 | 23.5% | 1 | 13 | 3 | -0.07% | -1.1% |
| BUY @ 2nd Alert (retest1) | 17 | 4 | 23.5% | 1 | 13 | 3 | -0.07% | -1.1% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 7 | 2 | 28.6% | 0 | 5 | 2 | -0.07% | -0.5% |
| SELL @ 2nd Alert (retest1) | 7 | 2 | 28.6% | 0 | 5 | 2 | -0.07% | -0.5% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 24 | 6 | 25.0% | 1 | 18 | 5 | -0.07% | -1.6% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-18 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-18 10:25:00 | 835.80 | 832.33 | 0.00 | ORB-long ORB[827.20,833.85] vol=1.9x ATR=2.41 |
| Stop hit — per-position SL triggered | 2026-02-18 11:15:00 | 833.39 | 833.19 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2026-02-20 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-20 09:55:00 | 857.00 | 852.74 | 0.00 | ORB-long ORB[835.65,841.40] vol=6.4x ATR=4.39 |
| Stop hit — per-position SL triggered | 2026-02-20 10:10:00 | 852.61 | 853.13 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2026-02-25 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-25 11:00:00 | 838.00 | 840.18 | 0.00 | ORB-short ORB[838.25,844.55] vol=2.9x ATR=1.58 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-25 12:40:00 | 835.62 | 839.08 | 0.00 | T1 1.5R @ 835.62 |
| Stop hit — per-position SL triggered | 2026-02-25 13:15:00 | 838.00 | 838.63 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2026-02-26 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-26 10:30:00 | 836.90 | 834.97 | 0.00 | ORB-long ORB[831.40,836.20] vol=5.8x ATR=2.10 |
| Stop hit — per-position SL triggered | 2026-02-26 10:40:00 | 834.80 | 834.94 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2026-03-05 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-05 11:10:00 | 805.70 | 802.96 | 0.00 | ORB-long ORB[797.20,802.35] vol=2.0x ATR=2.08 |
| Stop hit — per-position SL triggered | 2026-03-05 11:25:00 | 803.62 | 803.10 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2026-03-10 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-10 10:50:00 | 805.00 | 799.60 | 0.00 | ORB-long ORB[794.30,800.55] vol=2.1x ATR=2.01 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-10 10:55:00 | 808.01 | 800.10 | 0.00 | T1 1.5R @ 808.01 |
| Stop hit — per-position SL triggered | 2026-03-10 11:25:00 | 805.00 | 801.13 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2026-03-17 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-17 10:40:00 | 749.25 | 757.15 | 0.00 | ORB-short ORB[753.05,764.05] vol=4.0x ATR=3.55 |
| Stop hit — per-position SL triggered | 2026-03-17 10:55:00 | 752.80 | 755.62 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2026-03-19 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-19 11:00:00 | 754.80 | 756.18 | 0.00 | ORB-short ORB[754.95,763.95] vol=2.4x ATR=1.87 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-19 12:05:00 | 751.99 | 755.42 | 0.00 | T1 1.5R @ 751.99 |
| Stop hit — per-position SL triggered | 2026-03-19 12:15:00 | 754.80 | 756.84 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2026-04-09 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-09 11:10:00 | 869.95 | 864.87 | 0.00 | ORB-long ORB[853.10,865.00] vol=1.7x ATR=2.16 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-09 12:30:00 | 873.19 | 867.09 | 0.00 | T1 1.5R @ 873.19 |
| Stop hit — per-position SL triggered | 2026-04-09 12:45:00 | 869.95 | 867.55 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2026-04-10 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-10 09:45:00 | 871.55 | 867.60 | 0.00 | ORB-long ORB[858.85,870.85] vol=1.8x ATR=2.91 |
| Stop hit — per-position SL triggered | 2026-04-10 10:00:00 | 868.64 | 867.73 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2026-04-16 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-16 09:45:00 | 881.85 | 891.35 | 0.00 | ORB-short ORB[890.45,900.00] vol=1.5x ATR=3.14 |
| Stop hit — per-position SL triggered | 2026-04-16 09:55:00 | 884.99 | 890.37 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2026-04-20 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-20 10:45:00 | 954.35 | 939.77 | 0.00 | ORB-long ORB[928.40,939.30] vol=1.8x ATR=4.80 |
| Stop hit — per-position SL triggered | 2026-04-20 13:25:00 | 949.55 | 946.54 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2026-04-23 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-23 09:30:00 | 950.00 | 947.35 | 0.00 | ORB-long ORB[938.60,946.80] vol=5.7x ATR=3.19 |
| Stop hit — per-position SL triggered | 2026-04-23 09:35:00 | 946.81 | 947.38 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2026-04-27 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-27 10:30:00 | 943.55 | 937.68 | 0.00 | ORB-long ORB[929.85,941.95] vol=2.9x ATR=3.32 |
| Stop hit — per-position SL triggered | 2026-04-27 10:55:00 | 940.23 | 938.86 | 0.00 | SL hit |

### Cycle 15 — SELL (started 2026-04-28 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-28 09:50:00 | 943.95 | 946.58 | 0.00 | ORB-short ORB[945.50,954.30] vol=2.4x ATR=2.67 |
| Stop hit — per-position SL triggered | 2026-04-28 09:55:00 | 946.62 | 946.65 | 0.00 | SL hit |

### Cycle 16 — BUY (started 2026-04-29 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-29 10:25:00 | 979.65 | 972.59 | 0.00 | ORB-long ORB[968.60,977.00] vol=2.0x ATR=3.38 |
| Stop hit — per-position SL triggered | 2026-04-29 10:40:00 | 976.27 | 972.91 | 0.00 | SL hit |

### Cycle 17 — BUY (started 2026-05-04 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-04 09:55:00 | 979.75 | 972.13 | 0.00 | ORB-long ORB[957.95,968.00] vol=1.8x ATR=3.85 |
| Stop hit — per-position SL triggered | 2026-05-04 10:00:00 | 975.90 | 972.32 | 0.00 | SL hit |

### Cycle 18 — BUY (started 2026-05-06 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-06 09:45:00 | 985.35 | 978.32 | 0.00 | ORB-long ORB[970.85,980.00] vol=1.8x ATR=3.35 |
| Stop hit — per-position SL triggered | 2026-05-06 10:55:00 | 982.00 | 983.29 | 0.00 | SL hit |

### Cycle 19 — BUY (started 2026-05-08 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-08 11:00:00 | 1010.80 | 1007.80 | 0.00 | ORB-long ORB[1001.30,1010.00] vol=7.8x ATR=3.37 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-08 11:40:00 | 1015.85 | 1008.26 | 0.00 | T1 1.5R @ 1015.85 |
| Target hit | 2026-05-08 15:10:00 | 1026.25 | 1026.69 | 0.00 | Trail-exit close<VWAP |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-02-18 10:25:00 | 835.80 | 2026-02-18 11:15:00 | 833.39 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2026-02-20 09:55:00 | 857.00 | 2026-02-20 10:10:00 | 852.61 | STOP_HIT | 1.00 | -0.51% |
| SELL | retest1 | 2026-02-25 11:00:00 | 838.00 | 2026-02-25 12:40:00 | 835.62 | PARTIAL | 0.50 | 0.28% |
| SELL | retest1 | 2026-02-25 11:00:00 | 838.00 | 2026-02-25 13:15:00 | 838.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-26 10:30:00 | 836.90 | 2026-02-26 10:40:00 | 834.80 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2026-03-05 11:10:00 | 805.70 | 2026-03-05 11:25:00 | 803.62 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2026-03-10 10:50:00 | 805.00 | 2026-03-10 10:55:00 | 808.01 | PARTIAL | 0.50 | 0.37% |
| BUY | retest1 | 2026-03-10 10:50:00 | 805.00 | 2026-03-10 11:25:00 | 805.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-03-17 10:40:00 | 749.25 | 2026-03-17 10:55:00 | 752.80 | STOP_HIT | 1.00 | -0.47% |
| SELL | retest1 | 2026-03-19 11:00:00 | 754.80 | 2026-03-19 12:05:00 | 751.99 | PARTIAL | 0.50 | 0.37% |
| SELL | retest1 | 2026-03-19 11:00:00 | 754.80 | 2026-03-19 12:15:00 | 754.80 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-09 11:10:00 | 869.95 | 2026-04-09 12:30:00 | 873.19 | PARTIAL | 0.50 | 0.37% |
| BUY | retest1 | 2026-04-09 11:10:00 | 869.95 | 2026-04-09 12:45:00 | 869.95 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-10 09:45:00 | 871.55 | 2026-04-10 10:00:00 | 868.64 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2026-04-16 09:45:00 | 881.85 | 2026-04-16 09:55:00 | 884.99 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2026-04-20 10:45:00 | 954.35 | 2026-04-20 13:25:00 | 949.55 | STOP_HIT | 1.00 | -0.50% |
| BUY | retest1 | 2026-04-23 09:30:00 | 950.00 | 2026-04-23 09:35:00 | 946.81 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2026-04-27 10:30:00 | 943.55 | 2026-04-27 10:55:00 | 940.23 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2026-04-28 09:50:00 | 943.95 | 2026-04-28 09:55:00 | 946.62 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2026-04-29 10:25:00 | 979.65 | 2026-04-29 10:40:00 | 976.27 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2026-05-04 09:55:00 | 979.75 | 2026-05-04 10:00:00 | 975.90 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest1 | 2026-05-06 09:45:00 | 985.35 | 2026-05-06 10:55:00 | 982.00 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2026-05-08 11:00:00 | 1010.80 | 2026-05-08 11:40:00 | 1015.85 | PARTIAL | 0.50 | 0.50% |
| BUY | retest1 | 2026-05-08 11:00:00 | 1010.80 | 2026-05-08 15:10:00 | 1026.25 | TARGET_HIT | 0.50 | 1.53% |
