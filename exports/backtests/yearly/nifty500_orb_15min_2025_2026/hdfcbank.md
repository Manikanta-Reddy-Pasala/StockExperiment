# HDFC Bank Ltd. (HDFCBANK)

## Backtest Summary

- **Window:** 2025-05-12 09:15:00 → 2026-05-08 15:25:00 (15388 bars)
- **Last close:** 781.25
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
| ENTRY1 | 63 |
| ENTRY2 | 0 |
| PARTIAL | 25 |
| TARGET_HIT | 11 |
| STOP_HIT | 52 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 88 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 36 / 52
- **Target hits / Stop hits / Partials:** 11 / 52 / 25
- **Avg / median % per leg:** 0.08% / 0.00%
- **Sum % (uncompounded):** 6.96%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 47 | 19 | 40.4% | 6 | 28 | 13 | 0.11% | 5.4% |
| BUY @ 2nd Alert (retest1) | 47 | 19 | 40.4% | 6 | 28 | 13 | 0.11% | 5.4% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 41 | 17 | 41.5% | 5 | 24 | 12 | 0.04% | 1.6% |
| SELL @ 2nd Alert (retest1) | 41 | 17 | 41.5% | 5 | 24 | 12 | 0.04% | 1.6% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 88 | 36 | 40.9% | 11 | 52 | 25 | 0.08% | 7.0% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-05-13 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-13 09:30:00 | 970.00 | 975.04 | 0.00 | ORB-short ORB[970.40,980.85] vol=1.6x ATR=2.25 |
| Stop hit — per-position SL triggered | 2025-05-13 09:35:00 | 972.25 | 974.79 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2025-05-15 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-15 09:35:00 | 951.10 | 953.07 | 0.00 | ORB-short ORB[951.20,958.55] vol=2.2x ATR=1.71 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-15 10:05:00 | 948.53 | 951.94 | 0.00 | T1 1.5R @ 948.53 |
| Stop hit — per-position SL triggered | 2025-05-15 10:10:00 | 951.10 | 951.90 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2025-05-29 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-29 09:35:00 | 962.00 | 964.90 | 0.00 | ORB-short ORB[963.70,970.40] vol=2.2x ATR=1.84 |
| Stop hit — per-position SL triggered | 2025-05-29 09:40:00 | 963.84 | 964.80 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2025-06-03 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-03 10:20:00 | 975.35 | 970.32 | 0.00 | ORB-long ORB[966.95,972.50] vol=1.7x ATR=2.13 |
| Stop hit — per-position SL triggered | 2025-06-03 10:35:00 | 973.22 | 970.81 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2025-06-06 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-06 10:25:00 | 981.50 | 975.75 | 0.00 | ORB-long ORB[971.35,976.20] vol=2.1x ATR=2.55 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-06 10:40:00 | 985.33 | 977.39 | 0.00 | T1 1.5R @ 985.33 |
| Target hit | 2025-06-06 15:20:00 | 989.20 | 988.06 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 6 — BUY (started 2025-06-16 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-16 11:05:00 | 968.20 | 964.38 | 0.00 | ORB-long ORB[960.05,964.25] vol=1.8x ATR=1.43 |
| Stop hit — per-position SL triggered | 2025-06-16 11:15:00 | 966.77 | 964.70 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2025-06-19 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-19 11:00:00 | 970.00 | 969.29 | 0.00 | ORB-long ORB[962.80,969.50] vol=1.6x ATR=1.42 |
| Stop hit — per-position SL triggered | 2025-06-19 11:20:00 | 968.58 | 969.28 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2025-06-20 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-20 10:55:00 | 975.75 | 974.53 | 0.00 | ORB-long ORB[965.80,973.00] vol=2.3x ATR=1.54 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-20 11:35:00 | 978.06 | 974.84 | 0.00 | T1 1.5R @ 978.06 |
| Stop hit — per-position SL triggered | 2025-06-20 12:00:00 | 975.75 | 975.02 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2025-06-23 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-23 11:00:00 | 973.90 | 975.58 | 0.00 | ORB-short ORB[974.20,978.95] vol=2.8x ATR=1.35 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-23 12:20:00 | 971.88 | 974.29 | 0.00 | T1 1.5R @ 971.88 |
| Stop hit — per-position SL triggered | 2025-06-23 13:20:00 | 973.90 | 974.06 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2025-06-25 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-25 11:05:00 | 988.45 | 986.94 | 0.00 | ORB-long ORB[981.95,987.50] vol=2.5x ATR=1.49 |
| Stop hit — per-position SL triggered | 2025-06-25 12:00:00 | 986.96 | 987.08 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2025-07-02 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-02 11:00:00 | 998.00 | 1000.67 | 0.00 | ORB-short ORB[998.80,1009.95] vol=1.6x ATR=1.45 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-02 11:35:00 | 995.82 | 999.75 | 0.00 | T1 1.5R @ 995.82 |
| Target hit | 2025-07-02 15:20:00 | 992.25 | 994.97 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 12 — BUY (started 2025-07-03 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-03 11:05:00 | 1003.55 | 997.92 | 0.00 | ORB-long ORB[993.75,997.45] vol=1.8x ATR=1.39 |
| Stop hit — per-position SL triggered | 2025-07-03 11:25:00 | 1002.16 | 998.50 | 0.00 | SL hit |

### Cycle 13 — SELL (started 2025-07-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-07 10:15:00 | 993.35 | 995.79 | 0.00 | ORB-short ORB[993.50,999.20] vol=3.0x ATR=1.56 |
| Stop hit — per-position SL triggered | 2025-07-07 10:50:00 | 994.91 | 995.23 | 0.00 | SL hit |

### Cycle 14 — SELL (started 2025-07-11 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-11 11:05:00 | 994.85 | 998.50 | 0.00 | ORB-short ORB[999.00,1002.40] vol=3.3x ATR=1.12 |
| Stop hit — per-position SL triggered | 2025-07-11 11:10:00 | 995.97 | 998.29 | 0.00 | SL hit |

### Cycle 15 — SELL (started 2025-07-14 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-14 10:45:00 | 987.45 | 988.97 | 0.00 | ORB-short ORB[988.15,994.35] vol=1.5x ATR=1.18 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-14 11:05:00 | 985.68 | 988.54 | 0.00 | T1 1.5R @ 985.68 |
| Stop hit — per-position SL triggered | 2025-07-14 11:30:00 | 987.45 | 988.28 | 0.00 | SL hit |

### Cycle 16 — BUY (started 2025-07-15 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-15 11:10:00 | 998.00 | 993.34 | 0.00 | ORB-long ORB[986.50,993.60] vol=2.0x ATR=1.36 |
| Stop hit — per-position SL triggered | 2025-07-15 12:20:00 | 996.64 | 994.44 | 0.00 | SL hit |

### Cycle 17 — SELL (started 2025-07-24 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-24 09:30:00 | 1008.55 | 1012.50 | 0.00 | ORB-short ORB[1009.55,1018.85] vol=2.3x ATR=1.53 |
| Stop hit — per-position SL triggered | 2025-07-24 10:25:00 | 1010.08 | 1011.62 | 0.00 | SL hit |

### Cycle 18 — BUY (started 2025-07-28 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-28 11:00:00 | 1009.05 | 1006.88 | 0.00 | ORB-long ORB[1000.25,1006.70] vol=1.8x ATR=1.54 |
| Stop hit — per-position SL triggered | 2025-07-28 11:05:00 | 1007.51 | 1006.96 | 0.00 | SL hit |

### Cycle 19 — BUY (started 2025-07-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-29 11:15:00 | 1008.20 | 1003.38 | 0.00 | ORB-long ORB[999.70,1002.95] vol=2.2x ATR=1.37 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-29 12:20:00 | 1010.26 | 1004.60 | 0.00 | T1 1.5R @ 1010.26 |
| Stop hit — per-position SL triggered | 2025-07-29 14:35:00 | 1008.20 | 1007.30 | 0.00 | SL hit |

### Cycle 20 — SELL (started 2025-07-31 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-31 11:00:00 | 1005.55 | 1008.02 | 0.00 | ORB-short ORB[1006.05,1012.50] vol=3.2x ATR=1.43 |
| Stop hit — per-position SL triggered | 2025-07-31 11:05:00 | 1006.98 | 1007.94 | 0.00 | SL hit |

### Cycle 21 — SELL (started 2025-08-04 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-04 10:50:00 | 1000.85 | 1003.33 | 0.00 | ORB-short ORB[1003.95,1009.25] vol=2.0x ATR=1.52 |
| Stop hit — per-position SL triggered | 2025-08-04 11:20:00 | 1002.37 | 1002.69 | 0.00 | SL hit |

### Cycle 22 — SELL (started 2025-08-07 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-07 10:55:00 | 989.00 | 990.87 | 0.00 | ORB-short ORB[990.00,994.40] vol=1.6x ATR=1.31 |
| Stop hit — per-position SL triggered | 2025-08-07 11:15:00 | 990.31 | 990.64 | 0.00 | SL hit |

### Cycle 23 — SELL (started 2025-08-13 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-13 11:10:00 | 985.65 | 987.82 | 0.00 | ORB-short ORB[986.00,997.00] vol=1.6x ATR=1.37 |
| Stop hit — per-position SL triggered | 2025-08-13 12:20:00 | 987.02 | 987.26 | 0.00 | SL hit |

### Cycle 24 — SELL (started 2025-08-20 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-20 10:50:00 | 990.50 | 992.05 | 0.00 | ORB-short ORB[990.80,995.00] vol=2.0x ATR=1.17 |
| Stop hit — per-position SL triggered | 2025-08-20 11:15:00 | 991.67 | 991.80 | 0.00 | SL hit |

### Cycle 25 — SELL (started 2025-08-22 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-22 09:30:00 | 987.35 | 990.69 | 0.00 | ORB-short ORB[988.60,998.00] vol=1.8x ATR=1.56 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-22 10:00:00 | 985.01 | 989.27 | 0.00 | T1 1.5R @ 985.01 |
| Target hit | 2025-08-22 13:15:00 | 985.55 | 984.91 | 0.00 | Trail-exit close>VWAP |

### Cycle 26 — SELL (started 2025-09-05 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-05 10:55:00 | 955.60 | 960.47 | 0.00 | ORB-short ORB[958.90,967.50] vol=2.7x ATR=1.75 |
| Stop hit — per-position SL triggered | 2025-09-05 11:15:00 | 957.35 | 960.16 | 0.00 | SL hit |

### Cycle 27 — SELL (started 2025-09-09 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-09 11:10:00 | 967.85 | 970.81 | 0.00 | ORB-short ORB[969.15,974.40] vol=2.2x ATR=1.30 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-09 12:05:00 | 965.89 | 969.92 | 0.00 | T1 1.5R @ 965.89 |
| Stop hit — per-position SL triggered | 2025-09-09 12:15:00 | 967.85 | 969.84 | 0.00 | SL hit |

### Cycle 28 — BUY (started 2025-09-16 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-16 10:50:00 | 967.00 | 965.64 | 0.00 | ORB-long ORB[961.20,965.40] vol=3.5x ATR=1.08 |
| Stop hit — per-position SL triggered | 2025-09-16 11:40:00 | 965.92 | 965.80 | 0.00 | SL hit |

### Cycle 29 — SELL (started 2025-09-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-23 11:15:00 | 956.10 | 962.99 | 0.00 | ORB-short ORB[965.00,968.20] vol=1.5x ATR=1.37 |
| Stop hit — per-position SL triggered | 2025-09-23 11:30:00 | 957.47 | 962.49 | 0.00 | SL hit |

### Cycle 30 — SELL (started 2025-09-24 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-24 10:50:00 | 945.65 | 950.46 | 0.00 | ORB-short ORB[952.20,956.45] vol=1.6x ATR=1.58 |
| Stop hit — per-position SL triggered | 2025-09-24 10:55:00 | 947.23 | 950.36 | 0.00 | SL hit |

### Cycle 31 — BUY (started 2025-09-29 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-29 09:55:00 | 950.30 | 945.87 | 0.00 | ORB-long ORB[939.10,947.25] vol=3.9x ATR=1.92 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-29 10:20:00 | 953.18 | 946.41 | 0.00 | T1 1.5R @ 953.18 |
| Stop hit — per-position SL triggered | 2025-09-29 10:55:00 | 950.30 | 947.38 | 0.00 | SL hit |

### Cycle 32 — BUY (started 2025-10-06 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-06 10:55:00 | 975.85 | 971.00 | 0.00 | ORB-long ORB[962.50,975.75] vol=1.7x ATR=2.04 |
| Stop hit — per-position SL triggered | 2025-10-06 11:50:00 | 973.81 | 972.12 | 0.00 | SL hit |

### Cycle 33 — BUY (started 2025-10-07 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-07 09:40:00 | 980.05 | 974.64 | 0.00 | ORB-long ORB[971.00,976.55] vol=2.3x ATR=1.95 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-07 09:55:00 | 982.97 | 977.14 | 0.00 | T1 1.5R @ 982.97 |
| Target hit | 2025-10-07 15:05:00 | 983.25 | 983.78 | 0.00 | Trail-exit close<VWAP |

### Cycle 34 — SELL (started 2025-10-08 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-08 11:05:00 | 975.50 | 980.29 | 0.00 | ORB-short ORB[976.00,985.00] vol=2.6x ATR=1.57 |
| Stop hit — per-position SL triggered | 2025-10-08 11:25:00 | 977.07 | 979.70 | 0.00 | SL hit |

### Cycle 35 — BUY (started 2025-10-09 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-09 11:05:00 | 977.70 | 974.32 | 0.00 | ORB-long ORB[971.50,977.00] vol=1.7x ATR=1.77 |
| Stop hit — per-position SL triggered | 2025-10-09 11:40:00 | 975.93 | 975.04 | 0.00 | SL hit |

### Cycle 36 — BUY (started 2025-10-15 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-15 10:30:00 | 983.30 | 977.63 | 0.00 | ORB-long ORB[973.35,979.80] vol=1.7x ATR=1.82 |
| Stop hit — per-position SL triggered | 2025-10-15 11:50:00 | 981.48 | 979.11 | 0.00 | SL hit |

### Cycle 37 — BUY (started 2025-10-17 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-17 10:35:00 | 999.55 | 995.03 | 0.00 | ORB-long ORB[986.00,994.90] vol=1.8x ATR=1.95 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-17 10:50:00 | 1002.47 | 995.96 | 0.00 | T1 1.5R @ 1002.47 |
| Target hit | 2025-10-17 13:45:00 | 1000.95 | 1001.47 | 0.00 | Trail-exit close<VWAP |

### Cycle 38 — SELL (started 2025-10-24 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-24 11:05:00 | 1001.20 | 1005.46 | 0.00 | ORB-short ORB[1008.25,1011.95] vol=1.8x ATR=1.69 |
| Stop hit — per-position SL triggered | 2025-10-24 11:10:00 | 1002.89 | 1005.09 | 0.00 | SL hit |

### Cycle 39 — BUY (started 2025-10-27 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-27 11:00:00 | 1005.40 | 1004.62 | 0.00 | ORB-long ORB[996.20,1002.25] vol=1.5x ATR=1.68 |
| Stop hit — per-position SL triggered | 2025-10-27 11:40:00 | 1003.72 | 1004.75 | 0.00 | SL hit |

### Cycle 40 — SELL (started 2025-10-31 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-31 10:45:00 | 988.85 | 996.80 | 0.00 | ORB-short ORB[994.00,1002.80] vol=2.1x ATR=1.93 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-31 11:00:00 | 985.95 | 993.56 | 0.00 | T1 1.5R @ 985.95 |
| Stop hit — per-position SL triggered | 2025-10-31 11:55:00 | 988.85 | 991.83 | 0.00 | SL hit |

### Cycle 41 — BUY (started 2025-11-03 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-03 11:00:00 | 991.10 | 988.18 | 0.00 | ORB-long ORB[983.30,988.15] vol=1.6x ATR=1.86 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-03 11:55:00 | 993.89 | 989.70 | 0.00 | T1 1.5R @ 993.89 |
| Stop hit — per-position SL triggered | 2025-11-03 12:10:00 | 991.10 | 989.80 | 0.00 | SL hit |

### Cycle 42 — BUY (started 2025-11-04 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-04 10:35:00 | 994.50 | 990.90 | 0.00 | ORB-long ORB[987.00,991.70] vol=2.0x ATR=1.65 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-04 11:00:00 | 996.97 | 991.81 | 0.00 | T1 1.5R @ 996.97 |
| Stop hit — per-position SL triggered | 2025-11-04 11:05:00 | 994.50 | 991.94 | 0.00 | SL hit |

### Cycle 43 — BUY (started 2025-11-10 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-10 11:00:00 | 990.00 | 985.23 | 0.00 | ORB-long ORB[980.20,985.75] vol=2.5x ATR=1.76 |
| Stop hit — per-position SL triggered | 2025-11-10 13:10:00 | 988.24 | 986.77 | 0.00 | SL hit |

### Cycle 44 — BUY (started 2025-11-13 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-13 11:05:00 | 987.90 | 985.30 | 0.00 | ORB-long ORB[982.60,987.50] vol=1.5x ATR=1.28 |
| Stop hit — per-position SL triggered | 2025-11-13 12:35:00 | 986.62 | 986.07 | 0.00 | SL hit |

### Cycle 45 — BUY (started 2025-11-20 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-20 11:05:00 | 997.15 | 995.53 | 0.00 | ORB-long ORB[992.45,997.00] vol=1.9x ATR=1.28 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-20 12:20:00 | 999.07 | 996.45 | 0.00 | T1 1.5R @ 999.07 |
| Target hit | 2025-11-20 15:20:00 | 1008.90 | 1003.95 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 46 — BUY (started 2025-11-26 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-26 11:00:00 | 1000.00 | 994.22 | 0.00 | ORB-long ORB[981.30,992.10] vol=2.4x ATR=1.87 |
| Stop hit — per-position SL triggered | 2025-11-26 11:05:00 | 998.13 | 994.39 | 0.00 | SL hit |

### Cycle 47 — BUY (started 2025-11-27 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-27 10:55:00 | 1015.30 | 1011.03 | 0.00 | ORB-long ORB[1001.00,1010.65] vol=3.3x ATR=1.87 |
| Stop hit — per-position SL triggered | 2025-11-27 11:45:00 | 1013.43 | 1011.61 | 0.00 | SL hit |

### Cycle 48 — BUY (started 2025-11-28 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-28 10:50:00 | 1011.10 | 1007.91 | 0.00 | ORB-long ORB[1004.25,1008.00] vol=2.0x ATR=1.57 |
| Stop hit — per-position SL triggered | 2025-11-28 11:20:00 | 1009.53 | 1008.94 | 0.00 | SL hit |

### Cycle 49 — BUY (started 2025-12-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-04 11:15:00 | 1002.70 | 1002.05 | 0.00 | ORB-long ORB[997.00,1002.40] vol=2.4x ATR=1.19 |
| Stop hit — per-position SL triggered | 2025-12-04 11:30:00 | 1001.51 | 1002.10 | 0.00 | SL hit |

### Cycle 50 — BUY (started 2025-12-05 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-05 10:45:00 | 1007.20 | 998.91 | 0.00 | ORB-long ORB[990.20,997.00] vol=1.8x ATR=2.41 |
| Stop hit — per-position SL triggered | 2025-12-05 11:10:00 | 1004.79 | 999.90 | 0.00 | SL hit |

### Cycle 51 — SELL (started 2025-12-10 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-10 11:10:00 | 993.20 | 995.56 | 0.00 | ORB-short ORB[993.50,998.50] vol=2.9x ATR=1.46 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-10 11:25:00 | 991.00 | 995.02 | 0.00 | T1 1.5R @ 991.00 |
| Target hit | 2025-12-10 15:20:00 | 990.40 | 990.60 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 52 — BUY (started 2025-12-23 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-23 11:10:00 | 993.40 | 992.04 | 0.00 | ORB-long ORB[987.70,992.30] vol=3.7x ATR=1.07 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-23 13:25:00 | 995.00 | 992.74 | 0.00 | T1 1.5R @ 995.00 |
| Target hit | 2025-12-23 15:20:00 | 996.60 | 994.91 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 53 — SELL (started 2025-12-29 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-29 11:00:00 | 990.80 | 993.77 | 0.00 | ORB-short ORB[993.10,997.40] vol=2.5x ATR=1.29 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-29 11:15:00 | 988.87 | 993.41 | 0.00 | T1 1.5R @ 988.87 |
| Stop hit — per-position SL triggered | 2025-12-29 13:05:00 | 990.80 | 991.68 | 0.00 | SL hit |

### Cycle 54 — BUY (started 2026-01-02 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-02 09:40:00 | 998.00 | 995.19 | 0.00 | ORB-long ORB[988.80,996.95] vol=1.6x ATR=1.70 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-02 09:50:00 | 1000.55 | 996.17 | 0.00 | T1 1.5R @ 1000.55 |
| Stop hit — per-position SL triggered | 2026-01-02 09:55:00 | 998.00 | 996.31 | 0.00 | SL hit |

### Cycle 55 — BUY (started 2026-02-17 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-17 11:00:00 | 925.70 | 923.59 | 0.00 | ORB-long ORB[919.55,925.30] vol=3.0x ATR=1.58 |
| Stop hit — per-position SL triggered | 2026-02-17 12:30:00 | 924.12 | 924.74 | 0.00 | SL hit |

### Cycle 56 — SELL (started 2026-02-24 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-24 11:05:00 | 919.65 | 922.44 | 0.00 | ORB-short ORB[921.50,927.95] vol=1.8x ATR=1.45 |
| Stop hit — per-position SL triggered | 2026-02-24 11:20:00 | 921.10 | 922.28 | 0.00 | SL hit |

### Cycle 57 — SELL (started 2026-02-26 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-26 11:00:00 | 905.55 | 907.58 | 0.00 | ORB-short ORB[905.95,912.00] vol=1.6x ATR=1.33 |
| Stop hit — per-position SL triggered | 2026-02-26 11:25:00 | 906.88 | 907.22 | 0.00 | SL hit |

### Cycle 58 — SELL (started 2026-02-27 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-27 11:10:00 | 890.50 | 892.88 | 0.00 | ORB-short ORB[892.50,898.95] vol=4.8x ATR=1.33 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-27 11:40:00 | 888.51 | 892.36 | 0.00 | T1 1.5R @ 888.51 |
| Stop hit — per-position SL triggered | 2026-02-27 12:00:00 | 890.50 | 892.14 | 0.00 | SL hit |

### Cycle 59 — BUY (started 2026-03-05 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-05 10:55:00 | 874.50 | 869.31 | 0.00 | ORB-long ORB[863.50,873.85] vol=2.6x ATR=2.07 |
| Stop hit — per-position SL triggered | 2026-03-05 11:30:00 | 872.43 | 870.06 | 0.00 | SL hit |

### Cycle 60 — SELL (started 2026-03-11 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-11 10:50:00 | 836.20 | 841.39 | 0.00 | ORB-short ORB[840.60,848.85] vol=1.6x ATR=1.69 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-11 11:15:00 | 833.67 | 840.22 | 0.00 | T1 1.5R @ 833.67 |
| Target hit | 2026-03-11 15:05:00 | 834.80 | 834.08 | 0.00 | Trail-exit close>VWAP |

### Cycle 61 — BUY (started 2026-03-12 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-12 10:20:00 | 831.90 | 830.30 | 0.00 | ORB-long ORB[820.10,830.90] vol=3.4x ATR=2.42 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-12 12:05:00 | 835.53 | 831.33 | 0.00 | T1 1.5R @ 835.53 |
| Stop hit — per-position SL triggered | 2026-03-12 14:40:00 | 831.90 | 832.32 | 0.00 | SL hit |

### Cycle 62 — BUY (started 2026-04-02 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-02 11:10:00 | 731.65 | 730.53 | 0.00 | ORB-long ORB[726.65,731.00] vol=1.7x ATR=1.84 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-02 11:55:00 | 734.41 | 730.99 | 0.00 | T1 1.5R @ 734.41 |
| Target hit | 2026-04-02 15:20:00 | 750.00 | 740.51 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 63 — SELL (started 2026-05-08 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-08 11:05:00 | 782.95 | 784.83 | 0.00 | ORB-short ORB[784.00,788.75] vol=2.7x ATR=1.43 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-08 11:30:00 | 780.81 | 784.55 | 0.00 | T1 1.5R @ 780.81 |
| Target hit | 2026-05-08 15:10:00 | 781.45 | 781.44 | 0.00 | Trail-exit close>VWAP |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2025-05-13 09:30:00 | 970.00 | 2025-05-13 09:35:00 | 972.25 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2025-05-15 09:35:00 | 951.10 | 2025-05-15 10:05:00 | 948.53 | PARTIAL | 0.50 | 0.27% |
| SELL | retest1 | 2025-05-15 09:35:00 | 951.10 | 2025-05-15 10:10:00 | 951.10 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-05-29 09:35:00 | 962.00 | 2025-05-29 09:40:00 | 963.84 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest1 | 2025-06-03 10:20:00 | 975.35 | 2025-06-03 10:35:00 | 973.22 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2025-06-06 10:25:00 | 981.50 | 2025-06-06 10:40:00 | 985.33 | PARTIAL | 0.50 | 0.39% |
| BUY | retest1 | 2025-06-06 10:25:00 | 981.50 | 2025-06-06 15:20:00 | 989.20 | TARGET_HIT | 0.50 | 0.78% |
| BUY | retest1 | 2025-06-16 11:05:00 | 968.20 | 2025-06-16 11:15:00 | 966.77 | STOP_HIT | 1.00 | -0.15% |
| BUY | retest1 | 2025-06-19 11:00:00 | 970.00 | 2025-06-19 11:20:00 | 968.58 | STOP_HIT | 1.00 | -0.15% |
| BUY | retest1 | 2025-06-20 10:55:00 | 975.75 | 2025-06-20 11:35:00 | 978.06 | PARTIAL | 0.50 | 0.24% |
| BUY | retest1 | 2025-06-20 10:55:00 | 975.75 | 2025-06-20 12:00:00 | 975.75 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-06-23 11:00:00 | 973.90 | 2025-06-23 12:20:00 | 971.88 | PARTIAL | 0.50 | 0.21% |
| SELL | retest1 | 2025-06-23 11:00:00 | 973.90 | 2025-06-23 13:20:00 | 973.90 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-06-25 11:05:00 | 988.45 | 2025-06-25 12:00:00 | 986.96 | STOP_HIT | 1.00 | -0.15% |
| SELL | retest1 | 2025-07-02 11:00:00 | 998.00 | 2025-07-02 11:35:00 | 995.82 | PARTIAL | 0.50 | 0.22% |
| SELL | retest1 | 2025-07-02 11:00:00 | 998.00 | 2025-07-02 15:20:00 | 992.25 | TARGET_HIT | 0.50 | 0.58% |
| BUY | retest1 | 2025-07-03 11:05:00 | 1003.55 | 2025-07-03 11:25:00 | 1002.16 | STOP_HIT | 1.00 | -0.14% |
| SELL | retest1 | 2025-07-07 10:15:00 | 993.35 | 2025-07-07 10:50:00 | 994.91 | STOP_HIT | 1.00 | -0.16% |
| SELL | retest1 | 2025-07-11 11:05:00 | 994.85 | 2025-07-11 11:10:00 | 995.97 | STOP_HIT | 1.00 | -0.11% |
| SELL | retest1 | 2025-07-14 10:45:00 | 987.45 | 2025-07-14 11:05:00 | 985.68 | PARTIAL | 0.50 | 0.18% |
| SELL | retest1 | 2025-07-14 10:45:00 | 987.45 | 2025-07-14 11:30:00 | 987.45 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-07-15 11:10:00 | 998.00 | 2025-07-15 12:20:00 | 996.64 | STOP_HIT | 1.00 | -0.14% |
| SELL | retest1 | 2025-07-24 09:30:00 | 1008.55 | 2025-07-24 10:25:00 | 1010.08 | STOP_HIT | 1.00 | -0.15% |
| BUY | retest1 | 2025-07-28 11:00:00 | 1009.05 | 2025-07-28 11:05:00 | 1007.51 | STOP_HIT | 1.00 | -0.15% |
| BUY | retest1 | 2025-07-29 11:15:00 | 1008.20 | 2025-07-29 12:20:00 | 1010.26 | PARTIAL | 0.50 | 0.20% |
| BUY | retest1 | 2025-07-29 11:15:00 | 1008.20 | 2025-07-29 14:35:00 | 1008.20 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-07-31 11:00:00 | 1005.55 | 2025-07-31 11:05:00 | 1006.98 | STOP_HIT | 1.00 | -0.14% |
| SELL | retest1 | 2025-08-04 10:50:00 | 1000.85 | 2025-08-04 11:20:00 | 1002.37 | STOP_HIT | 1.00 | -0.15% |
| SELL | retest1 | 2025-08-07 10:55:00 | 989.00 | 2025-08-07 11:15:00 | 990.31 | STOP_HIT | 1.00 | -0.13% |
| SELL | retest1 | 2025-08-13 11:10:00 | 985.65 | 2025-08-13 12:20:00 | 987.02 | STOP_HIT | 1.00 | -0.14% |
| SELL | retest1 | 2025-08-20 10:50:00 | 990.50 | 2025-08-20 11:15:00 | 991.67 | STOP_HIT | 1.00 | -0.12% |
| SELL | retest1 | 2025-08-22 09:30:00 | 987.35 | 2025-08-22 10:00:00 | 985.01 | PARTIAL | 0.50 | 0.24% |
| SELL | retest1 | 2025-08-22 09:30:00 | 987.35 | 2025-08-22 13:15:00 | 985.55 | TARGET_HIT | 0.50 | 0.18% |
| SELL | retest1 | 2025-09-05 10:55:00 | 955.60 | 2025-09-05 11:15:00 | 957.35 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest1 | 2025-09-09 11:10:00 | 967.85 | 2025-09-09 12:05:00 | 965.89 | PARTIAL | 0.50 | 0.20% |
| SELL | retest1 | 2025-09-09 11:10:00 | 967.85 | 2025-09-09 12:15:00 | 967.85 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-09-16 10:50:00 | 967.00 | 2025-09-16 11:40:00 | 965.92 | STOP_HIT | 1.00 | -0.11% |
| SELL | retest1 | 2025-09-23 11:15:00 | 956.10 | 2025-09-23 11:30:00 | 957.47 | STOP_HIT | 1.00 | -0.14% |
| SELL | retest1 | 2025-09-24 10:50:00 | 945.65 | 2025-09-24 10:55:00 | 947.23 | STOP_HIT | 1.00 | -0.17% |
| BUY | retest1 | 2025-09-29 09:55:00 | 950.30 | 2025-09-29 10:20:00 | 953.18 | PARTIAL | 0.50 | 0.30% |
| BUY | retest1 | 2025-09-29 09:55:00 | 950.30 | 2025-09-29 10:55:00 | 950.30 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-10-06 10:55:00 | 975.85 | 2025-10-06 11:50:00 | 973.81 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2025-10-07 09:40:00 | 980.05 | 2025-10-07 09:55:00 | 982.97 | PARTIAL | 0.50 | 0.30% |
| BUY | retest1 | 2025-10-07 09:40:00 | 980.05 | 2025-10-07 15:05:00 | 983.25 | TARGET_HIT | 0.50 | 0.33% |
| SELL | retest1 | 2025-10-08 11:05:00 | 975.50 | 2025-10-08 11:25:00 | 977.07 | STOP_HIT | 1.00 | -0.16% |
| BUY | retest1 | 2025-10-09 11:05:00 | 977.70 | 2025-10-09 11:40:00 | 975.93 | STOP_HIT | 1.00 | -0.18% |
| BUY | retest1 | 2025-10-15 10:30:00 | 983.30 | 2025-10-15 11:50:00 | 981.48 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest1 | 2025-10-17 10:35:00 | 999.55 | 2025-10-17 10:50:00 | 1002.47 | PARTIAL | 0.50 | 0.29% |
| BUY | retest1 | 2025-10-17 10:35:00 | 999.55 | 2025-10-17 13:45:00 | 1000.95 | TARGET_HIT | 0.50 | 0.14% |
| SELL | retest1 | 2025-10-24 11:05:00 | 1001.20 | 2025-10-24 11:10:00 | 1002.89 | STOP_HIT | 1.00 | -0.17% |
| BUY | retest1 | 2025-10-27 11:00:00 | 1005.40 | 2025-10-27 11:40:00 | 1003.72 | STOP_HIT | 1.00 | -0.17% |
| SELL | retest1 | 2025-10-31 10:45:00 | 988.85 | 2025-10-31 11:00:00 | 985.95 | PARTIAL | 0.50 | 0.29% |
| SELL | retest1 | 2025-10-31 10:45:00 | 988.85 | 2025-10-31 11:55:00 | 988.85 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-11-03 11:00:00 | 991.10 | 2025-11-03 11:55:00 | 993.89 | PARTIAL | 0.50 | 0.28% |
| BUY | retest1 | 2025-11-03 11:00:00 | 991.10 | 2025-11-03 12:10:00 | 991.10 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-11-04 10:35:00 | 994.50 | 2025-11-04 11:00:00 | 996.97 | PARTIAL | 0.50 | 0.25% |
| BUY | retest1 | 2025-11-04 10:35:00 | 994.50 | 2025-11-04 11:05:00 | 994.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-11-10 11:00:00 | 990.00 | 2025-11-10 13:10:00 | 988.24 | STOP_HIT | 1.00 | -0.18% |
| BUY | retest1 | 2025-11-13 11:05:00 | 987.90 | 2025-11-13 12:35:00 | 986.62 | STOP_HIT | 1.00 | -0.13% |
| BUY | retest1 | 2025-11-20 11:05:00 | 997.15 | 2025-11-20 12:20:00 | 999.07 | PARTIAL | 0.50 | 0.19% |
| BUY | retest1 | 2025-11-20 11:05:00 | 997.15 | 2025-11-20 15:20:00 | 1008.90 | TARGET_HIT | 0.50 | 1.18% |
| BUY | retest1 | 2025-11-26 11:00:00 | 1000.00 | 2025-11-26 11:05:00 | 998.13 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest1 | 2025-11-27 10:55:00 | 1015.30 | 2025-11-27 11:45:00 | 1013.43 | STOP_HIT | 1.00 | -0.18% |
| BUY | retest1 | 2025-11-28 10:50:00 | 1011.10 | 2025-11-28 11:20:00 | 1009.53 | STOP_HIT | 1.00 | -0.16% |
| BUY | retest1 | 2025-12-04 11:15:00 | 1002.70 | 2025-12-04 11:30:00 | 1001.51 | STOP_HIT | 1.00 | -0.12% |
| BUY | retest1 | 2025-12-05 10:45:00 | 1007.20 | 2025-12-05 11:10:00 | 1004.79 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2025-12-10 11:10:00 | 993.20 | 2025-12-10 11:25:00 | 991.00 | PARTIAL | 0.50 | 0.22% |
| SELL | retest1 | 2025-12-10 11:10:00 | 993.20 | 2025-12-10 15:20:00 | 990.40 | TARGET_HIT | 0.50 | 0.28% |
| BUY | retest1 | 2025-12-23 11:10:00 | 993.40 | 2025-12-23 13:25:00 | 995.00 | PARTIAL | 0.50 | 0.16% |
| BUY | retest1 | 2025-12-23 11:10:00 | 993.40 | 2025-12-23 15:20:00 | 996.60 | TARGET_HIT | 0.50 | 0.32% |
| SELL | retest1 | 2025-12-29 11:00:00 | 990.80 | 2025-12-29 11:15:00 | 988.87 | PARTIAL | 0.50 | 0.20% |
| SELL | retest1 | 2025-12-29 11:00:00 | 990.80 | 2025-12-29 13:05:00 | 990.80 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-01-02 09:40:00 | 998.00 | 2026-01-02 09:50:00 | 1000.55 | PARTIAL | 0.50 | 0.26% |
| BUY | retest1 | 2026-01-02 09:40:00 | 998.00 | 2026-01-02 09:55:00 | 998.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-17 11:00:00 | 925.70 | 2026-02-17 12:30:00 | 924.12 | STOP_HIT | 1.00 | -0.17% |
| SELL | retest1 | 2026-02-24 11:05:00 | 919.65 | 2026-02-24 11:20:00 | 921.10 | STOP_HIT | 1.00 | -0.16% |
| SELL | retest1 | 2026-02-26 11:00:00 | 905.55 | 2026-02-26 11:25:00 | 906.88 | STOP_HIT | 1.00 | -0.15% |
| SELL | retest1 | 2026-02-27 11:10:00 | 890.50 | 2026-02-27 11:40:00 | 888.51 | PARTIAL | 0.50 | 0.22% |
| SELL | retest1 | 2026-02-27 11:10:00 | 890.50 | 2026-02-27 12:00:00 | 890.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-03-05 10:55:00 | 874.50 | 2026-03-05 11:30:00 | 872.43 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2026-03-11 10:50:00 | 836.20 | 2026-03-11 11:15:00 | 833.67 | PARTIAL | 0.50 | 0.30% |
| SELL | retest1 | 2026-03-11 10:50:00 | 836.20 | 2026-03-11 15:05:00 | 834.80 | TARGET_HIT | 0.50 | 0.17% |
| BUY | retest1 | 2026-03-12 10:20:00 | 831.90 | 2026-03-12 12:05:00 | 835.53 | PARTIAL | 0.50 | 0.44% |
| BUY | retest1 | 2026-03-12 10:20:00 | 831.90 | 2026-03-12 14:40:00 | 831.90 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-02 11:10:00 | 731.65 | 2026-04-02 11:55:00 | 734.41 | PARTIAL | 0.50 | 0.38% |
| BUY | retest1 | 2026-04-02 11:10:00 | 731.65 | 2026-04-02 15:20:00 | 750.00 | TARGET_HIT | 0.50 | 2.51% |
| SELL | retest1 | 2026-05-08 11:05:00 | 782.95 | 2026-05-08 11:30:00 | 780.81 | PARTIAL | 0.50 | 0.27% |
| SELL | retest1 | 2026-05-08 11:05:00 | 782.95 | 2026-05-08 15:10:00 | 781.45 | TARGET_HIT | 0.50 | 0.19% |
