# Amara Raja Energy & Mobility Ltd. (ARE&M)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 890.85
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 4 |
| ALERT1 | 4 |
| ALERT2 | 4 |
| ALERT2_SKIP | 0 |
| ALERT3 | 29 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 2 |
| ENTRY2 | 40 |
| PARTIAL | 14 |
| TARGET_HIT | 13 |
| STOP_HIT | 29 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 56 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 37 / 19
- **Target hits / Stop hits / Partials:** 13 / 29 / 14
- **Avg / median % per leg:** 3.21% / 1.96%
- **Sum % (uncompounded):** 179.54%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 6 | 2 | 33.3% | 2 | 4 | 0 | 2.01% | 12.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 6 | 2 | 33.3% | 2 | 4 | 0 | 2.01% | 12.0% |
| SELL (all) | 50 | 35 | 70.0% | 11 | 25 | 14 | 3.35% | 167.5% |
| SELL @ 2nd Alert (retest1) | 4 | 4 | 100.0% | 2 | 0 | 2 | 7.50% | 30.0% |
| SELL @ 3rd Alert (retest2) | 46 | 31 | 67.4% | 9 | 25 | 12 | 2.99% | 137.5% |
| retest1 (combined) | 4 | 4 | 100.0% | 2 | 0 | 2 | 7.50% | 30.0% |
| retest2 (combined) | 52 | 33 | 63.5% | 11 | 29 | 12 | 2.88% | 149.5% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-09-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-20 12:15:00 | 1382.30 | 1464.98 | 1465.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-20 13:15:00 | 1375.00 | 1464.08 | 1464.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-11 10:15:00 | 1408.20 | 1406.49 | 1429.10 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-14 09:15:00 | 1399.70 | 1406.52 | 1428.56 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-14 10:00:00 | 1393.00 | 1406.39 | 1428.38 | SELL ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-21 09:15:00 | 1329.71 | 1397.22 | 1420.01 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-22 09:15:00 | 1323.35 | 1392.46 | 1416.81 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Target hit | 2024-10-22 12:15:00 | 1259.73 | 1388.94 | 1414.68 | Target hit (10%) qty=0.50 alert=retest1 |

### Cycle 2 — BUY (started 2025-09-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-05 13:15:00 | 1019.00 | 985.81 | 985.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-08 09:15:00 | 1027.85 | 986.87 | 986.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-22 12:15:00 | 1010.00 | 1010.54 | 1000.52 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-22 12:30:00 | 1010.00 | 1010.54 | 1000.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 14:15:00 | 1004.85 | 1010.44 | 1000.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-22 15:00:00 | 1004.85 | 1010.44 | 1000.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 09:15:00 | 1008.00 | 1010.37 | 1000.64 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-23 12:15:00 | 1014.95 | 1010.34 | 1000.71 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-24 10:15:00 | 996.00 | 1010.29 | 1000.98 | SL hit (close<static) qty=1.00 sl=997.95 alert=retest2 |

### Cycle 3 — SELL (started 2025-11-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-10 12:15:00 | 965.50 | 997.67 | 997.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-10 13:15:00 | 961.00 | 997.30 | 997.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-05 09:15:00 | 937.60 | 933.12 | 950.25 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-05 09:45:00 | 936.40 | 933.12 | 950.25 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 14:15:00 | 897.50 | 873.02 | 899.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-09 15:00:00 | 897.50 | 873.02 | 899.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 09:15:00 | 907.00 | 873.59 | 899.92 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-12 09:15:00 | 873.00 | 877.56 | 900.31 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-02 09:15:00 | 829.35 | 862.60 | 883.34 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-03-09 09:15:00 | 785.70 | 849.23 | 873.51 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 4 — BUY (started 2026-05-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-05 15:15:00 | 887.90 | 821.89 | 821.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-07 09:15:00 | 890.25 | 826.48 | 823.95 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-06-04 13:15:00 | 1128.75 | 2024-06-05 11:15:00 | 1228.87 | TARGET_HIT | 1.00 | 8.87% |
| BUY | retest2 | 2024-06-05 09:45:00 | 1117.15 | 2024-06-06 09:15:00 | 1241.62 | TARGET_HIT | 1.00 | 11.14% |
| SELL | retest1 | 2024-10-14 09:15:00 | 1399.70 | 2024-10-21 09:15:00 | 1329.71 | PARTIAL | 0.50 | 5.00% |
| SELL | retest1 | 2024-10-14 10:00:00 | 1393.00 | 2024-10-22 09:15:00 | 1323.35 | PARTIAL | 0.50 | 5.00% |
| SELL | retest1 | 2024-10-14 09:15:00 | 1399.70 | 2024-10-22 12:15:00 | 1259.73 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest1 | 2024-10-14 10:00:00 | 1393.00 | 2024-10-23 09:15:00 | 1253.70 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-12-04 11:45:00 | 1314.95 | 2024-12-05 12:15:00 | 1340.95 | STOP_HIT | 1.00 | -1.98% |
| SELL | retest2 | 2024-12-04 12:30:00 | 1312.65 | 2024-12-05 12:15:00 | 1340.95 | STOP_HIT | 1.00 | -2.16% |
| SELL | retest2 | 2024-12-04 14:00:00 | 1314.95 | 2024-12-05 12:15:00 | 1340.95 | STOP_HIT | 1.00 | -1.98% |
| SELL | retest2 | 2024-12-10 10:00:00 | 1314.50 | 2024-12-18 10:15:00 | 1248.77 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-10 10:00:00 | 1314.50 | 2024-12-23 11:15:00 | 1183.05 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-03-27 14:30:00 | 1027.85 | 2025-04-07 09:15:00 | 925.06 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-03-27 15:00:00 | 1022.90 | 2025-04-07 09:15:00 | 920.61 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-03-28 10:45:00 | 1027.20 | 2025-04-07 09:15:00 | 924.48 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-04-02 13:30:00 | 1024.00 | 2025-04-07 09:15:00 | 921.60 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-04-22 12:30:00 | 1022.30 | 2025-04-25 09:15:00 | 971.18 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-04-22 13:15:00 | 1019.50 | 2025-04-25 09:15:00 | 969.47 | PARTIAL | 0.50 | 4.91% |
| SELL | retest2 | 2025-04-23 09:45:00 | 1020.50 | 2025-04-30 15:15:00 | 968.52 | PARTIAL | 0.50 | 5.09% |
| SELL | retest2 | 2025-04-22 12:30:00 | 1022.30 | 2025-05-06 14:15:00 | 920.07 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-04-22 13:15:00 | 1019.50 | 2025-05-06 14:15:00 | 917.55 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-04-23 09:45:00 | 1020.50 | 2025-05-06 14:15:00 | 918.45 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-05-14 10:15:00 | 1021.00 | 2025-05-16 11:15:00 | 1045.75 | STOP_HIT | 1.00 | -2.42% |
| SELL | retest2 | 2025-05-21 09:15:00 | 1005.75 | 2025-05-21 10:15:00 | 1015.50 | STOP_HIT | 1.00 | -0.97% |
| SELL | retest2 | 2025-06-03 11:30:00 | 1005.60 | 2025-06-11 09:15:00 | 1017.20 | STOP_HIT | 1.00 | -1.15% |
| SELL | retest2 | 2025-06-03 13:00:00 | 1004.30 | 2025-06-11 09:15:00 | 1017.20 | STOP_HIT | 1.00 | -1.28% |
| SELL | retest2 | 2025-06-03 15:00:00 | 1004.70 | 2025-06-11 09:15:00 | 1017.20 | STOP_HIT | 1.00 | -1.24% |
| SELL | retest2 | 2025-06-12 13:30:00 | 1006.20 | 2025-07-02 14:15:00 | 957.12 | PARTIAL | 0.50 | 4.88% |
| SELL | retest2 | 2025-06-12 14:00:00 | 1003.00 | 2025-07-02 15:15:00 | 955.89 | PARTIAL | 0.50 | 4.70% |
| SELL | retest2 | 2025-06-12 15:00:00 | 1007.50 | 2025-07-02 15:15:00 | 952.85 | PARTIAL | 0.50 | 5.42% |
| SELL | retest2 | 2025-06-12 13:30:00 | 1006.20 | 2025-07-10 09:15:00 | 987.80 | STOP_HIT | 0.50 | 1.83% |
| SELL | retest2 | 2025-06-12 14:00:00 | 1003.00 | 2025-07-10 09:15:00 | 987.80 | STOP_HIT | 0.50 | 1.52% |
| SELL | retest2 | 2025-06-12 15:00:00 | 1007.50 | 2025-07-10 09:15:00 | 987.80 | STOP_HIT | 0.50 | 1.96% |
| SELL | retest2 | 2025-07-21 12:15:00 | 1006.50 | 2025-07-24 10:15:00 | 1023.45 | STOP_HIT | 1.00 | -1.68% |
| SELL | retest2 | 2025-07-25 14:00:00 | 993.45 | 2025-07-25 15:15:00 | 998.95 | STOP_HIT | 1.00 | -0.55% |
| SELL | retest2 | 2025-07-25 14:45:00 | 993.30 | 2025-07-25 15:15:00 | 998.95 | STOP_HIT | 1.00 | -0.57% |
| SELL | retest2 | 2025-07-28 09:45:00 | 991.70 | 2025-08-06 09:15:00 | 942.12 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-29 15:00:00 | 993.90 | 2025-08-06 09:15:00 | 944.20 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-30 10:15:00 | 985.80 | 2025-08-06 11:15:00 | 936.51 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-30 11:30:00 | 985.25 | 2025-08-06 11:15:00 | 935.99 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-28 09:45:00 | 991.70 | 2025-08-19 09:15:00 | 978.20 | STOP_HIT | 0.50 | 1.36% |
| SELL | retest2 | 2025-07-29 15:00:00 | 993.90 | 2025-08-19 09:15:00 | 978.20 | STOP_HIT | 0.50 | 1.58% |
| SELL | retest2 | 2025-07-30 10:15:00 | 985.80 | 2025-08-19 09:15:00 | 978.20 | STOP_HIT | 0.50 | 0.77% |
| SELL | retest2 | 2025-07-30 11:30:00 | 985.25 | 2025-08-19 09:15:00 | 978.20 | STOP_HIT | 0.50 | 0.72% |
| SELL | retest2 | 2025-08-19 12:30:00 | 983.00 | 2025-08-29 13:15:00 | 990.45 | STOP_HIT | 1.00 | -0.76% |
| SELL | retest2 | 2025-08-19 13:15:00 | 983.55 | 2025-08-29 13:15:00 | 990.45 | STOP_HIT | 1.00 | -0.70% |
| SELL | retest2 | 2025-08-26 09:15:00 | 980.00 | 2025-09-01 09:15:00 | 1010.95 | STOP_HIT | 1.00 | -3.16% |
| SELL | retest2 | 2025-08-26 13:45:00 | 984.60 | 2025-09-01 09:15:00 | 1010.95 | STOP_HIT | 1.00 | -2.68% |
| BUY | retest2 | 2025-09-23 12:15:00 | 1014.95 | 2025-09-24 10:15:00 | 996.00 | STOP_HIT | 1.00 | -1.87% |
| BUY | retest2 | 2025-10-30 09:15:00 | 1015.50 | 2025-11-04 12:15:00 | 995.00 | STOP_HIT | 1.00 | -2.02% |
| BUY | retest2 | 2025-10-30 14:45:00 | 1015.80 | 2025-11-04 12:15:00 | 995.00 | STOP_HIT | 1.00 | -2.05% |
| BUY | retest2 | 2025-10-31 14:00:00 | 1015.60 | 2025-11-04 12:15:00 | 995.00 | STOP_HIT | 1.00 | -2.03% |
| SELL | retest2 | 2026-02-12 09:15:00 | 873.00 | 2026-03-02 09:15:00 | 829.35 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-12 09:15:00 | 873.00 | 2026-03-09 09:15:00 | 785.70 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-04-24 09:15:00 | 888.55 | 2026-05-05 15:15:00 | 887.90 | STOP_HIT | 1.00 | 0.07% |
| SELL | retest2 | 2026-04-28 10:15:00 | 892.55 | 2026-05-05 15:15:00 | 887.90 | STOP_HIT | 1.00 | 0.52% |
| SELL | retest2 | 2026-04-28 11:15:00 | 892.00 | 2026-05-05 15:15:00 | 887.90 | STOP_HIT | 1.00 | 0.46% |
