# Finolex Cables Ltd. (FINCABLES)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 1144.95
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 9 |
| ALERT1 | 9 |
| ALERT2 | 9 |
| ALERT2_SKIP | 3 |
| ALERT3 | 52 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 49 |
| PARTIAL | 11 |
| TARGET_HIT | 14 |
| STOP_HIT | 35 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 60 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 27 / 33
- **Target hits / Stop hits / Partials:** 14 / 35 / 11
- **Avg / median % per leg:** 2.31% / -0.93%
- **Sum % (uncompounded):** 138.78%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 30 | 7 | 23.3% | 7 | 23 | 0 | 0.85% | 25.4% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 30 | 7 | 23.3% | 7 | 23 | 0 | 0.85% | 25.4% |
| SELL (all) | 30 | 20 | 66.7% | 7 | 12 | 11 | 3.78% | 113.4% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 30 | 20 | 66.7% | 7 | 12 | 11 | 3.78% | 113.4% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 60 | 27 | 45.0% | 14 | 35 | 11 | 2.31% | 138.8% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-07-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-21 10:15:00 | 1011.05 | 859.74 | 859.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-24 11:15:00 | 1018.40 | 870.65 | 865.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-13 09:15:00 | 1038.15 | 1068.08 | 1015.24 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-09-13 09:30:00 | 1045.00 | 1068.08 | 1015.24 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-09 09:15:00 | 1060.60 | 1091.30 | 1052.79 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-11 10:15:00 | 1073.90 | 1085.78 | 1052.52 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-11 11:45:00 | 1073.50 | 1085.44 | 1052.68 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-11 12:30:00 | 1076.85 | 1085.33 | 1052.79 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-10-11 14:15:00 | 1046.85 | 1084.69 | 1052.79 | SL hit (close<static) qty=1.00 sl=1052.10 alert=retest2 |

### Cycle 2 — SELL (started 2023-10-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-23 13:15:00 | 904.20 | 1030.04 | 1030.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-23 14:15:00 | 899.95 | 1028.74 | 1029.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-28 09:15:00 | 937.85 | 936.92 | 965.00 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-11-28 09:30:00 | 942.10 | 936.92 | 965.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-30 09:15:00 | 972.05 | 939.07 | 964.21 | EMA400 retest candle locked (from downside) |

### Cycle 3 — BUY (started 2023-12-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-12 12:15:00 | 1100.10 | 982.21 | 981.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-14 09:15:00 | 1107.10 | 993.50 | 987.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-02 14:15:00 | 1040.20 | 1042.09 | 1020.67 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-01-02 15:00:00 | 1040.20 | 1042.09 | 1020.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-09 13:15:00 | 1022.25 | 1040.49 | 1023.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-09 14:00:00 | 1022.25 | 1040.49 | 1023.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-09 14:15:00 | 1039.50 | 1040.48 | 1023.26 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-09 15:15:00 | 1039.80 | 1040.48 | 1023.26 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-10 14:45:00 | 1041.90 | 1040.08 | 1023.65 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-11 09:45:00 | 1041.05 | 1040.22 | 1023.89 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-11 13:30:00 | 1040.00 | 1040.08 | 1024.14 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Target hit | 2024-02-07 09:15:00 | 1143.78 | 1071.02 | 1050.46 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 4 — SELL (started 2024-02-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-22 15:15:00 | 982.85 | 1040.34 | 1040.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-23 11:15:00 | 980.05 | 1038.59 | 1039.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-26 10:15:00 | 932.70 | 931.34 | 969.81 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-03-26 10:45:00 | 931.25 | 931.34 | 969.81 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-26 15:15:00 | 974.00 | 932.64 | 969.52 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-27 10:30:00 | 959.85 | 933.21 | 969.44 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-03-27 12:15:00 | 984.60 | 933.98 | 969.46 | SL hit (close>static) qty=1.00 sl=978.75 alert=retest2 |

### Cycle 5 — BUY (started 2024-04-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-30 09:15:00 | 1028.35 | 984.97 | 984.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-30 10:15:00 | 1054.85 | 985.66 | 985.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-13 09:15:00 | 1000.35 | 1009.53 | 999.05 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-13 09:15:00 | 1000.35 | 1009.53 | 999.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-13 09:15:00 | 1000.35 | 1009.53 | 999.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-13 10:00:00 | 1000.35 | 1009.53 | 999.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-13 10:15:00 | 1000.00 | 1009.43 | 999.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-13 10:30:00 | 996.90 | 1009.43 | 999.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-13 11:15:00 | 1000.95 | 1009.35 | 999.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-13 12:00:00 | 1000.95 | 1009.35 | 999.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-05 10:15:00 | 1464.45 | 1548.13 | 1465.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-05 11:00:00 | 1464.45 | 1548.13 | 1465.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-05 11:15:00 | 1462.30 | 1547.28 | 1465.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-05 12:00:00 | 1462.30 | 1547.28 | 1465.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-05 12:15:00 | 1455.45 | 1546.36 | 1465.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-05 13:00:00 | 1455.45 | 1546.36 | 1465.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-05 13:15:00 | 1469.15 | 1545.59 | 1465.28 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-06 09:15:00 | 1500.00 | 1543.88 | 1465.22 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-07 12:15:00 | 1476.35 | 1538.19 | 1466.15 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-07 14:15:00 | 1478.95 | 1536.93 | 1466.24 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-12 10:30:00 | 1481.45 | 1529.64 | 1468.52 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-12 12:15:00 | 1466.95 | 1528.47 | 1468.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-12 13:00:00 | 1466.95 | 1528.47 | 1468.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-12 13:15:00 | 1466.00 | 1527.85 | 1468.53 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-08-13 11:15:00 | 1443.65 | 1524.34 | 1468.22 | SL hit (close<static) qty=1.00 sl=1453.70 alert=retest2 |

### Cycle 6 — SELL (started 2024-09-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-13 13:15:00 | 1414.00 | 1451.54 | 1451.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-16 11:15:00 | 1407.00 | 1449.82 | 1450.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-23 10:15:00 | 1443.00 | 1435.84 | 1443.03 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-23 10:15:00 | 1443.00 | 1435.84 | 1443.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-23 10:15:00 | 1443.00 | 1435.84 | 1443.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-23 11:00:00 | 1443.00 | 1435.84 | 1443.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-23 11:15:00 | 1467.00 | 1436.15 | 1443.15 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-26 12:30:00 | 1429.00 | 1444.73 | 1447.03 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-30 09:15:00 | 1437.00 | 1445.29 | 1447.21 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-30 10:30:00 | 1437.80 | 1445.12 | 1447.10 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-01 14:00:00 | 1420.00 | 1443.12 | 1445.98 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-03 13:15:00 | 1365.15 | 1439.72 | 1444.16 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-03 13:15:00 | 1365.91 | 1439.72 | 1444.16 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-04 09:15:00 | 1357.55 | 1437.80 | 1443.13 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-04 09:15:00 | 1349.00 | 1437.80 | 1443.13 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2024-10-07 10:15:00 | 1286.10 | 1431.44 | 1439.68 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 7 — BUY (started 2025-05-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-27 11:15:00 | 983.55 | 922.63 | 922.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-27 15:15:00 | 995.00 | 925.17 | 923.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-16 09:15:00 | 947.00 | 955.16 | 942.63 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-16 09:15:00 | 947.00 | 955.16 | 942.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 09:15:00 | 947.00 | 955.16 | 942.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-16 09:45:00 | 943.25 | 955.16 | 942.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 11:15:00 | 947.15 | 955.83 | 944.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-19 11:45:00 | 949.20 | 955.83 | 944.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 12:15:00 | 936.10 | 955.63 | 944.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-19 13:00:00 | 936.10 | 955.63 | 944.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 13:15:00 | 940.50 | 955.48 | 944.28 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-25 09:15:00 | 942.85 | 950.75 | 942.99 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-25 09:45:00 | 944.25 | 950.68 | 942.99 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-26 09:15:00 | 945.25 | 950.01 | 942.88 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-26 10:15:00 | 942.50 | 949.92 | 942.87 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 10:15:00 | 937.85 | 949.80 | 942.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-26 10:45:00 | 936.90 | 949.80 | 942.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 11:15:00 | 936.65 | 949.67 | 942.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-26 12:00:00 | 936.65 | 949.67 | 942.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 15:15:00 | 938.90 | 949.24 | 942.73 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-27 09:30:00 | 945.30 | 949.18 | 942.74 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-18 13:15:00 | 932.70 | 956.34 | 950.65 | SL hit (close<static) qty=1.00 sl=932.90 alert=retest2 |

### Cycle 8 — SELL (started 2025-07-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-25 12:15:00 | 903.10 | 945.77 | 945.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-28 10:15:00 | 893.45 | 943.46 | 944.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-02 11:15:00 | 863.80 | 862.32 | 889.42 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-02 12:00:00 | 863.80 | 862.32 | 889.42 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 13:15:00 | 778.40 | 757.67 | 780.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-16 14:00:00 | 778.40 | 757.67 | 780.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 14:15:00 | 779.45 | 757.88 | 780.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-16 14:45:00 | 781.00 | 757.88 | 780.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 15:15:00 | 779.00 | 758.09 | 780.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-17 09:15:00 | 780.00 | 758.09 | 780.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 09:15:00 | 775.50 | 758.27 | 780.73 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-17 10:30:00 | 771.00 | 758.40 | 780.69 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-17 11:15:00 | 770.75 | 758.40 | 780.69 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-17 12:15:00 | 770.15 | 758.53 | 780.64 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-19 09:15:00 | 784.15 | 760.16 | 780.19 | SL hit (close>static) qty=1.00 sl=782.60 alert=retest2 |

### Cycle 9 — BUY (started 2026-02-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-17 09:15:00 | 814.50 | 767.78 | 767.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-17 12:15:00 | 816.85 | 769.11 | 768.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-16 09:15:00 | 852.30 | 853.01 | 820.53 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-16 10:15:00 | 840.60 | 853.01 | 820.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-23 11:15:00 | 824.80 | 854.77 | 826.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-23 12:00:00 | 824.80 | 854.77 | 826.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-23 12:15:00 | 814.20 | 854.36 | 826.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-23 13:00:00 | 814.20 | 854.36 | 826.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 10:15:00 | 816.00 | 852.48 | 826.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-24 10:45:00 | 817.60 | 852.48 | 826.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 11:15:00 | 826.95 | 852.23 | 826.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-24 11:30:00 | 817.40 | 852.23 | 826.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 12:15:00 | 835.30 | 852.06 | 826.64 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-25 14:30:00 | 837.90 | 850.20 | 826.80 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-27 09:15:00 | 810.30 | 849.62 | 826.74 | SL hit (close<static) qty=1.00 sl=825.55 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2023-10-11 10:15:00 | 1073.90 | 2023-10-11 14:15:00 | 1046.85 | STOP_HIT | 1.00 | -2.52% |
| BUY | retest2 | 2023-10-11 11:45:00 | 1073.50 | 2023-10-11 14:15:00 | 1046.85 | STOP_HIT | 1.00 | -2.48% |
| BUY | retest2 | 2023-10-11 12:30:00 | 1076.85 | 2023-10-11 14:15:00 | 1046.85 | STOP_HIT | 1.00 | -2.79% |
| BUY | retest2 | 2024-01-09 15:15:00 | 1039.80 | 2024-02-07 09:15:00 | 1143.78 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-01-10 14:45:00 | 1041.90 | 2024-02-07 09:15:00 | 1146.09 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-01-11 09:45:00 | 1041.05 | 2024-02-07 09:15:00 | 1145.15 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-01-11 13:30:00 | 1040.00 | 2024-02-07 09:15:00 | 1144.00 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-03-27 10:30:00 | 959.85 | 2024-03-27 12:15:00 | 984.60 | STOP_HIT | 1.00 | -2.58% |
| BUY | retest2 | 2024-08-06 09:15:00 | 1500.00 | 2024-08-13 11:15:00 | 1443.65 | STOP_HIT | 1.00 | -3.76% |
| BUY | retest2 | 2024-08-07 12:15:00 | 1476.35 | 2024-08-13 11:15:00 | 1443.65 | STOP_HIT | 1.00 | -2.21% |
| BUY | retest2 | 2024-08-07 14:15:00 | 1478.95 | 2024-08-13 11:15:00 | 1443.65 | STOP_HIT | 1.00 | -2.39% |
| BUY | retest2 | 2024-08-12 10:30:00 | 1481.45 | 2024-08-13 11:15:00 | 1443.65 | STOP_HIT | 1.00 | -2.55% |
| BUY | retest2 | 2024-08-14 12:45:00 | 1483.25 | 2024-08-16 12:15:00 | 1455.90 | STOP_HIT | 1.00 | -1.84% |
| BUY | retest2 | 2024-08-14 13:15:00 | 1472.80 | 2024-08-16 12:15:00 | 1455.90 | STOP_HIT | 1.00 | -1.15% |
| BUY | retest2 | 2024-08-16 10:15:00 | 1474.60 | 2024-08-16 12:15:00 | 1455.90 | STOP_HIT | 1.00 | -1.27% |
| BUY | retest2 | 2024-08-20 09:15:00 | 1474.20 | 2024-08-21 10:15:00 | 1455.95 | STOP_HIT | 1.00 | -1.24% |
| BUY | retest2 | 2024-08-23 09:30:00 | 1493.00 | 2024-08-26 09:15:00 | 1464.85 | STOP_HIT | 1.00 | -1.89% |
| BUY | retest2 | 2024-08-26 15:15:00 | 1490.00 | 2024-08-28 14:15:00 | 1464.00 | STOP_HIT | 1.00 | -1.74% |
| BUY | retest2 | 2024-08-27 10:45:00 | 1485.15 | 2024-08-28 14:15:00 | 1464.00 | STOP_HIT | 1.00 | -1.42% |
| BUY | retest2 | 2024-08-27 15:00:00 | 1483.30 | 2024-08-28 14:15:00 | 1464.00 | STOP_HIT | 1.00 | -1.30% |
| BUY | retest2 | 2024-08-30 09:30:00 | 1492.00 | 2024-08-30 14:15:00 | 1449.55 | STOP_HIT | 1.00 | -2.85% |
| BUY | retest2 | 2024-08-30 14:30:00 | 1476.40 | 2024-08-30 15:15:00 | 1448.00 | STOP_HIT | 1.00 | -1.92% |
| SELL | retest2 | 2024-09-26 12:30:00 | 1429.00 | 2024-10-03 13:15:00 | 1365.15 | PARTIAL | 0.50 | 4.47% |
| SELL | retest2 | 2024-09-30 09:15:00 | 1437.00 | 2024-10-03 13:15:00 | 1365.91 | PARTIAL | 0.50 | 4.95% |
| SELL | retest2 | 2024-09-30 10:30:00 | 1437.80 | 2024-10-04 09:15:00 | 1357.55 | PARTIAL | 0.50 | 5.58% |
| SELL | retest2 | 2024-10-01 14:00:00 | 1420.00 | 2024-10-04 09:15:00 | 1349.00 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-09-26 12:30:00 | 1429.00 | 2024-10-07 10:15:00 | 1286.10 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-09-30 09:15:00 | 1437.00 | 2024-10-07 10:15:00 | 1293.30 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-09-30 10:30:00 | 1437.80 | 2024-10-07 10:15:00 | 1294.02 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-10-01 14:00:00 | 1420.00 | 2024-10-08 09:15:00 | 1278.00 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-12-16 13:45:00 | 1248.45 | 2024-12-19 09:15:00 | 1186.03 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-17 09:15:00 | 1242.05 | 2024-12-20 10:15:00 | 1179.95 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-16 13:45:00 | 1248.45 | 2024-12-24 13:15:00 | 1246.00 | STOP_HIT | 0.50 | 0.20% |
| SELL | retest2 | 2024-12-17 09:15:00 | 1242.05 | 2024-12-24 13:15:00 | 1246.00 | STOP_HIT | 0.50 | -0.32% |
| SELL | retest2 | 2024-12-24 14:00:00 | 1246.00 | 2024-12-24 14:15:00 | 1284.80 | STOP_HIT | 1.00 | -3.11% |
| SELL | retest2 | 2024-12-24 15:15:00 | 1230.00 | 2024-12-30 14:15:00 | 1168.50 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-26 10:00:00 | 1221.50 | 2024-12-30 14:15:00 | 1160.42 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-26 11:15:00 | 1224.95 | 2024-12-30 14:15:00 | 1163.70 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-24 15:15:00 | 1230.00 | 2025-01-06 13:15:00 | 1107.00 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-12-26 10:00:00 | 1221.50 | 2025-01-06 15:15:00 | 1102.46 | TARGET_HIT | 0.50 | 9.75% |
| SELL | retest2 | 2024-12-26 11:15:00 | 1224.95 | 2025-01-08 12:15:00 | 1099.35 | TARGET_HIT | 0.50 | 10.25% |
| BUY | retest2 | 2025-06-25 09:15:00 | 942.85 | 2025-07-18 13:15:00 | 932.70 | STOP_HIT | 1.00 | -1.08% |
| BUY | retest2 | 2025-06-25 09:45:00 | 944.25 | 2025-07-18 13:15:00 | 932.70 | STOP_HIT | 1.00 | -1.22% |
| BUY | retest2 | 2025-06-26 09:15:00 | 945.25 | 2025-07-18 13:15:00 | 932.70 | STOP_HIT | 1.00 | -1.33% |
| BUY | retest2 | 2025-06-26 10:15:00 | 942.50 | 2025-07-18 13:15:00 | 932.70 | STOP_HIT | 1.00 | -1.04% |
| BUY | retest2 | 2025-06-27 09:30:00 | 945.30 | 2025-07-18 13:15:00 | 932.70 | STOP_HIT | 1.00 | -1.33% |
| SELL | retest2 | 2025-12-17 10:30:00 | 771.00 | 2025-12-19 09:15:00 | 784.15 | STOP_HIT | 1.00 | -1.71% |
| SELL | retest2 | 2025-12-17 11:15:00 | 770.75 | 2025-12-19 09:15:00 | 784.15 | STOP_HIT | 1.00 | -1.74% |
| SELL | retest2 | 2025-12-17 12:15:00 | 770.15 | 2025-12-19 09:15:00 | 784.15 | STOP_HIT | 1.00 | -1.82% |
| SELL | retest2 | 2025-12-23 14:30:00 | 770.15 | 2025-12-30 11:15:00 | 731.64 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-23 14:30:00 | 770.15 | 2026-01-01 11:15:00 | 770.85 | STOP_HIT | 0.50 | -0.09% |
| SELL | retest2 | 2026-01-09 11:45:00 | 770.50 | 2026-01-16 13:15:00 | 780.00 | STOP_HIT | 1.00 | -1.23% |
| SELL | retest2 | 2026-01-14 13:45:00 | 772.80 | 2026-01-16 13:15:00 | 780.00 | STOP_HIT | 1.00 | -0.93% |
| SELL | retest2 | 2026-01-19 10:00:00 | 772.05 | 2026-01-20 13:15:00 | 733.45 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-19 10:00:00 | 772.05 | 2026-02-04 09:15:00 | 743.95 | STOP_HIT | 0.50 | 3.64% |
| SELL | retest2 | 2026-02-06 15:15:00 | 769.80 | 2026-02-09 10:15:00 | 784.50 | STOP_HIT | 1.00 | -1.91% |
| BUY | retest2 | 2026-03-25 14:30:00 | 837.90 | 2026-03-27 09:15:00 | 810.30 | STOP_HIT | 1.00 | -3.29% |
| BUY | retest2 | 2026-04-08 09:30:00 | 838.30 | 2026-04-16 09:15:00 | 922.13 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-08 10:15:00 | 844.20 | 2026-04-16 09:15:00 | 928.62 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-13 10:15:00 | 841.15 | 2026-04-16 09:15:00 | 925.27 | TARGET_HIT | 1.00 | 10.00% |
