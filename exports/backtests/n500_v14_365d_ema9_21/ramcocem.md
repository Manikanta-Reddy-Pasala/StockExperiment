# The Ramco Cements Ltd. (RAMCOCEM)

## Backtest Summary

- **Window:** 2025-04-11 09:15:00 → 2026-05-08 15:15:00 (1850 bars)
- **Last close:** 953.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 71 |
| ALERT1 | 47 |
| ALERT2 | 46 |
| ALERT2_SKIP | 24 |
| ALERT3 | 152 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 2 |
| ENTRY2 | 99 |
| PARTIAL | 7 |
| TARGET_HIT | 0 |
| STOP_HIT | 101 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 108 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 30 / 78
- **Target hits / Stop hits / Partials:** 0 / 101 / 7
- **Avg / median % per leg:** -0.12% / -0.84%
- **Sum % (uncompounded):** -13.14%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 53 | 15 | 28.3% | 0 | 53 | 0 | -0.37% | -19.4% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 53 | 15 | 28.3% | 0 | 53 | 0 | -0.37% | -19.4% |
| SELL (all) | 55 | 15 | 27.3% | 0 | 48 | 7 | 0.11% | 6.3% |
| SELL @ 2nd Alert (retest1) | 3 | 2 | 66.7% | 0 | 2 | 1 | 1.68% | 5.0% |
| SELL @ 3rd Alert (retest2) | 52 | 13 | 25.0% | 0 | 46 | 6 | 0.02% | 1.3% |
| retest1 (combined) | 3 | 2 | 66.7% | 0 | 2 | 1 | 1.68% | 5.0% |
| retest2 (combined) | 105 | 28 | 26.7% | 0 | 99 | 6 | -0.17% | -18.2% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 13:15:00 | 950.00 | 947.12 | 947.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 15:15:00 | 965.00 | 951.15 | 948.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-16 14:15:00 | 1005.80 | 1005.97 | 997.70 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-16 14:45:00 | 1005.65 | 1005.97 | 997.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 09:15:00 | 1009.00 | 1006.00 | 999.11 | EMA400 retest candle locked (from upside) |

### Cycle 2 — SELL (started 2025-05-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-20 13:15:00 | 989.10 | 997.21 | 998.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-20 14:15:00 | 982.55 | 994.27 | 996.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-22 09:15:00 | 984.75 | 981.99 | 987.02 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-22 09:15:00 | 984.75 | 981.99 | 987.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 09:15:00 | 984.75 | 981.99 | 987.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-22 09:30:00 | 984.85 | 981.99 | 987.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 10:15:00 | 990.10 | 983.61 | 987.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-22 11:00:00 | 990.10 | 983.61 | 987.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 11:15:00 | 989.90 | 984.87 | 987.54 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-22 12:30:00 | 983.50 | 984.65 | 987.19 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-23 09:15:00 | 979.05 | 984.94 | 986.63 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-23 09:15:00 | 998.50 | 987.66 | 987.71 | SL hit (close>static) qty=1.00 sl=992.40 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-05-23 09:15:00 | 998.50 | 987.66 | 987.71 | SL hit (close>static) qty=1.00 sl=992.40 alert=retest2 |

### Cycle 3 — BUY (started 2025-05-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-23 10:15:00 | 996.70 | 989.46 | 988.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-23 11:15:00 | 1006.00 | 992.77 | 990.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-26 09:15:00 | 986.20 | 996.89 | 993.86 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-26 09:15:00 | 986.20 | 996.89 | 993.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 09:15:00 | 986.20 | 996.89 | 993.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-26 10:00:00 | 986.20 | 996.89 | 993.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 10:15:00 | 980.25 | 993.56 | 992.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-26 11:00:00 | 980.25 | 993.56 | 992.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 4 — SELL (started 2025-05-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-26 11:15:00 | 982.50 | 991.35 | 991.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-27 09:15:00 | 972.80 | 983.43 | 987.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-29 09:15:00 | 969.00 | 964.46 | 970.49 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-29 09:15:00 | 969.00 | 964.46 | 970.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 09:15:00 | 969.00 | 964.46 | 970.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-29 09:45:00 | 971.45 | 964.46 | 970.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 10:15:00 | 968.65 | 965.30 | 970.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-29 10:30:00 | 966.10 | 965.30 | 970.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 11:15:00 | 962.05 | 964.65 | 969.57 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-29 12:30:00 | 959.00 | 961.74 | 967.80 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-30 14:15:00 | 975.50 | 960.69 | 962.38 | SL hit (close>static) qty=1.00 sl=969.95 alert=retest2 |

### Cycle 5 — BUY (started 2025-06-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-02 09:15:00 | 973.10 | 964.66 | 963.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-02 10:15:00 | 983.75 | 968.48 | 965.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-03 14:15:00 | 988.40 | 988.51 | 981.66 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-03 14:45:00 | 987.75 | 988.51 | 981.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 15:15:00 | 982.00 | 987.21 | 981.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-04 09:15:00 | 980.50 | 987.21 | 981.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 09:15:00 | 991.20 | 988.01 | 982.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-04 09:30:00 | 978.15 | 988.01 | 982.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 09:15:00 | 996.90 | 1001.88 | 997.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-06 10:00:00 | 996.90 | 1001.88 | 997.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 10:15:00 | 1000.00 | 1001.51 | 997.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-06 10:30:00 | 996.00 | 1001.51 | 997.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 11:15:00 | 1059.80 | 1066.17 | 1056.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-12 11:45:00 | 1062.55 | 1066.17 | 1056.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 12:15:00 | 1058.65 | 1064.67 | 1056.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-12 13:00:00 | 1058.65 | 1064.67 | 1056.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 13:15:00 | 1052.95 | 1062.33 | 1056.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-12 14:00:00 | 1052.95 | 1062.33 | 1056.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 14:15:00 | 1054.10 | 1060.68 | 1056.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-12 15:15:00 | 1051.00 | 1060.68 | 1056.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 15:15:00 | 1051.00 | 1058.74 | 1055.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-13 09:15:00 | 1047.45 | 1058.74 | 1055.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 09:15:00 | 1049.45 | 1056.89 | 1055.09 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-13 10:30:00 | 1057.35 | 1057.31 | 1055.44 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-18 09:15:00 | 1048.50 | 1063.46 | 1063.75 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 6 — SELL (started 2025-06-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-18 09:15:00 | 1048.50 | 1063.46 | 1063.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-18 12:15:00 | 1039.05 | 1053.72 | 1058.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-20 10:15:00 | 1025.90 | 1024.65 | 1034.97 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-20 11:00:00 | 1025.90 | 1024.65 | 1034.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 14:15:00 | 1008.75 | 1021.72 | 1030.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 14:30:00 | 1026.80 | 1021.72 | 1030.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 12:15:00 | 1016.25 | 1016.54 | 1023.97 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-23 13:30:00 | 1011.05 | 1016.05 | 1023.08 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-23 15:15:00 | 1010.10 | 1015.74 | 1022.30 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-24 09:45:00 | 1007.65 | 1013.82 | 1020.23 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-24 13:30:00 | 1009.60 | 1015.52 | 1019.23 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 14:15:00 | 1016.30 | 1015.68 | 1018.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-24 15:00:00 | 1016.30 | 1015.68 | 1018.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 15:15:00 | 1021.10 | 1016.76 | 1019.16 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-25 09:30:00 | 1014.10 | 1018.13 | 1019.56 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-25 10:15:00 | 1035.70 | 1021.64 | 1021.03 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-25 10:15:00 | 1035.70 | 1021.64 | 1021.03 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-25 10:15:00 | 1035.70 | 1021.64 | 1021.03 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-25 10:15:00 | 1035.70 | 1021.64 | 1021.03 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-25 10:15:00 | 1035.70 | 1021.64 | 1021.03 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 7 — BUY (started 2025-06-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-25 10:15:00 | 1035.70 | 1021.64 | 1021.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-25 14:15:00 | 1041.75 | 1030.77 | 1026.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-27 09:15:00 | 1042.10 | 1045.81 | 1039.06 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-27 10:00:00 | 1042.10 | 1045.81 | 1039.06 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 11:15:00 | 1077.70 | 1083.36 | 1078.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-03 11:30:00 | 1077.50 | 1083.36 | 1078.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 12:15:00 | 1072.20 | 1081.13 | 1077.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-03 13:00:00 | 1072.20 | 1081.13 | 1077.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 13:15:00 | 1072.90 | 1079.48 | 1077.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-03 13:30:00 | 1072.90 | 1079.48 | 1077.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 15:15:00 | 1072.90 | 1077.45 | 1076.59 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-04 09:15:00 | 1077.90 | 1077.45 | 1076.59 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-04 10:00:00 | 1083.00 | 1078.56 | 1077.17 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-07 11:15:00 | 1069.10 | 1082.51 | 1081.73 | SL hit (close<static) qty=1.00 sl=1070.70 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-07 11:15:00 | 1069.10 | 1082.51 | 1081.73 | SL hit (close<static) qty=1.00 sl=1070.70 alert=retest2 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-07 12:30:00 | 1081.00 | 1082.53 | 1081.81 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-08 10:15:00 | 1079.50 | 1081.93 | 1081.85 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-08 10:15:00 | 1074.10 | 1080.37 | 1081.15 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-08 10:15:00 | 1074.10 | 1080.37 | 1081.15 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 8 — SELL (started 2025-07-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-08 10:15:00 | 1074.10 | 1080.37 | 1081.15 | EMA200 below EMA400 |

### Cycle 9 — BUY (started 2025-07-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-08 12:15:00 | 1086.00 | 1081.92 | 1081.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-08 13:15:00 | 1096.70 | 1084.87 | 1083.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-09 14:15:00 | 1099.60 | 1099.91 | 1093.76 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-09 14:15:00 | 1099.60 | 1099.91 | 1093.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 14:15:00 | 1099.60 | 1099.91 | 1093.76 | EMA400 retest candle locked (from upside) |

### Cycle 10 — SELL (started 2025-07-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-10 12:15:00 | 1086.00 | 1091.64 | 1091.68 | EMA200 below EMA400 |

### Cycle 11 — BUY (started 2025-07-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-10 13:15:00 | 1099.00 | 1093.11 | 1092.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-10 14:15:00 | 1105.50 | 1095.59 | 1093.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-14 14:15:00 | 1134.90 | 1139.22 | 1128.35 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-14 15:00:00 | 1134.90 | 1139.22 | 1128.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 15:15:00 | 1123.00 | 1135.97 | 1127.86 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-15 09:45:00 | 1137.20 | 1136.64 | 1128.90 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-15 12:30:00 | 1135.50 | 1135.81 | 1130.32 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-21 13:15:00 | 1151.90 | 1155.93 | 1156.30 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-21 13:15:00 | 1151.90 | 1155.93 | 1156.30 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 12 — SELL (started 2025-07-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-21 13:15:00 | 1151.90 | 1155.93 | 1156.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-21 14:15:00 | 1149.80 | 1154.71 | 1155.71 | Break + close below crossover candle low |

### Cycle 13 — BUY (started 2025-07-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-22 09:15:00 | 1172.10 | 1156.36 | 1156.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-22 11:15:00 | 1190.80 | 1166.23 | 1160.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-23 09:15:00 | 1170.80 | 1177.57 | 1169.60 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-23 09:15:00 | 1170.80 | 1177.57 | 1169.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 09:15:00 | 1170.80 | 1177.57 | 1169.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-23 10:00:00 | 1170.80 | 1177.57 | 1169.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 10:15:00 | 1168.60 | 1175.78 | 1169.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-23 10:45:00 | 1165.10 | 1175.78 | 1169.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 11:15:00 | 1178.50 | 1176.32 | 1170.33 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-23 12:45:00 | 1183.10 | 1178.24 | 1171.74 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-24 12:15:00 | 1182.70 | 1181.81 | 1176.99 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-24 13:00:00 | 1181.20 | 1181.69 | 1177.37 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-24 13:45:00 | 1181.50 | 1181.17 | 1177.53 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 14:15:00 | 1179.00 | 1180.73 | 1177.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-24 14:45:00 | 1176.20 | 1180.73 | 1177.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 15:15:00 | 1174.20 | 1179.43 | 1177.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-25 09:15:00 | 1167.40 | 1179.43 | 1177.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 09:15:00 | 1170.00 | 1177.54 | 1176.68 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-07-25 10:15:00 | 1165.90 | 1175.21 | 1175.70 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-25 10:15:00 | 1165.90 | 1175.21 | 1175.70 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-25 10:15:00 | 1165.90 | 1175.21 | 1175.70 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-25 10:15:00 | 1165.90 | 1175.21 | 1175.70 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 14 — SELL (started 2025-07-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-25 10:15:00 | 1165.90 | 1175.21 | 1175.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-25 11:15:00 | 1158.50 | 1171.87 | 1174.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-28 09:15:00 | 1167.60 | 1160.45 | 1166.51 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-28 09:15:00 | 1167.60 | 1160.45 | 1166.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 09:15:00 | 1167.60 | 1160.45 | 1166.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-28 09:45:00 | 1163.80 | 1160.45 | 1166.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 10:15:00 | 1172.80 | 1162.92 | 1167.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-28 11:00:00 | 1172.80 | 1162.92 | 1167.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 12:15:00 | 1169.90 | 1166.25 | 1167.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-28 12:30:00 | 1175.60 | 1166.25 | 1167.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 09:15:00 | 1168.40 | 1167.81 | 1168.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 10:00:00 | 1168.40 | 1167.81 | 1168.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 15 — BUY (started 2025-07-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-29 10:15:00 | 1181.50 | 1170.55 | 1169.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-29 11:15:00 | 1183.20 | 1173.08 | 1170.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-30 10:15:00 | 1182.40 | 1184.37 | 1178.68 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-30 10:15:00 | 1182.40 | 1184.37 | 1178.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 10:15:00 | 1182.40 | 1184.37 | 1178.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-30 11:00:00 | 1182.40 | 1184.37 | 1178.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 12:15:00 | 1186.00 | 1184.15 | 1179.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-30 12:30:00 | 1181.20 | 1184.15 | 1179.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 10:15:00 | 1183.50 | 1191.04 | 1185.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-31 11:00:00 | 1183.50 | 1191.04 | 1185.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 11:15:00 | 1183.00 | 1189.43 | 1185.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-31 11:30:00 | 1185.50 | 1189.43 | 1185.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 12:15:00 | 1187.60 | 1189.07 | 1185.53 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-31 13:45:00 | 1191.10 | 1189.51 | 1186.06 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-31 15:15:00 | 1170.10 | 1184.04 | 1184.07 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 16 — SELL (started 2025-07-31 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-31 15:15:00 | 1170.10 | 1184.04 | 1184.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-01 12:15:00 | 1163.90 | 1175.81 | 1179.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-04 09:15:00 | 1171.00 | 1164.72 | 1172.39 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-04 09:15:00 | 1171.00 | 1164.72 | 1172.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 09:15:00 | 1171.00 | 1164.72 | 1172.39 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-07 10:15:00 | 1144.80 | 1152.86 | 1157.73 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-07 12:45:00 | 1144.00 | 1148.66 | 1154.40 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-08 09:15:00 | 1087.56 | 1129.59 | 1143.45 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-08 09:15:00 | 1086.80 | 1129.59 | 1143.45 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-11 14:15:00 | 1070.00 | 1068.47 | 1089.68 | SL hit (close>ema200) qty=0.50 sl=1068.47 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-11 14:15:00 | 1070.00 | 1068.47 | 1089.68 | SL hit (close>ema200) qty=0.50 sl=1068.47 alert=retest2 |

### Cycle 17 — BUY (started 2025-08-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-18 10:15:00 | 1082.60 | 1073.93 | 1072.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-20 09:15:00 | 1100.00 | 1085.43 | 1080.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-21 15:15:00 | 1103.00 | 1104.44 | 1097.90 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-22 09:15:00 | 1082.00 | 1104.44 | 1097.90 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 09:15:00 | 1085.50 | 1100.65 | 1096.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-22 09:45:00 | 1081.50 | 1100.65 | 1096.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 10:15:00 | 1082.10 | 1096.94 | 1095.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-22 10:45:00 | 1082.20 | 1096.94 | 1095.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 18 — SELL (started 2025-08-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-22 11:15:00 | 1071.60 | 1091.87 | 1093.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-25 12:15:00 | 1063.30 | 1075.07 | 1082.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-28 15:15:00 | 1039.90 | 1038.96 | 1049.33 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-29 09:15:00 | 1031.10 | 1038.96 | 1049.33 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 10:15:00 | 1045.70 | 1040.70 | 1048.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-29 10:45:00 | 1045.50 | 1040.70 | 1048.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 13:15:00 | 1040.70 | 1041.24 | 1046.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-29 13:30:00 | 1048.00 | 1041.24 | 1046.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 14:15:00 | 1046.60 | 1042.31 | 1046.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-29 15:00:00 | 1046.60 | 1042.31 | 1046.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 15:15:00 | 1044.60 | 1042.77 | 1046.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 09:15:00 | 1048.40 | 1042.77 | 1046.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 09:15:00 | 1059.40 | 1046.10 | 1047.71 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-09-01 09:15:00 | 1059.40 | 1046.10 | 1047.71 | SL hit (close>ema400) qty=1.00 sl=1047.71 alert=retest1 |
| ALERT3_SIDEWAYS | 2025-09-01 10:00:00 | 1059.40 | 1046.10 | 1047.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 10:15:00 | 1055.10 | 1047.90 | 1048.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 11:15:00 | 1059.40 | 1047.90 | 1048.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 19 — BUY (started 2025-09-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-01 11:15:00 | 1063.10 | 1050.94 | 1049.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-02 09:15:00 | 1069.00 | 1058.74 | 1054.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-03 10:15:00 | 1069.90 | 1072.67 | 1065.71 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-03 11:00:00 | 1069.90 | 1072.67 | 1065.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 10:15:00 | 1074.50 | 1075.32 | 1070.67 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-04 11:15:00 | 1080.00 | 1075.32 | 1070.67 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-05 09:30:00 | 1079.20 | 1080.64 | 1076.07 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-05 10:15:00 | 1067.00 | 1077.91 | 1075.25 | SL hit (close<static) qty=1.00 sl=1069.20 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-05 10:15:00 | 1067.00 | 1077.91 | 1075.25 | SL hit (close<static) qty=1.00 sl=1069.20 alert=retest2 |

### Cycle 20 — SELL (started 2025-09-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-05 14:15:00 | 1068.50 | 1073.24 | 1073.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-05 15:15:00 | 1066.00 | 1071.79 | 1072.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-08 09:15:00 | 1076.70 | 1072.77 | 1073.26 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-08 09:15:00 | 1076.70 | 1072.77 | 1073.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 09:15:00 | 1076.70 | 1072.77 | 1073.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-08 10:00:00 | 1076.70 | 1072.77 | 1073.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 10:15:00 | 1068.20 | 1071.86 | 1072.80 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-08 11:15:00 | 1063.90 | 1071.86 | 1072.80 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-15 12:15:00 | 1053.80 | 1045.42 | 1044.28 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 21 — BUY (started 2025-09-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-15 12:15:00 | 1053.80 | 1045.42 | 1044.28 | EMA200 above EMA400 |

### Cycle 22 — SELL (started 2025-09-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-16 14:15:00 | 1037.50 | 1043.92 | 1044.71 | EMA200 below EMA400 |

### Cycle 23 — BUY (started 2025-09-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-17 09:15:00 | 1082.50 | 1051.49 | 1048.00 | EMA200 above EMA400 |

### Cycle 24 — SELL (started 2025-09-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-22 09:15:00 | 1052.20 | 1056.76 | 1056.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-22 13:15:00 | 1049.40 | 1054.81 | 1055.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-23 09:15:00 | 1054.70 | 1052.24 | 1054.29 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-23 09:15:00 | 1054.70 | 1052.24 | 1054.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 09:15:00 | 1054.70 | 1052.24 | 1054.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-23 09:30:00 | 1056.00 | 1052.24 | 1054.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 10:15:00 | 1060.80 | 1053.96 | 1054.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-23 11:00:00 | 1060.80 | 1053.96 | 1054.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 11:15:00 | 1058.90 | 1054.94 | 1055.25 | EMA400 retest candle locked (from downside) |

### Cycle 25 — BUY (started 2025-09-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-23 12:15:00 | 1060.20 | 1056.00 | 1055.70 | EMA200 above EMA400 |

### Cycle 26 — SELL (started 2025-09-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-23 13:15:00 | 1052.30 | 1055.26 | 1055.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-23 14:15:00 | 1045.00 | 1053.21 | 1054.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-24 11:15:00 | 1050.70 | 1050.52 | 1052.53 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-24 12:00:00 | 1050.70 | 1050.52 | 1052.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 10:15:00 | 1022.40 | 1015.78 | 1024.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-29 10:30:00 | 1022.80 | 1015.78 | 1024.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 15:15:00 | 991.80 | 986.21 | 994.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-03 09:15:00 | 996.35 | 986.21 | 994.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 09:15:00 | 997.15 | 988.40 | 994.29 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-03 10:30:00 | 988.20 | 986.72 | 992.99 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-06 09:45:00 | 984.00 | 985.45 | 989.61 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-07 10:15:00 | 1001.80 | 989.90 | 988.96 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-07 10:15:00 | 1001.80 | 989.90 | 988.96 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 27 — BUY (started 2025-10-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-07 10:15:00 | 1001.80 | 989.90 | 988.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-07 11:15:00 | 1011.80 | 994.28 | 991.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-08 10:15:00 | 1000.05 | 1001.71 | 997.12 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-08 10:15:00 | 1000.05 | 1001.71 | 997.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 10:15:00 | 1000.05 | 1001.71 | 997.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-08 10:30:00 | 1000.45 | 1001.71 | 997.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 12:15:00 | 999.50 | 1000.92 | 997.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-08 12:45:00 | 998.25 | 1000.92 | 997.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 13:15:00 | 999.40 | 1000.62 | 997.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-08 14:00:00 | 999.40 | 1000.62 | 997.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 14:15:00 | 998.10 | 1000.12 | 997.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-08 15:00:00 | 998.10 | 1000.12 | 997.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 15:15:00 | 1000.00 | 1000.09 | 997.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-09 09:15:00 | 1001.55 | 1000.09 | 997.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 09:15:00 | 1000.95 | 1000.26 | 998.22 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-09 11:15:00 | 1006.30 | 1001.25 | 998.86 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-09 12:45:00 | 1006.40 | 1003.29 | 1000.27 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-09 13:15:00 | 1009.10 | 1003.29 | 1000.27 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-09 15:00:00 | 1008.30 | 1005.07 | 1001.65 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 09:15:00 | 1023.25 | 1009.49 | 1004.30 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-10-14 10:15:00 | 1000.50 | 1008.96 | 1009.93 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-14 10:15:00 | 1000.50 | 1008.96 | 1009.93 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-14 10:15:00 | 1000.50 | 1008.96 | 1009.93 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-14 10:15:00 | 1000.50 | 1008.96 | 1009.93 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 28 — SELL (started 2025-10-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-14 10:15:00 | 1000.50 | 1008.96 | 1009.93 | EMA200 below EMA400 |

### Cycle 29 — BUY (started 2025-10-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-15 14:15:00 | 1016.00 | 1007.38 | 1007.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-16 09:15:00 | 1030.40 | 1013.65 | 1010.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-17 09:15:00 | 1020.45 | 1024.72 | 1019.15 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-17 09:15:00 | 1020.45 | 1024.72 | 1019.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 09:15:00 | 1020.45 | 1024.72 | 1019.15 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-17 14:30:00 | 1027.25 | 1023.12 | 1020.29 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-20 10:15:00 | 1011.85 | 1019.41 | 1019.22 | SL hit (close<static) qty=1.00 sl=1012.55 alert=retest2 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-20 12:30:00 | 1027.70 | 1021.41 | 1020.17 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-20 14:15:00 | 1028.85 | 1022.33 | 1020.70 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-20 15:00:00 | 1027.50 | 1023.36 | 1021.32 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 15:15:00 | 1023.00 | 1023.29 | 1021.47 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-21 13:45:00 | 1034.15 | 1027.28 | 1023.45 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-23 09:45:00 | 1034.60 | 1031.28 | 1026.01 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-30 13:15:00 | 1048.25 | 1051.06 | 1051.13 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-30 13:15:00 | 1048.25 | 1051.06 | 1051.13 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-30 13:15:00 | 1048.25 | 1051.06 | 1051.13 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-30 13:15:00 | 1048.25 | 1051.06 | 1051.13 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-30 13:15:00 | 1048.25 | 1051.06 | 1051.13 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 30 — SELL (started 2025-10-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-30 13:15:00 | 1048.25 | 1051.06 | 1051.13 | EMA200 below EMA400 |

### Cycle 31 — BUY (started 2025-10-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-30 14:15:00 | 1058.75 | 1052.60 | 1051.82 | EMA200 above EMA400 |

### Cycle 32 — SELL (started 2025-10-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-31 11:15:00 | 1046.40 | 1050.75 | 1051.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-31 12:15:00 | 1046.00 | 1049.80 | 1050.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-04 09:15:00 | 1036.30 | 1032.81 | 1038.94 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-04 09:15:00 | 1036.30 | 1032.81 | 1038.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 09:15:00 | 1036.30 | 1032.81 | 1038.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-04 10:00:00 | 1036.30 | 1032.81 | 1038.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 10:15:00 | 1034.00 | 1033.05 | 1038.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-04 10:45:00 | 1033.60 | 1033.05 | 1038.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 09:15:00 | 1017.60 | 1026.79 | 1032.64 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-06 10:15:00 | 1014.00 | 1026.79 | 1032.64 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-10 14:15:00 | 1032.70 | 1030.73 | 1030.63 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 33 — BUY (started 2025-11-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-10 14:15:00 | 1032.70 | 1030.73 | 1030.63 | EMA200 above EMA400 |

### Cycle 34 — SELL (started 2025-11-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-10 15:15:00 | 1026.80 | 1029.94 | 1030.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-11 09:15:00 | 1020.30 | 1028.01 | 1029.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-12 14:15:00 | 1006.80 | 1005.78 | 1012.52 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-12 15:00:00 | 1006.80 | 1005.78 | 1012.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 09:15:00 | 997.70 | 985.54 | 991.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-17 10:00:00 | 997.70 | 985.54 | 991.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 10:15:00 | 996.00 | 987.63 | 992.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-17 10:30:00 | 997.10 | 987.63 | 992.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 13:15:00 | 995.70 | 991.30 | 993.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-17 13:45:00 | 994.70 | 991.30 | 993.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 14:15:00 | 994.70 | 991.98 | 993.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-17 15:15:00 | 994.90 | 991.98 | 993.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 09:15:00 | 985.50 | 991.15 | 992.65 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-18 12:00:00 | 983.90 | 988.72 | 991.22 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-18 13:15:00 | 984.40 | 987.99 | 990.67 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-18 14:15:00 | 984.10 | 987.44 | 990.17 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-18 14:45:00 | 984.10 | 986.61 | 989.55 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 09:15:00 | 984.10 | 985.53 | 988.51 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-11-20 10:15:00 | 991.10 | 988.79 | 988.63 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-20 10:15:00 | 991.10 | 988.79 | 988.63 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-20 10:15:00 | 991.10 | 988.79 | 988.63 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-20 10:15:00 | 991.10 | 988.79 | 988.63 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 35 — BUY (started 2025-11-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-20 10:15:00 | 991.10 | 988.79 | 988.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-20 15:15:00 | 1000.00 | 991.54 | 989.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-24 13:15:00 | 1011.90 | 1013.19 | 1006.90 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-24 13:45:00 | 1008.90 | 1013.19 | 1006.90 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 15:15:00 | 1007.60 | 1011.72 | 1007.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-25 09:15:00 | 1000.20 | 1011.72 | 1007.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 09:15:00 | 1001.00 | 1009.58 | 1006.73 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-26 12:45:00 | 1017.30 | 1010.91 | 1008.91 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-26 14:15:00 | 1017.10 | 1011.63 | 1009.42 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-27 11:45:00 | 1017.10 | 1013.75 | 1011.41 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-27 13:15:00 | 1016.20 | 1013.86 | 1011.67 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 09:15:00 | 1018.00 | 1016.11 | 1013.57 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-28 10:15:00 | 1029.60 | 1016.11 | 1013.57 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-28 13:00:00 | 1025.20 | 1020.65 | 1016.51 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-01 15:15:00 | 1027.20 | 1025.98 | 1023.23 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-02 10:45:00 | 1023.10 | 1024.77 | 1023.32 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 11:15:00 | 1019.00 | 1023.61 | 1022.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-02 11:30:00 | 1019.20 | 1023.61 | 1022.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-12-02 12:15:00 | 1017.60 | 1022.41 | 1022.45 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-02 12:15:00 | 1017.60 | 1022.41 | 1022.45 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-02 12:15:00 | 1017.60 | 1022.41 | 1022.45 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-02 12:15:00 | 1017.60 | 1022.41 | 1022.45 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-02 12:15:00 | 1017.60 | 1022.41 | 1022.45 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-02 12:15:00 | 1017.60 | 1022.41 | 1022.45 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-02 12:15:00 | 1017.60 | 1022.41 | 1022.45 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-02 12:15:00 | 1017.60 | 1022.41 | 1022.45 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 36 — SELL (started 2025-12-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-02 12:15:00 | 1017.60 | 1022.41 | 1022.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-02 13:15:00 | 1015.50 | 1021.03 | 1021.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-02 14:15:00 | 1021.10 | 1021.04 | 1021.75 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-02 14:15:00 | 1021.10 | 1021.04 | 1021.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 14:15:00 | 1021.10 | 1021.04 | 1021.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-02 15:00:00 | 1021.10 | 1021.04 | 1021.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 15:15:00 | 1022.00 | 1021.23 | 1021.77 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-03 09:15:00 | 1015.00 | 1021.23 | 1021.77 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-03 14:15:00 | 1025.40 | 1020.61 | 1020.90 | SL hit (close>static) qty=1.00 sl=1024.50 alert=retest2 |

### Cycle 37 — BUY (started 2025-12-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-03 15:15:00 | 1025.00 | 1021.49 | 1021.27 | EMA200 above EMA400 |

### Cycle 38 — SELL (started 2025-12-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-04 09:15:00 | 1011.80 | 1019.55 | 1020.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-05 10:15:00 | 1009.90 | 1014.07 | 1016.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-05 13:15:00 | 1014.10 | 1012.61 | 1015.18 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-05 13:15:00 | 1014.10 | 1012.61 | 1015.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 13:15:00 | 1014.10 | 1012.61 | 1015.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-05 14:00:00 | 1014.10 | 1012.61 | 1015.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 10:15:00 | 1006.10 | 1009.97 | 1013.01 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-08 11:15:00 | 1005.10 | 1009.97 | 1013.01 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-08 12:00:00 | 1005.50 | 1009.08 | 1012.33 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-10 10:15:00 | 1023.70 | 1006.98 | 1006.06 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-10 10:15:00 | 1023.70 | 1006.98 | 1006.06 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 39 — BUY (started 2025-12-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-10 10:15:00 | 1023.70 | 1006.98 | 1006.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-12 09:15:00 | 1052.50 | 1029.92 | 1021.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-15 15:15:00 | 1058.60 | 1059.28 | 1049.02 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-16 09:15:00 | 1046.90 | 1059.28 | 1049.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 09:15:00 | 1041.90 | 1055.80 | 1048.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 10:00:00 | 1041.90 | 1055.80 | 1048.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 10:15:00 | 1045.30 | 1053.70 | 1048.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 11:00:00 | 1045.30 | 1053.70 | 1048.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 12:15:00 | 1050.00 | 1052.10 | 1048.28 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-16 13:30:00 | 1053.10 | 1052.74 | 1048.92 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-16 14:45:00 | 1051.90 | 1052.41 | 1049.12 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-16 15:15:00 | 1052.00 | 1052.41 | 1049.12 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-17 09:15:00 | 1040.60 | 1049.98 | 1048.58 | SL hit (close<static) qty=1.00 sl=1046.60 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-17 09:15:00 | 1040.60 | 1049.98 | 1048.58 | SL hit (close<static) qty=1.00 sl=1046.60 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-17 09:15:00 | 1040.60 | 1049.98 | 1048.58 | SL hit (close<static) qty=1.00 sl=1046.60 alert=retest2 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-17 14:30:00 | 1052.00 | 1049.10 | 1048.39 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 09:15:00 | 1043.50 | 1048.51 | 1048.28 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-12-18 09:15:00 | 1043.50 | 1048.51 | 1048.28 | SL hit (close<static) qty=1.00 sl=1046.60 alert=retest2 |
| ALERT3_SIDEWAYS | 2025-12-18 10:00:00 | 1043.50 | 1048.51 | 1048.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 40 — SELL (started 2025-12-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-18 10:15:00 | 1046.10 | 1048.02 | 1048.08 | EMA200 below EMA400 |

### Cycle 41 — BUY (started 2025-12-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-18 12:15:00 | 1053.80 | 1049.06 | 1048.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-18 14:15:00 | 1063.20 | 1052.28 | 1050.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-19 11:15:00 | 1051.40 | 1054.23 | 1052.01 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-19 11:15:00 | 1051.40 | 1054.23 | 1052.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 11:15:00 | 1051.40 | 1054.23 | 1052.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-19 12:00:00 | 1051.40 | 1054.23 | 1052.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 12:15:00 | 1050.80 | 1053.54 | 1051.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-19 12:30:00 | 1050.30 | 1053.54 | 1051.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 13:15:00 | 1051.00 | 1053.03 | 1051.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-19 13:30:00 | 1053.30 | 1053.03 | 1051.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 14:15:00 | 1050.00 | 1052.43 | 1051.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-19 15:00:00 | 1050.00 | 1052.43 | 1051.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 15:15:00 | 1050.50 | 1052.04 | 1051.55 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-22 09:15:00 | 1058.30 | 1052.04 | 1051.55 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-23 09:15:00 | 1046.00 | 1056.55 | 1055.66 | SL hit (close<static) qty=1.00 sl=1049.70 alert=retest2 |

### Cycle 42 — SELL (started 2025-12-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-23 10:15:00 | 1046.80 | 1054.60 | 1054.85 | EMA200 below EMA400 |

### Cycle 43 — BUY (started 2025-12-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-24 11:15:00 | 1062.60 | 1054.22 | 1053.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-26 09:15:00 | 1074.70 | 1061.44 | 1057.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-26 12:15:00 | 1065.30 | 1066.41 | 1061.09 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-26 12:45:00 | 1064.60 | 1066.41 | 1061.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 14:15:00 | 1064.60 | 1066.14 | 1061.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-26 14:30:00 | 1062.50 | 1066.14 | 1061.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 09:15:00 | 1062.70 | 1064.81 | 1061.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-29 09:30:00 | 1061.90 | 1064.81 | 1061.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 10:15:00 | 1057.80 | 1063.41 | 1061.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-29 11:00:00 | 1057.80 | 1063.41 | 1061.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 11:15:00 | 1065.00 | 1063.73 | 1061.92 | EMA400 retest candle locked (from upside) |

### Cycle 44 — SELL (started 2025-12-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-29 15:15:00 | 1056.60 | 1060.86 | 1061.06 | EMA200 below EMA400 |

### Cycle 45 — BUY (started 2025-12-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-30 09:15:00 | 1063.50 | 1061.39 | 1061.28 | EMA200 above EMA400 |

### Cycle 46 — SELL (started 2025-12-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-30 10:15:00 | 1052.00 | 1059.51 | 1060.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-30 11:15:00 | 1049.30 | 1057.47 | 1059.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-31 10:15:00 | 1056.90 | 1053.87 | 1056.27 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-31 10:15:00 | 1056.90 | 1053.87 | 1056.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 10:15:00 | 1056.90 | 1053.87 | 1056.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 11:00:00 | 1056.90 | 1053.87 | 1056.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 11:15:00 | 1055.00 | 1054.10 | 1056.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 12:15:00 | 1058.10 | 1054.10 | 1056.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 12:15:00 | 1059.90 | 1055.26 | 1056.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 13:00:00 | 1059.90 | 1055.26 | 1056.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 13:15:00 | 1055.90 | 1055.39 | 1056.44 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-31 14:30:00 | 1053.90 | 1055.03 | 1056.18 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-31 15:00:00 | 1053.60 | 1055.03 | 1056.18 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-01 09:15:00 | 1048.80 | 1055.02 | 1056.08 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-01 12:15:00 | 1059.00 | 1057.02 | 1056.78 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-01 12:15:00 | 1059.00 | 1057.02 | 1056.78 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-01 12:15:00 | 1059.00 | 1057.02 | 1056.78 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 47 — BUY (started 2026-01-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-01 12:15:00 | 1059.00 | 1057.02 | 1056.78 | EMA200 above EMA400 |

### Cycle 48 — SELL (started 2026-01-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-02 11:15:00 | 1053.00 | 1056.52 | 1056.77 | EMA200 below EMA400 |

### Cycle 49 — BUY (started 2026-01-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-02 15:15:00 | 1060.00 | 1057.30 | 1057.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-05 09:15:00 | 1088.00 | 1063.44 | 1059.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-05 15:15:00 | 1075.60 | 1076.38 | 1069.53 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-06 09:15:00 | 1075.50 | 1076.38 | 1069.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 14:15:00 | 1076.90 | 1077.49 | 1073.38 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-08 10:30:00 | 1087.00 | 1076.53 | 1074.78 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-09 09:15:00 | 1084.10 | 1080.54 | 1077.75 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-09 09:45:00 | 1085.50 | 1081.33 | 1078.37 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-12 10:00:00 | 1082.80 | 1087.03 | 1083.73 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 10:15:00 | 1080.10 | 1085.64 | 1083.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-12 10:45:00 | 1081.50 | 1085.64 | 1083.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 11:15:00 | 1078.00 | 1084.11 | 1082.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-12 11:45:00 | 1074.90 | 1084.11 | 1082.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 13:15:00 | 1088.50 | 1086.54 | 1084.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-12 14:00:00 | 1088.50 | 1086.54 | 1084.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 14:15:00 | 1082.90 | 1085.81 | 1084.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-12 15:00:00 | 1082.90 | 1085.81 | 1084.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 15:15:00 | 1085.00 | 1085.65 | 1084.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-13 09:15:00 | 1080.50 | 1085.65 | 1084.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2026-01-13 09:15:00 | 1073.00 | 1083.12 | 1083.24 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-13 09:15:00 | 1073.00 | 1083.12 | 1083.24 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-13 09:15:00 | 1073.00 | 1083.12 | 1083.24 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-13 09:15:00 | 1073.00 | 1083.12 | 1083.24 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 50 — SELL (started 2026-01-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-13 09:15:00 | 1073.00 | 1083.12 | 1083.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-14 13:15:00 | 1070.50 | 1077.92 | 1079.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-16 10:15:00 | 1071.50 | 1071.37 | 1075.46 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-16 11:00:00 | 1071.50 | 1071.37 | 1075.46 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 15:15:00 | 1075.00 | 1069.79 | 1072.75 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-19 09:15:00 | 1060.00 | 1069.79 | 1072.75 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-19 15:00:00 | 1065.00 | 1064.66 | 1068.22 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-20 09:45:00 | 1066.20 | 1065.39 | 1067.96 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-20 10:45:00 | 1067.00 | 1065.59 | 1067.81 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 11:15:00 | 1070.70 | 1066.61 | 1068.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-20 11:30:00 | 1073.30 | 1066.61 | 1068.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 12:15:00 | 1067.00 | 1066.69 | 1067.98 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-21 09:15:00 | 1056.40 | 1066.95 | 1067.78 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-21 09:15:00 | 1074.60 | 1068.48 | 1068.40 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-21 09:15:00 | 1074.60 | 1068.48 | 1068.40 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-21 09:15:00 | 1074.60 | 1068.48 | 1068.40 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-21 09:15:00 | 1074.60 | 1068.48 | 1068.40 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-21 09:15:00 | 1074.60 | 1068.48 | 1068.40 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 51 — BUY (started 2026-01-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-21 09:15:00 | 1074.60 | 1068.48 | 1068.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-21 12:15:00 | 1081.40 | 1072.78 | 1070.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-22 09:15:00 | 1074.00 | 1075.90 | 1072.99 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-22 10:00:00 | 1074.00 | 1075.90 | 1072.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 10:15:00 | 1071.20 | 1074.96 | 1072.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-22 11:00:00 | 1071.20 | 1074.96 | 1072.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 11:15:00 | 1074.60 | 1074.89 | 1072.98 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-22 12:45:00 | 1077.90 | 1075.19 | 1073.29 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-23 09:45:00 | 1079.80 | 1078.42 | 1075.60 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-23 12:00:00 | 1075.00 | 1077.20 | 1075.50 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-23 13:15:00 | 1075.00 | 1076.70 | 1075.43 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 13:15:00 | 1070.00 | 1075.36 | 1074.93 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-01-23 13:15:00 | 1070.00 | 1075.36 | 1074.93 | SL hit (close<static) qty=1.00 sl=1070.10 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-23 13:15:00 | 1070.00 | 1075.36 | 1074.93 | SL hit (close<static) qty=1.00 sl=1070.10 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-23 13:15:00 | 1070.00 | 1075.36 | 1074.93 | SL hit (close<static) qty=1.00 sl=1070.10 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-23 13:15:00 | 1070.00 | 1075.36 | 1074.93 | SL hit (close<static) qty=1.00 sl=1070.10 alert=retest2 |
| ALERT3_SIDEWAYS | 2026-01-23 14:00:00 | 1070.00 | 1075.36 | 1074.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 52 — SELL (started 2026-01-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-23 14:15:00 | 1056.60 | 1071.61 | 1073.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-23 15:15:00 | 1046.00 | 1066.49 | 1070.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-27 14:15:00 | 1073.20 | 1062.06 | 1065.66 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-27 14:15:00 | 1073.20 | 1062.06 | 1065.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 14:15:00 | 1073.20 | 1062.06 | 1065.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-27 15:00:00 | 1073.20 | 1062.06 | 1065.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 15:15:00 | 1055.00 | 1060.65 | 1064.69 | EMA400 retest candle locked (from downside) |

### Cycle 53 — BUY (started 2026-01-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-29 11:15:00 | 1078.10 | 1067.92 | 1066.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-29 12:15:00 | 1083.00 | 1070.93 | 1068.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-01 09:15:00 | 1093.70 | 1105.82 | 1094.94 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-01 09:15:00 | 1093.70 | 1105.82 | 1094.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 09:15:00 | 1093.70 | 1105.82 | 1094.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 10:00:00 | 1093.70 | 1105.82 | 1094.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 10:15:00 | 1095.20 | 1103.69 | 1094.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 10:45:00 | 1092.40 | 1103.69 | 1094.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 11:15:00 | 1101.60 | 1103.28 | 1095.57 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-02 09:15:00 | 1107.70 | 1097.78 | 1095.36 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-02 12:00:00 | 1112.00 | 1103.55 | 1098.90 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-11 10:15:00 | 1150.90 | 1165.18 | 1166.01 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-11 10:15:00 | 1150.90 | 1165.18 | 1166.01 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 54 — SELL (started 2026-02-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-11 10:15:00 | 1150.90 | 1165.18 | 1166.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-12 10:15:00 | 1127.00 | 1149.00 | 1156.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-12 13:15:00 | 1146.20 | 1144.12 | 1151.94 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-12 14:00:00 | 1146.20 | 1144.12 | 1151.94 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 15:15:00 | 1149.10 | 1144.89 | 1150.92 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-13 09:15:00 | 1131.10 | 1144.89 | 1150.92 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-13 14:00:00 | 1141.60 | 1142.05 | 1146.88 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-16 09:15:00 | 1131.20 | 1143.50 | 1146.75 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-16 13:45:00 | 1141.60 | 1140.76 | 1143.69 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 14:15:00 | 1147.30 | 1142.07 | 1144.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-16 15:00:00 | 1147.30 | 1142.07 | 1144.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2026-02-16 15:15:00 | 1160.00 | 1145.65 | 1145.47 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-16 15:15:00 | 1160.00 | 1145.65 | 1145.47 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-16 15:15:00 | 1160.00 | 1145.65 | 1145.47 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-16 15:15:00 | 1160.00 | 1145.65 | 1145.47 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 55 — BUY (started 2026-02-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-16 15:15:00 | 1160.00 | 1145.65 | 1145.47 | EMA200 above EMA400 |

### Cycle 56 — SELL (started 2026-02-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-17 09:15:00 | 1137.90 | 1144.10 | 1144.78 | EMA200 below EMA400 |

### Cycle 57 — BUY (started 2026-02-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-17 13:15:00 | 1159.30 | 1145.75 | 1145.07 | EMA200 above EMA400 |

### Cycle 58 — SELL (started 2026-02-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-19 09:15:00 | 1143.50 | 1147.64 | 1148.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-19 11:15:00 | 1133.80 | 1143.49 | 1145.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-23 09:15:00 | 1131.70 | 1120.54 | 1127.37 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-23 09:15:00 | 1131.70 | 1120.54 | 1127.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 09:15:00 | 1131.70 | 1120.54 | 1127.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-23 10:00:00 | 1131.70 | 1120.54 | 1127.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 10:15:00 | 1123.80 | 1121.19 | 1127.04 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-23 11:15:00 | 1118.50 | 1121.19 | 1127.04 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-23 12:15:00 | 1141.50 | 1126.92 | 1128.74 | SL hit (close>static) qty=1.00 sl=1131.70 alert=retest2 |

### Cycle 59 — BUY (started 2026-02-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-23 14:15:00 | 1138.70 | 1131.10 | 1130.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-23 15:15:00 | 1149.90 | 1134.86 | 1132.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-24 09:15:00 | 1131.00 | 1134.09 | 1132.09 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-24 09:15:00 | 1131.00 | 1134.09 | 1132.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 09:15:00 | 1131.00 | 1134.09 | 1132.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-24 09:30:00 | 1129.00 | 1134.09 | 1132.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 10:15:00 | 1134.90 | 1134.25 | 1132.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-24 10:30:00 | 1132.40 | 1134.25 | 1132.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 11:15:00 | 1133.00 | 1134.00 | 1132.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-24 12:00:00 | 1133.00 | 1134.00 | 1132.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 12:15:00 | 1133.90 | 1133.98 | 1132.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-24 12:30:00 | 1133.30 | 1133.98 | 1132.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 15:15:00 | 1151.40 | 1150.05 | 1143.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-26 09:15:00 | 1143.50 | 1150.05 | 1143.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 09:15:00 | 1147.10 | 1149.46 | 1144.27 | EMA400 retest candle locked (from upside) |

### Cycle 60 — SELL (started 2026-02-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-26 13:15:00 | 1122.90 | 1139.63 | 1140.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-27 11:15:00 | 1118.40 | 1131.94 | 1136.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-27 13:15:00 | 1132.10 | 1131.90 | 1135.88 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-27 13:30:00 | 1133.60 | 1131.90 | 1135.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 14:15:00 | 1097.40 | 1069.37 | 1081.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-05 15:00:00 | 1097.40 | 1069.37 | 1081.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 15:15:00 | 1106.90 | 1076.87 | 1083.66 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-06 09:15:00 | 1092.90 | 1076.87 | 1083.66 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-06 09:45:00 | 1096.80 | 1081.96 | 1085.35 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-06 10:30:00 | 1085.00 | 1081.81 | 1084.97 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-06 13:15:00 | 1102.50 | 1089.51 | 1088.03 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-06 13:15:00 | 1102.50 | 1089.51 | 1088.03 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-06 13:15:00 | 1102.50 | 1089.51 | 1088.03 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 61 — BUY (started 2026-03-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-06 13:15:00 | 1102.50 | 1089.51 | 1088.03 | EMA200 above EMA400 |

### Cycle 62 — SELL (started 2026-03-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-09 09:15:00 | 1054.80 | 1082.35 | 1085.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-09 10:15:00 | 1040.00 | 1073.88 | 1081.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-12 10:15:00 | 999.00 | 997.06 | 1011.57 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-12 11:00:00 | 999.00 | 997.06 | 1011.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 15:15:00 | 990.00 | 991.56 | 1003.12 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-13 09:30:00 | 975.60 | 987.31 | 1000.14 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-13 15:00:00 | 972.10 | 982.24 | 992.34 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-18 14:15:00 | 985.00 | 974.71 | 973.77 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-18 14:15:00 | 985.00 | 974.71 | 973.77 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 63 — BUY (started 2026-03-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 14:15:00 | 985.00 | 974.71 | 973.77 | EMA200 above EMA400 |

### Cycle 64 — SELL (started 2026-03-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 09:15:00 | 955.00 | 971.74 | 972.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 13:15:00 | 950.70 | 963.36 | 968.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-20 15:15:00 | 939.90 | 938.91 | 950.04 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-23 09:15:00 | 918.30 | 938.91 | 950.04 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-23 14:15:00 | 872.38 | 897.78 | 921.99 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Stop hit — per-position SL triggered | 2026-03-24 12:15:00 | 892.80 | 886.92 | 906.18 | SL hit (close>ema200) qty=0.50 sl=886.92 alert=retest1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 09:15:00 | 915.00 | 896.25 | 904.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-25 09:30:00 | 917.20 | 896.25 | 904.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 10:15:00 | 917.30 | 900.46 | 905.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-25 10:45:00 | 916.70 | 900.46 | 905.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 65 — BUY (started 2026-03-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 12:15:00 | 921.30 | 909.44 | 909.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 15:15:00 | 935.00 | 920.14 | 914.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-27 09:15:00 | 900.30 | 916.17 | 913.47 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-27 09:15:00 | 900.30 | 916.17 | 913.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 09:15:00 | 900.30 | 916.17 | 913.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 10:00:00 | 900.30 | 916.17 | 913.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 10:15:00 | 906.40 | 914.22 | 912.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 10:30:00 | 901.20 | 914.22 | 912.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 12:15:00 | 929.00 | 917.06 | 914.36 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-27 13:15:00 | 933.50 | 917.06 | 914.36 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-27 14:15:00 | 909.40 | 918.27 | 915.55 | SL hit (close<static) qty=1.00 sl=912.60 alert=retest2 |

### Cycle 66 — SELL (started 2026-03-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-30 09:15:00 | 898.40 | 912.25 | 913.16 | EMA200 below EMA400 |

### Cycle 67 — BUY (started 2026-04-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 09:15:00 | 946.20 | 916.55 | 913.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-01 12:15:00 | 950.70 | 932.34 | 922.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-02 09:15:00 | 918.30 | 933.56 | 926.80 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-02 09:15:00 | 918.30 | 933.56 | 926.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 09:15:00 | 918.30 | 933.56 | 926.80 | EMA400 retest candle locked (from upside) |

### Cycle 68 — SELL (started 2026-04-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-02 13:15:00 | 921.85 | 922.80 | 922.88 | EMA200 below EMA400 |

### Cycle 69 — BUY (started 2026-04-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-02 14:15:00 | 928.30 | 923.90 | 923.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-06 13:15:00 | 936.95 | 928.20 | 925.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-07 09:15:00 | 928.35 | 930.94 | 927.79 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-07 09:15:00 | 928.35 | 930.94 | 927.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-07 09:15:00 | 928.35 | 930.94 | 927.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-07 09:30:00 | 927.65 | 930.94 | 927.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-07 10:15:00 | 930.60 | 930.87 | 928.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-07 10:45:00 | 931.40 | 930.87 | 928.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 979.15 | 989.30 | 981.74 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 15:00:00 | 994.85 | 986.91 | 982.98 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-15 09:15:00 | 999.70 | 986.56 | 983.18 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-15 11:45:00 | 996.15 | 990.52 | 986.12 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-15 13:00:00 | 993.90 | 991.19 | 986.83 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-20 09:15:00 | 1014.70 | 1007.71 | 1002.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-20 09:45:00 | 1009.60 | 1007.71 | 1002.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-20 15:15:00 | 1006.00 | 1011.26 | 1007.21 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-21 10:00:00 | 1014.20 | 1011.85 | 1007.84 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-21 15:15:00 | 998.00 | 1009.27 | 1008.46 | SL hit (close<static) qty=1.00 sl=1003.10 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-22 09:15:00 | 994.50 | 1006.32 | 1007.19 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-22 09:15:00 | 994.50 | 1006.32 | 1007.19 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-22 09:15:00 | 994.50 | 1006.32 | 1007.19 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-22 09:15:00 | 994.50 | 1006.32 | 1007.19 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 70 — SELL (started 2026-04-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-22 09:15:00 | 994.50 | 1006.32 | 1007.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-22 12:15:00 | 992.30 | 1001.33 | 1004.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-24 14:15:00 | 964.15 | 963.60 | 973.28 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-24 14:45:00 | 967.05 | 963.60 | 973.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 13:15:00 | 968.55 | 962.24 | 968.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 14:00:00 | 968.55 | 962.24 | 968.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 14:15:00 | 966.20 | 963.03 | 967.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 15:15:00 | 965.00 | 963.03 | 967.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 15:15:00 | 965.00 | 963.43 | 967.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-28 09:15:00 | 974.95 | 963.43 | 967.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 09:15:00 | 972.20 | 965.18 | 968.11 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-28 11:15:00 | 958.45 | 965.72 | 968.08 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-28 15:00:00 | 957.75 | 961.96 | 965.32 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-29 09:30:00 | 958.80 | 961.90 | 964.73 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-29 12:30:00 | 962.80 | 960.29 | 963.28 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 14:15:00 | 949.00 | 940.31 | 948.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-30 14:30:00 | 947.00 | 940.31 | 948.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 15:15:00 | 929.00 | 938.05 | 946.43 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-04 11:45:00 | 928.50 | 936.09 | 943.50 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-04 14:45:00 | 928.25 | 931.77 | 939.46 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-05 10:15:00 | 910.86 | 924.72 | 934.12 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-05 10:15:00 | 914.66 | 924.72 | 934.12 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-05 11:15:00 | 910.53 | 921.73 | 931.91 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-05 11:15:00 | 909.86 | 921.73 | 931.91 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-05-05 12:15:00 | 922.05 | 921.80 | 931.01 | SL hit (close>ema200) qty=0.50 sl=921.80 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-05-05 12:15:00 | 922.05 | 921.80 | 931.01 | SL hit (close>ema200) qty=0.50 sl=921.80 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-05-05 12:15:00 | 922.05 | 921.80 | 931.01 | SL hit (close>ema200) qty=0.50 sl=921.80 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-05-05 12:15:00 | 922.05 | 921.80 | 931.01 | SL hit (close>ema200) qty=0.50 sl=921.80 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-06 13:45:00 | 927.90 | 926.57 | 928.28 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-05-06 15:15:00 | 943.30 | 931.17 | 930.16 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-05-06 15:15:00 | 943.30 | 931.17 | 930.16 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-05-06 15:15:00 | 943.30 | 931.17 | 930.16 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 71 — BUY (started 2026-05-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-06 15:15:00 | 943.30 | 931.17 | 930.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-07 13:15:00 | 951.15 | 940.12 | 935.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-08 15:15:00 | 953.00 | 954.94 | 948.22 | EMA200 retest candle locked (from upside) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-05-22 12:30:00 | 983.50 | 2025-05-23 09:15:00 | 998.50 | STOP_HIT | 1.00 | -1.53% |
| SELL | retest2 | 2025-05-23 09:15:00 | 979.05 | 2025-05-23 09:15:00 | 998.50 | STOP_HIT | 1.00 | -1.99% |
| SELL | retest2 | 2025-05-29 12:30:00 | 959.00 | 2025-05-30 14:15:00 | 975.50 | STOP_HIT | 1.00 | -1.72% |
| BUY | retest2 | 2025-06-13 10:30:00 | 1057.35 | 2025-06-18 09:15:00 | 1048.50 | STOP_HIT | 1.00 | -0.84% |
| SELL | retest2 | 2025-06-23 13:30:00 | 1011.05 | 2025-06-25 10:15:00 | 1035.70 | STOP_HIT | 1.00 | -2.44% |
| SELL | retest2 | 2025-06-23 15:15:00 | 1010.10 | 2025-06-25 10:15:00 | 1035.70 | STOP_HIT | 1.00 | -2.53% |
| SELL | retest2 | 2025-06-24 09:45:00 | 1007.65 | 2025-06-25 10:15:00 | 1035.70 | STOP_HIT | 1.00 | -2.78% |
| SELL | retest2 | 2025-06-24 13:30:00 | 1009.60 | 2025-06-25 10:15:00 | 1035.70 | STOP_HIT | 1.00 | -2.59% |
| SELL | retest2 | 2025-06-25 09:30:00 | 1014.10 | 2025-06-25 10:15:00 | 1035.70 | STOP_HIT | 1.00 | -2.13% |
| BUY | retest2 | 2025-07-04 09:15:00 | 1077.90 | 2025-07-07 11:15:00 | 1069.10 | STOP_HIT | 1.00 | -0.82% |
| BUY | retest2 | 2025-07-04 10:00:00 | 1083.00 | 2025-07-07 11:15:00 | 1069.10 | STOP_HIT | 1.00 | -1.28% |
| BUY | retest2 | 2025-07-07 12:30:00 | 1081.00 | 2025-07-08 10:15:00 | 1074.10 | STOP_HIT | 1.00 | -0.64% |
| BUY | retest2 | 2025-07-08 10:15:00 | 1079.50 | 2025-07-08 10:15:00 | 1074.10 | STOP_HIT | 1.00 | -0.50% |
| BUY | retest2 | 2025-07-15 09:45:00 | 1137.20 | 2025-07-21 13:15:00 | 1151.90 | STOP_HIT | 1.00 | 1.29% |
| BUY | retest2 | 2025-07-15 12:30:00 | 1135.50 | 2025-07-21 13:15:00 | 1151.90 | STOP_HIT | 1.00 | 1.44% |
| BUY | retest2 | 2025-07-23 12:45:00 | 1183.10 | 2025-07-25 10:15:00 | 1165.90 | STOP_HIT | 1.00 | -1.45% |
| BUY | retest2 | 2025-07-24 12:15:00 | 1182.70 | 2025-07-25 10:15:00 | 1165.90 | STOP_HIT | 1.00 | -1.42% |
| BUY | retest2 | 2025-07-24 13:00:00 | 1181.20 | 2025-07-25 10:15:00 | 1165.90 | STOP_HIT | 1.00 | -1.30% |
| BUY | retest2 | 2025-07-24 13:45:00 | 1181.50 | 2025-07-25 10:15:00 | 1165.90 | STOP_HIT | 1.00 | -1.32% |
| BUY | retest2 | 2025-07-31 13:45:00 | 1191.10 | 2025-07-31 15:15:00 | 1170.10 | STOP_HIT | 1.00 | -1.76% |
| SELL | retest2 | 2025-08-07 10:15:00 | 1144.80 | 2025-08-08 09:15:00 | 1087.56 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-07 12:45:00 | 1144.00 | 2025-08-08 09:15:00 | 1086.80 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-07 10:15:00 | 1144.80 | 2025-08-11 14:15:00 | 1070.00 | STOP_HIT | 0.50 | 6.53% |
| SELL | retest2 | 2025-08-07 12:45:00 | 1144.00 | 2025-08-11 14:15:00 | 1070.00 | STOP_HIT | 0.50 | 6.47% |
| SELL | retest1 | 2025-08-29 09:15:00 | 1031.10 | 2025-09-01 09:15:00 | 1059.40 | STOP_HIT | 1.00 | -2.74% |
| BUY | retest2 | 2025-09-04 11:15:00 | 1080.00 | 2025-09-05 10:15:00 | 1067.00 | STOP_HIT | 1.00 | -1.20% |
| BUY | retest2 | 2025-09-05 09:30:00 | 1079.20 | 2025-09-05 10:15:00 | 1067.00 | STOP_HIT | 1.00 | -1.13% |
| SELL | retest2 | 2025-09-08 11:15:00 | 1063.90 | 2025-09-15 12:15:00 | 1053.80 | STOP_HIT | 1.00 | 0.95% |
| SELL | retest2 | 2025-10-03 10:30:00 | 988.20 | 2025-10-07 10:15:00 | 1001.80 | STOP_HIT | 1.00 | -1.38% |
| SELL | retest2 | 2025-10-06 09:45:00 | 984.00 | 2025-10-07 10:15:00 | 1001.80 | STOP_HIT | 1.00 | -1.81% |
| BUY | retest2 | 2025-10-09 11:15:00 | 1006.30 | 2025-10-14 10:15:00 | 1000.50 | STOP_HIT | 1.00 | -0.58% |
| BUY | retest2 | 2025-10-09 12:45:00 | 1006.40 | 2025-10-14 10:15:00 | 1000.50 | STOP_HIT | 1.00 | -0.59% |
| BUY | retest2 | 2025-10-09 13:15:00 | 1009.10 | 2025-10-14 10:15:00 | 1000.50 | STOP_HIT | 1.00 | -0.85% |
| BUY | retest2 | 2025-10-09 15:00:00 | 1008.30 | 2025-10-14 10:15:00 | 1000.50 | STOP_HIT | 1.00 | -0.77% |
| BUY | retest2 | 2025-10-17 14:30:00 | 1027.25 | 2025-10-20 10:15:00 | 1011.85 | STOP_HIT | 1.00 | -1.50% |
| BUY | retest2 | 2025-10-20 12:30:00 | 1027.70 | 2025-10-30 13:15:00 | 1048.25 | STOP_HIT | 1.00 | 2.00% |
| BUY | retest2 | 2025-10-20 14:15:00 | 1028.85 | 2025-10-30 13:15:00 | 1048.25 | STOP_HIT | 1.00 | 1.89% |
| BUY | retest2 | 2025-10-20 15:00:00 | 1027.50 | 2025-10-30 13:15:00 | 1048.25 | STOP_HIT | 1.00 | 2.02% |
| BUY | retest2 | 2025-10-21 13:45:00 | 1034.15 | 2025-10-30 13:15:00 | 1048.25 | STOP_HIT | 1.00 | 1.36% |
| BUY | retest2 | 2025-10-23 09:45:00 | 1034.60 | 2025-10-30 13:15:00 | 1048.25 | STOP_HIT | 1.00 | 1.32% |
| SELL | retest2 | 2025-11-06 10:15:00 | 1014.00 | 2025-11-10 14:15:00 | 1032.70 | STOP_HIT | 1.00 | -1.84% |
| SELL | retest2 | 2025-11-18 12:00:00 | 983.90 | 2025-11-20 10:15:00 | 991.10 | STOP_HIT | 1.00 | -0.73% |
| SELL | retest2 | 2025-11-18 13:15:00 | 984.40 | 2025-11-20 10:15:00 | 991.10 | STOP_HIT | 1.00 | -0.68% |
| SELL | retest2 | 2025-11-18 14:15:00 | 984.10 | 2025-11-20 10:15:00 | 991.10 | STOP_HIT | 1.00 | -0.71% |
| SELL | retest2 | 2025-11-18 14:45:00 | 984.10 | 2025-11-20 10:15:00 | 991.10 | STOP_HIT | 1.00 | -0.71% |
| BUY | retest2 | 2025-11-26 12:45:00 | 1017.30 | 2025-12-02 12:15:00 | 1017.60 | STOP_HIT | 1.00 | 0.03% |
| BUY | retest2 | 2025-11-26 14:15:00 | 1017.10 | 2025-12-02 12:15:00 | 1017.60 | STOP_HIT | 1.00 | 0.05% |
| BUY | retest2 | 2025-11-27 11:45:00 | 1017.10 | 2025-12-02 12:15:00 | 1017.60 | STOP_HIT | 1.00 | 0.05% |
| BUY | retest2 | 2025-11-27 13:15:00 | 1016.20 | 2025-12-02 12:15:00 | 1017.60 | STOP_HIT | 1.00 | 0.14% |
| BUY | retest2 | 2025-11-28 10:15:00 | 1029.60 | 2025-12-02 12:15:00 | 1017.60 | STOP_HIT | 1.00 | -1.17% |
| BUY | retest2 | 2025-11-28 13:00:00 | 1025.20 | 2025-12-02 12:15:00 | 1017.60 | STOP_HIT | 1.00 | -0.74% |
| BUY | retest2 | 2025-12-01 15:15:00 | 1027.20 | 2025-12-02 12:15:00 | 1017.60 | STOP_HIT | 1.00 | -0.93% |
| BUY | retest2 | 2025-12-02 10:45:00 | 1023.10 | 2025-12-02 12:15:00 | 1017.60 | STOP_HIT | 1.00 | -0.54% |
| SELL | retest2 | 2025-12-03 09:15:00 | 1015.00 | 2025-12-03 14:15:00 | 1025.40 | STOP_HIT | 1.00 | -1.02% |
| SELL | retest2 | 2025-12-08 11:15:00 | 1005.10 | 2025-12-10 10:15:00 | 1023.70 | STOP_HIT | 1.00 | -1.85% |
| SELL | retest2 | 2025-12-08 12:00:00 | 1005.50 | 2025-12-10 10:15:00 | 1023.70 | STOP_HIT | 1.00 | -1.81% |
| BUY | retest2 | 2025-12-16 13:30:00 | 1053.10 | 2025-12-17 09:15:00 | 1040.60 | STOP_HIT | 1.00 | -1.19% |
| BUY | retest2 | 2025-12-16 14:45:00 | 1051.90 | 2025-12-17 09:15:00 | 1040.60 | STOP_HIT | 1.00 | -1.07% |
| BUY | retest2 | 2025-12-16 15:15:00 | 1052.00 | 2025-12-17 09:15:00 | 1040.60 | STOP_HIT | 1.00 | -1.08% |
| BUY | retest2 | 2025-12-17 14:30:00 | 1052.00 | 2025-12-18 09:15:00 | 1043.50 | STOP_HIT | 1.00 | -0.81% |
| BUY | retest2 | 2025-12-22 09:15:00 | 1058.30 | 2025-12-23 09:15:00 | 1046.00 | STOP_HIT | 1.00 | -1.16% |
| SELL | retest2 | 2025-12-31 14:30:00 | 1053.90 | 2026-01-01 12:15:00 | 1059.00 | STOP_HIT | 1.00 | -0.48% |
| SELL | retest2 | 2025-12-31 15:00:00 | 1053.60 | 2026-01-01 12:15:00 | 1059.00 | STOP_HIT | 1.00 | -0.51% |
| SELL | retest2 | 2026-01-01 09:15:00 | 1048.80 | 2026-01-01 12:15:00 | 1059.00 | STOP_HIT | 1.00 | -0.97% |
| BUY | retest2 | 2026-01-08 10:30:00 | 1087.00 | 2026-01-13 09:15:00 | 1073.00 | STOP_HIT | 1.00 | -1.29% |
| BUY | retest2 | 2026-01-09 09:15:00 | 1084.10 | 2026-01-13 09:15:00 | 1073.00 | STOP_HIT | 1.00 | -1.02% |
| BUY | retest2 | 2026-01-09 09:45:00 | 1085.50 | 2026-01-13 09:15:00 | 1073.00 | STOP_HIT | 1.00 | -1.15% |
| BUY | retest2 | 2026-01-12 10:00:00 | 1082.80 | 2026-01-13 09:15:00 | 1073.00 | STOP_HIT | 1.00 | -0.91% |
| SELL | retest2 | 2026-01-19 09:15:00 | 1060.00 | 2026-01-21 09:15:00 | 1074.60 | STOP_HIT | 1.00 | -1.38% |
| SELL | retest2 | 2026-01-19 15:00:00 | 1065.00 | 2026-01-21 09:15:00 | 1074.60 | STOP_HIT | 1.00 | -0.90% |
| SELL | retest2 | 2026-01-20 09:45:00 | 1066.20 | 2026-01-21 09:15:00 | 1074.60 | STOP_HIT | 1.00 | -0.79% |
| SELL | retest2 | 2026-01-20 10:45:00 | 1067.00 | 2026-01-21 09:15:00 | 1074.60 | STOP_HIT | 1.00 | -0.71% |
| SELL | retest2 | 2026-01-21 09:15:00 | 1056.40 | 2026-01-21 09:15:00 | 1074.60 | STOP_HIT | 1.00 | -1.72% |
| BUY | retest2 | 2026-01-22 12:45:00 | 1077.90 | 2026-01-23 13:15:00 | 1070.00 | STOP_HIT | 1.00 | -0.73% |
| BUY | retest2 | 2026-01-23 09:45:00 | 1079.80 | 2026-01-23 13:15:00 | 1070.00 | STOP_HIT | 1.00 | -0.91% |
| BUY | retest2 | 2026-01-23 12:00:00 | 1075.00 | 2026-01-23 13:15:00 | 1070.00 | STOP_HIT | 1.00 | -0.47% |
| BUY | retest2 | 2026-01-23 13:15:00 | 1075.00 | 2026-01-23 13:15:00 | 1070.00 | STOP_HIT | 1.00 | -0.47% |
| BUY | retest2 | 2026-02-02 09:15:00 | 1107.70 | 2026-02-11 10:15:00 | 1150.90 | STOP_HIT | 1.00 | 3.90% |
| BUY | retest2 | 2026-02-02 12:00:00 | 1112.00 | 2026-02-11 10:15:00 | 1150.90 | STOP_HIT | 1.00 | 3.50% |
| SELL | retest2 | 2026-02-13 09:15:00 | 1131.10 | 2026-02-16 15:15:00 | 1160.00 | STOP_HIT | 1.00 | -2.56% |
| SELL | retest2 | 2026-02-13 14:00:00 | 1141.60 | 2026-02-16 15:15:00 | 1160.00 | STOP_HIT | 1.00 | -1.61% |
| SELL | retest2 | 2026-02-16 09:15:00 | 1131.20 | 2026-02-16 15:15:00 | 1160.00 | STOP_HIT | 1.00 | -2.55% |
| SELL | retest2 | 2026-02-16 13:45:00 | 1141.60 | 2026-02-16 15:15:00 | 1160.00 | STOP_HIT | 1.00 | -1.61% |
| SELL | retest2 | 2026-02-23 11:15:00 | 1118.50 | 2026-02-23 12:15:00 | 1141.50 | STOP_HIT | 1.00 | -2.06% |
| SELL | retest2 | 2026-03-06 09:15:00 | 1092.90 | 2026-03-06 13:15:00 | 1102.50 | STOP_HIT | 1.00 | -0.88% |
| SELL | retest2 | 2026-03-06 09:45:00 | 1096.80 | 2026-03-06 13:15:00 | 1102.50 | STOP_HIT | 1.00 | -0.52% |
| SELL | retest2 | 2026-03-06 10:30:00 | 1085.00 | 2026-03-06 13:15:00 | 1102.50 | STOP_HIT | 1.00 | -1.61% |
| SELL | retest2 | 2026-03-13 09:30:00 | 975.60 | 2026-03-18 14:15:00 | 985.00 | STOP_HIT | 1.00 | -0.96% |
| SELL | retest2 | 2026-03-13 15:00:00 | 972.10 | 2026-03-18 14:15:00 | 985.00 | STOP_HIT | 1.00 | -1.33% |
| SELL | retest1 | 2026-03-23 09:15:00 | 918.30 | 2026-03-23 14:15:00 | 872.38 | PARTIAL | 0.50 | 5.00% |
| SELL | retest1 | 2026-03-23 09:15:00 | 918.30 | 2026-03-24 12:15:00 | 892.80 | STOP_HIT | 0.50 | 2.78% |
| BUY | retest2 | 2026-03-27 13:15:00 | 933.50 | 2026-03-27 14:15:00 | 909.40 | STOP_HIT | 1.00 | -2.58% |
| BUY | retest2 | 2026-04-13 15:00:00 | 994.85 | 2026-04-21 15:15:00 | 998.00 | STOP_HIT | 1.00 | 0.32% |
| BUY | retest2 | 2026-04-15 09:15:00 | 999.70 | 2026-04-22 09:15:00 | 994.50 | STOP_HIT | 1.00 | -0.52% |
| BUY | retest2 | 2026-04-15 11:45:00 | 996.15 | 2026-04-22 09:15:00 | 994.50 | STOP_HIT | 1.00 | -0.17% |
| BUY | retest2 | 2026-04-15 13:00:00 | 993.90 | 2026-04-22 09:15:00 | 994.50 | STOP_HIT | 1.00 | 0.06% |
| BUY | retest2 | 2026-04-21 10:00:00 | 1014.20 | 2026-04-22 09:15:00 | 994.50 | STOP_HIT | 1.00 | -1.94% |
| SELL | retest2 | 2026-04-28 11:15:00 | 958.45 | 2026-05-05 10:15:00 | 910.86 | PARTIAL | 0.50 | 4.97% |
| SELL | retest2 | 2026-04-28 15:00:00 | 957.75 | 2026-05-05 10:15:00 | 914.66 | PARTIAL | 0.50 | 4.50% |
| SELL | retest2 | 2026-04-29 09:30:00 | 958.80 | 2026-05-05 11:15:00 | 910.53 | PARTIAL | 0.50 | 5.03% |
| SELL | retest2 | 2026-04-29 12:30:00 | 962.80 | 2026-05-05 11:15:00 | 909.86 | PARTIAL | 0.50 | 5.50% |
| SELL | retest2 | 2026-04-28 11:15:00 | 958.45 | 2026-05-05 12:15:00 | 922.05 | STOP_HIT | 0.50 | 3.80% |
| SELL | retest2 | 2026-04-28 15:00:00 | 957.75 | 2026-05-05 12:15:00 | 922.05 | STOP_HIT | 0.50 | 3.73% |
| SELL | retest2 | 2026-04-29 09:30:00 | 958.80 | 2026-05-05 12:15:00 | 922.05 | STOP_HIT | 0.50 | 3.83% |
| SELL | retest2 | 2026-04-29 12:30:00 | 962.80 | 2026-05-05 12:15:00 | 922.05 | STOP_HIT | 0.50 | 4.23% |
| SELL | retest2 | 2026-05-04 11:45:00 | 928.50 | 2026-05-06 15:15:00 | 943.30 | STOP_HIT | 1.00 | -1.59% |
| SELL | retest2 | 2026-05-04 14:45:00 | 928.25 | 2026-05-06 15:15:00 | 943.30 | STOP_HIT | 1.00 | -1.62% |
| SELL | retest2 | 2026-05-06 13:45:00 | 927.90 | 2026-05-06 15:15:00 | 943.30 | STOP_HIT | 1.00 | -1.66% |
