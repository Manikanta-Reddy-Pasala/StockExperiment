# HDFC Asset Management Company Ltd. (HDFCAMC)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 2843.90
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 223 |
| ALERT1 | 155 |
| ALERT2 | 152 |
| ALERT2_SKIP | 76 |
| ALERT3 | 469 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 4 |
| ENTRY2 | 181 |
| PARTIAL | 8 |
| TARGET_HIT | 5 |
| STOP_HIT | 179 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 192 (incl. partial bookings)
- **Trades open at end:** 1
- **Winners / losers:** 64 / 128
- **Target hits / Stop hits / Partials:** 5 / 179 / 8
- **Avg / median % per leg:** -0.02% / -0.81%
- **Sum % (uncompounded):** -3.88%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 83 | 42 | 50.6% | 5 | 78 | 0 | 0.68% | 56.7% |
| BUY @ 2nd Alert (retest1) | 3 | 1 | 33.3% | 0 | 3 | 0 | -0.36% | -1.1% |
| BUY @ 3rd Alert (retest2) | 80 | 41 | 51.2% | 5 | 75 | 0 | 0.72% | 57.8% |
| SELL (all) | 109 | 22 | 20.2% | 0 | 101 | 8 | -0.56% | -60.6% |
| SELL @ 2nd Alert (retest1) | 2 | 2 | 100.0% | 0 | 1 | 1 | 3.80% | 7.6% |
| SELL @ 3rd Alert (retest2) | 107 | 20 | 18.7% | 0 | 100 | 7 | -0.64% | -68.2% |
| retest1 (combined) | 5 | 3 | 60.0% | 0 | 4 | 1 | 1.31% | 6.5% |
| retest2 (combined) | 187 | 61 | 32.6% | 5 | 175 | 7 | -0.06% | -10.4% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-05-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-15 10:15:00 | 926.58 | 918.48 | 917.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-05-16 09:15:00 | 929.18 | 923.21 | 920.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-05-16 13:15:00 | 924.65 | 924.76 | 922.47 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-05-16 14:00:00 | 924.65 | 924.76 | 922.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-16 14:15:00 | 919.93 | 923.79 | 922.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-16 15:00:00 | 919.93 | 923.79 | 922.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-16 15:15:00 | 920.28 | 923.09 | 922.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-17 09:15:00 | 920.88 | 923.09 | 922.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-17 10:15:00 | 921.18 | 922.15 | 921.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-17 10:30:00 | 918.35 | 922.15 | 921.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 2 — SELL (started 2023-05-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-17 11:15:00 | 908.58 | 919.43 | 920.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-05-17 13:15:00 | 906.43 | 914.93 | 918.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-05-19 12:15:00 | 902.00 | 900.22 | 906.37 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-05-19 13:00:00 | 902.00 | 900.22 | 906.37 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-19 14:15:00 | 904.53 | 901.60 | 905.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-05-19 15:00:00 | 904.53 | 901.60 | 905.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-19 15:15:00 | 904.50 | 902.18 | 905.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-05-22 09:15:00 | 907.03 | 902.18 | 905.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-22 09:15:00 | 905.98 | 902.94 | 905.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-05-22 09:45:00 | 905.60 | 902.94 | 905.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-22 10:15:00 | 901.23 | 902.60 | 905.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-05-22 10:30:00 | 906.90 | 902.60 | 905.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-25 10:15:00 | 880.75 | 877.68 | 883.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-05-25 10:30:00 | 881.25 | 877.68 | 883.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-25 11:15:00 | 880.58 | 878.26 | 882.99 | EMA400 retest candle locked (from downside) |

### Cycle 3 — BUY (started 2023-05-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-26 10:15:00 | 900.90 | 887.40 | 885.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-05-26 11:15:00 | 914.00 | 892.72 | 888.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-01 13:15:00 | 975.93 | 978.87 | 968.13 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-06-01 14:00:00 | 975.93 | 978.87 | 968.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-02 09:15:00 | 968.00 | 975.61 | 969.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-02 10:00:00 | 968.00 | 975.61 | 969.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-02 10:15:00 | 970.50 | 974.59 | 969.35 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-02 13:45:00 | 973.33 | 972.45 | 969.52 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-02 14:30:00 | 972.50 | 974.56 | 970.75 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-06-09 10:15:00 | 968.15 | 985.36 | 987.07 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 4 — SELL (started 2023-06-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-09 10:15:00 | 968.15 | 985.36 | 987.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-12 09:15:00 | 964.45 | 972.43 | 978.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-12 10:15:00 | 972.60 | 972.46 | 978.27 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-06-12 10:45:00 | 972.38 | 972.46 | 978.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-13 09:15:00 | 985.68 | 974.38 | 976.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-13 09:30:00 | 985.03 | 974.38 | 976.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 5 — BUY (started 2023-06-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-13 11:15:00 | 985.45 | 978.22 | 977.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-13 14:15:00 | 988.48 | 982.77 | 980.32 | Break + close above crossover candle high |

### Cycle 6 — SELL (started 2023-06-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-14 09:15:00 | 956.10 | 978.23 | 978.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-19 13:15:00 | 946.93 | 956.12 | 961.17 | Break + close below crossover candle low |

### Cycle 7 — BUY (started 2023-06-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-20 09:15:00 | 1010.00 | 964.04 | 963.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-20 14:15:00 | 1040.83 | 1005.26 | 986.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-21 13:15:00 | 1023.00 | 1026.24 | 1007.65 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-06-21 14:00:00 | 1023.00 | 1026.24 | 1007.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-22 09:15:00 | 1011.00 | 1023.09 | 1010.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-22 10:15:00 | 1012.35 | 1023.09 | 1010.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-22 10:15:00 | 1007.30 | 1019.93 | 1010.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-22 11:00:00 | 1007.30 | 1019.93 | 1010.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-22 11:15:00 | 1003.10 | 1016.56 | 1009.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-22 11:45:00 | 1004.00 | 1016.56 | 1009.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-22 14:15:00 | 1010.00 | 1012.20 | 1009.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-22 14:30:00 | 1011.40 | 1012.20 | 1009.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-22 15:15:00 | 1007.45 | 1011.25 | 1009.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-23 09:15:00 | 1000.73 | 1011.25 | 1009.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-23 09:15:00 | 1003.33 | 1009.67 | 1008.52 | EMA400 retest candle locked (from upside) |

### Cycle 8 — SELL (started 2023-06-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-23 11:15:00 | 1000.48 | 1006.31 | 1007.10 | EMA200 below EMA400 |

### Cycle 9 — BUY (started 2023-06-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-27 10:15:00 | 1011.83 | 1005.50 | 1005.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-27 13:15:00 | 1020.05 | 1009.32 | 1007.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-28 15:15:00 | 1022.50 | 1024.79 | 1018.77 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-30 09:15:00 | 1127.50 | 1024.79 | 1018.77 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-06 09:15:00 | 1137.03 | 1146.03 | 1138.62 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2023-07-06 09:15:00 | 1137.03 | 1146.03 | 1138.62 | SL hit (close<ema400) qty=1.00 sl=1138.62 alert=retest1 |

### Cycle 10 — SELL (started 2023-07-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-07 09:15:00 | 1130.22 | 1135.23 | 1135.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-07 10:15:00 | 1123.00 | 1132.78 | 1134.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-10 14:15:00 | 1124.85 | 1121.68 | 1125.70 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-07-10 15:00:00 | 1124.85 | 1121.68 | 1125.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-10 15:15:00 | 1127.97 | 1122.94 | 1125.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-11 09:15:00 | 1131.33 | 1122.94 | 1125.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-11 09:15:00 | 1138.30 | 1126.01 | 1127.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-11 10:00:00 | 1138.30 | 1126.01 | 1127.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 11 — BUY (started 2023-07-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-11 10:15:00 | 1150.68 | 1130.95 | 1129.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-12 09:15:00 | 1168.75 | 1144.94 | 1137.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-13 13:15:00 | 1164.08 | 1169.25 | 1158.77 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-07-13 14:00:00 | 1164.08 | 1169.25 | 1158.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-18 09:15:00 | 1199.00 | 1189.80 | 1181.05 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-18 10:30:00 | 1203.15 | 1191.54 | 1182.63 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-19 10:00:00 | 1202.95 | 1190.52 | 1185.41 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-19 11:00:00 | 1201.00 | 1192.61 | 1186.83 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-07-27 09:15:00 | 1233.53 | 1247.16 | 1248.44 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 12 — SELL (started 2023-07-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-27 09:15:00 | 1233.53 | 1247.16 | 1248.44 | EMA200 below EMA400 |

### Cycle 13 — BUY (started 2023-07-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-28 11:15:00 | 1268.80 | 1245.22 | 1243.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-31 09:15:00 | 1280.43 | 1263.23 | 1253.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-31 12:15:00 | 1270.75 | 1271.29 | 1260.53 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-07-31 13:00:00 | 1270.75 | 1271.29 | 1260.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-31 13:15:00 | 1261.68 | 1269.37 | 1260.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-31 14:00:00 | 1261.68 | 1269.37 | 1260.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-31 14:15:00 | 1265.00 | 1268.49 | 1261.03 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-01 09:45:00 | 1273.45 | 1269.09 | 1262.59 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-01 10:30:00 | 1268.75 | 1269.27 | 1263.26 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-01 12:45:00 | 1269.22 | 1269.73 | 1264.52 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-08-02 09:15:00 | 1251.50 | 1268.49 | 1265.85 | SL hit (close<static) qty=1.00 sl=1260.20 alert=retest2 |

### Cycle 14 — SELL (started 2023-08-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-02 10:15:00 | 1243.85 | 1263.56 | 1263.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-03 09:15:00 | 1230.50 | 1247.20 | 1254.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-04 14:15:00 | 1203.40 | 1200.76 | 1216.27 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-08-04 15:00:00 | 1203.40 | 1200.76 | 1216.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-07 09:15:00 | 1211.78 | 1203.28 | 1214.75 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-07 10:45:00 | 1204.50 | 1205.53 | 1214.72 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-08-07 12:15:00 | 1224.80 | 1211.25 | 1215.84 | SL hit (close>static) qty=1.00 sl=1223.47 alert=retest2 |

### Cycle 15 — BUY (started 2023-08-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-08 09:15:00 | 1250.97 | 1224.46 | 1221.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-09 09:15:00 | 1255.43 | 1241.25 | 1232.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-09 10:15:00 | 1238.50 | 1240.70 | 1232.93 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-08-09 11:00:00 | 1238.50 | 1240.70 | 1232.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-14 09:15:00 | 1247.25 | 1277.72 | 1273.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-14 09:30:00 | 1252.00 | 1277.72 | 1273.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-14 10:15:00 | 1248.47 | 1271.87 | 1270.87 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-14 11:15:00 | 1249.88 | 1271.87 | 1270.87 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-08-14 11:15:00 | 1250.50 | 1267.60 | 1269.02 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 16 — SELL (started 2023-08-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-14 11:15:00 | 1250.50 | 1267.60 | 1269.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-14 12:15:00 | 1242.68 | 1262.61 | 1266.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-16 09:15:00 | 1275.00 | 1258.33 | 1262.58 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-16 09:15:00 | 1275.00 | 1258.33 | 1262.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-16 09:15:00 | 1275.00 | 1258.33 | 1262.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-16 10:00:00 | 1275.00 | 1258.33 | 1262.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-16 10:15:00 | 1265.08 | 1259.68 | 1262.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-16 10:30:00 | 1274.30 | 1259.68 | 1262.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-16 11:15:00 | 1276.83 | 1263.11 | 1264.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-16 12:00:00 | 1276.83 | 1263.11 | 1264.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 17 — BUY (started 2023-08-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-16 12:15:00 | 1277.50 | 1265.99 | 1265.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-17 09:15:00 | 1279.55 | 1271.98 | 1268.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-17 11:15:00 | 1268.78 | 1272.33 | 1269.40 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-17 11:15:00 | 1268.78 | 1272.33 | 1269.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-17 11:15:00 | 1268.78 | 1272.33 | 1269.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-17 11:30:00 | 1267.47 | 1272.33 | 1269.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-17 12:15:00 | 1271.72 | 1272.21 | 1269.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-17 12:45:00 | 1270.95 | 1272.21 | 1269.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-17 13:15:00 | 1272.47 | 1272.26 | 1269.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-17 14:15:00 | 1268.97 | 1272.26 | 1269.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-17 14:15:00 | 1273.05 | 1272.42 | 1270.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-17 15:00:00 | 1273.05 | 1272.42 | 1270.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 18 — SELL (started 2023-08-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-18 09:15:00 | 1250.75 | 1268.02 | 1268.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-18 10:15:00 | 1240.22 | 1262.46 | 1265.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-21 09:15:00 | 1252.47 | 1252.40 | 1258.15 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-08-21 10:00:00 | 1252.47 | 1252.40 | 1258.15 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-21 10:15:00 | 1247.38 | 1251.40 | 1257.17 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-21 13:30:00 | 1238.18 | 1247.03 | 1253.52 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-21 14:45:00 | 1237.50 | 1245.03 | 1252.02 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-08-22 09:15:00 | 1267.30 | 1247.88 | 1252.00 | SL hit (close>static) qty=1.00 sl=1257.50 alert=retest2 |

### Cycle 19 — BUY (started 2023-08-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-23 09:15:00 | 1268.70 | 1255.00 | 1254.23 | EMA200 above EMA400 |

### Cycle 20 — SELL (started 2023-08-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-24 14:15:00 | 1250.05 | 1257.65 | 1258.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-24 15:15:00 | 1242.20 | 1254.56 | 1256.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-25 09:15:00 | 1257.18 | 1255.08 | 1256.98 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-25 09:15:00 | 1257.18 | 1255.08 | 1256.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-25 09:15:00 | 1257.18 | 1255.08 | 1256.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-25 10:00:00 | 1257.18 | 1255.08 | 1256.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-25 10:15:00 | 1253.45 | 1254.76 | 1256.66 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-25 15:15:00 | 1241.50 | 1252.62 | 1254.91 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-08-28 10:15:00 | 1259.38 | 1252.90 | 1254.31 | SL hit (close>static) qty=1.00 sl=1257.50 alert=retest2 |

### Cycle 21 — BUY (started 2023-08-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-28 11:15:00 | 1266.75 | 1255.67 | 1255.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-28 12:15:00 | 1270.00 | 1258.54 | 1256.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-29 11:15:00 | 1254.47 | 1260.78 | 1258.99 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-29 11:15:00 | 1254.47 | 1260.78 | 1258.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-29 11:15:00 | 1254.47 | 1260.78 | 1258.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-29 11:45:00 | 1253.25 | 1260.78 | 1258.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-29 12:15:00 | 1250.65 | 1258.75 | 1258.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-29 12:30:00 | 1247.08 | 1258.75 | 1258.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 22 — SELL (started 2023-08-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-29 13:15:00 | 1244.58 | 1255.92 | 1256.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-29 14:15:00 | 1237.00 | 1252.13 | 1255.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-30 09:15:00 | 1260.75 | 1251.92 | 1254.43 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-30 09:15:00 | 1260.75 | 1251.92 | 1254.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-30 09:15:00 | 1260.75 | 1251.92 | 1254.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-30 09:45:00 | 1267.40 | 1251.92 | 1254.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-30 10:15:00 | 1244.97 | 1250.53 | 1253.57 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-31 10:00:00 | 1239.68 | 1247.64 | 1250.91 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-08-31 14:15:00 | 1272.80 | 1244.06 | 1246.59 | SL hit (close>static) qty=1.00 sl=1263.25 alert=retest2 |

### Cycle 23 — BUY (started 2023-09-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-05 15:15:00 | 1234.00 | 1232.49 | 1232.48 | EMA200 above EMA400 |

### Cycle 24 — SELL (started 2023-09-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-06 09:15:00 | 1232.10 | 1232.41 | 1232.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-06 10:15:00 | 1228.65 | 1231.66 | 1232.10 | Break + close below crossover candle low |

### Cycle 25 — BUY (started 2023-09-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-07 09:15:00 | 1254.78 | 1234.14 | 1232.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-07 14:15:00 | 1261.22 | 1248.68 | 1241.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-12 09:15:00 | 1307.72 | 1309.47 | 1287.94 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-09-12 09:45:00 | 1314.50 | 1309.47 | 1287.94 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-13 09:15:00 | 1282.45 | 1304.73 | 1296.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-13 10:00:00 | 1282.45 | 1304.73 | 1296.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-13 10:15:00 | 1292.22 | 1302.23 | 1295.95 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-13 11:15:00 | 1296.25 | 1302.23 | 1295.95 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-09-22 14:15:00 | 1329.05 | 1339.48 | 1340.10 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 26 — SELL (started 2023-09-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-22 14:15:00 | 1329.05 | 1339.48 | 1340.10 | EMA200 below EMA400 |

### Cycle 27 — BUY (started 2023-09-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-26 09:15:00 | 1359.25 | 1340.07 | 1339.04 | EMA200 above EMA400 |

### Cycle 28 — SELL (started 2023-09-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-26 15:15:00 | 1334.70 | 1339.07 | 1339.37 | EMA200 below EMA400 |

### Cycle 29 — BUY (started 2023-09-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-27 09:15:00 | 1342.70 | 1339.79 | 1339.67 | EMA200 above EMA400 |

### Cycle 30 — SELL (started 2023-09-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-27 15:15:00 | 1333.80 | 1340.17 | 1340.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-28 09:15:00 | 1312.50 | 1334.64 | 1337.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-29 09:15:00 | 1310.90 | 1307.83 | 1319.86 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-09-29 10:00:00 | 1310.90 | 1307.83 | 1319.86 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-29 13:15:00 | 1319.20 | 1311.36 | 1317.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-29 14:00:00 | 1319.20 | 1311.36 | 1317.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-29 14:15:00 | 1323.73 | 1313.84 | 1318.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-29 15:00:00 | 1323.73 | 1313.84 | 1318.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-29 15:15:00 | 1322.50 | 1315.57 | 1318.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-03 09:15:00 | 1317.70 | 1315.57 | 1318.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-03 10:15:00 | 1329.50 | 1318.76 | 1319.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-03 11:00:00 | 1329.50 | 1318.76 | 1319.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 31 — BUY (started 2023-10-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-03 11:15:00 | 1332.73 | 1321.55 | 1320.79 | EMA200 above EMA400 |

### Cycle 32 — SELL (started 2023-10-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-04 11:15:00 | 1312.03 | 1320.88 | 1321.52 | EMA200 below EMA400 |

### Cycle 33 — BUY (started 2023-10-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-04 14:15:00 | 1329.83 | 1322.50 | 1322.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-05 09:15:00 | 1333.98 | 1325.31 | 1323.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-05 12:15:00 | 1330.63 | 1331.48 | 1327.21 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-10-05 13:00:00 | 1330.63 | 1331.48 | 1327.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-05 13:15:00 | 1328.15 | 1330.81 | 1327.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-05 14:00:00 | 1328.15 | 1330.81 | 1327.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-05 14:15:00 | 1323.00 | 1329.25 | 1326.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-05 15:00:00 | 1323.00 | 1329.25 | 1326.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-05 15:15:00 | 1324.98 | 1328.40 | 1326.73 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-06 09:15:00 | 1329.63 | 1328.40 | 1326.73 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2023-10-17 10:15:00 | 1462.59 | 1437.82 | 1421.90 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 34 — SELL (started 2023-10-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-23 11:15:00 | 1423.65 | 1433.02 | 1433.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-23 12:15:00 | 1419.00 | 1430.21 | 1432.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-25 09:15:00 | 1426.85 | 1423.52 | 1428.13 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-25 09:15:00 | 1426.85 | 1423.52 | 1428.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-25 09:15:00 | 1426.85 | 1423.52 | 1428.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-25 09:45:00 | 1428.45 | 1423.52 | 1428.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-25 10:15:00 | 1417.43 | 1422.30 | 1427.16 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-25 11:30:00 | 1413.58 | 1420.93 | 1426.09 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-25 12:00:00 | 1415.45 | 1420.93 | 1426.09 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-11-02 09:15:00 | 1386.50 | 1371.06 | 1371.05 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 35 — BUY (started 2023-11-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-02 09:15:00 | 1386.50 | 1371.06 | 1371.05 | EMA200 above EMA400 |

### Cycle 36 — SELL (started 2023-11-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-02 14:15:00 | 1364.08 | 1370.17 | 1370.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-03 11:15:00 | 1361.70 | 1366.07 | 1368.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-03 12:15:00 | 1378.60 | 1368.57 | 1369.36 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-03 12:15:00 | 1378.60 | 1368.57 | 1369.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-03 12:15:00 | 1378.60 | 1368.57 | 1369.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-03 13:00:00 | 1378.60 | 1368.57 | 1369.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 37 — BUY (started 2023-11-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-03 13:15:00 | 1378.48 | 1370.55 | 1370.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-06 09:15:00 | 1394.10 | 1377.34 | 1373.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-07 12:15:00 | 1376.53 | 1384.51 | 1381.03 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-07 12:15:00 | 1376.53 | 1384.51 | 1381.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-07 12:15:00 | 1376.53 | 1384.51 | 1381.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-07 13:00:00 | 1376.53 | 1384.51 | 1381.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-07 13:15:00 | 1376.50 | 1382.91 | 1380.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-07 14:00:00 | 1376.50 | 1382.91 | 1380.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-08 09:15:00 | 1383.88 | 1383.99 | 1381.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-08 09:45:00 | 1385.98 | 1383.99 | 1381.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-08 10:15:00 | 1381.75 | 1383.54 | 1381.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-08 11:00:00 | 1381.75 | 1383.54 | 1381.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-08 11:15:00 | 1375.00 | 1381.83 | 1381.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-08 11:45:00 | 1375.05 | 1381.83 | 1381.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-08 12:15:00 | 1377.68 | 1381.00 | 1380.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-08 12:45:00 | 1373.50 | 1381.00 | 1380.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 38 — SELL (started 2023-11-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-08 13:15:00 | 1372.28 | 1379.26 | 1380.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-09 09:15:00 | 1368.08 | 1375.57 | 1378.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-09 11:15:00 | 1379.50 | 1376.12 | 1377.86 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-09 11:15:00 | 1379.50 | 1376.12 | 1377.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-09 11:15:00 | 1379.50 | 1376.12 | 1377.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-09 11:45:00 | 1379.00 | 1376.12 | 1377.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-09 12:15:00 | 1374.40 | 1375.78 | 1377.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-09 12:30:00 | 1375.00 | 1375.78 | 1377.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-09 13:15:00 | 1382.50 | 1377.12 | 1378.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-09 14:00:00 | 1382.50 | 1377.12 | 1378.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-09 14:15:00 | 1380.48 | 1377.79 | 1378.22 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-10 09:15:00 | 1377.00 | 1378.05 | 1378.30 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-10 11:00:00 | 1378.25 | 1377.51 | 1377.98 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-11-10 11:15:00 | 1384.75 | 1378.96 | 1378.59 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 39 — BUY (started 2023-11-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-10 11:15:00 | 1384.75 | 1378.96 | 1378.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-10 12:15:00 | 1389.10 | 1380.99 | 1379.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-10 14:15:00 | 1382.20 | 1382.89 | 1380.76 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-11-10 15:00:00 | 1382.20 | 1382.89 | 1380.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-10 15:15:00 | 1390.00 | 1384.31 | 1381.60 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-12 18:15:00 | 1390.60 | 1384.31 | 1381.60 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-11-13 09:15:00 | 1370.53 | 1381.83 | 1380.96 | SL hit (close<static) qty=1.00 sl=1379.58 alert=retest2 |

### Cycle 40 — SELL (started 2023-11-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-22 12:15:00 | 1420.48 | 1436.59 | 1437.29 | EMA200 below EMA400 |

### Cycle 41 — BUY (started 2023-11-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-24 10:15:00 | 1438.98 | 1433.41 | 1433.34 | EMA200 above EMA400 |

### Cycle 42 — SELL (started 2023-11-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-24 11:15:00 | 1429.18 | 1432.56 | 1432.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-24 12:15:00 | 1422.68 | 1430.59 | 1432.03 | Break + close below crossover candle low |

### Cycle 43 — BUY (started 2023-11-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-28 09:15:00 | 1457.43 | 1432.83 | 1432.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-30 14:15:00 | 1485.93 | 1467.45 | 1458.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-01 13:15:00 | 1470.00 | 1471.79 | 1465.04 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-12-01 14:00:00 | 1470.00 | 1471.79 | 1465.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-01 14:15:00 | 1464.00 | 1470.23 | 1464.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-01 15:00:00 | 1464.00 | 1470.23 | 1464.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-01 15:15:00 | 1467.00 | 1469.59 | 1465.13 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-04 09:15:00 | 1492.50 | 1469.59 | 1465.13 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-12-08 13:15:00 | 1493.75 | 1495.98 | 1496.20 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 44 — SELL (started 2023-12-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-08 13:15:00 | 1493.75 | 1495.98 | 1496.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-11 13:15:00 | 1486.08 | 1492.60 | 1494.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-12 09:15:00 | 1503.18 | 1493.52 | 1494.23 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-12 09:15:00 | 1503.18 | 1493.52 | 1494.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-12 09:15:00 | 1503.18 | 1493.52 | 1494.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-12 09:45:00 | 1509.38 | 1493.52 | 1494.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 45 — BUY (started 2023-12-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-12 10:15:00 | 1508.08 | 1496.43 | 1495.49 | EMA200 above EMA400 |

### Cycle 46 — SELL (started 2023-12-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-13 11:15:00 | 1483.83 | 1495.55 | 1496.80 | EMA200 below EMA400 |

### Cycle 47 — BUY (started 2023-12-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-13 14:15:00 | 1506.10 | 1498.57 | 1497.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-13 15:15:00 | 1512.50 | 1501.36 | 1499.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-15 09:15:00 | 1516.35 | 1519.05 | 1512.12 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-15 09:15:00 | 1516.35 | 1519.05 | 1512.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-15 09:15:00 | 1516.35 | 1519.05 | 1512.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-15 09:30:00 | 1513.00 | 1519.05 | 1512.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-15 10:15:00 | 1512.20 | 1517.68 | 1512.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-15 10:45:00 | 1511.83 | 1517.68 | 1512.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-15 11:15:00 | 1510.95 | 1516.33 | 1512.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-15 11:45:00 | 1511.13 | 1516.33 | 1512.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-15 12:15:00 | 1511.10 | 1515.29 | 1511.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-15 13:00:00 | 1511.10 | 1515.29 | 1511.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-15 13:15:00 | 1510.95 | 1514.42 | 1511.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-15 13:30:00 | 1511.00 | 1514.42 | 1511.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-15 14:15:00 | 1510.00 | 1513.54 | 1511.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-15 15:00:00 | 1510.00 | 1513.54 | 1511.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-15 15:15:00 | 1515.50 | 1513.93 | 1512.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-18 09:15:00 | 1513.00 | 1513.93 | 1512.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-18 09:15:00 | 1523.53 | 1515.85 | 1513.07 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-18 10:15:00 | 1528.45 | 1515.85 | 1513.07 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-18 11:30:00 | 1525.38 | 1518.00 | 1514.63 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-19 10:30:00 | 1525.60 | 1519.64 | 1516.69 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-12-29 13:15:00 | 1595.00 | 1602.50 | 1603.49 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 48 — SELL (started 2023-12-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-29 13:15:00 | 1595.00 | 1602.50 | 1603.49 | EMA200 below EMA400 |

### Cycle 49 — BUY (started 2024-01-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-01 13:15:00 | 1611.20 | 1604.23 | 1603.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-02 09:15:00 | 1621.25 | 1609.92 | 1606.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-02 10:15:00 | 1608.50 | 1609.64 | 1606.81 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-02 10:15:00 | 1608.50 | 1609.64 | 1606.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-02 10:15:00 | 1608.50 | 1609.64 | 1606.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-02 11:00:00 | 1608.50 | 1609.64 | 1606.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-02 11:15:00 | 1608.23 | 1609.36 | 1606.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-02 11:45:00 | 1609.18 | 1609.36 | 1606.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-03 10:15:00 | 1618.35 | 1619.70 | 1614.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-03 10:45:00 | 1617.83 | 1619.70 | 1614.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-03 14:15:00 | 1617.85 | 1623.26 | 1617.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-03 15:00:00 | 1617.85 | 1623.26 | 1617.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-03 15:15:00 | 1622.48 | 1623.11 | 1618.36 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-04 09:15:00 | 1641.08 | 1623.11 | 1618.36 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-01-16 13:15:00 | 1713.50 | 1727.60 | 1728.42 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 50 — SELL (started 2024-01-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-16 13:15:00 | 1713.50 | 1727.60 | 1728.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-17 09:15:00 | 1696.00 | 1718.10 | 1723.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-18 10:15:00 | 1683.03 | 1678.05 | 1695.61 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-01-18 11:00:00 | 1683.03 | 1678.05 | 1695.61 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-18 11:15:00 | 1699.50 | 1682.34 | 1695.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-18 11:30:00 | 1688.05 | 1682.34 | 1695.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-18 12:15:00 | 1691.00 | 1684.07 | 1695.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-18 12:30:00 | 1699.85 | 1684.07 | 1695.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-18 14:15:00 | 1700.58 | 1688.60 | 1695.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-18 15:00:00 | 1700.58 | 1688.60 | 1695.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-18 15:15:00 | 1708.10 | 1692.50 | 1696.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-19 09:15:00 | 1706.50 | 1692.50 | 1696.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 51 — BUY (started 2024-01-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-19 10:15:00 | 1710.18 | 1699.60 | 1699.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-19 13:15:00 | 1726.98 | 1710.57 | 1704.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-20 09:15:00 | 1717.00 | 1720.34 | 1711.54 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-20 09:15:00 | 1717.00 | 1720.34 | 1711.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-20 09:15:00 | 1717.00 | 1720.34 | 1711.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-20 09:30:00 | 1717.75 | 1720.34 | 1711.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-20 13:15:00 | 1711.50 | 1717.36 | 1712.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-20 13:45:00 | 1706.35 | 1717.36 | 1712.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-20 14:15:00 | 1699.33 | 1713.75 | 1711.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-20 15:00:00 | 1699.33 | 1713.75 | 1711.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-20 15:15:00 | 1701.03 | 1711.21 | 1710.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-23 09:15:00 | 1720.03 | 1711.21 | 1710.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 52 — SELL (started 2024-01-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-23 09:15:00 | 1705.73 | 1710.11 | 1710.24 | EMA200 below EMA400 |

### Cycle 53 — BUY (started 2024-01-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-23 10:15:00 | 1717.28 | 1711.55 | 1710.88 | EMA200 above EMA400 |

### Cycle 54 — SELL (started 2024-01-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-23 13:15:00 | 1701.25 | 1710.40 | 1710.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-23 14:15:00 | 1687.80 | 1705.88 | 1708.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-24 09:15:00 | 1701.00 | 1698.60 | 1704.35 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-24 09:15:00 | 1701.00 | 1698.60 | 1704.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-24 09:15:00 | 1701.00 | 1698.60 | 1704.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-24 10:00:00 | 1701.00 | 1698.60 | 1704.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-24 10:15:00 | 1706.43 | 1700.17 | 1704.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-24 11:00:00 | 1706.43 | 1700.17 | 1704.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-24 11:15:00 | 1696.50 | 1699.44 | 1703.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-24 11:45:00 | 1708.18 | 1699.44 | 1703.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-24 12:15:00 | 1705.63 | 1700.67 | 1703.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-24 13:00:00 | 1705.63 | 1700.67 | 1703.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-24 13:15:00 | 1712.95 | 1703.13 | 1704.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-24 14:00:00 | 1712.95 | 1703.13 | 1704.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 55 — BUY (started 2024-01-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-24 14:15:00 | 1716.98 | 1705.90 | 1705.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-24 15:15:00 | 1720.50 | 1708.82 | 1707.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-25 13:15:00 | 1712.40 | 1716.49 | 1712.58 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-25 13:15:00 | 1712.40 | 1716.49 | 1712.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-25 13:15:00 | 1712.40 | 1716.49 | 1712.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-25 14:00:00 | 1712.40 | 1716.49 | 1712.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-25 14:15:00 | 1715.00 | 1716.19 | 1712.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-25 15:00:00 | 1715.00 | 1716.19 | 1712.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-25 15:15:00 | 1703.15 | 1713.58 | 1711.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-29 09:15:00 | 1737.75 | 1713.58 | 1711.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-29 09:15:00 | 1757.90 | 1722.45 | 1716.10 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-29 10:15:00 | 1764.50 | 1722.45 | 1716.10 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-29 11:45:00 | 1764.95 | 1737.36 | 1724.34 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-29 13:45:00 | 1762.93 | 1745.30 | 1730.40 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-29 14:15:00 | 1762.63 | 1745.30 | 1730.40 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-01 11:15:00 | 1779.43 | 1784.93 | 1772.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-01 12:00:00 | 1779.43 | 1784.93 | 1772.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-01 12:15:00 | 1779.93 | 1783.93 | 1773.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-01 12:30:00 | 1773.88 | 1783.93 | 1773.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-02 12:15:00 | 1782.80 | 1789.18 | 1781.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-02 13:00:00 | 1782.80 | 1789.18 | 1781.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-02 13:15:00 | 1781.00 | 1787.55 | 1781.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-02 14:00:00 | 1781.00 | 1787.55 | 1781.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-02 14:15:00 | 1784.03 | 1786.84 | 1781.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-02 15:15:00 | 1781.40 | 1786.84 | 1781.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-02 15:15:00 | 1781.40 | 1785.75 | 1781.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-05 09:15:00 | 1782.90 | 1785.75 | 1781.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-05 09:15:00 | 1795.08 | 1787.62 | 1782.99 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-02-05 13:15:00 | 1768.40 | 1780.59 | 1780.98 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 56 — SELL (started 2024-02-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-05 13:15:00 | 1768.40 | 1780.59 | 1780.98 | EMA200 below EMA400 |

### Cycle 57 — BUY (started 2024-02-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-06 09:15:00 | 1788.65 | 1782.07 | 1781.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-06 14:15:00 | 1811.88 | 1792.12 | 1786.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-08 10:15:00 | 1812.83 | 1815.45 | 1806.11 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-02-08 11:00:00 | 1812.83 | 1815.45 | 1806.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-09 09:15:00 | 1813.10 | 1819.87 | 1813.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-09 10:00:00 | 1813.10 | 1819.87 | 1813.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-09 10:15:00 | 1804.85 | 1816.86 | 1812.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-09 11:00:00 | 1804.85 | 1816.86 | 1812.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-09 11:15:00 | 1789.85 | 1811.46 | 1810.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-09 12:00:00 | 1789.85 | 1811.46 | 1810.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 58 — SELL (started 2024-02-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-09 12:15:00 | 1802.40 | 1809.65 | 1809.78 | EMA200 below EMA400 |

### Cycle 59 — BUY (started 2024-02-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-09 14:15:00 | 1817.73 | 1810.61 | 1810.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-12 09:15:00 | 1830.98 | 1815.46 | 1812.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-12 11:15:00 | 1800.53 | 1815.10 | 1813.00 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-12 11:15:00 | 1800.53 | 1815.10 | 1813.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-12 11:15:00 | 1800.53 | 1815.10 | 1813.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-12 11:30:00 | 1810.25 | 1815.10 | 1813.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-12 12:15:00 | 1797.80 | 1811.64 | 1811.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-12 13:00:00 | 1797.80 | 1811.64 | 1811.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-12 13:15:00 | 1816.85 | 1812.68 | 1812.09 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-12 14:15:00 | 1818.15 | 1812.68 | 1812.09 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-13 09:15:00 | 1827.00 | 1812.45 | 1812.10 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-02-19 13:15:00 | 1862.53 | 1892.67 | 1892.76 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 60 — SELL (started 2024-02-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-19 13:15:00 | 1862.53 | 1892.67 | 1892.76 | EMA200 below EMA400 |

### Cycle 61 — BUY (started 2024-02-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-20 11:15:00 | 1907.45 | 1892.21 | 1891.43 | EMA200 above EMA400 |

### Cycle 62 — SELL (started 2024-02-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-20 12:15:00 | 1881.45 | 1890.06 | 1890.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-21 14:15:00 | 1873.43 | 1884.70 | 1887.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-22 10:15:00 | 1886.00 | 1879.32 | 1883.88 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-22 10:15:00 | 1886.00 | 1879.32 | 1883.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-22 10:15:00 | 1886.00 | 1879.32 | 1883.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-22 11:00:00 | 1886.00 | 1879.32 | 1883.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-22 11:15:00 | 1892.85 | 1882.03 | 1884.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-22 12:00:00 | 1892.85 | 1882.03 | 1884.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-22 12:15:00 | 1891.85 | 1883.99 | 1885.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-22 12:30:00 | 1892.85 | 1883.99 | 1885.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-22 14:15:00 | 1889.70 | 1884.90 | 1885.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-22 14:30:00 | 1891.10 | 1884.90 | 1885.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 63 — BUY (started 2024-02-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-22 15:15:00 | 1891.60 | 1886.24 | 1886.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-23 09:15:00 | 1908.75 | 1890.74 | 1888.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-23 10:15:00 | 1890.50 | 1890.69 | 1888.34 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-23 10:15:00 | 1890.50 | 1890.69 | 1888.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-23 10:15:00 | 1890.50 | 1890.69 | 1888.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-23 10:45:00 | 1889.50 | 1890.69 | 1888.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-26 09:15:00 | 1886.53 | 1899.15 | 1894.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-26 10:00:00 | 1886.53 | 1899.15 | 1894.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-26 10:15:00 | 1896.73 | 1898.67 | 1895.11 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-26 12:30:00 | 1917.40 | 1903.20 | 1897.75 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-02-28 15:15:00 | 1914.00 | 1932.26 | 1933.12 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 64 — SELL (started 2024-02-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-28 15:15:00 | 1914.00 | 1932.26 | 1933.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-29 09:15:00 | 1872.50 | 1920.31 | 1927.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-01 13:15:00 | 1887.80 | 1884.27 | 1896.86 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-01 15:15:00 | 1889.38 | 1886.36 | 1895.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-01 15:15:00 | 1889.38 | 1886.36 | 1895.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-02 09:15:00 | 1899.75 | 1886.36 | 1895.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-02 09:15:00 | 1902.00 | 1889.49 | 1896.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-02 09:30:00 | 1897.00 | 1889.49 | 1896.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-02 11:15:00 | 1888.50 | 1889.29 | 1895.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-02 11:30:00 | 1892.53 | 1889.29 | 1895.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-04 09:15:00 | 1903.75 | 1890.80 | 1895.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-04 10:00:00 | 1903.75 | 1890.80 | 1895.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-04 10:15:00 | 1917.93 | 1896.22 | 1897.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-04 11:00:00 | 1917.93 | 1896.22 | 1897.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 65 — BUY (started 2024-03-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-04 12:15:00 | 1913.35 | 1899.72 | 1898.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-04 13:15:00 | 1924.00 | 1904.58 | 1900.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-05 09:15:00 | 1903.60 | 1909.92 | 1904.78 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-05 09:15:00 | 1903.60 | 1909.92 | 1904.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-05 09:15:00 | 1903.60 | 1909.92 | 1904.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-05 10:00:00 | 1903.60 | 1909.92 | 1904.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-05 10:15:00 | 1893.85 | 1906.71 | 1903.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-05 10:30:00 | 1893.18 | 1906.71 | 1903.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-05 11:15:00 | 1886.50 | 1902.67 | 1902.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-05 12:00:00 | 1886.50 | 1902.67 | 1902.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-05 13:15:00 | 1900.15 | 1902.94 | 1902.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-05 14:00:00 | 1900.15 | 1902.94 | 1902.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-05 14:15:00 | 1899.38 | 1902.23 | 1902.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-05 14:30:00 | 1900.25 | 1902.23 | 1902.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 66 — SELL (started 2024-03-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-05 15:15:00 | 1897.00 | 1901.18 | 1901.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-06 09:15:00 | 1888.60 | 1898.66 | 1900.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-11 09:15:00 | 1903.25 | 1881.38 | 1885.34 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-11 09:15:00 | 1903.25 | 1881.38 | 1885.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-11 09:15:00 | 1903.25 | 1881.38 | 1885.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-11 10:00:00 | 1903.25 | 1881.38 | 1885.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-11 10:15:00 | 1890.70 | 1883.24 | 1885.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-11 10:30:00 | 1912.80 | 1883.24 | 1885.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-11 11:15:00 | 1897.08 | 1886.01 | 1886.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-11 12:00:00 | 1897.08 | 1886.01 | 1886.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-11 12:15:00 | 1889.25 | 1886.66 | 1887.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-11 13:15:00 | 1900.73 | 1886.66 | 1887.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 67 — BUY (started 2024-03-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-11 13:15:00 | 1892.95 | 1887.92 | 1887.61 | EMA200 above EMA400 |

### Cycle 68 — SELL (started 2024-03-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-11 15:15:00 | 1877.50 | 1885.86 | 1886.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-12 10:15:00 | 1857.63 | 1879.99 | 1883.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-14 10:15:00 | 1835.10 | 1829.40 | 1844.03 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-03-14 11:00:00 | 1835.10 | 1829.40 | 1844.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-14 11:15:00 | 1852.85 | 1834.09 | 1844.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-14 12:00:00 | 1852.85 | 1834.09 | 1844.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-14 12:15:00 | 1855.75 | 1838.42 | 1845.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-14 13:00:00 | 1855.75 | 1838.42 | 1845.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 69 — BUY (started 2024-03-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-14 15:15:00 | 1870.68 | 1853.17 | 1851.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-15 13:15:00 | 1902.13 | 1867.67 | 1859.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-18 09:15:00 | 1880.78 | 1883.08 | 1869.52 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-18 09:15:00 | 1880.78 | 1883.08 | 1869.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-18 09:15:00 | 1880.78 | 1883.08 | 1869.52 | EMA400 retest candle locked (from upside) |

### Cycle 70 — SELL (started 2024-03-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-19 10:15:00 | 1828.95 | 1860.77 | 1864.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-19 15:15:00 | 1823.00 | 1838.68 | 1850.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-21 09:15:00 | 1825.48 | 1814.93 | 1828.53 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-21 09:15:00 | 1825.48 | 1814.93 | 1828.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-21 09:15:00 | 1825.48 | 1814.93 | 1828.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-21 10:00:00 | 1825.48 | 1814.93 | 1828.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-21 10:15:00 | 1831.50 | 1818.25 | 1828.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-21 10:30:00 | 1830.70 | 1818.25 | 1828.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-21 11:15:00 | 1812.55 | 1817.11 | 1827.33 | EMA400 retest candle locked (from downside) |

### Cycle 71 — BUY (started 2024-03-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-22 09:15:00 | 1894.40 | 1838.16 | 1833.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-26 09:15:00 | 1916.53 | 1880.74 | 1861.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-27 09:15:00 | 1897.40 | 1898.13 | 1881.46 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-03-27 09:30:00 | 1893.78 | 1898.13 | 1881.46 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-28 09:15:00 | 1884.50 | 1895.75 | 1888.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-28 09:45:00 | 1886.68 | 1895.75 | 1888.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-28 10:15:00 | 1878.98 | 1892.40 | 1887.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-28 10:45:00 | 1877.23 | 1892.40 | 1887.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-28 14:15:00 | 1874.70 | 1887.28 | 1886.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-28 15:00:00 | 1874.70 | 1887.28 | 1886.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 72 — SELL (started 2024-03-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-28 15:15:00 | 1878.45 | 1885.52 | 1885.75 | EMA200 below EMA400 |

### Cycle 73 — BUY (started 2024-04-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-01 09:15:00 | 1896.75 | 1887.76 | 1886.75 | EMA200 above EMA400 |

### Cycle 74 — SELL (started 2024-04-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-02 09:15:00 | 1834.75 | 1880.15 | 1884.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-08 09:15:00 | 1823.10 | 1837.58 | 1849.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-08 12:15:00 | 1836.25 | 1834.19 | 1844.49 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-04-08 13:00:00 | 1836.25 | 1834.19 | 1844.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-08 13:15:00 | 1830.53 | 1833.46 | 1843.22 | EMA400 retest candle locked (from downside) |

### Cycle 75 — BUY (started 2024-04-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-09 14:15:00 | 1851.85 | 1843.57 | 1843.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-10 09:15:00 | 1865.80 | 1849.74 | 1846.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-10 14:15:00 | 1852.90 | 1857.32 | 1852.27 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-10 14:15:00 | 1852.90 | 1857.32 | 1852.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-10 14:15:00 | 1852.90 | 1857.32 | 1852.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-10 15:00:00 | 1852.90 | 1857.32 | 1852.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-10 15:15:00 | 1852.38 | 1856.33 | 1852.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-12 09:15:00 | 1855.35 | 1856.33 | 1852.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-12 09:15:00 | 1847.10 | 1854.49 | 1851.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-12 09:45:00 | 1842.78 | 1854.49 | 1851.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-12 10:15:00 | 1861.83 | 1855.95 | 1852.72 | EMA400 retest candle locked (from upside) |

### Cycle 76 — SELL (started 2024-04-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-12 14:15:00 | 1840.83 | 1849.68 | 1850.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-12 15:15:00 | 1830.00 | 1845.74 | 1848.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-16 09:15:00 | 1861.40 | 1834.69 | 1838.40 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-16 09:15:00 | 1861.40 | 1834.69 | 1838.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-16 09:15:00 | 1861.40 | 1834.69 | 1838.40 | EMA400 retest candle locked (from downside) |

### Cycle 77 — BUY (started 2024-04-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-16 11:15:00 | 1874.38 | 1847.25 | 1843.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-18 09:15:00 | 1899.88 | 1868.83 | 1856.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-18 14:15:00 | 1883.60 | 1887.43 | 1872.26 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-04-18 15:00:00 | 1883.60 | 1887.43 | 1872.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-19 09:15:00 | 1848.30 | 1879.11 | 1871.08 | EMA400 retest candle locked (from upside) |

### Cycle 78 — SELL (started 2024-04-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-19 11:15:00 | 1840.00 | 1863.03 | 1864.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-22 15:15:00 | 1820.00 | 1834.24 | 1846.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-23 09:15:00 | 1843.68 | 1836.13 | 1845.95 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-23 09:15:00 | 1843.68 | 1836.13 | 1845.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-23 09:15:00 | 1843.68 | 1836.13 | 1845.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-23 10:00:00 | 1843.68 | 1836.13 | 1845.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-24 14:15:00 | 1830.38 | 1820.07 | 1828.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-24 14:45:00 | 1832.28 | 1820.07 | 1828.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-24 15:15:00 | 1827.55 | 1821.56 | 1828.38 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-25 09:15:00 | 1823.23 | 1821.56 | 1828.38 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-25 10:15:00 | 1825.60 | 1822.65 | 1828.25 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-25 10:45:00 | 1821.20 | 1823.08 | 1827.94 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-04-25 11:15:00 | 1836.45 | 1825.75 | 1828.71 | SL hit (close>static) qty=1.00 sl=1832.08 alert=retest2 |

### Cycle 79 — BUY (started 2024-04-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-25 13:15:00 | 1850.50 | 1832.98 | 1831.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-26 09:15:00 | 1861.08 | 1842.69 | 1836.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-26 14:15:00 | 1858.48 | 1859.97 | 1849.24 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-04-26 15:00:00 | 1858.48 | 1859.97 | 1849.24 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-06 10:15:00 | 1941.03 | 1954.34 | 1946.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-06 11:00:00 | 1941.03 | 1954.34 | 1946.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-06 11:15:00 | 1947.63 | 1952.99 | 1946.17 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-06 14:15:00 | 1949.95 | 1950.88 | 1946.32 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-07 09:15:00 | 1959.18 | 1948.18 | 1945.81 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-07 10:45:00 | 1948.98 | 1947.89 | 1946.06 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-07 12:15:00 | 1893.38 | 1935.52 | 1940.66 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 80 — SELL (started 2024-05-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-07 12:15:00 | 1893.38 | 1935.52 | 1940.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-07 13:15:00 | 1883.03 | 1925.02 | 1935.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-10 14:15:00 | 1821.15 | 1821.00 | 1842.13 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-10 15:00:00 | 1821.15 | 1821.00 | 1842.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-13 14:15:00 | 1831.40 | 1812.36 | 1825.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-13 15:00:00 | 1831.40 | 1812.36 | 1825.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-13 15:15:00 | 1831.60 | 1816.20 | 1825.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-14 09:15:00 | 1831.53 | 1816.20 | 1825.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-14 10:15:00 | 1848.85 | 1824.87 | 1828.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-14 11:00:00 | 1848.85 | 1824.87 | 1828.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 81 — BUY (started 2024-05-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-14 11:15:00 | 1854.50 | 1830.80 | 1830.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-15 11:15:00 | 1861.10 | 1847.44 | 1840.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-17 09:15:00 | 1906.90 | 1908.05 | 1886.21 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-17 09:45:00 | 1906.70 | 1908.05 | 1886.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-17 15:15:00 | 1892.50 | 1897.49 | 1889.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-18 09:15:00 | 1895.48 | 1897.49 | 1889.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-18 09:15:00 | 1891.00 | 1896.19 | 1889.82 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-22 12:15:00 | 1904.98 | 1897.21 | 1894.37 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-22 15:00:00 | 1903.98 | 1900.01 | 1896.50 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-23 12:15:00 | 1904.23 | 1896.39 | 1895.65 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-23 13:15:00 | 1905.40 | 1896.83 | 1895.92 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-24 14:15:00 | 1908.68 | 1907.35 | 1903.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-24 15:00:00 | 1908.68 | 1907.35 | 1903.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-27 09:15:00 | 1949.63 | 1915.67 | 1908.00 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-27 10:15:00 | 1959.05 | 1915.67 | 1908.00 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-29 14:45:00 | 1955.70 | 1969.64 | 1965.08 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-30 09:15:00 | 1882.50 | 1948.59 | 1956.08 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 82 — SELL (started 2024-05-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-30 09:15:00 | 1882.50 | 1948.59 | 1956.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-30 11:15:00 | 1877.83 | 1924.65 | 1943.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-31 10:15:00 | 1905.20 | 1898.07 | 1918.60 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-31 11:00:00 | 1905.20 | 1898.07 | 1918.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 13:15:00 | 1926.40 | 1906.80 | 1917.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-31 14:00:00 | 1926.40 | 1906.80 | 1917.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 14:15:00 | 1939.28 | 1913.30 | 1919.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-31 15:00:00 | 1939.28 | 1913.30 | 1919.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 15:15:00 | 1950.00 | 1920.64 | 1922.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-03 09:15:00 | 2049.45 | 1920.64 | 1922.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 83 — BUY (started 2024-06-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-03 09:15:00 | 2060.05 | 1948.52 | 1934.94 | EMA200 above EMA400 |

### Cycle 84 — SELL (started 2024-06-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-04 11:15:00 | 1749.43 | 1936.61 | 1951.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-04 14:15:00 | 1733.45 | 1846.21 | 1902.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-05 10:15:00 | 1835.43 | 1814.12 | 1870.93 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-06-05 11:00:00 | 1835.43 | 1814.12 | 1870.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 12:15:00 | 1864.93 | 1829.60 | 1868.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-05 12:45:00 | 1860.25 | 1829.60 | 1868.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 13:15:00 | 1849.90 | 1833.66 | 1866.75 | EMA400 retest candle locked (from downside) |

### Cycle 85 — BUY (started 2024-06-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-06 15:15:00 | 1885.50 | 1876.18 | 1875.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-07 09:15:00 | 1891.23 | 1879.19 | 1876.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-10 10:15:00 | 1881.18 | 1897.66 | 1890.45 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-10 10:15:00 | 1881.18 | 1897.66 | 1890.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-10 10:15:00 | 1881.18 | 1897.66 | 1890.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-10 11:00:00 | 1881.18 | 1897.66 | 1890.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-10 11:15:00 | 1885.45 | 1895.22 | 1890.00 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-10 12:30:00 | 1899.90 | 1894.69 | 1890.23 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-10 15:00:00 | 1899.73 | 1895.78 | 1891.51 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-11 11:15:00 | 1900.93 | 1892.64 | 1890.86 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-19 09:15:00 | 1965.00 | 1978.49 | 1979.21 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 86 — SELL (started 2024-06-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-19 09:15:00 | 1965.00 | 1978.49 | 1979.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-19 10:15:00 | 1957.00 | 1974.19 | 1977.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-21 09:15:00 | 1954.55 | 1950.35 | 1957.30 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-21 09:15:00 | 1954.55 | 1950.35 | 1957.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-21 09:15:00 | 1954.55 | 1950.35 | 1957.30 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-21 11:00:00 | 1948.08 | 1949.90 | 1956.46 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-21 12:30:00 | 1946.80 | 1949.13 | 1954.94 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-24 09:15:00 | 1916.65 | 1951.54 | 1954.76 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-24 10:15:00 | 1978.95 | 1952.79 | 1954.51 | SL hit (close>static) qty=1.00 sl=1974.18 alert=retest2 |

### Cycle 87 — BUY (started 2024-06-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-24 11:15:00 | 1978.15 | 1957.86 | 1956.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-24 12:15:00 | 1991.38 | 1964.56 | 1959.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-26 09:15:00 | 2029.95 | 2030.74 | 2006.08 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-26 09:45:00 | 2022.75 | 2030.74 | 2006.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 09:15:00 | 2026.48 | 2027.37 | 2016.21 | EMA400 retest candle locked (from upside) |

### Cycle 88 — SELL (started 2024-06-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-28 14:15:00 | 1994.75 | 2010.60 | 2012.72 | EMA200 below EMA400 |

### Cycle 89 — BUY (started 2024-07-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-01 10:15:00 | 2040.95 | 2016.72 | 2014.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-01 14:15:00 | 2056.25 | 2032.37 | 2023.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-02 09:15:00 | 2030.50 | 2034.98 | 2026.47 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-02 09:15:00 | 2030.50 | 2034.98 | 2026.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-02 09:15:00 | 2030.50 | 2034.98 | 2026.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-02 09:30:00 | 2020.03 | 2034.98 | 2026.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-02 10:15:00 | 2048.85 | 2037.75 | 2028.51 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-03 09:15:00 | 2064.00 | 2041.52 | 2033.81 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-03 15:00:00 | 2063.32 | 2061.16 | 2049.12 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-08 11:45:00 | 2059.53 | 2098.12 | 2092.44 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-08 14:15:00 | 2081.23 | 2088.08 | 2088.66 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 90 — SELL (started 2024-07-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-08 14:15:00 | 2081.23 | 2088.08 | 2088.66 | EMA200 below EMA400 |

### Cycle 91 — BUY (started 2024-07-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-09 14:15:00 | 2094.20 | 2088.27 | 2087.89 | EMA200 above EMA400 |

### Cycle 92 — SELL (started 2024-07-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-10 09:15:00 | 2052.93 | 2081.88 | 2085.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-10 10:15:00 | 2033.93 | 2072.29 | 2080.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-10 14:15:00 | 2079.68 | 2071.23 | 2076.97 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-10 14:15:00 | 2079.68 | 2071.23 | 2076.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-10 14:15:00 | 2079.68 | 2071.23 | 2076.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-10 14:45:00 | 2078.07 | 2071.23 | 2076.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-10 15:15:00 | 2075.13 | 2072.01 | 2076.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-11 09:15:00 | 2081.60 | 2072.01 | 2076.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-11 09:15:00 | 2075.00 | 2072.61 | 2076.64 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-11 10:15:00 | 2069.18 | 2072.61 | 2076.64 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-11 12:15:00 | 2093.80 | 2078.81 | 2078.60 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 93 — BUY (started 2024-07-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-11 12:15:00 | 2093.80 | 2078.81 | 2078.60 | EMA200 above EMA400 |

### Cycle 94 — SELL (started 2024-07-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-15 09:15:00 | 2075.00 | 2080.74 | 2080.82 | EMA200 below EMA400 |

### Cycle 95 — BUY (started 2024-07-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-15 11:15:00 | 2093.43 | 2081.94 | 2081.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-15 12:15:00 | 2102.75 | 2086.10 | 2083.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-16 12:15:00 | 2094.30 | 2096.54 | 2091.43 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-16 12:15:00 | 2094.30 | 2096.54 | 2091.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-16 12:15:00 | 2094.30 | 2096.54 | 2091.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-16 13:00:00 | 2094.30 | 2096.54 | 2091.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-16 13:15:00 | 2080.88 | 2093.41 | 2090.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-16 14:00:00 | 2080.88 | 2093.41 | 2090.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 96 — SELL (started 2024-07-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-16 14:15:00 | 2040.25 | 2082.78 | 2085.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-18 10:15:00 | 2037.40 | 2064.74 | 2075.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-18 12:15:00 | 2063.85 | 2061.00 | 2072.12 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-07-18 12:45:00 | 2066.23 | 2061.00 | 2072.12 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-18 13:15:00 | 2067.70 | 2062.34 | 2071.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-18 14:00:00 | 2067.70 | 2062.34 | 2071.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-18 14:15:00 | 2079.60 | 2065.79 | 2072.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-18 14:45:00 | 2072.50 | 2065.79 | 2072.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-18 15:15:00 | 2078.00 | 2068.23 | 2072.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-19 09:15:00 | 2070.57 | 2068.23 | 2072.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 10:15:00 | 2063.00 | 2044.54 | 2053.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 10:45:00 | 2056.00 | 2044.54 | 2053.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 11:15:00 | 2032.08 | 2042.05 | 2051.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 11:30:00 | 2045.78 | 2042.05 | 2051.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 13:15:00 | 2058.18 | 2045.73 | 2051.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 13:30:00 | 2055.90 | 2045.73 | 2051.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 14:15:00 | 2028.50 | 2042.28 | 2049.66 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-22 15:15:00 | 2020.00 | 2042.28 | 2049.66 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-23 12:15:00 | 1906.00 | 2032.87 | 2042.10 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-23 12:15:00 | 1919.00 | 2019.30 | 2035.09 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-07-24 09:15:00 | 2017.53 | 2004.15 | 2021.63 | SL hit (close>ema200) qty=0.50 sl=2004.15 alert=retest2 |

### Cycle 97 — BUY (started 2024-07-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-25 15:15:00 | 2039.45 | 2023.80 | 2022.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-26 09:15:00 | 2060.55 | 2031.15 | 2025.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-26 15:15:00 | 2045.00 | 2049.89 | 2039.60 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-29 09:15:00 | 2074.20 | 2049.89 | 2039.60 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-30 09:15:00 | 2056.07 | 2067.40 | 2057.60 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-07-30 09:15:00 | 2056.07 | 2067.40 | 2057.60 | SL hit (close<ema400) qty=1.00 sl=2057.60 alert=retest1 |

### Cycle 98 — SELL (started 2024-07-31 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-31 14:15:00 | 2057.43 | 2059.69 | 2059.96 | EMA200 below EMA400 |

### Cycle 99 — BUY (started 2024-08-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-01 09:15:00 | 2090.95 | 2065.51 | 2062.54 | EMA200 above EMA400 |

### Cycle 100 — SELL (started 2024-08-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-02 15:15:00 | 2053.90 | 2070.84 | 2072.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-05 09:15:00 | 1996.08 | 2055.89 | 2065.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-06 09:15:00 | 2030.00 | 2023.36 | 2039.43 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-06 09:15:00 | 2030.00 | 2023.36 | 2039.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-06 09:15:00 | 2030.00 | 2023.36 | 2039.43 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-06 10:15:00 | 2016.50 | 2023.36 | 2039.43 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-06 12:45:00 | 2015.20 | 2021.69 | 2034.50 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-06 13:30:00 | 2014.80 | 2017.18 | 2031.28 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-08 09:15:00 | 2061.50 | 2034.95 | 2031.66 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 101 — BUY (started 2024-08-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-08 09:15:00 | 2061.50 | 2034.95 | 2031.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-08 10:15:00 | 2077.53 | 2043.46 | 2035.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-08 14:15:00 | 2052.50 | 2055.77 | 2045.32 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-08 15:00:00 | 2052.50 | 2055.77 | 2045.32 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-12 09:15:00 | 2037.50 | 2062.51 | 2057.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-12 10:00:00 | 2037.50 | 2062.51 | 2057.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-12 10:15:00 | 2040.25 | 2058.06 | 2055.72 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-12 11:15:00 | 2046.35 | 2058.06 | 2055.72 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2024-08-27 11:15:00 | 2250.99 | 2220.34 | 2206.10 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 102 — SELL (started 2024-08-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-30 09:15:00 | 2211.00 | 2226.35 | 2227.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-02 14:15:00 | 2202.98 | 2215.13 | 2218.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-03 14:15:00 | 2230.10 | 2205.48 | 2210.42 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-03 14:15:00 | 2230.10 | 2205.48 | 2210.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-03 14:15:00 | 2230.10 | 2205.48 | 2210.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-03 15:00:00 | 2230.10 | 2205.48 | 2210.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-03 15:15:00 | 2221.53 | 2208.69 | 2211.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-04 09:15:00 | 2227.38 | 2208.69 | 2211.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 103 — BUY (started 2024-09-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-04 09:15:00 | 2236.13 | 2214.18 | 2213.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-04 13:15:00 | 2242.70 | 2225.27 | 2219.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-06 09:15:00 | 2246.35 | 2251.52 | 2241.82 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-06 09:15:00 | 2246.35 | 2251.52 | 2241.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-06 09:15:00 | 2246.35 | 2251.52 | 2241.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-06 10:00:00 | 2246.35 | 2251.52 | 2241.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-06 10:15:00 | 2227.30 | 2246.67 | 2240.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-06 11:00:00 | 2227.30 | 2246.67 | 2240.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-06 11:15:00 | 2229.05 | 2243.15 | 2239.46 | EMA400 retest candle locked (from upside) |

### Cycle 104 — SELL (started 2024-09-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-06 13:15:00 | 2225.88 | 2235.92 | 2236.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-06 14:15:00 | 2206.00 | 2229.93 | 2233.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-10 15:15:00 | 2198.98 | 2190.68 | 2200.74 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-11 09:15:00 | 2208.90 | 2190.68 | 2200.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-11 09:15:00 | 2208.55 | 2194.25 | 2201.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-11 09:30:00 | 2226.45 | 2194.25 | 2201.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-11 10:15:00 | 2220.00 | 2199.40 | 2203.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-11 11:00:00 | 2220.00 | 2199.40 | 2203.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-11 13:15:00 | 2195.00 | 2201.94 | 2203.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-11 13:30:00 | 2202.75 | 2201.94 | 2203.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-11 14:15:00 | 2200.60 | 2201.67 | 2203.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-11 15:00:00 | 2200.60 | 2201.67 | 2203.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-11 15:15:00 | 2194.50 | 2200.24 | 2202.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-12 09:15:00 | 2205.18 | 2200.24 | 2202.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-12 09:15:00 | 2200.00 | 2200.19 | 2202.42 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-12 13:00:00 | 2192.50 | 2197.57 | 2200.59 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-12 14:15:00 | 2188.53 | 2196.67 | 2199.91 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-13 10:15:00 | 2216.60 | 2202.67 | 2201.72 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 105 — BUY (started 2024-09-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-13 10:15:00 | 2216.60 | 2202.67 | 2201.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-16 10:15:00 | 2226.98 | 2210.79 | 2206.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-16 11:15:00 | 2203.28 | 2209.29 | 2205.95 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-16 11:15:00 | 2203.28 | 2209.29 | 2205.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-16 11:15:00 | 2203.28 | 2209.29 | 2205.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-16 12:00:00 | 2203.28 | 2209.29 | 2205.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-16 12:15:00 | 2211.68 | 2209.77 | 2206.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-16 12:30:00 | 2202.65 | 2209.77 | 2206.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-16 13:15:00 | 2210.18 | 2209.85 | 2206.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-16 14:00:00 | 2210.18 | 2209.85 | 2206.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-16 14:15:00 | 2211.90 | 2210.26 | 2207.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-16 14:30:00 | 2207.10 | 2210.26 | 2207.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-16 15:15:00 | 2215.00 | 2211.21 | 2207.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-17 09:15:00 | 2204.50 | 2211.21 | 2207.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-17 09:15:00 | 2211.18 | 2211.20 | 2208.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-17 10:15:00 | 2201.48 | 2211.20 | 2208.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-17 10:15:00 | 2209.73 | 2210.91 | 2208.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-17 10:30:00 | 2209.35 | 2210.91 | 2208.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-17 11:15:00 | 2205.68 | 2209.86 | 2208.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-17 12:00:00 | 2205.68 | 2209.86 | 2208.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-17 12:15:00 | 2205.35 | 2208.96 | 2207.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-17 12:45:00 | 2204.68 | 2208.96 | 2207.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-17 14:15:00 | 2214.50 | 2210.19 | 2208.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-17 15:00:00 | 2214.50 | 2210.19 | 2208.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-17 15:15:00 | 2213.15 | 2210.79 | 2209.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-18 09:15:00 | 2219.43 | 2210.79 | 2209.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-18 09:15:00 | 2225.68 | 2213.76 | 2210.57 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-18 10:30:00 | 2230.40 | 2215.04 | 2211.44 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-18 11:30:00 | 2229.35 | 2219.07 | 2213.60 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-18 12:00:00 | 2235.18 | 2219.07 | 2213.60 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-18 13:15:00 | 2201.23 | 2214.32 | 2212.32 | SL hit (close<static) qty=1.00 sl=2202.07 alert=retest2 |

### Cycle 106 — SELL (started 2024-09-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-19 09:15:00 | 2193.98 | 2210.80 | 2211.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-19 10:15:00 | 2182.53 | 2205.14 | 2208.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-19 14:15:00 | 2198.82 | 2186.52 | 2196.98 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-19 14:15:00 | 2198.82 | 2186.52 | 2196.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 14:15:00 | 2198.82 | 2186.52 | 2196.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-19 15:00:00 | 2198.82 | 2186.52 | 2196.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 15:15:00 | 2187.70 | 2186.76 | 2196.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-20 09:15:00 | 2182.50 | 2186.76 | 2196.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 09:15:00 | 2197.40 | 2188.88 | 2196.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-20 10:00:00 | 2197.40 | 2188.88 | 2196.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 10:15:00 | 2191.95 | 2189.50 | 2195.86 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-20 11:30:00 | 2186.00 | 2189.20 | 2195.15 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-20 12:15:00 | 2200.00 | 2191.36 | 2195.59 | SL hit (close>static) qty=1.00 sl=2199.05 alert=retest2 |

### Cycle 107 — BUY (started 2024-09-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-23 12:15:00 | 2201.40 | 2195.65 | 2195.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-24 09:15:00 | 2225.15 | 2206.22 | 2201.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-25 09:15:00 | 2227.57 | 2230.38 | 2218.99 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-25 10:00:00 | 2227.57 | 2230.38 | 2218.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 12:15:00 | 2217.25 | 2227.92 | 2220.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-25 13:00:00 | 2217.25 | 2227.92 | 2220.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 13:15:00 | 2216.00 | 2225.54 | 2220.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-25 13:30:00 | 2214.82 | 2225.54 | 2220.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 14:15:00 | 2239.15 | 2228.26 | 2222.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-25 14:30:00 | 2221.18 | 2228.26 | 2222.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-26 09:15:00 | 2217.05 | 2227.06 | 2222.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-26 09:45:00 | 2209.95 | 2227.06 | 2222.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-26 10:15:00 | 2216.05 | 2224.85 | 2222.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-26 10:45:00 | 2216.18 | 2224.85 | 2222.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-27 09:15:00 | 2226.00 | 2238.13 | 2230.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-27 10:00:00 | 2226.00 | 2238.13 | 2230.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-27 10:15:00 | 2222.13 | 2234.93 | 2230.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-27 11:00:00 | 2222.13 | 2234.93 | 2230.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-27 12:15:00 | 2227.93 | 2232.02 | 2229.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-27 12:45:00 | 2226.00 | 2232.02 | 2229.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-27 13:15:00 | 2221.23 | 2229.86 | 2228.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-27 14:00:00 | 2221.23 | 2229.86 | 2228.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 108 — SELL (started 2024-09-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-27 14:15:00 | 2206.98 | 2225.29 | 2226.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-30 09:15:00 | 2175.13 | 2212.81 | 2220.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-01 09:15:00 | 2173.60 | 2168.31 | 2188.99 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-01 09:30:00 | 2177.50 | 2168.31 | 2188.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-01 11:15:00 | 2184.38 | 2173.55 | 2187.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-01 11:30:00 | 2192.35 | 2173.55 | 2187.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-01 12:15:00 | 2184.98 | 2175.84 | 2187.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-01 13:15:00 | 2192.93 | 2175.84 | 2187.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-01 13:15:00 | 2188.80 | 2178.43 | 2187.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-01 13:30:00 | 2190.78 | 2178.43 | 2187.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-01 14:15:00 | 2179.48 | 2178.64 | 2187.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-01 15:15:00 | 2186.95 | 2178.64 | 2187.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-01 15:15:00 | 2186.95 | 2180.30 | 2186.99 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-03 09:15:00 | 2155.00 | 2180.30 | 2186.99 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-09 10:15:00 | 2168.50 | 2116.71 | 2110.04 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 109 — BUY (started 2024-10-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-09 10:15:00 | 2168.50 | 2116.71 | 2110.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-09 11:15:00 | 2184.07 | 2130.18 | 2116.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-11 09:15:00 | 2182.20 | 2194.61 | 2172.86 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-11 10:00:00 | 2182.20 | 2194.61 | 2172.86 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-11 10:15:00 | 2175.65 | 2190.81 | 2173.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-11 11:00:00 | 2175.65 | 2190.81 | 2173.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-11 11:15:00 | 2201.68 | 2192.99 | 2175.71 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-11 13:15:00 | 2203.75 | 2194.21 | 2177.84 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-11 13:45:00 | 2205.80 | 2196.36 | 2180.30 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2024-10-16 12:15:00 | 2424.12 | 2350.23 | 2294.85 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 110 — SELL (started 2024-10-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-21 14:15:00 | 2300.00 | 2340.11 | 2344.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-22 10:15:00 | 2286.95 | 2317.17 | 2332.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-29 11:15:00 | 2167.93 | 2160.12 | 2180.31 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-29 12:00:00 | 2167.93 | 2160.12 | 2180.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-29 12:15:00 | 2186.65 | 2165.43 | 2180.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-29 13:00:00 | 2186.65 | 2165.43 | 2180.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-29 13:15:00 | 2199.00 | 2172.14 | 2182.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-29 14:00:00 | 2199.00 | 2172.14 | 2182.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-29 14:15:00 | 2205.68 | 2178.85 | 2184.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-29 14:45:00 | 2217.38 | 2178.85 | 2184.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 111 — BUY (started 2024-10-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-30 09:15:00 | 2208.60 | 2190.43 | 2189.25 | EMA200 above EMA400 |

### Cycle 112 — SELL (started 2024-10-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-31 09:15:00 | 2172.90 | 2191.46 | 2191.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-31 11:15:00 | 2150.00 | 2180.39 | 2186.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-01 17:15:00 | 2170.93 | 2161.93 | 2173.01 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-01 17:15:00 | 2170.93 | 2161.93 | 2173.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-01 17:15:00 | 2170.93 | 2161.93 | 2173.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-01 18:00:00 | 2170.93 | 2161.93 | 2173.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-01 18:15:00 | 2165.40 | 2162.63 | 2172.32 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-04 09:15:00 | 2137.55 | 2162.63 | 2172.32 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-05 13:45:00 | 2149.73 | 2132.05 | 2144.09 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-06 10:15:00 | 2176.23 | 2149.36 | 2148.93 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 113 — BUY (started 2024-11-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-06 10:15:00 | 2176.23 | 2149.36 | 2148.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-06 11:15:00 | 2202.70 | 2160.03 | 2153.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-08 11:15:00 | 2239.05 | 2246.62 | 2224.29 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-08 12:00:00 | 2239.05 | 2246.62 | 2224.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-08 15:15:00 | 2239.90 | 2245.81 | 2231.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-11 09:15:00 | 2213.23 | 2245.81 | 2231.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-11 09:15:00 | 2220.70 | 2240.79 | 2230.21 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-11 11:15:00 | 2246.07 | 2240.85 | 2231.19 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-11 12:00:00 | 2243.55 | 2241.39 | 2232.32 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-11 14:15:00 | 2245.65 | 2240.45 | 2233.45 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-12 09:15:00 | 2251.50 | 2234.21 | 2231.66 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-12 10:15:00 | 2240.88 | 2236.97 | 2233.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-12 10:30:00 | 2229.95 | 2236.97 | 2233.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-12 12:15:00 | 2223.00 | 2236.85 | 2234.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-12 13:00:00 | 2223.00 | 2236.85 | 2234.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-12 13:15:00 | 2216.75 | 2232.83 | 2232.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-12 13:30:00 | 2208.05 | 2232.83 | 2232.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-11-12 14:15:00 | 2199.98 | 2226.26 | 2229.62 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 114 — SELL (started 2024-11-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-12 14:15:00 | 2199.98 | 2226.26 | 2229.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-12 15:15:00 | 2191.63 | 2219.34 | 2226.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-14 09:15:00 | 2174.43 | 2160.14 | 2184.08 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-14 10:00:00 | 2174.43 | 2160.14 | 2184.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-21 11:15:00 | 2102.60 | 2095.44 | 2108.42 | EMA400 retest candle locked (from downside) |

### Cycle 115 — BUY (started 2024-11-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-25 09:15:00 | 2171.35 | 2120.10 | 2113.53 | EMA200 above EMA400 |

### Cycle 116 — SELL (started 2024-11-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-27 09:15:00 | 2117.25 | 2132.50 | 2133.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-28 13:15:00 | 2097.98 | 2116.15 | 2122.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-28 15:15:00 | 2135.00 | 2118.23 | 2122.24 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-28 15:15:00 | 2135.00 | 2118.23 | 2122.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 15:15:00 | 2135.00 | 2118.23 | 2122.24 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-29 10:00:00 | 2099.88 | 2114.56 | 2120.21 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-29 13:30:00 | 2101.15 | 2109.71 | 2116.00 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-29 15:00:00 | 2100.38 | 2107.84 | 2114.58 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-02 09:15:00 | 2093.65 | 2107.20 | 2113.68 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-02 09:15:00 | 2114.25 | 2108.61 | 2113.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-02 10:00:00 | 2114.25 | 2108.61 | 2113.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-02 10:15:00 | 2142.98 | 2115.49 | 2116.39 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-12-02 10:15:00 | 2142.98 | 2115.49 | 2116.39 | SL hit (close>static) qty=1.00 sl=2135.00 alert=retest2 |

### Cycle 117 — BUY (started 2024-12-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-02 11:15:00 | 2130.25 | 2118.44 | 2117.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-04 09:15:00 | 2167.30 | 2135.97 | 2128.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-06 14:15:00 | 2182.53 | 2182.97 | 2172.74 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-06 15:00:00 | 2182.53 | 2182.97 | 2172.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-09 09:15:00 | 2189.90 | 2183.68 | 2174.80 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-09 10:15:00 | 2201.60 | 2183.68 | 2174.80 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-17 09:15:00 | 2222.82 | 2251.88 | 2253.02 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 118 — SELL (started 2024-12-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-17 09:15:00 | 2222.82 | 2251.88 | 2253.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-17 10:15:00 | 2219.00 | 2245.30 | 2249.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-19 12:15:00 | 2177.53 | 2175.54 | 2195.26 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-19 13:00:00 | 2177.53 | 2175.54 | 2195.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-19 14:15:00 | 2198.00 | 2178.75 | 2193.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-19 15:00:00 | 2198.00 | 2178.75 | 2193.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-19 15:15:00 | 2195.03 | 2182.00 | 2193.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-20 09:15:00 | 2195.60 | 2182.00 | 2193.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 09:15:00 | 2192.25 | 2184.05 | 2193.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-20 09:45:00 | 2201.18 | 2184.05 | 2193.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 10:15:00 | 2196.10 | 2186.46 | 2193.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-20 10:45:00 | 2199.93 | 2186.46 | 2193.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 11:15:00 | 2174.40 | 2184.05 | 2191.79 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-20 12:30:00 | 2165.03 | 2178.37 | 2188.51 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-27 11:15:00 | 2141.70 | 2134.30 | 2134.13 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 119 — BUY (started 2024-12-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-27 11:15:00 | 2141.70 | 2134.30 | 2134.13 | EMA200 above EMA400 |

### Cycle 120 — SELL (started 2024-12-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-30 12:15:00 | 2120.10 | 2132.13 | 2133.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-30 13:15:00 | 2100.00 | 2125.71 | 2130.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-30 14:15:00 | 2128.68 | 2126.30 | 2130.31 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-30 14:15:00 | 2128.68 | 2126.30 | 2130.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-30 14:15:00 | 2128.68 | 2126.30 | 2130.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-30 15:00:00 | 2128.68 | 2126.30 | 2130.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-30 15:15:00 | 2124.00 | 2125.84 | 2129.73 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-31 09:15:00 | 2106.68 | 2125.84 | 2129.73 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-31 14:15:00 | 2108.40 | 2113.21 | 2120.89 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-02 12:00:00 | 2108.05 | 2096.72 | 2101.92 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-02 13:15:00 | 2136.50 | 2109.76 | 2107.26 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 121 — BUY (started 2025-01-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-02 13:15:00 | 2136.50 | 2109.76 | 2107.26 | EMA200 above EMA400 |

### Cycle 122 — SELL (started 2025-01-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-03 13:15:00 | 2105.38 | 2108.55 | 2108.97 | EMA200 below EMA400 |

### Cycle 123 — BUY (started 2025-01-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-03 14:15:00 | 2112.07 | 2109.26 | 2109.25 | EMA200 above EMA400 |

### Cycle 124 — SELL (started 2025-01-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-06 09:15:00 | 2082.13 | 2104.51 | 2107.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-06 10:15:00 | 2048.65 | 2093.34 | 2101.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-08 13:15:00 | 2036.70 | 2029.05 | 2045.80 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-08 14:00:00 | 2036.70 | 2029.05 | 2045.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-14 09:15:00 | 1954.90 | 1934.08 | 1953.14 | EMA400 retest candle locked (from downside) |

### Cycle 125 — BUY (started 2025-01-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-15 09:15:00 | 2011.98 | 1961.07 | 1959.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-15 11:15:00 | 2038.08 | 1985.75 | 1971.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-17 09:15:00 | 2031.95 | 2038.20 | 2018.99 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-17 09:30:00 | 2033.45 | 2038.20 | 2018.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-17 10:15:00 | 2022.00 | 2034.96 | 2019.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-17 11:00:00 | 2022.00 | 2034.96 | 2019.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-17 11:15:00 | 2016.33 | 2031.23 | 2019.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-17 11:45:00 | 2016.10 | 2031.23 | 2019.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-17 12:15:00 | 2006.03 | 2026.19 | 2017.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-17 12:45:00 | 2006.65 | 2026.19 | 2017.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-17 13:15:00 | 2008.98 | 2022.75 | 2017.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-17 13:45:00 | 2009.58 | 2022.75 | 2017.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-17 15:15:00 | 2023.45 | 2021.65 | 2017.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-20 09:15:00 | 2015.20 | 2021.65 | 2017.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-20 09:15:00 | 2019.28 | 2021.18 | 2017.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-20 09:45:00 | 2008.23 | 2021.18 | 2017.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-20 10:15:00 | 2030.18 | 2022.98 | 2018.75 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-21 13:30:00 | 2039.13 | 2024.45 | 2021.41 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-21 14:15:00 | 2014.45 | 2022.45 | 2020.78 | SL hit (close<static) qty=1.00 sl=2017.50 alert=retest2 |

### Cycle 126 — SELL (started 2025-01-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-22 09:15:00 | 2003.00 | 2017.37 | 2018.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-22 12:15:00 | 1974.93 | 2002.43 | 2011.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-23 10:15:00 | 2002.03 | 1991.76 | 2001.06 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-23 10:15:00 | 2002.03 | 1991.76 | 2001.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 10:15:00 | 2002.03 | 1991.76 | 2001.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 10:45:00 | 2004.33 | 1991.76 | 2001.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 11:15:00 | 2001.80 | 1993.77 | 2001.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 12:00:00 | 2001.80 | 1993.77 | 2001.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 12:15:00 | 2010.00 | 1997.01 | 2001.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 12:30:00 | 2013.48 | 1997.01 | 2001.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 13:15:00 | 2009.03 | 1999.42 | 2002.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 13:45:00 | 2012.50 | 1999.42 | 2002.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 15:15:00 | 2000.00 | 1999.83 | 2002.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-24 09:15:00 | 2004.43 | 1999.83 | 2002.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-24 09:15:00 | 1986.30 | 1997.12 | 2000.78 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-24 12:30:00 | 1962.23 | 1984.77 | 1993.79 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-27 10:15:00 | 1864.12 | 1928.02 | 1960.67 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-01-28 11:15:00 | 1872.58 | 1871.75 | 1907.30 | SL hit (close>ema200) qty=0.50 sl=1871.75 alert=retest2 |

### Cycle 127 — BUY (started 2025-01-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-31 09:15:00 | 1922.13 | 1893.95 | 1891.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-31 10:15:00 | 1933.30 | 1901.82 | 1895.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-01 11:15:00 | 1922.25 | 1935.80 | 1920.93 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-01 11:15:00 | 1922.25 | 1935.80 | 1920.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 11:15:00 | 1922.25 | 1935.80 | 1920.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-01 11:45:00 | 1910.80 | 1935.80 | 1920.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 12:15:00 | 1916.60 | 1931.96 | 1920.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-01 12:30:00 | 1919.53 | 1931.96 | 1920.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 13:15:00 | 1958.65 | 1937.30 | 1924.00 | EMA400 retest candle locked (from upside) |

### Cycle 128 — SELL (started 2025-02-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-03 14:15:00 | 1907.45 | 1921.06 | 1922.57 | EMA200 below EMA400 |

### Cycle 129 — BUY (started 2025-02-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-04 09:15:00 | 1948.80 | 1923.51 | 1923.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-05 09:15:00 | 1984.73 | 1940.73 | 1932.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-06 11:15:00 | 1989.63 | 1992.13 | 1971.92 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-06 12:00:00 | 1989.63 | 1992.13 | 1971.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 12:15:00 | 1976.08 | 1988.92 | 1972.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-06 13:00:00 | 1976.08 | 1988.92 | 1972.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 13:15:00 | 1967.90 | 1984.71 | 1971.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-06 13:45:00 | 1970.00 | 1984.71 | 1971.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 14:15:00 | 1969.28 | 1981.63 | 1971.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-06 14:30:00 | 1967.40 | 1981.63 | 1971.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 15:15:00 | 1963.15 | 1977.93 | 1970.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-07 09:15:00 | 1941.68 | 1977.93 | 1970.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 10:15:00 | 1973.40 | 1971.65 | 1968.92 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-07 11:15:00 | 1984.05 | 1971.65 | 1968.92 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-07 14:15:00 | 1962.80 | 1966.89 | 1967.42 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 130 — SELL (started 2025-02-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-07 14:15:00 | 1962.80 | 1966.89 | 1967.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-10 09:15:00 | 1941.68 | 1960.51 | 1964.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-12 11:15:00 | 1845.15 | 1843.00 | 1879.56 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-12 12:00:00 | 1845.15 | 1843.00 | 1879.56 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 12:15:00 | 1871.65 | 1848.73 | 1878.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-12 12:45:00 | 1868.00 | 1848.73 | 1878.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-13 09:15:00 | 1872.13 | 1855.62 | 1872.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-13 09:45:00 | 1867.50 | 1855.62 | 1872.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-13 10:15:00 | 1881.60 | 1860.81 | 1873.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-13 10:45:00 | 1887.53 | 1860.81 | 1873.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-13 11:15:00 | 1876.03 | 1863.86 | 1873.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-13 11:45:00 | 1881.63 | 1863.86 | 1873.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-13 12:15:00 | 1879.50 | 1866.99 | 1874.19 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-14 10:30:00 | 1862.65 | 1874.06 | 1875.64 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-14 12:00:00 | 1871.15 | 1873.48 | 1875.23 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-14 15:00:00 | 1875.53 | 1873.33 | 1874.67 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-17 09:45:00 | 1866.03 | 1871.76 | 1873.75 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-17 10:15:00 | 1867.50 | 1870.91 | 1873.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-17 10:45:00 | 1883.93 | 1870.91 | 1873.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-17 11:15:00 | 1862.50 | 1869.23 | 1872.21 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-02-19 09:15:00 | 1905.83 | 1869.35 | 1867.20 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 131 — BUY (started 2025-02-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-19 09:15:00 | 1905.83 | 1869.35 | 1867.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-20 09:15:00 | 1923.80 | 1893.52 | 1882.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-21 09:15:00 | 1920.13 | 1924.94 | 1907.26 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-21 10:00:00 | 1920.13 | 1924.94 | 1907.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 10:15:00 | 1907.00 | 1921.35 | 1907.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-21 11:00:00 | 1907.00 | 1921.35 | 1907.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 11:15:00 | 1903.53 | 1917.79 | 1906.90 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-21 13:00:00 | 1909.33 | 1916.10 | 1907.12 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-21 14:00:00 | 1908.33 | 1914.54 | 1907.23 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-21 15:15:00 | 1890.98 | 1907.78 | 1905.30 | SL hit (close<static) qty=1.00 sl=1898.00 alert=retest2 |

### Cycle 132 — SELL (started 2025-02-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-24 09:15:00 | 1883.83 | 1902.99 | 1903.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-27 09:15:00 | 1857.58 | 1882.78 | 1890.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-03 09:15:00 | 1824.65 | 1821.83 | 1840.13 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-03 09:15:00 | 1824.65 | 1821.83 | 1840.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-03 09:15:00 | 1824.65 | 1821.83 | 1840.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-03 09:30:00 | 1832.10 | 1821.83 | 1840.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-03 10:15:00 | 1837.43 | 1824.95 | 1839.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-03 10:30:00 | 1840.35 | 1824.95 | 1839.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-03 12:15:00 | 1855.50 | 1832.01 | 1840.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-03 12:45:00 | 1846.25 | 1832.01 | 1840.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-03 13:15:00 | 1855.33 | 1836.67 | 1841.90 | EMA400 retest candle locked (from downside) |

### Cycle 133 — BUY (started 2025-03-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-04 09:15:00 | 1856.58 | 1846.85 | 1845.82 | EMA200 above EMA400 |

### Cycle 134 — SELL (started 2025-03-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-04 11:15:00 | 1830.53 | 1843.26 | 1844.35 | EMA200 below EMA400 |

### Cycle 135 — BUY (started 2025-03-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-04 14:15:00 | 1849.75 | 1845.75 | 1845.26 | EMA200 above EMA400 |

### Cycle 136 — SELL (started 2025-03-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-04 15:15:00 | 1840.05 | 1844.61 | 1844.79 | EMA200 below EMA400 |

### Cycle 137 — BUY (started 2025-03-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-05 09:15:00 | 1871.40 | 1849.97 | 1847.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-05 10:15:00 | 1905.00 | 1860.98 | 1852.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-07 12:15:00 | 1919.18 | 1925.56 | 1910.12 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-07 13:00:00 | 1919.18 | 1925.56 | 1910.12 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 09:15:00 | 1901.85 | 1917.24 | 1910.87 | EMA400 retest candle locked (from upside) |

### Cycle 138 — SELL (started 2025-03-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-10 11:15:00 | 1880.53 | 1903.18 | 1905.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-10 15:15:00 | 1867.50 | 1887.66 | 1896.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-11 13:15:00 | 1878.50 | 1876.52 | 1886.71 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-11 14:00:00 | 1878.50 | 1876.52 | 1886.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-11 14:15:00 | 1889.20 | 1879.05 | 1886.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-11 14:30:00 | 1887.08 | 1879.05 | 1886.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-11 15:15:00 | 1893.20 | 1881.88 | 1887.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-12 09:15:00 | 1892.78 | 1881.88 | 1887.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 09:15:00 | 1899.33 | 1885.37 | 1888.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-12 10:00:00 | 1899.33 | 1885.37 | 1888.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 10:15:00 | 1900.23 | 1888.34 | 1889.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-12 10:30:00 | 1906.28 | 1888.34 | 1889.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 13:15:00 | 1885.28 | 1882.80 | 1886.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-12 13:30:00 | 1884.95 | 1882.80 | 1886.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 14:15:00 | 1884.45 | 1883.13 | 1886.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-12 15:00:00 | 1884.45 | 1883.13 | 1886.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 15:15:00 | 1879.15 | 1882.33 | 1885.57 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-13 09:15:00 | 1839.88 | 1882.33 | 1885.57 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-17 10:00:00 | 1878.88 | 1868.24 | 1873.84 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-17 13:30:00 | 1878.20 | 1870.91 | 1873.37 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-17 14:00:00 | 1878.50 | 1870.91 | 1873.37 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 14:15:00 | 1872.43 | 1871.22 | 1873.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-17 14:45:00 | 1874.53 | 1871.22 | 1873.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 15:15:00 | 1871.83 | 1871.34 | 1873.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-18 09:15:00 | 1894.73 | 1871.34 | 1873.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-03-18 09:15:00 | 1912.05 | 1879.48 | 1876.69 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 139 — BUY (started 2025-03-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-18 09:15:00 | 1912.05 | 1879.48 | 1876.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-18 14:15:00 | 1914.65 | 1899.36 | 1888.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-25 09:15:00 | 2029.00 | 2031.61 | 2009.76 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-25 10:00:00 | 2029.00 | 2031.61 | 2009.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-26 14:15:00 | 2036.58 | 2049.40 | 2037.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-26 15:00:00 | 2036.58 | 2049.40 | 2037.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-26 15:15:00 | 2033.00 | 2046.12 | 2037.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-27 09:15:00 | 2062.85 | 2046.12 | 2037.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 09:15:00 | 2047.93 | 2046.48 | 2038.37 | EMA400 retest candle locked (from upside) |

### Cycle 140 — SELL (started 2025-03-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-28 12:15:00 | 2019.45 | 2039.38 | 2039.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-28 13:15:00 | 2012.15 | 2033.94 | 2036.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-01 09:15:00 | 2026.38 | 2024.55 | 2031.22 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-01 09:45:00 | 2027.48 | 2024.55 | 2031.22 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-01 10:15:00 | 1987.00 | 2017.04 | 2027.20 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-01 11:15:00 | 1975.38 | 2017.04 | 2027.20 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-02 09:15:00 | 1986.68 | 1994.67 | 2010.42 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-02 10:00:00 | 1982.50 | 1992.24 | 2007.88 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-02 11:30:00 | 1986.75 | 1993.94 | 2005.95 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 12:15:00 | 2017.13 | 1998.58 | 2006.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-02 13:00:00 | 2017.13 | 1998.58 | 2006.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 13:15:00 | 2030.00 | 2004.86 | 2009.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-02 14:00:00 | 2030.00 | 2004.86 | 2009.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-04-02 14:15:00 | 2039.05 | 2011.70 | 2011.79 | SL hit (close>static) qty=1.00 sl=2035.00 alert=retest2 |

### Cycle 141 — BUY (started 2025-04-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-02 15:15:00 | 2034.98 | 2016.36 | 2013.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-03 09:15:00 | 2041.03 | 2021.29 | 2016.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-03 12:15:00 | 2015.00 | 2020.53 | 2017.29 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-03 12:15:00 | 2015.00 | 2020.53 | 2017.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-03 12:15:00 | 2015.00 | 2020.53 | 2017.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-03 13:00:00 | 2015.00 | 2020.53 | 2017.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-03 13:15:00 | 2014.28 | 2019.28 | 2017.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-03 14:00:00 | 2014.28 | 2019.28 | 2017.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-03 14:15:00 | 2019.48 | 2019.32 | 2017.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-03 14:30:00 | 2018.18 | 2019.32 | 2017.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 142 — SELL (started 2025-04-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-04 09:15:00 | 1986.48 | 2014.22 | 2015.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-04 10:15:00 | 1976.35 | 2006.65 | 2011.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-08 09:15:00 | 1898.68 | 1885.97 | 1923.41 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-08 09:15:00 | 1898.68 | 1885.97 | 1923.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 09:15:00 | 1898.68 | 1885.97 | 1923.41 | EMA400 retest candle locked (from downside) |

### Cycle 143 — BUY (started 2025-04-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-11 10:15:00 | 1951.85 | 1920.71 | 1918.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-11 11:15:00 | 1987.73 | 1934.11 | 1925.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-17 09:15:00 | 2036.40 | 2051.76 | 2030.03 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-17 10:00:00 | 2036.40 | 2051.76 | 2030.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-25 09:15:00 | 2232.90 | 2270.48 | 2256.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-25 10:00:00 | 2232.90 | 2270.48 | 2256.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-25 10:15:00 | 2214.00 | 2259.18 | 2252.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-25 11:00:00 | 2214.00 | 2259.18 | 2252.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 144 — SELL (started 2025-04-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-25 12:15:00 | 2196.85 | 2239.17 | 2244.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-28 09:15:00 | 2191.00 | 2211.58 | 2227.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-28 12:15:00 | 2206.25 | 2204.76 | 2220.11 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-28 12:15:00 | 2206.25 | 2204.76 | 2220.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-28 12:15:00 | 2206.25 | 2204.76 | 2220.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-28 12:30:00 | 2208.60 | 2204.76 | 2220.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-28 14:15:00 | 2216.00 | 2207.12 | 2218.53 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-29 11:45:00 | 2197.50 | 2212.31 | 2217.96 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-30 11:00:00 | 2200.90 | 2208.69 | 2213.45 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-30 13:00:00 | 2196.45 | 2205.85 | 2211.31 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-30 13:45:00 | 2200.45 | 2203.11 | 2209.57 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-02 09:15:00 | 2244.45 | 2206.30 | 2208.94 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-05-02 09:15:00 | 2244.45 | 2206.30 | 2208.94 | SL hit (close>static) qty=1.00 sl=2219.45 alert=retest2 |

### Cycle 145 — BUY (started 2025-05-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-02 11:15:00 | 2214.15 | 2211.38 | 2211.01 | EMA200 above EMA400 |

### Cycle 146 — SELL (started 2025-05-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-02 12:15:00 | 2200.00 | 2209.11 | 2210.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-02 14:15:00 | 2193.45 | 2204.77 | 2207.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-05 10:15:00 | 2204.50 | 2201.90 | 2205.48 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-05 10:45:00 | 2197.85 | 2201.90 | 2205.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 11:15:00 | 2225.05 | 2206.53 | 2207.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-05 12:00:00 | 2225.05 | 2206.53 | 2207.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 147 — BUY (started 2025-05-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-05 12:15:00 | 2232.50 | 2211.73 | 2209.56 | EMA200 above EMA400 |

### Cycle 148 — SELL (started 2025-05-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-06 10:15:00 | 2173.50 | 2208.13 | 2209.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-06 14:15:00 | 2169.35 | 2189.39 | 2199.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-07 09:15:00 | 2187.45 | 2185.82 | 2195.84 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-07 09:15:00 | 2187.45 | 2185.82 | 2195.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 09:15:00 | 2187.45 | 2185.82 | 2195.84 | EMA400 retest candle locked (from downside) |

### Cycle 149 — BUY (started 2025-05-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-07 14:15:00 | 2214.05 | 2203.15 | 2201.79 | EMA200 above EMA400 |

### Cycle 150 — SELL (started 2025-05-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-08 09:15:00 | 2187.95 | 2200.98 | 2201.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-08 13:15:00 | 2156.55 | 2187.37 | 2194.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-09 14:15:00 | 2136.05 | 2132.53 | 2155.15 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-09 14:45:00 | 2132.00 | 2132.53 | 2155.15 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-12 09:15:00 | 2223.75 | 2151.54 | 2159.91 | EMA400 retest candle locked (from downside) |

### Cycle 151 — BUY (started 2025-05-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 10:15:00 | 2234.65 | 2168.16 | 2166.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 12:15:00 | 2255.30 | 2196.90 | 2180.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-13 13:15:00 | 2249.35 | 2253.27 | 2226.38 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-13 14:00:00 | 2249.35 | 2253.27 | 2226.38 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 14:15:00 | 2380.95 | 2396.17 | 2381.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 14:30:00 | 2374.40 | 2396.17 | 2381.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 15:15:00 | 2382.15 | 2393.36 | 2381.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-21 09:15:00 | 2356.05 | 2393.36 | 2381.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 09:15:00 | 2390.40 | 2392.77 | 2382.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-21 09:30:00 | 2372.10 | 2392.77 | 2382.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 10:15:00 | 2381.15 | 2390.45 | 2382.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-21 11:00:00 | 2381.15 | 2390.45 | 2382.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 11:15:00 | 2383.60 | 2389.08 | 2382.37 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-21 12:30:00 | 2403.00 | 2390.97 | 2383.84 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-22 14:45:00 | 2395.90 | 2400.43 | 2395.99 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-22 15:15:00 | 2399.05 | 2400.43 | 2395.99 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-27 10:30:00 | 2398.45 | 2407.16 | 2405.46 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 12:15:00 | 2402.90 | 2405.55 | 2404.98 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-05-27 13:15:00 | 2399.20 | 2404.28 | 2404.45 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 152 — SELL (started 2025-05-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-27 13:15:00 | 2399.20 | 2404.28 | 2404.45 | EMA200 below EMA400 |

### Cycle 153 — BUY (started 2025-05-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-28 09:15:00 | 2420.05 | 2407.21 | 2405.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-28 10:15:00 | 2437.30 | 2413.23 | 2408.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-28 14:15:00 | 2416.65 | 2419.68 | 2413.84 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-28 15:00:00 | 2416.65 | 2419.68 | 2413.84 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 15:15:00 | 2414.75 | 2418.70 | 2413.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-29 09:15:00 | 2405.55 | 2418.70 | 2413.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 09:15:00 | 2398.00 | 2414.56 | 2412.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-29 10:00:00 | 2398.00 | 2414.56 | 2412.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 10:15:00 | 2405.50 | 2412.74 | 2411.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-29 10:30:00 | 2400.20 | 2412.74 | 2411.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 12:15:00 | 2414.00 | 2413.20 | 2412.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-29 12:45:00 | 2413.05 | 2413.20 | 2412.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 13:15:00 | 2407.70 | 2412.10 | 2411.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-29 14:00:00 | 2407.70 | 2412.10 | 2411.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 14:15:00 | 2423.90 | 2414.46 | 2412.91 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-29 15:15:00 | 2428.05 | 2414.46 | 2412.91 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-30 10:15:00 | 2398.05 | 2411.90 | 2412.30 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 154 — SELL (started 2025-05-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-30 10:15:00 | 2398.05 | 2411.90 | 2412.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-30 12:15:00 | 2397.90 | 2408.24 | 2410.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-02 09:15:00 | 2401.20 | 2400.50 | 2405.53 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-02 10:00:00 | 2401.20 | 2400.50 | 2405.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 13:15:00 | 2406.35 | 2398.01 | 2402.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-02 14:00:00 | 2406.35 | 2398.01 | 2402.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 14:15:00 | 2412.50 | 2400.91 | 2403.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-02 15:00:00 | 2412.50 | 2400.91 | 2403.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 15:15:00 | 2410.00 | 2402.73 | 2403.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-03 09:15:00 | 2440.60 | 2402.73 | 2403.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 155 — BUY (started 2025-06-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-03 09:15:00 | 2460.10 | 2414.20 | 2408.96 | EMA200 above EMA400 |

### Cycle 156 — SELL (started 2025-06-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-03 14:15:00 | 2394.00 | 2406.36 | 2407.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-04 09:15:00 | 2380.95 | 2399.48 | 2403.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-05 09:15:00 | 2405.30 | 2383.17 | 2390.43 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-05 09:15:00 | 2405.30 | 2383.17 | 2390.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 09:15:00 | 2405.30 | 2383.17 | 2390.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-05 10:00:00 | 2405.30 | 2383.17 | 2390.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 10:15:00 | 2404.45 | 2387.42 | 2391.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-05 11:15:00 | 2420.30 | 2387.42 | 2391.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 157 — BUY (started 2025-06-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-05 12:15:00 | 2420.30 | 2398.40 | 2396.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-05 13:15:00 | 2428.00 | 2404.32 | 2399.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-10 11:15:00 | 2559.05 | 2574.86 | 2539.77 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-10 11:45:00 | 2566.20 | 2574.86 | 2539.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 09:15:00 | 2566.20 | 2567.28 | 2548.82 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-11 10:30:00 | 2583.20 | 2573.04 | 2553.11 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-12 10:15:00 | 2537.90 | 2563.29 | 2558.39 | SL hit (close<static) qty=1.00 sl=2544.60 alert=retest2 |

### Cycle 158 — SELL (started 2025-06-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-12 12:15:00 | 2546.35 | 2555.30 | 2555.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-12 13:15:00 | 2521.50 | 2548.54 | 2552.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-16 11:15:00 | 2509.00 | 2490.71 | 2508.55 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-16 11:15:00 | 2509.00 | 2490.71 | 2508.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 11:15:00 | 2509.00 | 2490.71 | 2508.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 12:00:00 | 2509.00 | 2490.71 | 2508.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 12:15:00 | 2494.00 | 2491.37 | 2507.23 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-16 14:15:00 | 2481.65 | 2490.75 | 2505.50 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-16 15:15:00 | 2481.25 | 2490.51 | 2504.05 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-17 11:15:00 | 2481.55 | 2489.37 | 2500.01 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-17 14:30:00 | 2484.35 | 2485.80 | 2494.43 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 15:15:00 | 2487.85 | 2486.21 | 2493.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-18 09:15:00 | 2487.65 | 2486.21 | 2493.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 09:15:00 | 2483.55 | 2485.68 | 2492.90 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-18 10:45:00 | 2466.50 | 2483.80 | 2491.39 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-18 11:15:00 | 2468.75 | 2483.80 | 2491.39 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-19 10:15:00 | 2471.50 | 2475.45 | 2482.80 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-20 11:15:00 | 2472.60 | 2457.15 | 2466.45 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 11:15:00 | 2477.00 | 2461.12 | 2467.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 11:30:00 | 2471.15 | 2461.12 | 2467.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 13:15:00 | 2454.35 | 2460.75 | 2466.21 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-06-23 10:15:00 | 2489.75 | 2472.40 | 2470.58 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 159 — BUY (started 2025-06-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-23 10:15:00 | 2489.75 | 2472.40 | 2470.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-24 09:15:00 | 2529.20 | 2483.34 | 2476.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-26 09:15:00 | 2530.00 | 2538.78 | 2524.55 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-26 10:00:00 | 2530.00 | 2538.78 | 2524.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 10:15:00 | 2515.75 | 2534.17 | 2523.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-26 11:00:00 | 2515.75 | 2534.17 | 2523.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 11:15:00 | 2519.20 | 2531.18 | 2523.34 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-26 12:15:00 | 2529.50 | 2531.18 | 2523.34 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-26 13:30:00 | 2526.45 | 2531.00 | 2524.51 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-02 10:15:00 | 2548.25 | 2567.59 | 2568.84 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 160 — SELL (started 2025-07-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-02 10:15:00 | 2548.25 | 2567.59 | 2568.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-02 12:15:00 | 2534.25 | 2558.27 | 2564.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-03 09:15:00 | 2556.25 | 2548.42 | 2556.63 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-03 09:15:00 | 2556.25 | 2548.42 | 2556.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 09:15:00 | 2556.25 | 2548.42 | 2556.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-03 10:00:00 | 2556.25 | 2548.42 | 2556.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 10:15:00 | 2549.50 | 2548.63 | 2555.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-03 10:45:00 | 2551.75 | 2548.63 | 2555.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 09:15:00 | 2525.00 | 2518.24 | 2529.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-07 10:00:00 | 2525.00 | 2518.24 | 2529.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 10:15:00 | 2537.00 | 2521.99 | 2529.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-07 11:00:00 | 2537.00 | 2521.99 | 2529.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 11:15:00 | 2514.25 | 2520.44 | 2528.43 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-07 12:15:00 | 2512.75 | 2520.44 | 2528.43 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-07 13:15:00 | 2508.00 | 2519.35 | 2527.21 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-08 09:15:00 | 2559.75 | 2519.82 | 2524.03 | SL hit (close>static) qty=1.00 sl=2537.50 alert=retest2 |

### Cycle 161 — BUY (started 2025-07-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-08 11:15:00 | 2556.00 | 2531.36 | 2528.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-09 13:15:00 | 2579.50 | 2563.57 | 2550.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-09 15:15:00 | 2565.00 | 2565.52 | 2554.00 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-10 09:15:00 | 2600.00 | 2565.52 | 2554.00 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 10:15:00 | 2572.75 | 2590.97 | 2579.24 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-07-11 10:15:00 | 2572.75 | 2590.97 | 2579.24 | SL hit (close<ema400) qty=1.00 sl=2579.24 alert=retest1 |

### Cycle 162 — SELL (started 2025-07-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-11 15:15:00 | 2545.50 | 2570.27 | 2572.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-14 11:15:00 | 2541.75 | 2560.18 | 2567.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-14 15:15:00 | 2550.50 | 2549.96 | 2559.17 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-15 09:15:00 | 2573.00 | 2549.96 | 2559.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 09:15:00 | 2589.25 | 2557.82 | 2561.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-15 10:00:00 | 2589.25 | 2557.82 | 2561.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 163 — BUY (started 2025-07-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-15 10:15:00 | 2611.75 | 2568.61 | 2566.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-15 14:15:00 | 2639.75 | 2599.54 | 2583.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-22 09:15:00 | 2797.50 | 2800.38 | 2769.60 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-24 09:15:00 | 2777.25 | 2792.49 | 2787.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 09:15:00 | 2777.25 | 2792.49 | 2787.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-24 10:00:00 | 2777.25 | 2792.49 | 2787.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 10:15:00 | 2780.00 | 2790.00 | 2786.51 | EMA400 retest candle locked (from upside) |

### Cycle 164 — SELL (started 2025-07-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-24 14:15:00 | 2781.25 | 2784.67 | 2784.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-25 10:15:00 | 2764.75 | 2778.86 | 2781.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-25 14:15:00 | 2775.50 | 2771.90 | 2776.91 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-25 14:15:00 | 2775.50 | 2771.90 | 2776.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 14:15:00 | 2775.50 | 2771.90 | 2776.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-25 15:00:00 | 2775.50 | 2771.90 | 2776.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 15:15:00 | 2763.25 | 2770.17 | 2775.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-28 09:15:00 | 2799.50 | 2770.17 | 2775.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 165 — BUY (started 2025-07-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-28 09:15:00 | 2820.75 | 2780.29 | 2779.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-30 12:15:00 | 2829.00 | 2813.66 | 2803.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-31 09:15:00 | 2808.25 | 2820.35 | 2811.04 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-31 09:15:00 | 2808.25 | 2820.35 | 2811.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 09:15:00 | 2808.25 | 2820.35 | 2811.04 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-31 13:00:00 | 2830.00 | 2816.99 | 2811.20 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-01 10:30:00 | 2832.25 | 2821.50 | 2816.19 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-01 11:00:00 | 2830.50 | 2821.50 | 2816.19 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-01 11:45:00 | 2829.75 | 2824.45 | 2818.02 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 13:15:00 | 2815.75 | 2824.36 | 2819.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-01 14:00:00 | 2815.75 | 2824.36 | 2819.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 14:15:00 | 2804.75 | 2820.44 | 2817.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-01 15:00:00 | 2804.75 | 2820.44 | 2817.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 15:15:00 | 2805.50 | 2817.45 | 2816.75 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-04 09:15:00 | 2833.00 | 2817.45 | 2816.75 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-07 10:15:00 | 2809.75 | 2830.38 | 2831.39 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 166 — SELL (started 2025-08-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-07 10:15:00 | 2809.75 | 2830.38 | 2831.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-07 12:15:00 | 2802.75 | 2821.55 | 2827.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-07 14:15:00 | 2832.00 | 2822.43 | 2826.38 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-07 14:15:00 | 2832.00 | 2822.43 | 2826.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 14:15:00 | 2832.00 | 2822.43 | 2826.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-07 14:45:00 | 2831.00 | 2822.43 | 2826.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 15:15:00 | 2828.00 | 2823.55 | 2826.53 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-08 09:15:00 | 2806.25 | 2823.55 | 2826.53 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-18 09:15:00 | 2836.50 | 2756.48 | 2751.89 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 167 — BUY (started 2025-08-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-18 09:15:00 | 2836.50 | 2756.48 | 2751.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-18 10:15:00 | 2847.50 | 2774.69 | 2760.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-20 09:15:00 | 2838.25 | 2846.99 | 2825.38 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-20 09:30:00 | 2843.00 | 2846.99 | 2825.38 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 10:15:00 | 2834.75 | 2844.54 | 2826.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-20 11:00:00 | 2834.75 | 2844.54 | 2826.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 11:15:00 | 2840.00 | 2843.63 | 2827.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-20 11:30:00 | 2828.00 | 2843.63 | 2827.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 09:15:00 | 2924.25 | 2937.98 | 2918.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-26 09:45:00 | 2922.50 | 2937.98 | 2918.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 11:15:00 | 2903.50 | 2928.45 | 2917.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-26 12:00:00 | 2903.50 | 2928.45 | 2917.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 12:15:00 | 2900.00 | 2922.76 | 2915.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-26 12:30:00 | 2900.00 | 2922.76 | 2915.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 168 — SELL (started 2025-08-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-26 14:15:00 | 2877.75 | 2909.20 | 2910.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-28 09:15:00 | 2847.25 | 2891.50 | 2901.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-01 12:15:00 | 2786.75 | 2771.02 | 2798.86 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-01 13:00:00 | 2786.75 | 2771.02 | 2798.86 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 09:15:00 | 2782.50 | 2779.04 | 2794.21 | EMA400 retest candle locked (from downside) |

### Cycle 169 — BUY (started 2025-09-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-03 13:15:00 | 2798.00 | 2790.91 | 2790.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-04 09:15:00 | 2808.00 | 2795.64 | 2792.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-04 10:15:00 | 2784.00 | 2793.31 | 2791.79 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-04 10:15:00 | 2784.00 | 2793.31 | 2791.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 10:15:00 | 2784.00 | 2793.31 | 2791.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 11:00:00 | 2784.00 | 2793.31 | 2791.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 11:15:00 | 2791.00 | 2792.85 | 2791.72 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-04 12:30:00 | 2794.75 | 2792.63 | 2791.72 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-04 14:15:00 | 2786.50 | 2790.46 | 2790.83 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 170 — SELL (started 2025-09-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-04 14:15:00 | 2786.50 | 2790.46 | 2790.83 | EMA200 below EMA400 |

### Cycle 171 — BUY (started 2025-09-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-05 10:15:00 | 2799.00 | 2792.32 | 2791.50 | EMA200 above EMA400 |

### Cycle 172 — SELL (started 2025-09-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-05 11:15:00 | 2783.00 | 2790.45 | 2790.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-05 12:15:00 | 2776.25 | 2787.61 | 2789.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-05 13:15:00 | 2795.00 | 2789.09 | 2789.92 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-05 13:15:00 | 2795.00 | 2789.09 | 2789.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 13:15:00 | 2795.00 | 2789.09 | 2789.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-05 14:00:00 | 2795.00 | 2789.09 | 2789.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 14:15:00 | 2786.50 | 2788.57 | 2789.61 | EMA400 retest candle locked (from downside) |

### Cycle 173 — BUY (started 2025-09-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-05 15:15:00 | 2797.75 | 2790.41 | 2790.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-08 11:15:00 | 2828.25 | 2799.99 | 2794.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-09 10:15:00 | 2829.75 | 2835.03 | 2818.89 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-09 10:15:00 | 2829.75 | 2835.03 | 2818.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 10:15:00 | 2829.75 | 2835.03 | 2818.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-09 10:30:00 | 2826.75 | 2835.03 | 2818.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 09:15:00 | 2889.00 | 2886.47 | 2862.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-11 09:30:00 | 2877.50 | 2886.47 | 2862.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 09:15:00 | 2862.50 | 2894.14 | 2889.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-15 10:00:00 | 2862.50 | 2894.14 | 2889.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 10:15:00 | 2864.75 | 2888.26 | 2887.14 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-15 11:45:00 | 2878.50 | 2887.21 | 2886.77 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-15 13:15:00 | 2871.50 | 2883.59 | 2885.17 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 174 — SELL (started 2025-09-15 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-15 13:15:00 | 2871.50 | 2883.59 | 2885.17 | EMA200 below EMA400 |

### Cycle 175 — BUY (started 2025-09-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-16 10:15:00 | 2906.00 | 2888.86 | 2886.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-18 09:15:00 | 2921.50 | 2904.59 | 2897.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-19 14:15:00 | 2923.00 | 2928.30 | 2918.45 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-19 15:00:00 | 2923.00 | 2928.30 | 2918.45 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 09:15:00 | 2942.75 | 2932.66 | 2922.24 | EMA400 retest candle locked (from upside) |

### Cycle 176 — SELL (started 2025-09-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-23 13:15:00 | 2920.00 | 2922.62 | 2922.74 | EMA200 below EMA400 |

### Cycle 177 — BUY (started 2025-09-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-23 14:15:00 | 2946.25 | 2927.35 | 2924.88 | EMA200 above EMA400 |

### Cycle 178 — SELL (started 2025-09-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-24 10:15:00 | 2877.00 | 2917.31 | 2921.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-26 09:15:00 | 2864.00 | 2883.66 | 2895.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-01 09:15:00 | 2776.00 | 2772.65 | 2800.16 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-03 09:15:00 | 2808.75 | 2784.44 | 2793.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 09:15:00 | 2808.75 | 2784.44 | 2793.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-03 10:00:00 | 2808.75 | 2784.44 | 2793.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 10:15:00 | 2792.50 | 2786.05 | 2792.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-03 10:30:00 | 2809.50 | 2786.05 | 2792.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 11:15:00 | 2796.00 | 2788.04 | 2793.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-03 12:00:00 | 2796.00 | 2788.04 | 2793.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 12:15:00 | 2787.75 | 2787.98 | 2792.76 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-03 13:15:00 | 2782.00 | 2787.98 | 2792.76 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-03 14:15:00 | 2797.25 | 2788.92 | 2792.30 | SL hit (close>static) qty=1.00 sl=2796.00 alert=retest2 |

### Cycle 179 — BUY (started 2025-10-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-06 10:15:00 | 2808.75 | 2796.80 | 2795.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-06 13:15:00 | 2827.75 | 2806.75 | 2800.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-07 09:15:00 | 2807.00 | 2814.04 | 2806.07 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-07 09:15:00 | 2807.00 | 2814.04 | 2806.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 09:15:00 | 2807.00 | 2814.04 | 2806.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-07 09:30:00 | 2814.25 | 2814.04 | 2806.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 10:15:00 | 2792.25 | 2809.68 | 2804.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-07 11:00:00 | 2792.25 | 2809.68 | 2804.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 180 — SELL (started 2025-10-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-07 11:15:00 | 2766.50 | 2801.05 | 2801.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-08 09:15:00 | 2754.25 | 2782.12 | 2791.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-09 10:15:00 | 2750.00 | 2747.35 | 2764.28 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-09 11:00:00 | 2750.00 | 2747.35 | 2764.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 11:15:00 | 2748.75 | 2747.63 | 2762.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-09 11:30:00 | 2758.75 | 2747.63 | 2762.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 09:15:00 | 2762.50 | 2746.14 | 2755.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-10 09:30:00 | 2762.50 | 2746.14 | 2755.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 10:15:00 | 2762.25 | 2749.36 | 2756.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-10 10:30:00 | 2762.00 | 2749.36 | 2756.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 12:15:00 | 2769.25 | 2756.08 | 2758.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-10 13:00:00 | 2769.25 | 2756.08 | 2758.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 14:15:00 | 2761.75 | 2758.04 | 2758.88 | EMA400 retest candle locked (from downside) |

### Cycle 181 — BUY (started 2025-10-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-13 09:15:00 | 2813.25 | 2769.48 | 2763.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-13 14:15:00 | 2836.00 | 2808.29 | 2787.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-14 12:15:00 | 2813.00 | 2821.23 | 2803.06 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-14 13:00:00 | 2813.00 | 2821.23 | 2803.06 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 13:15:00 | 2809.75 | 2818.93 | 2803.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-14 14:00:00 | 2809.75 | 2818.93 | 2803.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 14:15:00 | 2796.75 | 2814.49 | 2803.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-14 15:00:00 | 2796.75 | 2814.49 | 2803.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 15:15:00 | 2798.75 | 2811.35 | 2802.65 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-15 09:15:00 | 2835.75 | 2811.35 | 2802.65 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-20 10:15:00 | 2844.50 | 2866.74 | 2867.33 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 182 — SELL (started 2025-10-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-20 10:15:00 | 2844.50 | 2866.74 | 2867.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-23 13:15:00 | 2822.50 | 2846.30 | 2855.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-27 09:15:00 | 2790.00 | 2787.92 | 2811.15 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-27 09:30:00 | 2791.00 | 2787.92 | 2811.15 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 11:15:00 | 2803.75 | 2787.43 | 2797.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-28 11:45:00 | 2810.75 | 2787.43 | 2797.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 12:15:00 | 2816.75 | 2793.29 | 2798.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-28 13:00:00 | 2816.75 | 2793.29 | 2798.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 183 — BUY (started 2025-10-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-28 15:15:00 | 2832.00 | 2807.25 | 2804.45 | EMA200 above EMA400 |

### Cycle 184 — SELL (started 2025-10-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-29 09:15:00 | 2710.50 | 2787.90 | 2795.90 | EMA200 below EMA400 |

### Cycle 185 — BUY (started 2025-11-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-04 10:15:00 | 2724.50 | 2707.26 | 2707.20 | EMA200 above EMA400 |

### Cycle 186 — SELL (started 2025-11-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-04 15:15:00 | 2698.50 | 2707.31 | 2707.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-06 09:15:00 | 2689.75 | 2703.80 | 2706.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-06 12:15:00 | 2702.25 | 2698.55 | 2702.63 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-06 12:15:00 | 2702.25 | 2698.55 | 2702.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 12:15:00 | 2702.25 | 2698.55 | 2702.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-06 13:00:00 | 2702.25 | 2698.55 | 2702.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 13:15:00 | 2699.50 | 2698.74 | 2702.35 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-06 15:15:00 | 2690.50 | 2698.04 | 2701.70 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-07 10:15:00 | 2686.75 | 2696.83 | 2700.44 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-07 12:15:00 | 2712.75 | 2697.77 | 2699.71 | SL hit (close>static) qty=1.00 sl=2704.50 alert=retest2 |

### Cycle 187 — BUY (started 2025-11-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-07 14:15:00 | 2715.75 | 2703.48 | 2702.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-10 09:15:00 | 2735.75 | 2712.18 | 2706.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-11 09:15:00 | 2717.25 | 2732.01 | 2722.04 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-11 09:15:00 | 2717.25 | 2732.01 | 2722.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 09:15:00 | 2717.25 | 2732.01 | 2722.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-11 10:00:00 | 2717.25 | 2732.01 | 2722.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 10:15:00 | 2718.50 | 2729.30 | 2721.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-11 10:30:00 | 2728.00 | 2729.30 | 2721.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 11:15:00 | 2720.75 | 2727.59 | 2721.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-11 11:30:00 | 2717.75 | 2727.59 | 2721.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 12:15:00 | 2733.75 | 2728.83 | 2722.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-11 12:30:00 | 2721.50 | 2728.83 | 2722.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 09:15:00 | 2747.00 | 2737.05 | 2728.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-12 09:30:00 | 2733.50 | 2737.05 | 2728.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 09:15:00 | 2769.25 | 2746.94 | 2738.19 | EMA400 retest candle locked (from upside) |

### Cycle 188 — SELL (started 2025-11-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-14 10:15:00 | 2710.50 | 2737.70 | 2740.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-14 13:15:00 | 2692.25 | 2721.50 | 2731.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-17 09:15:00 | 2722.25 | 2717.63 | 2726.89 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-17 09:15:00 | 2722.25 | 2717.63 | 2726.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 09:15:00 | 2722.25 | 2717.63 | 2726.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-17 09:45:00 | 2726.00 | 2717.63 | 2726.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 10:15:00 | 2718.00 | 2717.71 | 2726.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-17 10:45:00 | 2722.00 | 2717.71 | 2726.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 13:15:00 | 2725.00 | 2718.18 | 2724.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-17 13:30:00 | 2721.50 | 2718.18 | 2724.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 14:15:00 | 2718.50 | 2718.24 | 2723.60 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-18 10:00:00 | 2709.00 | 2717.48 | 2722.39 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-24 10:15:00 | 2712.00 | 2699.17 | 2698.88 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 189 — BUY (started 2025-11-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-24 10:15:00 | 2712.00 | 2699.17 | 2698.88 | EMA200 above EMA400 |

### Cycle 190 — SELL (started 2025-11-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-24 11:15:00 | 2691.75 | 2697.69 | 2698.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-24 14:15:00 | 2687.25 | 2694.95 | 2696.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-26 10:15:00 | 2682.00 | 2674.03 | 2680.82 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-26 10:15:00 | 2682.00 | 2674.03 | 2680.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 10:15:00 | 2682.00 | 2674.03 | 2680.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-26 11:00:00 | 2682.00 | 2674.03 | 2680.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 11:15:00 | 2672.00 | 2673.63 | 2680.01 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-26 12:15:00 | 2665.00 | 2673.63 | 2680.01 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-26 14:15:00 | 2670.00 | 2674.78 | 2679.53 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-27 10:30:00 | 2668.00 | 2673.90 | 2677.76 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-27 13:00:00 | 2668.50 | 2671.80 | 2676.07 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 13:15:00 | 2677.00 | 2672.84 | 2676.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-27 14:00:00 | 2677.00 | 2672.84 | 2676.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 14:15:00 | 2681.00 | 2674.47 | 2676.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-27 15:00:00 | 2681.00 | 2674.47 | 2676.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 15:15:00 | 2677.50 | 2675.08 | 2676.67 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-28 09:15:00 | 2675.00 | 2675.08 | 2676.67 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-28 10:15:00 | 2667.50 | 2675.66 | 2676.79 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-28 11:15:00 | 2690.00 | 2678.98 | 2678.14 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 191 — BUY (started 2025-11-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-28 11:15:00 | 2690.00 | 2678.98 | 2678.14 | EMA200 above EMA400 |

### Cycle 192 — SELL (started 2025-11-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-28 15:15:00 | 2674.00 | 2678.06 | 2678.19 | EMA200 below EMA400 |

### Cycle 193 — BUY (started 2025-12-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-01 09:15:00 | 2683.40 | 2679.13 | 2678.66 | EMA200 above EMA400 |

### Cycle 194 — SELL (started 2025-12-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-01 10:15:00 | 2665.60 | 2676.42 | 2677.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-01 11:15:00 | 2659.60 | 2673.06 | 2675.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-03 12:15:00 | 2608.50 | 2605.86 | 2624.53 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-03 13:00:00 | 2608.50 | 2605.86 | 2624.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 09:15:00 | 2585.30 | 2583.22 | 2592.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-08 09:30:00 | 2588.10 | 2583.22 | 2592.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 10:15:00 | 2582.60 | 2583.10 | 2591.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-08 10:45:00 | 2595.30 | 2583.10 | 2591.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 09:15:00 | 2619.80 | 2567.67 | 2570.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-10 09:45:00 | 2615.00 | 2567.67 | 2570.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 195 — BUY (started 2025-12-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-10 10:15:00 | 2605.00 | 2575.14 | 2573.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-11 09:15:00 | 2631.10 | 2605.08 | 2591.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-15 09:15:00 | 2638.90 | 2659.98 | 2643.95 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-15 09:15:00 | 2638.90 | 2659.98 | 2643.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 09:15:00 | 2638.90 | 2659.98 | 2643.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-15 10:00:00 | 2638.90 | 2659.98 | 2643.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 10:15:00 | 2619.00 | 2651.78 | 2641.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-15 11:00:00 | 2619.00 | 2651.78 | 2641.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 11:15:00 | 2611.40 | 2643.71 | 2638.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-15 12:00:00 | 2611.40 | 2643.71 | 2638.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 196 — SELL (started 2025-12-15 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-15 13:15:00 | 2610.80 | 2632.40 | 2634.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-16 09:15:00 | 2584.60 | 2615.66 | 2625.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-18 09:15:00 | 2664.80 | 2581.04 | 2588.59 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-18 09:15:00 | 2664.80 | 2581.04 | 2588.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 09:15:00 | 2664.80 | 2581.04 | 2588.59 | EMA400 retest candle locked (from downside) |

### Cycle 197 — BUY (started 2025-12-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-18 10:15:00 | 2662.00 | 2597.23 | 2595.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-18 11:15:00 | 2682.00 | 2614.19 | 2603.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-19 10:15:00 | 2634.30 | 2669.84 | 2643.45 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-19 10:15:00 | 2634.30 | 2669.84 | 2643.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 10:15:00 | 2634.30 | 2669.84 | 2643.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-19 11:00:00 | 2634.30 | 2669.84 | 2643.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 11:15:00 | 2649.00 | 2665.67 | 2643.96 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-19 13:15:00 | 2655.00 | 2662.01 | 2644.27 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-24 14:45:00 | 2654.00 | 2678.80 | 2678.34 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-24 15:15:00 | 2660.10 | 2675.06 | 2676.68 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 198 — SELL (started 2025-12-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-24 15:15:00 | 2660.10 | 2675.06 | 2676.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-26 09:15:00 | 2653.60 | 2670.77 | 2674.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-29 13:15:00 | 2649.90 | 2643.37 | 2653.76 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-29 14:00:00 | 2649.90 | 2643.37 | 2653.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 14:15:00 | 2647.10 | 2644.11 | 2653.15 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-29 15:15:00 | 2640.00 | 2644.11 | 2653.15 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-31 09:15:00 | 2672.80 | 2643.39 | 2646.07 | SL hit (close>static) qty=1.00 sl=2656.80 alert=retest2 |

### Cycle 199 — BUY (started 2025-12-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-31 10:15:00 | 2666.90 | 2648.09 | 2647.96 | EMA200 above EMA400 |

### Cycle 200 — SELL (started 2026-01-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-01 12:15:00 | 2636.60 | 2652.02 | 2653.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-02 09:15:00 | 2634.80 | 2647.87 | 2650.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-02 11:15:00 | 2655.00 | 2648.68 | 2650.65 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-02 11:15:00 | 2655.00 | 2648.68 | 2650.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 11:15:00 | 2655.00 | 2648.68 | 2650.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-02 11:30:00 | 2653.00 | 2648.68 | 2650.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 12:15:00 | 2658.00 | 2650.54 | 2651.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-02 12:30:00 | 2654.50 | 2650.54 | 2651.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 201 — BUY (started 2026-01-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-02 13:15:00 | 2658.20 | 2652.07 | 2651.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-05 09:15:00 | 2672.00 | 2657.80 | 2654.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-05 13:15:00 | 2650.10 | 2661.14 | 2657.71 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-05 13:15:00 | 2650.10 | 2661.14 | 2657.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 13:15:00 | 2650.10 | 2661.14 | 2657.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-05 14:00:00 | 2650.10 | 2661.14 | 2657.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 14:15:00 | 2661.70 | 2661.25 | 2658.07 | EMA400 retest candle locked (from upside) |

### Cycle 202 — SELL (started 2026-01-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-06 10:15:00 | 2632.10 | 2651.88 | 2654.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-06 11:15:00 | 2624.40 | 2646.38 | 2651.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-07 10:15:00 | 2631.90 | 2630.77 | 2640.12 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-07 11:00:00 | 2631.90 | 2630.77 | 2640.12 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 11:15:00 | 2641.60 | 2632.93 | 2640.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-07 11:45:00 | 2639.90 | 2632.93 | 2640.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 12:15:00 | 2642.00 | 2634.75 | 2640.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-07 12:45:00 | 2642.90 | 2634.75 | 2640.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 13:15:00 | 2630.40 | 2633.88 | 2639.50 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-07 14:30:00 | 2630.00 | 2632.46 | 2638.35 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-12 09:15:00 | 2498.50 | 2538.77 | 2570.12 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-13 15:15:00 | 2490.00 | 2483.16 | 2506.02 | SL hit (close>ema200) qty=0.50 sl=2483.16 alert=retest2 |

### Cycle 203 — BUY (started 2026-01-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-14 14:15:00 | 2555.20 | 2522.51 | 2518.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-16 09:15:00 | 2684.70 | 2559.83 | 2536.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-16 14:15:00 | 2596.80 | 2597.15 | 2567.54 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-16 14:30:00 | 2596.90 | 2597.15 | 2567.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 09:15:00 | 2581.40 | 2593.34 | 2570.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-19 09:30:00 | 2570.70 | 2593.34 | 2570.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 11:15:00 | 2587.20 | 2590.46 | 2573.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-19 11:45:00 | 2576.90 | 2590.46 | 2573.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 13:15:00 | 2575.50 | 2585.68 | 2574.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-19 13:30:00 | 2578.00 | 2585.68 | 2574.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 14:15:00 | 2580.20 | 2584.58 | 2574.63 | EMA400 retest candle locked (from upside) |

### Cycle 204 — SELL (started 2026-01-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-20 10:15:00 | 2538.20 | 2568.95 | 2569.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-20 14:15:00 | 2522.00 | 2546.84 | 2557.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-22 09:15:00 | 2510.10 | 2497.40 | 2518.75 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-22 09:45:00 | 2507.90 | 2497.40 | 2518.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 09:15:00 | 2487.00 | 2495.52 | 2507.75 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-23 10:15:00 | 2485.00 | 2495.52 | 2507.75 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-23 11:30:00 | 2477.00 | 2487.70 | 2501.92 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-28 15:15:00 | 2477.30 | 2459.13 | 2459.04 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 205 — BUY (started 2026-01-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-28 15:15:00 | 2477.30 | 2459.13 | 2459.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-29 09:15:00 | 2515.90 | 2470.48 | 2464.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-30 12:15:00 | 2532.90 | 2533.08 | 2510.34 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-30 13:00:00 | 2532.90 | 2533.08 | 2510.34 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 14:15:00 | 2518.10 | 2527.94 | 2511.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-30 15:00:00 | 2518.10 | 2527.94 | 2511.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 15:15:00 | 2509.00 | 2524.16 | 2511.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 09:15:00 | 2485.50 | 2524.16 | 2511.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 09:15:00 | 2491.20 | 2517.56 | 2509.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 09:30:00 | 2498.70 | 2517.56 | 2509.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 10:15:00 | 2542.80 | 2522.61 | 2512.72 | EMA400 retest candle locked (from upside) |

### Cycle 206 — SELL (started 2026-02-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-01 14:15:00 | 2460.10 | 2499.38 | 2504.13 | EMA200 below EMA400 |

### Cycle 207 — BUY (started 2026-02-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-02 13:15:00 | 2528.80 | 2505.62 | 2503.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-02 14:15:00 | 2575.20 | 2519.53 | 2510.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-06 09:15:00 | 2720.30 | 2742.59 | 2708.95 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-06 10:00:00 | 2720.30 | 2742.59 | 2708.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 10:15:00 | 2712.60 | 2736.60 | 2709.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-06 10:45:00 | 2712.00 | 2736.60 | 2709.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 11:15:00 | 2709.70 | 2731.22 | 2709.32 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-09 09:15:00 | 2728.00 | 2721.47 | 2710.89 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-09 09:45:00 | 2727.50 | 2723.00 | 2712.54 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-18 14:15:00 | 2817.10 | 2834.21 | 2836.30 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 208 — SELL (started 2026-02-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-18 14:15:00 | 2817.10 | 2834.21 | 2836.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-19 09:15:00 | 2777.00 | 2818.74 | 2828.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-23 09:15:00 | 2758.80 | 2739.19 | 2761.70 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-23 09:15:00 | 2758.80 | 2739.19 | 2761.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 09:15:00 | 2758.80 | 2739.19 | 2761.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-23 09:45:00 | 2762.20 | 2739.19 | 2761.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 10:15:00 | 2726.80 | 2736.71 | 2758.53 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-23 12:00:00 | 2722.40 | 2733.85 | 2755.25 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-23 14:00:00 | 2717.50 | 2730.06 | 2749.78 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-24 09:15:00 | 2714.40 | 2730.21 | 2746.39 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-25 11:15:00 | 2760.00 | 2717.34 | 2724.03 | SL hit (close>static) qty=1.00 sl=2759.90 alert=retest2 |

### Cycle 209 — BUY (started 2026-02-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-25 15:15:00 | 2749.90 | 2728.36 | 2727.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-26 13:15:00 | 2753.50 | 2743.34 | 2735.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-27 09:15:00 | 2734.90 | 2744.87 | 2738.80 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-27 09:15:00 | 2734.90 | 2744.87 | 2738.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 09:15:00 | 2734.90 | 2744.87 | 2738.80 | EMA400 retest candle locked (from upside) |

### Cycle 210 — SELL (started 2026-02-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-27 12:15:00 | 2729.40 | 2734.93 | 2735.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-27 13:15:00 | 2720.90 | 2732.13 | 2733.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-05 15:15:00 | 2559.50 | 2558.72 | 2591.36 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-06 09:30:00 | 2538.70 | 2552.40 | 2585.51 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-09 09:15:00 | 2411.76 | 2509.52 | 2545.42 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 09:15:00 | 2472.50 | 2467.74 | 2500.96 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-03-10 09:15:00 | 2472.50 | 2467.74 | 2500.96 | SL hit (close>ema200) qty=0.50 sl=2467.74 alert=retest1 |

### Cycle 211 — BUY (started 2026-03-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 10:15:00 | 2447.90 | 2402.79 | 2397.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-18 12:15:00 | 2485.00 | 2426.38 | 2409.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 09:15:00 | 2414.80 | 2442.85 | 2425.10 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-19 09:15:00 | 2414.80 | 2442.85 | 2425.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 09:15:00 | 2414.80 | 2442.85 | 2425.10 | EMA400 retest candle locked (from upside) |

### Cycle 212 — SELL (started 2026-03-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 14:15:00 | 2398.00 | 2418.26 | 2418.35 | EMA200 below EMA400 |

### Cycle 213 — BUY (started 2026-03-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-20 10:15:00 | 2435.70 | 2421.39 | 2419.67 | EMA200 above EMA400 |

### Cycle 214 — SELL (started 2026-03-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-20 13:15:00 | 2403.90 | 2416.76 | 2417.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-20 14:15:00 | 2383.50 | 2410.10 | 2414.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 09:15:00 | 2306.50 | 2298.75 | 2340.50 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-24 09:30:00 | 2310.60 | 2298.75 | 2340.50 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 12:15:00 | 2347.40 | 2309.66 | 2335.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 12:45:00 | 2349.80 | 2309.66 | 2335.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 13:15:00 | 2355.40 | 2318.81 | 2336.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 14:00:00 | 2355.40 | 2318.81 | 2336.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 215 — BUY (started 2026-03-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 10:15:00 | 2418.70 | 2359.80 | 2351.88 | EMA200 above EMA400 |

### Cycle 216 — SELL (started 2026-03-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 10:15:00 | 2319.30 | 2358.05 | 2358.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-27 11:15:00 | 2304.70 | 2347.38 | 2353.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 2307.90 | 2266.50 | 2295.36 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 2307.90 | 2266.50 | 2295.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 2307.90 | 2266.50 | 2295.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 09:30:00 | 2306.80 | 2266.50 | 2295.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 10:15:00 | 2290.10 | 2271.22 | 2294.88 | EMA400 retest candle locked (from downside) |

### Cycle 217 — BUY (started 2026-04-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 14:15:00 | 2335.10 | 2307.18 | 2306.31 | EMA200 above EMA400 |

### Cycle 218 — SELL (started 2026-04-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-02 09:15:00 | 2261.00 | 2303.11 | 2304.93 | EMA200 below EMA400 |

### Cycle 219 — BUY (started 2026-04-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-02 11:15:00 | 2322.00 | 2307.16 | 2306.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-02 12:15:00 | 2327.90 | 2311.31 | 2308.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-06 10:15:00 | 2332.40 | 2332.81 | 2321.93 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-06 11:00:00 | 2332.40 | 2332.81 | 2321.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-06 12:15:00 | 2355.80 | 2336.58 | 2325.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-06 12:30:00 | 2328.30 | 2336.58 | 2325.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-07 09:15:00 | 2360.00 | 2360.29 | 2342.25 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-08 09:15:00 | 2501.90 | 2348.68 | 2343.30 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2026-04-17 09:15:00 | 2752.09 | 2682.83 | 2647.80 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 220 — SELL (started 2026-04-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-23 11:15:00 | 2743.00 | 2760.26 | 2761.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-23 12:15:00 | 2734.80 | 2755.17 | 2758.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-24 14:15:00 | 2733.60 | 2722.24 | 2734.12 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-24 14:15:00 | 2733.60 | 2722.24 | 2734.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 14:15:00 | 2733.60 | 2722.24 | 2734.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-24 14:45:00 | 2733.60 | 2722.24 | 2734.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 15:15:00 | 2740.00 | 2725.80 | 2734.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 09:15:00 | 2731.00 | 2725.80 | 2734.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 09:15:00 | 2729.50 | 2726.54 | 2734.19 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-27 10:45:00 | 2722.50 | 2727.85 | 2734.09 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-27 12:15:00 | 2764.70 | 2738.60 | 2738.10 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 221 — BUY (started 2026-04-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 12:15:00 | 2764.70 | 2738.60 | 2738.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-28 13:15:00 | 2768.60 | 2755.58 | 2748.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-30 09:15:00 | 2754.80 | 2779.33 | 2770.25 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-30 09:15:00 | 2754.80 | 2779.33 | 2770.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 09:15:00 | 2754.80 | 2779.33 | 2770.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-30 10:00:00 | 2754.80 | 2779.33 | 2770.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 10:15:00 | 2733.00 | 2770.07 | 2766.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-30 10:30:00 | 2728.20 | 2770.07 | 2766.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 222 — SELL (started 2026-04-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-30 11:15:00 | 2719.40 | 2759.93 | 2762.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-30 12:15:00 | 2710.40 | 2750.03 | 2757.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-05-04 09:15:00 | 2760.50 | 2738.97 | 2748.67 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-04 09:15:00 | 2760.50 | 2738.97 | 2748.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 09:15:00 | 2760.50 | 2738.97 | 2748.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-04 09:45:00 | 2768.40 | 2738.97 | 2748.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 10:15:00 | 2752.00 | 2741.57 | 2748.97 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-04 12:00:00 | 2737.00 | 2740.66 | 2747.88 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-04 14:30:00 | 2744.00 | 2743.20 | 2747.28 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-05 09:15:00 | 2709.20 | 2744.38 | 2747.44 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-05-05 10:15:00 | 2772.10 | 2752.10 | 2750.56 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 223 — BUY (started 2026-05-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-05 10:15:00 | 2772.10 | 2752.10 | 2750.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-05 11:15:00 | 2814.50 | 2764.58 | 2756.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-07 09:15:00 | 2788.10 | 2806.59 | 2794.07 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-07 09:15:00 | 2788.10 | 2806.59 | 2794.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-07 09:15:00 | 2788.10 | 2806.59 | 2794.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-07 10:00:00 | 2788.10 | 2806.59 | 2794.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-07 10:15:00 | 2800.20 | 2805.31 | 2794.63 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-07 12:00:00 | 2810.50 | 2806.35 | 2796.07 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2023-06-02 13:45:00 | 973.33 | 2023-06-09 10:15:00 | 968.15 | STOP_HIT | 1.00 | -0.53% |
| BUY | retest2 | 2023-06-02 14:30:00 | 972.50 | 2023-06-09 10:15:00 | 968.15 | STOP_HIT | 1.00 | -0.45% |
| BUY | retest1 | 2023-06-30 09:15:00 | 1127.50 | 2023-07-06 09:15:00 | 1137.03 | STOP_HIT | 1.00 | 0.85% |
| BUY | retest2 | 2023-07-18 10:30:00 | 1203.15 | 2023-07-27 09:15:00 | 1233.53 | STOP_HIT | 1.00 | 2.53% |
| BUY | retest2 | 2023-07-19 10:00:00 | 1202.95 | 2023-07-27 09:15:00 | 1233.53 | STOP_HIT | 1.00 | 2.54% |
| BUY | retest2 | 2023-07-19 11:00:00 | 1201.00 | 2023-07-27 09:15:00 | 1233.53 | STOP_HIT | 1.00 | 2.71% |
| BUY | retest2 | 2023-08-01 09:45:00 | 1273.45 | 2023-08-02 09:15:00 | 1251.50 | STOP_HIT | 1.00 | -1.72% |
| BUY | retest2 | 2023-08-01 10:30:00 | 1268.75 | 2023-08-02 09:15:00 | 1251.50 | STOP_HIT | 1.00 | -1.36% |
| BUY | retest2 | 2023-08-01 12:45:00 | 1269.22 | 2023-08-02 09:15:00 | 1251.50 | STOP_HIT | 1.00 | -1.40% |
| SELL | retest2 | 2023-08-07 10:45:00 | 1204.50 | 2023-08-07 12:15:00 | 1224.80 | STOP_HIT | 1.00 | -1.69% |
| BUY | retest2 | 2023-08-14 11:15:00 | 1249.88 | 2023-08-14 11:15:00 | 1250.50 | STOP_HIT | 1.00 | 0.05% |
| SELL | retest2 | 2023-08-21 13:30:00 | 1238.18 | 2023-08-22 09:15:00 | 1267.30 | STOP_HIT | 1.00 | -2.35% |
| SELL | retest2 | 2023-08-21 14:45:00 | 1237.50 | 2023-08-22 09:15:00 | 1267.30 | STOP_HIT | 1.00 | -2.41% |
| SELL | retest2 | 2023-08-25 15:15:00 | 1241.50 | 2023-08-28 10:15:00 | 1259.38 | STOP_HIT | 1.00 | -1.44% |
| SELL | retest2 | 2023-08-31 10:00:00 | 1239.68 | 2023-08-31 14:15:00 | 1272.80 | STOP_HIT | 1.00 | -2.67% |
| SELL | retest2 | 2023-09-01 09:15:00 | 1224.40 | 2023-09-05 15:15:00 | 1234.00 | STOP_HIT | 1.00 | -0.78% |
| SELL | retest2 | 2023-09-04 10:00:00 | 1240.13 | 2023-09-05 15:15:00 | 1234.00 | STOP_HIT | 1.00 | 0.49% |
| BUY | retest2 | 2023-09-13 11:15:00 | 1296.25 | 2023-09-22 14:15:00 | 1329.05 | STOP_HIT | 1.00 | 2.53% |
| BUY | retest2 | 2023-10-06 09:15:00 | 1329.63 | 2023-10-17 10:15:00 | 1462.59 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2023-10-25 11:30:00 | 1413.58 | 2023-11-02 09:15:00 | 1386.50 | STOP_HIT | 1.00 | 1.92% |
| SELL | retest2 | 2023-10-25 12:00:00 | 1415.45 | 2023-11-02 09:15:00 | 1386.50 | STOP_HIT | 1.00 | 2.05% |
| SELL | retest2 | 2023-11-10 09:15:00 | 1377.00 | 2023-11-10 11:15:00 | 1384.75 | STOP_HIT | 1.00 | -0.56% |
| SELL | retest2 | 2023-11-10 11:00:00 | 1378.25 | 2023-11-10 11:15:00 | 1384.75 | STOP_HIT | 1.00 | -0.47% |
| BUY | retest2 | 2023-11-12 18:15:00 | 1390.60 | 2023-11-13 09:15:00 | 1370.53 | STOP_HIT | 1.00 | -1.44% |
| BUY | retest2 | 2023-11-15 09:15:00 | 1397.20 | 2023-11-22 12:15:00 | 1420.48 | STOP_HIT | 1.00 | 1.67% |
| BUY | retest2 | 2023-11-15 12:45:00 | 1392.78 | 2023-11-22 12:15:00 | 1420.48 | STOP_HIT | 1.00 | 1.99% |
| BUY | retest2 | 2023-11-16 09:15:00 | 1391.30 | 2023-11-22 12:15:00 | 1420.48 | STOP_HIT | 1.00 | 2.10% |
| BUY | retest2 | 2023-11-17 10:15:00 | 1428.45 | 2023-11-22 12:15:00 | 1420.48 | STOP_HIT | 1.00 | -0.56% |
| BUY | retest2 | 2023-12-04 09:15:00 | 1492.50 | 2023-12-08 13:15:00 | 1493.75 | STOP_HIT | 1.00 | 0.08% |
| BUY | retest2 | 2023-12-18 10:15:00 | 1528.45 | 2023-12-29 13:15:00 | 1595.00 | STOP_HIT | 1.00 | 4.35% |
| BUY | retest2 | 2023-12-18 11:30:00 | 1525.38 | 2023-12-29 13:15:00 | 1595.00 | STOP_HIT | 1.00 | 4.56% |
| BUY | retest2 | 2023-12-19 10:30:00 | 1525.60 | 2023-12-29 13:15:00 | 1595.00 | STOP_HIT | 1.00 | 4.55% |
| BUY | retest2 | 2024-01-04 09:15:00 | 1641.08 | 2024-01-16 13:15:00 | 1713.50 | STOP_HIT | 1.00 | 4.41% |
| BUY | retest2 | 2024-01-29 10:15:00 | 1764.50 | 2024-02-05 13:15:00 | 1768.40 | STOP_HIT | 1.00 | 0.22% |
| BUY | retest2 | 2024-01-29 11:45:00 | 1764.95 | 2024-02-05 13:15:00 | 1768.40 | STOP_HIT | 1.00 | 0.20% |
| BUY | retest2 | 2024-01-29 13:45:00 | 1762.93 | 2024-02-05 13:15:00 | 1768.40 | STOP_HIT | 1.00 | 0.31% |
| BUY | retest2 | 2024-01-29 14:15:00 | 1762.63 | 2024-02-05 13:15:00 | 1768.40 | STOP_HIT | 1.00 | 0.33% |
| BUY | retest2 | 2024-02-12 14:15:00 | 1818.15 | 2024-02-19 13:15:00 | 1862.53 | STOP_HIT | 1.00 | 2.44% |
| BUY | retest2 | 2024-02-13 09:15:00 | 1827.00 | 2024-02-19 13:15:00 | 1862.53 | STOP_HIT | 1.00 | 1.94% |
| BUY | retest2 | 2024-02-26 12:30:00 | 1917.40 | 2024-02-28 15:15:00 | 1914.00 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest2 | 2024-04-25 09:15:00 | 1823.23 | 2024-04-25 11:15:00 | 1836.45 | STOP_HIT | 1.00 | -0.73% |
| SELL | retest2 | 2024-04-25 10:15:00 | 1825.60 | 2024-04-25 11:15:00 | 1836.45 | STOP_HIT | 1.00 | -0.59% |
| SELL | retest2 | 2024-04-25 10:45:00 | 1821.20 | 2024-04-25 11:15:00 | 1836.45 | STOP_HIT | 1.00 | -0.84% |
| BUY | retest2 | 2024-05-06 14:15:00 | 1949.95 | 2024-05-07 12:15:00 | 1893.38 | STOP_HIT | 1.00 | -2.90% |
| BUY | retest2 | 2024-05-07 09:15:00 | 1959.18 | 2024-05-07 12:15:00 | 1893.38 | STOP_HIT | 1.00 | -3.36% |
| BUY | retest2 | 2024-05-07 10:45:00 | 1948.98 | 2024-05-07 12:15:00 | 1893.38 | STOP_HIT | 1.00 | -2.85% |
| BUY | retest2 | 2024-05-22 12:15:00 | 1904.98 | 2024-05-30 09:15:00 | 1882.50 | STOP_HIT | 1.00 | -1.18% |
| BUY | retest2 | 2024-05-22 15:00:00 | 1903.98 | 2024-05-30 09:15:00 | 1882.50 | STOP_HIT | 1.00 | -1.13% |
| BUY | retest2 | 2024-05-23 12:15:00 | 1904.23 | 2024-05-30 09:15:00 | 1882.50 | STOP_HIT | 1.00 | -1.14% |
| BUY | retest2 | 2024-05-23 13:15:00 | 1905.40 | 2024-05-30 09:15:00 | 1882.50 | STOP_HIT | 1.00 | -1.20% |
| BUY | retest2 | 2024-05-27 10:15:00 | 1959.05 | 2024-05-30 09:15:00 | 1882.50 | STOP_HIT | 1.00 | -3.91% |
| BUY | retest2 | 2024-05-29 14:45:00 | 1955.70 | 2024-05-30 09:15:00 | 1882.50 | STOP_HIT | 1.00 | -3.74% |
| BUY | retest2 | 2024-06-10 12:30:00 | 1899.90 | 2024-06-19 09:15:00 | 1965.00 | STOP_HIT | 1.00 | 3.43% |
| BUY | retest2 | 2024-06-10 15:00:00 | 1899.73 | 2024-06-19 09:15:00 | 1965.00 | STOP_HIT | 1.00 | 3.44% |
| BUY | retest2 | 2024-06-11 11:15:00 | 1900.93 | 2024-06-19 09:15:00 | 1965.00 | STOP_HIT | 1.00 | 3.37% |
| SELL | retest2 | 2024-06-21 11:00:00 | 1948.08 | 2024-06-24 10:15:00 | 1978.95 | STOP_HIT | 1.00 | -1.58% |
| SELL | retest2 | 2024-06-21 12:30:00 | 1946.80 | 2024-06-24 10:15:00 | 1978.95 | STOP_HIT | 1.00 | -1.65% |
| SELL | retest2 | 2024-06-24 09:15:00 | 1916.65 | 2024-06-24 10:15:00 | 1978.95 | STOP_HIT | 1.00 | -3.25% |
| BUY | retest2 | 2024-07-03 09:15:00 | 2064.00 | 2024-07-08 14:15:00 | 2081.23 | STOP_HIT | 1.00 | 0.83% |
| BUY | retest2 | 2024-07-03 15:00:00 | 2063.32 | 2024-07-08 14:15:00 | 2081.23 | STOP_HIT | 1.00 | 0.87% |
| BUY | retest2 | 2024-07-08 11:45:00 | 2059.53 | 2024-07-08 14:15:00 | 2081.23 | STOP_HIT | 1.00 | 1.05% |
| SELL | retest2 | 2024-07-11 10:15:00 | 2069.18 | 2024-07-11 12:15:00 | 2093.80 | STOP_HIT | 1.00 | -1.19% |
| SELL | retest2 | 2024-07-22 15:15:00 | 2020.00 | 2024-07-23 12:15:00 | 1919.00 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-07-22 15:15:00 | 2020.00 | 2024-07-24 09:15:00 | 2017.53 | STOP_HIT | 0.50 | 0.12% |
| SELL | retest2 | 2024-07-23 12:15:00 | 1906.00 | 2024-07-25 15:15:00 | 2039.45 | STOP_HIT | 1.00 | -7.00% |
| SELL | retest2 | 2024-07-25 09:15:00 | 2021.03 | 2024-07-25 15:15:00 | 2039.45 | STOP_HIT | 1.00 | -0.91% |
| SELL | retest2 | 2024-07-25 12:30:00 | 2024.80 | 2024-07-25 15:15:00 | 2039.45 | STOP_HIT | 1.00 | -0.72% |
| BUY | retest1 | 2024-07-29 09:15:00 | 2074.20 | 2024-07-30 09:15:00 | 2056.07 | STOP_HIT | 1.00 | -0.87% |
| SELL | retest2 | 2024-08-06 10:15:00 | 2016.50 | 2024-08-08 09:15:00 | 2061.50 | STOP_HIT | 1.00 | -2.23% |
| SELL | retest2 | 2024-08-06 12:45:00 | 2015.20 | 2024-08-08 09:15:00 | 2061.50 | STOP_HIT | 1.00 | -2.30% |
| SELL | retest2 | 2024-08-06 13:30:00 | 2014.80 | 2024-08-08 09:15:00 | 2061.50 | STOP_HIT | 1.00 | -2.32% |
| BUY | retest2 | 2024-08-12 11:15:00 | 2046.35 | 2024-08-27 11:15:00 | 2250.99 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-09-12 13:00:00 | 2192.50 | 2024-09-13 10:15:00 | 2216.60 | STOP_HIT | 1.00 | -1.10% |
| SELL | retest2 | 2024-09-12 14:15:00 | 2188.53 | 2024-09-13 10:15:00 | 2216.60 | STOP_HIT | 1.00 | -1.28% |
| BUY | retest2 | 2024-09-18 10:30:00 | 2230.40 | 2024-09-18 13:15:00 | 2201.23 | STOP_HIT | 1.00 | -1.31% |
| BUY | retest2 | 2024-09-18 11:30:00 | 2229.35 | 2024-09-18 13:15:00 | 2201.23 | STOP_HIT | 1.00 | -1.26% |
| BUY | retest2 | 2024-09-18 12:00:00 | 2235.18 | 2024-09-18 13:15:00 | 2201.23 | STOP_HIT | 1.00 | -1.52% |
| BUY | retest2 | 2024-09-19 09:15:00 | 2231.80 | 2024-09-19 09:15:00 | 2193.98 | STOP_HIT | 1.00 | -1.69% |
| SELL | retest2 | 2024-09-20 11:30:00 | 2186.00 | 2024-09-20 12:15:00 | 2200.00 | STOP_HIT | 1.00 | -0.64% |
| SELL | retest2 | 2024-09-20 15:00:00 | 2184.20 | 2024-09-23 11:15:00 | 2202.50 | STOP_HIT | 1.00 | -0.84% |
| SELL | retest2 | 2024-09-23 09:15:00 | 2179.90 | 2024-09-23 11:15:00 | 2202.50 | STOP_HIT | 1.00 | -1.04% |
| SELL | retest2 | 2024-09-23 09:45:00 | 2185.50 | 2024-09-23 11:15:00 | 2202.50 | STOP_HIT | 1.00 | -0.78% |
| SELL | retest2 | 2024-10-03 09:15:00 | 2155.00 | 2024-10-09 10:15:00 | 2168.50 | STOP_HIT | 1.00 | -0.63% |
| BUY | retest2 | 2024-10-11 13:15:00 | 2203.75 | 2024-10-16 12:15:00 | 2424.12 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-10-11 13:45:00 | 2205.80 | 2024-10-16 13:15:00 | 2426.38 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-11-04 09:15:00 | 2137.55 | 2024-11-06 10:15:00 | 2176.23 | STOP_HIT | 1.00 | -1.81% |
| SELL | retest2 | 2024-11-05 13:45:00 | 2149.73 | 2024-11-06 10:15:00 | 2176.23 | STOP_HIT | 1.00 | -1.23% |
| BUY | retest2 | 2024-11-11 11:15:00 | 2246.07 | 2024-11-12 14:15:00 | 2199.98 | STOP_HIT | 1.00 | -2.05% |
| BUY | retest2 | 2024-11-11 12:00:00 | 2243.55 | 2024-11-12 14:15:00 | 2199.98 | STOP_HIT | 1.00 | -1.94% |
| BUY | retest2 | 2024-11-11 14:15:00 | 2245.65 | 2024-11-12 14:15:00 | 2199.98 | STOP_HIT | 1.00 | -2.03% |
| BUY | retest2 | 2024-11-12 09:15:00 | 2251.50 | 2024-11-12 14:15:00 | 2199.98 | STOP_HIT | 1.00 | -2.29% |
| SELL | retest2 | 2024-11-29 10:00:00 | 2099.88 | 2024-12-02 10:15:00 | 2142.98 | STOP_HIT | 1.00 | -2.05% |
| SELL | retest2 | 2024-11-29 13:30:00 | 2101.15 | 2024-12-02 10:15:00 | 2142.98 | STOP_HIT | 1.00 | -1.99% |
| SELL | retest2 | 2024-11-29 15:00:00 | 2100.38 | 2024-12-02 10:15:00 | 2142.98 | STOP_HIT | 1.00 | -2.03% |
| SELL | retest2 | 2024-12-02 09:15:00 | 2093.65 | 2024-12-02 10:15:00 | 2142.98 | STOP_HIT | 1.00 | -2.36% |
| BUY | retest2 | 2024-12-09 10:15:00 | 2201.60 | 2024-12-17 09:15:00 | 2222.82 | STOP_HIT | 1.00 | 0.96% |
| SELL | retest2 | 2024-12-20 12:30:00 | 2165.03 | 2024-12-27 11:15:00 | 2141.70 | STOP_HIT | 1.00 | 1.08% |
| SELL | retest2 | 2024-12-31 09:15:00 | 2106.68 | 2025-01-02 13:15:00 | 2136.50 | STOP_HIT | 1.00 | -1.42% |
| SELL | retest2 | 2024-12-31 14:15:00 | 2108.40 | 2025-01-02 13:15:00 | 2136.50 | STOP_HIT | 1.00 | -1.33% |
| SELL | retest2 | 2025-01-02 12:00:00 | 2108.05 | 2025-01-02 13:15:00 | 2136.50 | STOP_HIT | 1.00 | -1.35% |
| BUY | retest2 | 2025-01-21 13:30:00 | 2039.13 | 2025-01-21 14:15:00 | 2014.45 | STOP_HIT | 1.00 | -1.21% |
| SELL | retest2 | 2025-01-24 12:30:00 | 1962.23 | 2025-01-27 10:15:00 | 1864.12 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-24 12:30:00 | 1962.23 | 2025-01-28 11:15:00 | 1872.58 | STOP_HIT | 0.50 | 4.57% |
| BUY | retest2 | 2025-02-07 11:15:00 | 1984.05 | 2025-02-07 14:15:00 | 1962.80 | STOP_HIT | 1.00 | -1.07% |
| SELL | retest2 | 2025-02-14 10:30:00 | 1862.65 | 2025-02-19 09:15:00 | 1905.83 | STOP_HIT | 1.00 | -2.32% |
| SELL | retest2 | 2025-02-14 12:00:00 | 1871.15 | 2025-02-19 09:15:00 | 1905.83 | STOP_HIT | 1.00 | -1.85% |
| SELL | retest2 | 2025-02-14 15:00:00 | 1875.53 | 2025-02-19 09:15:00 | 1905.83 | STOP_HIT | 1.00 | -1.62% |
| SELL | retest2 | 2025-02-17 09:45:00 | 1866.03 | 2025-02-19 09:15:00 | 1905.83 | STOP_HIT | 1.00 | -2.13% |
| BUY | retest2 | 2025-02-21 13:00:00 | 1909.33 | 2025-02-21 15:15:00 | 1890.98 | STOP_HIT | 1.00 | -0.96% |
| BUY | retest2 | 2025-02-21 14:00:00 | 1908.33 | 2025-02-21 15:15:00 | 1890.98 | STOP_HIT | 1.00 | -0.91% |
| SELL | retest2 | 2025-03-13 09:15:00 | 1839.88 | 2025-03-18 09:15:00 | 1912.05 | STOP_HIT | 1.00 | -3.92% |
| SELL | retest2 | 2025-03-17 10:00:00 | 1878.88 | 2025-03-18 09:15:00 | 1912.05 | STOP_HIT | 1.00 | -1.77% |
| SELL | retest2 | 2025-03-17 13:30:00 | 1878.20 | 2025-03-18 09:15:00 | 1912.05 | STOP_HIT | 1.00 | -1.80% |
| SELL | retest2 | 2025-03-17 14:00:00 | 1878.50 | 2025-03-18 09:15:00 | 1912.05 | STOP_HIT | 1.00 | -1.79% |
| SELL | retest2 | 2025-04-01 11:15:00 | 1975.38 | 2025-04-02 14:15:00 | 2039.05 | STOP_HIT | 1.00 | -3.22% |
| SELL | retest2 | 2025-04-02 09:15:00 | 1986.68 | 2025-04-02 14:15:00 | 2039.05 | STOP_HIT | 1.00 | -2.64% |
| SELL | retest2 | 2025-04-02 10:00:00 | 1982.50 | 2025-04-02 14:15:00 | 2039.05 | STOP_HIT | 1.00 | -2.85% |
| SELL | retest2 | 2025-04-02 11:30:00 | 1986.75 | 2025-04-02 14:15:00 | 2039.05 | STOP_HIT | 1.00 | -2.63% |
| SELL | retest2 | 2025-04-29 11:45:00 | 2197.50 | 2025-05-02 09:15:00 | 2244.45 | STOP_HIT | 1.00 | -2.14% |
| SELL | retest2 | 2025-04-30 11:00:00 | 2200.90 | 2025-05-02 09:15:00 | 2244.45 | STOP_HIT | 1.00 | -1.98% |
| SELL | retest2 | 2025-04-30 13:00:00 | 2196.45 | 2025-05-02 09:15:00 | 2244.45 | STOP_HIT | 1.00 | -2.19% |
| SELL | retest2 | 2025-04-30 13:45:00 | 2200.45 | 2025-05-02 09:15:00 | 2244.45 | STOP_HIT | 1.00 | -2.00% |
| SELL | retest2 | 2025-05-02 11:15:00 | 2216.05 | 2025-05-02 11:15:00 | 2214.15 | STOP_HIT | 1.00 | 0.09% |
| BUY | retest2 | 2025-05-21 12:30:00 | 2403.00 | 2025-05-27 13:15:00 | 2399.20 | STOP_HIT | 1.00 | -0.16% |
| BUY | retest2 | 2025-05-22 14:45:00 | 2395.90 | 2025-05-27 13:15:00 | 2399.20 | STOP_HIT | 1.00 | 0.14% |
| BUY | retest2 | 2025-05-22 15:15:00 | 2399.05 | 2025-05-27 13:15:00 | 2399.20 | STOP_HIT | 1.00 | 0.01% |
| BUY | retest2 | 2025-05-27 10:30:00 | 2398.45 | 2025-05-27 13:15:00 | 2399.20 | STOP_HIT | 1.00 | 0.03% |
| BUY | retest2 | 2025-05-29 15:15:00 | 2428.05 | 2025-05-30 10:15:00 | 2398.05 | STOP_HIT | 1.00 | -1.24% |
| BUY | retest2 | 2025-06-11 10:30:00 | 2583.20 | 2025-06-12 10:15:00 | 2537.90 | STOP_HIT | 1.00 | -1.75% |
| SELL | retest2 | 2025-06-16 14:15:00 | 2481.65 | 2025-06-23 10:15:00 | 2489.75 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest2 | 2025-06-16 15:15:00 | 2481.25 | 2025-06-23 10:15:00 | 2489.75 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest2 | 2025-06-17 11:15:00 | 2481.55 | 2025-06-23 10:15:00 | 2489.75 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest2 | 2025-06-17 14:30:00 | 2484.35 | 2025-06-23 10:15:00 | 2489.75 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest2 | 2025-06-18 10:45:00 | 2466.50 | 2025-06-23 10:15:00 | 2489.75 | STOP_HIT | 1.00 | -0.94% |
| SELL | retest2 | 2025-06-18 11:15:00 | 2468.75 | 2025-06-23 10:15:00 | 2489.75 | STOP_HIT | 1.00 | -0.85% |
| SELL | retest2 | 2025-06-19 10:15:00 | 2471.50 | 2025-06-23 10:15:00 | 2489.75 | STOP_HIT | 1.00 | -0.74% |
| SELL | retest2 | 2025-06-20 11:15:00 | 2472.60 | 2025-06-23 10:15:00 | 2489.75 | STOP_HIT | 1.00 | -0.69% |
| BUY | retest2 | 2025-06-26 12:15:00 | 2529.50 | 2025-07-02 10:15:00 | 2548.25 | STOP_HIT | 1.00 | 0.74% |
| BUY | retest2 | 2025-06-26 13:30:00 | 2526.45 | 2025-07-02 10:15:00 | 2548.25 | STOP_HIT | 1.00 | 0.86% |
| SELL | retest2 | 2025-07-07 12:15:00 | 2512.75 | 2025-07-08 09:15:00 | 2559.75 | STOP_HIT | 1.00 | -1.87% |
| SELL | retest2 | 2025-07-07 13:15:00 | 2508.00 | 2025-07-08 09:15:00 | 2559.75 | STOP_HIT | 1.00 | -2.06% |
| BUY | retest1 | 2025-07-10 09:15:00 | 2600.00 | 2025-07-11 10:15:00 | 2572.75 | STOP_HIT | 1.00 | -1.05% |
| BUY | retest2 | 2025-07-31 13:00:00 | 2830.00 | 2025-08-07 10:15:00 | 2809.75 | STOP_HIT | 1.00 | -0.72% |
| BUY | retest2 | 2025-08-01 10:30:00 | 2832.25 | 2025-08-07 10:15:00 | 2809.75 | STOP_HIT | 1.00 | -0.79% |
| BUY | retest2 | 2025-08-01 11:00:00 | 2830.50 | 2025-08-07 10:15:00 | 2809.75 | STOP_HIT | 1.00 | -0.73% |
| BUY | retest2 | 2025-08-01 11:45:00 | 2829.75 | 2025-08-07 10:15:00 | 2809.75 | STOP_HIT | 1.00 | -0.71% |
| BUY | retest2 | 2025-08-04 09:15:00 | 2833.00 | 2025-08-07 10:15:00 | 2809.75 | STOP_HIT | 1.00 | -0.82% |
| SELL | retest2 | 2025-08-08 09:15:00 | 2806.25 | 2025-08-18 09:15:00 | 2836.50 | STOP_HIT | 1.00 | -1.08% |
| BUY | retest2 | 2025-09-04 12:30:00 | 2794.75 | 2025-09-04 14:15:00 | 2786.50 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest2 | 2025-09-15 11:45:00 | 2878.50 | 2025-09-15 13:15:00 | 2871.50 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest2 | 2025-10-03 13:15:00 | 2782.00 | 2025-10-03 14:15:00 | 2797.25 | STOP_HIT | 1.00 | -0.55% |
| BUY | retest2 | 2025-10-15 09:15:00 | 2835.75 | 2025-10-20 10:15:00 | 2844.50 | STOP_HIT | 1.00 | 0.31% |
| SELL | retest2 | 2025-11-06 15:15:00 | 2690.50 | 2025-11-07 12:15:00 | 2712.75 | STOP_HIT | 1.00 | -0.83% |
| SELL | retest2 | 2025-11-07 10:15:00 | 2686.75 | 2025-11-07 12:15:00 | 2712.75 | STOP_HIT | 1.00 | -0.97% |
| SELL | retest2 | 2025-11-18 10:00:00 | 2709.00 | 2025-11-24 10:15:00 | 2712.00 | STOP_HIT | 1.00 | -0.11% |
| SELL | retest2 | 2025-11-26 12:15:00 | 2665.00 | 2025-11-28 11:15:00 | 2690.00 | STOP_HIT | 1.00 | -0.94% |
| SELL | retest2 | 2025-11-26 14:15:00 | 2670.00 | 2025-11-28 11:15:00 | 2690.00 | STOP_HIT | 1.00 | -0.75% |
| SELL | retest2 | 2025-11-27 10:30:00 | 2668.00 | 2025-11-28 11:15:00 | 2690.00 | STOP_HIT | 1.00 | -0.82% |
| SELL | retest2 | 2025-11-27 13:00:00 | 2668.50 | 2025-11-28 11:15:00 | 2690.00 | STOP_HIT | 1.00 | -0.81% |
| SELL | retest2 | 2025-11-28 09:15:00 | 2675.00 | 2025-11-28 11:15:00 | 2690.00 | STOP_HIT | 1.00 | -0.56% |
| SELL | retest2 | 2025-11-28 10:15:00 | 2667.50 | 2025-11-28 11:15:00 | 2690.00 | STOP_HIT | 1.00 | -0.84% |
| BUY | retest2 | 2025-12-19 13:15:00 | 2655.00 | 2025-12-24 15:15:00 | 2660.10 | STOP_HIT | 1.00 | 0.19% |
| BUY | retest2 | 2025-12-24 14:45:00 | 2654.00 | 2025-12-24 15:15:00 | 2660.10 | STOP_HIT | 1.00 | 0.23% |
| SELL | retest2 | 2025-12-29 15:15:00 | 2640.00 | 2025-12-31 09:15:00 | 2672.80 | STOP_HIT | 1.00 | -1.24% |
| SELL | retest2 | 2026-01-07 14:30:00 | 2630.00 | 2026-01-12 09:15:00 | 2498.50 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-07 14:30:00 | 2630.00 | 2026-01-13 15:15:00 | 2490.00 | STOP_HIT | 0.50 | 5.32% |
| SELL | retest2 | 2026-01-23 10:15:00 | 2485.00 | 2026-01-28 15:15:00 | 2477.30 | STOP_HIT | 1.00 | 0.31% |
| SELL | retest2 | 2026-01-23 11:30:00 | 2477.00 | 2026-01-28 15:15:00 | 2477.30 | STOP_HIT | 1.00 | -0.01% |
| BUY | retest2 | 2026-02-09 09:15:00 | 2728.00 | 2026-02-18 14:15:00 | 2817.10 | STOP_HIT | 1.00 | 3.27% |
| BUY | retest2 | 2026-02-09 09:45:00 | 2727.50 | 2026-02-18 14:15:00 | 2817.10 | STOP_HIT | 1.00 | 3.29% |
| SELL | retest2 | 2026-02-23 12:00:00 | 2722.40 | 2026-02-25 11:15:00 | 2760.00 | STOP_HIT | 1.00 | -1.38% |
| SELL | retest2 | 2026-02-23 14:00:00 | 2717.50 | 2026-02-25 11:15:00 | 2760.00 | STOP_HIT | 1.00 | -1.56% |
| SELL | retest2 | 2026-02-24 09:15:00 | 2714.40 | 2026-02-25 11:15:00 | 2760.00 | STOP_HIT | 1.00 | -1.68% |
| SELL | retest2 | 2026-02-25 12:45:00 | 2718.10 | 2026-02-25 15:15:00 | 2749.90 | STOP_HIT | 1.00 | -1.17% |
| SELL | retest1 | 2026-03-06 09:30:00 | 2538.70 | 2026-03-09 09:15:00 | 2411.76 | PARTIAL | 0.50 | 5.00% |
| SELL | retest1 | 2026-03-06 09:30:00 | 2538.70 | 2026-03-10 09:15:00 | 2472.50 | STOP_HIT | 0.50 | 2.61% |
| SELL | retest2 | 2026-03-11 13:00:00 | 2464.20 | 2026-03-16 10:15:00 | 2340.99 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-11 13:45:00 | 2467.30 | 2026-03-16 10:15:00 | 2343.93 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-12 12:30:00 | 2462.70 | 2026-03-16 10:15:00 | 2339.56 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-12 13:45:00 | 2456.90 | 2026-03-16 10:15:00 | 2334.05 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-11 13:00:00 | 2464.20 | 2026-03-16 14:15:00 | 2369.90 | STOP_HIT | 0.50 | 3.83% |
| SELL | retest2 | 2026-03-11 13:45:00 | 2467.30 | 2026-03-16 14:15:00 | 2369.90 | STOP_HIT | 0.50 | 3.95% |
| SELL | retest2 | 2026-03-12 12:30:00 | 2462.70 | 2026-03-16 14:15:00 | 2369.90 | STOP_HIT | 0.50 | 3.77% |
| SELL | retest2 | 2026-03-12 13:45:00 | 2456.90 | 2026-03-16 14:15:00 | 2369.90 | STOP_HIT | 0.50 | 3.54% |
| SELL | retest2 | 2026-03-17 11:15:00 | 2372.70 | 2026-03-18 09:15:00 | 2437.60 | STOP_HIT | 1.00 | -2.74% |
| SELL | retest2 | 2026-03-17 12:15:00 | 2375.60 | 2026-03-18 09:15:00 | 2437.60 | STOP_HIT | 1.00 | -2.61% |
| SELL | retest2 | 2026-03-17 13:00:00 | 2375.90 | 2026-03-18 09:15:00 | 2437.60 | STOP_HIT | 1.00 | -2.60% |
| BUY | retest2 | 2026-04-08 09:15:00 | 2501.90 | 2026-04-17 09:15:00 | 2752.09 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2026-04-27 10:45:00 | 2722.50 | 2026-04-27 12:15:00 | 2764.70 | STOP_HIT | 1.00 | -1.55% |
| SELL | retest2 | 2026-05-04 12:00:00 | 2737.00 | 2026-05-05 10:15:00 | 2772.10 | STOP_HIT | 1.00 | -1.28% |
| SELL | retest2 | 2026-05-04 14:30:00 | 2744.00 | 2026-05-05 10:15:00 | 2772.10 | STOP_HIT | 1.00 | -1.02% |
| SELL | retest2 | 2026-05-05 09:15:00 | 2709.20 | 2026-05-05 10:15:00 | 2772.10 | STOP_HIT | 1.00 | -2.32% |
