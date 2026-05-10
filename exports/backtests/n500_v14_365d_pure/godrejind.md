# Godrej Industries Ltd. (GODREJIND)

## Backtest Summary

- **Window:** 2024-04-05 09:15:00 → 2026-05-08 15:15:00 (3605 bars)
- **Last close:** 1202.00
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
| ALERT2 | 3 |
| ALERT2_SKIP | 1 |
| ALERT3 | 17 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 27 |
| PARTIAL | 8 |
| TARGET_HIT | 6 |
| STOP_HIT | 25 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 35 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 16 / 19
- **Target hits / Stop hits / Partials:** 2 / 25 / 8
- **Avg / median % per leg:** -0.91% / -0.85%
- **Sum % (uncompounded):** -31.74%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 14 | 3 | 21.4% | 1 | 13 | 0 | -0.23% | -3.3% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 14 | 3 | 21.4% | 1 | 13 | 0 | -0.23% | -3.3% |
| SELL (all) | 21 | 13 | 61.9% | 1 | 12 | 8 | -1.36% | -28.5% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 21 | 13 | 61.9% | 1 | 12 | 8 | -1.36% | -28.5% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 35 | 16 | 45.7% | 2 | 25 | 8 | -0.91% | -31.7% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-07-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-22 15:15:00 | 1143.00 | 1176.65 | 1176.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-23 09:15:00 | 1139.20 | 1176.27 | 1176.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-13 13:15:00 | 1203.40 | 1137.86 | 1152.58 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-13 13:15:00 | 1203.40 | 1137.86 | 1152.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 13:15:00 | 1203.40 | 1137.86 | 1152.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-13 14:00:00 | 1203.40 | 1137.86 | 1152.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 14:15:00 | 1169.10 | 1138.17 | 1152.67 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-14 12:00:00 | 1157.00 | 1139.32 | 1152.96 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-18 10:15:00 | 1226.50 | 1141.95 | 1153.89 | SL hit (close>static) qty=1.00 sl=1220.10 alert=retest2 |

### Cycle 2 — BUY (started 2025-08-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-20 14:15:00 | 1285.00 | 1164.97 | 1164.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-22 09:15:00 | 1299.00 | 1175.51 | 1170.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-02 14:15:00 | 1198.40 | 1202.73 | 1186.98 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-02 15:00:00 | 1198.40 | 1202.73 | 1186.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 11:15:00 | 1185.10 | 1201.95 | 1187.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 12:00:00 | 1185.10 | 1201.95 | 1187.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 12:15:00 | 1188.20 | 1201.81 | 1187.42 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-04 15:15:00 | 1204.70 | 1201.54 | 1187.43 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-05 11:15:00 | 1184.00 | 1201.26 | 1187.57 | SL hit (close<static) qty=1.00 sl=1185.10 alert=retest2 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-05 13:30:00 | 1197.70 | 1201.13 | 1187.64 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-19 12:15:00 | 1181.70 | 1207.17 | 1195.05 | SL hit (close<static) qty=1.00 sl=1185.10 alert=retest2 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-22 10:15:00 | 1192.90 | 1206.02 | 1194.70 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-22 11:15:00 | 1184.90 | 1205.60 | 1194.61 | SL hit (close<static) qty=1.00 sl=1185.10 alert=retest2 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-24 09:45:00 | 1192.40 | 1202.73 | 1193.77 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 10:15:00 | 1193.40 | 1202.63 | 1193.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-24 10:45:00 | 1186.90 | 1202.63 | 1193.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 11:15:00 | 1190.90 | 1202.52 | 1193.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-24 11:45:00 | 1191.40 | 1202.52 | 1193.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 13:15:00 | 1194.60 | 1202.36 | 1193.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-24 13:30:00 | 1190.50 | 1202.36 | 1193.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 14:15:00 | 1190.00 | 1202.23 | 1193.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-24 15:00:00 | 1190.00 | 1202.23 | 1193.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 15:15:00 | 1193.00 | 1202.14 | 1193.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-25 09:15:00 | 1191.50 | 1202.14 | 1193.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 09:15:00 | 1195.80 | 1202.08 | 1193.74 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-25 10:45:00 | 1213.40 | 1202.25 | 1193.87 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-26 10:15:00 | 1201.70 | 1203.10 | 1194.56 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-26 12:30:00 | 1197.00 | 1202.92 | 1194.59 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-26 15:15:00 | 1199.00 | 1202.77 | 1194.60 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 09:15:00 | 1196.40 | 1202.67 | 1194.63 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-30 09:45:00 | 1201.00 | 1202.26 | 1194.70 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-30 15:00:00 | 1202.30 | 1202.03 | 1194.77 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-01 09:15:00 | 1172.70 | 1201.75 | 1194.70 | SL hit (close<static) qty=1.00 sl=1185.10 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-01 09:15:00 | 1172.70 | 1201.75 | 1194.70 | SL hit (close<static) qty=1.00 sl=1184.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-01 09:15:00 | 1172.70 | 1201.75 | 1194.70 | SL hit (close<static) qty=1.00 sl=1184.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-01 09:15:00 | 1172.70 | 1201.75 | 1194.70 | SL hit (close<static) qty=1.00 sl=1184.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-01 09:15:00 | 1172.70 | 1201.75 | 1194.70 | SL hit (close<static) qty=1.00 sl=1184.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-01 09:15:00 | 1172.70 | 1201.75 | 1194.70 | SL hit (close<static) qty=1.00 sl=1184.60 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-01 09:15:00 | 1172.70 | 1201.75 | 1194.70 | SL hit (close<static) qty=1.00 sl=1184.60 alert=retest2 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-01 11:15:00 | 1201.10 | 1201.72 | 1194.72 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-03 10:15:00 | 1172.90 | 1201.06 | 1194.63 | SL hit (close<static) qty=1.00 sl=1184.60 alert=retest2 |

### Cycle 3 — SELL (started 2025-10-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-08 09:15:00 | 1115.10 | 1188.82 | 1188.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-08 10:15:00 | 1107.60 | 1188.01 | 1188.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-31 12:15:00 | 1122.20 | 1118.41 | 1143.50 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-31 12:30:00 | 1125.80 | 1118.41 | 1143.50 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 14:15:00 | 1055.30 | 1020.46 | 1047.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-07 15:00:00 | 1055.30 | 1020.46 | 1047.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 15:15:00 | 1050.50 | 1020.75 | 1047.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-08 09:15:00 | 1045.50 | 1020.75 | 1047.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 10:15:00 | 1034.10 | 1021.19 | 1047.06 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-08 11:15:00 | 1030.10 | 1021.19 | 1047.06 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-08 15:00:00 | 1026.10 | 1021.63 | 1046.77 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-21 10:15:00 | 978.59 | 1014.50 | 1037.07 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-21 11:15:00 | 974.79 | 1014.15 | 1036.78 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-03 09:15:00 | 1001.00 | 998.61 | 1021.96 | SL hit (close>ema200) qty=0.50 sl=998.61 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-03 09:15:00 | 1001.00 | 998.61 | 1021.96 | SL hit (close>ema200) qty=0.50 sl=998.61 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-03 13:30:00 | 1024.40 | 999.58 | 1021.99 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-10 10:00:00 | 1032.50 | 1002.03 | 1020.06 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 10:15:00 | 1024.50 | 1004.44 | 1020.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-11 11:00:00 | 1024.50 | 1004.44 | 1020.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 13:15:00 | 1014.15 | 1004.89 | 1020.57 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-12 09:15:00 | 997.30 | 1005.10 | 1020.52 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-12 12:30:00 | 1011.30 | 1005.18 | 1020.25 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-12 13:15:00 | 1012.90 | 1005.18 | 1020.25 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-12 14:00:00 | 1011.05 | 1005.23 | 1020.20 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-13 13:15:00 | 980.88 | 1004.24 | 1019.19 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-13 14:15:00 | 973.18 | 1003.84 | 1018.91 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-13 14:15:00 | 962.25 | 1003.84 | 1018.91 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-13 15:15:00 | 960.73 | 1003.51 | 1018.67 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-13 15:15:00 | 960.50 | 1003.51 | 1018.67 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 09:15:00 | 1021.50 | 998.57 | 1014.46 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-02-19 09:15:00 | 1021.50 | 998.57 | 1014.46 | SL hit (close>ema200) qty=0.50 sl=998.57 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-19 09:15:00 | 1021.50 | 998.57 | 1014.46 | SL hit (close>ema200) qty=0.50 sl=998.57 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-19 09:15:00 | 1021.50 | 998.57 | 1014.46 | SL hit (close>static) qty=1.00 sl=1021.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-19 09:15:00 | 1021.50 | 998.57 | 1014.46 | SL hit (close>ema200) qty=0.50 sl=998.57 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-19 09:15:00 | 1021.50 | 998.57 | 1014.46 | SL hit (close>ema200) qty=0.50 sl=998.57 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-19 09:15:00 | 1021.50 | 998.57 | 1014.46 | SL hit (close>ema200) qty=0.50 sl=998.57 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-02 09:15:00 | 977.60 | 1010.47 | 1017.89 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-05 11:15:00 | 928.72 | 1001.15 | 1012.46 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-03-12 09:15:00 | 879.84 | 977.45 | 998.01 | Target hit (10%) qty=0.50 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-24 09:15:00 | 968.00 | 900.11 | 922.44 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-24 13:00:00 | 977.10 | 903.07 | 923.49 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-29 09:45:00 | 974.25 | 917.59 | 929.32 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-05-07 09:15:00 | 1208.15 | 938.06 | 938.33 | SL hit (close>static) qty=1.00 sl=1043.90 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-05-07 09:15:00 | 1208.15 | 938.06 | 938.33 | SL hit (close>static) qty=1.00 sl=1043.90 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-05-07 09:15:00 | 1208.15 | 938.06 | 938.33 | SL hit (close>static) qty=1.00 sl=1043.90 alert=retest2 |

### Cycle 4 — BUY (started 2026-05-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-07 10:15:00 | 1155.05 | 940.22 | 939.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-07 11:15:00 | 1212.00 | 942.92 | 940.77 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-05-12 09:15:00 | 1117.00 | 2025-06-06 09:15:00 | 1225.29 | TARGET_HIT | 1.00 | 9.69% |
| BUY | retest2 | 2025-07-14 09:30:00 | 1094.20 | 2025-07-22 15:15:00 | 1143.00 | STOP_HIT | 1.00 | 4.46% |
| BUY | retest2 | 2025-07-14 11:15:00 | 1087.30 | 2025-07-22 15:15:00 | 1143.00 | STOP_HIT | 1.00 | 5.12% |
| SELL | retest2 | 2025-08-14 12:00:00 | 1157.00 | 2025-08-18 10:15:00 | 1226.50 | STOP_HIT | 1.00 | -6.01% |
| BUY | retest2 | 2025-09-04 15:15:00 | 1204.70 | 2025-09-05 11:15:00 | 1184.00 | STOP_HIT | 1.00 | -1.72% |
| BUY | retest2 | 2025-09-05 13:30:00 | 1197.70 | 2025-09-19 12:15:00 | 1181.70 | STOP_HIT | 1.00 | -1.34% |
| BUY | retest2 | 2025-09-22 10:15:00 | 1192.90 | 2025-09-22 11:15:00 | 1184.90 | STOP_HIT | 1.00 | -0.67% |
| BUY | retest2 | 2025-09-24 09:45:00 | 1192.40 | 2025-10-01 09:15:00 | 1172.70 | STOP_HIT | 1.00 | -1.65% |
| BUY | retest2 | 2025-09-25 10:45:00 | 1213.40 | 2025-10-01 09:15:00 | 1172.70 | STOP_HIT | 1.00 | -3.35% |
| BUY | retest2 | 2025-09-26 10:15:00 | 1201.70 | 2025-10-01 09:15:00 | 1172.70 | STOP_HIT | 1.00 | -2.41% |
| BUY | retest2 | 2025-09-26 12:30:00 | 1197.00 | 2025-10-01 09:15:00 | 1172.70 | STOP_HIT | 1.00 | -2.03% |
| BUY | retest2 | 2025-09-26 15:15:00 | 1199.00 | 2025-10-01 09:15:00 | 1172.70 | STOP_HIT | 1.00 | -2.19% |
| BUY | retest2 | 2025-09-30 09:45:00 | 1201.00 | 2025-10-01 09:15:00 | 1172.70 | STOP_HIT | 1.00 | -2.36% |
| BUY | retest2 | 2025-09-30 15:00:00 | 1202.30 | 2025-10-01 09:15:00 | 1172.70 | STOP_HIT | 1.00 | -2.46% |
| BUY | retest2 | 2025-10-01 11:15:00 | 1201.10 | 2025-10-03 10:15:00 | 1172.90 | STOP_HIT | 1.00 | -2.35% |
| SELL | retest2 | 2026-01-08 11:15:00 | 1030.10 | 2026-01-21 10:15:00 | 978.59 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-08 15:00:00 | 1026.10 | 2026-01-21 11:15:00 | 974.79 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-08 11:15:00 | 1030.10 | 2026-02-03 09:15:00 | 1001.00 | STOP_HIT | 0.50 | 2.82% |
| SELL | retest2 | 2026-01-08 15:00:00 | 1026.10 | 2026-02-03 09:15:00 | 1001.00 | STOP_HIT | 0.50 | 2.45% |
| SELL | retest2 | 2026-02-03 13:30:00 | 1024.40 | 2026-02-13 13:15:00 | 980.88 | PARTIAL | 0.50 | 4.25% |
| SELL | retest2 | 2026-02-10 10:00:00 | 1032.50 | 2026-02-13 14:15:00 | 973.18 | PARTIAL | 0.50 | 5.75% |
| SELL | retest2 | 2026-02-12 09:15:00 | 997.30 | 2026-02-13 14:15:00 | 962.25 | PARTIAL | 0.50 | 3.51% |
| SELL | retest2 | 2026-02-12 12:30:00 | 1011.30 | 2026-02-13 15:15:00 | 960.73 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-12 13:15:00 | 1012.90 | 2026-02-13 15:15:00 | 960.50 | PARTIAL | 0.50 | 5.17% |
| SELL | retest2 | 2026-02-03 13:30:00 | 1024.40 | 2026-02-19 09:15:00 | 1021.50 | STOP_HIT | 0.50 | 0.28% |
| SELL | retest2 | 2026-02-10 10:00:00 | 1032.50 | 2026-02-19 09:15:00 | 1021.50 | STOP_HIT | 0.50 | 1.07% |
| SELL | retest2 | 2026-02-12 09:15:00 | 997.30 | 2026-02-19 09:15:00 | 1021.50 | STOP_HIT | 0.50 | -2.43% |
| SELL | retest2 | 2026-02-12 12:30:00 | 1011.30 | 2026-02-19 09:15:00 | 1021.50 | STOP_HIT | 0.50 | -1.01% |
| SELL | retest2 | 2026-02-12 13:15:00 | 1012.90 | 2026-02-19 09:15:00 | 1021.50 | STOP_HIT | 0.50 | -0.85% |
| SELL | retest2 | 2026-02-12 14:00:00 | 1011.05 | 2026-02-19 09:15:00 | 1021.50 | STOP_HIT | 1.00 | -1.03% |
| SELL | retest2 | 2026-03-02 09:15:00 | 977.60 | 2026-03-05 11:15:00 | 928.72 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-02 09:15:00 | 977.60 | 2026-03-12 09:15:00 | 879.84 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-04-24 09:15:00 | 968.00 | 2026-05-07 09:15:00 | 1208.15 | STOP_HIT | 1.00 | -24.81% |
| SELL | retest2 | 2026-04-24 13:00:00 | 977.10 | 2026-05-07 09:15:00 | 1208.15 | STOP_HIT | 1.00 | -23.65% |
| SELL | retest2 | 2026-04-29 09:45:00 | 974.25 | 2026-05-07 09:15:00 | 1208.15 | STOP_HIT | 1.00 | -24.01% |
