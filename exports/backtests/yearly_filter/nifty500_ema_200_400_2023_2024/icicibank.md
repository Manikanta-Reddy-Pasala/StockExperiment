# ICICI Bank Ltd. (ICICIBANK)

## Backtest Summary

- **Window:** 2022-04-08 09:15:00 → 2026-05-08 15:15:00 (7047 bars)
- **Last close:** 1267.80
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
| ALERT2 | 10 |
| ALERT2_SKIP | 5 |
| ALERT3 | 52 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 59 |
| PARTIAL | 12 |
| TARGET_HIT | 9 |
| STOP_HIT | 49 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 70 (incl. partial bookings)
- **Trades open at end:** 1
- **Winners / losers:** 41 / 29
- **Target hits / Stop hits / Partials:** 9 / 49 / 12
- **Avg / median % per leg:** 1.87% / 1.33%
- **Sum % (uncompounded):** 130.56%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 29 | 17 | 58.6% | 9 | 20 | 0 | 2.93% | 85.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 29 | 17 | 58.6% | 9 | 20 | 0 | 2.93% | 85.0% |
| SELL (all) | 41 | 24 | 58.5% | 0 | 29 | 12 | 1.11% | 45.5% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 41 | 24 | 58.5% | 0 | 29 | 12 | 1.11% | 45.5% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 70 | 41 | 58.6% | 9 | 49 | 12 | 1.87% | 130.6% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-10-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-05 13:15:00 | 939.20 | 959.56 | 959.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-09 12:15:00 | 937.60 | 957.39 | 958.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-12 09:15:00 | 959.00 | 956.24 | 957.75 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-12 09:15:00 | 959.00 | 956.24 | 957.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-12 09:15:00 | 959.00 | 956.24 | 957.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-12 10:00:00 | 959.00 | 956.24 | 957.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-12 10:15:00 | 956.50 | 956.24 | 957.75 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-12 11:15:00 | 955.10 | 956.24 | 957.75 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-12 14:45:00 | 954.95 | 956.15 | 957.67 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-13 14:15:00 | 955.05 | 955.81 | 957.45 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-17 11:30:00 | 955.40 | 955.44 | 957.16 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-17 12:15:00 | 956.35 | 955.45 | 957.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-17 12:45:00 | 956.55 | 955.45 | 957.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-17 13:15:00 | 954.30 | 955.44 | 957.15 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-17 14:15:00 | 953.00 | 955.44 | 957.15 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-17 14:45:00 | 953.25 | 955.42 | 957.13 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-18 09:15:00 | 948.40 | 955.41 | 957.12 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-26 09:15:00 | 907.35 | 948.74 | 953.27 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-26 09:15:00 | 907.20 | 948.74 | 953.27 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-26 09:15:00 | 907.30 | 948.74 | 953.27 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-26 09:15:00 | 907.63 | 948.74 | 953.27 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-26 10:15:00 | 905.35 | 948.28 | 953.02 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-26 10:15:00 | 905.59 | 948.28 | 953.02 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-26 11:15:00 | 900.98 | 947.81 | 952.75 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2023-11-06 09:15:00 | 942.25 | 937.34 | 945.79 | SL hit (close>ema200) qty=0.50 sl=937.34 alert=retest2 |

### Cycle 2 — BUY (started 2023-12-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-06 14:15:00 | 1003.20 | 945.21 | 945.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-08 13:15:00 | 1008.15 | 951.90 | 948.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-02 10:15:00 | 986.45 | 988.42 | 973.63 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-01-02 11:00:00 | 986.45 | 988.42 | 973.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-09 15:15:00 | 978.00 | 988.06 | 976.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-10 09:15:00 | 982.15 | 988.06 | 976.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-10 09:15:00 | 987.60 | 988.05 | 976.15 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-10 10:30:00 | 988.10 | 988.07 | 976.22 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-18 10:45:00 | 988.35 | 991.34 | 980.28 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-18 12:00:00 | 991.50 | 991.34 | 980.33 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-19 09:15:00 | 1000.60 | 991.10 | 980.42 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-08 14:15:00 | 988.80 | 1008.30 | 995.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-08 15:00:00 | 988.80 | 1008.30 | 995.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-08 15:15:00 | 991.90 | 1008.14 | 995.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-09 09:15:00 | 993.80 | 1008.14 | 995.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-12 12:15:00 | 988.60 | 1007.71 | 995.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-12 13:00:00 | 988.60 | 1007.71 | 995.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-12 13:15:00 | 995.55 | 1007.58 | 995.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-12 13:45:00 | 990.15 | 1007.58 | 995.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-12 14:15:00 | 994.50 | 1007.45 | 995.50 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-13 09:15:00 | 1009.15 | 1007.32 | 995.49 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2024-03-01 14:15:00 | 1086.91 | 1033.29 | 1015.31 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 3 — SELL (started 2025-01-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-15 09:15:00 | 1237.35 | 1279.98 | 1280.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-17 09:15:00 | 1226.10 | 1275.07 | 1277.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-28 09:15:00 | 1256.25 | 1252.06 | 1264.07 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-28 10:00:00 | 1256.25 | 1252.06 | 1264.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-04 09:15:00 | 1277.30 | 1251.77 | 1261.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-04 09:45:00 | 1278.60 | 1251.77 | 1261.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-04 10:15:00 | 1273.80 | 1251.99 | 1261.68 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-04 13:15:00 | 1266.90 | 1252.36 | 1261.77 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-04 15:15:00 | 1266.70 | 1252.62 | 1261.81 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-06 11:00:00 | 1265.65 | 1254.32 | 1262.24 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-06 12:30:00 | 1264.90 | 1254.59 | 1262.29 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 11:15:00 | 1259.75 | 1255.31 | 1262.43 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-07 12:30:00 | 1257.30 | 1255.31 | 1262.39 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-10 09:45:00 | 1258.05 | 1255.28 | 1262.24 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-10 15:15:00 | 1259.60 | 1255.32 | 1262.09 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-11 10:30:00 | 1256.15 | 1255.45 | 1262.05 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 09:15:00 | 1249.70 | 1255.19 | 1261.73 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-12 10:30:00 | 1246.75 | 1255.09 | 1261.65 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-12 11:00:00 | 1245.70 | 1255.09 | 1261.65 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-14 12:00:00 | 1245.90 | 1254.84 | 1261.04 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-17 09:30:00 | 1242.85 | 1254.69 | 1260.82 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-19 14:15:00 | 1262.10 | 1253.41 | 1259.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-19 15:00:00 | 1262.10 | 1253.41 | 1259.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-19 15:15:00 | 1261.50 | 1253.49 | 1259.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-20 09:15:00 | 1256.55 | 1253.49 | 1259.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-20 09:15:00 | 1255.50 | 1253.51 | 1259.56 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-20 10:45:00 | 1251.80 | 1253.52 | 1259.54 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-20 11:45:00 | 1249.20 | 1253.46 | 1259.48 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-28 13:15:00 | 1203.56 | 1244.88 | 1253.83 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-28 13:15:00 | 1203.37 | 1244.88 | 1253.83 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-28 15:15:00 | 1202.37 | 1244.06 | 1253.34 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-28 15:15:00 | 1201.65 | 1244.06 | 1253.34 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-03-11 09:15:00 | 1234.70 | 1233.99 | 1245.93 | SL hit (close>ema200) qty=0.50 sl=1233.99 alert=retest2 |

### Cycle 4 — BUY (started 2025-03-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-21 11:15:00 | 1341.20 | 1254.85 | 1254.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-24 09:15:00 | 1358.20 | 1259.18 | 1256.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-07 09:15:00 | 1288.00 | 1295.39 | 1278.50 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-07 10:00:00 | 1288.00 | 1295.39 | 1278.50 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-07 10:15:00 | 1276.05 | 1295.20 | 1278.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-07 11:00:00 | 1276.05 | 1295.20 | 1278.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-07 11:15:00 | 1270.15 | 1294.95 | 1278.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-07 12:00:00 | 1270.15 | 1294.95 | 1278.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-07 12:15:00 | 1270.25 | 1294.70 | 1278.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-07 13:00:00 | 1270.25 | 1294.70 | 1278.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-07 14:15:00 | 1289.55 | 1294.50 | 1278.47 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-07 15:15:00 | 1294.10 | 1294.50 | 1278.47 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-08 10:45:00 | 1292.00 | 1294.53 | 1278.72 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-08 11:45:00 | 1296.85 | 1294.59 | 1278.83 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-09 12:00:00 | 1291.20 | 1294.67 | 1279.41 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Target hit | 2025-04-21 09:15:00 | 1423.51 | 1310.63 | 1290.14 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 5 — SELL (started 2025-09-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-03 12:15:00 | 1391.00 | 1429.41 | 1429.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-24 09:15:00 | 1381.30 | 1416.09 | 1421.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-15 09:15:00 | 1389.80 | 1388.78 | 1401.92 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-16 09:15:00 | 1402.60 | 1389.31 | 1401.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 09:15:00 | 1402.60 | 1389.31 | 1401.74 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-20 14:15:00 | 1396.80 | 1393.78 | 1403.01 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-06 10:15:00 | 1326.96 | 1376.58 | 1390.68 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-13 09:15:00 | 1382.00 | 1368.33 | 1383.90 | SL hit (close>ema200) qty=0.50 sl=1368.33 alert=retest2 |

### Cycle 6 — BUY (started 2026-01-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-12 13:15:00 | 1416.60 | 1377.36 | 1377.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-13 09:15:00 | 1422.70 | 1378.55 | 1377.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-19 09:15:00 | 1362.10 | 1386.14 | 1381.98 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-19 09:15:00 | 1362.10 | 1386.14 | 1381.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 09:15:00 | 1362.10 | 1386.14 | 1381.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-19 10:00:00 | 1362.10 | 1386.14 | 1381.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 10:15:00 | 1370.20 | 1385.98 | 1381.92 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-19 11:45:00 | 1372.50 | 1385.83 | 1381.86 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-19 13:45:00 | 1380.10 | 1385.62 | 1381.80 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-21 09:15:00 | 1348.60 | 1384.81 | 1381.58 | SL hit (close<static) qty=1.00 sl=1360.00 alert=retest2 |

### Cycle 7 — SELL (started 2026-01-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-23 13:15:00 | 1347.60 | 1378.57 | 1378.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-23 14:15:00 | 1344.30 | 1378.23 | 1378.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-29 13:15:00 | 1378.00 | 1375.40 | 1376.90 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-29 13:15:00 | 1378.00 | 1375.40 | 1376.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 13:15:00 | 1378.00 | 1375.40 | 1376.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-29 14:00:00 | 1378.00 | 1375.40 | 1376.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 14:15:00 | 1383.20 | 1375.48 | 1376.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-29 14:45:00 | 1384.10 | 1375.48 | 1376.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 15:15:00 | 1385.00 | 1375.58 | 1376.97 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-30 09:15:00 | 1375.10 | 1375.58 | 1376.97 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-03 09:15:00 | 1390.10 | 1371.24 | 1374.55 | SL hit (close>static) qty=1.00 sl=1386.00 alert=retest2 |

### Cycle 8 — BUY (started 2026-02-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-06 12:15:00 | 1401.90 | 1377.60 | 1377.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-06 13:15:00 | 1403.70 | 1377.86 | 1377.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-19 14:15:00 | 1386.80 | 1391.26 | 1385.53 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-19 15:00:00 | 1386.80 | 1391.26 | 1385.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 15:15:00 | 1388.70 | 1391.24 | 1385.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-20 09:15:00 | 1389.30 | 1391.24 | 1385.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 09:15:00 | 1398.10 | 1391.31 | 1385.61 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-20 10:15:00 | 1400.80 | 1391.31 | 1385.61 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-23 09:15:00 | 1404.80 | 1391.58 | 1385.91 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-23 14:15:00 | 1400.40 | 1392.06 | 1386.30 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-25 09:30:00 | 1400.00 | 1392.14 | 1386.62 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 10:15:00 | 1389.40 | 1393.23 | 1387.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-27 10:45:00 | 1386.40 | 1393.23 | 1387.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 12:15:00 | 1383.70 | 1393.09 | 1387.57 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-02-27 12:15:00 | 1383.70 | 1393.09 | 1387.57 | SL hit (close<static) qty=1.00 sl=1384.00 alert=retest2 |

### Cycle 9 — SELL (started 2026-03-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-06 14:15:00 | 1312.00 | 1383.02 | 1383.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-09 09:15:00 | 1271.90 | 1381.25 | 1382.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 09:15:00 | 1302.40 | 1281.40 | 1317.76 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-08 09:15:00 | 1302.40 | 1281.40 | 1317.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 09:15:00 | 1302.40 | 1281.40 | 1317.76 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-09 09:45:00 | 1291.50 | 1283.11 | 1317.38 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-09 10:30:00 | 1291.90 | 1283.23 | 1317.27 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-09 12:00:00 | 1290.30 | 1283.30 | 1317.13 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-13 13:15:00 | 1350.30 | 1287.89 | 1316.89 | SL hit (close>static) qty=1.00 sl=1333.50 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2023-06-19 14:30:00 | 924.90 | 2023-06-20 09:15:00 | 917.40 | STOP_HIT | 1.00 | -0.81% |
| BUY | retest2 | 2023-06-20 13:15:00 | 921.75 | 2023-10-05 13:15:00 | 939.20 | STOP_HIT | 1.00 | 1.89% |
| BUY | retest2 | 2023-06-20 14:15:00 | 921.60 | 2023-10-05 13:15:00 | 939.20 | STOP_HIT | 1.00 | 1.91% |
| BUY | retest2 | 2023-06-20 15:00:00 | 925.45 | 2023-10-05 13:15:00 | 939.20 | STOP_HIT | 1.00 | 1.49% |
| BUY | retest2 | 2023-06-21 12:30:00 | 923.50 | 2023-10-05 13:15:00 | 939.20 | STOP_HIT | 1.00 | 1.70% |
| BUY | retest2 | 2023-06-23 11:00:00 | 923.25 | 2023-10-05 13:15:00 | 939.20 | STOP_HIT | 1.00 | 1.73% |
| BUY | retest2 | 2023-06-23 13:15:00 | 924.25 | 2023-10-05 13:15:00 | 939.20 | STOP_HIT | 1.00 | 1.62% |
| SELL | retest2 | 2023-10-12 11:15:00 | 955.10 | 2023-10-26 09:15:00 | 907.35 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-10-12 14:45:00 | 954.95 | 2023-10-26 09:15:00 | 907.20 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-10-13 14:15:00 | 955.05 | 2023-10-26 09:15:00 | 907.30 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-10-17 11:30:00 | 955.40 | 2023-10-26 09:15:00 | 907.63 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-10-17 14:15:00 | 953.00 | 2023-10-26 10:15:00 | 905.35 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-10-17 14:45:00 | 953.25 | 2023-10-26 10:15:00 | 905.59 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-10-18 09:15:00 | 948.40 | 2023-10-26 11:15:00 | 900.98 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-10-12 11:15:00 | 955.10 | 2023-11-06 09:15:00 | 942.25 | STOP_HIT | 0.50 | 1.35% |
| SELL | retest2 | 2023-10-12 14:45:00 | 954.95 | 2023-11-06 09:15:00 | 942.25 | STOP_HIT | 0.50 | 1.33% |
| SELL | retest2 | 2023-10-13 14:15:00 | 955.05 | 2023-11-06 09:15:00 | 942.25 | STOP_HIT | 0.50 | 1.34% |
| SELL | retest2 | 2023-10-17 11:30:00 | 955.40 | 2023-11-06 09:15:00 | 942.25 | STOP_HIT | 0.50 | 1.38% |
| SELL | retest2 | 2023-10-17 14:15:00 | 953.00 | 2023-11-06 09:15:00 | 942.25 | STOP_HIT | 0.50 | 1.13% |
| SELL | retest2 | 2023-10-17 14:45:00 | 953.25 | 2023-11-06 09:15:00 | 942.25 | STOP_HIT | 0.50 | 1.15% |
| SELL | retest2 | 2023-10-18 09:15:00 | 948.40 | 2023-11-06 09:15:00 | 942.25 | STOP_HIT | 0.50 | 0.65% |
| BUY | retest2 | 2024-01-10 10:30:00 | 988.10 | 2024-03-01 14:15:00 | 1086.91 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-01-18 10:45:00 | 988.35 | 2024-03-01 14:15:00 | 1087.19 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-01-18 12:00:00 | 991.50 | 2024-03-04 09:15:00 | 1090.65 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-01-19 09:15:00 | 1000.60 | 2024-03-06 09:15:00 | 1100.66 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-02-13 09:15:00 | 1009.15 | 2024-03-06 09:15:00 | 1110.07 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-02-04 13:15:00 | 1266.90 | 2025-02-28 13:15:00 | 1203.56 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-04 15:15:00 | 1266.70 | 2025-02-28 13:15:00 | 1203.37 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-06 11:00:00 | 1265.65 | 2025-02-28 15:15:00 | 1202.37 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-06 12:30:00 | 1264.90 | 2025-02-28 15:15:00 | 1201.65 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-04 13:15:00 | 1266.90 | 2025-03-11 09:15:00 | 1234.70 | STOP_HIT | 0.50 | 2.54% |
| SELL | retest2 | 2025-02-04 15:15:00 | 1266.70 | 2025-03-11 09:15:00 | 1234.70 | STOP_HIT | 0.50 | 2.53% |
| SELL | retest2 | 2025-02-06 11:00:00 | 1265.65 | 2025-03-11 09:15:00 | 1234.70 | STOP_HIT | 0.50 | 2.45% |
| SELL | retest2 | 2025-02-06 12:30:00 | 1264.90 | 2025-03-11 09:15:00 | 1234.70 | STOP_HIT | 0.50 | 2.39% |
| SELL | retest2 | 2025-02-07 12:30:00 | 1257.30 | 2025-03-17 09:15:00 | 1266.65 | STOP_HIT | 1.00 | -0.74% |
| SELL | retest2 | 2025-02-10 09:45:00 | 1258.05 | 2025-03-17 09:15:00 | 1266.65 | STOP_HIT | 1.00 | -0.68% |
| SELL | retest2 | 2025-02-10 15:15:00 | 1259.60 | 2025-03-17 09:15:00 | 1266.65 | STOP_HIT | 1.00 | -0.56% |
| SELL | retest2 | 2025-02-11 10:30:00 | 1256.15 | 2025-03-17 09:15:00 | 1266.65 | STOP_HIT | 1.00 | -0.84% |
| SELL | retest2 | 2025-02-12 10:30:00 | 1246.75 | 2025-03-17 10:15:00 | 1270.25 | STOP_HIT | 1.00 | -1.88% |
| SELL | retest2 | 2025-02-12 11:00:00 | 1245.70 | 2025-03-17 10:15:00 | 1270.25 | STOP_HIT | 1.00 | -1.97% |
| SELL | retest2 | 2025-02-14 12:00:00 | 1245.90 | 2025-03-17 10:15:00 | 1270.25 | STOP_HIT | 1.00 | -1.95% |
| SELL | retest2 | 2025-02-17 09:30:00 | 1242.85 | 2025-03-17 10:15:00 | 1270.25 | STOP_HIT | 1.00 | -2.20% |
| SELL | retest2 | 2025-02-20 10:45:00 | 1251.80 | 2025-03-17 10:15:00 | 1270.25 | STOP_HIT | 1.00 | -1.47% |
| SELL | retest2 | 2025-02-20 11:45:00 | 1249.20 | 2025-03-17 10:15:00 | 1270.25 | STOP_HIT | 1.00 | -1.69% |
| SELL | retest2 | 2025-03-13 11:15:00 | 1250.40 | 2025-03-17 10:15:00 | 1270.25 | STOP_HIT | 1.00 | -1.59% |
| SELL | retest2 | 2025-03-13 13:15:00 | 1250.80 | 2025-03-17 10:15:00 | 1270.25 | STOP_HIT | 1.00 | -1.56% |
| BUY | retest2 | 2025-04-07 15:15:00 | 1294.10 | 2025-04-21 09:15:00 | 1423.51 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-04-08 10:45:00 | 1292.00 | 2025-04-21 09:15:00 | 1421.20 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-04-08 11:45:00 | 1296.85 | 2025-04-21 09:15:00 | 1426.54 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-04-09 12:00:00 | 1291.20 | 2025-04-21 09:15:00 | 1420.32 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-07-17 12:30:00 | 1419.50 | 2025-07-18 10:15:00 | 1413.50 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest2 | 2025-07-17 14:15:00 | 1419.30 | 2025-07-18 10:15:00 | 1413.50 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest2 | 2025-07-18 13:45:00 | 1420.30 | 2025-08-25 09:15:00 | 1425.00 | STOP_HIT | 1.00 | 0.33% |
| BUY | retest2 | 2025-08-13 11:30:00 | 1420.10 | 2025-08-25 09:15:00 | 1425.00 | STOP_HIT | 1.00 | 0.35% |
| BUY | retest2 | 2025-08-21 09:15:00 | 1438.80 | 2025-08-25 09:15:00 | 1425.00 | STOP_HIT | 1.00 | -0.96% |
| BUY | retest2 | 2025-08-22 12:15:00 | 1439.00 | 2025-08-28 09:15:00 | 1408.40 | STOP_HIT | 1.00 | -2.13% |
| BUY | retest2 | 2025-08-22 14:45:00 | 1439.50 | 2025-08-28 09:15:00 | 1408.40 | STOP_HIT | 1.00 | -2.16% |
| SELL | retest2 | 2025-10-20 14:15:00 | 1396.80 | 2025-11-06 10:15:00 | 1326.96 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-20 14:15:00 | 1396.80 | 2025-11-13 09:15:00 | 1382.00 | STOP_HIT | 0.50 | 1.06% |
| SELL | retest2 | 2026-01-12 09:15:00 | 1395.20 | 2026-01-12 12:15:00 | 1420.10 | STOP_HIT | 1.00 | -1.78% |
| BUY | retest2 | 2026-01-19 11:45:00 | 1372.50 | 2026-01-21 09:15:00 | 1348.60 | STOP_HIT | 1.00 | -1.74% |
| BUY | retest2 | 2026-01-19 13:45:00 | 1380.10 | 2026-01-21 09:15:00 | 1348.60 | STOP_HIT | 1.00 | -2.28% |
| SELL | retest2 | 2026-01-30 09:15:00 | 1375.10 | 2026-02-03 09:15:00 | 1390.10 | STOP_HIT | 1.00 | -1.09% |
| BUY | retest2 | 2026-02-20 10:15:00 | 1400.80 | 2026-02-27 12:15:00 | 1383.70 | STOP_HIT | 1.00 | -1.22% |
| BUY | retest2 | 2026-02-23 09:15:00 | 1404.80 | 2026-02-27 12:15:00 | 1383.70 | STOP_HIT | 1.00 | -1.50% |
| BUY | retest2 | 2026-02-23 14:15:00 | 1400.40 | 2026-02-27 12:15:00 | 1383.70 | STOP_HIT | 1.00 | -1.19% |
| BUY | retest2 | 2026-02-25 09:30:00 | 1400.00 | 2026-02-27 12:15:00 | 1383.70 | STOP_HIT | 1.00 | -1.16% |
| SELL | retest2 | 2026-04-09 09:45:00 | 1291.50 | 2026-04-13 13:15:00 | 1350.30 | STOP_HIT | 1.00 | -4.55% |
| SELL | retest2 | 2026-04-09 10:30:00 | 1291.90 | 2026-04-13 13:15:00 | 1350.30 | STOP_HIT | 1.00 | -4.52% |
| SELL | retest2 | 2026-04-09 12:00:00 | 1290.30 | 2026-04-13 13:15:00 | 1350.30 | STOP_HIT | 1.00 | -4.65% |
