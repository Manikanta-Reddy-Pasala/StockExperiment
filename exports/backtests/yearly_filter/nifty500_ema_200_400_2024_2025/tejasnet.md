# Tejas Networks Ltd. (TEJASNET)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 515.50
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
| ALERT2_SKIP | 2 |
| ALERT3 | 42 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 40 |
| PARTIAL | 9 |
| TARGET_HIT | 9 |
| STOP_HIT | 31 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 49 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 23 / 26
- **Target hits / Stop hits / Partials:** 9 / 31 / 9
- **Avg / median % per leg:** 1.91% / -0.55%
- **Sum % (uncompounded):** 93.66%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 24 | 0 | 0.0% | 0 | 24 | 0 | -2.33% | -56.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 24 | 0 | 0.0% | 0 | 24 | 0 | -2.33% | -56.0% |
| SELL (all) | 25 | 23 | 92.0% | 9 | 7 | 9 | 5.99% | 149.7% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 25 | 23 | 92.0% | 9 | 7 | 9 | 5.99% | 149.7% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 49 | 23 | 46.9% | 9 | 31 | 9 | 1.91% | 93.7% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-09-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-27 11:15:00 | 1222.00 | 1264.53 | 1264.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-27 12:15:00 | 1216.00 | 1264.05 | 1264.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-21 09:15:00 | 1348.10 | 1210.45 | 1230.72 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-21 09:15:00 | 1348.10 | 1210.45 | 1230.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-21 09:15:00 | 1348.10 | 1210.45 | 1230.72 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-21 11:15:00 | 1298.25 | 1211.77 | 1231.28 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-23 12:15:00 | 1335.40 | 1223.66 | 1236.04 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-23 13:00:00 | 1336.05 | 1224.78 | 1236.54 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-23 14:00:00 | 1327.80 | 1225.80 | 1236.99 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-25 09:15:00 | 1268.63 | 1233.17 | 1240.23 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-10-25 09:15:00 | 1256.25 | 1233.17 | 1240.23 | SL hit (close>static) qty=0.50 sl=1233.17 alert=retest2 |

### Cycle 2 — BUY (started 2024-11-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-04 11:15:00 | 1325.00 | 1246.24 | 1245.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-06 09:15:00 | 1361.80 | 1253.99 | 1249.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-13 09:15:00 | 1261.95 | 1285.39 | 1267.80 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-13 09:15:00 | 1261.95 | 1285.39 | 1267.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-13 09:15:00 | 1261.95 | 1285.39 | 1267.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-13 09:30:00 | 1256.45 | 1285.39 | 1267.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-13 10:15:00 | 1272.65 | 1285.26 | 1267.82 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-18 12:15:00 | 1293.00 | 1281.89 | 1267.33 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-18 13:45:00 | 1289.10 | 1281.98 | 1267.52 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-19 09:15:00 | 1309.30 | 1281.92 | 1267.64 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-22 09:15:00 | 1303.05 | 1283.04 | 1269.21 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-22 09:15:00 | 1289.05 | 1283.10 | 1269.30 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-25 09:15:00 | 1322.70 | 1283.07 | 1269.70 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-25 10:15:00 | 1313.25 | 1283.28 | 1269.87 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-25 11:30:00 | 1305.15 | 1283.76 | 1270.24 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-25 12:00:00 | 1308.40 | 1283.76 | 1270.24 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-12 09:15:00 | 1292.00 | 1314.72 | 1294.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-12 09:45:00 | 1291.00 | 1314.72 | 1294.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-12 10:15:00 | 1290.70 | 1314.49 | 1293.99 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-12 12:45:00 | 1297.05 | 1314.06 | 1293.98 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-12 13:30:00 | 1297.50 | 1314.07 | 1294.09 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-17 11:00:00 | 1297.70 | 1314.04 | 1295.80 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-17 12:00:00 | 1297.35 | 1313.87 | 1295.81 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-17 12:15:00 | 1303.00 | 1313.77 | 1295.84 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-12-18 09:15:00 | 1285.85 | 1312.83 | 1295.73 | SL hit (close<static) qty=1.00 sl=1287.40 alert=retest2 |

### Cycle 3 — SELL (started 2024-12-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-27 11:15:00 | 1193.00 | 1282.04 | 1282.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-30 10:15:00 | 1189.40 | 1277.24 | 1279.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-21 09:15:00 | 809.45 | 766.41 | 873.92 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-21 10:00:00 | 809.45 | 766.41 | 873.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-03 09:15:00 | 855.60 | 777.12 | 853.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-03 10:00:00 | 855.60 | 777.12 | 853.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-03 10:15:00 | 898.95 | 778.33 | 854.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-03 11:00:00 | 898.95 | 778.33 | 854.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-03 11:15:00 | 872.50 | 779.27 | 854.18 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-03 12:15:00 | 868.70 | 779.27 | 854.18 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-03 13:00:00 | 870.95 | 780.18 | 854.27 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-04 09:15:00 | 867.80 | 783.00 | 854.58 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-04 14:15:00 | 827.40 | 786.27 | 854.13 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-04-04 14:15:00 | 838.00 | 786.27 | 854.13 | SL hit (close>static) qty=0.50 sl=786.27 alert=retest2 |

### Cycle 4 — BUY (started 2026-03-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 14:15:00 | 451.35 | 411.09 | 410.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-06 10:15:00 | 478.55 | 419.76 | 417.75 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-07-24 09:15:00 | 1329.00 | 2024-07-31 13:15:00 | 1253.85 | STOP_HIT | 1.00 | -5.65% |
| BUY | retest2 | 2024-08-07 12:30:00 | 1289.30 | 2024-08-08 12:15:00 | 1256.25 | STOP_HIT | 1.00 | -2.56% |
| BUY | retest2 | 2024-08-07 13:00:00 | 1283.25 | 2024-08-08 12:15:00 | 1256.25 | STOP_HIT | 1.00 | -2.10% |
| BUY | retest2 | 2024-08-07 15:00:00 | 1282.85 | 2024-08-08 12:15:00 | 1256.25 | STOP_HIT | 1.00 | -2.07% |
| BUY | retest2 | 2024-08-21 10:15:00 | 1282.20 | 2024-08-27 09:15:00 | 1251.10 | STOP_HIT | 1.00 | -2.43% |
| BUY | retest2 | 2024-08-21 13:00:00 | 1274.35 | 2024-08-27 09:15:00 | 1251.10 | STOP_HIT | 1.00 | -1.82% |
| BUY | retest2 | 2024-08-27 13:00:00 | 1274.00 | 2024-09-11 13:15:00 | 1262.00 | STOP_HIT | 1.00 | -0.94% |
| BUY | retest2 | 2024-09-11 09:30:00 | 1275.00 | 2024-09-12 13:15:00 | 1262.00 | STOP_HIT | 1.00 | -1.02% |
| BUY | retest2 | 2024-09-11 13:15:00 | 1277.70 | 2024-09-12 13:15:00 | 1262.00 | STOP_HIT | 1.00 | -1.23% |
| BUY | retest2 | 2024-09-12 09:15:00 | 1287.30 | 2024-09-13 11:15:00 | 1262.00 | STOP_HIT | 1.00 | -1.97% |
| BUY | retest2 | 2024-09-12 11:30:00 | 1284.00 | 2024-09-17 10:15:00 | 1257.05 | STOP_HIT | 1.00 | -2.10% |
| BUY | retest2 | 2024-09-13 09:15:00 | 1277.50 | 2024-09-17 10:15:00 | 1257.05 | STOP_HIT | 1.00 | -1.60% |
| SELL | retest2 | 2024-10-21 11:15:00 | 1298.25 | 2024-10-25 09:15:00 | 1268.63 | PARTIAL | 0.50 | 2.28% |
| SELL | retest2 | 2024-10-21 11:15:00 | 1298.25 | 2024-10-25 09:15:00 | 1256.25 | STOP_HIT | 0.50 | 3.24% |
| SELL | retest2 | 2024-10-23 12:15:00 | 1335.40 | 2024-10-25 09:15:00 | 1269.25 | PARTIAL | 0.50 | 4.95% |
| SELL | retest2 | 2024-10-23 12:15:00 | 1335.40 | 2024-10-25 09:15:00 | 1256.25 | STOP_HIT | 0.50 | 5.93% |
| SELL | retest2 | 2024-10-23 13:00:00 | 1336.05 | 2024-10-25 09:15:00 | 1261.41 | PARTIAL | 0.50 | 5.59% |
| SELL | retest2 | 2024-10-23 13:00:00 | 1336.05 | 2024-10-25 09:15:00 | 1256.25 | STOP_HIT | 0.50 | 5.97% |
| SELL | retest2 | 2024-10-23 14:00:00 | 1327.80 | 2024-10-28 09:15:00 | 1233.34 | PARTIAL | 0.50 | 7.11% |
| SELL | retest2 | 2024-10-23 14:00:00 | 1327.80 | 2024-10-28 09:15:00 | 1234.95 | STOP_HIT | 0.50 | 6.99% |
| BUY | retest2 | 2024-11-18 12:15:00 | 1293.00 | 2024-12-18 09:15:00 | 1285.85 | STOP_HIT | 1.00 | -0.55% |
| BUY | retest2 | 2024-11-18 13:45:00 | 1289.10 | 2024-12-18 09:15:00 | 1285.85 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest2 | 2024-11-19 09:15:00 | 1309.30 | 2024-12-18 09:15:00 | 1285.85 | STOP_HIT | 1.00 | -1.79% |
| BUY | retest2 | 2024-11-22 09:15:00 | 1303.05 | 2024-12-18 09:15:00 | 1285.85 | STOP_HIT | 1.00 | -1.32% |
| BUY | retest2 | 2024-11-25 09:15:00 | 1322.70 | 2024-12-18 12:15:00 | 1268.10 | STOP_HIT | 1.00 | -4.13% |
| BUY | retest2 | 2024-11-25 10:15:00 | 1313.25 | 2024-12-18 12:15:00 | 1268.10 | STOP_HIT | 1.00 | -3.44% |
| BUY | retest2 | 2024-11-25 11:30:00 | 1305.15 | 2024-12-18 12:15:00 | 1268.10 | STOP_HIT | 1.00 | -2.84% |
| BUY | retest2 | 2024-11-25 12:00:00 | 1308.40 | 2024-12-18 12:15:00 | 1268.10 | STOP_HIT | 1.00 | -3.08% |
| BUY | retest2 | 2024-12-12 12:45:00 | 1297.05 | 2024-12-18 14:15:00 | 1254.90 | STOP_HIT | 1.00 | -3.25% |
| BUY | retest2 | 2024-12-12 13:30:00 | 1297.50 | 2024-12-18 14:15:00 | 1254.90 | STOP_HIT | 1.00 | -3.28% |
| BUY | retest2 | 2024-12-17 11:00:00 | 1297.70 | 2024-12-18 14:15:00 | 1254.90 | STOP_HIT | 1.00 | -3.30% |
| BUY | retest2 | 2024-12-17 12:00:00 | 1297.35 | 2024-12-18 14:15:00 | 1254.90 | STOP_HIT | 1.00 | -3.27% |
| SELL | retest2 | 2025-04-03 12:15:00 | 868.70 | 2025-04-04 14:15:00 | 827.40 | PARTIAL | 0.50 | 4.75% |
| SELL | retest2 | 2025-04-03 12:15:00 | 868.70 | 2025-04-04 14:15:00 | 838.00 | STOP_HIT | 0.50 | 3.53% |
| SELL | retest2 | 2025-04-03 13:00:00 | 870.95 | 2025-04-07 09:15:00 | 781.83 | TARGET_HIT | 1.00 | 10.23% |
| SELL | retest2 | 2025-04-04 09:15:00 | 867.80 | 2025-04-07 09:15:00 | 781.02 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-04-16 10:00:00 | 868.00 | 2025-04-22 09:15:00 | 904.00 | STOP_HIT | 1.00 | -4.15% |
| SELL | retest2 | 2025-04-21 13:30:00 | 856.15 | 2025-04-22 09:15:00 | 904.00 | STOP_HIT | 1.00 | -5.59% |
| SELL | retest2 | 2025-04-21 15:00:00 | 856.70 | 2025-04-28 09:15:00 | 781.20 | TARGET_HIT | 1.00 | 8.81% |
| SELL | retest2 | 2025-04-25 09:45:00 | 854.15 | 2025-04-28 09:15:00 | 768.74 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-04-25 15:15:00 | 857.35 | 2025-04-28 09:15:00 | 771.62 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-09-29 10:45:00 | 595.50 | 2025-10-20 09:15:00 | 565.73 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-29 13:30:00 | 595.65 | 2025-10-20 09:15:00 | 565.87 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-07 09:15:00 | 596.80 | 2025-10-20 09:15:00 | 566.96 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-07 10:00:00 | 595.80 | 2025-10-20 09:15:00 | 566.01 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-29 10:45:00 | 595.50 | 2025-10-20 13:15:00 | 537.12 | TARGET_HIT | 0.50 | 9.80% |
| SELL | retest2 | 2025-09-29 13:30:00 | 595.65 | 2025-10-24 13:15:00 | 535.95 | TARGET_HIT | 0.50 | 10.02% |
| SELL | retest2 | 2025-10-07 09:15:00 | 596.80 | 2025-10-24 13:15:00 | 536.09 | TARGET_HIT | 0.50 | 10.17% |
| SELL | retest2 | 2025-10-07 10:00:00 | 595.80 | 2025-10-24 13:15:00 | 536.22 | TARGET_HIT | 0.50 | 10.00% |
