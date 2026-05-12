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
| CROSSOVER | 8 |
| ALERT1 | 8 |
| ALERT2 | 7 |
| ALERT2_SKIP | 3 |
| ALERT3 | 55 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 57 |
| PARTIAL | 9 |
| TARGET_HIT | 9 |
| STOP_HIT | 48 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 66 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 23 / 43
- **Target hits / Stop hits / Partials:** 9 / 48 / 9
- **Avg / median % per leg:** 0.61% / -1.97%
- **Sum % (uncompounded):** 40.53%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 29 | 0 | 0.0% | 0 | 29 | 0 | -2.58% | -74.9% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 29 | 0 | 0.0% | 0 | 29 | 0 | -2.58% | -74.9% |
| SELL (all) | 37 | 23 | 62.2% | 9 | 19 | 9 | 3.12% | 115.4% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 37 | 23 | 62.2% | 9 | 19 | 9 | 3.12% | 115.4% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 66 | 23 | 34.8% | 9 | 48 | 9 | 0.61% | 40.5% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-11-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-28 10:15:00 | 805.30 | 843.72 | 843.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-28 11:15:00 | 803.00 | 843.32 | 843.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-08 11:15:00 | 841.90 | 831.85 | 836.86 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-08 11:15:00 | 841.90 | 831.85 | 836.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-08 11:15:00 | 841.90 | 831.85 | 836.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-08 11:30:00 | 845.60 | 831.85 | 836.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-08 12:15:00 | 837.40 | 831.91 | 836.87 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-12-08 13:15:00 | 831.65 | 831.91 | 836.87 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-12-08 15:00:00 | 833.10 | 831.94 | 836.84 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-12-11 09:30:00 | 831.95 | 831.90 | 836.77 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-12-18 10:15:00 | 832.00 | 828.59 | 834.19 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-18 10:15:00 | 834.55 | 828.65 | 834.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-18 10:45:00 | 833.00 | 828.65 | 834.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-18 11:15:00 | 838.00 | 828.74 | 834.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-18 12:00:00 | 838.00 | 828.74 | 834.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-18 12:15:00 | 838.30 | 828.84 | 834.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-18 13:00:00 | 838.30 | 828.84 | 834.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-19 10:15:00 | 831.25 | 829.16 | 834.26 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-12-19 11:15:00 | 828.40 | 829.16 | 834.26 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-12-19 14:15:00 | 826.50 | 829.21 | 834.21 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-12-20 09:15:00 | 852.10 | 829.45 | 834.25 | SL hit (close>static) qty=1.00 sl=842.40 alert=retest2 |

### Cycle 2 — BUY (started 2023-12-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-28 11:15:00 | 861.45 | 838.10 | 838.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-29 11:15:00 | 865.00 | 839.73 | 838.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-16 12:15:00 | 849.10 | 852.40 | 846.95 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-01-16 13:00:00 | 849.10 | 852.40 | 846.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-16 15:15:00 | 848.40 | 852.31 | 846.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-17 09:15:00 | 845.85 | 852.31 | 846.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-17 09:15:00 | 843.55 | 852.22 | 846.97 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-17 10:45:00 | 852.45 | 852.28 | 847.03 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-18 12:15:00 | 854.60 | 852.21 | 847.20 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-18 12:45:00 | 851.05 | 852.16 | 847.20 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-18 15:00:00 | 849.95 | 852.10 | 847.22 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-18 15:15:00 | 848.35 | 852.06 | 847.22 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-19 09:45:00 | 852.35 | 852.07 | 847.25 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-01-20 09:15:00 | 819.90 | 852.47 | 847.62 | SL hit (close<static) qty=1.00 sl=836.50 alert=retest2 |

### Cycle 3 — SELL (started 2024-01-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-24 10:15:00 | 771.80 | 842.94 | 843.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-29 09:15:00 | 762.90 | 835.00 | 838.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-04 13:15:00 | 762.35 | 758.64 | 782.16 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-03-04 13:45:00 | 761.55 | 758.64 | 782.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-07 14:15:00 | 768.20 | 755.24 | 777.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-07 14:30:00 | 773.65 | 755.24 | 777.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-02 14:15:00 | 756.65 | 717.84 | 745.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-02 14:30:00 | 761.90 | 717.84 | 745.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-02 15:15:00 | 758.00 | 718.24 | 745.53 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-03 09:15:00 | 743.20 | 718.24 | 745.53 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-03 12:30:00 | 754.50 | 719.69 | 745.72 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-03 15:15:00 | 753.00 | 720.44 | 745.84 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-04 09:45:00 | 753.10 | 721.10 | 745.92 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-04-04 11:15:00 | 770.60 | 721.95 | 746.10 | SL hit (close>static) qty=1.00 sl=762.80 alert=retest2 |

### Cycle 4 — BUY (started 2024-04-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-23 09:15:00 | 1088.25 | 764.22 | 763.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-24 13:15:00 | 1090.70 | 797.28 | 780.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-04 10:15:00 | 1052.35 | 1098.74 | 1004.22 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-04 11:00:00 | 1052.35 | 1098.74 | 1004.22 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 09:15:00 | 1040.50 | 1096.10 | 1005.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-05 09:45:00 | 1025.45 | 1096.10 | 1005.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 10:15:00 | 1297.30 | 1366.66 | 1267.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-23 11:00:00 | 1297.30 | 1366.66 | 1267.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 12:15:00 | 1265.00 | 1364.95 | 1267.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-23 12:30:00 | 1270.05 | 1364.95 | 1267.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 13:15:00 | 1281.95 | 1364.12 | 1267.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-23 13:45:00 | 1280.55 | 1364.12 | 1267.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 14:15:00 | 1269.45 | 1363.18 | 1267.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-23 15:00:00 | 1269.45 | 1363.18 | 1267.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 15:15:00 | 1280.00 | 1362.35 | 1267.35 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-24 09:15:00 | 1329.00 | 1362.35 | 1267.35 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-31 13:15:00 | 1253.85 | 1341.03 | 1272.98 | SL hit (close<static) qty=1.00 sl=1260.10 alert=retest2 |

### Cycle 5 — SELL (started 2024-09-27 11:15:00)

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

### Cycle 6 — BUY (started 2024-11-04 11:15:00)

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

### Cycle 7 — SELL (started 2024-12-27 11:15:00)

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

### Cycle 8 — BUY (started 2026-03-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 14:15:00 | 451.35 | 411.09 | 410.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-06 10:15:00 | 478.55 | 419.76 | 417.75 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2023-12-08 13:15:00 | 831.65 | 2023-12-20 09:15:00 | 852.10 | STOP_HIT | 1.00 | -2.46% |
| SELL | retest2 | 2023-12-08 15:00:00 | 833.10 | 2023-12-20 09:15:00 | 852.10 | STOP_HIT | 1.00 | -2.28% |
| SELL | retest2 | 2023-12-11 09:30:00 | 831.95 | 2023-12-20 09:15:00 | 852.10 | STOP_HIT | 1.00 | -2.42% |
| SELL | retest2 | 2023-12-18 10:15:00 | 832.00 | 2023-12-20 09:15:00 | 852.10 | STOP_HIT | 1.00 | -2.42% |
| SELL | retest2 | 2023-12-19 11:15:00 | 828.40 | 2023-12-20 09:15:00 | 852.10 | STOP_HIT | 1.00 | -2.86% |
| SELL | retest2 | 2023-12-19 14:15:00 | 826.50 | 2023-12-20 09:15:00 | 852.10 | STOP_HIT | 1.00 | -3.10% |
| SELL | retest2 | 2023-12-20 15:15:00 | 828.35 | 2023-12-21 09:15:00 | 845.10 | STOP_HIT | 1.00 | -2.02% |
| BUY | retest2 | 2024-01-17 10:45:00 | 852.45 | 2024-01-20 09:15:00 | 819.90 | STOP_HIT | 1.00 | -3.82% |
| BUY | retest2 | 2024-01-18 12:15:00 | 854.60 | 2024-01-20 09:15:00 | 819.90 | STOP_HIT | 1.00 | -4.06% |
| BUY | retest2 | 2024-01-18 12:45:00 | 851.05 | 2024-01-20 09:15:00 | 819.90 | STOP_HIT | 1.00 | -3.66% |
| BUY | retest2 | 2024-01-18 15:00:00 | 849.95 | 2024-01-20 09:15:00 | 819.90 | STOP_HIT | 1.00 | -3.54% |
| BUY | retest2 | 2024-01-19 09:45:00 | 852.35 | 2024-01-20 09:15:00 | 819.90 | STOP_HIT | 1.00 | -3.81% |
| SELL | retest2 | 2024-04-03 09:15:00 | 743.20 | 2024-04-04 11:15:00 | 770.60 | STOP_HIT | 1.00 | -3.69% |
| SELL | retest2 | 2024-04-03 12:30:00 | 754.50 | 2024-04-04 11:15:00 | 770.60 | STOP_HIT | 1.00 | -2.13% |
| SELL | retest2 | 2024-04-03 15:15:00 | 753.00 | 2024-04-04 11:15:00 | 770.60 | STOP_HIT | 1.00 | -2.34% |
| SELL | retest2 | 2024-04-04 09:45:00 | 753.10 | 2024-04-04 11:15:00 | 770.60 | STOP_HIT | 1.00 | -2.32% |
| SELL | retest2 | 2024-04-15 14:30:00 | 764.10 | 2024-04-16 09:15:00 | 811.50 | STOP_HIT | 1.00 | -6.20% |
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
