# DCM Shriram Ltd. (DCMSHRIRAM)

## Backtest Summary

- **Window:** 2025-03-13 09:15:00 → 2026-05-08 15:15:00 (1976 bars)
- **Last close:** 1237.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 78 |
| ALERT1 | 52 |
| ALERT2 | 49 |
| ALERT2_SKIP | 25 |
| ALERT3 | 120 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 1 |
| ENTRY2 | 51 |
| PARTIAL | 9 |
| TARGET_HIT | 8 |
| STOP_HIT | 44 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 61 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 31 / 30
- **Target hits / Stop hits / Partials:** 8 / 44 / 9
- **Avg / median % per leg:** 1.70% / 0.12%
- **Sum % (uncompounded):** 103.78%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 19 | 7 | 36.8% | 0 | 19 | 0 | 0.51% | 9.7% |
| BUY @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -0.99% | -1.0% |
| BUY @ 3rd Alert (retest2) | 18 | 7 | 38.9% | 0 | 18 | 0 | 0.59% | 10.7% |
| SELL (all) | 42 | 24 | 57.1% | 8 | 25 | 9 | 2.24% | 94.1% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 42 | 24 | 57.1% | 8 | 25 | 9 | 2.24% | 94.1% |
| retest1 (combined) | 1 | 0 | 0.0% | 0 | 1 | 0 | -0.99% | -1.0% |
| retest2 (combined) | 60 | 31 | 51.7% | 8 | 43 | 9 | 1.75% | 104.8% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 10:15:00 | 1030.50 | 1005.70 | 1003.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-13 10:15:00 | 1043.30 | 1026.54 | 1016.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-15 09:15:00 | 1041.00 | 1041.22 | 1034.19 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-15 09:15:00 | 1041.00 | 1041.22 | 1034.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-15 09:15:00 | 1041.00 | 1041.22 | 1034.19 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-19 10:15:00 | 1053.00 | 1043.02 | 1040.56 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-20 15:15:00 | 1055.00 | 1059.60 | 1055.14 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-23 14:15:00 | 1052.60 | 1058.50 | 1059.10 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 2 — SELL (started 2025-05-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-23 14:15:00 | 1052.60 | 1058.50 | 1059.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-26 10:15:00 | 1050.60 | 1056.43 | 1057.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-27 09:15:00 | 1053.30 | 1047.31 | 1051.22 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-27 09:15:00 | 1053.30 | 1047.31 | 1051.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 09:15:00 | 1053.30 | 1047.31 | 1051.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-27 09:30:00 | 1057.05 | 1047.31 | 1051.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 10:15:00 | 1034.50 | 1044.74 | 1049.70 | EMA400 retest candle locked (from downside) |

### Cycle 3 — BUY (started 2025-05-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-28 09:15:00 | 1054.85 | 1052.08 | 1051.94 | EMA200 above EMA400 |

### Cycle 4 — SELL (started 2025-05-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-28 11:15:00 | 1038.75 | 1050.15 | 1051.13 | EMA200 below EMA400 |

### Cycle 5 — BUY (started 2025-05-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-29 11:15:00 | 1061.85 | 1051.96 | 1051.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-30 11:15:00 | 1068.85 | 1061.03 | 1056.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-30 14:15:00 | 1059.45 | 1062.51 | 1058.73 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-30 15:00:00 | 1059.45 | 1062.51 | 1058.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 15:15:00 | 1085.00 | 1067.01 | 1061.12 | EMA400 retest candle locked (from upside) |

### Cycle 6 — SELL (started 2025-06-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-04 11:15:00 | 1062.10 | 1068.62 | 1068.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-04 13:15:00 | 1056.60 | 1064.92 | 1067.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-05 09:15:00 | 1060.20 | 1059.54 | 1063.71 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-05 09:30:00 | 1059.50 | 1059.54 | 1063.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 11:15:00 | 1061.40 | 1060.56 | 1063.49 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-05 12:45:00 | 1060.00 | 1060.31 | 1063.11 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-06 09:15:00 | 1082.20 | 1062.39 | 1062.88 | SL hit (close>static) qty=1.00 sl=1066.40 alert=retest2 |

### Cycle 7 — BUY (started 2025-06-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-06 10:15:00 | 1081.60 | 1066.23 | 1064.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-10 09:15:00 | 1086.50 | 1074.71 | 1070.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-12 09:15:00 | 1096.60 | 1101.25 | 1093.04 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-12 09:15:00 | 1096.60 | 1101.25 | 1093.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 09:15:00 | 1096.60 | 1101.25 | 1093.04 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-12 11:45:00 | 1114.50 | 1103.82 | 1095.64 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-16 10:45:00 | 1115.30 | 1121.44 | 1115.70 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-27 15:15:00 | 1190.00 | 1193.85 | 1194.17 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 8 — SELL (started 2025-06-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-27 15:15:00 | 1190.00 | 1193.85 | 1194.17 | EMA200 below EMA400 |

### Cycle 9 — BUY (started 2025-06-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-30 09:15:00 | 1226.40 | 1200.36 | 1197.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-30 12:15:00 | 1248.70 | 1217.45 | 1206.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-01 09:15:00 | 1216.10 | 1223.53 | 1213.52 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-01 10:00:00 | 1216.10 | 1223.53 | 1213.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 10:15:00 | 1215.70 | 1221.96 | 1213.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-01 10:45:00 | 1211.10 | 1221.96 | 1213.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 11:15:00 | 1211.60 | 1219.89 | 1213.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-01 11:45:00 | 1209.00 | 1219.89 | 1213.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 12:15:00 | 1208.40 | 1217.59 | 1213.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-01 12:45:00 | 1208.80 | 1217.59 | 1213.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 09:15:00 | 1209.20 | 1215.77 | 1213.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-02 10:00:00 | 1209.20 | 1215.77 | 1213.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 10:15:00 | 1217.40 | 1216.10 | 1213.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-02 10:45:00 | 1220.70 | 1216.10 | 1213.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 13:15:00 | 1236.50 | 1221.82 | 1217.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-02 13:30:00 | 1223.20 | 1221.82 | 1217.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 10:15:00 | 1380.50 | 1392.76 | 1382.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-10 11:00:00 | 1380.50 | 1392.76 | 1382.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 11:15:00 | 1377.00 | 1389.61 | 1381.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-10 12:00:00 | 1377.00 | 1389.61 | 1381.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 12:15:00 | 1379.60 | 1387.61 | 1381.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-10 13:00:00 | 1379.60 | 1387.61 | 1381.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 13:15:00 | 1368.20 | 1383.72 | 1380.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-10 14:00:00 | 1368.20 | 1383.72 | 1380.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 10 — SELL (started 2025-07-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-10 15:15:00 | 1359.00 | 1374.55 | 1376.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-11 10:15:00 | 1350.00 | 1367.03 | 1372.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-14 13:15:00 | 1359.70 | 1354.20 | 1360.34 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-14 13:15:00 | 1359.70 | 1354.20 | 1360.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 13:15:00 | 1359.70 | 1354.20 | 1360.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-14 14:00:00 | 1359.70 | 1354.20 | 1360.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 14:15:00 | 1366.00 | 1356.56 | 1360.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-14 15:00:00 | 1366.00 | 1356.56 | 1360.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 15:15:00 | 1365.00 | 1358.25 | 1361.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-15 09:15:00 | 1379.70 | 1358.25 | 1361.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 11 — BUY (started 2025-07-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-15 09:15:00 | 1406.00 | 1367.80 | 1365.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-16 09:15:00 | 1419.90 | 1397.48 | 1383.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-16 15:15:00 | 1409.00 | 1409.32 | 1397.07 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-17 09:15:00 | 1413.00 | 1409.32 | 1397.07 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 15:15:00 | 1399.00 | 1406.44 | 1401.46 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-07-17 15:15:00 | 1399.00 | 1406.44 | 1401.46 | SL hit (close<ema400) qty=1.00 sl=1401.46 alert=retest1 |

### Cycle 12 — SELL (started 2025-07-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-21 14:15:00 | 1392.10 | 1402.98 | 1403.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-21 15:15:00 | 1382.00 | 1398.79 | 1401.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-22 12:15:00 | 1413.80 | 1396.26 | 1398.52 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-22 12:15:00 | 1413.80 | 1396.26 | 1398.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 12:15:00 | 1413.80 | 1396.26 | 1398.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-22 13:00:00 | 1413.80 | 1396.26 | 1398.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 13:15:00 | 1401.50 | 1397.31 | 1398.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-22 13:30:00 | 1403.70 | 1397.31 | 1398.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 14:15:00 | 1398.70 | 1397.59 | 1398.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-22 14:45:00 | 1405.70 | 1397.59 | 1398.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 15:15:00 | 1396.00 | 1397.27 | 1398.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-23 09:15:00 | 1385.20 | 1397.27 | 1398.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 13 — BUY (started 2025-07-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-23 09:15:00 | 1413.10 | 1400.44 | 1399.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-23 14:15:00 | 1438.60 | 1413.98 | 1407.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-24 11:15:00 | 1427.60 | 1428.96 | 1417.83 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-24 11:45:00 | 1428.90 | 1428.96 | 1417.83 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 14:15:00 | 1429.40 | 1430.50 | 1421.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-24 14:45:00 | 1433.20 | 1430.50 | 1421.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 09:15:00 | 1402.70 | 1425.18 | 1420.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-25 09:45:00 | 1404.60 | 1425.18 | 1420.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 14 — SELL (started 2025-07-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-25 11:15:00 | 1392.80 | 1414.58 | 1416.34 | EMA200 below EMA400 |

### Cycle 15 — BUY (started 2025-07-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-29 13:15:00 | 1418.90 | 1400.63 | 1399.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-29 15:15:00 | 1425.00 | 1408.44 | 1403.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-31 14:15:00 | 1426.30 | 1435.55 | 1425.69 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-31 14:15:00 | 1426.30 | 1435.55 | 1425.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 14:15:00 | 1426.30 | 1435.55 | 1425.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-31 15:00:00 | 1426.30 | 1435.55 | 1425.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 15:15:00 | 1427.10 | 1433.86 | 1425.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-01 09:15:00 | 1423.80 | 1433.86 | 1425.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 09:15:00 | 1419.50 | 1430.99 | 1425.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-01 10:00:00 | 1419.50 | 1430.99 | 1425.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 10:15:00 | 1409.40 | 1426.67 | 1423.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-01 11:00:00 | 1409.40 | 1426.67 | 1423.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 16 — SELL (started 2025-08-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-01 12:15:00 | 1418.00 | 1421.45 | 1421.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-01 14:15:00 | 1371.70 | 1409.11 | 1415.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-05 12:15:00 | 1380.00 | 1379.97 | 1390.10 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-05 12:45:00 | 1382.10 | 1379.97 | 1390.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 15:15:00 | 1382.00 | 1380.62 | 1387.89 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-06 09:15:00 | 1367.10 | 1380.62 | 1387.89 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-06 10:30:00 | 1377.00 | 1380.42 | 1386.57 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-06 12:15:00 | 1378.80 | 1380.36 | 1385.99 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-06 14:00:00 | 1379.80 | 1380.36 | 1385.02 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 09:15:00 | 1334.40 | 1346.55 | 1360.59 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-08 10:30:00 | 1330.60 | 1343.84 | 1358.08 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-08 11:15:00 | 1327.60 | 1343.84 | 1358.08 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-11 09:30:00 | 1332.00 | 1347.99 | 1354.07 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-11 14:15:00 | 1365.40 | 1356.70 | 1356.40 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 17 — BUY (started 2025-08-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-11 14:15:00 | 1365.40 | 1356.70 | 1356.40 | EMA200 above EMA400 |

### Cycle 18 — SELL (started 2025-08-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-12 12:15:00 | 1354.10 | 1356.23 | 1356.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-12 13:15:00 | 1347.10 | 1354.40 | 1355.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-19 10:15:00 | 1280.00 | 1279.03 | 1294.18 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-19 10:30:00 | 1283.90 | 1279.03 | 1294.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 12:15:00 | 1268.10 | 1263.48 | 1275.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-20 13:00:00 | 1268.10 | 1263.48 | 1275.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 15:15:00 | 1258.20 | 1262.91 | 1272.63 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-25 09:15:00 | 1249.60 | 1269.21 | 1271.38 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-02 14:15:00 | 1241.70 | 1239.29 | 1239.28 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 19 — BUY (started 2025-09-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-02 14:15:00 | 1241.70 | 1239.29 | 1239.28 | EMA200 above EMA400 |

### Cycle 20 — SELL (started 2025-09-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-02 15:15:00 | 1237.20 | 1238.87 | 1239.09 | EMA200 below EMA400 |

### Cycle 21 — BUY (started 2025-09-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-03 09:15:00 | 1244.60 | 1240.02 | 1239.59 | EMA200 above EMA400 |

### Cycle 22 — SELL (started 2025-09-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-03 13:15:00 | 1235.00 | 1238.84 | 1239.24 | EMA200 below EMA400 |

### Cycle 23 — BUY (started 2025-09-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-04 09:15:00 | 1245.20 | 1240.12 | 1239.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-04 15:15:00 | 1252.00 | 1247.38 | 1244.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-05 09:15:00 | 1239.30 | 1245.77 | 1243.60 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-05 09:15:00 | 1239.30 | 1245.77 | 1243.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 09:15:00 | 1239.30 | 1245.77 | 1243.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-05 10:00:00 | 1239.30 | 1245.77 | 1243.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 10:15:00 | 1242.60 | 1245.13 | 1243.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-05 10:30:00 | 1246.50 | 1245.13 | 1243.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 11:15:00 | 1238.50 | 1243.81 | 1243.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-05 12:00:00 | 1238.50 | 1243.81 | 1243.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 12:15:00 | 1243.00 | 1243.65 | 1243.05 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-05 15:15:00 | 1248.30 | 1243.25 | 1242.96 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-08 10:15:00 | 1238.30 | 1242.08 | 1242.54 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 24 — SELL (started 2025-09-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-08 10:15:00 | 1238.30 | 1242.08 | 1242.54 | EMA200 below EMA400 |

### Cycle 25 — BUY (started 2025-09-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-08 12:15:00 | 1245.00 | 1243.20 | 1243.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-09 09:15:00 | 1251.10 | 1245.60 | 1244.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-09 14:15:00 | 1234.80 | 1245.26 | 1244.92 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-09 14:15:00 | 1234.80 | 1245.26 | 1244.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 14:15:00 | 1234.80 | 1245.26 | 1244.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-09 15:00:00 | 1234.80 | 1245.26 | 1244.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 26 — SELL (started 2025-09-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-09 15:15:00 | 1236.90 | 1243.59 | 1244.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-10 12:15:00 | 1232.80 | 1240.49 | 1242.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-11 09:15:00 | 1243.70 | 1237.66 | 1240.17 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-11 09:15:00 | 1243.70 | 1237.66 | 1240.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 09:15:00 | 1243.70 | 1237.66 | 1240.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-11 09:30:00 | 1244.70 | 1237.66 | 1240.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 10:15:00 | 1242.50 | 1238.62 | 1240.38 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-11 11:30:00 | 1240.00 | 1241.50 | 1241.53 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-11 12:15:00 | 1248.10 | 1242.82 | 1242.13 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 27 — BUY (started 2025-09-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-11 12:15:00 | 1248.10 | 1242.82 | 1242.13 | EMA200 above EMA400 |

### Cycle 28 — SELL (started 2025-09-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-11 14:15:00 | 1232.60 | 1241.44 | 1241.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-12 14:15:00 | 1224.40 | 1234.53 | 1237.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-16 09:15:00 | 1226.20 | 1225.40 | 1230.38 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-16 09:15:00 | 1226.20 | 1225.40 | 1230.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 09:15:00 | 1226.20 | 1225.40 | 1230.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-16 09:30:00 | 1226.30 | 1225.40 | 1230.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 09:15:00 | 1235.10 | 1227.60 | 1229.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-17 10:00:00 | 1235.10 | 1227.60 | 1229.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 29 — BUY (started 2025-09-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-17 10:15:00 | 1240.00 | 1230.08 | 1230.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-17 11:15:00 | 1255.30 | 1235.12 | 1232.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-19 09:15:00 | 1278.10 | 1305.68 | 1289.31 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-19 09:15:00 | 1278.10 | 1305.68 | 1289.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 09:15:00 | 1278.10 | 1305.68 | 1289.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-19 09:30:00 | 1280.80 | 1305.68 | 1289.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 10:15:00 | 1277.00 | 1299.94 | 1288.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-19 11:00:00 | 1277.00 | 1299.94 | 1288.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 30 — SELL (started 2025-09-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-19 14:15:00 | 1253.10 | 1279.06 | 1280.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-22 10:15:00 | 1248.50 | 1265.50 | 1273.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-23 13:15:00 | 1241.40 | 1239.54 | 1251.28 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-23 14:00:00 | 1241.40 | 1239.54 | 1251.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 14:15:00 | 1209.80 | 1233.59 | 1247.51 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-23 15:15:00 | 1206.00 | 1233.59 | 1247.51 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-24 13:15:00 | 1192.70 | 1221.75 | 1235.38 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-26 09:15:00 | 1145.70 | 1175.88 | 1197.86 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-29 09:15:00 | 1158.90 | 1156.95 | 1174.79 | SL hit (close>ema200) qty=0.50 sl=1156.95 alert=retest2 |

### Cycle 31 — BUY (started 2025-10-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-01 15:15:00 | 1170.10 | 1160.13 | 1159.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-03 09:15:00 | 1183.20 | 1164.74 | 1161.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-06 14:15:00 | 1189.80 | 1192.65 | 1183.84 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-06 15:00:00 | 1189.80 | 1192.65 | 1183.84 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 12:15:00 | 1191.20 | 1191.56 | 1186.56 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-08 09:15:00 | 1193.00 | 1191.09 | 1187.56 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-08 10:15:00 | 1183.50 | 1189.22 | 1187.30 | SL hit (close<static) qty=1.00 sl=1185.70 alert=retest2 |

### Cycle 32 — SELL (started 2025-10-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-08 14:15:00 | 1171.10 | 1183.66 | 1185.12 | EMA200 below EMA400 |

### Cycle 33 — BUY (started 2025-10-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-09 13:15:00 | 1193.50 | 1186.07 | 1185.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-10 09:15:00 | 1199.50 | 1190.36 | 1187.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-13 09:15:00 | 1177.20 | 1192.46 | 1190.87 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-13 09:15:00 | 1177.20 | 1192.46 | 1190.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 09:15:00 | 1177.20 | 1192.46 | 1190.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-13 10:15:00 | 1175.40 | 1192.46 | 1190.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 34 — SELL (started 2025-10-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-13 10:15:00 | 1175.00 | 1188.97 | 1189.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-13 12:15:00 | 1165.20 | 1181.69 | 1185.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-15 09:15:00 | 1172.00 | 1168.27 | 1173.26 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-15 09:15:00 | 1172.00 | 1168.27 | 1173.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 09:15:00 | 1172.00 | 1168.27 | 1173.26 | EMA400 retest candle locked (from downside) |

### Cycle 35 — BUY (started 2025-10-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-15 14:15:00 | 1201.20 | 1176.42 | 1175.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-15 15:15:00 | 1211.00 | 1183.33 | 1178.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-21 14:15:00 | 1233.90 | 1236.57 | 1226.42 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-21 14:30:00 | 1233.90 | 1236.57 | 1226.42 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 09:15:00 | 1241.50 | 1237.56 | 1227.79 | EMA400 retest candle locked (from upside) |

### Cycle 36 — SELL (started 2025-10-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-27 09:15:00 | 1219.00 | 1230.48 | 1231.44 | EMA200 below EMA400 |

### Cycle 37 — BUY (started 2025-10-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-27 11:15:00 | 1244.80 | 1233.64 | 1232.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-27 13:15:00 | 1275.90 | 1243.38 | 1237.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-29 14:15:00 | 1315.00 | 1326.50 | 1301.36 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-29 15:00:00 | 1315.00 | 1326.50 | 1301.36 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 09:15:00 | 1310.80 | 1322.05 | 1313.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-31 09:45:00 | 1313.20 | 1322.05 | 1313.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 10:15:00 | 1294.70 | 1316.58 | 1312.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-31 11:00:00 | 1294.70 | 1316.58 | 1312.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 11:15:00 | 1294.50 | 1312.16 | 1310.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-31 11:45:00 | 1293.10 | 1312.16 | 1310.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 38 — SELL (started 2025-10-31 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-31 12:15:00 | 1295.70 | 1308.87 | 1309.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-31 14:15:00 | 1275.50 | 1300.55 | 1305.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-03 15:15:00 | 1281.00 | 1279.08 | 1289.00 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-04 09:15:00 | 1266.10 | 1279.08 | 1289.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 13:15:00 | 1252.40 | 1233.68 | 1243.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-07 14:00:00 | 1252.40 | 1233.68 | 1243.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 14:15:00 | 1242.70 | 1235.48 | 1243.03 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-10 09:15:00 | 1231.00 | 1239.19 | 1244.03 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-10 12:15:00 | 1255.80 | 1248.14 | 1247.18 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 39 — BUY (started 2025-11-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-10 12:15:00 | 1255.80 | 1248.14 | 1247.18 | EMA200 above EMA400 |

### Cycle 40 — SELL (started 2025-11-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-11 10:15:00 | 1240.90 | 1246.44 | 1246.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-11 14:15:00 | 1235.30 | 1242.16 | 1244.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-12 09:15:00 | 1242.10 | 1241.33 | 1243.65 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-12 09:15:00 | 1242.10 | 1241.33 | 1243.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 09:15:00 | 1242.10 | 1241.33 | 1243.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-12 09:45:00 | 1244.20 | 1241.33 | 1243.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 10:15:00 | 1244.00 | 1241.86 | 1243.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-12 10:30:00 | 1244.30 | 1241.86 | 1243.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 11:15:00 | 1240.20 | 1241.53 | 1243.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-12 11:30:00 | 1243.00 | 1241.53 | 1243.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 09:15:00 | 1239.80 | 1239.29 | 1241.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-13 10:15:00 | 1242.60 | 1239.29 | 1241.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 10:15:00 | 1240.90 | 1239.61 | 1241.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-13 11:00:00 | 1240.90 | 1239.61 | 1241.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 11:15:00 | 1237.80 | 1239.25 | 1240.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-13 11:30:00 | 1241.30 | 1239.25 | 1240.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 14:15:00 | 1236.20 | 1236.53 | 1239.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-13 14:45:00 | 1237.50 | 1236.53 | 1239.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 09:15:00 | 1232.00 | 1235.38 | 1238.16 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-17 10:15:00 | 1208.40 | 1237.29 | 1238.13 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-17 11:00:00 | 1224.30 | 1234.69 | 1236.87 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-21 10:15:00 | 1242.40 | 1191.85 | 1191.59 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 41 — BUY (started 2025-11-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-21 10:15:00 | 1242.40 | 1191.85 | 1191.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-21 11:15:00 | 1257.00 | 1204.88 | 1197.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-24 09:15:00 | 1223.40 | 1234.07 | 1217.72 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-24 09:15:00 | 1223.40 | 1234.07 | 1217.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 09:15:00 | 1223.40 | 1234.07 | 1217.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-24 09:45:00 | 1220.20 | 1234.07 | 1217.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 11:15:00 | 1215.30 | 1228.54 | 1217.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-24 11:45:00 | 1214.30 | 1228.54 | 1217.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 12:15:00 | 1206.90 | 1224.22 | 1216.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-24 12:45:00 | 1210.40 | 1224.22 | 1216.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 13:15:00 | 1200.20 | 1219.41 | 1215.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-24 14:00:00 | 1200.20 | 1219.41 | 1215.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 42 — SELL (started 2025-11-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-24 15:15:00 | 1198.00 | 1212.01 | 1212.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-25 09:15:00 | 1192.60 | 1208.13 | 1210.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-26 10:15:00 | 1204.00 | 1198.12 | 1202.66 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-26 10:15:00 | 1204.00 | 1198.12 | 1202.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 10:15:00 | 1204.00 | 1198.12 | 1202.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-26 10:45:00 | 1200.70 | 1198.12 | 1202.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 11:15:00 | 1208.50 | 1200.20 | 1203.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-26 12:00:00 | 1208.50 | 1200.20 | 1203.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 12:15:00 | 1203.50 | 1200.86 | 1203.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-26 13:15:00 | 1212.80 | 1200.86 | 1203.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 13:15:00 | 1206.00 | 1201.89 | 1203.47 | EMA400 retest candle locked (from downside) |

### Cycle 43 — BUY (started 2025-11-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-26 14:15:00 | 1266.50 | 1214.81 | 1209.20 | EMA200 above EMA400 |

### Cycle 44 — SELL (started 2025-12-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-01 13:15:00 | 1214.50 | 1218.21 | 1218.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-01 14:15:00 | 1206.70 | 1215.91 | 1217.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-02 10:15:00 | 1232.70 | 1216.57 | 1217.16 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-02 10:15:00 | 1232.70 | 1216.57 | 1217.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 10:15:00 | 1232.70 | 1216.57 | 1217.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-02 10:30:00 | 1225.70 | 1216.57 | 1217.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 11:15:00 | 1217.50 | 1216.76 | 1217.19 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-02 12:15:00 | 1215.00 | 1216.76 | 1217.19 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-02 12:15:00 | 1223.60 | 1218.13 | 1217.78 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 45 — BUY (started 2025-12-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-02 12:15:00 | 1223.60 | 1218.13 | 1217.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-04 12:15:00 | 1230.30 | 1224.61 | 1221.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-05 11:15:00 | 1227.20 | 1227.44 | 1224.79 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-05 12:00:00 | 1227.20 | 1227.44 | 1224.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 12:15:00 | 1230.00 | 1227.95 | 1225.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-05 12:30:00 | 1226.60 | 1227.95 | 1225.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 13:15:00 | 1222.00 | 1226.76 | 1224.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-05 14:00:00 | 1222.00 | 1226.76 | 1224.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 14:15:00 | 1221.10 | 1225.63 | 1224.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-05 14:30:00 | 1220.00 | 1225.63 | 1224.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 15:15:00 | 1221.90 | 1224.88 | 1224.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-08 09:15:00 | 1212.00 | 1224.88 | 1224.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 10:15:00 | 1225.40 | 1226.06 | 1225.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-08 11:00:00 | 1225.40 | 1226.06 | 1225.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 11:15:00 | 1227.90 | 1226.43 | 1225.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-08 12:00:00 | 1227.90 | 1226.43 | 1225.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 12:15:00 | 1227.20 | 1226.58 | 1225.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-08 12:30:00 | 1222.00 | 1226.58 | 1225.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 13:15:00 | 1226.40 | 1226.55 | 1225.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-08 14:00:00 | 1226.40 | 1226.55 | 1225.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 46 — SELL (started 2025-12-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-08 14:15:00 | 1202.00 | 1221.64 | 1223.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-08 15:15:00 | 1200.20 | 1217.35 | 1221.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-09 11:15:00 | 1217.30 | 1214.95 | 1218.98 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-09 12:00:00 | 1217.30 | 1214.95 | 1218.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 13:15:00 | 1221.30 | 1216.39 | 1218.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-09 13:45:00 | 1221.00 | 1216.39 | 1218.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 14:15:00 | 1228.00 | 1218.71 | 1219.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-09 15:00:00 | 1228.00 | 1218.71 | 1219.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 47 — BUY (started 2025-12-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-09 15:15:00 | 1229.00 | 1220.77 | 1220.60 | EMA200 above EMA400 |

### Cycle 48 — SELL (started 2025-12-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-10 12:15:00 | 1209.70 | 1220.01 | 1220.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-10 14:15:00 | 1208.80 | 1216.10 | 1218.56 | Break + close below crossover candle low |

### Cycle 49 — BUY (started 2025-12-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-11 09:15:00 | 1290.10 | 1229.89 | 1224.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-11 10:15:00 | 1307.60 | 1245.43 | 1231.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-11 15:15:00 | 1260.00 | 1268.86 | 1250.86 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-12 09:15:00 | 1255.00 | 1268.86 | 1250.86 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 09:15:00 | 1257.80 | 1266.65 | 1251.49 | EMA400 retest candle locked (from upside) |

### Cycle 50 — SELL (started 2025-12-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-16 10:15:00 | 1242.20 | 1250.27 | 1251.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-16 13:15:00 | 1240.30 | 1246.02 | 1248.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-18 11:15:00 | 1225.10 | 1224.92 | 1233.64 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-18 12:00:00 | 1225.10 | 1224.92 | 1233.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-22 09:15:00 | 1234.60 | 1217.97 | 1221.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-22 10:00:00 | 1234.60 | 1217.97 | 1221.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-22 10:15:00 | 1232.50 | 1220.88 | 1222.76 | EMA400 retest candle locked (from downside) |

### Cycle 51 — BUY (started 2025-12-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-22 12:15:00 | 1239.80 | 1227.07 | 1225.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-22 13:15:00 | 1245.90 | 1230.83 | 1227.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-24 11:15:00 | 1247.00 | 1247.79 | 1241.39 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-24 11:45:00 | 1246.00 | 1247.79 | 1241.39 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 11:15:00 | 1258.20 | 1255.55 | 1248.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-26 11:45:00 | 1252.90 | 1255.55 | 1248.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 14:15:00 | 1245.10 | 1254.06 | 1249.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-26 15:00:00 | 1245.10 | 1254.06 | 1249.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 15:15:00 | 1245.00 | 1252.25 | 1249.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-29 09:15:00 | 1233.70 | 1252.25 | 1249.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 10:15:00 | 1246.20 | 1247.96 | 1247.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-29 10:45:00 | 1244.10 | 1247.96 | 1247.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 52 — SELL (started 2025-12-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-29 11:15:00 | 1246.30 | 1247.63 | 1247.63 | EMA200 below EMA400 |

### Cycle 53 — BUY (started 2025-12-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-30 09:15:00 | 1253.30 | 1247.78 | 1247.57 | EMA200 above EMA400 |

### Cycle 54 — SELL (started 2025-12-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-30 15:15:00 | 1241.10 | 1247.24 | 1247.75 | EMA200 below EMA400 |

### Cycle 55 — BUY (started 2025-12-31 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-31 15:15:00 | 1250.90 | 1247.47 | 1247.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-01 09:15:00 | 1261.10 | 1250.19 | 1248.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-02 11:15:00 | 1262.50 | 1267.38 | 1261.04 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-02 11:15:00 | 1262.50 | 1267.38 | 1261.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 11:15:00 | 1262.50 | 1267.38 | 1261.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-02 12:00:00 | 1262.50 | 1267.38 | 1261.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 12:15:00 | 1257.90 | 1265.49 | 1260.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-02 13:00:00 | 1257.90 | 1265.49 | 1260.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 13:15:00 | 1256.80 | 1263.75 | 1260.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-02 13:45:00 | 1258.40 | 1263.75 | 1260.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 09:15:00 | 1249.40 | 1259.10 | 1258.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-05 09:30:00 | 1244.70 | 1259.10 | 1258.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 56 — SELL (started 2026-01-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-05 10:15:00 | 1251.00 | 1257.48 | 1258.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-06 09:15:00 | 1242.80 | 1252.94 | 1255.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-12 12:15:00 | 1179.00 | 1177.94 | 1190.15 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-12 13:00:00 | 1179.00 | 1177.94 | 1190.15 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 15:15:00 | 1198.00 | 1182.34 | 1189.13 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 09:30:00 | 1175.20 | 1183.45 | 1189.02 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 14:00:00 | 1175.10 | 1182.34 | 1186.91 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-16 12:30:00 | 1174.10 | 1179.22 | 1181.85 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-20 10:15:00 | 1116.44 | 1138.88 | 1154.35 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-20 10:15:00 | 1116.34 | 1138.88 | 1154.35 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-20 10:15:00 | 1115.39 | 1138.88 | 1154.35 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-01-21 10:15:00 | 1057.68 | 1098.59 | 1124.04 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 57 — BUY (started 2026-01-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-23 10:15:00 | 1132.00 | 1112.34 | 1111.54 | EMA200 above EMA400 |

### Cycle 58 — SELL (started 2026-01-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-23 15:15:00 | 1099.50 | 1110.51 | 1111.67 | EMA200 below EMA400 |

### Cycle 59 — BUY (started 2026-01-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-27 13:15:00 | 1121.40 | 1113.43 | 1112.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-27 14:15:00 | 1129.60 | 1116.66 | 1114.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-28 09:15:00 | 1116.30 | 1118.28 | 1115.37 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-28 09:15:00 | 1116.30 | 1118.28 | 1115.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 09:15:00 | 1116.30 | 1118.28 | 1115.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-28 09:45:00 | 1115.80 | 1118.28 | 1115.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 10:15:00 | 1112.10 | 1117.04 | 1115.07 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-28 11:30:00 | 1133.60 | 1118.29 | 1115.82 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-02 09:15:00 | 1147.30 | 1160.22 | 1161.27 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 60 — SELL (started 2026-02-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-02 09:15:00 | 1147.30 | 1160.22 | 1161.27 | EMA200 below EMA400 |

### Cycle 61 — BUY (started 2026-02-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 09:15:00 | 1173.40 | 1157.85 | 1157.77 | EMA200 above EMA400 |

### Cycle 62 — SELL (started 2026-02-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-05 10:15:00 | 1147.70 | 1160.76 | 1162.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-06 09:15:00 | 1144.50 | 1154.76 | 1158.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-06 12:15:00 | 1161.10 | 1155.49 | 1157.58 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-06 12:15:00 | 1161.10 | 1155.49 | 1157.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 12:15:00 | 1161.10 | 1155.49 | 1157.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-06 13:00:00 | 1161.10 | 1155.49 | 1157.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 13:15:00 | 1159.30 | 1156.26 | 1157.74 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-06 14:15:00 | 1155.00 | 1156.26 | 1157.74 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-09 09:15:00 | 1148.10 | 1157.48 | 1158.04 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-10 09:15:00 | 1171.90 | 1155.94 | 1155.18 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 63 — BUY (started 2026-02-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-10 09:15:00 | 1171.90 | 1155.94 | 1155.18 | EMA200 above EMA400 |

### Cycle 64 — SELL (started 2026-02-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-11 10:15:00 | 1148.60 | 1157.39 | 1158.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-11 13:15:00 | 1143.20 | 1151.89 | 1155.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-16 09:15:00 | 1126.00 | 1122.42 | 1131.59 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-16 09:45:00 | 1127.00 | 1122.42 | 1131.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 09:15:00 | 1123.90 | 1123.18 | 1127.51 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-17 14:00:00 | 1108.10 | 1116.30 | 1122.51 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-17 14:30:00 | 1108.30 | 1114.62 | 1121.18 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-18 13:30:00 | 1108.60 | 1112.13 | 1116.85 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-18 14:30:00 | 1108.20 | 1111.01 | 1115.91 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 09:15:00 | 1107.40 | 1110.09 | 1114.63 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-19 11:00:00 | 1105.70 | 1109.21 | 1113.82 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-27 09:15:00 | 1052.69 | 1064.11 | 1070.31 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-27 09:15:00 | 1052.88 | 1064.11 | 1070.31 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-27 09:15:00 | 1053.17 | 1064.11 | 1070.31 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-27 09:15:00 | 1052.79 | 1064.11 | 1070.31 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-27 09:15:00 | 1050.41 | 1064.11 | 1070.31 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-03-02 09:15:00 | 997.29 | 1040.77 | 1054.35 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 65 — BUY (started 2026-03-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-10 12:15:00 | 1004.30 | 989.22 | 987.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-11 09:15:00 | 1012.50 | 998.97 | 993.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-11 12:15:00 | 1002.70 | 1002.72 | 996.74 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-11 13:00:00 | 1002.70 | 1002.72 | 996.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 14:15:00 | 987.70 | 998.91 | 996.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-11 15:00:00 | 987.70 | 998.91 | 996.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 15:15:00 | 986.50 | 996.43 | 995.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-12 09:15:00 | 977.40 | 996.43 | 995.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 66 — SELL (started 2026-03-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-12 09:15:00 | 975.00 | 992.15 | 993.31 | EMA200 below EMA400 |

### Cycle 67 — BUY (started 2026-03-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-12 13:15:00 | 1001.00 | 995.05 | 994.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-12 14:15:00 | 1016.00 | 999.24 | 996.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-13 09:15:00 | 1001.40 | 1002.63 | 998.50 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-13 09:15:00 | 1001.40 | 1002.63 | 998.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-13 09:15:00 | 1001.40 | 1002.63 | 998.50 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-16 09:30:00 | 1025.00 | 1014.29 | 1006.45 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-20 09:15:00 | 1044.10 | 1060.49 | 1062.41 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 68 — SELL (started 2026-03-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-20 09:15:00 | 1044.10 | 1060.49 | 1062.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-20 10:15:00 | 1025.70 | 1053.53 | 1059.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-23 13:15:00 | 1036.00 | 1033.97 | 1042.58 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-23 13:45:00 | 1033.80 | 1033.97 | 1042.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-23 14:15:00 | 1026.50 | 1032.48 | 1041.12 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-24 09:45:00 | 1019.30 | 1029.46 | 1038.17 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-24 11:15:00 | 1048.50 | 1033.36 | 1038.44 | SL hit (close>static) qty=1.00 sl=1041.90 alert=retest2 |

### Cycle 69 — BUY (started 2026-03-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-24 12:15:00 | 1075.90 | 1041.87 | 1041.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 09:15:00 | 1095.00 | 1064.50 | 1053.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-30 09:15:00 | 1107.90 | 1110.57 | 1098.18 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-30 09:15:00 | 1107.90 | 1110.57 | 1098.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-30 09:15:00 | 1107.90 | 1110.57 | 1098.18 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-30 13:45:00 | 1116.90 | 1111.79 | 1102.58 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-06 13:45:00 | 1116.70 | 1125.94 | 1125.50 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-08 13:15:00 | 1124.50 | 1137.45 | 1137.53 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 70 — SELL (started 2026-04-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-08 13:15:00 | 1124.50 | 1137.45 | 1137.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-08 15:15:00 | 1117.00 | 1131.11 | 1134.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-10 09:15:00 | 1121.10 | 1116.14 | 1123.01 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-10 09:15:00 | 1121.10 | 1116.14 | 1123.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-10 09:15:00 | 1121.10 | 1116.14 | 1123.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-10 09:45:00 | 1121.60 | 1116.14 | 1123.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-10 10:15:00 | 1121.50 | 1117.22 | 1122.88 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-10 14:15:00 | 1112.60 | 1118.15 | 1121.96 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-10 15:15:00 | 1115.00 | 1118.44 | 1121.75 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-13 13:15:00 | 1114.70 | 1118.46 | 1120.37 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-13 14:00:00 | 1115.80 | 1117.92 | 1119.95 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 14:15:00 | 1119.70 | 1118.28 | 1119.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-13 14:45:00 | 1117.00 | 1118.28 | 1119.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 15:15:00 | 1121.00 | 1118.82 | 1120.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-15 09:15:00 | 1126.50 | 1118.82 | 1120.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2026-04-15 09:15:00 | 1155.00 | 1126.06 | 1123.21 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 71 — BUY (started 2026-04-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-15 09:15:00 | 1155.00 | 1126.06 | 1123.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-15 10:15:00 | 1160.70 | 1132.99 | 1126.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-16 14:15:00 | 1191.30 | 1195.44 | 1172.82 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-16 15:00:00 | 1191.30 | 1195.44 | 1172.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-16 15:15:00 | 1180.00 | 1192.35 | 1173.47 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-17 09:15:00 | 1193.20 | 1192.35 | 1173.47 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-22 09:15:00 | 1202.90 | 1199.34 | 1198.81 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-23 10:45:00 | 1192.70 | 1201.01 | 1200.89 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-23 11:15:00 | 1192.20 | 1199.25 | 1200.10 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 72 — SELL (started 2026-04-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-23 11:15:00 | 1192.20 | 1199.25 | 1200.10 | EMA200 below EMA400 |

### Cycle 73 — BUY (started 2026-04-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-23 15:15:00 | 1205.00 | 1200.64 | 1200.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-24 09:15:00 | 1205.90 | 1201.69 | 1200.92 | Break + close above crossover candle high |

### Cycle 74 — SELL (started 2026-04-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-24 10:15:00 | 1193.80 | 1200.11 | 1200.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-24 11:15:00 | 1188.60 | 1197.81 | 1199.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-24 15:15:00 | 1194.00 | 1193.55 | 1196.44 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-27 09:15:00 | 1194.60 | 1193.55 | 1196.44 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 09:15:00 | 1198.10 | 1194.46 | 1196.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 09:45:00 | 1200.30 | 1194.46 | 1196.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 10:15:00 | 1199.00 | 1195.37 | 1196.81 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-27 11:15:00 | 1193.00 | 1195.37 | 1196.81 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-27 14:00:00 | 1195.00 | 1195.85 | 1196.72 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-27 15:15:00 | 1204.00 | 1197.57 | 1197.35 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 75 — BUY (started 2026-04-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 15:15:00 | 1204.00 | 1197.57 | 1197.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-28 09:15:00 | 1204.20 | 1198.90 | 1197.98 | Break + close above crossover candle high |

### Cycle 76 — SELL (started 2026-04-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-28 13:15:00 | 1186.00 | 1197.49 | 1197.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-28 14:15:00 | 1181.00 | 1194.19 | 1196.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-29 15:15:00 | 1185.00 | 1179.79 | 1186.09 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-30 09:15:00 | 1190.60 | 1179.79 | 1186.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 09:15:00 | 1212.00 | 1186.23 | 1188.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-30 09:45:00 | 1227.00 | 1186.23 | 1188.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 10:15:00 | 1196.00 | 1188.19 | 1189.13 | EMA400 retest candle locked (from downside) |

### Cycle 77 — BUY (started 2026-04-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-30 11:15:00 | 1200.90 | 1190.73 | 1190.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-04 09:15:00 | 1242.10 | 1208.35 | 1199.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-05 10:15:00 | 1229.50 | 1237.42 | 1223.74 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-05 10:15:00 | 1229.50 | 1237.42 | 1223.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 10:15:00 | 1229.50 | 1237.42 | 1223.74 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-05 11:45:00 | 1242.80 | 1238.52 | 1225.49 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-05-08 10:15:00 | 1247.80 | 1263.51 | 1263.84 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 78 — SELL (started 2026-05-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-08 10:15:00 | 1247.80 | 1263.51 | 1263.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-05-08 11:15:00 | 1245.50 | 1259.91 | 1262.18 | Break + close below crossover candle low |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-05-19 10:15:00 | 1053.00 | 2025-05-23 14:15:00 | 1052.60 | STOP_HIT | 1.00 | -0.04% |
| BUY | retest2 | 2025-05-20 15:15:00 | 1055.00 | 2025-05-23 14:15:00 | 1052.60 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest2 | 2025-06-05 12:45:00 | 1060.00 | 2025-06-06 09:15:00 | 1082.20 | STOP_HIT | 1.00 | -2.09% |
| BUY | retest2 | 2025-06-12 11:45:00 | 1114.50 | 2025-06-27 15:15:00 | 1190.00 | STOP_HIT | 1.00 | 6.77% |
| BUY | retest2 | 2025-06-16 10:45:00 | 1115.30 | 2025-06-27 15:15:00 | 1190.00 | STOP_HIT | 1.00 | 6.70% |
| BUY | retest1 | 2025-07-17 09:15:00 | 1413.00 | 2025-07-17 15:15:00 | 1399.00 | STOP_HIT | 1.00 | -0.99% |
| BUY | retest2 | 2025-07-18 09:15:00 | 1408.90 | 2025-07-18 09:15:00 | 1395.50 | STOP_HIT | 1.00 | -0.95% |
| BUY | retest2 | 2025-07-18 13:00:00 | 1409.20 | 2025-07-21 14:15:00 | 1392.10 | STOP_HIT | 1.00 | -1.21% |
| BUY | retest2 | 2025-07-18 14:30:00 | 1410.40 | 2025-07-21 14:15:00 | 1392.10 | STOP_HIT | 1.00 | -1.30% |
| BUY | retest2 | 2025-07-18 15:15:00 | 1410.30 | 2025-07-21 14:15:00 | 1392.10 | STOP_HIT | 1.00 | -1.29% |
| SELL | retest2 | 2025-08-06 09:15:00 | 1367.10 | 2025-08-11 14:15:00 | 1365.40 | STOP_HIT | 1.00 | 0.12% |
| SELL | retest2 | 2025-08-06 10:30:00 | 1377.00 | 2025-08-11 14:15:00 | 1365.40 | STOP_HIT | 1.00 | 0.84% |
| SELL | retest2 | 2025-08-06 12:15:00 | 1378.80 | 2025-08-11 14:15:00 | 1365.40 | STOP_HIT | 1.00 | 0.97% |
| SELL | retest2 | 2025-08-06 14:00:00 | 1379.80 | 2025-08-11 14:15:00 | 1365.40 | STOP_HIT | 1.00 | 1.04% |
| SELL | retest2 | 2025-08-08 10:30:00 | 1330.60 | 2025-08-11 14:15:00 | 1365.40 | STOP_HIT | 1.00 | -2.62% |
| SELL | retest2 | 2025-08-08 11:15:00 | 1327.60 | 2025-08-11 14:15:00 | 1365.40 | STOP_HIT | 1.00 | -2.85% |
| SELL | retest2 | 2025-08-11 09:30:00 | 1332.00 | 2025-08-11 14:15:00 | 1365.40 | STOP_HIT | 1.00 | -2.51% |
| SELL | retest2 | 2025-08-25 09:15:00 | 1249.60 | 2025-09-02 14:15:00 | 1241.70 | STOP_HIT | 1.00 | 0.63% |
| BUY | retest2 | 2025-09-05 15:15:00 | 1248.30 | 2025-09-08 10:15:00 | 1238.30 | STOP_HIT | 1.00 | -0.80% |
| SELL | retest2 | 2025-09-11 11:30:00 | 1240.00 | 2025-09-11 12:15:00 | 1248.10 | STOP_HIT | 1.00 | -0.65% |
| SELL | retest2 | 2025-09-23 15:15:00 | 1206.00 | 2025-09-26 09:15:00 | 1145.70 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-23 15:15:00 | 1206.00 | 2025-09-29 09:15:00 | 1158.90 | STOP_HIT | 0.50 | 3.91% |
| SELL | retest2 | 2025-09-24 13:15:00 | 1192.70 | 2025-10-01 15:15:00 | 1170.10 | STOP_HIT | 1.00 | 1.89% |
| BUY | retest2 | 2025-10-08 09:15:00 | 1193.00 | 2025-10-08 10:15:00 | 1183.50 | STOP_HIT | 1.00 | -0.80% |
| SELL | retest2 | 2025-11-10 09:15:00 | 1231.00 | 2025-11-10 12:15:00 | 1255.80 | STOP_HIT | 1.00 | -2.01% |
| SELL | retest2 | 2025-11-17 10:15:00 | 1208.40 | 2025-11-21 10:15:00 | 1242.40 | STOP_HIT | 1.00 | -2.81% |
| SELL | retest2 | 2025-11-17 11:00:00 | 1224.30 | 2025-11-21 10:15:00 | 1242.40 | STOP_HIT | 1.00 | -1.48% |
| SELL | retest2 | 2025-12-02 12:15:00 | 1215.00 | 2025-12-02 12:15:00 | 1223.60 | STOP_HIT | 1.00 | -0.71% |
| SELL | retest2 | 2026-01-13 09:30:00 | 1175.20 | 2026-01-20 10:15:00 | 1116.44 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-13 14:00:00 | 1175.10 | 2026-01-20 10:15:00 | 1116.34 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-16 12:30:00 | 1174.10 | 2026-01-20 10:15:00 | 1115.39 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-13 09:30:00 | 1175.20 | 2026-01-21 10:15:00 | 1057.68 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-01-13 14:00:00 | 1175.10 | 2026-01-21 10:15:00 | 1057.59 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-01-16 12:30:00 | 1174.10 | 2026-01-21 10:15:00 | 1056.69 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2026-01-28 11:30:00 | 1133.60 | 2026-02-02 09:15:00 | 1147.30 | STOP_HIT | 1.00 | 1.21% |
| SELL | retest2 | 2026-02-06 14:15:00 | 1155.00 | 2026-02-10 09:15:00 | 1171.90 | STOP_HIT | 1.00 | -1.46% |
| SELL | retest2 | 2026-02-09 09:15:00 | 1148.10 | 2026-02-10 09:15:00 | 1171.90 | STOP_HIT | 1.00 | -2.07% |
| SELL | retest2 | 2026-02-17 14:00:00 | 1108.10 | 2026-02-27 09:15:00 | 1052.69 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-17 14:30:00 | 1108.30 | 2026-02-27 09:15:00 | 1052.88 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-18 13:30:00 | 1108.60 | 2026-02-27 09:15:00 | 1053.17 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-18 14:30:00 | 1108.20 | 2026-02-27 09:15:00 | 1052.79 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-19 11:00:00 | 1105.70 | 2026-02-27 09:15:00 | 1050.41 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-17 14:00:00 | 1108.10 | 2026-03-02 09:15:00 | 997.29 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-02-17 14:30:00 | 1108.30 | 2026-03-02 09:15:00 | 997.47 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-02-18 13:30:00 | 1108.60 | 2026-03-02 09:15:00 | 997.74 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-02-18 14:30:00 | 1108.20 | 2026-03-02 09:15:00 | 997.38 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-02-19 11:00:00 | 1105.70 | 2026-03-02 09:15:00 | 995.13 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2026-03-16 09:30:00 | 1025.00 | 2026-03-20 09:15:00 | 1044.10 | STOP_HIT | 1.00 | 1.86% |
| SELL | retest2 | 2026-03-24 09:45:00 | 1019.30 | 2026-03-24 11:15:00 | 1048.50 | STOP_HIT | 1.00 | -2.86% |
| BUY | retest2 | 2026-03-30 13:45:00 | 1116.90 | 2026-04-08 13:15:00 | 1124.50 | STOP_HIT | 1.00 | 0.68% |
| BUY | retest2 | 2026-04-06 13:45:00 | 1116.70 | 2026-04-08 13:15:00 | 1124.50 | STOP_HIT | 1.00 | 0.70% |
| SELL | retest2 | 2026-04-10 14:15:00 | 1112.60 | 2026-04-15 09:15:00 | 1155.00 | STOP_HIT | 1.00 | -3.81% |
| SELL | retest2 | 2026-04-10 15:15:00 | 1115.00 | 2026-04-15 09:15:00 | 1155.00 | STOP_HIT | 1.00 | -3.59% |
| SELL | retest2 | 2026-04-13 13:15:00 | 1114.70 | 2026-04-15 09:15:00 | 1155.00 | STOP_HIT | 1.00 | -3.62% |
| SELL | retest2 | 2026-04-13 14:00:00 | 1115.80 | 2026-04-15 09:15:00 | 1155.00 | STOP_HIT | 1.00 | -3.51% |
| BUY | retest2 | 2026-04-17 09:15:00 | 1193.20 | 2026-04-23 11:15:00 | 1192.20 | STOP_HIT | 1.00 | -0.08% |
| BUY | retest2 | 2026-04-22 09:15:00 | 1202.90 | 2026-04-23 11:15:00 | 1192.20 | STOP_HIT | 1.00 | -0.89% |
| BUY | retest2 | 2026-04-23 10:45:00 | 1192.70 | 2026-04-23 11:15:00 | 1192.20 | STOP_HIT | 1.00 | -0.04% |
| SELL | retest2 | 2026-04-27 11:15:00 | 1193.00 | 2026-04-27 15:15:00 | 1204.00 | STOP_HIT | 1.00 | -0.92% |
| SELL | retest2 | 2026-04-27 14:00:00 | 1195.00 | 2026-04-27 15:15:00 | 1204.00 | STOP_HIT | 1.00 | -0.75% |
| BUY | retest2 | 2026-05-05 11:45:00 | 1242.80 | 2026-05-08 10:15:00 | 1247.80 | STOP_HIT | 1.00 | 0.40% |
