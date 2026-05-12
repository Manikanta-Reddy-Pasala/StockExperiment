# KPIT Technologies Ltd. (KPITTECH)

## Backtest Summary

- **Window:** 2025-03-13 09:15:00 → 2026-05-08 15:15:00 (1976 bars)
- **Last close:** 725.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 68 |
| ALERT1 | 48 |
| ALERT2 | 48 |
| ALERT2_SKIP | 20 |
| ALERT3 | 143 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 4 |
| ENTRY2 | 57 |
| PARTIAL | 8 |
| TARGET_HIT | 8 |
| STOP_HIT | 53 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 69 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 20 / 49
- **Target hits / Stop hits / Partials:** 8 / 53 / 8
- **Avg / median % per leg:** 0.69% / -1.07%
- **Sum % (uncompounded):** 47.58%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 25 | 2 | 8.0% | 0 | 25 | 0 | -1.00% | -25.1% |
| BUY @ 2nd Alert (retest1) | 2 | 0 | 0.0% | 0 | 2 | 0 | -0.49% | -1.0% |
| BUY @ 3rd Alert (retest2) | 23 | 2 | 8.7% | 0 | 23 | 0 | -1.05% | -24.1% |
| SELL (all) | 44 | 18 | 40.9% | 8 | 28 | 8 | 1.65% | 72.6% |
| SELL @ 2nd Alert (retest1) | 2 | 2 | 100.0% | 0 | 2 | 0 | 0.89% | 1.8% |
| SELL @ 3rd Alert (retest2) | 42 | 16 | 38.1% | 8 | 26 | 8 | 1.69% | 70.9% |
| retest1 (combined) | 4 | 2 | 50.0% | 0 | 4 | 0 | 0.20% | 0.8% |
| retest2 (combined) | 65 | 18 | 27.7% | 8 | 49 | 8 | 0.72% | 46.8% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 09:15:00 | 1310.50 | 1264.74 | 1264.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 13:15:00 | 1332.60 | 1296.50 | 1280.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-14 13:15:00 | 1354.00 | 1354.91 | 1334.59 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-14 14:00:00 | 1354.00 | 1354.91 | 1334.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-16 11:15:00 | 1369.70 | 1377.25 | 1365.32 | EMA400 retest candle locked (from upside) |

### Cycle 2 — SELL (started 2025-05-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-19 11:15:00 | 1344.70 | 1359.61 | 1361.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-19 13:15:00 | 1342.30 | 1354.18 | 1358.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-21 10:15:00 | 1336.90 | 1331.50 | 1341.10 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-21 11:00:00 | 1336.90 | 1331.50 | 1341.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 09:15:00 | 1343.00 | 1326.69 | 1330.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-23 09:30:00 | 1345.10 | 1326.69 | 1330.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 10:15:00 | 1336.00 | 1328.56 | 1330.69 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-23 11:30:00 | 1332.50 | 1330.14 | 1331.22 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-23 13:15:00 | 1337.00 | 1332.48 | 1332.15 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 3 — BUY (started 2025-05-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-23 13:15:00 | 1337.00 | 1332.48 | 1332.15 | EMA200 above EMA400 |

### Cycle 4 — SELL (started 2025-05-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-23 15:15:00 | 1326.00 | 1330.79 | 1331.42 | EMA200 below EMA400 |

### Cycle 5 — BUY (started 2025-05-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-26 09:15:00 | 1337.90 | 1332.21 | 1332.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-27 09:15:00 | 1348.40 | 1337.21 | 1334.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-27 12:15:00 | 1332.70 | 1337.07 | 1335.34 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-27 12:15:00 | 1332.70 | 1337.07 | 1335.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 12:15:00 | 1332.70 | 1337.07 | 1335.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-27 13:00:00 | 1332.70 | 1337.07 | 1335.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 13:15:00 | 1325.50 | 1334.76 | 1334.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-27 14:00:00 | 1325.50 | 1334.76 | 1334.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 6 — SELL (started 2025-05-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-27 14:15:00 | 1330.80 | 1333.97 | 1334.11 | EMA200 below EMA400 |

### Cycle 7 — BUY (started 2025-05-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-29 09:15:00 | 1340.50 | 1333.79 | 1333.43 | EMA200 above EMA400 |

### Cycle 8 — SELL (started 2025-05-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-30 12:15:00 | 1332.00 | 1336.09 | 1336.57 | EMA200 below EMA400 |

### Cycle 9 — BUY (started 2025-05-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-30 13:15:00 | 1340.20 | 1336.91 | 1336.90 | EMA200 above EMA400 |

### Cycle 10 — SELL (started 2025-05-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-30 15:15:00 | 1334.90 | 1336.54 | 1336.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-02 09:15:00 | 1329.00 | 1335.03 | 1336.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-02 13:15:00 | 1331.50 | 1331.45 | 1333.72 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-02 13:15:00 | 1331.50 | 1331.45 | 1333.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 13:15:00 | 1331.50 | 1331.45 | 1333.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-02 14:00:00 | 1331.50 | 1331.45 | 1333.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 14:15:00 | 1332.80 | 1331.72 | 1333.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-02 15:00:00 | 1332.80 | 1331.72 | 1333.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 09:15:00 | 1341.40 | 1333.73 | 1334.22 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-03 11:30:00 | 1330.10 | 1333.19 | 1333.89 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-03 14:15:00 | 1328.10 | 1332.93 | 1333.65 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-09 09:15:00 | 1356.20 | 1325.19 | 1322.26 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 11 — BUY (started 2025-06-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-09 09:15:00 | 1356.20 | 1325.19 | 1322.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-09 15:15:00 | 1357.90 | 1346.78 | 1336.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-10 12:15:00 | 1359.50 | 1361.61 | 1347.83 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-10 12:45:00 | 1362.00 | 1361.61 | 1347.83 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 09:15:00 | 1375.50 | 1376.11 | 1366.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-12 09:30:00 | 1367.80 | 1376.11 | 1366.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 13:15:00 | 1360.40 | 1374.49 | 1368.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-12 14:00:00 | 1360.40 | 1374.49 | 1368.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 14:15:00 | 1363.00 | 1372.20 | 1368.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-12 15:15:00 | 1359.00 | 1372.20 | 1368.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 15:15:00 | 1359.00 | 1369.56 | 1367.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-13 09:15:00 | 1343.50 | 1369.56 | 1367.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 12 — SELL (started 2025-06-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-13 09:15:00 | 1346.90 | 1365.03 | 1365.49 | EMA200 below EMA400 |

### Cycle 13 — BUY (started 2025-06-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-16 11:15:00 | 1389.60 | 1366.48 | 1363.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-16 12:15:00 | 1393.50 | 1371.88 | 1366.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-17 12:15:00 | 1396.60 | 1398.91 | 1386.06 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-17 12:45:00 | 1393.90 | 1398.91 | 1386.06 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 09:15:00 | 1409.10 | 1400.56 | 1390.98 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-18 12:45:00 | 1413.90 | 1404.73 | 1395.40 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-18 14:30:00 | 1418.20 | 1410.32 | 1399.68 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-19 09:15:00 | 1386.80 | 1407.00 | 1400.10 | SL hit (close<static) qty=1.00 sl=1390.80 alert=retest2 |

### Cycle 14 — SELL (started 2025-06-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-19 13:15:00 | 1390.50 | 1395.61 | 1396.07 | EMA200 below EMA400 |

### Cycle 15 — BUY (started 2025-06-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-23 09:15:00 | 1401.90 | 1395.76 | 1395.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-23 10:15:00 | 1410.10 | 1398.63 | 1396.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-23 13:15:00 | 1396.10 | 1400.10 | 1397.97 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-23 13:15:00 | 1396.10 | 1400.10 | 1397.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 13:15:00 | 1396.10 | 1400.10 | 1397.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-23 14:00:00 | 1396.10 | 1400.10 | 1397.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 14:15:00 | 1389.30 | 1397.94 | 1397.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-23 15:00:00 | 1389.30 | 1397.94 | 1397.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 15:15:00 | 1391.70 | 1396.69 | 1396.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-24 09:15:00 | 1333.80 | 1396.69 | 1396.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 16 — SELL (started 2025-06-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-24 09:15:00 | 1324.40 | 1382.23 | 1390.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-24 11:15:00 | 1312.50 | 1357.62 | 1376.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-27 09:15:00 | 1287.40 | 1281.32 | 1298.89 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-27 14:45:00 | 1268.80 | 1279.74 | 1291.76 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-30 10:15:00 | 1273.30 | 1277.86 | 1288.69 | SELL ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 09:15:00 | 1254.30 | 1249.83 | 1260.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-02 09:30:00 | 1256.70 | 1249.83 | 1260.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 12:15:00 | 1251.60 | 1251.59 | 1258.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-02 12:30:00 | 1258.60 | 1251.59 | 1258.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 09:15:00 | 1254.40 | 1249.40 | 1255.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-03 09:45:00 | 1256.50 | 1249.40 | 1255.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 10:15:00 | 1259.70 | 1251.46 | 1255.51 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-07-03 10:15:00 | 1259.70 | 1251.46 | 1255.51 | SL hit (close>ema400) qty=1.00 sl=1255.51 alert=retest1 |

### Cycle 17 — BUY (started 2025-07-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-03 14:15:00 | 1260.20 | 1257.86 | 1257.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-03 15:15:00 | 1263.90 | 1259.07 | 1258.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-04 12:15:00 | 1260.00 | 1260.46 | 1259.36 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-04 12:15:00 | 1260.00 | 1260.46 | 1259.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 12:15:00 | 1260.00 | 1260.46 | 1259.36 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-04 13:45:00 | 1262.80 | 1260.73 | 1259.58 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-04 14:15:00 | 1262.80 | 1260.73 | 1259.58 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-07 09:45:00 | 1263.30 | 1262.57 | 1260.82 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-09 13:15:00 | 1262.80 | 1267.62 | 1266.94 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-09 13:15:00 | 1261.60 | 1266.42 | 1266.46 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 18 — SELL (started 2025-07-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-09 13:15:00 | 1261.60 | 1266.42 | 1266.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-09 15:15:00 | 1258.00 | 1263.88 | 1265.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-10 15:15:00 | 1256.00 | 1254.34 | 1258.53 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-11 09:15:00 | 1260.30 | 1254.34 | 1258.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 09:15:00 | 1264.60 | 1256.39 | 1259.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-11 10:00:00 | 1264.60 | 1256.39 | 1259.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 10:15:00 | 1268.60 | 1258.83 | 1259.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-11 10:45:00 | 1267.20 | 1258.83 | 1259.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 19 — BUY (started 2025-07-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-11 11:15:00 | 1272.00 | 1261.47 | 1261.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-15 09:15:00 | 1280.30 | 1268.45 | 1265.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-16 13:15:00 | 1291.60 | 1292.12 | 1284.01 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-16 14:00:00 | 1291.60 | 1292.12 | 1284.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 15:15:00 | 1286.40 | 1290.12 | 1284.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-17 09:15:00 | 1294.50 | 1290.12 | 1284.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 09:15:00 | 1288.30 | 1289.76 | 1284.81 | EMA400 retest candle locked (from upside) |

### Cycle 20 — SELL (started 2025-07-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-18 09:15:00 | 1271.00 | 1280.61 | 1281.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-21 09:15:00 | 1260.60 | 1272.06 | 1276.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-21 15:15:00 | 1266.80 | 1265.79 | 1270.68 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-22 09:15:00 | 1267.70 | 1265.79 | 1270.68 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 09:15:00 | 1259.80 | 1264.60 | 1269.69 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-22 12:00:00 | 1254.00 | 1261.84 | 1267.52 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-22 12:45:00 | 1255.70 | 1260.55 | 1266.41 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-23 10:15:00 | 1278.90 | 1264.68 | 1265.90 | SL hit (close>static) qty=1.00 sl=1273.70 alert=retest2 |

### Cycle 21 — BUY (started 2025-07-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-23 12:15:00 | 1272.80 | 1267.86 | 1267.23 | EMA200 above EMA400 |

### Cycle 22 — SELL (started 2025-07-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-24 10:15:00 | 1259.60 | 1267.31 | 1267.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-24 14:15:00 | 1253.00 | 1261.14 | 1264.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-29 11:15:00 | 1203.20 | 1201.04 | 1216.11 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-29 12:00:00 | 1203.20 | 1201.04 | 1216.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 14:15:00 | 1220.00 | 1206.10 | 1214.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 14:45:00 | 1221.50 | 1206.10 | 1214.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 15:15:00 | 1218.00 | 1208.48 | 1215.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-30 09:15:00 | 1207.90 | 1208.48 | 1215.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 09:15:00 | 1215.30 | 1209.84 | 1215.10 | EMA400 retest candle locked (from downside) |

### Cycle 23 — BUY (started 2025-07-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-30 14:15:00 | 1273.50 | 1225.48 | 1220.67 | EMA200 above EMA400 |

### Cycle 24 — SELL (started 2025-08-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-01 10:15:00 | 1213.00 | 1227.17 | 1227.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-01 14:15:00 | 1205.60 | 1220.74 | 1224.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-04 13:15:00 | 1216.90 | 1214.59 | 1219.02 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-04 14:00:00 | 1216.90 | 1214.59 | 1219.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 14:15:00 | 1218.70 | 1215.41 | 1218.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 14:45:00 | 1218.00 | 1215.41 | 1218.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 15:15:00 | 1218.00 | 1215.93 | 1218.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-05 09:15:00 | 1213.90 | 1215.93 | 1218.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 09:15:00 | 1205.50 | 1213.85 | 1217.68 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-06 10:00:00 | 1196.90 | 1207.01 | 1211.82 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-06 11:30:00 | 1200.10 | 1206.17 | 1210.58 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-07 09:30:00 | 1198.20 | 1205.08 | 1208.52 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-08 10:00:00 | 1200.20 | 1204.78 | 1206.40 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 14:15:00 | 1191.00 | 1199.25 | 1202.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-08 14:30:00 | 1197.50 | 1199.25 | 1202.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 10:15:00 | 1197.80 | 1197.03 | 1200.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-11 10:30:00 | 1203.00 | 1197.03 | 1200.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 11:15:00 | 1200.80 | 1197.79 | 1200.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-11 12:00:00 | 1200.80 | 1197.79 | 1200.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 12:15:00 | 1202.20 | 1198.67 | 1200.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-11 12:45:00 | 1205.00 | 1198.67 | 1200.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 13:15:00 | 1202.70 | 1199.47 | 1201.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-11 14:00:00 | 1202.70 | 1199.47 | 1201.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 14:15:00 | 1198.50 | 1199.28 | 1200.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-11 14:30:00 | 1198.60 | 1199.28 | 1200.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-08-12 09:15:00 | 1214.90 | 1202.25 | 1201.87 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 25 — BUY (started 2025-08-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-12 09:15:00 | 1214.90 | 1202.25 | 1201.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-12 13:15:00 | 1224.50 | 1212.09 | 1207.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-13 09:15:00 | 1209.40 | 1214.16 | 1209.46 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-13 09:15:00 | 1209.40 | 1214.16 | 1209.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 09:15:00 | 1209.40 | 1214.16 | 1209.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-13 10:00:00 | 1209.40 | 1214.16 | 1209.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 10:15:00 | 1219.30 | 1215.18 | 1210.35 | EMA400 retest candle locked (from upside) |

### Cycle 26 — SELL (started 2025-08-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-18 11:15:00 | 1202.00 | 1212.83 | 1212.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-19 09:15:00 | 1195.30 | 1207.90 | 1210.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-19 11:15:00 | 1212.00 | 1208.23 | 1210.14 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-19 11:15:00 | 1212.00 | 1208.23 | 1210.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 11:15:00 | 1212.00 | 1208.23 | 1210.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-19 11:45:00 | 1211.60 | 1208.23 | 1210.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 12:15:00 | 1201.70 | 1206.92 | 1209.37 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-19 13:15:00 | 1200.60 | 1206.92 | 1209.37 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-20 09:15:00 | 1197.70 | 1203.67 | 1207.04 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-20 11:15:00 | 1218.00 | 1207.24 | 1207.87 | SL hit (close>static) qty=1.00 sl=1212.60 alert=retest2 |

### Cycle 27 — BUY (started 2025-08-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-20 12:15:00 | 1217.30 | 1209.25 | 1208.73 | EMA200 above EMA400 |

### Cycle 28 — SELL (started 2025-08-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-22 10:15:00 | 1199.80 | 1209.76 | 1210.87 | EMA200 below EMA400 |

### Cycle 29 — BUY (started 2025-08-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-25 10:15:00 | 1227.80 | 1211.46 | 1210.07 | EMA200 above EMA400 |

### Cycle 30 — SELL (started 2025-08-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-26 10:15:00 | 1204.80 | 1210.10 | 1210.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-26 14:15:00 | 1200.60 | 1208.08 | 1209.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-29 12:15:00 | 1191.60 | 1190.67 | 1196.53 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-29 12:45:00 | 1192.80 | 1190.67 | 1196.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 13:15:00 | 1194.40 | 1191.42 | 1196.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-29 13:45:00 | 1190.90 | 1191.42 | 1196.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 09:15:00 | 1202.90 | 1193.27 | 1195.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 09:45:00 | 1204.50 | 1193.27 | 1195.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 10:15:00 | 1206.50 | 1195.92 | 1196.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 11:00:00 | 1206.50 | 1195.92 | 1196.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 31 — BUY (started 2025-09-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-01 11:15:00 | 1215.80 | 1199.90 | 1198.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-01 12:15:00 | 1222.70 | 1204.46 | 1200.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-02 13:15:00 | 1207.90 | 1214.32 | 1209.82 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-02 13:15:00 | 1207.90 | 1214.32 | 1209.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 13:15:00 | 1207.90 | 1214.32 | 1209.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-02 14:00:00 | 1207.90 | 1214.32 | 1209.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 14:15:00 | 1207.30 | 1212.91 | 1209.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-02 15:00:00 | 1207.30 | 1212.91 | 1209.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 15:15:00 | 1207.00 | 1211.73 | 1209.35 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-03 09:15:00 | 1211.90 | 1211.73 | 1209.35 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-03 11:00:00 | 1211.20 | 1211.84 | 1209.83 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-12 12:15:00 | 1251.90 | 1259.71 | 1260.03 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 32 — SELL (started 2025-09-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-12 12:15:00 | 1251.90 | 1259.71 | 1260.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-12 15:15:00 | 1250.00 | 1255.93 | 1258.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-15 14:15:00 | 1251.00 | 1250.99 | 1254.09 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-15 14:15:00 | 1251.00 | 1250.99 | 1254.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 14:15:00 | 1251.00 | 1250.99 | 1254.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-15 14:30:00 | 1254.80 | 1250.99 | 1254.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 15:15:00 | 1253.00 | 1251.39 | 1253.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-16 09:15:00 | 1249.30 | 1251.39 | 1253.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 09:15:00 | 1251.60 | 1251.43 | 1253.78 | EMA400 retest candle locked (from downside) |

### Cycle 33 — BUY (started 2025-09-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-17 09:15:00 | 1273.60 | 1253.96 | 1253.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-17 11:15:00 | 1276.40 | 1261.06 | 1256.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-19 10:15:00 | 1294.80 | 1299.46 | 1288.37 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-19 10:30:00 | 1295.70 | 1299.46 | 1288.37 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 11:15:00 | 1286.90 | 1296.95 | 1288.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-19 12:00:00 | 1286.90 | 1296.95 | 1288.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 12:15:00 | 1283.40 | 1294.24 | 1287.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-19 13:00:00 | 1283.40 | 1294.24 | 1287.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 14:15:00 | 1284.80 | 1291.26 | 1287.49 | EMA400 retest candle locked (from upside) |

### Cycle 34 — SELL (started 2025-09-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-22 09:15:00 | 1262.70 | 1284.72 | 1285.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-23 14:15:00 | 1248.30 | 1257.19 | 1265.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-25 09:15:00 | 1265.70 | 1248.21 | 1253.97 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-25 09:15:00 | 1265.70 | 1248.21 | 1253.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 09:15:00 | 1265.70 | 1248.21 | 1253.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-25 09:45:00 | 1270.90 | 1248.21 | 1253.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 10:15:00 | 1267.60 | 1252.08 | 1255.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-25 10:45:00 | 1271.00 | 1252.08 | 1255.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 13:15:00 | 1254.40 | 1253.49 | 1255.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-25 13:45:00 | 1253.30 | 1253.49 | 1255.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 14:15:00 | 1255.60 | 1253.91 | 1255.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-25 15:15:00 | 1260.70 | 1253.91 | 1255.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 15:15:00 | 1260.70 | 1255.27 | 1255.73 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-26 09:15:00 | 1251.00 | 1255.27 | 1255.73 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-30 12:15:00 | 1188.45 | 1201.21 | 1216.76 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-09-30 13:15:00 | 1125.90 | 1186.22 | 1208.54 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 35 — BUY (started 2025-10-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-07 14:15:00 | 1172.00 | 1159.51 | 1158.81 | EMA200 above EMA400 |

### Cycle 36 — SELL (started 2025-10-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-09 09:15:00 | 1152.10 | 1159.37 | 1159.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-09 11:15:00 | 1146.80 | 1155.90 | 1158.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-09 14:15:00 | 1161.60 | 1155.82 | 1157.47 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-09 14:15:00 | 1161.60 | 1155.82 | 1157.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 14:15:00 | 1161.60 | 1155.82 | 1157.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-09 15:00:00 | 1161.60 | 1155.82 | 1157.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 15:15:00 | 1161.90 | 1157.04 | 1157.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-10 09:15:00 | 1164.10 | 1157.04 | 1157.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 37 — BUY (started 2025-10-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-10 09:15:00 | 1166.30 | 1158.89 | 1158.64 | EMA200 above EMA400 |

### Cycle 38 — SELL (started 2025-10-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-10 14:15:00 | 1154.60 | 1158.92 | 1159.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-13 09:15:00 | 1145.40 | 1155.12 | 1157.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-13 13:15:00 | 1153.50 | 1150.07 | 1153.57 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-13 13:15:00 | 1153.50 | 1150.07 | 1153.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 13:15:00 | 1153.50 | 1150.07 | 1153.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-13 14:00:00 | 1153.50 | 1150.07 | 1153.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 14:15:00 | 1154.60 | 1150.97 | 1153.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-13 14:30:00 | 1155.40 | 1150.97 | 1153.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 15:15:00 | 1155.00 | 1151.78 | 1153.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-14 09:15:00 | 1160.00 | 1151.78 | 1153.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 09:15:00 | 1153.70 | 1152.16 | 1153.78 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-14 11:15:00 | 1145.00 | 1151.29 | 1153.24 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-14 12:15:00 | 1144.80 | 1150.11 | 1152.52 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-14 13:00:00 | 1142.60 | 1148.61 | 1151.62 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-15 09:15:00 | 1164.90 | 1154.54 | 1153.73 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 39 — BUY (started 2025-10-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-15 09:15:00 | 1164.90 | 1154.54 | 1153.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-15 14:15:00 | 1166.50 | 1160.28 | 1157.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-16 09:15:00 | 1159.10 | 1160.49 | 1157.85 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-16 10:15:00 | 1150.50 | 1160.49 | 1157.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 10:15:00 | 1153.20 | 1159.03 | 1157.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-16 10:30:00 | 1153.60 | 1159.03 | 1157.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 11:15:00 | 1154.80 | 1158.19 | 1157.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-16 11:30:00 | 1150.90 | 1158.19 | 1157.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 09:15:00 | 1156.00 | 1158.22 | 1157.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-17 09:45:00 | 1153.60 | 1158.22 | 1157.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 10:15:00 | 1155.80 | 1157.73 | 1157.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-17 10:30:00 | 1155.40 | 1157.73 | 1157.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 11:15:00 | 1158.40 | 1157.87 | 1157.47 | EMA400 retest candle locked (from upside) |

### Cycle 40 — SELL (started 2025-10-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-17 12:15:00 | 1145.10 | 1155.31 | 1156.34 | EMA200 below EMA400 |

### Cycle 41 — BUY (started 2025-10-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-23 09:15:00 | 1168.50 | 1153.62 | 1152.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-24 15:15:00 | 1179.00 | 1171.20 | 1165.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-28 11:15:00 | 1190.90 | 1198.98 | 1188.33 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-28 12:00:00 | 1190.90 | 1198.98 | 1188.33 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 12:15:00 | 1191.40 | 1197.46 | 1188.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-28 12:30:00 | 1190.40 | 1197.46 | 1188.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 13:15:00 | 1186.40 | 1195.25 | 1188.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-28 14:00:00 | 1186.40 | 1195.25 | 1188.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 14:15:00 | 1191.40 | 1194.48 | 1188.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-28 14:30:00 | 1181.20 | 1194.48 | 1188.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 15:15:00 | 1192.00 | 1193.98 | 1188.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-29 09:15:00 | 1186.00 | 1193.98 | 1188.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 09:15:00 | 1182.10 | 1191.61 | 1188.36 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-29 14:15:00 | 1197.30 | 1191.50 | 1189.19 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-30 09:15:00 | 1170.00 | 1187.65 | 1188.06 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 42 — SELL (started 2025-10-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-30 09:15:00 | 1170.00 | 1187.65 | 1188.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-31 11:15:00 | 1168.10 | 1173.01 | 1178.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-31 13:15:00 | 1173.10 | 1172.06 | 1176.80 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-31 14:00:00 | 1173.10 | 1172.06 | 1176.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 15:15:00 | 1168.90 | 1162.29 | 1167.11 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-04 09:45:00 | 1160.70 | 1162.30 | 1166.67 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-10 10:15:00 | 1172.20 | 1158.46 | 1156.99 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 43 — BUY (started 2025-11-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-10 10:15:00 | 1172.20 | 1158.46 | 1156.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-10 12:15:00 | 1192.00 | 1167.12 | 1161.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-13 15:15:00 | 1234.10 | 1237.39 | 1224.50 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-14 09:15:00 | 1239.10 | 1237.39 | 1224.50 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 12:15:00 | 1223.50 | 1234.21 | 1227.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-14 13:00:00 | 1223.50 | 1234.21 | 1227.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 13:15:00 | 1225.10 | 1232.39 | 1226.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-14 13:45:00 | 1222.80 | 1232.39 | 1226.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 15:15:00 | 1219.30 | 1228.88 | 1226.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-17 09:15:00 | 1221.00 | 1228.88 | 1226.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 09:15:00 | 1221.30 | 1227.36 | 1225.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-17 10:15:00 | 1216.20 | 1227.36 | 1225.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 44 — SELL (started 2025-11-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-17 11:15:00 | 1220.00 | 1223.75 | 1224.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-18 09:15:00 | 1196.30 | 1215.26 | 1219.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-18 11:15:00 | 1212.00 | 1211.51 | 1217.10 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-18 11:15:00 | 1212.00 | 1211.51 | 1217.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 11:15:00 | 1212.00 | 1211.51 | 1217.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-18 12:00:00 | 1212.00 | 1211.51 | 1217.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 09:15:00 | 1215.00 | 1205.97 | 1211.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-19 10:00:00 | 1215.00 | 1205.97 | 1211.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 10:15:00 | 1206.20 | 1206.02 | 1211.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-19 10:45:00 | 1210.30 | 1206.02 | 1211.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 11:15:00 | 1214.30 | 1207.67 | 1211.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-19 12:00:00 | 1214.30 | 1207.67 | 1211.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 12:15:00 | 1208.00 | 1207.74 | 1211.12 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-19 15:00:00 | 1204.20 | 1207.44 | 1210.42 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-20 14:15:00 | 1204.20 | 1207.90 | 1209.48 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-24 11:45:00 | 1194.20 | 1187.33 | 1192.28 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-27 09:15:00 | 1205.60 | 1191.02 | 1189.04 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 45 — BUY (started 2025-11-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-27 09:15:00 | 1205.60 | 1191.02 | 1189.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-27 10:15:00 | 1214.30 | 1195.67 | 1191.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-02 09:15:00 | 1249.00 | 1249.27 | 1234.61 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-03 11:15:00 | 1263.50 | 1254.57 | 1245.05 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-04 09:15:00 | 1268.00 | 1259.38 | 1251.57 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 09:15:00 | 1264.40 | 1267.99 | 1262.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-05 09:45:00 | 1262.90 | 1267.99 | 1262.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 10:15:00 | 1265.20 | 1267.43 | 1262.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-05 10:30:00 | 1267.70 | 1267.43 | 1262.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 11:15:00 | 1259.50 | 1265.84 | 1262.12 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-12-05 11:15:00 | 1259.50 | 1265.84 | 1262.12 | SL hit (close<ema400) qty=1.00 sl=1262.12 alert=retest1 |

### Cycle 46 — SELL (started 2025-12-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-08 10:15:00 | 1245.60 | 1259.24 | 1260.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-08 11:15:00 | 1234.60 | 1254.31 | 1258.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-11 09:15:00 | 1211.70 | 1205.48 | 1215.27 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-11 10:00:00 | 1211.70 | 1205.48 | 1215.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 10:15:00 | 1216.10 | 1207.60 | 1215.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-11 11:00:00 | 1216.10 | 1207.60 | 1215.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 11:15:00 | 1228.70 | 1211.82 | 1216.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-11 12:00:00 | 1228.70 | 1211.82 | 1216.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 12:15:00 | 1228.90 | 1215.24 | 1217.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-11 12:45:00 | 1230.40 | 1215.24 | 1217.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 47 — BUY (started 2025-12-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-11 14:15:00 | 1231.00 | 1220.72 | 1219.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-12 11:15:00 | 1237.00 | 1227.34 | 1223.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-15 09:15:00 | 1221.90 | 1230.78 | 1227.28 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-15 09:15:00 | 1221.90 | 1230.78 | 1227.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 09:15:00 | 1221.90 | 1230.78 | 1227.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-15 10:00:00 | 1221.90 | 1230.78 | 1227.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 10:15:00 | 1233.50 | 1231.32 | 1227.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-15 10:45:00 | 1225.80 | 1231.32 | 1227.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 11:15:00 | 1226.50 | 1230.36 | 1227.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-15 11:30:00 | 1227.80 | 1230.36 | 1227.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 12:15:00 | 1219.30 | 1228.15 | 1226.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-15 13:00:00 | 1219.30 | 1228.15 | 1226.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 48 — SELL (started 2025-12-15 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-15 13:15:00 | 1211.30 | 1224.78 | 1225.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-15 14:15:00 | 1207.20 | 1221.26 | 1223.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-19 09:15:00 | 1163.80 | 1163.50 | 1172.84 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-19 09:15:00 | 1163.80 | 1163.50 | 1172.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 09:15:00 | 1163.80 | 1163.50 | 1172.84 | EMA400 retest candle locked (from downside) |

### Cycle 49 — BUY (started 2025-12-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-19 13:15:00 | 1214.50 | 1179.30 | 1177.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-19 14:15:00 | 1225.00 | 1188.44 | 1181.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-24 12:15:00 | 1228.80 | 1230.70 | 1222.93 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-24 12:45:00 | 1230.10 | 1230.70 | 1222.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 14:15:00 | 1217.60 | 1227.66 | 1222.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-24 14:45:00 | 1218.20 | 1227.66 | 1222.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 15:15:00 | 1218.90 | 1225.91 | 1222.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-26 09:15:00 | 1234.30 | 1225.91 | 1222.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 10:15:00 | 1221.10 | 1224.79 | 1222.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-26 11:00:00 | 1221.10 | 1224.79 | 1222.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 11:15:00 | 1223.40 | 1224.51 | 1222.66 | EMA400 retest candle locked (from upside) |

### Cycle 50 — SELL (started 2025-12-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-26 14:15:00 | 1206.90 | 1218.68 | 1220.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-29 10:15:00 | 1199.30 | 1212.49 | 1216.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-31 14:15:00 | 1172.60 | 1171.10 | 1180.34 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-31 14:30:00 | 1174.30 | 1171.10 | 1180.34 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 09:15:00 | 1168.90 | 1170.98 | 1178.70 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-01 12:30:00 | 1162.00 | 1168.18 | 1175.40 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-02 09:15:00 | 1161.30 | 1166.58 | 1172.74 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-02 10:30:00 | 1157.00 | 1165.09 | 1170.97 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-05 09:15:00 | 1157.60 | 1165.08 | 1168.57 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 09:15:00 | 1156.50 | 1163.36 | 1167.47 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-05 10:30:00 | 1151.60 | 1161.55 | 1166.27 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-05 14:15:00 | 1151.20 | 1159.30 | 1164.00 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-05 15:00:00 | 1151.50 | 1157.74 | 1162.86 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-07 09:15:00 | 1193.60 | 1154.69 | 1156.11 | SL hit (close>static) qty=1.00 sl=1184.00 alert=retest2 |

### Cycle 51 — BUY (started 2026-01-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-07 10:15:00 | 1191.30 | 1162.01 | 1159.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-07 13:15:00 | 1200.40 | 1177.77 | 1167.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-08 10:15:00 | 1191.50 | 1193.17 | 1179.59 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-08 11:00:00 | 1191.50 | 1193.17 | 1179.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 12:15:00 | 1187.50 | 1192.52 | 1181.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-08 12:45:00 | 1186.10 | 1192.52 | 1181.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 13:15:00 | 1175.00 | 1189.02 | 1181.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-08 14:00:00 | 1175.00 | 1189.02 | 1181.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 14:15:00 | 1181.40 | 1187.49 | 1181.10 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-09 09:45:00 | 1186.90 | 1183.86 | 1180.43 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-09 10:15:00 | 1189.20 | 1183.86 | 1180.43 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-09 13:45:00 | 1187.10 | 1181.32 | 1179.96 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-09 14:30:00 | 1185.80 | 1181.72 | 1180.27 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-12 09:15:00 | 1163.40 | 1177.99 | 1178.82 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 52 — SELL (started 2026-01-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-12 09:15:00 | 1163.40 | 1177.99 | 1178.82 | EMA200 below EMA400 |

### Cycle 53 — BUY (started 2026-01-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-14 09:15:00 | 1188.20 | 1174.66 | 1173.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-16 10:15:00 | 1216.00 | 1189.94 | 1182.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-16 13:15:00 | 1193.80 | 1196.65 | 1187.87 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-16 14:00:00 | 1193.80 | 1196.65 | 1187.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 15:15:00 | 1187.00 | 1193.42 | 1187.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-19 09:15:00 | 1167.10 | 1193.42 | 1187.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 09:15:00 | 1167.80 | 1188.29 | 1186.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-19 10:00:00 | 1167.80 | 1188.29 | 1186.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 54 — SELL (started 2026-01-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-19 10:15:00 | 1161.50 | 1182.93 | 1183.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-20 09:15:00 | 1144.70 | 1163.93 | 1172.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-22 09:15:00 | 1131.50 | 1118.94 | 1132.04 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-22 09:15:00 | 1131.50 | 1118.94 | 1132.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 09:15:00 | 1131.50 | 1118.94 | 1132.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 09:45:00 | 1134.90 | 1118.94 | 1132.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 10:15:00 | 1119.00 | 1118.95 | 1130.86 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-22 11:30:00 | 1116.50 | 1118.26 | 1129.46 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-22 12:45:00 | 1117.40 | 1118.05 | 1128.35 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-22 14:00:00 | 1110.10 | 1116.46 | 1126.69 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-23 09:30:00 | 1116.70 | 1115.32 | 1123.42 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 14:15:00 | 1113.10 | 1111.27 | 1118.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-23 14:45:00 | 1121.80 | 1111.27 | 1118.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 09:15:00 | 1099.20 | 1103.76 | 1109.51 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-28 10:15:00 | 1095.90 | 1103.76 | 1109.51 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-29 09:15:00 | 1097.70 | 1100.85 | 1104.87 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-29 12:15:00 | 1060.67 | 1089.72 | 1097.76 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-29 12:15:00 | 1061.53 | 1089.72 | 1097.76 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-29 12:15:00 | 1060.87 | 1089.72 | 1097.76 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-29 13:15:00 | 1054.59 | 1083.46 | 1094.19 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-29 14:15:00 | 1041.11 | 1074.17 | 1088.99 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-29 14:15:00 | 1042.82 | 1074.17 | 1088.99 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-02-01 11:15:00 | 1004.85 | 1030.16 | 1051.47 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 55 — BUY (started 2026-02-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-10 10:15:00 | 980.70 | 968.68 | 967.51 | EMA200 above EMA400 |

### Cycle 56 — SELL (started 2026-02-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-11 11:15:00 | 960.00 | 967.37 | 968.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-11 15:15:00 | 957.50 | 962.65 | 965.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-16 09:15:00 | 875.50 | 870.29 | 893.43 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-16 10:00:00 | 875.50 | 870.29 | 893.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 09:15:00 | 890.90 | 879.63 | 887.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 10:00:00 | 890.90 | 879.63 | 887.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 10:15:00 | 880.00 | 879.70 | 886.80 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-17 13:45:00 | 879.00 | 880.11 | 885.32 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-20 09:15:00 | 835.05 | 851.55 | 861.15 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-02-24 09:15:00 | 791.10 | 811.34 | 827.54 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 57 — BUY (started 2026-03-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 09:15:00 | 681.50 | 655.51 | 655.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-18 10:15:00 | 691.15 | 662.64 | 658.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 09:15:00 | 661.95 | 677.68 | 669.95 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-19 09:15:00 | 661.95 | 677.68 | 669.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 09:15:00 | 661.95 | 677.68 | 669.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-19 10:00:00 | 661.95 | 677.68 | 669.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 10:15:00 | 661.25 | 674.39 | 669.16 | EMA400 retest candle locked (from upside) |

### Cycle 58 — SELL (started 2026-03-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 14:15:00 | 657.10 | 665.07 | 665.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-23 09:15:00 | 654.00 | 662.00 | 663.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-23 14:15:00 | 658.80 | 657.95 | 660.89 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-23 14:45:00 | 659.45 | 657.95 | 660.89 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-23 15:15:00 | 656.80 | 657.72 | 660.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 09:15:00 | 660.95 | 657.72 | 660.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 09:15:00 | 653.20 | 656.81 | 659.85 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-24 10:15:00 | 652.95 | 656.81 | 659.85 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-24 14:15:00 | 666.05 | 661.66 | 661.11 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 59 — BUY (started 2026-03-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-24 14:15:00 | 666.05 | 661.66 | 661.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 09:15:00 | 680.70 | 665.69 | 663.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-27 09:15:00 | 666.85 | 672.17 | 668.83 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-27 09:15:00 | 666.85 | 672.17 | 668.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 09:15:00 | 666.85 | 672.17 | 668.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 10:00:00 | 666.85 | 672.17 | 668.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 10:15:00 | 668.20 | 671.38 | 668.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 10:45:00 | 666.15 | 671.38 | 668.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 11:15:00 | 666.05 | 670.31 | 668.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 12:15:00 | 665.95 | 670.31 | 668.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 12:15:00 | 663.25 | 668.90 | 668.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 13:00:00 | 663.25 | 668.90 | 668.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 60 — SELL (started 2026-03-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 13:15:00 | 660.70 | 667.26 | 667.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 09:15:00 | 648.00 | 661.32 | 664.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 679.30 | 651.91 | 656.33 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 679.30 | 651.91 | 656.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 679.30 | 651.91 | 656.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 10:00:00 | 679.30 | 651.91 | 656.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 10:15:00 | 675.40 | 656.61 | 658.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 10:30:00 | 678.50 | 656.61 | 658.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 61 — BUY (started 2026-04-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 11:15:00 | 678.90 | 661.06 | 659.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-02 12:15:00 | 680.20 | 672.60 | 667.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-09 09:15:00 | 710.00 | 713.64 | 705.83 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-09 10:00:00 | 710.00 | 713.64 | 705.83 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-10 09:15:00 | 712.00 | 715.85 | 711.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-10 09:45:00 | 713.40 | 715.85 | 711.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-10 10:15:00 | 711.75 | 715.03 | 711.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-10 11:00:00 | 711.75 | 715.03 | 711.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-10 11:15:00 | 719.35 | 715.89 | 712.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-10 11:45:00 | 712.80 | 715.89 | 712.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-10 14:15:00 | 716.55 | 716.08 | 713.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-10 14:30:00 | 713.50 | 716.08 | 713.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 709.10 | 714.91 | 713.12 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 10:15:00 | 713.60 | 714.91 | 713.12 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 10:45:00 | 718.00 | 715.40 | 713.51 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-13 14:15:00 | 710.35 | 712.45 | 712.55 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 62 — SELL (started 2026-04-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-13 14:15:00 | 710.35 | 712.45 | 712.55 | EMA200 below EMA400 |

### Cycle 63 — BUY (started 2026-04-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-15 09:15:00 | 731.55 | 715.81 | 714.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-16 10:15:00 | 739.15 | 729.02 | 722.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-17 12:15:00 | 742.05 | 742.26 | 735.08 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-17 12:30:00 | 741.70 | 742.26 | 735.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-20 09:15:00 | 728.85 | 741.59 | 737.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-20 10:00:00 | 728.85 | 741.59 | 737.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-20 10:15:00 | 734.10 | 740.10 | 736.97 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-20 11:15:00 | 738.05 | 740.10 | 736.97 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-20 13:15:00 | 738.90 | 739.28 | 737.13 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-20 13:45:00 | 738.15 | 739.30 | 737.33 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-21 09:30:00 | 739.85 | 738.25 | 737.20 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-22 09:15:00 | 727.75 | 740.00 | 739.37 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-04-22 09:15:00 | 727.75 | 740.00 | 739.37 | SL hit (close<static) qty=1.00 sl=728.20 alert=retest2 |

### Cycle 64 — SELL (started 2026-04-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-22 10:15:00 | 726.65 | 737.33 | 738.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-22 11:15:00 | 722.45 | 734.35 | 736.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-22 13:15:00 | 736.30 | 733.73 | 736.01 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-22 13:15:00 | 736.30 | 733.73 | 736.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-22 13:15:00 | 736.30 | 733.73 | 736.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-22 13:45:00 | 740.00 | 733.73 | 736.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-22 14:15:00 | 735.40 | 734.06 | 735.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-22 14:45:00 | 737.10 | 734.06 | 735.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-22 15:15:00 | 735.80 | 734.41 | 735.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-23 09:15:00 | 735.70 | 734.41 | 735.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-23 09:15:00 | 739.60 | 735.45 | 736.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-23 09:45:00 | 741.85 | 735.45 | 736.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-23 10:15:00 | 740.20 | 736.40 | 736.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-23 11:00:00 | 740.20 | 736.40 | 736.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 65 — BUY (started 2026-04-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-23 11:15:00 | 740.35 | 737.19 | 736.97 | EMA200 above EMA400 |

### Cycle 66 — SELL (started 2026-04-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-23 15:15:00 | 732.00 | 736.32 | 736.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-24 09:15:00 | 719.50 | 732.96 | 735.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-27 09:15:00 | 726.05 | 715.25 | 722.33 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-27 09:15:00 | 726.05 | 715.25 | 722.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 09:15:00 | 726.05 | 715.25 | 722.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 09:45:00 | 728.20 | 715.25 | 722.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 10:15:00 | 729.15 | 718.03 | 722.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 10:45:00 | 729.60 | 718.03 | 722.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 67 — BUY (started 2026-04-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 13:15:00 | 739.80 | 728.20 | 726.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-29 09:15:00 | 750.85 | 737.65 | 733.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-29 14:15:00 | 740.55 | 740.95 | 736.98 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-29 15:00:00 | 740.55 | 740.95 | 736.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 09:15:00 | 747.25 | 741.90 | 738.08 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-30 12:30:00 | 754.30 | 744.84 | 740.48 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-30 13:45:00 | 756.60 | 747.87 | 742.25 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-06 13:15:00 | 788.50 | 773.38 | 768.48 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-05-06 14:15:00 | 748.65 | 762.30 | 763.92 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 68 — SELL (started 2026-05-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-06 14:15:00 | 748.65 | 762.30 | 763.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-05-07 09:15:00 | 720.50 | 751.89 | 758.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-05-08 10:15:00 | 728.85 | 727.64 | 738.89 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-05-08 10:45:00 | 730.85 | 727.64 | 738.89 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 14:15:00 | 727.10 | 728.12 | 735.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-08 14:30:00 | 734.45 | 728.12 | 735.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-05-23 11:30:00 | 1332.50 | 2025-05-23 13:15:00 | 1337.00 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest2 | 2025-06-03 11:30:00 | 1330.10 | 2025-06-09 09:15:00 | 1356.20 | STOP_HIT | 1.00 | -1.96% |
| SELL | retest2 | 2025-06-03 14:15:00 | 1328.10 | 2025-06-09 09:15:00 | 1356.20 | STOP_HIT | 1.00 | -2.12% |
| BUY | retest2 | 2025-06-18 12:45:00 | 1413.90 | 2025-06-19 09:15:00 | 1386.80 | STOP_HIT | 1.00 | -1.92% |
| BUY | retest2 | 2025-06-18 14:30:00 | 1418.20 | 2025-06-19 09:15:00 | 1386.80 | STOP_HIT | 1.00 | -2.21% |
| SELL | retest1 | 2025-06-27 14:45:00 | 1268.80 | 2025-07-03 10:15:00 | 1259.70 | STOP_HIT | 1.00 | 0.72% |
| SELL | retest1 | 2025-06-30 10:15:00 | 1273.30 | 2025-07-03 10:15:00 | 1259.70 | STOP_HIT | 1.00 | 1.07% |
| BUY | retest2 | 2025-07-04 13:45:00 | 1262.80 | 2025-07-09 13:15:00 | 1261.60 | STOP_HIT | 1.00 | -0.10% |
| BUY | retest2 | 2025-07-04 14:15:00 | 1262.80 | 2025-07-09 13:15:00 | 1261.60 | STOP_HIT | 1.00 | -0.10% |
| BUY | retest2 | 2025-07-07 09:45:00 | 1263.30 | 2025-07-09 13:15:00 | 1261.60 | STOP_HIT | 1.00 | -0.13% |
| BUY | retest2 | 2025-07-09 13:15:00 | 1262.80 | 2025-07-09 13:15:00 | 1261.60 | STOP_HIT | 1.00 | -0.10% |
| SELL | retest2 | 2025-07-22 12:00:00 | 1254.00 | 2025-07-23 10:15:00 | 1278.90 | STOP_HIT | 1.00 | -1.99% |
| SELL | retest2 | 2025-07-22 12:45:00 | 1255.70 | 2025-07-23 10:15:00 | 1278.90 | STOP_HIT | 1.00 | -1.85% |
| SELL | retest2 | 2025-08-06 10:00:00 | 1196.90 | 2025-08-12 09:15:00 | 1214.90 | STOP_HIT | 1.00 | -1.50% |
| SELL | retest2 | 2025-08-06 11:30:00 | 1200.10 | 2025-08-12 09:15:00 | 1214.90 | STOP_HIT | 1.00 | -1.23% |
| SELL | retest2 | 2025-08-07 09:30:00 | 1198.20 | 2025-08-12 09:15:00 | 1214.90 | STOP_HIT | 1.00 | -1.39% |
| SELL | retest2 | 2025-08-08 10:00:00 | 1200.20 | 2025-08-12 09:15:00 | 1214.90 | STOP_HIT | 1.00 | -1.22% |
| SELL | retest2 | 2025-08-19 13:15:00 | 1200.60 | 2025-08-20 11:15:00 | 1218.00 | STOP_HIT | 1.00 | -1.45% |
| SELL | retest2 | 2025-08-20 09:15:00 | 1197.70 | 2025-08-20 11:15:00 | 1218.00 | STOP_HIT | 1.00 | -1.69% |
| BUY | retest2 | 2025-09-03 09:15:00 | 1211.90 | 2025-09-12 12:15:00 | 1251.90 | STOP_HIT | 1.00 | 3.30% |
| BUY | retest2 | 2025-09-03 11:00:00 | 1211.20 | 2025-09-12 12:15:00 | 1251.90 | STOP_HIT | 1.00 | 3.36% |
| SELL | retest2 | 2025-09-26 09:15:00 | 1251.00 | 2025-09-30 12:15:00 | 1188.45 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-26 09:15:00 | 1251.00 | 2025-09-30 13:15:00 | 1125.90 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-10-14 11:15:00 | 1145.00 | 2025-10-15 09:15:00 | 1164.90 | STOP_HIT | 1.00 | -1.74% |
| SELL | retest2 | 2025-10-14 12:15:00 | 1144.80 | 2025-10-15 09:15:00 | 1164.90 | STOP_HIT | 1.00 | -1.76% |
| SELL | retest2 | 2025-10-14 13:00:00 | 1142.60 | 2025-10-15 09:15:00 | 1164.90 | STOP_HIT | 1.00 | -1.95% |
| BUY | retest2 | 2025-10-29 14:15:00 | 1197.30 | 2025-10-30 09:15:00 | 1170.00 | STOP_HIT | 1.00 | -2.28% |
| SELL | retest2 | 2025-11-04 09:45:00 | 1160.70 | 2025-11-10 10:15:00 | 1172.20 | STOP_HIT | 1.00 | -0.99% |
| SELL | retest2 | 2025-11-19 15:00:00 | 1204.20 | 2025-11-27 09:15:00 | 1205.60 | STOP_HIT | 1.00 | -0.12% |
| SELL | retest2 | 2025-11-20 14:15:00 | 1204.20 | 2025-11-27 09:15:00 | 1205.60 | STOP_HIT | 1.00 | -0.12% |
| SELL | retest2 | 2025-11-24 11:45:00 | 1194.20 | 2025-11-27 09:15:00 | 1205.60 | STOP_HIT | 1.00 | -0.95% |
| BUY | retest1 | 2025-12-03 11:15:00 | 1263.50 | 2025-12-05 11:15:00 | 1259.50 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2025-12-04 09:15:00 | 1268.00 | 2025-12-05 11:15:00 | 1259.50 | STOP_HIT | 1.00 | -0.67% |
| BUY | retest2 | 2025-12-05 13:45:00 | 1269.20 | 2025-12-08 09:15:00 | 1249.50 | STOP_HIT | 1.00 | -1.55% |
| SELL | retest2 | 2026-01-01 12:30:00 | 1162.00 | 2026-01-07 09:15:00 | 1193.60 | STOP_HIT | 1.00 | -2.72% |
| SELL | retest2 | 2026-01-02 09:15:00 | 1161.30 | 2026-01-07 09:15:00 | 1193.60 | STOP_HIT | 1.00 | -2.78% |
| SELL | retest2 | 2026-01-02 10:30:00 | 1157.00 | 2026-01-07 09:15:00 | 1193.60 | STOP_HIT | 1.00 | -3.16% |
| SELL | retest2 | 2026-01-05 09:15:00 | 1157.60 | 2026-01-07 09:15:00 | 1193.60 | STOP_HIT | 1.00 | -3.11% |
| SELL | retest2 | 2026-01-05 10:30:00 | 1151.60 | 2026-01-07 09:15:00 | 1193.60 | STOP_HIT | 1.00 | -3.65% |
| SELL | retest2 | 2026-01-05 14:15:00 | 1151.20 | 2026-01-07 09:15:00 | 1193.60 | STOP_HIT | 1.00 | -3.68% |
| SELL | retest2 | 2026-01-05 15:00:00 | 1151.50 | 2026-01-07 09:15:00 | 1193.60 | STOP_HIT | 1.00 | -3.66% |
| BUY | retest2 | 2026-01-09 09:45:00 | 1186.90 | 2026-01-12 09:15:00 | 1163.40 | STOP_HIT | 1.00 | -1.98% |
| BUY | retest2 | 2026-01-09 10:15:00 | 1189.20 | 2026-01-12 09:15:00 | 1163.40 | STOP_HIT | 1.00 | -2.17% |
| BUY | retest2 | 2026-01-09 13:45:00 | 1187.10 | 2026-01-12 09:15:00 | 1163.40 | STOP_HIT | 1.00 | -2.00% |
| BUY | retest2 | 2026-01-09 14:30:00 | 1185.80 | 2026-01-12 09:15:00 | 1163.40 | STOP_HIT | 1.00 | -1.89% |
| SELL | retest2 | 2026-01-22 11:30:00 | 1116.50 | 2026-01-29 12:15:00 | 1060.67 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-22 12:45:00 | 1117.40 | 2026-01-29 12:15:00 | 1061.53 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-22 14:00:00 | 1110.10 | 2026-01-29 12:15:00 | 1060.87 | PARTIAL | 0.50 | 4.44% |
| SELL | retest2 | 2026-01-23 09:30:00 | 1116.70 | 2026-01-29 13:15:00 | 1054.59 | PARTIAL | 0.50 | 5.56% |
| SELL | retest2 | 2026-01-28 10:15:00 | 1095.90 | 2026-01-29 14:15:00 | 1041.11 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-29 09:15:00 | 1097.70 | 2026-01-29 14:15:00 | 1042.82 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-22 11:30:00 | 1116.50 | 2026-02-01 11:15:00 | 1004.85 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-01-22 12:45:00 | 1117.40 | 2026-02-01 11:15:00 | 1005.66 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-01-22 14:00:00 | 1110.10 | 2026-02-01 11:15:00 | 999.09 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-01-23 09:30:00 | 1116.70 | 2026-02-01 11:15:00 | 1005.03 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-01-28 10:15:00 | 1095.90 | 2026-02-01 12:15:00 | 986.31 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-01-29 09:15:00 | 1097.70 | 2026-02-01 12:15:00 | 987.93 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-02-17 13:45:00 | 879.00 | 2026-02-20 09:15:00 | 835.05 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-17 13:45:00 | 879.00 | 2026-02-24 09:15:00 | 791.10 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-03-24 10:15:00 | 652.95 | 2026-03-24 14:15:00 | 666.05 | STOP_HIT | 1.00 | -2.01% |
| BUY | retest2 | 2026-04-13 10:15:00 | 713.60 | 2026-04-13 14:15:00 | 710.35 | STOP_HIT | 1.00 | -0.46% |
| BUY | retest2 | 2026-04-13 10:45:00 | 718.00 | 2026-04-13 14:15:00 | 710.35 | STOP_HIT | 1.00 | -1.07% |
| BUY | retest2 | 2026-04-20 11:15:00 | 738.05 | 2026-04-22 09:15:00 | 727.75 | STOP_HIT | 1.00 | -1.40% |
| BUY | retest2 | 2026-04-20 13:15:00 | 738.90 | 2026-04-22 09:15:00 | 727.75 | STOP_HIT | 1.00 | -1.51% |
| BUY | retest2 | 2026-04-20 13:45:00 | 738.15 | 2026-04-22 09:15:00 | 727.75 | STOP_HIT | 1.00 | -1.41% |
| BUY | retest2 | 2026-04-21 09:30:00 | 739.85 | 2026-04-22 09:15:00 | 727.75 | STOP_HIT | 1.00 | -1.64% |
| BUY | retest2 | 2026-04-30 12:30:00 | 754.30 | 2026-05-06 14:15:00 | 748.65 | STOP_HIT | 1.00 | -0.75% |
| BUY | retest2 | 2026-04-30 13:45:00 | 756.60 | 2026-05-06 14:15:00 | 748.65 | STOP_HIT | 1.00 | -1.05% |
| BUY | retest2 | 2026-05-06 13:15:00 | 788.50 | 2026-05-06 14:15:00 | 748.65 | STOP_HIT | 1.00 | -5.05% |
