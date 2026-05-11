# Mahindra & Mahindra Ltd. (M&M)

## Backtest Summary

- **Window:** 2023-03-14 09:15:00 → 2026-05-08 15:15:00 (5436 bars)
- **Last close:** 3331.50
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 222 |
| ALERT1 | 150 |
| ALERT2 | 149 |
| ALERT2_SKIP | 75 |
| ALERT3 | 402 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 8 |
| ENTRY2 | 179 |
| PARTIAL | 12 |
| TARGET_HIT | 7 |
| STOP_HIT | 180 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 199 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 55 / 144
- **Target hits / Stop hits / Partials:** 7 / 180 / 12
- **Avg / median % per leg:** 0.17% / -0.50%
- **Sum % (uncompounded):** 34.78%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 88 | 27 | 30.7% | 6 | 80 | 2 | 0.61% | 53.3% |
| BUY @ 2nd Alert (retest1) | 9 | 4 | 44.4% | 0 | 7 | 2 | 1.27% | 11.5% |
| BUY @ 3rd Alert (retest2) | 79 | 23 | 29.1% | 6 | 73 | 0 | 0.53% | 41.8% |
| SELL (all) | 111 | 28 | 25.2% | 1 | 100 | 10 | -0.17% | -18.5% |
| SELL @ 2nd Alert (retest1) | 2 | 2 | 100.0% | 1 | 0 | 1 | 7.50% | 15.0% |
| SELL @ 3rd Alert (retest2) | 109 | 26 | 23.9% | 0 | 100 | 9 | -0.31% | -33.5% |
| retest1 (combined) | 11 | 6 | 54.5% | 1 | 7 | 3 | 2.41% | 26.5% |
| retest2 (combined) | 188 | 49 | 26.1% | 6 | 173 | 9 | 0.04% | 8.3% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-05-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-17 13:15:00 | 1262.75 | 1267.88 | 1268.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-05-18 09:15:00 | 1255.00 | 1263.98 | 1266.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-05-19 11:15:00 | 1252.15 | 1251.42 | 1257.01 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-19 13:15:00 | 1256.25 | 1252.70 | 1256.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-19 13:15:00 | 1256.25 | 1252.70 | 1256.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-05-19 14:00:00 | 1256.25 | 1252.70 | 1256.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-19 14:15:00 | 1260.70 | 1254.30 | 1257.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-05-19 14:45:00 | 1260.95 | 1254.30 | 1257.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-19 15:15:00 | 1258.00 | 1255.04 | 1257.10 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-05-22 09:15:00 | 1257.00 | 1255.04 | 1257.10 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-05-22 10:15:00 | 1265.45 | 1258.08 | 1258.19 | SL hit (close>static) qty=1.00 sl=1261.15 alert=retest2 |

### Cycle 2 — BUY (started 2023-05-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-22 11:15:00 | 1268.60 | 1260.19 | 1259.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-05-24 13:15:00 | 1277.40 | 1268.17 | 1265.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-05-25 09:15:00 | 1266.80 | 1270.28 | 1267.37 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-25 09:15:00 | 1266.80 | 1270.28 | 1267.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-25 09:15:00 | 1266.80 | 1270.28 | 1267.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-25 09:30:00 | 1268.95 | 1270.28 | 1267.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-25 10:15:00 | 1262.40 | 1268.71 | 1266.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-25 10:30:00 | 1259.20 | 1268.71 | 1266.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-25 11:15:00 | 1265.00 | 1267.97 | 1266.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-25 11:30:00 | 1262.50 | 1267.97 | 1266.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-25 12:15:00 | 1265.80 | 1267.53 | 1266.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-25 12:30:00 | 1265.05 | 1267.53 | 1266.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-26 13:15:00 | 1276.75 | 1280.28 | 1275.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-26 14:00:00 | 1276.75 | 1280.28 | 1275.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-26 14:15:00 | 1282.35 | 1280.69 | 1275.75 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-05-29 09:15:00 | 1334.45 | 1279.96 | 1275.87 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-06-08 15:15:00 | 1379.95 | 1388.89 | 1389.87 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 3 — SELL (started 2023-06-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-08 15:15:00 | 1379.95 | 1388.89 | 1389.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-09 09:15:00 | 1365.05 | 1384.13 | 1387.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-12 11:15:00 | 1377.00 | 1372.89 | 1377.59 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-12 11:15:00 | 1377.00 | 1372.89 | 1377.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-12 11:15:00 | 1377.00 | 1372.89 | 1377.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-12 12:00:00 | 1377.00 | 1372.89 | 1377.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-12 12:15:00 | 1377.40 | 1373.79 | 1377.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-12 13:00:00 | 1377.40 | 1373.79 | 1377.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-12 13:15:00 | 1383.00 | 1375.63 | 1378.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-12 14:00:00 | 1383.00 | 1375.63 | 1378.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-12 14:15:00 | 1381.30 | 1376.77 | 1378.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-12 14:45:00 | 1383.10 | 1376.77 | 1378.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-13 13:15:00 | 1373.55 | 1376.02 | 1377.42 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-13 14:15:00 | 1370.00 | 1376.02 | 1377.42 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-14 09:15:00 | 1370.90 | 1375.31 | 1376.84 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-14 11:00:00 | 1370.80 | 1373.62 | 1375.76 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-06-14 12:15:00 | 1379.75 | 1375.01 | 1376.03 | SL hit (close>static) qty=1.00 sl=1377.80 alert=retest2 |

### Cycle 4 — BUY (started 2023-06-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-14 15:15:00 | 1378.15 | 1376.91 | 1376.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-15 09:15:00 | 1388.20 | 1379.17 | 1377.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-19 13:15:00 | 1404.60 | 1405.86 | 1399.47 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-19 13:15:00 | 1404.60 | 1405.86 | 1399.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-19 13:15:00 | 1404.60 | 1405.86 | 1399.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-19 13:30:00 | 1401.50 | 1405.86 | 1399.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-19 15:15:00 | 1400.50 | 1404.41 | 1399.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-20 09:15:00 | 1385.55 | 1404.41 | 1399.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-20 09:15:00 | 1385.15 | 1400.56 | 1398.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-20 10:15:00 | 1381.10 | 1400.56 | 1398.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-20 10:15:00 | 1386.85 | 1397.82 | 1397.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-20 10:30:00 | 1382.50 | 1397.82 | 1397.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 5 — SELL (started 2023-06-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-20 11:15:00 | 1384.70 | 1395.19 | 1396.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-21 13:15:00 | 1377.95 | 1387.62 | 1391.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-22 09:15:00 | 1389.75 | 1384.36 | 1388.72 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-22 09:15:00 | 1389.75 | 1384.36 | 1388.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-22 09:15:00 | 1389.75 | 1384.36 | 1388.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-22 10:00:00 | 1389.75 | 1384.36 | 1388.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-22 10:15:00 | 1380.00 | 1383.49 | 1387.93 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-22 11:15:00 | 1379.05 | 1383.49 | 1387.93 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-22 12:00:00 | 1378.00 | 1382.39 | 1387.03 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-22 12:30:00 | 1378.90 | 1382.41 | 1386.61 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-22 14:15:00 | 1379.00 | 1382.89 | 1386.45 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-26 09:15:00 | 1390.00 | 1377.88 | 1380.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-26 09:45:00 | 1389.70 | 1377.88 | 1380.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-26 10:15:00 | 1392.85 | 1380.88 | 1381.40 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2023-06-26 10:15:00 | 1392.85 | 1380.88 | 1381.40 | SL hit (close>static) qty=1.00 sl=1391.80 alert=retest2 |

### Cycle 6 — BUY (started 2023-06-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-26 11:15:00 | 1391.60 | 1383.02 | 1382.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-26 14:15:00 | 1395.05 | 1387.52 | 1384.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-27 12:15:00 | 1398.35 | 1399.18 | 1392.42 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-06-27 13:00:00 | 1398.35 | 1399.18 | 1392.42 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-28 14:15:00 | 1394.80 | 1401.86 | 1398.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-28 15:00:00 | 1394.80 | 1401.86 | 1398.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-28 15:15:00 | 1397.00 | 1400.89 | 1398.30 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-30 09:15:00 | 1416.25 | 1400.89 | 1398.30 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2023-07-07 09:15:00 | 1557.88 | 1522.78 | 1496.76 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 7 — SELL (started 2023-07-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-14 12:15:00 | 1557.30 | 1563.42 | 1564.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-14 13:15:00 | 1549.40 | 1560.62 | 1562.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-18 14:15:00 | 1536.00 | 1535.31 | 1542.47 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-07-18 15:00:00 | 1536.00 | 1535.31 | 1542.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-19 11:15:00 | 1539.45 | 1534.75 | 1539.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-19 12:00:00 | 1539.45 | 1534.75 | 1539.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-19 12:15:00 | 1541.20 | 1536.04 | 1539.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-19 12:30:00 | 1541.65 | 1536.04 | 1539.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-19 13:15:00 | 1537.20 | 1536.27 | 1539.72 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-20 09:30:00 | 1532.50 | 1537.74 | 1539.72 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-07-20 13:15:00 | 1547.25 | 1540.05 | 1540.13 | SL hit (close>static) qty=1.00 sl=1542.95 alert=retest2 |

### Cycle 8 — BUY (started 2023-07-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-20 14:15:00 | 1546.45 | 1541.33 | 1540.71 | EMA200 above EMA400 |

### Cycle 9 — SELL (started 2023-07-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-21 13:15:00 | 1530.05 | 1539.49 | 1540.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-21 14:15:00 | 1523.35 | 1536.26 | 1538.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-24 09:15:00 | 1549.20 | 1536.41 | 1538.31 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-24 09:15:00 | 1549.20 | 1536.41 | 1538.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-24 09:15:00 | 1549.20 | 1536.41 | 1538.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-24 09:45:00 | 1545.90 | 1536.41 | 1538.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-24 10:15:00 | 1547.95 | 1538.72 | 1539.19 | EMA400 retest candle locked (from downside) |

### Cycle 10 — BUY (started 2023-07-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-24 11:15:00 | 1543.65 | 1539.70 | 1539.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-25 09:15:00 | 1557.90 | 1547.16 | 1543.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-26 10:15:00 | 1557.65 | 1559.85 | 1553.57 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-07-26 11:00:00 | 1557.65 | 1559.85 | 1553.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-26 11:15:00 | 1540.80 | 1556.04 | 1552.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-26 12:00:00 | 1540.80 | 1556.04 | 1552.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-26 12:15:00 | 1537.75 | 1552.38 | 1551.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-26 12:45:00 | 1533.90 | 1552.38 | 1551.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 11 — SELL (started 2023-07-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-26 14:15:00 | 1546.05 | 1550.04 | 1550.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-27 09:15:00 | 1445.00 | 1528.37 | 1540.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-28 14:15:00 | 1469.20 | 1468.47 | 1486.84 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-07-28 15:00:00 | 1469.20 | 1468.47 | 1486.84 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-31 14:15:00 | 1478.30 | 1470.46 | 1478.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-31 15:00:00 | 1478.30 | 1470.46 | 1478.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-31 15:15:00 | 1474.00 | 1471.17 | 1478.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-01 09:15:00 | 1489.05 | 1471.17 | 1478.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-01 09:15:00 | 1487.40 | 1474.42 | 1479.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-01 10:00:00 | 1487.40 | 1474.42 | 1479.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-01 10:15:00 | 1490.85 | 1477.70 | 1480.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-01 11:00:00 | 1490.85 | 1477.70 | 1480.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-01 11:15:00 | 1493.65 | 1480.89 | 1481.36 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-01 12:15:00 | 1487.70 | 1480.89 | 1481.36 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-08-01 12:15:00 | 1488.50 | 1482.41 | 1482.01 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 12 — BUY (started 2023-08-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-01 12:15:00 | 1488.50 | 1482.41 | 1482.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-01 14:15:00 | 1494.90 | 1486.52 | 1484.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-02 09:15:00 | 1486.95 | 1487.48 | 1484.97 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-02 09:15:00 | 1486.95 | 1487.48 | 1484.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-02 09:15:00 | 1486.95 | 1487.48 | 1484.97 | EMA400 retest candle locked (from upside) |

### Cycle 13 — SELL (started 2023-08-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-02 11:15:00 | 1473.90 | 1483.38 | 1483.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-02 13:15:00 | 1468.10 | 1479.00 | 1481.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-02 14:15:00 | 1487.20 | 1480.64 | 1481.90 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-02 14:15:00 | 1487.20 | 1480.64 | 1481.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-02 14:15:00 | 1487.20 | 1480.64 | 1481.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-02 15:00:00 | 1487.20 | 1480.64 | 1481.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-02 15:15:00 | 1481.80 | 1480.87 | 1481.89 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-03 09:45:00 | 1478.60 | 1480.41 | 1481.59 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-08-03 11:15:00 | 1484.45 | 1482.48 | 1482.40 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 14 — BUY (started 2023-08-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-03 11:15:00 | 1484.45 | 1482.48 | 1482.40 | EMA200 above EMA400 |

### Cycle 15 — SELL (started 2023-08-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-03 12:15:00 | 1469.90 | 1479.96 | 1481.26 | EMA200 below EMA400 |

### Cycle 16 — BUY (started 2023-08-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-07 09:15:00 | 1513.65 | 1477.49 | 1476.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-07 10:15:00 | 1515.80 | 1485.15 | 1480.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-08 09:15:00 | 1510.35 | 1510.92 | 1497.57 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-08-08 09:45:00 | 1512.00 | 1510.92 | 1497.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-08 10:15:00 | 1492.10 | 1507.16 | 1497.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-08 11:00:00 | 1492.10 | 1507.16 | 1497.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-08 11:15:00 | 1499.25 | 1505.57 | 1497.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-08 11:30:00 | 1488.75 | 1505.57 | 1497.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-08 12:15:00 | 1490.00 | 1502.46 | 1496.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-08 13:00:00 | 1490.00 | 1502.46 | 1496.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-08 13:15:00 | 1499.60 | 1501.89 | 1496.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-08 13:30:00 | 1494.65 | 1501.89 | 1496.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-08 14:15:00 | 1498.75 | 1501.26 | 1497.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-08 14:45:00 | 1497.50 | 1501.26 | 1497.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-08 15:15:00 | 1500.00 | 1501.01 | 1497.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-09 09:15:00 | 1512.60 | 1501.01 | 1497.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-09 09:15:00 | 1519.80 | 1504.77 | 1499.37 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-09 13:15:00 | 1521.45 | 1511.90 | 1504.33 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-08-18 15:15:00 | 1547.10 | 1556.59 | 1556.65 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 17 — SELL (started 2023-08-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-18 15:15:00 | 1547.10 | 1556.59 | 1556.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-21 09:15:00 | 1538.30 | 1552.94 | 1554.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-22 10:15:00 | 1547.75 | 1544.36 | 1548.16 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-22 10:15:00 | 1547.75 | 1544.36 | 1548.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-22 10:15:00 | 1547.75 | 1544.36 | 1548.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-22 11:00:00 | 1547.75 | 1544.36 | 1548.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-22 11:15:00 | 1550.35 | 1545.56 | 1548.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-22 11:30:00 | 1551.60 | 1545.56 | 1548.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-22 12:15:00 | 1551.75 | 1546.80 | 1548.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-22 12:30:00 | 1552.50 | 1546.80 | 1548.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-22 15:15:00 | 1550.00 | 1548.50 | 1549.09 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-23 09:15:00 | 1544.90 | 1548.50 | 1549.09 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-23 09:45:00 | 1544.10 | 1548.47 | 1549.03 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-23 10:15:00 | 1545.00 | 1548.47 | 1549.03 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-23 15:00:00 | 1541.85 | 1543.88 | 1546.16 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-24 09:15:00 | 1550.90 | 1545.43 | 1546.48 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-24 10:30:00 | 1541.65 | 1543.60 | 1545.55 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-24 12:45:00 | 1540.45 | 1542.10 | 1544.51 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-08-28 13:15:00 | 1548.20 | 1538.43 | 1537.92 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 18 — BUY (started 2023-08-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-28 13:15:00 | 1548.20 | 1538.43 | 1537.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-29 13:15:00 | 1552.60 | 1546.54 | 1543.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-31 15:15:00 | 1577.70 | 1579.62 | 1571.69 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-01 09:15:00 | 1599.60 | 1579.62 | 1571.69 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-01 12:45:00 | 1587.00 | 1586.51 | 1578.02 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-04 09:15:00 | 1578.00 | 1587.95 | 1581.86 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2023-09-04 09:15:00 | 1578.00 | 1587.95 | 1581.86 | SL hit (close<ema400) qty=1.00 sl=1581.86 alert=retest1 |

### Cycle 19 — SELL (started 2023-09-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-05 09:15:00 | 1573.65 | 1578.35 | 1578.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-05 13:15:00 | 1566.80 | 1574.46 | 1576.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-06 11:15:00 | 1574.05 | 1572.55 | 1574.71 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-06 11:15:00 | 1574.05 | 1572.55 | 1574.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-06 11:15:00 | 1574.05 | 1572.55 | 1574.71 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-06 12:30:00 | 1563.05 | 1571.66 | 1574.11 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-09-06 14:15:00 | 1578.40 | 1571.46 | 1573.50 | SL hit (close>static) qty=1.00 sl=1575.90 alert=retest2 |

### Cycle 20 — BUY (started 2023-09-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-11 10:15:00 | 1578.15 | 1571.39 | 1570.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-11 11:15:00 | 1582.20 | 1573.55 | 1571.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-12 09:15:00 | 1578.05 | 1578.93 | 1575.48 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-12 09:15:00 | 1578.05 | 1578.93 | 1575.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-12 09:15:00 | 1578.05 | 1578.93 | 1575.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-12 09:45:00 | 1576.50 | 1578.93 | 1575.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-12 10:15:00 | 1572.70 | 1577.69 | 1575.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-12 10:30:00 | 1572.00 | 1577.69 | 1575.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 21 — SELL (started 2023-09-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-12 11:15:00 | 1537.50 | 1569.65 | 1571.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-12 12:15:00 | 1516.50 | 1559.02 | 1566.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-12 14:15:00 | 1559.10 | 1558.69 | 1565.25 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-09-12 15:00:00 | 1559.10 | 1558.69 | 1565.25 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-13 09:15:00 | 1536.10 | 1553.26 | 1561.59 | EMA400 retest candle locked (from downside) |

### Cycle 22 — BUY (started 2023-09-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-14 15:15:00 | 1573.95 | 1557.58 | 1555.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-15 09:15:00 | 1577.30 | 1561.52 | 1557.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-20 14:15:00 | 1634.85 | 1642.33 | 1624.97 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-09-20 15:00:00 | 1634.85 | 1642.33 | 1624.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-21 09:15:00 | 1613.00 | 1635.40 | 1624.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-21 10:00:00 | 1613.00 | 1635.40 | 1624.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-21 10:15:00 | 1600.75 | 1628.47 | 1622.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-21 11:00:00 | 1600.75 | 1628.47 | 1622.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 23 — SELL (started 2023-09-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-21 12:15:00 | 1583.90 | 1613.50 | 1616.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-28 09:15:00 | 1580.00 | 1590.91 | 1594.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-29 13:15:00 | 1562.05 | 1561.01 | 1570.64 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-09-29 14:00:00 | 1562.05 | 1561.01 | 1570.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-03 09:15:00 | 1550.15 | 1556.98 | 1566.22 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-03 11:00:00 | 1544.90 | 1554.56 | 1564.29 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-06 09:30:00 | 1543.35 | 1535.36 | 1536.08 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-10-06 10:15:00 | 1551.50 | 1538.59 | 1537.48 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 24 — BUY (started 2023-10-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-06 10:15:00 | 1551.50 | 1538.59 | 1537.48 | EMA200 above EMA400 |

### Cycle 25 — SELL (started 2023-10-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-09 11:15:00 | 1529.20 | 1538.82 | 1539.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-09 12:15:00 | 1519.60 | 1534.98 | 1537.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-10 09:15:00 | 1537.50 | 1528.14 | 1532.71 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-10 09:15:00 | 1537.50 | 1528.14 | 1532.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-10 09:15:00 | 1537.50 | 1528.14 | 1532.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-10 09:45:00 | 1535.15 | 1528.14 | 1532.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-10 10:15:00 | 1529.45 | 1528.40 | 1532.42 | EMA400 retest candle locked (from downside) |

### Cycle 26 — BUY (started 2023-10-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-11 09:15:00 | 1547.15 | 1535.87 | 1534.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-11 12:15:00 | 1558.85 | 1543.05 | 1538.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-13 09:15:00 | 1560.50 | 1561.63 | 1553.95 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-13 09:15:00 | 1560.50 | 1561.63 | 1553.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-13 09:15:00 | 1560.50 | 1561.63 | 1553.95 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-13 13:30:00 | 1571.00 | 1562.54 | 1556.57 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-16 13:30:00 | 1573.75 | 1565.24 | 1560.81 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-18 11:15:00 | 1569.40 | 1572.73 | 1569.70 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-18 12:00:00 | 1570.20 | 1572.22 | 1569.75 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-18 12:15:00 | 1574.85 | 1572.75 | 1570.21 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-18 13:45:00 | 1575.40 | 1573.90 | 1570.96 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-18 14:45:00 | 1576.60 | 1574.20 | 1571.37 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-10-19 09:15:00 | 1566.25 | 1573.06 | 1571.37 | SL hit (close<static) qty=1.00 sl=1568.20 alert=retest2 |

### Cycle 27 — SELL (started 2023-10-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-19 15:15:00 | 1567.60 | 1571.21 | 1571.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-20 09:15:00 | 1565.70 | 1570.11 | 1570.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-23 09:15:00 | 1569.65 | 1561.23 | 1564.48 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-23 09:15:00 | 1569.65 | 1561.23 | 1564.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-23 09:15:00 | 1569.65 | 1561.23 | 1564.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-23 09:30:00 | 1571.00 | 1561.23 | 1564.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-23 10:15:00 | 1563.10 | 1561.60 | 1564.35 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-23 11:15:00 | 1560.10 | 1561.60 | 1564.35 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-23 13:00:00 | 1562.10 | 1561.67 | 1563.91 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-23 15:15:00 | 1561.00 | 1563.89 | 1564.63 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-25 10:00:00 | 1561.25 | 1562.90 | 1564.02 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-25 10:15:00 | 1571.50 | 1564.62 | 1564.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-25 11:00:00 | 1571.50 | 1564.62 | 1564.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2023-10-25 11:15:00 | 1568.65 | 1565.42 | 1565.06 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 28 — BUY (started 2023-10-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-25 11:15:00 | 1568.65 | 1565.42 | 1565.06 | EMA200 above EMA400 |

### Cycle 29 — SELL (started 2023-10-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-25 12:15:00 | 1548.10 | 1561.96 | 1563.52 | EMA200 below EMA400 |

### Cycle 30 — BUY (started 2023-10-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-25 15:15:00 | 1570.95 | 1564.74 | 1564.43 | EMA200 above EMA400 |

### Cycle 31 — SELL (started 2023-10-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-26 09:15:00 | 1531.15 | 1558.02 | 1561.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-26 11:15:00 | 1515.80 | 1544.55 | 1554.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-27 10:15:00 | 1523.25 | 1522.57 | 1536.65 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-10-27 11:00:00 | 1523.25 | 1522.57 | 1536.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-02 14:15:00 | 1468.35 | 1462.77 | 1470.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-02 15:00:00 | 1468.35 | 1462.77 | 1470.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-02 15:15:00 | 1475.25 | 1465.26 | 1470.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-03 09:15:00 | 1478.20 | 1465.26 | 1470.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-03 09:15:00 | 1481.75 | 1468.56 | 1471.70 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-03 11:00:00 | 1476.10 | 1470.07 | 1472.10 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-03 11:45:00 | 1472.80 | 1470.03 | 1471.90 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-06 09:45:00 | 1476.05 | 1470.23 | 1471.05 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-11-06 10:15:00 | 1478.15 | 1471.81 | 1471.70 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 32 — BUY (started 2023-11-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-06 10:15:00 | 1478.15 | 1471.81 | 1471.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-06 11:15:00 | 1479.95 | 1473.44 | 1472.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-07 11:15:00 | 1481.50 | 1484.61 | 1479.66 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-07 11:15:00 | 1481.50 | 1484.61 | 1479.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-07 11:15:00 | 1481.50 | 1484.61 | 1479.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-07 12:00:00 | 1481.50 | 1484.61 | 1479.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-07 14:15:00 | 1480.00 | 1484.74 | 1481.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-07 15:00:00 | 1480.00 | 1484.74 | 1481.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-07 15:15:00 | 1477.00 | 1483.19 | 1480.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-08 09:15:00 | 1477.80 | 1483.19 | 1480.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-08 09:15:00 | 1481.05 | 1482.77 | 1480.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-08 09:45:00 | 1481.00 | 1482.77 | 1480.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-08 10:15:00 | 1478.45 | 1481.90 | 1480.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-08 11:00:00 | 1478.45 | 1481.90 | 1480.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-08 11:15:00 | 1480.00 | 1481.52 | 1480.45 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-08 15:00:00 | 1488.50 | 1482.33 | 1481.02 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-11-20 13:15:00 | 1554.05 | 1560.38 | 1560.52 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 33 — SELL (started 2023-11-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-20 13:15:00 | 1554.05 | 1560.38 | 1560.52 | EMA200 below EMA400 |

### Cycle 34 — BUY (started 2023-11-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-21 12:15:00 | 1563.40 | 1560.22 | 1559.95 | EMA200 above EMA400 |

### Cycle 35 — SELL (started 2023-11-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-22 09:15:00 | 1555.95 | 1559.76 | 1559.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-22 10:15:00 | 1547.80 | 1557.37 | 1558.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-23 09:15:00 | 1550.70 | 1547.95 | 1552.29 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-23 09:15:00 | 1550.70 | 1547.95 | 1552.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-23 09:15:00 | 1550.70 | 1547.95 | 1552.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-23 09:30:00 | 1557.25 | 1547.95 | 1552.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-23 10:15:00 | 1543.55 | 1547.07 | 1551.49 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-23 12:00:00 | 1541.40 | 1545.94 | 1550.57 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-23 12:45:00 | 1539.75 | 1544.02 | 1549.28 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-11-24 11:15:00 | 1553.05 | 1548.03 | 1548.99 | SL hit (close>static) qty=1.00 sl=1552.85 alert=retest2 |

### Cycle 36 — BUY (started 2023-11-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-24 13:15:00 | 1556.40 | 1550.78 | 1550.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-28 09:15:00 | 1563.10 | 1553.72 | 1551.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-28 13:15:00 | 1550.30 | 1554.57 | 1552.95 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-28 13:15:00 | 1550.30 | 1554.57 | 1552.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-28 13:15:00 | 1550.30 | 1554.57 | 1552.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-28 13:45:00 | 1547.40 | 1554.57 | 1552.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-28 14:15:00 | 1567.00 | 1557.06 | 1554.22 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-29 09:15:00 | 1578.15 | 1558.74 | 1555.25 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-12-08 12:15:00 | 1670.00 | 1682.69 | 1683.87 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 37 — SELL (started 2023-12-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-08 12:15:00 | 1670.00 | 1682.69 | 1683.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-11 11:15:00 | 1653.00 | 1668.72 | 1675.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-13 09:15:00 | 1648.40 | 1645.44 | 1655.15 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-12-13 09:45:00 | 1639.95 | 1645.44 | 1655.15 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-13 13:15:00 | 1657.10 | 1645.14 | 1651.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-13 13:30:00 | 1656.35 | 1645.14 | 1651.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-13 14:15:00 | 1666.10 | 1649.33 | 1652.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-13 15:00:00 | 1666.10 | 1649.33 | 1652.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 38 — BUY (started 2023-12-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-14 09:15:00 | 1681.00 | 1658.66 | 1656.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-14 12:15:00 | 1689.55 | 1672.64 | 1664.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-18 11:15:00 | 1714.20 | 1716.42 | 1702.23 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-12-18 12:00:00 | 1714.20 | 1716.42 | 1702.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-19 09:15:00 | 1695.95 | 1710.60 | 1704.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-19 09:30:00 | 1692.20 | 1710.60 | 1704.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-19 10:15:00 | 1696.25 | 1707.73 | 1704.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-19 10:45:00 | 1694.80 | 1707.73 | 1704.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-19 14:15:00 | 1698.95 | 1702.63 | 1702.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-19 15:00:00 | 1698.95 | 1702.63 | 1702.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 39 — SELL (started 2023-12-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-19 15:15:00 | 1699.60 | 1702.02 | 1702.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-20 10:15:00 | 1698.65 | 1700.94 | 1701.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-22 10:15:00 | 1646.85 | 1645.29 | 1660.45 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-12-22 10:45:00 | 1646.75 | 1645.29 | 1660.45 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-26 09:15:00 | 1664.65 | 1643.35 | 1651.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-26 10:00:00 | 1664.65 | 1643.35 | 1651.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-26 10:15:00 | 1663.00 | 1647.28 | 1652.88 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-12-26 12:15:00 | 1658.10 | 1650.04 | 1653.63 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-12-26 13:00:00 | 1657.75 | 1651.58 | 1654.01 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-12-26 15:15:00 | 1657.10 | 1654.51 | 1654.98 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-12-27 09:15:00 | 1678.95 | 1659.82 | 1657.33 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 40 — BUY (started 2023-12-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-27 09:15:00 | 1678.95 | 1659.82 | 1657.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-27 13:15:00 | 1680.90 | 1669.10 | 1663.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-29 14:15:00 | 1727.45 | 1728.43 | 1710.80 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-12-29 15:15:00 | 1729.00 | 1728.43 | 1710.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-01 09:15:00 | 1707.60 | 1724.36 | 1712.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-01 10:00:00 | 1707.60 | 1724.36 | 1712.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-01 10:15:00 | 1703.95 | 1720.27 | 1711.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-01 10:45:00 | 1704.55 | 1720.27 | 1711.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 41 — SELL (started 2024-01-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-02 09:15:00 | 1671.10 | 1701.38 | 1704.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-02 10:15:00 | 1658.65 | 1692.83 | 1700.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-03 13:15:00 | 1664.45 | 1664.11 | 1675.57 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-01-03 14:00:00 | 1664.45 | 1664.11 | 1675.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-04 09:15:00 | 1664.40 | 1661.63 | 1671.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-04 09:30:00 | 1672.20 | 1661.63 | 1671.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-04 10:15:00 | 1667.75 | 1662.85 | 1671.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-04 11:00:00 | 1667.75 | 1662.85 | 1671.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-04 11:15:00 | 1665.85 | 1663.45 | 1670.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-04 11:45:00 | 1669.85 | 1663.45 | 1670.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-05 09:15:00 | 1646.10 | 1651.20 | 1661.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-05 09:30:00 | 1651.60 | 1651.20 | 1661.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-08 09:15:00 | 1638.90 | 1643.88 | 1652.04 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-08 10:15:00 | 1636.30 | 1643.88 | 1652.04 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-11 09:45:00 | 1635.60 | 1630.24 | 1631.39 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-11 10:15:00 | 1635.30 | 1630.24 | 1631.39 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-01-11 12:15:00 | 1640.95 | 1633.29 | 1632.57 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 42 — BUY (started 2024-01-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-11 12:15:00 | 1640.95 | 1633.29 | 1632.57 | EMA200 above EMA400 |

### Cycle 43 — SELL (started 2024-01-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-11 14:15:00 | 1628.70 | 1631.99 | 1632.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-12 09:15:00 | 1610.30 | 1627.56 | 1630.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-12 13:15:00 | 1624.30 | 1621.47 | 1625.68 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-12 13:15:00 | 1624.30 | 1621.47 | 1625.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-12 13:15:00 | 1624.30 | 1621.47 | 1625.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-12 13:45:00 | 1625.05 | 1621.47 | 1625.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-12 14:15:00 | 1622.30 | 1621.64 | 1625.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-12 14:30:00 | 1625.65 | 1621.64 | 1625.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-12 15:15:00 | 1626.50 | 1622.61 | 1625.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-15 09:15:00 | 1633.35 | 1622.61 | 1625.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-15 09:15:00 | 1621.95 | 1622.48 | 1625.15 | EMA400 retest candle locked (from downside) |

### Cycle 44 — BUY (started 2024-01-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-15 14:15:00 | 1634.70 | 1627.78 | 1626.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-15 15:15:00 | 1638.00 | 1629.82 | 1627.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-16 12:15:00 | 1627.95 | 1630.33 | 1628.87 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-16 12:15:00 | 1627.95 | 1630.33 | 1628.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-16 12:15:00 | 1627.95 | 1630.33 | 1628.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-16 12:45:00 | 1630.00 | 1630.33 | 1628.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-16 13:15:00 | 1622.95 | 1628.85 | 1628.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-16 13:30:00 | 1624.05 | 1628.85 | 1628.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 45 — SELL (started 2024-01-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-16 14:15:00 | 1618.40 | 1626.76 | 1627.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-17 09:15:00 | 1600.00 | 1620.49 | 1624.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-18 10:15:00 | 1609.20 | 1603.75 | 1611.45 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-01-18 10:30:00 | 1608.85 | 1603.75 | 1611.45 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-18 12:15:00 | 1621.00 | 1607.46 | 1611.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-18 13:00:00 | 1621.00 | 1607.46 | 1611.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-18 13:15:00 | 1618.40 | 1609.64 | 1612.41 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-18 14:45:00 | 1612.10 | 1611.70 | 1613.09 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-01-19 09:15:00 | 1633.70 | 1616.99 | 1615.31 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 46 — BUY (started 2024-01-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-19 09:15:00 | 1633.70 | 1616.99 | 1615.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-19 12:15:00 | 1645.70 | 1627.69 | 1621.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-20 11:15:00 | 1641.60 | 1643.27 | 1633.57 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-01-20 11:30:00 | 1643.15 | 1643.27 | 1633.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-20 12:15:00 | 1642.35 | 1643.08 | 1634.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-20 12:45:00 | 1633.75 | 1643.08 | 1634.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-20 13:15:00 | 1633.25 | 1641.12 | 1634.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-20 14:00:00 | 1633.25 | 1641.12 | 1634.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-20 14:15:00 | 1623.75 | 1637.64 | 1633.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-20 15:00:00 | 1623.75 | 1637.64 | 1633.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-20 15:15:00 | 1624.00 | 1634.92 | 1632.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-23 09:15:00 | 1638.95 | 1634.92 | 1632.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-23 10:15:00 | 1637.85 | 1635.68 | 1633.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-23 10:30:00 | 1632.05 | 1635.68 | 1633.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-23 11:15:00 | 1618.50 | 1632.24 | 1631.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-23 12:00:00 | 1618.50 | 1632.24 | 1631.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 47 — SELL (started 2024-01-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-23 12:15:00 | 1599.45 | 1625.68 | 1628.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-23 15:15:00 | 1595.10 | 1611.85 | 1621.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-24 10:15:00 | 1609.10 | 1607.06 | 1617.10 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-01-24 11:00:00 | 1609.10 | 1607.06 | 1617.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-24 12:15:00 | 1620.05 | 1609.19 | 1616.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-24 13:00:00 | 1620.05 | 1609.19 | 1616.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-24 13:15:00 | 1623.95 | 1612.14 | 1616.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-24 14:00:00 | 1623.95 | 1612.14 | 1616.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-24 14:15:00 | 1626.70 | 1615.06 | 1617.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-24 15:00:00 | 1626.70 | 1615.06 | 1617.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-25 10:15:00 | 1605.60 | 1616.34 | 1618.07 | EMA400 retest candle locked (from downside) |

### Cycle 48 — BUY (started 2024-01-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-25 12:15:00 | 1631.55 | 1619.50 | 1619.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-29 09:15:00 | 1638.50 | 1628.03 | 1623.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-30 10:15:00 | 1630.00 | 1636.28 | 1631.31 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-30 10:15:00 | 1630.00 | 1636.28 | 1631.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-30 10:15:00 | 1630.00 | 1636.28 | 1631.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-30 11:00:00 | 1630.00 | 1636.28 | 1631.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-30 11:15:00 | 1643.20 | 1637.66 | 1632.39 | EMA400 retest candle locked (from upside) |

### Cycle 49 — SELL (started 2024-01-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-30 15:15:00 | 1617.20 | 1629.95 | 1630.25 | EMA200 below EMA400 |

### Cycle 50 — BUY (started 2024-01-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-31 09:15:00 | 1647.70 | 1633.50 | 1631.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-31 10:15:00 | 1649.00 | 1636.60 | 1633.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-31 13:15:00 | 1638.65 | 1640.32 | 1636.24 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-31 13:15:00 | 1638.65 | 1640.32 | 1636.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-31 13:15:00 | 1638.65 | 1640.32 | 1636.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-31 13:45:00 | 1640.75 | 1640.32 | 1636.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-31 14:15:00 | 1654.00 | 1643.05 | 1637.86 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-31 15:15:00 | 1656.80 | 1643.05 | 1637.86 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-02 09:15:00 | 1664.00 | 1658.84 | 1652.51 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-02-08 13:15:00 | 1678.35 | 1705.49 | 1707.77 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 51 — SELL (started 2024-02-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-08 13:15:00 | 1678.35 | 1705.49 | 1707.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-09 09:15:00 | 1641.00 | 1687.79 | 1698.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-12 11:15:00 | 1655.70 | 1654.76 | 1669.88 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-02-12 12:00:00 | 1655.70 | 1654.76 | 1669.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-12 13:15:00 | 1670.00 | 1658.79 | 1669.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-12 14:00:00 | 1670.00 | 1658.79 | 1669.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-12 14:15:00 | 1661.60 | 1659.35 | 1668.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-12 14:30:00 | 1673.60 | 1659.35 | 1668.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-13 09:15:00 | 1669.90 | 1661.82 | 1668.03 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-13 11:45:00 | 1651.25 | 1659.41 | 1665.97 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-14 10:00:00 | 1651.00 | 1650.36 | 1658.02 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-02-15 09:15:00 | 1724.35 | 1667.65 | 1662.27 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 52 — BUY (started 2024-02-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-15 09:15:00 | 1724.35 | 1667.65 | 1662.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-15 11:15:00 | 1745.05 | 1691.38 | 1674.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-20 09:15:00 | 1828.45 | 1837.19 | 1806.97 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-20 10:45:00 | 1844.35 | 1839.00 | 1810.54 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-23 11:15:00 | 1936.57 | 1901.70 | 1876.67 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Stop hit — per-position SL triggered | 2024-02-26 14:15:00 | 1930.05 | 1931.64 | 1912.61 | SL hit (close<ema200) qty=0.50 sl=1931.64 alert=retest1 |

### Cycle 53 — SELL (started 2024-02-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-28 14:15:00 | 1899.55 | 1920.14 | 1920.18 | EMA200 below EMA400 |

### Cycle 54 — BUY (started 2024-02-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-29 10:15:00 | 1936.00 | 1920.24 | 1919.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-29 11:15:00 | 1938.10 | 1923.81 | 1921.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-29 15:15:00 | 1932.00 | 1932.12 | 1926.95 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-01 09:15:00 | 1966.00 | 1932.12 | 1926.95 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-01 10:30:00 | 1945.20 | 1937.64 | 1930.55 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-01 11:30:00 | 1947.90 | 1940.43 | 1932.47 | BUY ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-04 09:15:00 | 1930.80 | 1953.37 | 1946.56 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-03-04 09:15:00 | 1930.80 | 1953.37 | 1946.56 | SL hit (close<ema400) qty=1.00 sl=1946.56 alert=retest1 |

### Cycle 55 — SELL (started 2024-03-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-04 13:15:00 | 1925.10 | 1941.25 | 1942.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-04 14:15:00 | 1919.55 | 1936.91 | 1940.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-05 09:15:00 | 1943.70 | 1936.52 | 1939.46 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-05 09:15:00 | 1943.70 | 1936.52 | 1939.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-05 09:15:00 | 1943.70 | 1936.52 | 1939.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-05 09:45:00 | 1954.15 | 1936.52 | 1939.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-05 10:15:00 | 1937.60 | 1936.74 | 1939.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-05 11:15:00 | 1946.45 | 1936.74 | 1939.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-05 11:15:00 | 1942.70 | 1937.93 | 1939.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-05 12:15:00 | 1948.90 | 1937.93 | 1939.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 56 — BUY (started 2024-03-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-05 12:15:00 | 1956.60 | 1941.66 | 1941.15 | EMA200 above EMA400 |

### Cycle 57 — SELL (started 2024-03-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-06 10:15:00 | 1935.15 | 1940.49 | 1940.94 | EMA200 below EMA400 |

### Cycle 58 — BUY (started 2024-03-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-06 13:15:00 | 1963.55 | 1944.14 | 1942.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-06 14:15:00 | 1970.05 | 1949.32 | 1944.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-07 09:15:00 | 1900.80 | 1943.25 | 1943.13 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-07 09:15:00 | 1900.80 | 1943.25 | 1943.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-07 09:15:00 | 1900.80 | 1943.25 | 1943.13 | EMA400 retest candle locked (from upside) |

### Cycle 59 — SELL (started 2024-03-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-07 10:15:00 | 1891.75 | 1932.95 | 1938.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-11 09:15:00 | 1883.85 | 1903.21 | 1918.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-12 09:15:00 | 1894.20 | 1894.03 | 1905.41 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-03-12 10:00:00 | 1894.20 | 1894.03 | 1905.41 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-13 09:15:00 | 1885.75 | 1890.22 | 1897.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-13 09:30:00 | 1892.00 | 1890.22 | 1897.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-14 09:15:00 | 1890.50 | 1872.49 | 1882.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-14 10:00:00 | 1890.50 | 1872.49 | 1882.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-14 10:15:00 | 1884.30 | 1874.85 | 1883.04 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-14 14:00:00 | 1877.45 | 1879.05 | 1883.28 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-15 09:15:00 | 1865.05 | 1881.58 | 1883.83 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-03-21 10:15:00 | 1854.30 | 1847.57 | 1847.11 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 60 — BUY (started 2024-03-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-21 10:15:00 | 1854.30 | 1847.57 | 1847.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-21 14:15:00 | 1870.65 | 1854.41 | 1850.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-22 15:15:00 | 1875.00 | 1875.45 | 1865.70 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-03-26 09:15:00 | 1896.80 | 1875.45 | 1865.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-26 09:15:00 | 1870.20 | 1874.40 | 1866.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-26 10:00:00 | 1870.20 | 1874.40 | 1866.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-26 14:15:00 | 1863.35 | 1873.18 | 1868.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-26 14:45:00 | 1865.45 | 1873.18 | 1868.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-26 15:15:00 | 1869.75 | 1872.49 | 1868.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-27 09:15:00 | 1861.80 | 1872.49 | 1868.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-27 09:15:00 | 1870.00 | 1871.99 | 1868.93 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-27 15:15:00 | 1884.00 | 1872.55 | 1870.33 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-28 09:30:00 | 1895.05 | 1879.63 | 1874.05 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2024-04-08 09:15:00 | 2072.40 | 2021.20 | 2000.79 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 61 — SELL (started 2024-04-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-15 10:15:00 | 2050.20 | 2066.89 | 2068.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-16 09:15:00 | 2046.60 | 2056.90 | 2062.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-16 11:15:00 | 2056.65 | 2055.99 | 2060.73 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-16 11:15:00 | 2056.65 | 2055.99 | 2060.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-16 11:15:00 | 2056.65 | 2055.99 | 2060.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-16 11:30:00 | 2061.05 | 2055.99 | 2060.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-18 09:15:00 | 2061.80 | 2047.58 | 2053.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-18 10:00:00 | 2061.80 | 2047.58 | 2053.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-18 10:15:00 | 2069.25 | 2051.91 | 2055.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-18 11:00:00 | 2069.25 | 2051.91 | 2055.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 62 — BUY (started 2024-04-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-18 11:15:00 | 2079.50 | 2057.43 | 2057.43 | EMA200 above EMA400 |

### Cycle 63 — SELL (started 2024-04-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-18 13:15:00 | 2036.90 | 2056.47 | 2057.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-18 14:15:00 | 2026.15 | 2050.41 | 2054.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-19 11:15:00 | 2060.55 | 2045.54 | 2049.92 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-19 11:15:00 | 2060.55 | 2045.54 | 2049.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-19 11:15:00 | 2060.55 | 2045.54 | 2049.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-19 12:00:00 | 2060.55 | 2045.54 | 2049.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-19 12:15:00 | 2051.50 | 2046.73 | 2050.07 | EMA400 retest candle locked (from downside) |

### Cycle 64 — BUY (started 2024-04-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-19 14:15:00 | 2083.90 | 2057.19 | 2054.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-22 14:15:00 | 2092.55 | 2079.40 | 2069.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-23 10:15:00 | 2084.60 | 2084.77 | 2074.42 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-04-23 10:45:00 | 2084.95 | 2084.77 | 2074.42 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-23 11:15:00 | 2082.60 | 2084.33 | 2075.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-23 11:45:00 | 2071.50 | 2084.33 | 2075.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-23 12:15:00 | 2072.00 | 2081.87 | 2074.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-23 13:00:00 | 2072.00 | 2081.87 | 2074.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-23 13:15:00 | 2075.60 | 2080.61 | 2074.94 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-23 14:15:00 | 2079.70 | 2080.61 | 2074.94 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-04-23 14:15:00 | 2061.05 | 2076.70 | 2073.68 | SL hit (close<static) qty=1.00 sl=2069.65 alert=retest2 |

### Cycle 65 — SELL (started 2024-04-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-24 14:15:00 | 2058.80 | 2071.89 | 2072.60 | EMA200 below EMA400 |

### Cycle 66 — BUY (started 2024-04-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-25 10:15:00 | 2081.05 | 2074.13 | 2073.36 | EMA200 above EMA400 |

### Cycle 67 — SELL (started 2024-04-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-25 12:15:00 | 2068.25 | 2072.67 | 2072.81 | EMA200 below EMA400 |

### Cycle 68 — BUY (started 2024-04-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-25 13:15:00 | 2088.30 | 2075.80 | 2074.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-25 14:15:00 | 2095.85 | 2079.81 | 2076.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-26 09:15:00 | 2071.30 | 2080.11 | 2077.08 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-26 09:15:00 | 2071.30 | 2080.11 | 2077.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-26 09:15:00 | 2071.30 | 2080.11 | 2077.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-26 09:30:00 | 2072.20 | 2080.11 | 2077.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-26 10:15:00 | 2068.25 | 2077.74 | 2076.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-26 11:00:00 | 2068.25 | 2077.74 | 2076.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-26 11:15:00 | 2070.90 | 2076.37 | 2075.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-26 12:15:00 | 2066.25 | 2076.37 | 2075.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 69 — SELL (started 2024-04-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-26 12:15:00 | 2058.65 | 2072.83 | 2074.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-26 13:15:00 | 2053.95 | 2069.05 | 2072.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-29 11:15:00 | 2055.00 | 2053.32 | 2062.15 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-04-29 12:00:00 | 2055.00 | 2053.32 | 2062.15 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-29 14:15:00 | 2061.00 | 2054.32 | 2060.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-29 14:45:00 | 2065.30 | 2054.32 | 2060.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-29 15:15:00 | 2065.55 | 2056.57 | 2060.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-30 09:15:00 | 2121.00 | 2056.57 | 2060.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 70 — BUY (started 2024-04-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-30 09:15:00 | 2142.10 | 2073.67 | 2068.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-30 10:15:00 | 2156.00 | 2090.14 | 2076.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-07 09:15:00 | 2215.00 | 2215.73 | 2197.37 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-07 10:00:00 | 2215.00 | 2215.73 | 2197.37 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-07 10:15:00 | 2191.65 | 2210.92 | 2196.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-07 11:00:00 | 2191.65 | 2210.92 | 2196.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-07 11:15:00 | 2176.10 | 2203.95 | 2194.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-07 12:00:00 | 2176.10 | 2203.95 | 2194.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-07 15:15:00 | 2195.70 | 2193.17 | 2191.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-08 09:15:00 | 2179.05 | 2193.17 | 2191.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 71 — SELL (started 2024-05-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-08 09:15:00 | 2178.25 | 2190.18 | 2190.36 | EMA200 below EMA400 |

### Cycle 72 — BUY (started 2024-05-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-09 09:15:00 | 2237.40 | 2193.61 | 2190.38 | EMA200 above EMA400 |

### Cycle 73 — SELL (started 2024-05-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-10 14:15:00 | 2191.40 | 2199.63 | 2200.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-13 09:15:00 | 2171.35 | 2193.04 | 2197.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-13 15:15:00 | 2206.90 | 2188.70 | 2191.92 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-13 15:15:00 | 2206.90 | 2188.70 | 2191.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-13 15:15:00 | 2206.90 | 2188.70 | 2191.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-14 09:15:00 | 2201.75 | 2188.70 | 2191.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 74 — BUY (started 2024-05-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-14 09:15:00 | 2216.85 | 2194.33 | 2194.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-14 10:15:00 | 2260.00 | 2207.46 | 2200.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-16 12:15:00 | 2294.00 | 2298.47 | 2275.10 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-16 13:00:00 | 2294.00 | 2298.47 | 2275.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-27 09:15:00 | 2558.00 | 2578.28 | 2559.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-27 09:45:00 | 2559.45 | 2578.28 | 2559.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-27 10:15:00 | 2559.30 | 2574.48 | 2559.73 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-27 11:45:00 | 2569.70 | 2574.25 | 2560.97 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-27 14:15:00 | 2565.00 | 2572.57 | 2562.53 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-27 14:15:00 | 2550.05 | 2568.06 | 2561.40 | SL hit (close<static) qty=1.00 sl=2553.90 alert=retest2 |

### Cycle 75 — SELL (started 2024-05-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-28 15:15:00 | 2550.00 | 2558.96 | 2559.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-29 09:15:00 | 2512.00 | 2549.57 | 2555.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-31 09:15:00 | 2548.15 | 2514.51 | 2525.03 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-31 09:15:00 | 2548.15 | 2514.51 | 2525.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 09:15:00 | 2548.15 | 2514.51 | 2525.03 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-31 13:00:00 | 2503.85 | 2519.61 | 2525.60 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-31 15:00:00 | 2503.80 | 2516.18 | 2522.99 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-03 09:15:00 | 2635.60 | 2540.35 | 2532.81 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 76 — BUY (started 2024-06-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-03 09:15:00 | 2635.60 | 2540.35 | 2532.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-05 10:15:00 | 2725.00 | 2616.33 | 2590.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-06 11:15:00 | 2684.35 | 2703.01 | 2661.27 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-06 11:45:00 | 2688.25 | 2703.01 | 2661.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-07 09:15:00 | 2700.00 | 2698.51 | 2674.19 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-07 11:00:00 | 2734.85 | 2705.78 | 2679.70 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2024-06-18 09:15:00 | 3008.34 | 2924.54 | 2881.81 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 77 — SELL (started 2024-06-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-20 11:15:00 | 2887.40 | 2920.31 | 2921.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-20 12:15:00 | 2860.00 | 2908.25 | 2916.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-21 09:15:00 | 2893.15 | 2891.13 | 2904.21 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-06-21 10:00:00 | 2893.15 | 2891.13 | 2904.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-21 10:15:00 | 2888.75 | 2890.66 | 2902.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-21 10:30:00 | 2890.00 | 2890.66 | 2902.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-21 13:15:00 | 2911.90 | 2894.73 | 2901.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-21 14:00:00 | 2911.90 | 2894.73 | 2901.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-21 14:15:00 | 2832.85 | 2882.36 | 2895.37 | EMA400 retest candle locked (from downside) |

### Cycle 78 — BUY (started 2024-06-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-24 15:15:00 | 2921.20 | 2895.45 | 2893.78 | EMA200 above EMA400 |

### Cycle 79 — SELL (started 2024-06-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-26 10:15:00 | 2875.60 | 2893.83 | 2895.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-26 14:15:00 | 2851.10 | 2880.21 | 2888.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-27 14:15:00 | 2890.90 | 2859.59 | 2870.12 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-27 14:15:00 | 2890.90 | 2859.59 | 2870.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 14:15:00 | 2890.90 | 2859.59 | 2870.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-27 15:00:00 | 2890.90 | 2859.59 | 2870.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 15:15:00 | 2885.00 | 2864.67 | 2871.48 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-28 09:15:00 | 2870.20 | 2864.67 | 2871.48 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-28 14:30:00 | 2881.05 | 2871.49 | 2872.67 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-01 13:15:00 | 2881.75 | 2873.14 | 2872.51 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 80 — BUY (started 2024-07-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-01 13:15:00 | 2881.75 | 2873.14 | 2872.51 | EMA200 above EMA400 |

### Cycle 81 — SELL (started 2024-07-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-02 12:15:00 | 2862.00 | 2871.14 | 2871.95 | EMA200 below EMA400 |

### Cycle 82 — BUY (started 2024-07-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-02 13:15:00 | 2879.10 | 2872.73 | 2872.60 | EMA200 above EMA400 |

### Cycle 83 — SELL (started 2024-07-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-02 14:15:00 | 2865.70 | 2871.32 | 2871.97 | EMA200 below EMA400 |

### Cycle 84 — BUY (started 2024-07-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-03 09:15:00 | 2885.00 | 2872.72 | 2872.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-04 09:15:00 | 2910.60 | 2882.26 | 2877.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-05 09:15:00 | 2850.35 | 2887.63 | 2884.84 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-05 09:15:00 | 2850.35 | 2887.63 | 2884.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-05 09:15:00 | 2850.35 | 2887.63 | 2884.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-05 09:30:00 | 2853.55 | 2887.63 | 2884.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 85 — SELL (started 2024-07-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-05 10:15:00 | 2850.00 | 2880.11 | 2881.67 | EMA200 below EMA400 |

### Cycle 86 — BUY (started 2024-07-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-09 12:15:00 | 2909.70 | 2878.87 | 2876.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-09 13:15:00 | 2932.50 | 2889.60 | 2881.46 | Break + close above crossover candle high |

### Cycle 87 — SELL (started 2024-07-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-10 09:15:00 | 2760.05 | 2873.72 | 2877.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-10 10:15:00 | 2732.00 | 2845.37 | 2864.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-15 09:15:00 | 2726.10 | 2712.26 | 2734.25 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-15 09:15:00 | 2726.10 | 2712.26 | 2734.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-15 09:15:00 | 2726.10 | 2712.26 | 2734.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-15 09:45:00 | 2725.15 | 2712.26 | 2734.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-15 14:15:00 | 2732.05 | 2719.36 | 2729.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-15 15:00:00 | 2732.05 | 2719.36 | 2729.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-15 15:15:00 | 2730.00 | 2721.49 | 2729.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-16 09:15:00 | 2762.20 | 2721.49 | 2729.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-16 09:15:00 | 2749.05 | 2727.00 | 2731.51 | EMA400 retest candle locked (from downside) |

### Cycle 88 — BUY (started 2024-07-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-16 11:15:00 | 2748.50 | 2734.96 | 2734.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-16 12:15:00 | 2753.95 | 2738.76 | 2736.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-19 09:15:00 | 2789.00 | 2789.81 | 2769.93 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-19 10:00:00 | 2789.00 | 2789.81 | 2769.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-19 12:15:00 | 2767.70 | 2783.75 | 2772.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-19 13:00:00 | 2767.70 | 2783.75 | 2772.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-19 13:15:00 | 2768.05 | 2780.61 | 2771.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-19 14:00:00 | 2768.05 | 2780.61 | 2771.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-19 14:15:00 | 2748.30 | 2774.15 | 2769.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-19 15:00:00 | 2748.30 | 2774.15 | 2769.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-19 15:15:00 | 2756.20 | 2770.56 | 2768.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-22 09:15:00 | 2740.45 | 2770.56 | 2768.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 12:15:00 | 2814.90 | 2809.71 | 2795.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-23 12:30:00 | 2782.10 | 2809.71 | 2795.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-24 09:15:00 | 2778.50 | 2807.11 | 2799.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-24 10:00:00 | 2778.50 | 2807.11 | 2799.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-24 10:15:00 | 2783.90 | 2802.47 | 2797.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-24 10:30:00 | 2764.95 | 2802.47 | 2797.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 89 — SELL (started 2024-07-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-24 12:15:00 | 2784.40 | 2794.64 | 2794.94 | EMA200 below EMA400 |

### Cycle 90 — BUY (started 2024-07-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-24 14:15:00 | 2805.15 | 2795.67 | 2795.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-24 15:15:00 | 2815.70 | 2799.68 | 2797.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-25 09:15:00 | 2795.85 | 2798.91 | 2797.03 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-25 09:15:00 | 2795.85 | 2798.91 | 2797.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-25 09:15:00 | 2795.85 | 2798.91 | 2797.03 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-25 11:45:00 | 2805.60 | 2799.92 | 2797.77 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-01 10:15:00 | 2839.05 | 2901.58 | 2904.53 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 91 — SELL (started 2024-08-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-01 10:15:00 | 2839.05 | 2901.58 | 2904.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-01 11:15:00 | 2827.35 | 2886.74 | 2897.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-06 09:15:00 | 2700.10 | 2697.41 | 2743.76 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-06 09:15:00 | 2700.10 | 2697.41 | 2743.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-06 09:15:00 | 2700.10 | 2697.41 | 2743.76 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-06 12:45:00 | 2681.95 | 2695.48 | 2731.59 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-07 09:45:00 | 2685.00 | 2668.47 | 2705.22 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-08 13:00:00 | 2680.30 | 2681.90 | 2690.78 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-08 13:45:00 | 2681.65 | 2682.39 | 2690.19 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-08 14:15:00 | 2684.20 | 2682.75 | 2689.65 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-08-09 09:15:00 | 2740.70 | 2693.93 | 2693.51 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 92 — BUY (started 2024-08-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-09 09:15:00 | 2740.70 | 2693.93 | 2693.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-09 15:15:00 | 2764.80 | 2733.98 | 2716.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-12 09:15:00 | 2719.75 | 2731.14 | 2716.67 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-12 09:45:00 | 2720.10 | 2731.14 | 2716.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-12 10:15:00 | 2726.30 | 2730.17 | 2717.55 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-12 11:15:00 | 2734.00 | 2730.17 | 2717.55 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-13 10:00:00 | 2730.60 | 2724.82 | 2719.73 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-13 10:45:00 | 2730.80 | 2725.49 | 2720.50 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-13 11:15:00 | 2734.95 | 2725.49 | 2720.50 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-13 12:15:00 | 2725.95 | 2726.04 | 2721.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-13 13:00:00 | 2725.95 | 2726.04 | 2721.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-13 13:15:00 | 2711.15 | 2723.07 | 2720.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-13 14:00:00 | 2711.15 | 2723.07 | 2720.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-13 14:15:00 | 2716.65 | 2721.78 | 2720.32 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-14 09:15:00 | 2748.40 | 2721.03 | 2720.11 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-20 10:15:00 | 2753.80 | 2775.19 | 2775.66 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 93 — SELL (started 2024-08-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-20 10:15:00 | 2753.80 | 2775.19 | 2775.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-22 12:15:00 | 2734.40 | 2754.38 | 2761.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-23 09:15:00 | 2748.00 | 2744.96 | 2753.84 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-23 09:15:00 | 2748.00 | 2744.96 | 2753.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 09:15:00 | 2748.00 | 2744.96 | 2753.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-23 09:45:00 | 2753.00 | 2744.96 | 2753.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 10:15:00 | 2760.00 | 2747.97 | 2754.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-23 11:00:00 | 2760.00 | 2747.97 | 2754.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 11:15:00 | 2752.10 | 2748.79 | 2754.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-23 11:30:00 | 2755.90 | 2748.79 | 2754.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 12:15:00 | 2764.35 | 2751.91 | 2755.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-23 13:00:00 | 2764.35 | 2751.91 | 2755.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 13:15:00 | 2763.45 | 2754.21 | 2755.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-23 14:00:00 | 2763.45 | 2754.21 | 2755.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 15:15:00 | 2761.00 | 2756.44 | 2756.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-26 09:15:00 | 2776.95 | 2756.44 | 2756.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 94 — BUY (started 2024-08-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-26 09:15:00 | 2781.35 | 2761.42 | 2758.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-26 14:15:00 | 2791.60 | 2773.91 | 2766.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-27 09:15:00 | 2776.95 | 2777.79 | 2769.64 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-27 09:45:00 | 2780.00 | 2777.79 | 2769.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-27 13:15:00 | 2788.50 | 2783.35 | 2775.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-27 13:30:00 | 2777.10 | 2783.35 | 2775.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-29 09:15:00 | 2780.35 | 2795.63 | 2789.47 | EMA400 retest candle locked (from upside) |

### Cycle 95 — SELL (started 2024-08-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-29 11:15:00 | 2763.95 | 2784.83 | 2785.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-29 13:15:00 | 2750.35 | 2775.08 | 2780.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-30 09:15:00 | 2777.10 | 2771.01 | 2776.96 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-30 09:15:00 | 2777.10 | 2771.01 | 2776.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-30 09:15:00 | 2777.10 | 2771.01 | 2776.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-30 10:00:00 | 2777.10 | 2771.01 | 2776.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-30 10:15:00 | 2797.10 | 2776.23 | 2778.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-30 11:00:00 | 2797.10 | 2776.23 | 2778.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 96 — BUY (started 2024-08-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-30 11:15:00 | 2813.00 | 2783.58 | 2781.90 | EMA200 above EMA400 |

### Cycle 97 — SELL (started 2024-09-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-02 11:15:00 | 2759.45 | 2783.53 | 2785.03 | EMA200 below EMA400 |

### Cycle 98 — BUY (started 2024-09-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-03 12:15:00 | 2788.35 | 2783.71 | 2783.20 | EMA200 above EMA400 |

### Cycle 99 — SELL (started 2024-09-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-04 09:15:00 | 2742.70 | 2776.16 | 2780.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-04 12:15:00 | 2735.60 | 2758.26 | 2770.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-04 15:15:00 | 2755.20 | 2753.62 | 2764.56 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-05 09:15:00 | 2740.75 | 2753.62 | 2764.56 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-09 15:15:00 | 2705.20 | 2704.30 | 2713.66 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-10 09:15:00 | 2690.40 | 2704.30 | 2713.66 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-10 13:30:00 | 2704.75 | 2699.23 | 2706.76 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-12 13:15:00 | 2724.40 | 2685.57 | 2686.56 | SL hit (close>static) qty=1.00 sl=2714.30 alert=retest2 |

### Cycle 100 — BUY (started 2024-09-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-12 14:15:00 | 2744.45 | 2697.34 | 2691.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-16 09:15:00 | 2767.45 | 2735.39 | 2718.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-16 14:15:00 | 2756.35 | 2758.18 | 2737.79 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-16 14:30:00 | 2755.00 | 2758.18 | 2737.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 12:15:00 | 2795.90 | 2811.82 | 2796.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-19 13:00:00 | 2795.90 | 2811.82 | 2796.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 13:15:00 | 2800.20 | 2809.50 | 2796.68 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-20 09:15:00 | 2833.00 | 2805.84 | 2797.11 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2024-09-25 09:15:00 | 3116.30 | 3067.57 | 3014.28 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 101 — SELL (started 2024-09-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-30 15:15:00 | 3115.00 | 3124.28 | 3124.86 | EMA200 below EMA400 |

### Cycle 102 — BUY (started 2024-10-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-01 09:15:00 | 3132.80 | 3125.98 | 3125.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-01 11:15:00 | 3179.75 | 3137.97 | 3131.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-03 11:15:00 | 3146.95 | 3154.34 | 3145.25 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-03 11:15:00 | 3146.95 | 3154.34 | 3145.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-03 11:15:00 | 3146.95 | 3154.34 | 3145.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-03 11:45:00 | 3137.10 | 3154.34 | 3145.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-03 12:15:00 | 3163.65 | 3156.20 | 3146.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-03 12:45:00 | 3140.10 | 3156.20 | 3146.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-03 13:15:00 | 3144.90 | 3153.94 | 3146.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-03 13:30:00 | 3143.85 | 3153.94 | 3146.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-03 14:15:00 | 3128.95 | 3148.94 | 3145.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-03 14:30:00 | 3133.95 | 3148.94 | 3145.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 103 — SELL (started 2024-10-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-04 09:15:00 | 3094.95 | 3136.33 | 3139.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-04 10:15:00 | 3079.00 | 3124.86 | 3134.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-07 14:15:00 | 3059.15 | 3041.56 | 3069.18 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-07 15:00:00 | 3059.15 | 3041.56 | 3069.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 09:15:00 | 3101.15 | 3056.75 | 3071.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-08 10:15:00 | 3130.85 | 3056.75 | 3071.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 10:15:00 | 3131.00 | 3071.60 | 3076.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-08 10:45:00 | 3127.15 | 3071.60 | 3076.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 104 — BUY (started 2024-10-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-08 11:15:00 | 3123.25 | 3081.93 | 3081.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-08 14:15:00 | 3168.15 | 3116.77 | 3098.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-09 14:15:00 | 3152.75 | 3157.33 | 3132.64 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-09 15:00:00 | 3152.75 | 3157.33 | 3132.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-11 09:15:00 | 3180.85 | 3185.07 | 3165.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-11 09:30:00 | 3172.25 | 3185.07 | 3165.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-11 10:15:00 | 3150.60 | 3178.17 | 3164.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-11 10:45:00 | 3149.05 | 3178.17 | 3164.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-11 11:15:00 | 3147.35 | 3172.01 | 3162.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-11 12:15:00 | 3139.85 | 3172.01 | 3162.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 105 — SELL (started 2024-10-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-11 13:15:00 | 3124.55 | 3155.81 | 3156.43 | EMA200 below EMA400 |

### Cycle 106 — BUY (started 2024-10-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-14 13:15:00 | 3160.75 | 3155.48 | 3155.08 | EMA200 above EMA400 |

### Cycle 107 — SELL (started 2024-10-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-15 09:15:00 | 3098.95 | 3144.00 | 3149.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-15 12:15:00 | 3096.15 | 3121.60 | 3137.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-15 13:15:00 | 3127.65 | 3122.81 | 3136.23 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-15 14:00:00 | 3127.65 | 3122.81 | 3136.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-15 14:15:00 | 3153.65 | 3128.98 | 3137.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-15 15:00:00 | 3153.65 | 3128.98 | 3137.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-15 15:15:00 | 3150.35 | 3133.25 | 3138.96 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-16 09:15:00 | 3103.10 | 3133.25 | 3138.96 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-18 09:15:00 | 2947.94 | 2982.02 | 3031.00 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-10-21 13:15:00 | 2988.60 | 2951.14 | 2978.19 | SL hit (close>ema200) qty=0.50 sl=2951.14 alert=retest2 |

### Cycle 108 — BUY (started 2024-11-01 17:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-01 17:15:00 | 2798.85 | 2734.95 | 2734.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-01 18:15:00 | 2826.45 | 2753.25 | 2742.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-05 11:15:00 | 2832.05 | 2852.56 | 2818.26 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-05 11:45:00 | 2825.50 | 2852.56 | 2818.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 09:15:00 | 2875.85 | 2905.33 | 2879.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-07 10:00:00 | 2875.85 | 2905.33 | 2879.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 10:15:00 | 2896.45 | 2903.56 | 2880.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-07 10:30:00 | 2869.75 | 2903.56 | 2880.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 11:15:00 | 2897.30 | 2902.31 | 2882.33 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-07 12:15:00 | 2935.00 | 2902.31 | 2882.33 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-07 12:45:00 | 2930.00 | 2907.28 | 2886.41 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-08 10:45:00 | 2921.20 | 2906.09 | 2893.52 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-08 11:15:00 | 2918.00 | 2906.09 | 2893.52 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-11 12:15:00 | 2925.00 | 2937.37 | 2923.11 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-11 14:30:00 | 2936.00 | 2931.06 | 2922.36 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-11 15:00:00 | 2935.00 | 2931.06 | 2922.36 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-12 09:30:00 | 2936.00 | 2929.01 | 2922.83 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-12 10:15:00 | 2938.75 | 2929.01 | 2922.83 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-12 10:15:00 | 2921.70 | 2927.55 | 2922.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-12 10:30:00 | 2919.35 | 2927.55 | 2922.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-12 11:15:00 | 2922.65 | 2926.57 | 2922.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-12 11:45:00 | 2912.85 | 2926.57 | 2922.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-12 12:15:00 | 2904.95 | 2922.24 | 2921.10 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-11-12 12:15:00 | 2904.95 | 2922.24 | 2921.10 | SL hit (close<static) qty=1.00 sl=2913.30 alert=retest2 |

### Cycle 109 — SELL (started 2024-11-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-12 13:15:00 | 2902.90 | 2918.37 | 2919.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-13 09:15:00 | 2804.25 | 2887.61 | 2904.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-14 14:15:00 | 2807.90 | 2807.56 | 2834.02 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-14 15:00:00 | 2807.90 | 2807.56 | 2834.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-18 09:15:00 | 2817.55 | 2811.27 | 2831.22 | EMA400 retest candle locked (from downside) |

### Cycle 110 — BUY (started 2024-11-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-19 09:15:00 | 2948.90 | 2855.45 | 2844.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-19 12:15:00 | 2982.80 | 2907.01 | 2873.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-21 15:15:00 | 2925.05 | 2933.80 | 2913.56 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-22 09:15:00 | 2938.75 | 2933.80 | 2913.56 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-22 09:15:00 | 2955.00 | 2938.04 | 2917.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-22 09:45:00 | 2922.50 | 2938.04 | 2917.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-22 10:15:00 | 2933.10 | 2937.05 | 2918.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-22 10:45:00 | 2931.10 | 2937.05 | 2918.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-25 09:15:00 | 3085.69 | 3003.66 | 2961.60 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-26 09:15:00 | 3002.00 | 3036.41 | 3005.31 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-11-26 09:15:00 | 3002.00 | 3036.41 | 3005.31 | SL hit (close<ema200) qty=0.50 sl=3036.41 alert=retest1 |

### Cycle 111 — SELL (started 2024-11-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-28 09:15:00 | 2952.15 | 2995.23 | 2998.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-28 10:15:00 | 2931.30 | 2982.45 | 2992.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-29 09:15:00 | 2935.60 | 2933.59 | 2959.03 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-29 09:45:00 | 2947.40 | 2933.59 | 2959.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 10:15:00 | 2960.70 | 2939.01 | 2959.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-29 11:00:00 | 2960.70 | 2939.01 | 2959.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 11:15:00 | 2953.10 | 2941.83 | 2958.63 | EMA400 retest candle locked (from downside) |

### Cycle 112 — BUY (started 2024-12-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-02 10:15:00 | 2996.00 | 2968.76 | 2966.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-02 11:15:00 | 3018.00 | 2978.61 | 2970.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-04 15:15:00 | 3025.00 | 3029.77 | 3018.04 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-05 09:15:00 | 3040.90 | 3029.77 | 3018.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-05 09:15:00 | 3034.95 | 3030.81 | 3019.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-05 09:45:00 | 3026.20 | 3030.81 | 3019.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-05 10:15:00 | 3033.00 | 3031.25 | 3020.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-05 10:45:00 | 3023.00 | 3031.25 | 3020.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-05 11:15:00 | 3053.00 | 3035.60 | 3023.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-05 11:45:00 | 3023.55 | 3035.60 | 3023.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-09 09:15:00 | 3061.05 | 3072.39 | 3059.35 | EMA400 retest candle locked (from upside) |

### Cycle 113 — SELL (started 2024-12-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-10 09:15:00 | 3019.15 | 3050.63 | 3053.73 | EMA200 below EMA400 |

### Cycle 114 — BUY (started 2024-12-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-11 09:15:00 | 3091.30 | 3058.25 | 3054.21 | EMA200 above EMA400 |

### Cycle 115 — SELL (started 2024-12-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-13 09:15:00 | 3020.20 | 3057.25 | 3061.60 | EMA200 below EMA400 |

### Cycle 116 — BUY (started 2024-12-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-13 14:15:00 | 3076.20 | 3063.93 | 3062.81 | EMA200 above EMA400 |

### Cycle 117 — SELL (started 2024-12-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-16 10:15:00 | 3045.90 | 3062.14 | 3062.55 | EMA200 below EMA400 |

### Cycle 118 — BUY (started 2024-12-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-16 11:15:00 | 3066.90 | 3063.09 | 3062.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-16 14:15:00 | 3084.75 | 3069.44 | 3066.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-17 10:15:00 | 3059.80 | 3070.82 | 3067.84 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-17 10:15:00 | 3059.80 | 3070.82 | 3067.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-17 10:15:00 | 3059.80 | 3070.82 | 3067.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-17 11:00:00 | 3059.80 | 3070.82 | 3067.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 119 — SELL (started 2024-12-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-17 11:15:00 | 3031.80 | 3063.02 | 3064.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-19 09:15:00 | 3004.90 | 3040.40 | 3048.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-24 09:15:00 | 2934.85 | 2923.57 | 2950.73 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-24 10:00:00 | 2934.85 | 2923.57 | 2950.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-24 10:15:00 | 2935.00 | 2925.86 | 2949.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-24 10:45:00 | 2938.10 | 2925.86 | 2949.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-26 09:15:00 | 2958.00 | 2933.59 | 2943.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-26 09:30:00 | 2948.15 | 2933.59 | 2943.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-26 10:15:00 | 2972.50 | 2941.37 | 2945.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-26 11:00:00 | 2972.50 | 2941.37 | 2945.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-26 11:15:00 | 2975.05 | 2948.10 | 2948.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-26 12:00:00 | 2975.05 | 2948.10 | 2948.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 120 — BUY (started 2024-12-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-26 12:15:00 | 2975.65 | 2953.61 | 2951.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-26 15:15:00 | 2984.20 | 2966.14 | 2957.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-30 09:15:00 | 3025.00 | 3030.67 | 3004.62 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-30 09:30:00 | 3019.50 | 3030.67 | 3004.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-30 14:15:00 | 2992.00 | 3025.53 | 3012.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-30 15:00:00 | 2992.00 | 3025.53 | 3012.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-30 15:15:00 | 3015.00 | 3023.42 | 3012.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-31 09:15:00 | 2983.05 | 3023.42 | 3012.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-31 09:15:00 | 2978.00 | 3014.34 | 3009.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-31 09:45:00 | 2980.75 | 3014.34 | 3009.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-31 11:15:00 | 3001.00 | 3010.89 | 3008.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-31 12:00:00 | 3001.00 | 3010.89 | 3008.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-31 12:15:00 | 3002.25 | 3009.16 | 3008.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-31 12:45:00 | 2998.40 | 3009.16 | 3008.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-31 14:15:00 | 3004.85 | 3008.38 | 3008.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-31 14:30:00 | 3004.95 | 3008.38 | 3008.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-31 15:15:00 | 3014.85 | 3009.68 | 3008.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-01 09:15:00 | 3006.00 | 3009.68 | 3008.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-01 09:15:00 | 3013.20 | 3010.38 | 3009.16 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-01 10:30:00 | 3026.00 | 3020.10 | 3013.69 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-06 14:15:00 | 3103.40 | 3142.28 | 3142.88 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 121 — SELL (started 2025-01-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-06 14:15:00 | 3103.40 | 3142.28 | 3142.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-06 15:15:00 | 3095.95 | 3133.02 | 3138.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-08 09:15:00 | 3118.65 | 3114.15 | 3123.79 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-08 09:15:00 | 3118.65 | 3114.15 | 3123.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-08 09:15:00 | 3118.65 | 3114.15 | 3123.79 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-08 12:00:00 | 3100.80 | 3113.30 | 3121.84 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-09 09:30:00 | 3096.00 | 3101.77 | 3111.03 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-09 11:45:00 | 3100.95 | 3108.36 | 3112.57 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-09 12:15:00 | 3151.60 | 3117.00 | 3116.12 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 122 — BUY (started 2025-01-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-09 12:15:00 | 3151.60 | 3117.00 | 3116.12 | EMA200 above EMA400 |

### Cycle 123 — SELL (started 2025-01-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-10 09:15:00 | 3078.85 | 3115.85 | 3116.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-13 09:15:00 | 3017.75 | 3086.75 | 3101.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-14 10:15:00 | 3072.95 | 3038.76 | 3061.57 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-14 10:15:00 | 3072.95 | 3038.76 | 3061.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-14 10:15:00 | 3072.95 | 3038.76 | 3061.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-14 11:00:00 | 3072.95 | 3038.76 | 3061.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-14 11:15:00 | 3066.95 | 3044.40 | 3062.06 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-14 14:15:00 | 3052.35 | 3050.35 | 3061.90 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-20 09:15:00 | 2899.73 | 2924.82 | 2954.46 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-01-23 09:15:00 | 2891.95 | 2842.90 | 2859.34 | SL hit (close>ema200) qty=0.50 sl=2842.90 alert=retest2 |

### Cycle 124 — BUY (started 2025-01-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-23 13:15:00 | 2893.50 | 2867.45 | 2866.97 | EMA200 above EMA400 |

### Cycle 125 — SELL (started 2025-01-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-24 12:15:00 | 2834.80 | 2861.99 | 2865.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-24 14:15:00 | 2800.70 | 2844.25 | 2856.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-27 12:15:00 | 2834.00 | 2817.15 | 2835.65 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-27 13:00:00 | 2834.00 | 2817.15 | 2835.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-27 13:15:00 | 2829.30 | 2819.58 | 2835.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-27 14:00:00 | 2829.30 | 2819.58 | 2835.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-27 14:15:00 | 2831.10 | 2821.88 | 2834.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-27 15:00:00 | 2831.10 | 2821.88 | 2834.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-27 15:15:00 | 2830.00 | 2823.51 | 2834.29 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-28 09:15:00 | 2796.55 | 2823.51 | 2834.29 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-28 10:15:00 | 2854.50 | 2828.56 | 2834.64 | SL hit (close>static) qty=1.00 sl=2835.75 alert=retest2 |

### Cycle 126 — BUY (started 2025-01-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-28 11:15:00 | 2894.95 | 2841.84 | 2840.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-28 12:15:00 | 2898.75 | 2853.22 | 2845.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-31 14:15:00 | 2989.00 | 2993.64 | 2962.68 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-31 15:00:00 | 2989.00 | 2993.64 | 2962.68 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 09:15:00 | 3134.40 | 3169.70 | 3151.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-06 09:30:00 | 3127.25 | 3169.70 | 3151.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 10:15:00 | 3139.15 | 3163.59 | 3150.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-06 10:30:00 | 3144.25 | 3163.59 | 3150.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 11:15:00 | 3160.15 | 3162.90 | 3151.66 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-07 09:15:00 | 3177.20 | 3152.51 | 3149.60 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-07 09:45:00 | 3174.85 | 3157.97 | 3152.35 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-07 10:45:00 | 3187.25 | 3163.19 | 3155.23 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-07 13:15:00 | 3200.00 | 3166.38 | 3158.23 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 13:15:00 | 3124.35 | 3157.97 | 3155.15 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-02-07 13:15:00 | 3124.35 | 3157.97 | 3155.15 | SL hit (close<static) qty=1.00 sl=3126.55 alert=retest2 |

### Cycle 127 — SELL (started 2025-02-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-10 15:15:00 | 3130.75 | 3162.17 | 3164.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-11 09:15:00 | 3118.65 | 3153.47 | 3160.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-13 09:15:00 | 3026.75 | 3021.29 | 3064.74 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-13 13:30:00 | 2989.50 | 3008.46 | 3044.70 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-17 09:15:00 | 2840.03 | 2919.61 | 2969.70 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Target hit | 2025-02-20 09:15:00 | 2690.55 | 2771.84 | 2802.44 | Target hit (10%) qty=0.50 alert=retest1 |

### Cycle 128 — BUY (started 2025-02-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-20 15:15:00 | 2832.35 | 2817.45 | 2815.73 | EMA200 above EMA400 |

### Cycle 129 — SELL (started 2025-02-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-21 09:15:00 | 2694.05 | 2792.77 | 2804.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-21 10:15:00 | 2674.45 | 2769.11 | 2792.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-24 11:15:00 | 2699.05 | 2696.30 | 2732.21 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-24 11:45:00 | 2701.80 | 2696.30 | 2732.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-25 09:15:00 | 2774.90 | 2718.70 | 2729.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-25 09:45:00 | 2774.50 | 2718.70 | 2729.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-25 10:15:00 | 2790.00 | 2732.96 | 2735.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-25 10:45:00 | 2784.00 | 2732.96 | 2735.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 130 — BUY (started 2025-02-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-25 11:15:00 | 2798.00 | 2745.97 | 2741.02 | EMA200 above EMA400 |

### Cycle 131 — SELL (started 2025-02-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-27 11:15:00 | 2716.10 | 2746.92 | 2747.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-28 09:15:00 | 2598.45 | 2705.37 | 2726.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-03 09:15:00 | 2635.40 | 2628.75 | 2669.09 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-03 09:15:00 | 2635.40 | 2628.75 | 2669.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-03 09:15:00 | 2635.40 | 2628.75 | 2669.09 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-03 11:45:00 | 2597.00 | 2622.17 | 2658.95 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-03 13:45:00 | 2610.90 | 2621.01 | 2652.07 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-03 15:00:00 | 2611.75 | 2619.16 | 2648.40 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-04 09:15:00 | 2593.65 | 2618.12 | 2645.27 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-05 09:15:00 | 2704.55 | 2627.30 | 2632.64 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-03-05 09:15:00 | 2704.55 | 2627.30 | 2632.64 | SL hit (close>static) qty=1.00 sl=2695.95 alert=retest2 |

### Cycle 132 — BUY (started 2025-03-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-05 10:15:00 | 2716.35 | 2645.11 | 2640.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-05 11:15:00 | 2731.90 | 2662.47 | 2648.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-07 11:15:00 | 2721.30 | 2730.79 | 2710.68 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-07 12:00:00 | 2721.30 | 2730.79 | 2710.68 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 09:15:00 | 2706.50 | 2724.73 | 2715.33 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-10 11:15:00 | 2727.30 | 2724.28 | 2715.98 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-11 09:15:00 | 2648.75 | 2702.11 | 2708.77 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 133 — SELL (started 2025-03-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-11 09:15:00 | 2648.75 | 2702.11 | 2708.77 | EMA200 below EMA400 |

### Cycle 134 — BUY (started 2025-03-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-17 11:15:00 | 2679.30 | 2664.61 | 2663.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-17 13:15:00 | 2705.55 | 2673.60 | 2667.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-19 14:15:00 | 2791.25 | 2793.84 | 2761.21 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-19 15:00:00 | 2791.25 | 2793.84 | 2761.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-21 14:15:00 | 2801.85 | 2835.59 | 2817.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-21 15:00:00 | 2801.85 | 2835.59 | 2817.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-21 15:15:00 | 2787.20 | 2825.91 | 2814.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-24 09:15:00 | 2749.90 | 2825.91 | 2814.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 135 — SELL (started 2025-03-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-24 10:15:00 | 2725.00 | 2791.86 | 2800.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-28 09:15:00 | 2654.90 | 2719.37 | 2739.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-01 09:15:00 | 2706.00 | 2683.16 | 2705.84 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-01 09:15:00 | 2706.00 | 2683.16 | 2705.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-01 09:15:00 | 2706.00 | 2683.16 | 2705.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-01 09:30:00 | 2705.90 | 2683.16 | 2705.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-01 10:15:00 | 2642.05 | 2674.94 | 2700.04 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-01 11:45:00 | 2638.20 | 2670.33 | 2695.66 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-01 13:15:00 | 2640.75 | 2666.33 | 2691.54 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-01 13:45:00 | 2638.75 | 2660.39 | 2686.55 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-02 11:45:00 | 2637.00 | 2648.64 | 2669.99 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-07 09:15:00 | 2506.29 | 2578.85 | 2608.50 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-07 09:15:00 | 2508.71 | 2578.85 | 2608.50 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-07 09:15:00 | 2506.81 | 2578.85 | 2608.50 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-07 09:15:00 | 2505.15 | 2578.85 | 2608.50 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-04-08 11:15:00 | 2512.25 | 2508.66 | 2542.90 | SL hit (close>ema200) qty=0.50 sl=2508.66 alert=retest2 |

### Cycle 136 — BUY (started 2025-04-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-11 09:15:00 | 2597.45 | 2547.70 | 2545.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-11 15:15:00 | 2600.60 | 2574.62 | 2561.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-16 10:15:00 | 2630.50 | 2639.61 | 2615.14 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-16 11:00:00 | 2630.50 | 2639.61 | 2615.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-17 09:15:00 | 2620.60 | 2633.54 | 2622.68 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-17 10:15:00 | 2635.00 | 2633.54 | 2622.68 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-04-23 09:15:00 | 2898.50 | 2805.82 | 2759.79 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 137 — SELL (started 2025-05-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-09 09:15:00 | 3021.90 | 3045.17 | 3048.32 | EMA200 below EMA400 |

### Cycle 138 — BUY (started 2025-05-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 12:15:00 | 3060.00 | 3036.87 | 3036.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 13:15:00 | 3083.60 | 3046.21 | 3041.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-13 12:15:00 | 3060.20 | 3074.31 | 3060.97 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-13 12:15:00 | 3060.20 | 3074.31 | 3060.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-13 12:15:00 | 3060.20 | 3074.31 | 3060.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-13 12:45:00 | 3058.20 | 3074.31 | 3060.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-13 13:15:00 | 3042.00 | 3067.85 | 3059.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-13 14:00:00 | 3042.00 | 3067.85 | 3059.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-13 14:15:00 | 3051.80 | 3064.64 | 3058.57 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-14 09:15:00 | 3078.80 | 3063.31 | 3058.52 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-20 10:15:00 | 3098.50 | 3121.02 | 3121.60 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 139 — SELL (started 2025-05-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-20 10:15:00 | 3098.50 | 3121.02 | 3121.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-20 11:15:00 | 3092.80 | 3115.38 | 3118.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-21 09:15:00 | 3109.80 | 3093.53 | 3104.54 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-21 09:15:00 | 3109.80 | 3093.53 | 3104.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 09:15:00 | 3109.80 | 3093.53 | 3104.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-21 10:00:00 | 3109.80 | 3093.53 | 3104.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 10:15:00 | 3111.70 | 3097.16 | 3105.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-21 10:30:00 | 3114.00 | 3097.16 | 3105.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 11:15:00 | 3086.10 | 3094.95 | 3103.45 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-21 12:15:00 | 3082.20 | 3094.95 | 3103.45 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-22 09:15:00 | 3042.70 | 3091.64 | 3099.08 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-26 10:15:00 | 3076.50 | 3029.20 | 3039.61 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-26 12:00:00 | 3076.00 | 3044.97 | 3045.37 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-26 12:15:00 | 3073.20 | 3050.62 | 3047.90 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 140 — BUY (started 2025-05-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-26 12:15:00 | 3073.20 | 3050.62 | 3047.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-26 13:15:00 | 3084.00 | 3057.29 | 3051.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-27 09:15:00 | 3036.90 | 3059.16 | 3054.14 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-27 09:15:00 | 3036.90 | 3059.16 | 3054.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 09:15:00 | 3036.90 | 3059.16 | 3054.14 | EMA400 retest candle locked (from upside) |

### Cycle 141 — SELL (started 2025-05-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-27 13:15:00 | 3025.90 | 3048.29 | 3050.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-28 09:15:00 | 3007.90 | 3038.22 | 3045.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-29 14:15:00 | 3010.00 | 3004.76 | 3016.19 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-29 15:00:00 | 3010.00 | 3004.76 | 3016.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 15:15:00 | 3013.60 | 3006.53 | 3015.95 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-30 09:15:00 | 2998.20 | 3006.53 | 3015.95 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-02 12:15:00 | 3028.10 | 3003.49 | 3002.71 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 142 — BUY (started 2025-06-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-02 12:15:00 | 3028.10 | 3003.49 | 3002.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-02 13:15:00 | 3031.60 | 3009.11 | 3005.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-04 12:15:00 | 3052.10 | 3053.74 | 3038.74 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-04 12:30:00 | 3054.70 | 3053.74 | 3038.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 10:15:00 | 3027.70 | 3053.33 | 3044.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-05 10:45:00 | 3013.00 | 3053.33 | 3044.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 11:15:00 | 3069.90 | 3056.64 | 3046.71 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-06 10:30:00 | 3071.90 | 3052.04 | 3047.63 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-06 11:30:00 | 3075.50 | 3059.56 | 3051.45 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-10 15:15:00 | 3075.20 | 3084.48 | 3081.11 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-12 09:15:00 | 3058.40 | 3081.02 | 3082.98 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 143 — SELL (started 2025-06-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-12 09:15:00 | 3058.40 | 3081.02 | 3082.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-12 10:15:00 | 3038.30 | 3072.48 | 3078.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-16 10:15:00 | 3032.50 | 3012.19 | 3027.44 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-16 10:15:00 | 3032.50 | 3012.19 | 3027.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 10:15:00 | 3032.50 | 3012.19 | 3027.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 10:45:00 | 3026.20 | 3012.19 | 3027.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 11:15:00 | 3044.70 | 3018.69 | 3029.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 12:00:00 | 3044.70 | 3018.69 | 3029.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 12:15:00 | 3023.00 | 3019.55 | 3028.46 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-17 09:15:00 | 3006.50 | 3022.48 | 3027.72 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-17 11:45:00 | 3013.40 | 3016.24 | 3023.16 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-18 09:15:00 | 3067.00 | 3021.64 | 3022.12 | SL hit (close>static) qty=1.00 sl=3045.00 alert=retest2 |

### Cycle 144 — BUY (started 2025-06-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-18 10:15:00 | 3051.50 | 3027.61 | 3024.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-19 12:15:00 | 3079.70 | 3058.04 | 3044.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-23 09:15:00 | 3132.70 | 3153.66 | 3117.69 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-23 09:45:00 | 3135.00 | 3153.66 | 3117.69 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 13:15:00 | 3147.40 | 3171.67 | 3150.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-24 14:00:00 | 3147.40 | 3171.67 | 3150.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 14:15:00 | 3149.50 | 3167.24 | 3150.35 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-24 15:15:00 | 3156.20 | 3167.24 | 3150.35 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-30 13:15:00 | 3178.00 | 3191.76 | 3193.62 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 145 — SELL (started 2025-06-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-30 13:15:00 | 3178.00 | 3191.76 | 3193.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-02 13:15:00 | 3163.70 | 3178.04 | 3183.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-03 09:15:00 | 3203.60 | 3179.50 | 3182.79 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-03 09:15:00 | 3203.60 | 3179.50 | 3182.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 09:15:00 | 3203.60 | 3179.50 | 3182.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-03 10:00:00 | 3203.60 | 3179.50 | 3182.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 146 — BUY (started 2025-07-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-03 10:15:00 | 3211.90 | 3185.98 | 3185.44 | EMA200 above EMA400 |

### Cycle 147 — SELL (started 2025-07-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-03 15:15:00 | 3175.60 | 3186.05 | 3186.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-04 09:15:00 | 3163.00 | 3181.44 | 3184.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-04 14:15:00 | 3165.90 | 3162.28 | 3172.15 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-04 15:00:00 | 3165.90 | 3162.28 | 3172.15 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 09:15:00 | 3163.30 | 3162.44 | 3170.50 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-08 09:15:00 | 3145.20 | 3162.47 | 3167.37 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-09 09:15:00 | 3183.00 | 3157.57 | 3158.96 | SL hit (close>static) qty=1.00 sl=3170.80 alert=retest2 |

### Cycle 148 — BUY (started 2025-07-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-09 10:15:00 | 3177.60 | 3161.57 | 3160.65 | EMA200 above EMA400 |

### Cycle 149 — SELL (started 2025-07-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-10 11:15:00 | 3143.80 | 3162.39 | 3163.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-11 09:15:00 | 3093.90 | 3148.21 | 3156.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-14 13:15:00 | 3096.00 | 3093.35 | 3111.50 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-14 14:00:00 | 3096.00 | 3093.35 | 3111.50 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 09:15:00 | 3119.10 | 3098.11 | 3109.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-15 10:00:00 | 3119.10 | 3098.11 | 3109.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 10:15:00 | 3120.00 | 3102.49 | 3110.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-15 10:45:00 | 3118.10 | 3102.49 | 3110.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 11:15:00 | 3131.90 | 3108.37 | 3112.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-15 11:30:00 | 3144.00 | 3108.37 | 3112.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 150 — BUY (started 2025-07-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-15 12:15:00 | 3143.80 | 3115.46 | 3114.96 | EMA200 above EMA400 |

### Cycle 151 — SELL (started 2025-07-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-16 10:15:00 | 3100.00 | 3115.47 | 3116.09 | EMA200 below EMA400 |

### Cycle 152 — BUY (started 2025-07-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-16 13:15:00 | 3194.90 | 3130.95 | 3122.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-17 15:15:00 | 3216.00 | 3185.59 | 3163.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-18 11:15:00 | 3189.10 | 3190.55 | 3171.56 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-18 11:30:00 | 3185.20 | 3190.55 | 3171.56 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 09:15:00 | 3219.80 | 3196.38 | 3181.13 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-21 14:15:00 | 3235.20 | 3212.41 | 3194.38 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-22 12:00:00 | 3237.20 | 3227.14 | 3209.98 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-25 10:30:00 | 3235.00 | 3249.63 | 3249.37 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-25 11:15:00 | 3231.30 | 3245.96 | 3247.73 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 153 — SELL (started 2025-07-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-25 11:15:00 | 3231.30 | 3245.96 | 3247.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-28 10:15:00 | 3214.00 | 3237.70 | 3242.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-29 13:15:00 | 3211.20 | 3209.50 | 3220.99 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-29 13:45:00 | 3208.50 | 3209.50 | 3220.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 14:15:00 | 3209.30 | 3186.62 | 3200.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-30 14:45:00 | 3209.50 | 3186.62 | 3200.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 15:15:00 | 3224.90 | 3194.28 | 3202.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-31 09:15:00 | 3167.40 | 3194.28 | 3202.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 09:15:00 | 3197.90 | 3195.00 | 3202.20 | EMA400 retest candle locked (from downside) |

### Cycle 154 — BUY (started 2025-07-31 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-31 12:15:00 | 3217.80 | 3206.45 | 3206.25 | EMA200 above EMA400 |

### Cycle 155 — SELL (started 2025-07-31 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-31 15:15:00 | 3197.50 | 3206.26 | 3206.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-01 09:15:00 | 3164.40 | 3197.88 | 3202.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-04 09:15:00 | 3205.70 | 3179.29 | 3187.48 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-04 09:15:00 | 3205.70 | 3179.29 | 3187.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 09:15:00 | 3205.70 | 3179.29 | 3187.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 09:30:00 | 3195.90 | 3179.29 | 3187.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 10:15:00 | 3213.50 | 3186.13 | 3189.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 11:00:00 | 3213.50 | 3186.13 | 3189.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 156 — BUY (started 2025-08-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-04 13:15:00 | 3202.90 | 3193.25 | 3192.57 | EMA200 above EMA400 |

### Cycle 157 — SELL (started 2025-08-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-05 09:15:00 | 3181.20 | 3192.48 | 3192.55 | EMA200 below EMA400 |

### Cycle 158 — BUY (started 2025-08-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-05 11:15:00 | 3202.50 | 3194.57 | 3193.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-05 14:15:00 | 3215.00 | 3200.33 | 3196.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-06 09:15:00 | 3192.30 | 3200.75 | 3197.52 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-06 09:15:00 | 3192.30 | 3200.75 | 3197.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 09:15:00 | 3192.30 | 3200.75 | 3197.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-06 10:00:00 | 3192.30 | 3200.75 | 3197.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 10:15:00 | 3185.00 | 3197.60 | 3196.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-06 11:00:00 | 3185.00 | 3197.60 | 3196.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 11:15:00 | 3198.40 | 3197.76 | 3196.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-06 11:30:00 | 3188.30 | 3197.76 | 3196.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 12:15:00 | 3223.50 | 3202.91 | 3199.02 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-06 15:15:00 | 3240.10 | 3209.35 | 3202.73 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-07 10:15:00 | 3188.00 | 3208.35 | 3204.40 | SL hit (close<static) qty=1.00 sl=3197.20 alert=retest2 |

### Cycle 159 — SELL (started 2025-08-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-07 13:15:00 | 3180.20 | 3200.59 | 3201.75 | EMA200 below EMA400 |

### Cycle 160 — BUY (started 2025-08-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-07 14:15:00 | 3211.00 | 3202.68 | 3202.59 | EMA200 above EMA400 |

### Cycle 161 — SELL (started 2025-08-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-08 09:15:00 | 3195.30 | 3201.73 | 3202.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-08 11:15:00 | 3179.10 | 3194.82 | 3198.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-11 13:15:00 | 3179.90 | 3166.19 | 3176.54 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-11 13:15:00 | 3179.90 | 3166.19 | 3176.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 13:15:00 | 3179.90 | 3166.19 | 3176.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-11 13:45:00 | 3179.80 | 3166.19 | 3176.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 14:15:00 | 3186.50 | 3170.25 | 3177.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-11 15:00:00 | 3186.50 | 3170.25 | 3177.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 15:15:00 | 3184.10 | 3173.02 | 3178.05 | EMA400 retest candle locked (from downside) |

### Cycle 162 — BUY (started 2025-08-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-12 09:15:00 | 3239.20 | 3186.26 | 3183.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-13 09:15:00 | 3267.50 | 3232.52 | 3212.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-14 14:15:00 | 3263.50 | 3272.69 | 3256.09 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-14 15:00:00 | 3263.50 | 3272.69 | 3256.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 09:15:00 | 3375.80 | 3368.48 | 3349.87 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-20 10:15:00 | 3383.60 | 3368.48 | 3349.87 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-20 13:30:00 | 3387.30 | 3373.63 | 3358.13 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-21 13:30:00 | 3379.90 | 3385.45 | 3373.36 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-21 15:15:00 | 3377.50 | 3383.64 | 3373.64 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 15:15:00 | 3377.50 | 3382.41 | 3373.99 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-22 09:15:00 | 3390.40 | 3382.41 | 3373.99 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-26 09:45:00 | 3387.00 | 3397.37 | 3392.58 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-26 12:15:00 | 3382.00 | 3391.77 | 3390.76 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-26 12:15:00 | 3371.50 | 3387.71 | 3389.01 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 163 — SELL (started 2025-08-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-26 12:15:00 | 3371.50 | 3387.71 | 3389.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-26 14:15:00 | 3326.70 | 3371.83 | 3381.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-01 09:15:00 | 3238.00 | 3230.20 | 3272.74 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-01 10:00:00 | 3238.00 | 3230.20 | 3272.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 11:15:00 | 3292.20 | 3247.19 | 3273.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 12:00:00 | 3292.20 | 3247.19 | 3273.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 12:15:00 | 3315.60 | 3260.87 | 3277.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 13:00:00 | 3315.60 | 3260.87 | 3277.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 14:15:00 | 3312.90 | 3276.47 | 3281.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 15:00:00 | 3312.90 | 3276.47 | 3281.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 164 — BUY (started 2025-09-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-02 10:15:00 | 3290.20 | 3285.88 | 3285.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-02 11:15:00 | 3299.20 | 3288.54 | 3286.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-02 13:15:00 | 3283.20 | 3288.49 | 3287.07 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-02 13:15:00 | 3283.20 | 3288.49 | 3287.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 13:15:00 | 3283.20 | 3288.49 | 3287.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-02 14:00:00 | 3283.20 | 3288.49 | 3287.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 165 — SELL (started 2025-09-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-02 14:15:00 | 3238.40 | 3278.47 | 3282.64 | EMA200 below EMA400 |

### Cycle 166 — BUY (started 2025-09-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-04 09:15:00 | 3477.50 | 3314.23 | 3293.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-05 10:15:00 | 3562.30 | 3480.41 | 3407.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-10 09:15:00 | 3667.70 | 3685.52 | 3636.93 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-10 11:15:00 | 3619.50 | 3669.03 | 3637.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 11:15:00 | 3619.50 | 3669.03 | 3637.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-10 12:00:00 | 3619.50 | 3669.03 | 3637.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 12:15:00 | 3604.00 | 3656.03 | 3634.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-10 12:30:00 | 3610.00 | 3656.03 | 3634.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 167 — SELL (started 2025-09-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-11 09:15:00 | 3594.60 | 3624.46 | 3624.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-11 10:15:00 | 3584.00 | 3616.37 | 3620.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-12 09:15:00 | 3606.60 | 3604.71 | 3611.88 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-12 09:15:00 | 3606.60 | 3604.71 | 3611.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 09:15:00 | 3606.60 | 3604.71 | 3611.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-12 09:30:00 | 3607.50 | 3604.71 | 3611.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 10:15:00 | 3600.30 | 3603.83 | 3610.82 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-12 11:15:00 | 3597.00 | 3603.83 | 3610.82 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-12 11:45:00 | 3597.10 | 3602.76 | 3609.70 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-12 12:15:00 | 3594.30 | 3602.76 | 3609.70 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-16 15:15:00 | 3613.60 | 3581.07 | 3578.79 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 168 — BUY (started 2025-09-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-16 15:15:00 | 3613.60 | 3581.07 | 3578.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-17 09:15:00 | 3620.00 | 3588.86 | 3582.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-18 11:15:00 | 3619.40 | 3623.54 | 3609.46 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-18 12:00:00 | 3619.40 | 3623.54 | 3609.46 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 12:15:00 | 3622.00 | 3623.23 | 3610.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-18 12:30:00 | 3611.00 | 3623.23 | 3610.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 13:15:00 | 3612.60 | 3621.10 | 3610.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-18 14:00:00 | 3612.60 | 3621.10 | 3610.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 14:15:00 | 3649.10 | 3626.70 | 3614.27 | EMA400 retest candle locked (from upside) |

### Cycle 169 — SELL (started 2025-09-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-22 09:15:00 | 3587.00 | 3610.60 | 3613.64 | EMA200 below EMA400 |

### Cycle 170 — BUY (started 2025-09-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-23 12:15:00 | 3627.70 | 3609.22 | 3608.03 | EMA200 above EMA400 |

### Cycle 171 — SELL (started 2025-09-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-24 11:15:00 | 3592.00 | 3606.89 | 3608.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-24 14:15:00 | 3571.70 | 3596.44 | 3602.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-29 13:15:00 | 3450.00 | 3441.96 | 3478.18 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-29 13:45:00 | 3454.10 | 3441.96 | 3478.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 13:15:00 | 3432.70 | 3427.69 | 3451.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-30 13:45:00 | 3445.50 | 3427.69 | 3451.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 09:15:00 | 3475.60 | 3438.40 | 3450.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-01 09:45:00 | 3457.90 | 3438.40 | 3450.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 10:15:00 | 3464.20 | 3443.56 | 3451.42 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-01 12:30:00 | 3455.30 | 3448.76 | 3452.57 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-01 15:15:00 | 3468.00 | 3456.73 | 3455.67 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 172 — BUY (started 2025-10-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-01 15:15:00 | 3468.00 | 3456.73 | 3455.67 | EMA200 above EMA400 |

### Cycle 173 — SELL (started 2025-10-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-03 09:15:00 | 3414.80 | 3448.34 | 3451.95 | EMA200 below EMA400 |

### Cycle 174 — BUY (started 2025-10-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-03 15:15:00 | 3460.00 | 3452.49 | 3452.15 | EMA200 above EMA400 |

### Cycle 175 — SELL (started 2025-10-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-06 09:15:00 | 3448.30 | 3451.65 | 3451.80 | EMA200 below EMA400 |

### Cycle 176 — BUY (started 2025-10-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-06 10:15:00 | 3464.50 | 3454.22 | 3452.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-06 12:15:00 | 3476.60 | 3459.88 | 3455.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-08 09:15:00 | 3479.40 | 3485.97 | 3476.24 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-08 10:00:00 | 3479.40 | 3485.97 | 3476.24 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 10:15:00 | 3461.90 | 3481.16 | 3474.93 | EMA400 retest candle locked (from upside) |

### Cycle 177 — SELL (started 2025-10-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-08 12:15:00 | 3447.30 | 3468.87 | 3470.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-08 14:15:00 | 3425.80 | 3454.41 | 3462.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-09 10:15:00 | 3446.70 | 3445.61 | 3456.11 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-09 11:00:00 | 3446.70 | 3445.61 | 3456.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 13:15:00 | 3440.90 | 3443.08 | 3452.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-09 14:00:00 | 3440.90 | 3443.08 | 3452.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 10:15:00 | 3452.90 | 3443.48 | 3449.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-10 10:45:00 | 3448.10 | 3443.48 | 3449.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 11:15:00 | 3444.00 | 3443.59 | 3448.89 | EMA400 retest candle locked (from downside) |

### Cycle 178 — BUY (started 2025-10-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-13 09:15:00 | 3464.80 | 3454.43 | 3453.05 | EMA200 above EMA400 |

### Cycle 179 — SELL (started 2025-10-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-14 11:15:00 | 3437.80 | 3450.99 | 3452.53 | EMA200 below EMA400 |

### Cycle 180 — BUY (started 2025-10-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-15 09:15:00 | 3484.50 | 3457.44 | 3454.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-15 12:15:00 | 3495.30 | 3472.55 | 3462.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-20 10:15:00 | 3613.70 | 3622.18 | 3583.59 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-20 11:00:00 | 3613.70 | 3622.18 | 3583.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 13:15:00 | 3600.90 | 3620.09 | 3607.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-23 14:00:00 | 3600.90 | 3620.09 | 3607.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 14:15:00 | 3617.60 | 3619.59 | 3608.38 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-24 09:15:00 | 3650.10 | 3621.17 | 3610.12 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-24 10:30:00 | 3636.50 | 3628.20 | 3615.43 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-27 12:15:00 | 3598.10 | 3616.44 | 3616.87 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 181 — SELL (started 2025-10-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-27 12:15:00 | 3598.10 | 3616.44 | 3616.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-28 11:15:00 | 3590.30 | 3604.74 | 3610.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-03 09:15:00 | 3540.80 | 3508.89 | 3521.49 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-03 09:15:00 | 3540.80 | 3508.89 | 3521.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 09:15:00 | 3540.80 | 3508.89 | 3521.49 | EMA400 retest candle locked (from downside) |

### Cycle 182 — BUY (started 2025-11-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-03 12:15:00 | 3548.60 | 3528.99 | 3528.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-03 13:15:00 | 3554.00 | 3534.00 | 3530.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-07 09:15:00 | 3597.00 | 3610.27 | 3589.39 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-07 09:15:00 | 3597.00 | 3610.27 | 3589.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 09:15:00 | 3597.00 | 3610.27 | 3589.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-07 09:30:00 | 3597.30 | 3610.27 | 3589.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 10:15:00 | 3636.70 | 3615.55 | 3593.69 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-07 11:30:00 | 3647.90 | 3622.78 | 3598.96 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-14 10:15:00 | 3710.00 | 3713.96 | 3714.16 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 183 — SELL (started 2025-11-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-14 10:15:00 | 3710.00 | 3713.96 | 3714.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-14 13:15:00 | 3680.00 | 3700.88 | 3707.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-17 10:15:00 | 3721.10 | 3703.50 | 3706.50 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-17 10:15:00 | 3721.10 | 3703.50 | 3706.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 10:15:00 | 3721.10 | 3703.50 | 3706.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-17 11:00:00 | 3721.10 | 3703.50 | 3706.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 184 — BUY (started 2025-11-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-17 11:15:00 | 3743.20 | 3711.44 | 3709.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-17 13:15:00 | 3747.60 | 3723.54 | 3715.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-18 09:15:00 | 3696.80 | 3721.10 | 3717.00 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-18 09:15:00 | 3696.80 | 3721.10 | 3717.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 09:15:00 | 3696.80 | 3721.10 | 3717.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-18 10:00:00 | 3696.80 | 3721.10 | 3717.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 10:15:00 | 3701.70 | 3717.22 | 3715.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-18 10:30:00 | 3695.20 | 3717.22 | 3715.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 12:15:00 | 3715.00 | 3717.22 | 3715.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-18 12:30:00 | 3710.90 | 3717.22 | 3715.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 13:15:00 | 3717.70 | 3717.31 | 3716.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-18 13:45:00 | 3715.70 | 3717.31 | 3716.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 185 — SELL (started 2025-11-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-18 14:15:00 | 3693.30 | 3712.51 | 3714.01 | EMA200 below EMA400 |

### Cycle 186 — BUY (started 2025-11-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-19 11:15:00 | 3734.80 | 3716.63 | 3715.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-20 09:15:00 | 3751.30 | 3727.67 | 3721.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-20 14:15:00 | 3717.00 | 3730.21 | 3725.93 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-20 14:15:00 | 3717.00 | 3730.21 | 3725.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 14:15:00 | 3717.00 | 3730.21 | 3725.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-20 15:00:00 | 3717.00 | 3730.21 | 3725.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 15:15:00 | 3711.80 | 3726.53 | 3724.65 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-21 09:15:00 | 3764.00 | 3726.53 | 3724.65 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-24 10:15:00 | 3719.50 | 3741.46 | 3737.58 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-24 11:15:00 | 3702.80 | 3729.32 | 3732.46 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 187 — SELL (started 2025-11-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-24 11:15:00 | 3702.80 | 3729.32 | 3732.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-24 12:15:00 | 3696.50 | 3722.76 | 3729.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-25 13:15:00 | 3692.90 | 3689.74 | 3704.60 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-25 14:00:00 | 3692.90 | 3689.74 | 3704.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 12:15:00 | 3702.00 | 3684.57 | 3694.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-26 13:00:00 | 3702.00 | 3684.57 | 3694.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 13:15:00 | 3697.70 | 3687.19 | 3694.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-26 13:30:00 | 3705.00 | 3687.19 | 3694.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 14:15:00 | 3688.20 | 3687.39 | 3693.86 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-27 10:45:00 | 3675.50 | 3684.65 | 3690.99 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-27 13:00:00 | 3670.90 | 3679.68 | 3687.50 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-27 14:00:00 | 3676.00 | 3678.94 | 3686.45 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-27 14:30:00 | 3677.40 | 3679.28 | 3685.92 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-28 09:15:00 | 3746.00 | 3692.58 | 3690.81 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 188 — BUY (started 2025-11-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-28 09:15:00 | 3746.00 | 3692.58 | 3690.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-01 09:15:00 | 3785.00 | 3746.71 | 3723.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-01 10:15:00 | 3746.70 | 3746.71 | 3725.75 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-01 10:45:00 | 3738.40 | 3746.71 | 3725.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 15:15:00 | 3733.00 | 3745.43 | 3733.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-02 09:15:00 | 3734.30 | 3745.43 | 3733.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 09:15:00 | 3723.40 | 3741.03 | 3732.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-02 10:00:00 | 3723.40 | 3741.03 | 3732.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 10:15:00 | 3720.00 | 3736.82 | 3731.40 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-02 12:30:00 | 3727.00 | 3730.48 | 3729.27 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-02 13:45:00 | 3728.90 | 3730.04 | 3729.18 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-02 14:15:00 | 3716.40 | 3727.31 | 3728.02 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 189 — SELL (started 2025-12-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-02 14:15:00 | 3716.40 | 3727.31 | 3728.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-03 09:15:00 | 3691.10 | 3719.04 | 3724.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-04 10:15:00 | 3677.00 | 3667.91 | 3687.71 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-04 11:00:00 | 3677.00 | 3667.91 | 3687.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 09:15:00 | 3694.90 | 3672.36 | 3680.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-05 10:00:00 | 3694.90 | 3672.36 | 3680.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 10:15:00 | 3704.80 | 3678.85 | 3682.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-05 11:00:00 | 3704.80 | 3678.85 | 3682.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 11:15:00 | 3704.10 | 3683.90 | 3684.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-05 11:45:00 | 3703.40 | 3683.90 | 3684.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 190 — BUY (started 2025-12-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-05 12:15:00 | 3710.90 | 3689.30 | 3687.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-05 14:15:00 | 3714.10 | 3698.05 | 3691.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-08 09:15:00 | 3692.90 | 3698.66 | 3693.24 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-08 09:15:00 | 3692.90 | 3698.66 | 3693.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 09:15:00 | 3692.90 | 3698.66 | 3693.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-08 10:00:00 | 3692.90 | 3698.66 | 3693.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 10:15:00 | 3692.70 | 3697.47 | 3693.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-08 10:30:00 | 3684.00 | 3697.47 | 3693.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 11:15:00 | 3689.90 | 3695.96 | 3692.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-08 11:45:00 | 3687.50 | 3695.96 | 3692.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 12:15:00 | 3696.10 | 3695.98 | 3693.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-08 13:15:00 | 3689.00 | 3695.98 | 3693.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 13:15:00 | 3682.40 | 3693.27 | 3692.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-08 14:00:00 | 3682.40 | 3693.27 | 3692.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 191 — SELL (started 2025-12-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-08 14:15:00 | 3683.60 | 3691.33 | 3691.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-09 09:15:00 | 3617.00 | 3674.49 | 3683.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-10 09:15:00 | 3687.00 | 3656.35 | 3666.32 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-10 09:15:00 | 3687.00 | 3656.35 | 3666.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 09:15:00 | 3687.00 | 3656.35 | 3666.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-10 10:00:00 | 3687.00 | 3656.35 | 3666.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 10:15:00 | 3680.40 | 3661.16 | 3667.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-10 10:30:00 | 3688.80 | 3661.16 | 3667.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 13:15:00 | 3635.20 | 3659.01 | 3665.36 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-10 14:15:00 | 3628.30 | 3659.01 | 3665.36 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-11 10:15:00 | 3631.40 | 3647.31 | 3657.70 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-12 09:15:00 | 3679.60 | 3657.60 | 3657.24 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 192 — BUY (started 2025-12-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-12 09:15:00 | 3679.60 | 3657.60 | 3657.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-12 10:15:00 | 3694.00 | 3664.88 | 3660.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-15 09:15:00 | 3633.90 | 3666.56 | 3664.63 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-15 09:15:00 | 3633.90 | 3666.56 | 3664.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 09:15:00 | 3633.90 | 3666.56 | 3664.63 | EMA400 retest candle locked (from upside) |

### Cycle 193 — SELL (started 2025-12-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-15 10:15:00 | 3624.60 | 3658.17 | 3660.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-15 13:15:00 | 3610.00 | 3639.14 | 3650.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-16 11:15:00 | 3630.50 | 3625.35 | 3637.98 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-16 12:00:00 | 3630.50 | 3625.35 | 3637.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 09:15:00 | 3616.80 | 3623.63 | 3632.36 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-17 10:15:00 | 3612.80 | 3623.63 | 3632.36 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-17 15:15:00 | 3612.50 | 3611.96 | 3622.25 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-19 13:45:00 | 3610.60 | 3595.31 | 3598.87 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-22 09:15:00 | 3610.30 | 3600.78 | 3600.72 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 194 — BUY (started 2025-12-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-22 09:15:00 | 3610.30 | 3600.78 | 3600.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-23 09:15:00 | 3620.70 | 3610.24 | 3606.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-23 10:15:00 | 3610.00 | 3610.20 | 3606.38 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-23 10:15:00 | 3610.00 | 3610.20 | 3606.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 10:15:00 | 3610.00 | 3610.20 | 3606.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-23 10:30:00 | 3609.10 | 3610.20 | 3606.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 11:15:00 | 3620.10 | 3612.18 | 3607.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-23 11:45:00 | 3611.70 | 3612.18 | 3607.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 09:15:00 | 3636.00 | 3623.43 | 3615.60 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-24 11:15:00 | 3640.20 | 3626.02 | 3617.49 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-24 12:00:00 | 3641.00 | 3629.02 | 3619.62 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-29 13:15:00 | 3593.00 | 3621.02 | 3623.60 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 195 — SELL (started 2025-12-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-29 13:15:00 | 3593.00 | 3621.02 | 3623.60 | EMA200 below EMA400 |

### Cycle 196 — BUY (started 2025-12-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-30 11:15:00 | 3636.90 | 3623.03 | 3622.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-30 13:15:00 | 3647.80 | 3629.13 | 3625.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-05 13:15:00 | 3793.40 | 3803.53 | 3775.73 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-05 14:00:00 | 3793.40 | 3803.53 | 3775.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 09:15:00 | 3774.00 | 3793.89 | 3777.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-06 10:15:00 | 3760.50 | 3793.89 | 3777.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 10:15:00 | 3768.60 | 3788.83 | 3777.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-06 10:45:00 | 3760.00 | 3788.83 | 3777.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 14:15:00 | 3783.20 | 3779.83 | 3775.63 | EMA400 retest candle locked (from upside) |

### Cycle 197 — SELL (started 2026-01-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-07 11:15:00 | 3753.30 | 3773.88 | 3774.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-07 12:15:00 | 3732.50 | 3765.60 | 3770.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-09 09:15:00 | 3740.20 | 3732.53 | 3744.78 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-09 09:15:00 | 3740.20 | 3732.53 | 3744.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 09:15:00 | 3740.20 | 3732.53 | 3744.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-09 10:00:00 | 3740.20 | 3732.53 | 3744.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 10:15:00 | 3715.30 | 3729.09 | 3742.10 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-09 11:30:00 | 3706.40 | 3723.79 | 3738.51 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-16 10:15:00 | 3715.80 | 3671.99 | 3669.55 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 198 — BUY (started 2026-01-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-16 10:15:00 | 3715.80 | 3671.99 | 3669.55 | EMA200 above EMA400 |

### Cycle 199 — SELL (started 2026-01-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-16 15:15:00 | 3660.00 | 3670.77 | 3670.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-19 09:15:00 | 3643.20 | 3665.26 | 3668.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-19 13:15:00 | 3658.40 | 3654.51 | 3661.33 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-19 13:15:00 | 3658.40 | 3654.51 | 3661.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 13:15:00 | 3658.40 | 3654.51 | 3661.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-19 13:45:00 | 3663.70 | 3654.51 | 3661.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 14:15:00 | 3660.80 | 3655.77 | 3661.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-19 14:30:00 | 3666.90 | 3655.77 | 3661.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 15:15:00 | 3662.90 | 3657.19 | 3661.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-20 09:15:00 | 3648.30 | 3657.19 | 3661.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 09:15:00 | 3648.10 | 3655.37 | 3660.21 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-20 11:00:00 | 3626.30 | 3649.56 | 3657.13 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-20 12:00:00 | 3632.60 | 3646.17 | 3654.90 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-27 09:15:00 | 3444.99 | 3532.10 | 3556.18 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-27 09:15:00 | 3450.97 | 3532.10 | 3556.18 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-28 10:15:00 | 3441.60 | 3430.22 | 3476.21 | SL hit (close>ema200) qty=0.50 sl=3430.22 alert=retest2 |

### Cycle 200 — BUY (started 2026-02-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-01 09:15:00 | 3476.00 | 3421.27 | 3416.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-01 10:15:00 | 3497.00 | 3436.42 | 3423.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-01 12:15:00 | 3437.50 | 3443.19 | 3429.59 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-01 12:15:00 | 3437.50 | 3443.19 | 3429.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 12:15:00 | 3437.50 | 3443.19 | 3429.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 12:30:00 | 3424.80 | 3443.19 | 3429.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 13:15:00 | 3429.00 | 3440.35 | 3429.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 14:00:00 | 3429.00 | 3440.35 | 3429.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 14:15:00 | 3366.50 | 3425.58 | 3423.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 14:45:00 | 3368.10 | 3425.58 | 3423.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 201 — SELL (started 2026-02-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-01 15:15:00 | 3362.00 | 3412.87 | 3418.19 | EMA200 below EMA400 |

### Cycle 202 — BUY (started 2026-02-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-02 14:15:00 | 3460.50 | 3417.70 | 3415.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-03 09:15:00 | 3543.80 | 3449.03 | 3430.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-05 09:15:00 | 3539.00 | 3560.65 | 3529.27 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-05 09:15:00 | 3539.00 | 3560.65 | 3529.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 09:15:00 | 3539.00 | 3560.65 | 3529.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-05 10:00:00 | 3539.00 | 3560.65 | 3529.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 09:15:00 | 3550.40 | 3565.78 | 3548.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-06 10:00:00 | 3550.40 | 3565.78 | 3548.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 10:15:00 | 3572.00 | 3567.02 | 3550.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-06 10:30:00 | 3551.90 | 3567.02 | 3550.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 12:15:00 | 3545.10 | 3562.83 | 3551.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-06 12:30:00 | 3539.40 | 3562.83 | 3551.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 13:15:00 | 3564.80 | 3563.22 | 3552.56 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-06 14:30:00 | 3575.00 | 3566.30 | 3554.93 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-12 12:15:00 | 3607.70 | 3652.06 | 3652.77 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 203 — SELL (started 2026-02-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-12 12:15:00 | 3607.70 | 3652.06 | 3652.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-12 13:15:00 | 3603.10 | 3642.27 | 3648.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-17 14:15:00 | 3489.80 | 3488.80 | 3519.69 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-17 15:00:00 | 3489.80 | 3488.80 | 3519.69 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 14:15:00 | 3528.90 | 3499.24 | 3509.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-18 15:00:00 | 3528.90 | 3499.24 | 3509.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 15:15:00 | 3543.00 | 3507.99 | 3512.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-19 09:15:00 | 3526.00 | 3507.99 | 3512.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 09:15:00 | 3450.00 | 3429.81 | 3450.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-23 09:45:00 | 3450.50 | 3429.81 | 3450.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 10:15:00 | 3446.00 | 3433.05 | 3449.73 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-24 09:15:00 | 3426.00 | 3438.33 | 3446.54 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-25 09:15:00 | 3463.50 | 3436.28 | 3438.58 | SL hit (close>static) qty=1.00 sl=3457.30 alert=retest2 |

### Cycle 204 — BUY (started 2026-02-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-25 10:15:00 | 3486.90 | 3446.40 | 3442.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-25 11:15:00 | 3491.80 | 3455.48 | 3447.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-26 11:15:00 | 3470.60 | 3477.10 | 3464.61 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-26 12:00:00 | 3470.60 | 3477.10 | 3464.61 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 12:15:00 | 3481.90 | 3478.06 | 3466.18 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-26 14:30:00 | 3487.00 | 3480.19 | 3469.25 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-26 15:00:00 | 3485.40 | 3480.19 | 3469.25 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-27 09:15:00 | 3446.00 | 3473.32 | 3468.03 | SL hit (close<static) qty=1.00 sl=3461.60 alert=retest2 |

### Cycle 205 — SELL (started 2026-02-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-27 11:15:00 | 3445.00 | 3463.96 | 3464.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-27 14:15:00 | 3392.60 | 3444.32 | 3454.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-05 09:15:00 | 3293.90 | 3281.60 | 3322.48 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-05 09:15:00 | 3293.90 | 3281.60 | 3322.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 09:15:00 | 3293.90 | 3281.60 | 3322.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-05 09:45:00 | 3315.20 | 3281.60 | 3322.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 12:15:00 | 3316.00 | 3292.82 | 3317.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-05 12:30:00 | 3310.10 | 3292.82 | 3317.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 13:15:00 | 3308.20 | 3295.89 | 3316.82 | EMA400 retest candle locked (from downside) |

### Cycle 206 — BUY (started 2026-03-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-06 11:15:00 | 3349.70 | 3329.11 | 3327.45 | EMA200 above EMA400 |

### Cycle 207 — SELL (started 2026-03-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-09 09:15:00 | 3209.30 | 3311.12 | 3321.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-09 10:15:00 | 3175.70 | 3284.03 | 3308.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-10 09:15:00 | 3258.70 | 3219.26 | 3256.23 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-10 09:15:00 | 3258.70 | 3219.26 | 3256.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 09:15:00 | 3258.70 | 3219.26 | 3256.23 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-11 10:30:00 | 3206.30 | 3248.84 | 3257.26 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-12 10:15:00 | 3045.99 | 3150.37 | 3198.46 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-16 11:15:00 | 2965.30 | 2963.40 | 3025.17 | SL hit (close>ema200) qty=0.50 sl=2963.40 alert=retest2 |

### Cycle 208 — BUY (started 2026-03-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-17 11:15:00 | 3100.10 | 3050.02 | 3043.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-17 12:15:00 | 3117.00 | 3063.42 | 3049.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 09:15:00 | 3124.30 | 3176.08 | 3138.01 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-19 09:15:00 | 3124.30 | 3176.08 | 3138.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 09:15:00 | 3124.30 | 3176.08 | 3138.01 | EMA400 retest candle locked (from upside) |

### Cycle 209 — SELL (started 2026-03-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 14:15:00 | 3044.60 | 3109.98 | 3117.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-23 09:15:00 | 2968.00 | 3055.04 | 3080.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 09:15:00 | 2993.50 | 2982.77 | 3022.70 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-24 09:15:00 | 2993.50 | 2982.77 | 3022.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 09:15:00 | 2993.50 | 2982.77 | 3022.70 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-24 10:30:00 | 2981.00 | 2981.53 | 3018.51 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-24 11:45:00 | 2985.40 | 2978.53 | 3013.78 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-24 13:15:00 | 3046.70 | 3000.93 | 3018.39 | SL hit (close>static) qty=1.00 sl=3035.00 alert=retest2 |

### Cycle 210 — BUY (started 2026-03-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 09:15:00 | 3115.40 | 3033.95 | 3029.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 10:15:00 | 3151.40 | 3057.44 | 3040.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-27 09:15:00 | 3059.00 | 3099.74 | 3075.72 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-27 09:15:00 | 3059.00 | 3099.74 | 3075.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 09:15:00 | 3059.00 | 3099.74 | 3075.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 10:00:00 | 3059.00 | 3099.74 | 3075.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 10:15:00 | 3065.90 | 3092.97 | 3074.83 | EMA400 retest candle locked (from upside) |

### Cycle 211 — SELL (started 2026-03-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 14:15:00 | 3037.30 | 3065.58 | 3065.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 09:15:00 | 3009.40 | 3050.89 | 3058.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 3047.90 | 3001.86 | 3022.67 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 3047.90 | 3001.86 | 3022.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 3047.90 | 3001.86 | 3022.67 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-01 10:15:00 | 3031.50 | 3001.86 | 3022.67 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-01 11:00:00 | 3033.70 | 3008.23 | 3023.67 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-01 14:30:00 | 3030.50 | 3031.17 | 3031.55 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-08 09:15:00 | 3221.60 | 3041.69 | 3019.32 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 212 — BUY (started 2026-04-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-08 09:15:00 | 3221.60 | 3041.69 | 3019.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-10 10:15:00 | 3250.60 | 3190.39 | 3150.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-13 09:15:00 | 3192.10 | 3228.70 | 3192.24 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-13 09:15:00 | 3192.10 | 3228.70 | 3192.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 3192.10 | 3228.70 | 3192.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-13 09:30:00 | 3201.90 | 3228.70 | 3192.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 10:15:00 | 3224.50 | 3227.86 | 3195.17 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 11:15:00 | 3228.40 | 3227.86 | 3195.17 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-15 09:15:00 | 3243.70 | 3226.85 | 3207.83 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-16 14:15:00 | 3228.00 | 3236.13 | 3233.49 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-16 15:15:00 | 3220.00 | 3230.39 | 3231.18 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 213 — SELL (started 2026-04-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-16 15:15:00 | 3220.00 | 3230.39 | 3231.18 | EMA200 below EMA400 |

### Cycle 214 — BUY (started 2026-04-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-17 09:15:00 | 3239.80 | 3232.27 | 3231.96 | EMA200 above EMA400 |

### Cycle 215 — SELL (started 2026-04-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-17 10:15:00 | 3228.60 | 3231.54 | 3231.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-17 11:15:00 | 3212.00 | 3227.63 | 3229.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-20 11:15:00 | 3214.70 | 3206.98 | 3215.58 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-20 11:15:00 | 3214.70 | 3206.98 | 3215.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-20 11:15:00 | 3214.70 | 3206.98 | 3215.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-20 12:00:00 | 3214.70 | 3206.98 | 3215.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-20 12:15:00 | 3212.30 | 3208.04 | 3215.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-20 12:45:00 | 3221.90 | 3208.04 | 3215.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-20 13:15:00 | 3228.80 | 3212.19 | 3216.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-20 14:00:00 | 3228.80 | 3212.19 | 3216.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-20 14:15:00 | 3224.80 | 3214.71 | 3217.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-20 14:30:00 | 3226.80 | 3214.71 | 3217.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 216 — BUY (started 2026-04-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-21 09:15:00 | 3242.40 | 3219.90 | 3219.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-21 10:15:00 | 3257.50 | 3227.42 | 3222.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-22 09:15:00 | 3206.10 | 3233.99 | 3229.79 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-22 09:15:00 | 3206.10 | 3233.99 | 3229.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-22 09:15:00 | 3206.10 | 3233.99 | 3229.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-22 10:00:00 | 3206.10 | 3233.99 | 3229.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-22 10:15:00 | 3201.10 | 3227.41 | 3227.18 | EMA400 retest candle locked (from upside) |

### Cycle 217 — SELL (started 2026-04-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-22 11:15:00 | 3200.90 | 3222.11 | 3224.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-22 12:15:00 | 3191.80 | 3216.05 | 3221.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-24 09:15:00 | 3086.10 | 3084.51 | 3129.91 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-24 10:00:00 | 3086.10 | 3084.51 | 3129.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 09:15:00 | 3084.40 | 3061.10 | 3092.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 09:45:00 | 3083.60 | 3061.10 | 3092.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 10:15:00 | 3083.30 | 3065.54 | 3091.82 | EMA400 retest candle locked (from downside) |

### Cycle 218 — BUY (started 2026-04-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-28 10:15:00 | 3117.50 | 3102.60 | 3100.68 | EMA200 above EMA400 |

### Cycle 219 — SELL (started 2026-04-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-28 11:15:00 | 3082.50 | 3098.58 | 3099.02 | EMA200 below EMA400 |

### Cycle 220 — BUY (started 2026-04-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-29 09:15:00 | 3175.40 | 3109.99 | 3103.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-29 10:15:00 | 3183.80 | 3124.75 | 3110.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-30 09:15:00 | 3060.00 | 3130.45 | 3123.60 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-30 09:15:00 | 3060.00 | 3130.45 | 3123.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 09:15:00 | 3060.00 | 3130.45 | 3123.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-30 10:00:00 | 3060.00 | 3130.45 | 3123.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 221 — SELL (started 2026-04-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-30 10:15:00 | 3050.40 | 3114.44 | 3116.94 | EMA200 below EMA400 |

### Cycle 222 — BUY (started 2026-05-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-05 12:15:00 | 3174.20 | 3114.94 | 3111.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-05 14:15:00 | 3213.90 | 3140.02 | 3123.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-08 12:15:00 | 3339.00 | 3348.17 | 3305.61 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-05-08 13:00:00 | 3339.00 | 3348.17 | 3305.61 | Sideways (15m bar) within 4 candles — skip ENTRY1 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2023-05-22 09:15:00 | 1257.00 | 2023-05-22 10:15:00 | 1265.45 | STOP_HIT | 1.00 | -0.67% |
| BUY | retest2 | 2023-05-29 09:15:00 | 1334.45 | 2023-06-08 15:15:00 | 1379.95 | STOP_HIT | 1.00 | 3.41% |
| SELL | retest2 | 2023-06-13 14:15:00 | 1370.00 | 2023-06-14 12:15:00 | 1379.75 | STOP_HIT | 1.00 | -0.71% |
| SELL | retest2 | 2023-06-14 09:15:00 | 1370.90 | 2023-06-14 12:15:00 | 1379.75 | STOP_HIT | 1.00 | -0.65% |
| SELL | retest2 | 2023-06-14 11:00:00 | 1370.80 | 2023-06-14 12:15:00 | 1379.75 | STOP_HIT | 1.00 | -0.65% |
| SELL | retest2 | 2023-06-22 11:15:00 | 1379.05 | 2023-06-26 10:15:00 | 1392.85 | STOP_HIT | 1.00 | -1.00% |
| SELL | retest2 | 2023-06-22 12:00:00 | 1378.00 | 2023-06-26 10:15:00 | 1392.85 | STOP_HIT | 1.00 | -1.08% |
| SELL | retest2 | 2023-06-22 12:30:00 | 1378.90 | 2023-06-26 10:15:00 | 1392.85 | STOP_HIT | 1.00 | -1.01% |
| SELL | retest2 | 2023-06-22 14:15:00 | 1379.00 | 2023-06-26 10:15:00 | 1392.85 | STOP_HIT | 1.00 | -1.00% |
| BUY | retest2 | 2023-06-30 09:15:00 | 1416.25 | 2023-07-07 09:15:00 | 1557.88 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2023-07-20 09:30:00 | 1532.50 | 2023-07-20 13:15:00 | 1547.25 | STOP_HIT | 1.00 | -0.96% |
| SELL | retest2 | 2023-08-01 12:15:00 | 1487.70 | 2023-08-01 12:15:00 | 1488.50 | STOP_HIT | 1.00 | -0.05% |
| SELL | retest2 | 2023-08-03 09:45:00 | 1478.60 | 2023-08-03 11:15:00 | 1484.45 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest2 | 2023-08-09 13:15:00 | 1521.45 | 2023-08-18 15:15:00 | 1547.10 | STOP_HIT | 1.00 | 1.69% |
| SELL | retest2 | 2023-08-23 09:15:00 | 1544.90 | 2023-08-28 13:15:00 | 1548.20 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest2 | 2023-08-23 09:45:00 | 1544.10 | 2023-08-28 13:15:00 | 1548.20 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest2 | 2023-08-23 10:15:00 | 1545.00 | 2023-08-28 13:15:00 | 1548.20 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest2 | 2023-08-23 15:00:00 | 1541.85 | 2023-08-28 13:15:00 | 1548.20 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest2 | 2023-08-24 10:30:00 | 1541.65 | 2023-08-28 13:15:00 | 1548.20 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest2 | 2023-08-24 12:45:00 | 1540.45 | 2023-08-28 13:15:00 | 1548.20 | STOP_HIT | 1.00 | -0.50% |
| BUY | retest1 | 2023-09-01 09:15:00 | 1599.60 | 2023-09-04 09:15:00 | 1578.00 | STOP_HIT | 1.00 | -1.35% |
| BUY | retest1 | 2023-09-01 12:45:00 | 1587.00 | 2023-09-04 09:15:00 | 1578.00 | STOP_HIT | 1.00 | -0.57% |
| BUY | retest2 | 2023-09-04 12:30:00 | 1575.55 | 2023-09-05 09:15:00 | 1573.65 | STOP_HIT | 1.00 | -0.12% |
| SELL | retest2 | 2023-09-06 12:30:00 | 1563.05 | 2023-09-06 14:15:00 | 1578.40 | STOP_HIT | 1.00 | -0.98% |
| SELL | retest2 | 2023-09-07 09:15:00 | 1565.60 | 2023-09-11 10:15:00 | 1578.15 | STOP_HIT | 1.00 | -0.80% |
| SELL | retest2 | 2023-09-07 11:00:00 | 1568.10 | 2023-09-11 10:15:00 | 1578.15 | STOP_HIT | 1.00 | -0.64% |
| SELL | retest2 | 2023-09-08 10:30:00 | 1568.40 | 2023-09-11 10:15:00 | 1578.15 | STOP_HIT | 1.00 | -0.62% |
| SELL | retest2 | 2023-10-03 11:00:00 | 1544.90 | 2023-10-06 10:15:00 | 1551.50 | STOP_HIT | 1.00 | -0.43% |
| SELL | retest2 | 2023-10-06 09:30:00 | 1543.35 | 2023-10-06 10:15:00 | 1551.50 | STOP_HIT | 1.00 | -0.53% |
| BUY | retest2 | 2023-10-13 13:30:00 | 1571.00 | 2023-10-19 09:15:00 | 1566.25 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest2 | 2023-10-16 13:30:00 | 1573.75 | 2023-10-19 09:15:00 | 1566.25 | STOP_HIT | 1.00 | -0.48% |
| BUY | retest2 | 2023-10-18 11:15:00 | 1569.40 | 2023-10-19 15:15:00 | 1567.60 | STOP_HIT | 1.00 | -0.11% |
| BUY | retest2 | 2023-10-18 12:00:00 | 1570.20 | 2023-10-19 15:15:00 | 1567.60 | STOP_HIT | 1.00 | -0.17% |
| BUY | retest2 | 2023-10-18 13:45:00 | 1575.40 | 2023-10-19 15:15:00 | 1567.60 | STOP_HIT | 1.00 | -0.50% |
| BUY | retest2 | 2023-10-18 14:45:00 | 1576.60 | 2023-10-19 15:15:00 | 1567.60 | STOP_HIT | 1.00 | -0.57% |
| BUY | retest2 | 2023-10-19 11:15:00 | 1576.80 | 2023-10-19 15:15:00 | 1567.60 | STOP_HIT | 1.00 | -0.58% |
| BUY | retest2 | 2023-10-19 12:00:00 | 1576.75 | 2023-10-19 15:15:00 | 1567.60 | STOP_HIT | 1.00 | -0.58% |
| SELL | retest2 | 2023-10-23 11:15:00 | 1560.10 | 2023-10-25 11:15:00 | 1568.65 | STOP_HIT | 1.00 | -0.55% |
| SELL | retest2 | 2023-10-23 13:00:00 | 1562.10 | 2023-10-25 11:15:00 | 1568.65 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest2 | 2023-10-23 15:15:00 | 1561.00 | 2023-10-25 11:15:00 | 1568.65 | STOP_HIT | 1.00 | -0.49% |
| SELL | retest2 | 2023-10-25 10:00:00 | 1561.25 | 2023-10-25 11:15:00 | 1568.65 | STOP_HIT | 1.00 | -0.47% |
| SELL | retest2 | 2023-11-03 11:00:00 | 1476.10 | 2023-11-06 10:15:00 | 1478.15 | STOP_HIT | 1.00 | -0.14% |
| SELL | retest2 | 2023-11-03 11:45:00 | 1472.80 | 2023-11-06 10:15:00 | 1478.15 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest2 | 2023-11-06 09:45:00 | 1476.05 | 2023-11-06 10:15:00 | 1478.15 | STOP_HIT | 1.00 | -0.14% |
| BUY | retest2 | 2023-11-08 15:00:00 | 1488.50 | 2023-11-20 13:15:00 | 1554.05 | STOP_HIT | 1.00 | 4.40% |
| SELL | retest2 | 2023-11-23 12:00:00 | 1541.40 | 2023-11-24 11:15:00 | 1553.05 | STOP_HIT | 1.00 | -0.76% |
| SELL | retest2 | 2023-11-23 12:45:00 | 1539.75 | 2023-11-24 11:15:00 | 1553.05 | STOP_HIT | 1.00 | -0.86% |
| BUY | retest2 | 2023-11-29 09:15:00 | 1578.15 | 2023-12-08 12:15:00 | 1670.00 | STOP_HIT | 1.00 | 5.82% |
| SELL | retest2 | 2023-12-26 12:15:00 | 1658.10 | 2023-12-27 09:15:00 | 1678.95 | STOP_HIT | 1.00 | -1.26% |
| SELL | retest2 | 2023-12-26 13:00:00 | 1657.75 | 2023-12-27 09:15:00 | 1678.95 | STOP_HIT | 1.00 | -1.28% |
| SELL | retest2 | 2023-12-26 15:15:00 | 1657.10 | 2023-12-27 09:15:00 | 1678.95 | STOP_HIT | 1.00 | -1.32% |
| SELL | retest2 | 2024-01-08 10:15:00 | 1636.30 | 2024-01-11 12:15:00 | 1640.95 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest2 | 2024-01-11 09:45:00 | 1635.60 | 2024-01-11 12:15:00 | 1640.95 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest2 | 2024-01-11 10:15:00 | 1635.30 | 2024-01-11 12:15:00 | 1640.95 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest2 | 2024-01-18 14:45:00 | 1612.10 | 2024-01-19 09:15:00 | 1633.70 | STOP_HIT | 1.00 | -1.34% |
| BUY | retest2 | 2024-01-31 15:15:00 | 1656.80 | 2024-02-08 13:15:00 | 1678.35 | STOP_HIT | 1.00 | 1.30% |
| BUY | retest2 | 2024-02-02 09:15:00 | 1664.00 | 2024-02-08 13:15:00 | 1678.35 | STOP_HIT | 1.00 | 0.86% |
| SELL | retest2 | 2024-02-13 11:45:00 | 1651.25 | 2024-02-15 09:15:00 | 1724.35 | STOP_HIT | 1.00 | -4.43% |
| SELL | retest2 | 2024-02-14 10:00:00 | 1651.00 | 2024-02-15 09:15:00 | 1724.35 | STOP_HIT | 1.00 | -4.44% |
| BUY | retest1 | 2024-02-20 10:45:00 | 1844.35 | 2024-02-23 11:15:00 | 1936.57 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2024-02-20 10:45:00 | 1844.35 | 2024-02-26 14:15:00 | 1930.05 | STOP_HIT | 0.50 | 4.65% |
| BUY | retest1 | 2024-03-01 09:15:00 | 1966.00 | 2024-03-04 09:15:00 | 1930.80 | STOP_HIT | 1.00 | -1.79% |
| BUY | retest1 | 2024-03-01 10:30:00 | 1945.20 | 2024-03-04 09:15:00 | 1930.80 | STOP_HIT | 1.00 | -0.74% |
| BUY | retest1 | 2024-03-01 11:30:00 | 1947.90 | 2024-03-04 09:15:00 | 1930.80 | STOP_HIT | 1.00 | -0.88% |
| BUY | retest2 | 2024-03-04 12:30:00 | 1943.40 | 2024-03-04 13:15:00 | 1925.10 | STOP_HIT | 1.00 | -0.94% |
| SELL | retest2 | 2024-03-14 14:00:00 | 1877.45 | 2024-03-21 10:15:00 | 1854.30 | STOP_HIT | 1.00 | 1.23% |
| SELL | retest2 | 2024-03-15 09:15:00 | 1865.05 | 2024-03-21 10:15:00 | 1854.30 | STOP_HIT | 1.00 | 0.58% |
| BUY | retest2 | 2024-03-27 15:15:00 | 1884.00 | 2024-04-08 09:15:00 | 2072.40 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-03-28 09:30:00 | 1895.05 | 2024-04-08 10:15:00 | 2084.56 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-04-23 14:15:00 | 2079.70 | 2024-04-23 14:15:00 | 2061.05 | STOP_HIT | 1.00 | -0.90% |
| BUY | retest2 | 2024-04-24 11:30:00 | 2082.50 | 2024-04-24 14:15:00 | 2058.80 | STOP_HIT | 1.00 | -1.14% |
| BUY | retest2 | 2024-05-27 11:45:00 | 2569.70 | 2024-05-27 14:15:00 | 2550.05 | STOP_HIT | 1.00 | -0.76% |
| BUY | retest2 | 2024-05-27 14:15:00 | 2565.00 | 2024-05-27 14:15:00 | 2550.05 | STOP_HIT | 1.00 | -0.58% |
| BUY | retest2 | 2024-05-28 13:00:00 | 2578.70 | 2024-05-28 14:15:00 | 2550.25 | STOP_HIT | 1.00 | -1.10% |
| SELL | retest2 | 2024-05-31 13:00:00 | 2503.85 | 2024-06-03 09:15:00 | 2635.60 | STOP_HIT | 1.00 | -5.26% |
| SELL | retest2 | 2024-05-31 15:00:00 | 2503.80 | 2024-06-03 09:15:00 | 2635.60 | STOP_HIT | 1.00 | -5.26% |
| BUY | retest2 | 2024-06-07 11:00:00 | 2734.85 | 2024-06-18 09:15:00 | 3008.34 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-06-28 09:15:00 | 2870.20 | 2024-07-01 13:15:00 | 2881.75 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest2 | 2024-06-28 14:30:00 | 2881.05 | 2024-07-01 13:15:00 | 2881.75 | STOP_HIT | 1.00 | -0.02% |
| BUY | retest2 | 2024-07-25 11:45:00 | 2805.60 | 2024-08-01 10:15:00 | 2839.05 | STOP_HIT | 1.00 | 1.19% |
| SELL | retest2 | 2024-08-06 12:45:00 | 2681.95 | 2024-08-09 09:15:00 | 2740.70 | STOP_HIT | 1.00 | -2.19% |
| SELL | retest2 | 2024-08-07 09:45:00 | 2685.00 | 2024-08-09 09:15:00 | 2740.70 | STOP_HIT | 1.00 | -2.07% |
| SELL | retest2 | 2024-08-08 13:00:00 | 2680.30 | 2024-08-09 09:15:00 | 2740.70 | STOP_HIT | 1.00 | -2.25% |
| SELL | retest2 | 2024-08-08 13:45:00 | 2681.65 | 2024-08-09 09:15:00 | 2740.70 | STOP_HIT | 1.00 | -2.20% |
| BUY | retest2 | 2024-08-12 11:15:00 | 2734.00 | 2024-08-20 10:15:00 | 2753.80 | STOP_HIT | 1.00 | 0.72% |
| BUY | retest2 | 2024-08-13 10:00:00 | 2730.60 | 2024-08-20 10:15:00 | 2753.80 | STOP_HIT | 1.00 | 0.85% |
| BUY | retest2 | 2024-08-13 10:45:00 | 2730.80 | 2024-08-20 10:15:00 | 2753.80 | STOP_HIT | 1.00 | 0.84% |
| BUY | retest2 | 2024-08-13 11:15:00 | 2734.95 | 2024-08-20 10:15:00 | 2753.80 | STOP_HIT | 1.00 | 0.69% |
| BUY | retest2 | 2024-08-14 09:15:00 | 2748.40 | 2024-08-20 10:15:00 | 2753.80 | STOP_HIT | 1.00 | 0.20% |
| SELL | retest2 | 2024-09-10 09:15:00 | 2690.40 | 2024-09-12 13:15:00 | 2724.40 | STOP_HIT | 1.00 | -1.26% |
| SELL | retest2 | 2024-09-10 13:30:00 | 2704.75 | 2024-09-12 13:15:00 | 2724.40 | STOP_HIT | 1.00 | -0.73% |
| BUY | retest2 | 2024-09-20 09:15:00 | 2833.00 | 2024-09-25 09:15:00 | 3116.30 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-10-16 09:15:00 | 3103.10 | 2024-10-18 09:15:00 | 2947.94 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-16 09:15:00 | 3103.10 | 2024-10-21 13:15:00 | 2988.60 | STOP_HIT | 0.50 | 3.69% |
| BUY | retest2 | 2024-11-07 12:15:00 | 2935.00 | 2024-11-12 12:15:00 | 2904.95 | STOP_HIT | 1.00 | -1.02% |
| BUY | retest2 | 2024-11-07 12:45:00 | 2930.00 | 2024-11-12 12:15:00 | 2904.95 | STOP_HIT | 1.00 | -0.85% |
| BUY | retest2 | 2024-11-08 10:45:00 | 2921.20 | 2024-11-12 12:15:00 | 2904.95 | STOP_HIT | 1.00 | -0.56% |
| BUY | retest2 | 2024-11-08 11:15:00 | 2918.00 | 2024-11-12 12:15:00 | 2904.95 | STOP_HIT | 1.00 | -0.45% |
| BUY | retest2 | 2024-11-11 14:30:00 | 2936.00 | 2024-11-12 13:15:00 | 2902.90 | STOP_HIT | 1.00 | -1.13% |
| BUY | retest2 | 2024-11-11 15:00:00 | 2935.00 | 2024-11-12 13:15:00 | 2902.90 | STOP_HIT | 1.00 | -1.09% |
| BUY | retest2 | 2024-11-12 09:30:00 | 2936.00 | 2024-11-12 13:15:00 | 2902.90 | STOP_HIT | 1.00 | -1.13% |
| BUY | retest2 | 2024-11-12 10:15:00 | 2938.75 | 2024-11-12 13:15:00 | 2902.90 | STOP_HIT | 1.00 | -1.22% |
| BUY | retest1 | 2024-11-22 09:15:00 | 2938.75 | 2024-11-25 09:15:00 | 3085.69 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2024-11-22 09:15:00 | 2938.75 | 2024-11-26 09:15:00 | 3002.00 | STOP_HIT | 0.50 | 2.15% |
| BUY | retest2 | 2025-01-01 10:30:00 | 3026.00 | 2025-01-06 14:15:00 | 3103.40 | STOP_HIT | 1.00 | 2.56% |
| SELL | retest2 | 2025-01-08 12:00:00 | 3100.80 | 2025-01-09 12:15:00 | 3151.60 | STOP_HIT | 1.00 | -1.64% |
| SELL | retest2 | 2025-01-09 09:30:00 | 3096.00 | 2025-01-09 12:15:00 | 3151.60 | STOP_HIT | 1.00 | -1.80% |
| SELL | retest2 | 2025-01-09 11:45:00 | 3100.95 | 2025-01-09 12:15:00 | 3151.60 | STOP_HIT | 1.00 | -1.63% |
| SELL | retest2 | 2025-01-14 14:15:00 | 3052.35 | 2025-01-20 09:15:00 | 2899.73 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-14 14:15:00 | 3052.35 | 2025-01-23 09:15:00 | 2891.95 | STOP_HIT | 0.50 | 5.25% |
| SELL | retest2 | 2025-01-28 09:15:00 | 2796.55 | 2025-01-28 10:15:00 | 2854.50 | STOP_HIT | 1.00 | -2.07% |
| SELL | retest2 | 2025-01-28 10:45:00 | 2824.75 | 2025-01-28 11:15:00 | 2894.95 | STOP_HIT | 1.00 | -2.49% |
| BUY | retest2 | 2025-02-07 09:15:00 | 3177.20 | 2025-02-07 13:15:00 | 3124.35 | STOP_HIT | 1.00 | -1.66% |
| BUY | retest2 | 2025-02-07 09:45:00 | 3174.85 | 2025-02-07 13:15:00 | 3124.35 | STOP_HIT | 1.00 | -1.59% |
| BUY | retest2 | 2025-02-07 10:45:00 | 3187.25 | 2025-02-07 13:15:00 | 3124.35 | STOP_HIT | 1.00 | -1.97% |
| BUY | retest2 | 2025-02-07 13:15:00 | 3200.00 | 2025-02-07 13:15:00 | 3124.35 | STOP_HIT | 1.00 | -2.36% |
| BUY | retest2 | 2025-02-10 09:30:00 | 3209.20 | 2025-02-10 15:15:00 | 3130.75 | STOP_HIT | 1.00 | -2.44% |
| SELL | retest1 | 2025-02-13 13:30:00 | 2989.50 | 2025-02-17 09:15:00 | 2840.03 | PARTIAL | 0.50 | 5.00% |
| SELL | retest1 | 2025-02-13 13:30:00 | 2989.50 | 2025-02-20 09:15:00 | 2690.55 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-03-03 11:45:00 | 2597.00 | 2025-03-05 09:15:00 | 2704.55 | STOP_HIT | 1.00 | -4.14% |
| SELL | retest2 | 2025-03-03 13:45:00 | 2610.90 | 2025-03-05 09:15:00 | 2704.55 | STOP_HIT | 1.00 | -3.59% |
| SELL | retest2 | 2025-03-03 15:00:00 | 2611.75 | 2025-03-05 09:15:00 | 2704.55 | STOP_HIT | 1.00 | -3.55% |
| SELL | retest2 | 2025-03-04 09:15:00 | 2593.65 | 2025-03-05 09:15:00 | 2704.55 | STOP_HIT | 1.00 | -4.28% |
| BUY | retest2 | 2025-03-10 11:15:00 | 2727.30 | 2025-03-11 09:15:00 | 2648.75 | STOP_HIT | 1.00 | -2.88% |
| SELL | retest2 | 2025-04-01 11:45:00 | 2638.20 | 2025-04-07 09:15:00 | 2506.29 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-04-01 13:15:00 | 2640.75 | 2025-04-07 09:15:00 | 2508.71 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-04-01 13:45:00 | 2638.75 | 2025-04-07 09:15:00 | 2506.81 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-04-02 11:45:00 | 2637.00 | 2025-04-07 09:15:00 | 2505.15 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-04-01 11:45:00 | 2638.20 | 2025-04-08 11:15:00 | 2512.25 | STOP_HIT | 0.50 | 4.77% |
| SELL | retest2 | 2025-04-01 13:15:00 | 2640.75 | 2025-04-08 11:15:00 | 2512.25 | STOP_HIT | 0.50 | 4.87% |
| SELL | retest2 | 2025-04-01 13:45:00 | 2638.75 | 2025-04-08 11:15:00 | 2512.25 | STOP_HIT | 0.50 | 4.79% |
| SELL | retest2 | 2025-04-02 11:45:00 | 2637.00 | 2025-04-08 11:15:00 | 2512.25 | STOP_HIT | 0.50 | 4.73% |
| SELL | retest2 | 2025-04-09 14:30:00 | 2522.75 | 2025-04-11 09:15:00 | 2597.45 | STOP_HIT | 1.00 | -2.96% |
| BUY | retest2 | 2025-04-17 10:15:00 | 2635.00 | 2025-04-23 09:15:00 | 2898.50 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-05-14 09:15:00 | 3078.80 | 2025-05-20 10:15:00 | 3098.50 | STOP_HIT | 1.00 | 0.64% |
| SELL | retest2 | 2025-05-21 12:15:00 | 3082.20 | 2025-05-26 12:15:00 | 3073.20 | STOP_HIT | 1.00 | 0.29% |
| SELL | retest2 | 2025-05-22 09:15:00 | 3042.70 | 2025-05-26 12:15:00 | 3073.20 | STOP_HIT | 1.00 | -1.00% |
| SELL | retest2 | 2025-05-26 10:15:00 | 3076.50 | 2025-05-26 12:15:00 | 3073.20 | STOP_HIT | 1.00 | 0.11% |
| SELL | retest2 | 2025-05-26 12:00:00 | 3076.00 | 2025-05-26 12:15:00 | 3073.20 | STOP_HIT | 1.00 | 0.09% |
| SELL | retest2 | 2025-05-30 09:15:00 | 2998.20 | 2025-06-02 12:15:00 | 3028.10 | STOP_HIT | 1.00 | -1.00% |
| BUY | retest2 | 2025-06-06 10:30:00 | 3071.90 | 2025-06-12 09:15:00 | 3058.40 | STOP_HIT | 1.00 | -0.44% |
| BUY | retest2 | 2025-06-06 11:30:00 | 3075.50 | 2025-06-12 09:15:00 | 3058.40 | STOP_HIT | 1.00 | -0.56% |
| BUY | retest2 | 2025-06-10 15:15:00 | 3075.20 | 2025-06-12 09:15:00 | 3058.40 | STOP_HIT | 1.00 | -0.55% |
| SELL | retest2 | 2025-06-17 09:15:00 | 3006.50 | 2025-06-18 09:15:00 | 3067.00 | STOP_HIT | 1.00 | -2.01% |
| SELL | retest2 | 2025-06-17 11:45:00 | 3013.40 | 2025-06-18 09:15:00 | 3067.00 | STOP_HIT | 1.00 | -1.78% |
| BUY | retest2 | 2025-06-24 15:15:00 | 3156.20 | 2025-06-30 13:15:00 | 3178.00 | STOP_HIT | 1.00 | 0.69% |
| SELL | retest2 | 2025-07-08 09:15:00 | 3145.20 | 2025-07-09 09:15:00 | 3183.00 | STOP_HIT | 1.00 | -1.20% |
| BUY | retest2 | 2025-07-21 14:15:00 | 3235.20 | 2025-07-25 11:15:00 | 3231.30 | STOP_HIT | 1.00 | -0.12% |
| BUY | retest2 | 2025-07-22 12:00:00 | 3237.20 | 2025-07-25 11:15:00 | 3231.30 | STOP_HIT | 1.00 | -0.18% |
| BUY | retest2 | 2025-07-25 10:30:00 | 3235.00 | 2025-07-25 11:15:00 | 3231.30 | STOP_HIT | 1.00 | -0.11% |
| BUY | retest2 | 2025-08-06 15:15:00 | 3240.10 | 2025-08-07 10:15:00 | 3188.00 | STOP_HIT | 1.00 | -1.61% |
| BUY | retest2 | 2025-08-20 10:15:00 | 3383.60 | 2025-08-26 12:15:00 | 3371.50 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest2 | 2025-08-20 13:30:00 | 3387.30 | 2025-08-26 12:15:00 | 3371.50 | STOP_HIT | 1.00 | -0.47% |
| BUY | retest2 | 2025-08-21 13:30:00 | 3379.90 | 2025-08-26 12:15:00 | 3371.50 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest2 | 2025-08-21 15:15:00 | 3377.50 | 2025-08-26 12:15:00 | 3371.50 | STOP_HIT | 1.00 | -0.18% |
| BUY | retest2 | 2025-08-22 09:15:00 | 3390.40 | 2025-08-26 12:15:00 | 3371.50 | STOP_HIT | 1.00 | -0.56% |
| BUY | retest2 | 2025-08-26 09:45:00 | 3387.00 | 2025-08-26 12:15:00 | 3371.50 | STOP_HIT | 1.00 | -0.46% |
| BUY | retest2 | 2025-08-26 12:15:00 | 3382.00 | 2025-08-26 12:15:00 | 3371.50 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest2 | 2025-09-12 11:15:00 | 3597.00 | 2025-09-16 15:15:00 | 3613.60 | STOP_HIT | 1.00 | -0.46% |
| SELL | retest2 | 2025-09-12 11:45:00 | 3597.10 | 2025-09-16 15:15:00 | 3613.60 | STOP_HIT | 1.00 | -0.46% |
| SELL | retest2 | 2025-09-12 12:15:00 | 3594.30 | 2025-09-16 15:15:00 | 3613.60 | STOP_HIT | 1.00 | -0.54% |
| SELL | retest2 | 2025-10-01 12:30:00 | 3455.30 | 2025-10-01 15:15:00 | 3468.00 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest2 | 2025-10-24 09:15:00 | 3650.10 | 2025-10-27 12:15:00 | 3598.10 | STOP_HIT | 1.00 | -1.42% |
| BUY | retest2 | 2025-10-24 10:30:00 | 3636.50 | 2025-10-27 12:15:00 | 3598.10 | STOP_HIT | 1.00 | -1.06% |
| BUY | retest2 | 2025-11-07 11:30:00 | 3647.90 | 2025-11-14 10:15:00 | 3710.00 | STOP_HIT | 1.00 | 1.70% |
| BUY | retest2 | 2025-11-21 09:15:00 | 3764.00 | 2025-11-24 11:15:00 | 3702.80 | STOP_HIT | 1.00 | -1.63% |
| BUY | retest2 | 2025-11-24 10:15:00 | 3719.50 | 2025-11-24 11:15:00 | 3702.80 | STOP_HIT | 1.00 | -0.45% |
| SELL | retest2 | 2025-11-27 10:45:00 | 3675.50 | 2025-11-28 09:15:00 | 3746.00 | STOP_HIT | 1.00 | -1.92% |
| SELL | retest2 | 2025-11-27 13:00:00 | 3670.90 | 2025-11-28 09:15:00 | 3746.00 | STOP_HIT | 1.00 | -2.05% |
| SELL | retest2 | 2025-11-27 14:00:00 | 3676.00 | 2025-11-28 09:15:00 | 3746.00 | STOP_HIT | 1.00 | -1.90% |
| SELL | retest2 | 2025-11-27 14:30:00 | 3677.40 | 2025-11-28 09:15:00 | 3746.00 | STOP_HIT | 1.00 | -1.87% |
| BUY | retest2 | 2025-12-02 12:30:00 | 3727.00 | 2025-12-02 14:15:00 | 3716.40 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest2 | 2025-12-02 13:45:00 | 3728.90 | 2025-12-02 14:15:00 | 3716.40 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest2 | 2025-12-10 14:15:00 | 3628.30 | 2025-12-12 09:15:00 | 3679.60 | STOP_HIT | 1.00 | -1.41% |
| SELL | retest2 | 2025-12-11 10:15:00 | 3631.40 | 2025-12-12 09:15:00 | 3679.60 | STOP_HIT | 1.00 | -1.33% |
| SELL | retest2 | 2025-12-17 10:15:00 | 3612.80 | 2025-12-22 09:15:00 | 3610.30 | STOP_HIT | 1.00 | 0.07% |
| SELL | retest2 | 2025-12-17 15:15:00 | 3612.50 | 2025-12-22 09:15:00 | 3610.30 | STOP_HIT | 1.00 | 0.06% |
| SELL | retest2 | 2025-12-19 13:45:00 | 3610.60 | 2025-12-22 09:15:00 | 3610.30 | STOP_HIT | 1.00 | 0.01% |
| BUY | retest2 | 2025-12-24 11:15:00 | 3640.20 | 2025-12-29 13:15:00 | 3593.00 | STOP_HIT | 1.00 | -1.30% |
| BUY | retest2 | 2025-12-24 12:00:00 | 3641.00 | 2025-12-29 13:15:00 | 3593.00 | STOP_HIT | 1.00 | -1.32% |
| SELL | retest2 | 2026-01-09 11:30:00 | 3706.40 | 2026-01-16 10:15:00 | 3715.80 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest2 | 2026-01-20 11:00:00 | 3626.30 | 2026-01-27 09:15:00 | 3444.99 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-20 12:00:00 | 3632.60 | 2026-01-27 09:15:00 | 3450.97 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-20 11:00:00 | 3626.30 | 2026-01-28 10:15:00 | 3441.60 | STOP_HIT | 0.50 | 5.09% |
| SELL | retest2 | 2026-01-20 12:00:00 | 3632.60 | 2026-01-28 10:15:00 | 3441.60 | STOP_HIT | 0.50 | 5.26% |
| BUY | retest2 | 2026-02-06 14:30:00 | 3575.00 | 2026-02-12 12:15:00 | 3607.70 | STOP_HIT | 1.00 | 0.91% |
| SELL | retest2 | 2026-02-24 09:15:00 | 3426.00 | 2026-02-25 09:15:00 | 3463.50 | STOP_HIT | 1.00 | -1.09% |
| BUY | retest2 | 2026-02-26 14:30:00 | 3487.00 | 2026-02-27 09:15:00 | 3446.00 | STOP_HIT | 1.00 | -1.18% |
| BUY | retest2 | 2026-02-26 15:00:00 | 3485.40 | 2026-02-27 09:15:00 | 3446.00 | STOP_HIT | 1.00 | -1.13% |
| SELL | retest2 | 2026-03-11 10:30:00 | 3206.30 | 2026-03-12 10:15:00 | 3045.99 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-11 10:30:00 | 3206.30 | 2026-03-16 11:15:00 | 2965.30 | STOP_HIT | 0.50 | 7.52% |
| SELL | retest2 | 2026-03-24 10:30:00 | 2981.00 | 2026-03-24 13:15:00 | 3046.70 | STOP_HIT | 1.00 | -2.20% |
| SELL | retest2 | 2026-03-24 11:45:00 | 2985.40 | 2026-03-24 13:15:00 | 3046.70 | STOP_HIT | 1.00 | -2.05% |
| SELL | retest2 | 2026-04-01 10:15:00 | 3031.50 | 2026-04-08 09:15:00 | 3221.60 | STOP_HIT | 1.00 | -6.27% |
| SELL | retest2 | 2026-04-01 11:00:00 | 3033.70 | 2026-04-08 09:15:00 | 3221.60 | STOP_HIT | 1.00 | -6.19% |
| SELL | retest2 | 2026-04-01 14:30:00 | 3030.50 | 2026-04-08 09:15:00 | 3221.60 | STOP_HIT | 1.00 | -6.31% |
| BUY | retest2 | 2026-04-13 11:15:00 | 3228.40 | 2026-04-16 15:15:00 | 3220.00 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest2 | 2026-04-15 09:15:00 | 3243.70 | 2026-04-16 15:15:00 | 3220.00 | STOP_HIT | 1.00 | -0.73% |
| BUY | retest2 | 2026-04-16 14:15:00 | 3228.00 | 2026-04-16 15:15:00 | 3220.00 | STOP_HIT | 1.00 | -0.25% |
