# ICICIBANK (ICICIBANK)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 1267.80
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 7 |
| ALERT1 | 7 |
| ALERT2 | 7 |
| ALERT2_SKIP | 4 |
| ALERT3 | 36 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 40 |
| PARTIAL | 5 |
| TARGET_HIT | 5 |
| STOP_HIT | 35 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 44 (incl. partial bookings)
- **Trades open at end:** 1
- **Winners / losers:** 16 / 28
- **Target hits / Stop hits / Partials:** 4 / 35 / 5
- **Avg / median % per leg:** 0.63% / -0.84%
- **Sum % (uncompounded):** 27.72%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 17 | 6 | 35.3% | 4 | 13 | 0 | 1.50% | 25.5% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 17 | 6 | 35.3% | 4 | 13 | 0 | 1.50% | 25.5% |
| SELL (all) | 27 | 10 | 37.0% | 0 | 22 | 5 | 0.08% | 2.2% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 27 | 10 | 37.0% | 0 | 22 | 5 | 0.08% | 2.2% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 44 | 16 | 36.4% | 4 | 35 | 5 | 0.63% | 27.7% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-01-15 09:15:00)

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

### Cycle 2 — BUY (started 2025-03-21 11:15:00)

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

### Cycle 3 — SELL (started 2025-09-03 12:15:00)

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

### Cycle 4 — BUY (started 2026-01-12 13:15:00)

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

### Cycle 5 — SELL (started 2026-01-23 13:15:00)

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

### Cycle 6 — BUY (started 2026-02-06 12:15:00)

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

### Cycle 7 — SELL (started 2026-03-06 14:15:00)

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
