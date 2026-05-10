# Dr. Lal Path Labs Ltd. (LALPATHLAB)

## Backtest Summary

- **Window:** 2025-04-11 09:15:00 → 2026-05-08 15:15:00 (1850 bars)
- **Last close:** 1655.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 75 |
| ALERT1 | 51 |
| ALERT2 | 50 |
| ALERT2_SKIP | 32 |
| ALERT3 | 136 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 1 |
| ENTRY2 | 60 |
| PARTIAL | 12 |
| TARGET_HIT | 1 |
| STOP_HIT | 59 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 72 (incl. partial bookings)
- **Trades open at end:** 1
- **Winners / losers:** 31 / 41
- **Target hits / Stop hits / Partials:** 1 / 59 / 12
- **Avg / median % per leg:** 1.08% / -0.59%
- **Sum % (uncompounded):** 77.64%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 23 | 3 | 13.0% | 1 | 22 | 0 | -0.33% | -7.5% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 23 | 3 | 13.0% | 1 | 22 | 0 | -0.33% | -7.5% |
| SELL (all) | 49 | 28 | 57.1% | 0 | 37 | 12 | 1.74% | 85.2% |
| SELL @ 2nd Alert (retest1) | 2 | 2 | 100.0% | 0 | 1 | 1 | 4.66% | 9.3% |
| SELL @ 3rd Alert (retest2) | 47 | 26 | 55.3% | 0 | 36 | 11 | 1.61% | 75.8% |
| retest1 (combined) | 2 | 2 | 100.0% | 0 | 1 | 1 | 4.66% | 9.3% |
| retest2 (combined) | 70 | 29 | 41.4% | 1 | 58 | 11 | 0.98% | 68.3% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 12:15:00 | 1411.80 | 1402.20 | 1401.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 15:15:00 | 1418.85 | 1407.39 | 1404.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-13 10:15:00 | 1404.95 | 1407.48 | 1404.84 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-13 10:15:00 | 1404.95 | 1407.48 | 1404.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-13 10:15:00 | 1404.95 | 1407.48 | 1404.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-13 10:30:00 | 1407.50 | 1407.48 | 1404.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-13 11:15:00 | 1388.05 | 1403.59 | 1403.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-13 11:30:00 | 1389.30 | 1403.59 | 1403.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 2 — SELL (started 2025-05-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-13 12:15:00 | 1376.00 | 1398.07 | 1400.83 | EMA200 below EMA400 |

### Cycle 3 — BUY (started 2025-05-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-14 14:15:00 | 1404.00 | 1399.86 | 1399.65 | EMA200 above EMA400 |

### Cycle 4 — SELL (started 2025-05-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-14 15:15:00 | 1397.50 | 1399.39 | 1399.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-15 09:15:00 | 1393.65 | 1398.24 | 1398.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-15 13:15:00 | 1397.75 | 1396.83 | 1397.89 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-15 13:15:00 | 1397.75 | 1396.83 | 1397.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-15 13:15:00 | 1397.75 | 1396.83 | 1397.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-15 14:00:00 | 1397.75 | 1396.83 | 1397.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-15 14:15:00 | 1396.15 | 1396.70 | 1397.73 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-15 15:15:00 | 1394.00 | 1396.70 | 1397.73 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-16 09:30:00 | 1391.05 | 1393.03 | 1395.85 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-16 14:15:00 | 1403.05 | 1391.04 | 1393.21 | SL hit (close>static) qty=1.00 sl=1398.95 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-05-16 14:15:00 | 1403.05 | 1391.04 | 1393.21 | SL hit (close>static) qty=1.00 sl=1398.95 alert=retest2 |

### Cycle 5 — BUY (started 2025-05-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-19 10:15:00 | 1397.90 | 1395.17 | 1394.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-20 13:15:00 | 1405.85 | 1399.77 | 1397.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-21 11:15:00 | 1406.65 | 1410.18 | 1404.47 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-21 12:00:00 | 1406.65 | 1410.18 | 1404.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 12:15:00 | 1413.75 | 1410.90 | 1405.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-21 12:45:00 | 1411.65 | 1410.90 | 1405.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 15:15:00 | 1419.00 | 1413.90 | 1408.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-22 09:15:00 | 1404.95 | 1413.90 | 1408.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 09:15:00 | 1418.70 | 1414.86 | 1409.14 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-23 09:30:00 | 1430.00 | 1417.08 | 1412.88 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-23 11:15:00 | 1424.90 | 1418.34 | 1413.84 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-23 12:15:00 | 1424.65 | 1419.33 | 1414.70 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-23 15:00:00 | 1427.30 | 1421.23 | 1416.74 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 12:15:00 | 1426.55 | 1431.76 | 1424.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-26 12:45:00 | 1425.75 | 1431.76 | 1424.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 13:15:00 | 1427.80 | 1430.97 | 1424.69 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-05-28 09:15:00 | 1413.45 | 1422.34 | 1423.24 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-05-28 09:15:00 | 1413.45 | 1422.34 | 1423.24 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-05-28 09:15:00 | 1413.45 | 1422.34 | 1423.24 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-05-28 09:15:00 | 1413.45 | 1422.34 | 1423.24 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 6 — SELL (started 2025-05-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-28 09:15:00 | 1413.45 | 1422.34 | 1423.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-28 10:15:00 | 1405.75 | 1419.02 | 1421.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-29 09:15:00 | 1417.50 | 1415.13 | 1418.00 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-29 09:15:00 | 1417.50 | 1415.13 | 1418.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 09:15:00 | 1417.50 | 1415.13 | 1418.00 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-30 14:45:00 | 1407.50 | 1411.13 | 1414.72 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-03 09:30:00 | 1400.30 | 1397.12 | 1402.12 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-03 12:15:00 | 1413.55 | 1405.69 | 1405.28 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-03 12:15:00 | 1413.55 | 1405.69 | 1405.28 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 7 — BUY (started 2025-06-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-03 12:15:00 | 1413.55 | 1405.69 | 1405.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-04 09:15:00 | 1421.00 | 1410.47 | 1407.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-06 14:15:00 | 1424.05 | 1428.84 | 1425.37 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-06 14:15:00 | 1424.05 | 1428.84 | 1425.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 14:15:00 | 1424.05 | 1428.84 | 1425.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-06 15:00:00 | 1424.05 | 1428.84 | 1425.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 15:15:00 | 1424.20 | 1427.91 | 1425.27 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-09 09:15:00 | 1430.05 | 1427.91 | 1425.27 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-10 09:30:00 | 1425.90 | 1436.34 | 1431.92 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-16 09:15:00 | 1453.10 | 1473.85 | 1475.49 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-16 09:15:00 | 1453.10 | 1473.85 | 1475.49 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 8 — SELL (started 2025-06-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-16 09:15:00 | 1453.10 | 1473.85 | 1475.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-19 10:15:00 | 1445.90 | 1454.20 | 1459.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-20 11:15:00 | 1452.65 | 1445.59 | 1450.73 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-20 11:15:00 | 1452.65 | 1445.59 | 1450.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 11:15:00 | 1452.65 | 1445.59 | 1450.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 12:00:00 | 1452.65 | 1445.59 | 1450.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 12:15:00 | 1448.45 | 1446.16 | 1450.52 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-23 09:15:00 | 1424.05 | 1450.83 | 1451.90 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-30 09:15:00 | 1400.60 | 1386.36 | 1385.72 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 9 — BUY (started 2025-06-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-30 09:15:00 | 1400.60 | 1386.36 | 1385.72 | EMA200 above EMA400 |

### Cycle 10 — SELL (started 2025-07-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-01 14:15:00 | 1384.90 | 1389.94 | 1389.98 | EMA200 below EMA400 |

### Cycle 11 — BUY (started 2025-07-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-01 15:15:00 | 1395.00 | 1390.95 | 1390.43 | EMA200 above EMA400 |

### Cycle 12 — SELL (started 2025-07-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-02 10:15:00 | 1386.95 | 1389.46 | 1389.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-03 09:15:00 | 1376.15 | 1385.38 | 1387.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-04 09:15:00 | 1394.45 | 1381.93 | 1383.69 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-04 09:15:00 | 1394.45 | 1381.93 | 1383.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 09:15:00 | 1394.45 | 1381.93 | 1383.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-04 10:00:00 | 1394.45 | 1381.93 | 1383.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 13 — BUY (started 2025-07-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-04 10:15:00 | 1414.00 | 1388.34 | 1386.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-04 14:15:00 | 1424.70 | 1403.80 | 1395.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-08 10:15:00 | 1417.35 | 1422.03 | 1412.86 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-08 11:00:00 | 1417.35 | 1422.03 | 1412.86 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 10:15:00 | 1457.70 | 1465.18 | 1450.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-10 10:30:00 | 1458.00 | 1465.18 | 1450.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 09:15:00 | 1463.90 | 1463.95 | 1456.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-11 09:30:00 | 1459.95 | 1463.95 | 1456.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 13:15:00 | 1453.00 | 1460.97 | 1457.23 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-11 14:45:00 | 1464.50 | 1461.45 | 1457.79 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-14 10:15:00 | 1441.75 | 1456.01 | 1456.14 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 14 — SELL (started 2025-07-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-14 10:15:00 | 1441.75 | 1456.01 | 1456.14 | EMA200 below EMA400 |

### Cycle 15 — BUY (started 2025-07-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-15 11:15:00 | 1460.75 | 1455.13 | 1454.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-15 12:15:00 | 1463.15 | 1456.73 | 1455.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-17 11:15:00 | 1482.85 | 1486.46 | 1478.95 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-17 12:00:00 | 1482.85 | 1486.46 | 1478.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 09:15:00 | 1500.40 | 1491.39 | 1484.33 | EMA400 retest candle locked (from upside) |

### Cycle 16 — SELL (started 2025-07-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-21 14:15:00 | 1480.95 | 1486.36 | 1486.76 | EMA200 below EMA400 |

### Cycle 17 — BUY (started 2025-07-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-22 10:15:00 | 1499.25 | 1488.66 | 1487.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-22 11:15:00 | 1501.60 | 1491.25 | 1488.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-23 12:15:00 | 1504.05 | 1504.13 | 1498.10 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-23 12:30:00 | 1506.70 | 1504.13 | 1498.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 10:15:00 | 1525.50 | 1534.75 | 1529.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-28 10:30:00 | 1529.35 | 1534.75 | 1529.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 11:15:00 | 1522.10 | 1532.22 | 1528.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-28 12:00:00 | 1522.10 | 1532.22 | 1528.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 18 — SELL (started 2025-07-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-28 13:15:00 | 1508.90 | 1523.88 | 1525.10 | EMA200 below EMA400 |

### Cycle 19 — BUY (started 2025-07-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-30 09:15:00 | 1554.90 | 1523.55 | 1522.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-31 13:15:00 | 1572.00 | 1542.63 | 1537.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-04 09:15:00 | 1562.35 | 1575.77 | 1563.41 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-04 09:15:00 | 1562.35 | 1575.77 | 1563.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 09:15:00 | 1562.35 | 1575.77 | 1563.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-04 09:45:00 | 1566.70 | 1575.77 | 1563.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 10:15:00 | 1570.00 | 1574.61 | 1564.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-04 10:45:00 | 1559.80 | 1574.61 | 1564.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 11:15:00 | 1573.90 | 1574.47 | 1564.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-04 11:30:00 | 1571.40 | 1574.47 | 1564.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 12:15:00 | 1567.10 | 1573.00 | 1565.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-04 13:00:00 | 1567.10 | 1573.00 | 1565.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 13:15:00 | 1564.55 | 1571.31 | 1565.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-04 13:30:00 | 1565.95 | 1571.31 | 1565.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 14:15:00 | 1549.75 | 1567.00 | 1563.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-04 15:00:00 | 1549.75 | 1567.00 | 1563.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 15:15:00 | 1550.25 | 1563.65 | 1562.45 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-05 09:15:00 | 1587.80 | 1563.65 | 1562.45 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-08-19 09:15:00 | 1746.58 | 1707.75 | 1680.97 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 20 — SELL (started 2025-08-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-21 13:15:00 | 1691.75 | 1701.88 | 1703.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-25 09:15:00 | 1682.60 | 1691.08 | 1695.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-26 09:15:00 | 1683.85 | 1683.41 | 1688.76 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-26 10:00:00 | 1683.85 | 1683.41 | 1688.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 10:15:00 | 1688.00 | 1684.33 | 1688.69 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-26 11:15:00 | 1682.00 | 1684.33 | 1688.69 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-29 10:00:00 | 1675.20 | 1666.14 | 1671.18 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-03 11:15:00 | 1664.50 | 1645.02 | 1643.37 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-03 11:15:00 | 1664.50 | 1645.02 | 1643.37 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 21 — BUY (started 2025-09-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-03 11:15:00 | 1664.50 | 1645.02 | 1643.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-04 09:15:00 | 1697.50 | 1666.20 | 1655.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-04 14:15:00 | 1655.05 | 1674.51 | 1664.98 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-04 14:15:00 | 1655.05 | 1674.51 | 1664.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 14:15:00 | 1655.05 | 1674.51 | 1664.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 15:00:00 | 1655.05 | 1674.51 | 1664.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 15:15:00 | 1655.00 | 1670.61 | 1664.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-05 09:15:00 | 1658.65 | 1670.61 | 1664.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 09:15:00 | 1659.90 | 1668.47 | 1663.69 | EMA400 retest candle locked (from upside) |

### Cycle 22 — SELL (started 2025-09-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-05 11:15:00 | 1630.50 | 1657.34 | 1659.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-05 12:15:00 | 1624.95 | 1650.86 | 1656.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-10 09:15:00 | 1596.85 | 1595.72 | 1608.22 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-10 09:15:00 | 1596.85 | 1595.72 | 1608.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 09:15:00 | 1596.85 | 1595.72 | 1608.22 | EMA400 retest candle locked (from downside) |

### Cycle 23 — BUY (started 2025-09-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-11 11:15:00 | 1624.10 | 1611.19 | 1610.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-11 13:15:00 | 1635.00 | 1617.04 | 1613.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-15 09:15:00 | 1639.95 | 1654.97 | 1641.85 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-15 09:15:00 | 1639.95 | 1654.97 | 1641.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 09:15:00 | 1639.95 | 1654.97 | 1641.85 | EMA400 retest candle locked (from upside) |

### Cycle 24 — SELL (started 2025-09-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-16 15:15:00 | 1633.50 | 1644.02 | 1644.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-17 12:15:00 | 1625.00 | 1639.34 | 1642.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-18 09:15:00 | 1638.55 | 1635.78 | 1639.28 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-18 09:15:00 | 1638.55 | 1635.78 | 1639.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 09:15:00 | 1638.55 | 1635.78 | 1639.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-18 09:45:00 | 1634.65 | 1635.78 | 1639.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 10:15:00 | 1641.30 | 1636.88 | 1639.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-18 10:45:00 | 1643.20 | 1636.88 | 1639.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 11:15:00 | 1641.00 | 1637.71 | 1639.60 | EMA400 retest candle locked (from downside) |

### Cycle 25 — BUY (started 2025-09-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-19 09:15:00 | 1686.05 | 1649.14 | 1644.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-19 14:15:00 | 1734.75 | 1678.57 | 1661.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-22 12:15:00 | 1690.00 | 1690.36 | 1675.07 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-22 13:00:00 | 1690.00 | 1690.36 | 1675.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 09:15:00 | 1654.65 | 1684.32 | 1677.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-23 10:00:00 | 1654.65 | 1684.32 | 1677.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 10:15:00 | 1660.65 | 1679.59 | 1675.76 | EMA400 retest candle locked (from upside) |

### Cycle 26 — SELL (started 2025-09-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-23 13:15:00 | 1665.05 | 1672.28 | 1672.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-24 09:15:00 | 1652.00 | 1666.18 | 1669.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-24 15:15:00 | 1657.70 | 1654.48 | 1661.03 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-25 09:15:00 | 1641.25 | 1654.48 | 1661.03 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-29 11:15:00 | 1559.19 | 1587.17 | 1609.74 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-10-01 09:15:00 | 1570.40 | 1565.47 | 1578.20 | SL hit (close>ema200) qty=0.50 sl=1565.47 alert=retest1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 10:15:00 | 1585.50 | 1569.47 | 1578.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-01 10:30:00 | 1585.05 | 1569.47 | 1578.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 11:15:00 | 1587.30 | 1573.04 | 1579.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-01 11:30:00 | 1585.00 | 1573.04 | 1579.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 09:15:00 | 1575.40 | 1578.64 | 1580.64 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-03 10:45:00 | 1565.00 | 1575.67 | 1579.11 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-03 13:00:00 | 1564.40 | 1571.16 | 1576.32 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-03 15:00:00 | 1565.00 | 1568.73 | 1574.25 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-06 09:45:00 | 1565.50 | 1568.82 | 1573.27 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 10:15:00 | 1581.65 | 1571.39 | 1574.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-06 10:45:00 | 1589.90 | 1571.39 | 1574.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 11:15:00 | 1567.30 | 1570.57 | 1573.42 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-10-06 14:15:00 | 1601.25 | 1575.73 | 1574.98 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-06 14:15:00 | 1601.25 | 1575.73 | 1574.98 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-06 14:15:00 | 1601.25 | 1575.73 | 1574.98 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-06 14:15:00 | 1601.25 | 1575.73 | 1574.98 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 27 — BUY (started 2025-10-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-06 14:15:00 | 1601.25 | 1575.73 | 1574.98 | EMA200 above EMA400 |

### Cycle 28 — SELL (started 2025-10-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-07 11:15:00 | 1568.70 | 1573.81 | 1574.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-07 12:15:00 | 1562.60 | 1571.57 | 1573.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-07 14:15:00 | 1572.50 | 1571.02 | 1572.73 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-07 14:15:00 | 1572.50 | 1571.02 | 1572.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 14:15:00 | 1572.50 | 1571.02 | 1572.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-07 15:00:00 | 1572.50 | 1571.02 | 1572.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 15:15:00 | 1573.00 | 1571.42 | 1572.76 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-08 09:15:00 | 1559.35 | 1571.42 | 1572.76 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-13 09:15:00 | 1569.90 | 1548.23 | 1546.31 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 29 — BUY (started 2025-10-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-13 09:15:00 | 1569.90 | 1548.23 | 1546.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-13 11:15:00 | 1578.35 | 1558.51 | 1551.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-14 09:15:00 | 1577.90 | 1583.99 | 1568.83 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-14 10:00:00 | 1577.90 | 1583.99 | 1568.83 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 14:15:00 | 1589.80 | 1587.84 | 1576.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-14 14:45:00 | 1586.90 | 1587.84 | 1576.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 09:15:00 | 1565.50 | 1582.36 | 1575.99 | EMA400 retest candle locked (from upside) |

### Cycle 30 — SELL (started 2025-10-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-15 14:15:00 | 1557.30 | 1569.76 | 1571.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-15 15:15:00 | 1555.00 | 1566.81 | 1569.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-16 11:15:00 | 1579.00 | 1566.61 | 1568.74 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-16 11:15:00 | 1579.00 | 1566.61 | 1568.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 11:15:00 | 1579.00 | 1566.61 | 1568.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-16 12:00:00 | 1579.00 | 1566.61 | 1568.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 31 — BUY (started 2025-10-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-16 12:15:00 | 1589.85 | 1571.26 | 1570.66 | EMA200 above EMA400 |

### Cycle 32 — SELL (started 2025-10-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-17 13:15:00 | 1562.25 | 1572.15 | 1572.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-17 14:15:00 | 1557.25 | 1569.17 | 1571.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-20 11:15:00 | 1564.30 | 1561.97 | 1566.38 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-20 11:45:00 | 1562.35 | 1561.97 | 1566.38 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 12:15:00 | 1587.50 | 1567.08 | 1568.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-20 12:45:00 | 1589.35 | 1567.08 | 1568.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 33 — BUY (started 2025-10-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-20 13:15:00 | 1580.95 | 1569.85 | 1569.45 | EMA200 above EMA400 |

### Cycle 34 — SELL (started 2025-10-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-23 09:15:00 | 1542.75 | 1564.43 | 1567.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-23 10:15:00 | 1536.00 | 1558.75 | 1564.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-24 14:15:00 | 1525.20 | 1522.70 | 1536.51 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-24 15:00:00 | 1525.20 | 1522.70 | 1536.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 09:15:00 | 1560.80 | 1529.98 | 1537.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-27 09:45:00 | 1585.50 | 1529.98 | 1537.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 10:15:00 | 1565.00 | 1536.98 | 1539.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-27 10:30:00 | 1564.95 | 1536.98 | 1539.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 35 — BUY (started 2025-10-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-27 12:15:00 | 1568.75 | 1547.15 | 1544.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-27 14:15:00 | 1572.50 | 1554.67 | 1548.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-28 09:15:00 | 1558.25 | 1558.56 | 1551.40 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-28 09:15:00 | 1558.25 | 1558.56 | 1551.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 09:15:00 | 1558.25 | 1558.56 | 1551.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-28 10:00:00 | 1558.25 | 1558.56 | 1551.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 11:15:00 | 1540.95 | 1554.33 | 1550.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-28 12:00:00 | 1540.95 | 1554.33 | 1550.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 12:15:00 | 1550.00 | 1553.47 | 1550.61 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-28 13:15:00 | 1560.00 | 1553.47 | 1550.61 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-29 09:15:00 | 1534.65 | 1546.71 | 1548.03 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 36 — SELL (started 2025-10-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-29 09:15:00 | 1534.65 | 1546.71 | 1548.03 | EMA200 below EMA400 |

### Cycle 37 — BUY (started 2025-10-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-30 10:15:00 | 1553.85 | 1548.15 | 1547.77 | EMA200 above EMA400 |

### Cycle 38 — SELL (started 2025-10-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-30 11:15:00 | 1537.50 | 1546.02 | 1546.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-31 12:15:00 | 1533.85 | 1540.83 | 1543.53 | Break + close below crossover candle low |

### Cycle 39 — BUY (started 2025-10-31 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-31 13:15:00 | 1569.90 | 1546.64 | 1545.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-03 09:15:00 | 1616.00 | 1566.86 | 1555.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-04 09:15:00 | 1591.05 | 1608.10 | 1587.99 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-04 09:30:00 | 1590.00 | 1608.10 | 1587.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 10:15:00 | 1573.90 | 1601.26 | 1586.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-04 11:00:00 | 1573.90 | 1601.26 | 1586.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 11:15:00 | 1582.10 | 1597.43 | 1586.29 | EMA400 retest candle locked (from upside) |

### Cycle 40 — SELL (started 2025-11-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-06 09:15:00 | 1560.30 | 1579.43 | 1580.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-06 10:15:00 | 1551.35 | 1573.82 | 1577.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-06 12:15:00 | 1573.70 | 1571.59 | 1575.96 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-06 12:15:00 | 1573.70 | 1571.59 | 1575.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 12:15:00 | 1573.70 | 1571.59 | 1575.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-06 12:45:00 | 1575.15 | 1571.59 | 1575.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 13:15:00 | 1575.00 | 1572.27 | 1575.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-06 13:30:00 | 1575.90 | 1572.27 | 1575.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 14:15:00 | 1546.30 | 1567.08 | 1573.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-06 14:30:00 | 1576.15 | 1567.08 | 1573.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 10:15:00 | 1548.80 | 1546.98 | 1554.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-10 11:00:00 | 1548.80 | 1546.98 | 1554.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 11:15:00 | 1561.25 | 1549.84 | 1555.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-10 11:45:00 | 1564.65 | 1549.84 | 1555.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 12:15:00 | 1555.70 | 1551.01 | 1555.48 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-10 13:45:00 | 1546.50 | 1550.60 | 1554.89 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-13 10:15:00 | 1571.70 | 1545.81 | 1543.41 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 41 — BUY (started 2025-11-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-13 10:15:00 | 1571.70 | 1545.81 | 1543.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-13 11:15:00 | 1578.95 | 1552.44 | 1546.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-17 09:15:00 | 1571.15 | 1573.66 | 1566.42 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-17 09:15:00 | 1571.15 | 1573.66 | 1566.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 09:15:00 | 1571.15 | 1573.66 | 1566.42 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-17 15:00:00 | 1580.00 | 1573.88 | 1569.09 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-18 10:15:00 | 1577.90 | 1575.21 | 1570.56 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-18 10:45:00 | 1578.85 | 1575.67 | 1571.19 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-18 11:15:00 | 1578.05 | 1575.67 | 1571.19 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 09:15:00 | 1585.60 | 1603.51 | 1595.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-20 10:00:00 | 1585.60 | 1603.51 | 1595.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 10:15:00 | 1568.15 | 1596.44 | 1593.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-20 11:00:00 | 1568.15 | 1596.44 | 1593.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-11-20 11:15:00 | 1565.05 | 1590.16 | 1590.87 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-20 11:15:00 | 1565.05 | 1590.16 | 1590.87 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-20 11:15:00 | 1565.05 | 1590.16 | 1590.87 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-20 11:15:00 | 1565.05 | 1590.16 | 1590.87 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 42 — SELL (started 2025-11-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-20 11:15:00 | 1565.05 | 1590.16 | 1590.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-21 09:15:00 | 1549.35 | 1570.24 | 1579.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-21 13:15:00 | 1577.65 | 1567.22 | 1574.63 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-21 13:15:00 | 1577.65 | 1567.22 | 1574.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 13:15:00 | 1577.65 | 1567.22 | 1574.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-21 13:45:00 | 1575.60 | 1567.22 | 1574.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 14:15:00 | 1571.50 | 1568.08 | 1574.35 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-24 09:15:00 | 1562.20 | 1568.36 | 1573.91 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-24 11:15:00 | 1564.80 | 1567.52 | 1572.52 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-24 13:45:00 | 1566.60 | 1568.01 | 1571.61 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-25 15:15:00 | 1565.90 | 1557.86 | 1562.06 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 15:15:00 | 1565.90 | 1559.47 | 1562.41 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-26 09:15:00 | 1555.50 | 1559.47 | 1562.41 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-26 11:30:00 | 1555.55 | 1557.23 | 1560.54 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-27 10:15:00 | 1555.70 | 1553.36 | 1556.86 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-27 13:00:00 | 1555.00 | 1556.73 | 1557.84 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 13:15:00 | 1555.00 | 1556.39 | 1557.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-27 13:30:00 | 1556.50 | 1556.39 | 1557.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 14:15:00 | 1557.20 | 1556.55 | 1557.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-27 15:00:00 | 1557.20 | 1556.55 | 1557.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 15:15:00 | 1560.00 | 1557.24 | 1557.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-28 09:15:00 | 1553.65 | 1557.24 | 1557.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 09:15:00 | 1542.50 | 1554.29 | 1556.38 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-28 10:15:00 | 1531.15 | 1554.29 | 1556.38 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-03 11:15:00 | 1484.09 | 1512.98 | 1520.84 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-03 11:15:00 | 1486.56 | 1512.98 | 1520.84 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-03 11:15:00 | 1488.27 | 1512.98 | 1520.84 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-03 11:15:00 | 1487.61 | 1512.98 | 1520.84 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-04 09:15:00 | 1505.80 | 1504.82 | 1513.18 | SL hit (close>ema200) qty=0.50 sl=1504.82 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-04 09:15:00 | 1505.80 | 1504.82 | 1513.18 | SL hit (close>ema200) qty=0.50 sl=1504.82 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-04 09:15:00 | 1505.80 | 1504.82 | 1513.18 | SL hit (close>ema200) qty=0.50 sl=1504.82 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-04 09:15:00 | 1505.80 | 1504.82 | 1513.18 | SL hit (close>ema200) qty=0.50 sl=1504.82 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-08 15:15:00 | 1477.72 | 1489.83 | 1498.35 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-08 15:15:00 | 1477.77 | 1489.83 | 1498.35 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-08 15:15:00 | 1477.91 | 1489.83 | 1498.35 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-08 15:15:00 | 1477.25 | 1489.83 | 1498.35 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-09 09:15:00 | 1454.59 | 1482.51 | 1494.25 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-11 13:15:00 | 1434.85 | 1434.08 | 1447.51 | SL hit (close>ema200) qty=0.50 sl=1434.08 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-11 13:15:00 | 1434.85 | 1434.08 | 1447.51 | SL hit (close>ema200) qty=0.50 sl=1434.08 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-11 13:15:00 | 1434.85 | 1434.08 | 1447.51 | SL hit (close>ema200) qty=0.50 sl=1434.08 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-11 13:15:00 | 1434.85 | 1434.08 | 1447.51 | SL hit (close>ema200) qty=0.50 sl=1434.08 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-11 13:15:00 | 1434.85 | 1434.08 | 1447.51 | SL hit (close>ema200) qty=0.50 sl=1434.08 alert=retest2 |

### Cycle 43 — BUY (started 2025-12-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-22 13:15:00 | 1416.20 | 1400.37 | 1398.57 | EMA200 above EMA400 |

### Cycle 44 — SELL (started 2025-12-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-26 12:15:00 | 1404.10 | 1405.41 | 1405.50 | EMA200 below EMA400 |

### Cycle 45 — BUY (started 2025-12-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-26 13:15:00 | 1420.10 | 1408.35 | 1406.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-29 14:15:00 | 1434.90 | 1419.08 | 1413.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-02 09:15:00 | 1484.90 | 1493.57 | 1477.21 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-02 10:15:00 | 1478.70 | 1493.57 | 1477.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 10:15:00 | 1470.60 | 1488.97 | 1476.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-02 10:45:00 | 1471.10 | 1488.97 | 1476.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 11:15:00 | 1471.10 | 1485.40 | 1476.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-02 12:00:00 | 1471.10 | 1485.40 | 1476.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 14:15:00 | 1476.90 | 1478.07 | 1474.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-02 14:45:00 | 1464.50 | 1478.07 | 1474.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 09:15:00 | 1480.00 | 1477.98 | 1475.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-05 09:30:00 | 1474.00 | 1477.98 | 1475.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 11:15:00 | 1485.00 | 1491.58 | 1486.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-06 12:00:00 | 1485.00 | 1491.58 | 1486.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 12:15:00 | 1486.00 | 1490.47 | 1486.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-06 13:15:00 | 1481.90 | 1490.47 | 1486.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 13:15:00 | 1476.60 | 1487.69 | 1485.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-06 13:45:00 | 1473.50 | 1487.69 | 1485.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 14:15:00 | 1477.20 | 1485.59 | 1484.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-06 15:00:00 | 1477.20 | 1485.59 | 1484.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 46 — SELL (started 2026-01-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-07 09:15:00 | 1469.60 | 1481.50 | 1482.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-07 10:15:00 | 1455.30 | 1476.26 | 1480.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-07 13:15:00 | 1470.00 | 1468.61 | 1475.24 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-07 13:30:00 | 1475.20 | 1468.61 | 1475.24 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 14:15:00 | 1477.10 | 1470.31 | 1475.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-07 14:45:00 | 1476.20 | 1470.31 | 1475.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 15:15:00 | 1470.00 | 1470.25 | 1474.92 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-08 09:15:00 | 1457.10 | 1470.25 | 1474.92 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-08 10:00:00 | 1456.20 | 1467.44 | 1473.22 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-12 11:15:00 | 1384.24 | 1412.88 | 1431.19 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-12 12:15:00 | 1383.39 | 1406.92 | 1426.82 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-13 10:15:00 | 1405.50 | 1396.03 | 1412.67 | SL hit (close>ema200) qty=0.50 sl=1396.03 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-13 10:15:00 | 1405.50 | 1396.03 | 1412.67 | SL hit (close>ema200) qty=0.50 sl=1396.03 alert=retest2 |

### Cycle 47 — BUY (started 2026-01-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-14 10:15:00 | 1431.10 | 1415.11 | 1414.75 | EMA200 above EMA400 |

### Cycle 48 — SELL (started 2026-01-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-14 15:15:00 | 1408.00 | 1414.34 | 1414.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-16 09:15:00 | 1398.40 | 1411.16 | 1413.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-16 15:15:00 | 1404.30 | 1400.87 | 1406.03 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-16 15:15:00 | 1404.30 | 1400.87 | 1406.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 15:15:00 | 1404.30 | 1400.87 | 1406.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-19 09:15:00 | 1388.00 | 1400.87 | 1406.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 09:15:00 | 1386.80 | 1398.06 | 1404.28 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-19 13:15:00 | 1380.10 | 1394.56 | 1401.04 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-19 14:00:00 | 1380.00 | 1391.65 | 1399.12 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-20 09:15:00 | 1366.00 | 1390.31 | 1397.12 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-20 15:00:00 | 1374.80 | 1376.93 | 1386.16 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-21 09:15:00 | 1359.20 | 1372.95 | 1382.72 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-01-22 10:15:00 | 1398.50 | 1385.55 | 1384.28 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-22 10:15:00 | 1398.50 | 1385.55 | 1384.28 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-22 10:15:00 | 1398.50 | 1385.55 | 1384.28 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-22 10:15:00 | 1398.50 | 1385.55 | 1384.28 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 49 — BUY (started 2026-01-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-22 10:15:00 | 1398.50 | 1385.55 | 1384.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-22 12:15:00 | 1414.30 | 1393.97 | 1388.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-23 09:15:00 | 1400.20 | 1403.09 | 1395.40 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-23 09:15:00 | 1400.20 | 1403.09 | 1395.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 09:15:00 | 1400.20 | 1403.09 | 1395.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-23 10:15:00 | 1395.90 | 1403.09 | 1395.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 10:15:00 | 1403.80 | 1403.24 | 1396.17 | EMA400 retest candle locked (from upside) |

### Cycle 50 — SELL (started 2026-01-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-23 14:15:00 | 1377.40 | 1390.35 | 1391.73 | EMA200 below EMA400 |

### Cycle 51 — BUY (started 2026-01-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-27 11:15:00 | 1405.40 | 1392.15 | 1391.87 | EMA200 above EMA400 |

### Cycle 52 — SELL (started 2026-01-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-28 11:15:00 | 1391.60 | 1392.95 | 1392.97 | EMA200 below EMA400 |

### Cycle 53 — BUY (started 2026-01-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-28 13:15:00 | 1393.70 | 1393.01 | 1392.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-28 14:15:00 | 1403.70 | 1395.15 | 1393.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-29 09:15:00 | 1387.20 | 1394.35 | 1393.86 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-29 09:15:00 | 1387.20 | 1394.35 | 1393.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 09:15:00 | 1387.20 | 1394.35 | 1393.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-29 10:00:00 | 1387.20 | 1394.35 | 1393.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 10:15:00 | 1399.60 | 1395.40 | 1394.38 | EMA400 retest candle locked (from upside) |

### Cycle 54 — SELL (started 2026-01-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-29 12:15:00 | 1383.30 | 1392.26 | 1393.08 | EMA200 below EMA400 |

### Cycle 55 — BUY (started 2026-01-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-30 09:15:00 | 1395.60 | 1393.47 | 1393.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-30 10:15:00 | 1406.40 | 1396.06 | 1394.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-30 15:15:00 | 1398.90 | 1401.06 | 1398.04 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-30 15:15:00 | 1398.90 | 1401.06 | 1398.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 15:15:00 | 1398.90 | 1401.06 | 1398.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 09:15:00 | 1382.80 | 1401.06 | 1398.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 09:15:00 | 1382.10 | 1397.27 | 1396.59 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-01 11:00:00 | 1413.50 | 1400.52 | 1398.12 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-02 13:30:00 | 1405.50 | 1407.22 | 1406.47 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-02 14:30:00 | 1406.90 | 1409.10 | 1407.39 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-03 10:00:00 | 1406.30 | 1411.10 | 1408.75 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 10:15:00 | 1402.20 | 1409.32 | 1408.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-03 10:30:00 | 1400.00 | 1409.32 | 1408.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 11:15:00 | 1411.10 | 1409.68 | 1408.42 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-02-03 14:15:00 | 1405.10 | 1407.35 | 1407.54 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-03 14:15:00 | 1405.10 | 1407.35 | 1407.54 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-03 14:15:00 | 1405.10 | 1407.35 | 1407.54 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-03 14:15:00 | 1405.10 | 1407.35 | 1407.54 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 56 — SELL (started 2026-02-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-03 14:15:00 | 1405.10 | 1407.35 | 1407.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-04 09:15:00 | 1398.50 | 1404.89 | 1406.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-04 11:15:00 | 1404.10 | 1403.63 | 1405.45 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-04 12:00:00 | 1404.10 | 1403.63 | 1405.45 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 12:15:00 | 1407.00 | 1404.30 | 1405.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-04 12:45:00 | 1405.00 | 1404.30 | 1405.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 13:15:00 | 1405.10 | 1404.46 | 1405.54 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-04 15:15:00 | 1398.50 | 1405.21 | 1405.79 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-05 13:15:00 | 1414.20 | 1405.53 | 1405.10 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 57 — BUY (started 2026-02-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-05 13:15:00 | 1414.20 | 1405.53 | 1405.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-06 09:15:00 | 1423.80 | 1410.48 | 1407.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-09 13:15:00 | 1435.50 | 1438.58 | 1429.08 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-09 14:00:00 | 1435.50 | 1438.58 | 1429.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 14:15:00 | 1447.40 | 1440.34 | 1430.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-09 14:45:00 | 1429.80 | 1440.34 | 1430.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 15:15:00 | 1426.50 | 1437.58 | 1430.36 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-10 09:45:00 | 1452.50 | 1438.76 | 1431.55 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-10 10:45:00 | 1458.70 | 1442.79 | 1434.04 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-10 14:45:00 | 1454.50 | 1450.09 | 1440.64 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-11 09:15:00 | 1462.10 | 1447.87 | 1440.49 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 12:15:00 | 1445.10 | 1451.17 | 1444.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-11 13:00:00 | 1445.10 | 1451.17 | 1444.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 13:15:00 | 1444.40 | 1449.82 | 1444.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-11 14:00:00 | 1444.40 | 1449.82 | 1444.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 14:15:00 | 1443.20 | 1448.49 | 1444.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-11 14:30:00 | 1441.70 | 1448.49 | 1444.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 15:15:00 | 1445.50 | 1447.89 | 1444.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-12 09:15:00 | 1445.40 | 1447.89 | 1444.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 09:15:00 | 1447.30 | 1447.78 | 1445.06 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-12 10:15:00 | 1455.30 | 1447.78 | 1445.06 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-12 14:15:00 | 1432.80 | 1443.71 | 1444.06 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-12 14:15:00 | 1432.80 | 1443.71 | 1444.06 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-12 14:15:00 | 1432.80 | 1443.71 | 1444.06 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-12 14:15:00 | 1432.80 | 1443.71 | 1444.06 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-12 14:15:00 | 1432.80 | 1443.71 | 1444.06 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 58 — SELL (started 2026-02-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-12 14:15:00 | 1432.80 | 1443.71 | 1444.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-13 09:15:00 | 1416.00 | 1436.93 | 1440.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-16 09:15:00 | 1438.10 | 1417.85 | 1426.33 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-16 09:15:00 | 1438.10 | 1417.85 | 1426.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 09:15:00 | 1438.10 | 1417.85 | 1426.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-16 09:30:00 | 1449.00 | 1417.85 | 1426.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 10:15:00 | 1444.00 | 1423.08 | 1427.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-16 10:45:00 | 1442.60 | 1423.08 | 1427.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 14:15:00 | 1419.50 | 1424.62 | 1427.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-16 14:45:00 | 1420.20 | 1424.62 | 1427.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 15:15:00 | 1422.70 | 1424.24 | 1427.04 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-17 09:15:00 | 1411.60 | 1424.24 | 1427.04 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-17 10:15:00 | 1436.00 | 1426.38 | 1427.51 | SL hit (close>static) qty=1.00 sl=1434.90 alert=retest2 |

### Cycle 59 — BUY (started 2026-02-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-17 12:15:00 | 1446.40 | 1431.26 | 1429.59 | EMA200 above EMA400 |

### Cycle 60 — SELL (started 2026-02-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-18 11:15:00 | 1422.10 | 1429.71 | 1430.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-18 13:15:00 | 1406.00 | 1423.59 | 1427.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-19 11:15:00 | 1418.50 | 1416.70 | 1421.59 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-19 11:15:00 | 1418.50 | 1416.70 | 1421.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 11:15:00 | 1418.50 | 1416.70 | 1421.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-19 11:45:00 | 1424.10 | 1416.70 | 1421.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 12:15:00 | 1419.90 | 1417.34 | 1421.44 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-19 13:15:00 | 1408.20 | 1417.34 | 1421.44 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-25 09:15:00 | 1422.20 | 1406.23 | 1404.10 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 61 — BUY (started 2026-02-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-25 09:15:00 | 1422.20 | 1406.23 | 1404.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-25 15:15:00 | 1429.50 | 1419.62 | 1412.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-26 09:15:00 | 1412.00 | 1418.10 | 1412.65 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-26 09:15:00 | 1412.00 | 1418.10 | 1412.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 09:15:00 | 1412.00 | 1418.10 | 1412.65 | EMA400 retest candle locked (from upside) |

### Cycle 62 — SELL (started 2026-02-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-26 12:15:00 | 1401.10 | 1409.74 | 1409.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-27 09:15:00 | 1393.70 | 1404.73 | 1407.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-27 12:15:00 | 1407.30 | 1401.56 | 1404.78 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-27 12:15:00 | 1407.30 | 1401.56 | 1404.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 12:15:00 | 1407.30 | 1401.56 | 1404.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-27 12:45:00 | 1403.70 | 1401.56 | 1404.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 13:15:00 | 1400.00 | 1401.25 | 1404.35 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-02 09:15:00 | 1390.30 | 1400.97 | 1403.68 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-02 10:15:00 | 1414.70 | 1404.52 | 1404.88 | SL hit (close>static) qty=1.00 sl=1408.00 alert=retest2 |

### Cycle 63 — BUY (started 2026-03-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-02 11:15:00 | 1409.40 | 1405.50 | 1405.29 | EMA200 above EMA400 |

### Cycle 64 — SELL (started 2026-03-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-04 09:15:00 | 1379.30 | 1401.50 | 1403.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-04 11:15:00 | 1361.50 | 1389.58 | 1397.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-06 09:15:00 | 1356.60 | 1345.47 | 1360.00 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-06 09:15:00 | 1356.60 | 1345.47 | 1360.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 09:15:00 | 1356.60 | 1345.47 | 1360.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-06 09:45:00 | 1355.10 | 1345.47 | 1360.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 10:15:00 | 1355.40 | 1347.46 | 1359.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-06 10:45:00 | 1357.90 | 1347.46 | 1359.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 14:15:00 | 1359.00 | 1352.44 | 1358.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-06 14:45:00 | 1360.00 | 1352.44 | 1358.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 15:15:00 | 1354.60 | 1352.87 | 1358.00 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-09 09:15:00 | 1332.10 | 1352.87 | 1358.00 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-09 15:15:00 | 1373.00 | 1357.73 | 1357.03 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 65 — BUY (started 2026-03-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-09 15:15:00 | 1373.00 | 1357.73 | 1357.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-10 09:15:00 | 1379.70 | 1362.12 | 1359.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-11 15:15:00 | 1395.80 | 1396.89 | 1385.14 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-11 15:15:00 | 1395.80 | 1396.89 | 1385.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 15:15:00 | 1395.80 | 1396.89 | 1385.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-12 09:15:00 | 1383.90 | 1396.89 | 1385.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 09:15:00 | 1364.70 | 1390.45 | 1383.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-12 10:00:00 | 1364.70 | 1390.45 | 1383.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 10:15:00 | 1376.60 | 1387.68 | 1382.67 | EMA400 retest candle locked (from upside) |

### Cycle 66 — SELL (started 2026-03-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-12 13:15:00 | 1370.00 | 1379.71 | 1379.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-13 09:15:00 | 1355.40 | 1374.02 | 1377.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-17 14:15:00 | 1336.50 | 1336.34 | 1344.74 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-17 15:00:00 | 1336.50 | 1336.34 | 1344.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 09:15:00 | 1339.40 | 1336.23 | 1343.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-18 09:45:00 | 1340.70 | 1336.23 | 1343.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 10:15:00 | 1340.10 | 1337.00 | 1342.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-18 10:30:00 | 1345.10 | 1337.00 | 1342.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 11:15:00 | 1340.30 | 1337.66 | 1342.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-18 11:45:00 | 1344.60 | 1337.66 | 1342.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 12:15:00 | 1343.80 | 1338.89 | 1342.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-18 12:45:00 | 1341.40 | 1338.89 | 1342.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 13:15:00 | 1348.30 | 1340.77 | 1343.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-18 14:00:00 | 1348.30 | 1340.77 | 1343.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 14:15:00 | 1347.00 | 1342.02 | 1343.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-18 15:15:00 | 1348.90 | 1342.02 | 1343.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 15:15:00 | 1348.90 | 1343.39 | 1344.09 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-19 09:15:00 | 1330.80 | 1343.39 | 1344.09 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-24 14:15:00 | 1327.10 | 1315.73 | 1314.84 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 67 — BUY (started 2026-03-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-24 14:15:00 | 1327.10 | 1315.73 | 1314.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 11:15:00 | 1342.60 | 1324.21 | 1319.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-27 09:15:00 | 1315.20 | 1332.44 | 1326.59 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-27 09:15:00 | 1315.20 | 1332.44 | 1326.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 09:15:00 | 1315.20 | 1332.44 | 1326.59 | EMA400 retest candle locked (from upside) |

### Cycle 68 — SELL (started 2026-03-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-30 09:15:00 | 1304.60 | 1321.02 | 1322.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-02 09:15:00 | 1291.00 | 1313.17 | 1316.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-02 12:15:00 | 1316.00 | 1310.32 | 1313.85 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-02 12:15:00 | 1316.00 | 1310.32 | 1313.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 12:15:00 | 1316.00 | 1310.32 | 1313.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-02 13:00:00 | 1316.00 | 1310.32 | 1313.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 13:15:00 | 1322.80 | 1312.81 | 1314.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-02 13:30:00 | 1326.00 | 1312.81 | 1314.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 69 — BUY (started 2026-04-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-02 14:15:00 | 1334.70 | 1317.19 | 1316.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-06 10:15:00 | 1340.00 | 1322.34 | 1319.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-07 13:15:00 | 1359.50 | 1360.07 | 1345.42 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-07 13:45:00 | 1358.90 | 1360.07 | 1345.42 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 1390.00 | 1399.37 | 1391.17 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 10:45:00 | 1401.10 | 1399.50 | 1391.97 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-13 15:15:00 | 1377.50 | 1389.59 | 1389.68 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 70 — SELL (started 2026-04-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-13 15:15:00 | 1377.50 | 1389.59 | 1389.68 | EMA200 below EMA400 |

### Cycle 71 — BUY (started 2026-04-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-15 09:15:00 | 1423.00 | 1396.28 | 1392.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-15 11:15:00 | 1435.00 | 1406.98 | 1398.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-16 09:15:00 | 1421.00 | 1423.97 | 1411.73 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-16 09:15:00 | 1421.00 | 1423.97 | 1411.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-16 09:15:00 | 1421.00 | 1423.97 | 1411.73 | EMA400 retest candle locked (from upside) |

### Cycle 72 — SELL (started 2026-04-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-17 15:15:00 | 1404.40 | 1409.78 | 1410.26 | EMA200 below EMA400 |

### Cycle 73 — BUY (started 2026-04-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-20 13:15:00 | 1419.70 | 1411.94 | 1411.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-21 10:15:00 | 1428.10 | 1418.36 | 1414.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-23 09:15:00 | 1449.00 | 1451.65 | 1440.78 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-23 09:15:00 | 1449.00 | 1451.65 | 1440.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-23 09:15:00 | 1449.00 | 1451.65 | 1440.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-23 09:30:00 | 1445.90 | 1451.65 | 1440.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-23 12:15:00 | 1443.10 | 1449.97 | 1442.80 | EMA400 retest candle locked (from upside) |

### Cycle 74 — SELL (started 2026-04-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-24 09:15:00 | 1406.30 | 1434.94 | 1437.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-24 11:15:00 | 1402.30 | 1424.60 | 1432.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-27 09:15:00 | 1420.00 | 1417.11 | 1424.89 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-27 09:15:00 | 1420.00 | 1417.11 | 1424.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 09:15:00 | 1420.00 | 1417.11 | 1424.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 09:45:00 | 1425.30 | 1417.11 | 1424.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 09:15:00 | 1411.20 | 1415.06 | 1419.70 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-29 10:45:00 | 1402.60 | 1411.00 | 1415.31 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-29 14:45:00 | 1400.90 | 1402.91 | 1409.59 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-05-04 09:15:00 | 1487.70 | 1401.87 | 1401.04 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-05-04 09:15:00 | 1487.70 | 1401.87 | 1401.04 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 75 — BUY (started 2026-05-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-04 09:15:00 | 1487.70 | 1401.87 | 1401.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-04 10:15:00 | 1629.50 | 1447.40 | 1421.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-05 12:15:00 | 1541.50 | 1543.48 | 1502.20 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-05-05 12:45:00 | 1541.70 | 1543.48 | 1502.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-07 15:15:00 | 1546.10 | 1573.29 | 1560.11 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-08 09:15:00 | 1611.40 | 1573.29 | 1560.11 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-05-15 15:15:00 | 1394.00 | 2025-05-16 14:15:00 | 1403.05 | STOP_HIT | 1.00 | -0.65% |
| SELL | retest2 | 2025-05-16 09:30:00 | 1391.05 | 2025-05-16 14:15:00 | 1403.05 | STOP_HIT | 1.00 | -0.86% |
| BUY | retest2 | 2025-05-23 09:30:00 | 1430.00 | 2025-05-28 09:15:00 | 1413.45 | STOP_HIT | 1.00 | -1.16% |
| BUY | retest2 | 2025-05-23 11:15:00 | 1424.90 | 2025-05-28 09:15:00 | 1413.45 | STOP_HIT | 1.00 | -0.80% |
| BUY | retest2 | 2025-05-23 12:15:00 | 1424.65 | 2025-05-28 09:15:00 | 1413.45 | STOP_HIT | 1.00 | -0.79% |
| BUY | retest2 | 2025-05-23 15:00:00 | 1427.30 | 2025-05-28 09:15:00 | 1413.45 | STOP_HIT | 1.00 | -0.97% |
| SELL | retest2 | 2025-05-30 14:45:00 | 1407.50 | 2025-06-03 12:15:00 | 1413.55 | STOP_HIT | 1.00 | -0.43% |
| SELL | retest2 | 2025-06-03 09:30:00 | 1400.30 | 2025-06-03 12:15:00 | 1413.55 | STOP_HIT | 1.00 | -0.95% |
| BUY | retest2 | 2025-06-09 09:15:00 | 1430.05 | 2025-06-16 09:15:00 | 1453.10 | STOP_HIT | 1.00 | 1.61% |
| BUY | retest2 | 2025-06-10 09:30:00 | 1425.90 | 2025-06-16 09:15:00 | 1453.10 | STOP_HIT | 1.00 | 1.91% |
| SELL | retest2 | 2025-06-23 09:15:00 | 1424.05 | 2025-06-30 09:15:00 | 1400.60 | STOP_HIT | 1.00 | 1.65% |
| BUY | retest2 | 2025-07-11 14:45:00 | 1464.50 | 2025-07-14 10:15:00 | 1441.75 | STOP_HIT | 1.00 | -1.55% |
| BUY | retest2 | 2025-08-05 09:15:00 | 1587.80 | 2025-08-19 09:15:00 | 1746.58 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-08-26 11:15:00 | 1682.00 | 2025-09-03 11:15:00 | 1664.50 | STOP_HIT | 1.00 | 1.04% |
| SELL | retest2 | 2025-08-29 10:00:00 | 1675.20 | 2025-09-03 11:15:00 | 1664.50 | STOP_HIT | 1.00 | 0.64% |
| SELL | retest1 | 2025-09-25 09:15:00 | 1641.25 | 2025-09-29 11:15:00 | 1559.19 | PARTIAL | 0.50 | 5.00% |
| SELL | retest1 | 2025-09-25 09:15:00 | 1641.25 | 2025-10-01 09:15:00 | 1570.40 | STOP_HIT | 0.50 | 4.32% |
| SELL | retest2 | 2025-10-03 10:45:00 | 1565.00 | 2025-10-06 14:15:00 | 1601.25 | STOP_HIT | 1.00 | -2.32% |
| SELL | retest2 | 2025-10-03 13:00:00 | 1564.40 | 2025-10-06 14:15:00 | 1601.25 | STOP_HIT | 1.00 | -2.36% |
| SELL | retest2 | 2025-10-03 15:00:00 | 1565.00 | 2025-10-06 14:15:00 | 1601.25 | STOP_HIT | 1.00 | -2.32% |
| SELL | retest2 | 2025-10-06 09:45:00 | 1565.50 | 2025-10-06 14:15:00 | 1601.25 | STOP_HIT | 1.00 | -2.28% |
| SELL | retest2 | 2025-10-08 09:15:00 | 1559.35 | 2025-10-13 09:15:00 | 1569.90 | STOP_HIT | 1.00 | -0.68% |
| BUY | retest2 | 2025-10-28 13:15:00 | 1560.00 | 2025-10-29 09:15:00 | 1534.65 | STOP_HIT | 1.00 | -1.62% |
| SELL | retest2 | 2025-11-10 13:45:00 | 1546.50 | 2025-11-13 10:15:00 | 1571.70 | STOP_HIT | 1.00 | -1.63% |
| BUY | retest2 | 2025-11-17 15:00:00 | 1580.00 | 2025-11-20 11:15:00 | 1565.05 | STOP_HIT | 1.00 | -0.95% |
| BUY | retest2 | 2025-11-18 10:15:00 | 1577.90 | 2025-11-20 11:15:00 | 1565.05 | STOP_HIT | 1.00 | -0.81% |
| BUY | retest2 | 2025-11-18 10:45:00 | 1578.85 | 2025-11-20 11:15:00 | 1565.05 | STOP_HIT | 1.00 | -0.87% |
| BUY | retest2 | 2025-11-18 11:15:00 | 1578.05 | 2025-11-20 11:15:00 | 1565.05 | STOP_HIT | 1.00 | -0.82% |
| SELL | retest2 | 2025-11-24 09:15:00 | 1562.20 | 2025-12-03 11:15:00 | 1484.09 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-24 11:15:00 | 1564.80 | 2025-12-03 11:15:00 | 1486.56 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-24 13:45:00 | 1566.60 | 2025-12-03 11:15:00 | 1488.27 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-25 15:15:00 | 1565.90 | 2025-12-03 11:15:00 | 1487.61 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-24 09:15:00 | 1562.20 | 2025-12-04 09:15:00 | 1505.80 | STOP_HIT | 0.50 | 3.61% |
| SELL | retest2 | 2025-11-24 11:15:00 | 1564.80 | 2025-12-04 09:15:00 | 1505.80 | STOP_HIT | 0.50 | 3.77% |
| SELL | retest2 | 2025-11-24 13:45:00 | 1566.60 | 2025-12-04 09:15:00 | 1505.80 | STOP_HIT | 0.50 | 3.88% |
| SELL | retest2 | 2025-11-25 15:15:00 | 1565.90 | 2025-12-04 09:15:00 | 1505.80 | STOP_HIT | 0.50 | 3.84% |
| SELL | retest2 | 2025-11-26 09:15:00 | 1555.50 | 2025-12-08 15:15:00 | 1477.72 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-26 11:30:00 | 1555.55 | 2025-12-08 15:15:00 | 1477.77 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-27 10:15:00 | 1555.70 | 2025-12-08 15:15:00 | 1477.91 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-27 13:00:00 | 1555.00 | 2025-12-08 15:15:00 | 1477.25 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-28 10:15:00 | 1531.15 | 2025-12-09 09:15:00 | 1454.59 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-26 09:15:00 | 1555.50 | 2025-12-11 13:15:00 | 1434.85 | STOP_HIT | 0.50 | 7.76% |
| SELL | retest2 | 2025-11-26 11:30:00 | 1555.55 | 2025-12-11 13:15:00 | 1434.85 | STOP_HIT | 0.50 | 7.76% |
| SELL | retest2 | 2025-11-27 10:15:00 | 1555.70 | 2025-12-11 13:15:00 | 1434.85 | STOP_HIT | 0.50 | 7.77% |
| SELL | retest2 | 2025-11-27 13:00:00 | 1555.00 | 2025-12-11 13:15:00 | 1434.85 | STOP_HIT | 0.50 | 7.73% |
| SELL | retest2 | 2025-11-28 10:15:00 | 1531.15 | 2025-12-11 13:15:00 | 1434.85 | STOP_HIT | 0.50 | 6.29% |
| SELL | retest2 | 2026-01-08 09:15:00 | 1457.10 | 2026-01-12 11:15:00 | 1384.24 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-08 10:00:00 | 1456.20 | 2026-01-12 12:15:00 | 1383.39 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-08 09:15:00 | 1457.10 | 2026-01-13 10:15:00 | 1405.50 | STOP_HIT | 0.50 | 3.54% |
| SELL | retest2 | 2026-01-08 10:00:00 | 1456.20 | 2026-01-13 10:15:00 | 1405.50 | STOP_HIT | 0.50 | 3.48% |
| SELL | retest2 | 2026-01-19 13:15:00 | 1380.10 | 2026-01-22 10:15:00 | 1398.50 | STOP_HIT | 1.00 | -1.33% |
| SELL | retest2 | 2026-01-19 14:00:00 | 1380.00 | 2026-01-22 10:15:00 | 1398.50 | STOP_HIT | 1.00 | -1.34% |
| SELL | retest2 | 2026-01-20 09:15:00 | 1366.00 | 2026-01-22 10:15:00 | 1398.50 | STOP_HIT | 1.00 | -2.38% |
| SELL | retest2 | 2026-01-20 15:00:00 | 1374.80 | 2026-01-22 10:15:00 | 1398.50 | STOP_HIT | 1.00 | -1.72% |
| BUY | retest2 | 2026-02-01 11:00:00 | 1413.50 | 2026-02-03 14:15:00 | 1405.10 | STOP_HIT | 1.00 | -0.59% |
| BUY | retest2 | 2026-02-02 13:30:00 | 1405.50 | 2026-02-03 14:15:00 | 1405.10 | STOP_HIT | 1.00 | -0.03% |
| BUY | retest2 | 2026-02-02 14:30:00 | 1406.90 | 2026-02-03 14:15:00 | 1405.10 | STOP_HIT | 1.00 | -0.13% |
| BUY | retest2 | 2026-02-03 10:00:00 | 1406.30 | 2026-02-03 14:15:00 | 1405.10 | STOP_HIT | 1.00 | -0.09% |
| SELL | retest2 | 2026-02-04 15:15:00 | 1398.50 | 2026-02-05 13:15:00 | 1414.20 | STOP_HIT | 1.00 | -1.12% |
| BUY | retest2 | 2026-02-10 09:45:00 | 1452.50 | 2026-02-12 14:15:00 | 1432.80 | STOP_HIT | 1.00 | -1.36% |
| BUY | retest2 | 2026-02-10 10:45:00 | 1458.70 | 2026-02-12 14:15:00 | 1432.80 | STOP_HIT | 1.00 | -1.78% |
| BUY | retest2 | 2026-02-10 14:45:00 | 1454.50 | 2026-02-12 14:15:00 | 1432.80 | STOP_HIT | 1.00 | -1.49% |
| BUY | retest2 | 2026-02-11 09:15:00 | 1462.10 | 2026-02-12 14:15:00 | 1432.80 | STOP_HIT | 1.00 | -2.00% |
| BUY | retest2 | 2026-02-12 10:15:00 | 1455.30 | 2026-02-12 14:15:00 | 1432.80 | STOP_HIT | 1.00 | -1.55% |
| SELL | retest2 | 2026-02-17 09:15:00 | 1411.60 | 2026-02-17 10:15:00 | 1436.00 | STOP_HIT | 1.00 | -1.73% |
| SELL | retest2 | 2026-02-19 13:15:00 | 1408.20 | 2026-02-25 09:15:00 | 1422.20 | STOP_HIT | 1.00 | -0.99% |
| SELL | retest2 | 2026-03-02 09:15:00 | 1390.30 | 2026-03-02 10:15:00 | 1414.70 | STOP_HIT | 1.00 | -1.76% |
| SELL | retest2 | 2026-03-09 09:15:00 | 1332.10 | 2026-03-09 15:15:00 | 1373.00 | STOP_HIT | 1.00 | -3.07% |
| SELL | retest2 | 2026-03-19 09:15:00 | 1330.80 | 2026-03-24 14:15:00 | 1327.10 | STOP_HIT | 1.00 | 0.28% |
| BUY | retest2 | 2026-04-13 10:45:00 | 1401.10 | 2026-04-13 15:15:00 | 1377.50 | STOP_HIT | 1.00 | -1.68% |
| SELL | retest2 | 2026-04-29 10:45:00 | 1402.60 | 2026-05-04 09:15:00 | 1487.70 | STOP_HIT | 1.00 | -6.07% |
| SELL | retest2 | 2026-04-29 14:45:00 | 1400.90 | 2026-05-04 09:15:00 | 1487.70 | STOP_HIT | 1.00 | -6.20% |
