# ADANIPORTS (ADANIPORTS)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 1760.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 10 |
| ALERT1 | 7 |
| ALERT2 | 6 |
| ALERT2_SKIP | 3 |
| ALERT3 | 25 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 1 |
| ENTRY2 | 23 |
| PARTIAL | 0 |
| TARGET_HIT | 4 |
| STOP_HIT | 20 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 24 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 4 / 20
- **Target hits / Stop hits / Partials:** 4 / 20 / 0
- **Avg / median % per leg:** -0.01% / -1.16%
- **Sum % (uncompounded):** -0.16%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 15 | 4 | 26.7% | 4 | 11 | 0 | 1.41% | 21.2% |
| BUY @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -5.32% | -5.3% |
| BUY @ 3rd Alert (retest2) | 14 | 4 | 28.6% | 4 | 10 | 0 | 1.90% | 26.5% |
| SELL (all) | 9 | 0 | 0.0% | 0 | 9 | 0 | -2.38% | -21.4% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 9 | 0 | 0.0% | 0 | 9 | 0 | -2.38% | -21.4% |
| retest1 (combined) | 1 | 0 | 0.0% | 0 | 1 | 0 | -5.32% | -5.3% |
| retest2 (combined) | 23 | 4 | 17.4% | 4 | 19 | 0 | 0.22% | 5.2% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-09-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-19 15:15:00 | 1410.90 | 1463.23 | 1463.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-07 09:15:00 | 1385.90 | 1454.80 | 1458.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-30 10:15:00 | 1411.90 | 1399.79 | 1423.04 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-30 11:00:00 | 1411.90 | 1399.79 | 1423.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-06 12:15:00 | 1142.10 | 1101.70 | 1139.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-06 13:00:00 | 1142.10 | 1101.70 | 1139.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-06 13:15:00 | 1147.25 | 1102.15 | 1139.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-06 14:00:00 | 1147.25 | 1102.15 | 1139.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-06 14:15:00 | 1136.80 | 1102.50 | 1139.53 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-06 15:15:00 | 1130.80 | 1102.50 | 1139.53 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-07 09:15:00 | 1151.00 | 1103.26 | 1139.55 | SL hit (close>static) qty=1.00 sl=1147.95 alert=retest2 |

### Cycle 2 — BUY (started 2025-04-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-03 15:15:00 | 1201.30 | 1155.00 | 1154.82 | EMA200 above EMA400 |

### Cycle 3 — SELL (started 2025-04-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-07 09:15:00 | 1085.85 | 1154.57 | 1154.61 | EMA200 below EMA400 |

### Cycle 4 — BUY (started 2025-04-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-16 09:15:00 | 1214.90 | 1154.39 | 1154.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-16 12:15:00 | 1219.80 | 1156.17 | 1155.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-16 09:15:00 | 1396.70 | 1397.66 | 1333.10 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-17 09:30:00 | 1411.80 | 1397.89 | 1335.44 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 12:15:00 | 1336.70 | 1394.02 | 1338.55 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-06-19 12:15:00 | 1336.70 | 1394.02 | 1338.55 | SL hit (close<ema400) qty=1.00 sl=1338.55 alert=retest1 |

### Cycle 5 — SELL (started 2025-08-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-14 15:15:00 | 1301.40 | 1376.66 | 1376.84 | EMA200 below EMA400 |

### Cycle 6 — BUY (started 2025-09-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-17 14:15:00 | 1407.20 | 1368.43 | 1368.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-18 09:15:00 | 1409.40 | 1369.21 | 1368.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-29 10:15:00 | 1388.00 | 1391.40 | 1381.47 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-29 10:45:00 | 1388.00 | 1391.40 | 1381.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 11:15:00 | 1380.00 | 1391.29 | 1381.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-29 11:30:00 | 1381.30 | 1391.29 | 1381.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 12:15:00 | 1387.40 | 1391.25 | 1381.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-29 12:45:00 | 1382.60 | 1391.25 | 1381.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 14:15:00 | 1382.40 | 1391.15 | 1381.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-29 15:00:00 | 1382.40 | 1391.15 | 1381.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 15:15:00 | 1392.00 | 1391.16 | 1381.59 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-30 09:30:00 | 1396.10 | 1391.24 | 1381.67 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-08 12:30:00 | 1392.20 | 1395.71 | 1385.80 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-09 09:15:00 | 1396.00 | 1395.67 | 1385.92 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-09 10:15:00 | 1393.00 | 1395.61 | 1385.94 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Target hit | 2025-11-28 13:15:00 | 1531.42 | 1472.55 | 1446.42 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 7 — SELL (started 2026-01-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-19 15:15:00 | 1403.00 | 1465.11 | 1465.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-20 10:15:00 | 1396.30 | 1463.83 | 1464.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-03 09:15:00 | 1502.40 | 1425.99 | 1441.97 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-03 09:15:00 | 1502.40 | 1425.99 | 1441.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 09:15:00 | 1502.40 | 1425.99 | 1441.97 | EMA400 retest candle locked (from downside) |

### Cycle 8 — BUY (started 2026-02-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-09 10:15:00 | 1562.90 | 1456.70 | 1456.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-09 11:15:00 | 1568.50 | 1457.81 | 1456.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-02 09:15:00 | 1481.00 | 1510.47 | 1490.33 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-02 09:15:00 | 1481.00 | 1510.47 | 1490.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 09:15:00 | 1481.00 | 1510.47 | 1490.33 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-05 14:45:00 | 1497.80 | 1500.63 | 1487.00 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-09 09:15:00 | 1429.60 | 1498.70 | 1486.62 | SL hit (close<static) qty=1.00 sl=1452.20 alert=retest2 |

### Cycle 9 — SELL (started 2026-03-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-13 11:15:00 | 1362.40 | 1476.01 | 1476.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 14:15:00 | 1352.40 | 1448.47 | 1461.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 09:15:00 | 1456.60 | 1403.04 | 1430.23 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-08 09:15:00 | 1456.60 | 1403.04 | 1430.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 09:15:00 | 1456.60 | 1403.04 | 1430.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-08 09:45:00 | 1473.30 | 1403.04 | 1430.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 10:15:00 | 1448.40 | 1406.80 | 1431.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-09 11:00:00 | 1448.40 | 1406.80 | 1431.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 10:15:00 | 1461.70 | 1413.60 | 1432.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-13 11:00:00 | 1461.70 | 1413.60 | 1432.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 10 — BUY (started 2026-04-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-20 15:15:00 | 1572.10 | 1449.50 | 1449.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-21 09:15:00 | 1598.10 | 1450.98 | 1449.90 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-03-06 15:15:00 | 1130.80 | 2025-03-07 09:15:00 | 1151.00 | STOP_HIT | 1.00 | -1.79% |
| SELL | retest2 | 2025-03-11 09:30:00 | 1133.70 | 2025-03-18 09:15:00 | 1152.80 | STOP_HIT | 1.00 | -1.68% |
| SELL | retest2 | 2025-03-11 10:15:00 | 1134.85 | 2025-03-18 09:15:00 | 1152.80 | STOP_HIT | 1.00 | -1.58% |
| SELL | retest2 | 2025-03-11 15:00:00 | 1136.10 | 2025-03-18 09:15:00 | 1152.80 | STOP_HIT | 1.00 | -1.47% |
| SELL | retest2 | 2025-03-12 11:15:00 | 1119.35 | 2025-03-18 09:15:00 | 1152.80 | STOP_HIT | 1.00 | -2.99% |
| SELL | retest2 | 2025-03-12 11:45:00 | 1113.30 | 2025-03-18 09:15:00 | 1152.80 | STOP_HIT | 1.00 | -3.55% |
| SELL | retest2 | 2025-03-13 09:15:00 | 1117.35 | 2025-03-18 09:15:00 | 1152.80 | STOP_HIT | 1.00 | -3.17% |
| SELL | retest2 | 2025-03-13 15:00:00 | 1119.55 | 2025-03-18 09:15:00 | 1152.80 | STOP_HIT | 1.00 | -2.97% |
| SELL | retest2 | 2025-03-17 12:30:00 | 1128.25 | 2025-03-18 09:15:00 | 1152.80 | STOP_HIT | 1.00 | -2.18% |
| BUY | retest1 | 2025-06-17 09:30:00 | 1411.80 | 2025-06-19 12:15:00 | 1336.70 | STOP_HIT | 1.00 | -5.32% |
| BUY | retest2 | 2025-06-20 10:15:00 | 1345.50 | 2025-08-07 11:15:00 | 1328.70 | STOP_HIT | 1.00 | -1.25% |
| BUY | retest2 | 2025-06-20 13:15:00 | 1343.40 | 2025-08-07 11:15:00 | 1328.70 | STOP_HIT | 1.00 | -1.09% |
| BUY | retest2 | 2025-06-20 14:45:00 | 1344.30 | 2025-08-07 11:15:00 | 1328.70 | STOP_HIT | 1.00 | -1.16% |
| BUY | retest2 | 2025-06-23 09:45:00 | 1344.00 | 2025-08-07 11:15:00 | 1328.70 | STOP_HIT | 1.00 | -1.14% |
| BUY | retest2 | 2025-09-30 09:30:00 | 1396.10 | 2025-11-28 13:15:00 | 1531.42 | TARGET_HIT | 1.00 | 9.69% |
| BUY | retest2 | 2025-10-08 12:30:00 | 1392.20 | 2025-11-28 13:15:00 | 1532.30 | TARGET_HIT | 1.00 | 10.06% |
| BUY | retest2 | 2025-10-09 09:15:00 | 1396.00 | 2025-12-01 09:15:00 | 1535.71 | TARGET_HIT | 1.00 | 10.01% |
| BUY | retest2 | 2025-10-09 10:15:00 | 1393.00 | 2025-12-01 09:15:00 | 1535.60 | TARGET_HIT | 1.00 | 10.24% |
| BUY | retest2 | 2025-12-31 13:15:00 | 1474.80 | 2026-01-07 11:15:00 | 1462.20 | STOP_HIT | 1.00 | -0.85% |
| BUY | retest2 | 2026-01-01 09:15:00 | 1473.90 | 2026-01-07 11:15:00 | 1462.20 | STOP_HIT | 1.00 | -0.79% |
| BUY | retest2 | 2026-01-06 14:45:00 | 1474.40 | 2026-01-07 11:15:00 | 1462.20 | STOP_HIT | 1.00 | -0.83% |
| BUY | retest2 | 2026-01-06 15:15:00 | 1476.00 | 2026-01-07 11:15:00 | 1462.20 | STOP_HIT | 1.00 | -0.93% |
| BUY | retest2 | 2026-01-08 09:30:00 | 1482.20 | 2026-01-08 12:15:00 | 1469.50 | STOP_HIT | 1.00 | -0.86% |
| BUY | retest2 | 2026-03-05 14:45:00 | 1497.80 | 2026-03-09 09:15:00 | 1429.60 | STOP_HIT | 1.00 | -4.55% |
