# Astral Ltd. (ASTRAL)

## Backtest Summary

- **Window:** 2025-03-13 09:15:00 → 2026-05-08 15:15:00 (1976 bars)
- **Last close:** 1567.40
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 75 |
| ALERT1 | 53 |
| ALERT2 | 53 |
| ALERT2_SKIP | 26 |
| ALERT3 | 162 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 39 |
| PARTIAL | 3 |
| TARGET_HIT | 1 |
| STOP_HIT | 36 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 40 (incl. partial bookings)
- **Trades open at end:** 2
- **Winners / losers:** 28 / 12
- **Target hits / Stop hits / Partials:** 1 / 36 / 3
- **Avg / median % per leg:** 1.08% / 0.68%
- **Sum % (uncompounded):** 43.08%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 11 | 9 | 81.8% | 0 | 11 | 0 | 1.02% | 11.2% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 11 | 9 | 81.8% | 0 | 11 | 0 | 1.02% | 11.2% |
| SELL (all) | 29 | 19 | 65.5% | 1 | 25 | 3 | 1.10% | 31.9% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 29 | 19 | 65.5% | 1 | 25 | 3 | 1.10% | 31.9% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 40 | 28 | 70.0% | 1 | 36 | 3 | 1.08% | 43.1% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 10:15:00 | 1328.30 | 1294.47 | 1293.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-13 09:15:00 | 1345.10 | 1325.13 | 1311.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-14 09:15:00 | 1320.00 | 1336.56 | 1326.38 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-14 09:15:00 | 1320.00 | 1336.56 | 1326.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-14 09:15:00 | 1320.00 | 1336.56 | 1326.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-14 09:45:00 | 1314.50 | 1336.56 | 1326.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-14 10:15:00 | 1322.10 | 1333.67 | 1325.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-14 10:45:00 | 1312.90 | 1333.67 | 1325.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-14 11:15:00 | 1324.00 | 1331.73 | 1325.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-14 11:45:00 | 1320.20 | 1331.73 | 1325.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-14 12:15:00 | 1328.60 | 1331.11 | 1326.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-14 12:45:00 | 1319.90 | 1331.11 | 1326.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-14 13:15:00 | 1325.60 | 1330.00 | 1326.02 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-15 09:15:00 | 1341.10 | 1329.10 | 1326.26 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-21 09:15:00 | 1358.50 | 1374.03 | 1375.13 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 2 — SELL (started 2025-05-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-21 09:15:00 | 1358.50 | 1374.03 | 1375.13 | EMA200 below EMA400 |

### Cycle 3 — BUY (started 2025-05-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-22 09:15:00 | 1426.90 | 1382.58 | 1377.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-22 11:15:00 | 1438.50 | 1399.46 | 1386.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-28 10:15:00 | 1511.80 | 1515.90 | 1496.20 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-28 11:00:00 | 1511.80 | 1515.90 | 1496.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 10:15:00 | 1503.70 | 1516.38 | 1506.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-29 11:00:00 | 1503.70 | 1516.38 | 1506.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 11:15:00 | 1501.00 | 1513.31 | 1506.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-29 11:45:00 | 1500.00 | 1513.31 | 1506.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 13:15:00 | 1503.20 | 1510.07 | 1505.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-29 13:45:00 | 1501.80 | 1510.07 | 1505.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 14:15:00 | 1500.10 | 1508.08 | 1505.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-29 15:00:00 | 1500.10 | 1508.08 | 1505.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 15:15:00 | 1503.20 | 1507.10 | 1505.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-30 09:15:00 | 1503.60 | 1507.10 | 1505.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 10:15:00 | 1513.80 | 1507.46 | 1505.54 | EMA400 retest candle locked (from upside) |

### Cycle 4 — SELL (started 2025-05-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-30 15:15:00 | 1500.10 | 1504.39 | 1504.91 | EMA200 below EMA400 |

### Cycle 5 — BUY (started 2025-06-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-02 15:15:00 | 1508.60 | 1504.03 | 1503.94 | EMA200 above EMA400 |

### Cycle 6 — SELL (started 2025-06-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-03 11:15:00 | 1492.50 | 1503.44 | 1503.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-03 12:15:00 | 1484.40 | 1499.64 | 1502.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-04 11:15:00 | 1494.90 | 1494.56 | 1498.01 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-04 12:00:00 | 1494.90 | 1494.56 | 1498.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 12:15:00 | 1504.10 | 1496.47 | 1498.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-04 13:00:00 | 1504.10 | 1496.47 | 1498.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 13:15:00 | 1510.00 | 1499.18 | 1499.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-04 13:45:00 | 1511.20 | 1499.18 | 1499.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 7 — BUY (started 2025-06-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-04 14:15:00 | 1521.20 | 1503.58 | 1501.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-05 09:15:00 | 1534.30 | 1513.23 | 1506.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-05 13:15:00 | 1514.40 | 1519.87 | 1512.46 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-05 13:15:00 | 1514.40 | 1519.87 | 1512.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 13:15:00 | 1514.40 | 1519.87 | 1512.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-05 13:45:00 | 1514.60 | 1519.87 | 1512.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 14:15:00 | 1510.60 | 1518.02 | 1512.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-05 15:00:00 | 1510.60 | 1518.02 | 1512.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 15:15:00 | 1509.90 | 1516.40 | 1512.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-06 09:15:00 | 1508.60 | 1516.40 | 1512.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 09:15:00 | 1506.40 | 1514.40 | 1511.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-06 09:45:00 | 1508.70 | 1514.40 | 1511.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 10:15:00 | 1509.00 | 1513.32 | 1511.33 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-06 14:00:00 | 1518.00 | 1513.48 | 1511.82 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-12 10:15:00 | 1513.00 | 1523.93 | 1524.97 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 8 — SELL (started 2025-06-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-12 10:15:00 | 1513.00 | 1523.93 | 1524.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-12 13:15:00 | 1499.70 | 1516.63 | 1521.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-13 10:15:00 | 1508.10 | 1507.99 | 1514.91 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-13 10:45:00 | 1512.50 | 1507.99 | 1514.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 11:15:00 | 1510.10 | 1508.41 | 1514.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-13 11:30:00 | 1516.80 | 1508.41 | 1514.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 12:15:00 | 1513.50 | 1509.43 | 1514.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-13 12:45:00 | 1512.40 | 1509.43 | 1514.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 13:15:00 | 1524.10 | 1512.36 | 1515.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-13 14:00:00 | 1524.10 | 1512.36 | 1515.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 14:15:00 | 1529.50 | 1515.79 | 1516.56 | EMA400 retest candle locked (from downside) |

### Cycle 9 — BUY (started 2025-06-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-13 15:15:00 | 1528.80 | 1518.39 | 1517.67 | EMA200 above EMA400 |

### Cycle 10 — SELL (started 2025-06-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-16 09:15:00 | 1506.10 | 1515.93 | 1516.62 | EMA200 below EMA400 |

### Cycle 11 — BUY (started 2025-06-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-16 11:15:00 | 1538.70 | 1520.59 | 1518.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-16 12:15:00 | 1540.80 | 1524.64 | 1520.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-18 15:15:00 | 1560.00 | 1571.34 | 1561.08 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-18 15:15:00 | 1560.00 | 1571.34 | 1561.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 15:15:00 | 1560.00 | 1571.34 | 1561.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-19 09:15:00 | 1558.00 | 1571.34 | 1561.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 09:15:00 | 1533.00 | 1563.67 | 1558.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-19 10:00:00 | 1533.00 | 1563.67 | 1558.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 10:15:00 | 1543.00 | 1559.54 | 1557.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-19 10:30:00 | 1538.00 | 1559.54 | 1557.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 12 — SELL (started 2025-06-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-19 11:15:00 | 1538.00 | 1555.23 | 1555.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-19 12:15:00 | 1524.00 | 1548.98 | 1552.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-20 10:15:00 | 1547.70 | 1535.41 | 1543.08 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-20 10:15:00 | 1547.70 | 1535.41 | 1543.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 10:15:00 | 1547.70 | 1535.41 | 1543.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 11:00:00 | 1547.70 | 1535.41 | 1543.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 11:15:00 | 1552.60 | 1538.85 | 1543.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 11:45:00 | 1553.80 | 1538.85 | 1543.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 13 — BUY (started 2025-06-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-20 14:15:00 | 1561.00 | 1547.20 | 1546.82 | EMA200 above EMA400 |

### Cycle 14 — SELL (started 2025-06-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-23 09:15:00 | 1495.50 | 1537.31 | 1542.41 | EMA200 below EMA400 |

### Cycle 15 — BUY (started 2025-06-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-26 14:15:00 | 1523.00 | 1513.34 | 1512.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-27 09:15:00 | 1538.00 | 1519.34 | 1515.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-27 12:15:00 | 1515.50 | 1520.48 | 1517.34 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-27 12:15:00 | 1515.50 | 1520.48 | 1517.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 12:15:00 | 1515.50 | 1520.48 | 1517.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-27 12:45:00 | 1518.60 | 1520.48 | 1517.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 13:15:00 | 1512.00 | 1518.78 | 1516.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-27 14:00:00 | 1512.00 | 1518.78 | 1516.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 16 — SELL (started 2025-06-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-27 14:15:00 | 1500.70 | 1515.16 | 1515.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-30 09:15:00 | 1496.20 | 1508.94 | 1512.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-30 14:15:00 | 1509.00 | 1507.29 | 1510.02 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-30 14:15:00 | 1509.00 | 1507.29 | 1510.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 14:15:00 | 1509.00 | 1507.29 | 1510.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-30 14:30:00 | 1511.00 | 1507.29 | 1510.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 09:15:00 | 1496.60 | 1505.09 | 1508.54 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-01 11:00:00 | 1495.70 | 1503.21 | 1507.38 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-01 12:30:00 | 1494.50 | 1500.27 | 1505.24 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-02 12:00:00 | 1494.90 | 1494.77 | 1499.85 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-03 12:15:00 | 1493.00 | 1494.28 | 1496.98 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 15:15:00 | 1498.50 | 1494.52 | 1496.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-04 09:15:00 | 1503.40 | 1494.52 | 1496.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 09:15:00 | 1503.50 | 1496.32 | 1496.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-04 09:45:00 | 1506.00 | 1496.32 | 1496.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 10:15:00 | 1493.80 | 1495.81 | 1496.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-04 10:45:00 | 1496.10 | 1495.81 | 1496.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 14:15:00 | 1486.20 | 1484.74 | 1487.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-07 15:00:00 | 1486.20 | 1484.74 | 1487.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 15:15:00 | 1486.10 | 1485.01 | 1487.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-08 09:15:00 | 1484.80 | 1485.01 | 1487.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 09:15:00 | 1477.00 | 1483.41 | 1486.61 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-08 10:15:00 | 1475.30 | 1483.41 | 1486.61 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-08 14:15:00 | 1491.60 | 1483.46 | 1484.89 | SL hit (close>static) qty=1.00 sl=1488.70 alert=retest2 |

### Cycle 17 — BUY (started 2025-07-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-09 10:15:00 | 1493.70 | 1486.33 | 1485.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-10 09:15:00 | 1501.90 | 1491.59 | 1488.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-10 12:15:00 | 1493.10 | 1493.67 | 1490.71 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-10 13:00:00 | 1493.10 | 1493.67 | 1490.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 13:15:00 | 1489.00 | 1492.73 | 1490.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-10 14:00:00 | 1489.00 | 1492.73 | 1490.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 14:15:00 | 1493.00 | 1492.79 | 1490.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-10 15:00:00 | 1493.00 | 1492.79 | 1490.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 15:15:00 | 1492.90 | 1492.81 | 1490.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-11 09:15:00 | 1491.40 | 1492.81 | 1490.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 09:15:00 | 1481.00 | 1490.45 | 1490.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-11 10:00:00 | 1481.00 | 1490.45 | 1490.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 18 — SELL (started 2025-07-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-11 10:15:00 | 1475.20 | 1487.40 | 1488.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-14 11:15:00 | 1469.40 | 1476.07 | 1481.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-14 12:15:00 | 1478.00 | 1476.46 | 1480.80 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-14 13:00:00 | 1478.00 | 1476.46 | 1480.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 13:15:00 | 1493.40 | 1479.85 | 1481.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-14 14:00:00 | 1493.40 | 1479.85 | 1481.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 14:15:00 | 1496.50 | 1483.18 | 1483.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-14 15:00:00 | 1496.50 | 1483.18 | 1483.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 19 — BUY (started 2025-07-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-14 15:15:00 | 1494.70 | 1485.48 | 1484.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-15 13:15:00 | 1498.40 | 1492.40 | 1488.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-16 12:15:00 | 1495.20 | 1497.21 | 1493.28 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-16 12:45:00 | 1494.50 | 1497.21 | 1493.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 13:15:00 | 1496.80 | 1497.13 | 1493.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-16 13:45:00 | 1492.20 | 1497.13 | 1493.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 14:15:00 | 1493.20 | 1496.35 | 1493.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-16 14:45:00 | 1493.30 | 1496.35 | 1493.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 15:15:00 | 1490.00 | 1495.08 | 1493.24 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-17 09:15:00 | 1505.00 | 1495.08 | 1493.24 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-21 14:15:00 | 1500.10 | 1503.96 | 1504.14 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 20 — SELL (started 2025-07-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-21 14:15:00 | 1500.10 | 1503.96 | 1504.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-22 09:15:00 | 1496.20 | 1502.21 | 1503.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-23 14:15:00 | 1478.00 | 1467.71 | 1478.94 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-23 14:15:00 | 1478.00 | 1467.71 | 1478.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 14:15:00 | 1478.00 | 1467.71 | 1478.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-23 15:00:00 | 1478.00 | 1467.71 | 1478.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 15:15:00 | 1475.40 | 1469.25 | 1478.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-24 09:15:00 | 1459.20 | 1469.25 | 1478.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 09:15:00 | 1467.70 | 1468.94 | 1477.63 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-25 11:15:00 | 1455.00 | 1465.44 | 1471.22 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-25 12:15:00 | 1452.30 | 1463.90 | 1469.99 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-04 14:15:00 | 1428.40 | 1416.13 | 1415.20 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 21 — BUY (started 2025-08-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-04 14:15:00 | 1428.40 | 1416.13 | 1415.20 | EMA200 above EMA400 |

### Cycle 22 — SELL (started 2025-08-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-07 09:15:00 | 1412.90 | 1417.43 | 1417.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-07 10:15:00 | 1403.90 | 1414.73 | 1416.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-07 14:15:00 | 1416.50 | 1409.89 | 1413.13 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-07 14:15:00 | 1416.50 | 1409.89 | 1413.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 14:15:00 | 1416.50 | 1409.89 | 1413.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-07 15:00:00 | 1416.50 | 1409.89 | 1413.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 15:15:00 | 1414.00 | 1410.71 | 1413.21 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-08 09:15:00 | 1408.00 | 1410.71 | 1413.21 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-12 09:15:00 | 1337.60 | 1362.11 | 1379.84 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-08-12 14:15:00 | 1267.20 | 1303.95 | 1340.59 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 23 — BUY (started 2025-08-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-18 12:15:00 | 1320.30 | 1304.57 | 1304.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-19 09:15:00 | 1326.20 | 1312.89 | 1308.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-22 15:15:00 | 1422.00 | 1424.08 | 1408.02 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-25 09:15:00 | 1417.30 | 1424.08 | 1408.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 15:15:00 | 1414.70 | 1419.51 | 1413.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-26 09:15:00 | 1404.60 | 1419.51 | 1413.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 09:15:00 | 1397.10 | 1415.03 | 1411.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-26 09:45:00 | 1397.00 | 1415.03 | 1411.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 10:15:00 | 1390.60 | 1410.15 | 1409.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-26 10:45:00 | 1390.00 | 1410.15 | 1409.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 24 — SELL (started 2025-08-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-26 11:15:00 | 1393.60 | 1406.84 | 1408.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-28 09:15:00 | 1370.10 | 1392.20 | 1400.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-29 11:15:00 | 1366.60 | 1366.20 | 1378.49 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-29 11:45:00 | 1365.20 | 1366.20 | 1378.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 09:15:00 | 1385.30 | 1367.02 | 1373.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 09:45:00 | 1383.90 | 1367.02 | 1373.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 10:15:00 | 1386.50 | 1370.92 | 1374.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 11:00:00 | 1386.50 | 1370.92 | 1374.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 25 — BUY (started 2025-09-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-01 12:15:00 | 1394.00 | 1378.99 | 1378.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-01 14:15:00 | 1407.40 | 1387.47 | 1382.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-03 15:15:00 | 1421.80 | 1423.01 | 1413.60 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-04 09:15:00 | 1413.80 | 1423.01 | 1413.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 09:15:00 | 1411.00 | 1420.61 | 1413.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 10:00:00 | 1411.00 | 1420.61 | 1413.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 10:15:00 | 1410.30 | 1418.55 | 1413.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 10:30:00 | 1409.50 | 1418.55 | 1413.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 11:15:00 | 1411.80 | 1417.20 | 1412.97 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-04 13:00:00 | 1414.00 | 1416.56 | 1413.06 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-10 14:15:00 | 1438.60 | 1446.78 | 1447.43 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 26 — SELL (started 2025-09-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-10 14:15:00 | 1438.60 | 1446.78 | 1447.43 | EMA200 below EMA400 |

### Cycle 27 — BUY (started 2025-09-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-11 10:15:00 | 1456.70 | 1448.44 | 1447.93 | EMA200 above EMA400 |

### Cycle 28 — SELL (started 2025-09-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-11 14:15:00 | 1440.80 | 1446.40 | 1447.14 | EMA200 below EMA400 |

### Cycle 29 — BUY (started 2025-09-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-12 09:15:00 | 1466.80 | 1449.46 | 1448.34 | EMA200 above EMA400 |

### Cycle 30 — SELL (started 2025-09-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-17 15:15:00 | 1457.60 | 1462.70 | 1462.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-18 09:15:00 | 1456.00 | 1461.36 | 1462.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-18 14:15:00 | 1457.80 | 1457.22 | 1459.55 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-18 15:00:00 | 1457.80 | 1457.22 | 1459.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 15:15:00 | 1462.70 | 1458.32 | 1459.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-19 09:15:00 | 1466.60 | 1458.32 | 1459.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 09:15:00 | 1458.60 | 1458.38 | 1459.73 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-19 10:30:00 | 1456.90 | 1457.74 | 1459.32 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-26 14:15:00 | 1384.06 | 1395.60 | 1409.73 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-30 14:15:00 | 1366.10 | 1364.57 | 1377.25 | SL hit (close>ema200) qty=0.50 sl=1364.57 alert=retest2 |

### Cycle 31 — BUY (started 2025-10-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-03 15:15:00 | 1385.10 | 1374.87 | 1373.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-06 15:15:00 | 1389.00 | 1381.15 | 1377.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-08 10:15:00 | 1404.60 | 1406.86 | 1396.47 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-08 11:00:00 | 1404.60 | 1406.86 | 1396.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 12:15:00 | 1406.20 | 1406.40 | 1398.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-08 12:30:00 | 1401.80 | 1406.40 | 1398.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 15:15:00 | 1400.00 | 1404.21 | 1399.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-09 10:00:00 | 1399.10 | 1403.19 | 1399.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 10:15:00 | 1398.70 | 1402.29 | 1399.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-09 10:45:00 | 1398.90 | 1402.29 | 1399.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 11:15:00 | 1399.80 | 1401.79 | 1399.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-09 11:45:00 | 1395.70 | 1401.79 | 1399.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 12:15:00 | 1400.90 | 1401.61 | 1399.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-09 12:30:00 | 1397.40 | 1401.61 | 1399.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 15:15:00 | 1398.00 | 1402.20 | 1400.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-10 09:15:00 | 1393.30 | 1402.20 | 1400.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 09:15:00 | 1411.90 | 1404.14 | 1401.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-10 09:30:00 | 1394.80 | 1404.14 | 1401.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 09:15:00 | 1408.40 | 1417.67 | 1411.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-13 10:00:00 | 1408.40 | 1417.67 | 1411.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 10:15:00 | 1407.60 | 1415.66 | 1411.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-13 11:15:00 | 1404.30 | 1415.66 | 1411.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 11:15:00 | 1408.60 | 1414.25 | 1410.88 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-13 12:45:00 | 1412.20 | 1414.16 | 1411.14 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-14 13:15:00 | 1412.30 | 1413.75 | 1412.98 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-24 14:15:00 | 1431.30 | 1444.55 | 1446.26 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 32 — SELL (started 2025-10-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-24 14:15:00 | 1431.30 | 1444.55 | 1446.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-27 13:15:00 | 1415.00 | 1430.65 | 1437.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-27 14:15:00 | 1435.70 | 1431.66 | 1437.58 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-27 15:00:00 | 1435.70 | 1431.66 | 1437.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 15:15:00 | 1440.00 | 1433.33 | 1437.80 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-28 09:15:00 | 1432.80 | 1433.33 | 1437.80 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-28 09:15:00 | 1447.00 | 1436.06 | 1438.64 | SL hit (close>static) qty=1.00 sl=1440.60 alert=retest2 |

### Cycle 33 — BUY (started 2025-10-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-28 11:15:00 | 1444.40 | 1440.44 | 1440.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-28 15:15:00 | 1454.90 | 1446.47 | 1443.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-30 09:15:00 | 1454.50 | 1457.84 | 1452.31 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-30 10:00:00 | 1454.50 | 1457.84 | 1452.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 12:15:00 | 1467.00 | 1459.60 | 1454.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-30 12:30:00 | 1454.50 | 1459.60 | 1454.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 09:15:00 | 1462.80 | 1462.08 | 1457.46 | EMA400 retest candle locked (from upside) |

### Cycle 34 — SELL (started 2025-10-31 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-31 15:15:00 | 1447.60 | 1455.05 | 1455.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-03 09:15:00 | 1442.60 | 1452.56 | 1454.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-03 10:15:00 | 1461.60 | 1454.37 | 1455.10 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-03 10:15:00 | 1461.60 | 1454.37 | 1455.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 10:15:00 | 1461.60 | 1454.37 | 1455.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-03 10:45:00 | 1465.00 | 1454.37 | 1455.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 11:15:00 | 1458.40 | 1455.17 | 1455.40 | EMA400 retest candle locked (from downside) |

### Cycle 35 — BUY (started 2025-11-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-03 12:15:00 | 1465.00 | 1457.14 | 1456.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-03 13:15:00 | 1480.10 | 1461.73 | 1458.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-04 13:15:00 | 1467.00 | 1469.55 | 1465.36 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-04 13:15:00 | 1467.00 | 1469.55 | 1465.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 13:15:00 | 1467.00 | 1469.55 | 1465.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-04 14:00:00 | 1467.00 | 1469.55 | 1465.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 14:15:00 | 1467.50 | 1469.14 | 1465.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-04 14:45:00 | 1464.50 | 1469.14 | 1465.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 15:15:00 | 1466.00 | 1468.51 | 1465.60 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-06 09:15:00 | 1523.00 | 1468.51 | 1465.60 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-13 13:15:00 | 1552.10 | 1562.81 | 1563.38 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 36 — SELL (started 2025-11-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-13 13:15:00 | 1552.10 | 1562.81 | 1563.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-14 09:15:00 | 1520.10 | 1551.42 | 1557.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-19 14:15:00 | 1448.50 | 1447.19 | 1462.77 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-19 15:00:00 | 1448.50 | 1447.19 | 1462.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 11:15:00 | 1460.90 | 1452.41 | 1460.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-20 11:45:00 | 1462.10 | 1452.41 | 1460.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 12:15:00 | 1461.70 | 1454.27 | 1460.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-20 12:30:00 | 1462.00 | 1454.27 | 1460.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 13:15:00 | 1461.70 | 1455.75 | 1460.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-20 13:45:00 | 1461.10 | 1455.75 | 1460.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 14:15:00 | 1463.40 | 1457.28 | 1460.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-20 15:00:00 | 1463.40 | 1457.28 | 1460.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 15:15:00 | 1463.00 | 1458.43 | 1461.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-21 09:15:00 | 1464.70 | 1458.43 | 1461.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 09:15:00 | 1457.30 | 1458.20 | 1460.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-21 09:30:00 | 1460.10 | 1458.20 | 1460.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 09:15:00 | 1450.40 | 1452.60 | 1456.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-24 09:45:00 | 1455.70 | 1452.60 | 1456.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 12:15:00 | 1449.70 | 1450.90 | 1454.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-24 12:30:00 | 1452.80 | 1450.90 | 1454.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 13:15:00 | 1450.90 | 1450.90 | 1454.09 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-24 14:15:00 | 1448.60 | 1450.90 | 1454.09 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-24 14:15:00 | 1474.30 | 1455.58 | 1455.93 | SL hit (close>static) qty=1.00 sl=1455.80 alert=retest2 |

### Cycle 37 — BUY (started 2025-11-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-24 15:15:00 | 1486.00 | 1461.66 | 1458.66 | EMA200 above EMA400 |

### Cycle 38 — SELL (started 2025-11-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-28 09:15:00 | 1446.30 | 1463.34 | 1464.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-28 14:15:00 | 1439.60 | 1454.36 | 1459.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-01 14:15:00 | 1439.90 | 1438.95 | 1447.31 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-01 15:00:00 | 1439.90 | 1438.95 | 1447.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 09:15:00 | 1433.70 | 1418.33 | 1424.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-04 10:00:00 | 1433.70 | 1418.33 | 1424.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 10:15:00 | 1430.60 | 1420.78 | 1425.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-04 10:30:00 | 1434.60 | 1420.78 | 1425.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 12:15:00 | 1438.80 | 1426.18 | 1427.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-04 13:00:00 | 1438.80 | 1426.18 | 1427.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 39 — BUY (started 2025-12-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-04 13:15:00 | 1444.50 | 1429.85 | 1428.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-05 11:15:00 | 1447.90 | 1439.40 | 1434.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-08 09:15:00 | 1433.00 | 1447.01 | 1441.08 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-08 09:15:00 | 1433.00 | 1447.01 | 1441.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 09:15:00 | 1433.00 | 1447.01 | 1441.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-08 10:00:00 | 1433.00 | 1447.01 | 1441.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 10:15:00 | 1433.70 | 1444.35 | 1440.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-08 11:00:00 | 1433.70 | 1444.35 | 1440.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 40 — SELL (started 2025-12-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-08 13:15:00 | 1428.00 | 1436.53 | 1437.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-10 10:15:00 | 1412.80 | 1427.20 | 1431.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-11 12:15:00 | 1404.30 | 1402.28 | 1412.63 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-11 12:45:00 | 1403.40 | 1402.28 | 1412.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 15:15:00 | 1411.60 | 1405.05 | 1411.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-12 09:15:00 | 1411.50 | 1405.05 | 1411.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 09:15:00 | 1412.50 | 1406.54 | 1411.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-12 09:45:00 | 1417.20 | 1406.54 | 1411.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 10:15:00 | 1413.00 | 1407.83 | 1411.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-12 10:45:00 | 1413.00 | 1407.83 | 1411.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 11:15:00 | 1413.50 | 1408.97 | 1411.80 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-12 12:30:00 | 1409.20 | 1409.05 | 1411.58 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-12 14:15:00 | 1417.60 | 1411.55 | 1412.33 | SL hit (close>static) qty=1.00 sl=1416.40 alert=retest2 |

### Cycle 41 — BUY (started 2025-12-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-15 10:15:00 | 1418.90 | 1412.92 | 1412.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-15 13:15:00 | 1425.50 | 1416.96 | 1414.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-17 09:15:00 | 1444.00 | 1451.20 | 1439.19 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-17 10:00:00 | 1444.00 | 1451.20 | 1439.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 10:15:00 | 1432.50 | 1447.46 | 1438.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-17 11:00:00 | 1432.50 | 1447.46 | 1438.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 11:15:00 | 1435.90 | 1445.15 | 1438.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-17 12:00:00 | 1435.90 | 1445.15 | 1438.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 13:15:00 | 1435.30 | 1442.64 | 1438.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-17 14:00:00 | 1435.30 | 1442.64 | 1438.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 14:15:00 | 1429.90 | 1440.09 | 1437.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-17 14:45:00 | 1431.30 | 1440.09 | 1437.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 42 — SELL (started 2025-12-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-18 09:15:00 | 1414.80 | 1432.62 | 1434.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-18 14:15:00 | 1407.20 | 1421.72 | 1428.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-19 13:15:00 | 1417.40 | 1415.22 | 1421.15 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-19 13:45:00 | 1416.70 | 1415.22 | 1421.15 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 14:15:00 | 1423.00 | 1416.78 | 1421.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 15:00:00 | 1423.00 | 1416.78 | 1421.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 15:15:00 | 1420.30 | 1417.48 | 1421.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-22 09:15:00 | 1429.80 | 1417.48 | 1421.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-22 09:15:00 | 1429.30 | 1419.85 | 1421.96 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-22 11:30:00 | 1423.00 | 1420.72 | 1422.04 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-30 10:15:00 | 1351.85 | 1368.46 | 1380.88 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-31 09:15:00 | 1379.50 | 1365.14 | 1372.84 | SL hit (close>ema200) qty=0.50 sl=1365.14 alert=retest2 |

### Cycle 43 — BUY (started 2025-12-31 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-31 14:15:00 | 1389.30 | 1378.46 | 1377.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-31 15:15:00 | 1391.50 | 1381.07 | 1378.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-06 13:15:00 | 1482.90 | 1485.78 | 1468.46 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-06 14:00:00 | 1482.90 | 1485.78 | 1468.46 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 09:15:00 | 1482.40 | 1494.90 | 1485.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-08 10:00:00 | 1482.40 | 1494.90 | 1485.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 10:15:00 | 1472.40 | 1490.40 | 1483.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-08 11:00:00 | 1472.40 | 1490.40 | 1483.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 11:15:00 | 1475.70 | 1487.46 | 1483.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-08 12:00:00 | 1475.70 | 1487.46 | 1483.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 13:15:00 | 1475.50 | 1482.96 | 1481.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-08 13:30:00 | 1465.80 | 1482.96 | 1481.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 44 — SELL (started 2026-01-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-08 15:15:00 | 1476.50 | 1480.67 | 1480.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-09 09:15:00 | 1466.60 | 1477.85 | 1479.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-12 15:15:00 | 1442.00 | 1440.68 | 1451.41 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-13 09:15:00 | 1442.00 | 1440.68 | 1451.41 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 09:15:00 | 1434.90 | 1439.52 | 1449.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-13 09:45:00 | 1435.30 | 1439.52 | 1449.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 09:15:00 | 1447.30 | 1437.62 | 1443.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-14 09:45:00 | 1449.20 | 1437.62 | 1443.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 10:15:00 | 1467.30 | 1443.55 | 1445.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-14 11:00:00 | 1467.30 | 1443.55 | 1445.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 45 — BUY (started 2026-01-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-14 12:15:00 | 1463.40 | 1449.55 | 1447.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-16 09:15:00 | 1491.00 | 1463.04 | 1455.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-16 14:15:00 | 1462.00 | 1468.16 | 1461.40 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-16 15:00:00 | 1462.00 | 1468.16 | 1461.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 15:15:00 | 1462.00 | 1466.93 | 1461.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-19 09:15:00 | 1449.00 | 1466.93 | 1461.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 09:15:00 | 1447.30 | 1463.00 | 1460.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-19 10:00:00 | 1447.30 | 1463.00 | 1460.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 10:15:00 | 1442.50 | 1458.90 | 1458.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-19 10:30:00 | 1440.70 | 1458.90 | 1458.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 46 — SELL (started 2026-01-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-19 11:15:00 | 1436.10 | 1454.34 | 1456.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-19 12:15:00 | 1427.80 | 1449.03 | 1453.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-22 09:15:00 | 1390.60 | 1370.67 | 1392.37 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-22 09:15:00 | 1390.60 | 1370.67 | 1392.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 09:15:00 | 1390.60 | 1370.67 | 1392.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 10:00:00 | 1390.60 | 1370.67 | 1392.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 11:15:00 | 1392.20 | 1376.60 | 1391.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 12:00:00 | 1392.20 | 1376.60 | 1391.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 12:15:00 | 1391.00 | 1379.48 | 1391.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 12:45:00 | 1392.00 | 1379.48 | 1391.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 13:15:00 | 1399.60 | 1383.50 | 1392.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 13:45:00 | 1402.50 | 1383.50 | 1392.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 14:15:00 | 1406.70 | 1388.14 | 1393.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 15:00:00 | 1406.70 | 1388.14 | 1393.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 47 — BUY (started 2026-01-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-23 09:15:00 | 1411.40 | 1397.09 | 1396.85 | EMA200 above EMA400 |

### Cycle 48 — SELL (started 2026-01-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-23 12:15:00 | 1389.40 | 1395.67 | 1396.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-23 13:15:00 | 1384.10 | 1393.36 | 1395.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-23 15:15:00 | 1399.50 | 1393.47 | 1394.88 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-23 15:15:00 | 1399.50 | 1393.47 | 1394.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 15:15:00 | 1399.50 | 1393.47 | 1394.88 | EMA400 retest candle locked (from downside) |

### Cycle 49 — BUY (started 2026-01-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-28 09:15:00 | 1428.80 | 1400.20 | 1396.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-28 10:15:00 | 1445.50 | 1409.26 | 1401.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-01 11:15:00 | 1465.90 | 1473.55 | 1459.78 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-01 11:15:00 | 1465.90 | 1473.55 | 1459.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 11:15:00 | 1465.90 | 1473.55 | 1459.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 12:00:00 | 1465.90 | 1473.55 | 1459.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 12:15:00 | 1467.00 | 1472.24 | 1460.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 12:30:00 | 1450.00 | 1472.24 | 1460.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 14:15:00 | 1460.50 | 1468.86 | 1460.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 14:45:00 | 1460.00 | 1468.86 | 1460.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 15:15:00 | 1461.70 | 1467.43 | 1460.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-02 09:15:00 | 1465.10 | 1467.43 | 1460.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 09:15:00 | 1454.00 | 1464.74 | 1460.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-02 09:45:00 | 1453.90 | 1464.74 | 1460.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 10:15:00 | 1428.70 | 1457.54 | 1457.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-02 10:45:00 | 1424.40 | 1457.54 | 1457.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 50 — SELL (started 2026-02-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-02 11:15:00 | 1438.60 | 1453.75 | 1455.74 | EMA200 below EMA400 |

### Cycle 51 — BUY (started 2026-02-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-02 15:15:00 | 1467.50 | 1455.87 | 1455.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-03 09:15:00 | 1486.70 | 1462.03 | 1458.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-04 12:15:00 | 1492.60 | 1494.81 | 1483.75 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-04 12:45:00 | 1488.60 | 1494.81 | 1483.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 09:15:00 | 1484.30 | 1498.43 | 1494.17 | EMA400 retest candle locked (from upside) |

### Cycle 52 — SELL (started 2026-02-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-06 13:15:00 | 1476.00 | 1488.86 | 1490.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-06 14:15:00 | 1470.80 | 1485.25 | 1488.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-09 10:15:00 | 1480.80 | 1480.28 | 1485.25 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-09 10:45:00 | 1480.00 | 1480.28 | 1485.25 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 11:15:00 | 1482.20 | 1480.67 | 1484.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-09 11:45:00 | 1482.50 | 1480.67 | 1484.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 12:15:00 | 1489.20 | 1482.37 | 1485.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-09 13:00:00 | 1489.20 | 1482.37 | 1485.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 13:15:00 | 1492.40 | 1484.38 | 1486.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-09 13:30:00 | 1496.80 | 1484.38 | 1486.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 53 — BUY (started 2026-02-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-10 09:15:00 | 1505.00 | 1490.24 | 1488.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-10 11:15:00 | 1526.20 | 1500.72 | 1493.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-12 15:15:00 | 1588.00 | 1589.63 | 1567.76 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-13 09:15:00 | 1567.00 | 1589.63 | 1567.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 09:15:00 | 1569.50 | 1585.61 | 1567.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-13 09:30:00 | 1569.80 | 1585.61 | 1567.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 10:15:00 | 1576.50 | 1583.78 | 1568.70 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-13 11:45:00 | 1588.20 | 1585.19 | 1570.71 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-13 14:15:00 | 1587.10 | 1586.66 | 1574.01 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-20 09:15:00 | 1617.40 | 1627.64 | 1628.19 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 54 — SELL (started 2026-02-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-20 09:15:00 | 1617.40 | 1627.64 | 1628.19 | EMA200 below EMA400 |

### Cycle 55 — BUY (started 2026-02-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-20 11:15:00 | 1641.00 | 1630.80 | 1629.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-23 09:15:00 | 1643.00 | 1635.12 | 1632.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-23 14:15:00 | 1634.10 | 1636.09 | 1634.05 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-23 14:15:00 | 1634.10 | 1636.09 | 1634.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 14:15:00 | 1634.10 | 1636.09 | 1634.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-23 14:45:00 | 1636.60 | 1636.09 | 1634.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 15:15:00 | 1634.00 | 1635.67 | 1634.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-24 09:15:00 | 1638.10 | 1635.67 | 1634.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 09:15:00 | 1635.40 | 1635.62 | 1634.17 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-24 11:00:00 | 1648.40 | 1638.17 | 1635.46 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-24 13:45:00 | 1647.30 | 1642.43 | 1638.30 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-02 09:15:00 | 1651.70 | 1668.40 | 1668.62 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 56 — SELL (started 2026-03-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-02 09:15:00 | 1651.70 | 1668.40 | 1668.62 | EMA200 below EMA400 |

### Cycle 57 — BUY (started 2026-03-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-06 09:15:00 | 1694.70 | 1658.64 | 1655.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-06 11:15:00 | 1698.50 | 1672.43 | 1662.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-06 15:15:00 | 1677.60 | 1681.77 | 1671.06 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-09 09:15:00 | 1634.80 | 1681.77 | 1671.06 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-09 09:15:00 | 1623.50 | 1670.11 | 1666.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-09 10:00:00 | 1623.50 | 1670.11 | 1666.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 58 — SELL (started 2026-03-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-09 10:15:00 | 1618.60 | 1659.81 | 1662.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-09 11:15:00 | 1608.30 | 1649.51 | 1657.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-10 09:15:00 | 1653.00 | 1629.41 | 1641.94 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-10 09:15:00 | 1653.00 | 1629.41 | 1641.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 09:15:00 | 1653.00 | 1629.41 | 1641.94 | EMA400 retest candle locked (from downside) |

### Cycle 59 — BUY (started 2026-03-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-10 14:15:00 | 1658.00 | 1647.87 | 1647.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-10 15:15:00 | 1665.00 | 1651.30 | 1649.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-12 09:15:00 | 1664.00 | 1698.35 | 1680.76 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-12 09:15:00 | 1664.00 | 1698.35 | 1680.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 09:15:00 | 1664.00 | 1698.35 | 1680.76 | EMA400 retest candle locked (from upside) |

### Cycle 60 — SELL (started 2026-03-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-13 10:15:00 | 1637.70 | 1673.55 | 1676.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-13 11:15:00 | 1633.70 | 1665.58 | 1672.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-17 09:15:00 | 1642.40 | 1622.96 | 1635.07 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-17 09:15:00 | 1642.40 | 1622.96 | 1635.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 09:15:00 | 1642.40 | 1622.96 | 1635.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 09:45:00 | 1640.80 | 1622.96 | 1635.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 10:15:00 | 1649.90 | 1628.34 | 1636.42 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-17 11:30:00 | 1640.60 | 1631.18 | 1636.97 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-17 14:15:00 | 1652.40 | 1641.59 | 1640.81 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 61 — BUY (started 2026-03-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-17 14:15:00 | 1652.40 | 1641.59 | 1640.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-18 11:15:00 | 1662.60 | 1650.75 | 1645.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 09:15:00 | 1639.30 | 1658.30 | 1652.71 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-19 09:15:00 | 1639.30 | 1658.30 | 1652.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 09:15:00 | 1639.30 | 1658.30 | 1652.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-19 09:45:00 | 1637.70 | 1658.30 | 1652.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 10:15:00 | 1628.80 | 1652.40 | 1650.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-19 11:00:00 | 1628.80 | 1652.40 | 1650.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 62 — SELL (started 2026-03-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 11:15:00 | 1630.10 | 1647.94 | 1648.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-20 15:15:00 | 1625.00 | 1635.46 | 1639.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 12:15:00 | 1588.00 | 1576.24 | 1592.81 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-24 12:15:00 | 1588.00 | 1576.24 | 1592.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 12:15:00 | 1588.00 | 1576.24 | 1592.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 12:30:00 | 1591.30 | 1576.24 | 1592.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 13:15:00 | 1597.00 | 1580.39 | 1593.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 13:45:00 | 1595.70 | 1580.39 | 1593.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 14:15:00 | 1593.20 | 1582.95 | 1593.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 14:30:00 | 1598.80 | 1582.95 | 1593.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 15:15:00 | 1592.80 | 1584.92 | 1593.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-25 09:15:00 | 1644.60 | 1584.92 | 1593.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 63 — BUY (started 2026-03-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 09:15:00 | 1656.10 | 1599.16 | 1598.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 10:15:00 | 1673.60 | 1614.05 | 1605.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-27 09:15:00 | 1630.60 | 1639.13 | 1625.22 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-27 09:15:00 | 1630.60 | 1639.13 | 1625.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 09:15:00 | 1630.60 | 1639.13 | 1625.22 | EMA400 retest candle locked (from upside) |

### Cycle 64 — SELL (started 2026-03-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-30 10:15:00 | 1608.80 | 1622.47 | 1623.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-01 13:15:00 | 1582.90 | 1604.68 | 1611.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-02 14:15:00 | 1550.40 | 1539.07 | 1565.30 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-02 15:00:00 | 1550.40 | 1539.07 | 1565.30 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 15:15:00 | 1570.00 | 1545.26 | 1565.73 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-06 09:15:00 | 1491.20 | 1545.26 | 1565.73 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-08 09:15:00 | 1568.90 | 1542.40 | 1541.45 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 65 — BUY (started 2026-04-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-08 09:15:00 | 1568.90 | 1542.40 | 1541.45 | EMA200 above EMA400 |

### Cycle 66 — SELL (started 2026-04-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-08 13:15:00 | 1516.20 | 1537.04 | 1539.61 | EMA200 below EMA400 |

### Cycle 67 — BUY (started 2026-04-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-09 11:15:00 | 1562.30 | 1542.16 | 1540.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-09 12:15:00 | 1563.00 | 1546.33 | 1542.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-15 13:15:00 | 1622.80 | 1626.44 | 1613.56 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-15 13:30:00 | 1621.80 | 1626.44 | 1613.56 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-16 09:15:00 | 1574.70 | 1615.53 | 1611.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-16 10:00:00 | 1574.70 | 1615.53 | 1611.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 68 — SELL (started 2026-04-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-16 10:15:00 | 1570.60 | 1606.54 | 1608.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-16 11:15:00 | 1556.80 | 1596.60 | 1603.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-17 10:15:00 | 1586.30 | 1580.96 | 1590.24 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-17 10:15:00 | 1586.30 | 1580.96 | 1590.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-17 10:15:00 | 1586.30 | 1580.96 | 1590.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-17 11:00:00 | 1586.30 | 1580.96 | 1590.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-17 14:15:00 | 1608.40 | 1586.31 | 1589.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-17 15:00:00 | 1608.40 | 1586.31 | 1589.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 69 — BUY (started 2026-04-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-17 15:15:00 | 1628.00 | 1594.65 | 1593.21 | EMA200 above EMA400 |

### Cycle 70 — SELL (started 2026-04-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-20 15:15:00 | 1590.90 | 1593.01 | 1593.21 | EMA200 below EMA400 |

### Cycle 71 — BUY (started 2026-04-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-21 09:15:00 | 1600.60 | 1594.53 | 1593.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-21 10:15:00 | 1604.90 | 1596.60 | 1594.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-21 11:15:00 | 1592.50 | 1595.78 | 1594.67 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-21 11:15:00 | 1592.50 | 1595.78 | 1594.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 11:15:00 | 1592.50 | 1595.78 | 1594.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-21 12:00:00 | 1592.50 | 1595.78 | 1594.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 72 — SELL (started 2026-04-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-21 12:15:00 | 1585.80 | 1593.78 | 1593.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-21 14:15:00 | 1581.00 | 1589.66 | 1591.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-22 10:15:00 | 1593.10 | 1589.25 | 1591.03 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-22 10:15:00 | 1593.10 | 1589.25 | 1591.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-22 10:15:00 | 1593.10 | 1589.25 | 1591.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-22 11:00:00 | 1593.10 | 1589.25 | 1591.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-22 11:15:00 | 1588.80 | 1589.16 | 1590.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-22 12:00:00 | 1588.80 | 1589.16 | 1590.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-22 12:15:00 | 1580.00 | 1587.33 | 1589.84 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-23 09:30:00 | 1576.20 | 1587.46 | 1589.40 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-23 10:00:00 | 1573.90 | 1587.46 | 1589.40 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-23 14:00:00 | 1576.70 | 1579.94 | 1584.60 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-23 14:30:00 | 1576.10 | 1579.09 | 1583.79 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 09:15:00 | 1581.50 | 1579.72 | 1583.27 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-24 10:30:00 | 1561.60 | 1576.45 | 1581.46 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-27 11:15:00 | 1566.00 | 1567.69 | 1573.20 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-27 13:00:00 | 1561.80 | 1566.40 | 1571.65 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-29 11:00:00 | 1564.40 | 1554.36 | 1556.57 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 12:15:00 | 1559.40 | 1555.94 | 1556.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-29 12:45:00 | 1559.00 | 1555.94 | 1556.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 13:15:00 | 1548.50 | 1554.45 | 1556.17 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-30 09:15:00 | 1531.80 | 1554.59 | 1555.92 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-05-04 09:15:00 | 1569.50 | 1541.44 | 1545.08 | SL hit (close>static) qty=1.00 sl=1559.60 alert=retest2 |

### Cycle 73 — BUY (started 2026-05-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-04 11:15:00 | 1553.70 | 1547.81 | 1547.58 | EMA200 above EMA400 |

### Cycle 74 — SELL (started 2026-05-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-05 12:15:00 | 1539.20 | 1547.30 | 1548.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-05-05 13:15:00 | 1534.10 | 1544.66 | 1547.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-05-06 09:15:00 | 1542.40 | 1540.28 | 1544.12 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-06 09:15:00 | 1542.40 | 1540.28 | 1544.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 09:15:00 | 1542.40 | 1540.28 | 1544.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-06 10:15:00 | 1547.60 | 1540.28 | 1544.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 10:15:00 | 1564.50 | 1545.13 | 1545.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-06 11:00:00 | 1564.50 | 1545.13 | 1545.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 75 — BUY (started 2026-05-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-06 11:15:00 | 1571.80 | 1550.46 | 1548.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-06 14:15:00 | 1574.20 | 1561.27 | 1554.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-07 14:15:00 | 1569.70 | 1575.98 | 1567.39 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-07 14:15:00 | 1569.70 | 1575.98 | 1567.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-07 14:15:00 | 1569.70 | 1575.98 | 1567.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-07 15:00:00 | 1569.70 | 1575.98 | 1567.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 09:15:00 | 1588.10 | 1578.76 | 1570.17 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-08 11:45:00 | 1590.90 | 1581.66 | 1573.03 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-08 12:15:00 | 1590.60 | 1581.66 | 1573.03 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-05-15 09:15:00 | 1341.10 | 2025-05-21 09:15:00 | 1358.50 | STOP_HIT | 1.00 | 1.30% |
| BUY | retest2 | 2025-06-06 14:00:00 | 1518.00 | 2025-06-12 10:15:00 | 1513.00 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest2 | 2025-07-01 11:00:00 | 1495.70 | 2025-07-08 14:15:00 | 1491.60 | STOP_HIT | 1.00 | 0.27% |
| SELL | retest2 | 2025-07-01 12:30:00 | 1494.50 | 2025-07-09 10:15:00 | 1493.70 | STOP_HIT | 1.00 | 0.05% |
| SELL | retest2 | 2025-07-02 12:00:00 | 1494.90 | 2025-07-09 10:15:00 | 1493.70 | STOP_HIT | 1.00 | 0.08% |
| SELL | retest2 | 2025-07-03 12:15:00 | 1493.00 | 2025-07-09 10:15:00 | 1493.70 | STOP_HIT | 1.00 | -0.05% |
| SELL | retest2 | 2025-07-08 10:15:00 | 1475.30 | 2025-07-09 10:15:00 | 1493.70 | STOP_HIT | 1.00 | -1.25% |
| BUY | retest2 | 2025-07-17 09:15:00 | 1505.00 | 2025-07-21 14:15:00 | 1500.10 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest2 | 2025-07-25 11:15:00 | 1455.00 | 2025-08-04 14:15:00 | 1428.40 | STOP_HIT | 1.00 | 1.83% |
| SELL | retest2 | 2025-07-25 12:15:00 | 1452.30 | 2025-08-04 14:15:00 | 1428.40 | STOP_HIT | 1.00 | 1.65% |
| SELL | retest2 | 2025-08-08 09:15:00 | 1408.00 | 2025-08-12 09:15:00 | 1337.60 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-08 09:15:00 | 1408.00 | 2025-08-12 14:15:00 | 1267.20 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2025-09-04 13:00:00 | 1414.00 | 2025-09-10 14:15:00 | 1438.60 | STOP_HIT | 1.00 | 1.74% |
| SELL | retest2 | 2025-09-19 10:30:00 | 1456.90 | 2025-09-26 14:15:00 | 1384.06 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-19 10:30:00 | 1456.90 | 2025-09-30 14:15:00 | 1366.10 | STOP_HIT | 0.50 | 6.23% |
| BUY | retest2 | 2025-10-13 12:45:00 | 1412.20 | 2025-10-24 14:15:00 | 1431.30 | STOP_HIT | 1.00 | 1.35% |
| BUY | retest2 | 2025-10-14 13:15:00 | 1412.30 | 2025-10-24 14:15:00 | 1431.30 | STOP_HIT | 1.00 | 1.35% |
| SELL | retest2 | 2025-10-28 09:15:00 | 1432.80 | 2025-10-28 09:15:00 | 1447.00 | STOP_HIT | 1.00 | -0.99% |
| BUY | retest2 | 2025-11-06 09:15:00 | 1523.00 | 2025-11-13 13:15:00 | 1552.10 | STOP_HIT | 1.00 | 1.91% |
| SELL | retest2 | 2025-11-24 14:15:00 | 1448.60 | 2025-11-24 14:15:00 | 1474.30 | STOP_HIT | 1.00 | -1.77% |
| SELL | retest2 | 2025-12-12 12:30:00 | 1409.20 | 2025-12-12 14:15:00 | 1417.60 | STOP_HIT | 1.00 | -0.60% |
| SELL | retest2 | 2025-12-12 14:30:00 | 1411.30 | 2025-12-15 10:15:00 | 1418.90 | STOP_HIT | 1.00 | -0.54% |
| SELL | retest2 | 2025-12-15 09:15:00 | 1407.30 | 2025-12-15 10:15:00 | 1418.90 | STOP_HIT | 1.00 | -0.82% |
| SELL | retest2 | 2025-12-22 11:30:00 | 1423.00 | 2025-12-30 10:15:00 | 1351.85 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-22 11:30:00 | 1423.00 | 2025-12-31 09:15:00 | 1379.50 | STOP_HIT | 0.50 | 3.06% |
| BUY | retest2 | 2026-02-13 11:45:00 | 1588.20 | 2026-02-20 09:15:00 | 1617.40 | STOP_HIT | 1.00 | 1.84% |
| BUY | retest2 | 2026-02-13 14:15:00 | 1587.10 | 2026-02-20 09:15:00 | 1617.40 | STOP_HIT | 1.00 | 1.91% |
| BUY | retest2 | 2026-02-24 11:00:00 | 1648.40 | 2026-03-02 09:15:00 | 1651.70 | STOP_HIT | 1.00 | 0.20% |
| BUY | retest2 | 2026-02-24 13:45:00 | 1647.30 | 2026-03-02 09:15:00 | 1651.70 | STOP_HIT | 1.00 | 0.27% |
| SELL | retest2 | 2026-03-17 11:30:00 | 1640.60 | 2026-03-17 14:15:00 | 1652.40 | STOP_HIT | 1.00 | -0.72% |
| SELL | retest2 | 2026-04-06 09:15:00 | 1491.20 | 2026-04-08 09:15:00 | 1568.90 | STOP_HIT | 1.00 | -5.21% |
| SELL | retest2 | 2026-04-23 09:30:00 | 1576.20 | 2026-05-04 09:15:00 | 1569.50 | STOP_HIT | 1.00 | 0.43% |
| SELL | retest2 | 2026-04-23 10:00:00 | 1573.90 | 2026-05-04 11:15:00 | 1553.70 | STOP_HIT | 1.00 | 1.28% |
| SELL | retest2 | 2026-04-23 14:00:00 | 1576.70 | 2026-05-04 11:15:00 | 1553.70 | STOP_HIT | 1.00 | 1.46% |
| SELL | retest2 | 2026-04-23 14:30:00 | 1576.10 | 2026-05-04 11:15:00 | 1553.70 | STOP_HIT | 1.00 | 1.42% |
| SELL | retest2 | 2026-04-24 10:30:00 | 1561.60 | 2026-05-04 11:15:00 | 1553.70 | STOP_HIT | 1.00 | 0.51% |
| SELL | retest2 | 2026-04-27 11:15:00 | 1566.00 | 2026-05-04 11:15:00 | 1553.70 | STOP_HIT | 1.00 | 0.79% |
| SELL | retest2 | 2026-04-27 13:00:00 | 1561.80 | 2026-05-04 11:15:00 | 1553.70 | STOP_HIT | 1.00 | 0.52% |
| SELL | retest2 | 2026-04-29 11:00:00 | 1564.40 | 2026-05-04 11:15:00 | 1553.70 | STOP_HIT | 1.00 | 0.68% |
| SELL | retest2 | 2026-04-30 09:15:00 | 1531.80 | 2026-05-04 11:15:00 | 1553.70 | STOP_HIT | 1.00 | -1.43% |
