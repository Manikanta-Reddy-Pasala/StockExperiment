# Techno Electric & Engineering Company Ltd. (TECHNOE)

## Backtest Summary

- **Window:** 2024-03-13 09:15:00 → 2026-05-08 15:15:00 (3710 bars)
- **Last close:** 1268.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 127 |
| ALERT1 | 91 |
| ALERT2 | 92 |
| ALERT2_SKIP | 53 |
| ALERT3 | 242 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 13 |
| ENTRY2 | 113 |
| PARTIAL | 19 |
| TARGET_HIT | 10 |
| STOP_HIT | 114 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 143 (incl. partial bookings)
- **Trades open at end:** 2
- **Winners / losers:** 48 / 95
- **Target hits / Stop hits / Partials:** 10 / 114 / 19
- **Avg / median % per leg:** 0.03% / -1.62%
- **Sum % (uncompounded):** 4.16%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 67 | 15 | 22.4% | 8 | 59 | 0 | -0.26% | -17.5% |
| BUY @ 2nd Alert (retest1) | 12 | 0 | 0.0% | 0 | 12 | 0 | -2.27% | -27.3% |
| BUY @ 3rd Alert (retest2) | 55 | 15 | 27.3% | 8 | 47 | 0 | 0.18% | 9.8% |
| SELL (all) | 76 | 33 | 43.4% | 2 | 55 | 19 | 0.28% | 21.6% |
| SELL @ 2nd Alert (retest1) | 2 | 2 | 100.0% | 0 | 1 | 1 | 4.18% | 8.4% |
| SELL @ 3rd Alert (retest2) | 74 | 31 | 41.9% | 2 | 54 | 18 | 0.18% | 13.3% |
| retest1 (combined) | 14 | 2 | 14.3% | 0 | 13 | 1 | -1.35% | -18.9% |
| retest2 (combined) | 129 | 46 | 35.7% | 10 | 101 | 18 | 0.18% | 23.1% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-05-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-27 09:15:00 | 1077.35 | 1145.22 | 1145.60 | EMA200 below EMA400 |

### Cycle 2 — BUY (started 2024-05-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-30 09:15:00 | 1175.00 | 1094.27 | 1086.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-30 12:15:00 | 1194.00 | 1137.27 | 1110.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-04 09:15:00 | 1397.35 | 1444.79 | 1361.59 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-04 09:15:00 | 1397.35 | 1444.79 | 1361.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 09:15:00 | 1397.35 | 1444.79 | 1361.59 | EMA400 retest candle locked (from upside) |

### Cycle 3 — SELL (started 2024-06-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-05 10:15:00 | 1218.10 | 1319.90 | 1333.65 | EMA200 below EMA400 |

### Cycle 4 — BUY (started 2024-06-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-10 14:15:00 | 1270.00 | 1259.87 | 1259.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-11 11:15:00 | 1301.95 | 1272.76 | 1265.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-14 09:15:00 | 1440.85 | 1445.97 | 1404.52 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-14 09:15:00 | 1440.85 | 1445.97 | 1404.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-14 09:15:00 | 1440.85 | 1445.97 | 1404.52 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-18 09:30:00 | 1498.00 | 1456.21 | 1428.77 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-18 10:15:00 | 1500.00 | 1456.21 | 1428.77 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-19 15:15:00 | 1430.00 | 1450.44 | 1452.22 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 5 — SELL (started 2024-06-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-19 15:15:00 | 1430.00 | 1450.44 | 1452.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-21 12:15:00 | 1411.00 | 1432.57 | 1439.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-21 13:15:00 | 1440.00 | 1434.06 | 1439.75 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-21 13:15:00 | 1440.00 | 1434.06 | 1439.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-21 13:15:00 | 1440.00 | 1434.06 | 1439.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-21 13:45:00 | 1439.95 | 1434.06 | 1439.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-21 14:15:00 | 1440.80 | 1435.41 | 1439.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-21 15:00:00 | 1440.80 | 1435.41 | 1439.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-21 15:15:00 | 1440.00 | 1436.33 | 1439.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-24 09:15:00 | 1498.90 | 1436.33 | 1439.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 6 — BUY (started 2024-06-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-24 09:15:00 | 1513.85 | 1451.83 | 1446.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-25 09:15:00 | 1582.00 | 1510.74 | 1483.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-26 09:15:00 | 1540.95 | 1543.99 | 1518.35 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-26 10:00:00 | 1540.95 | 1543.99 | 1518.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-26 10:15:00 | 1538.00 | 1542.79 | 1520.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-26 11:00:00 | 1538.00 | 1542.79 | 1520.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-26 12:15:00 | 1528.00 | 1540.91 | 1523.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-26 12:45:00 | 1525.60 | 1540.91 | 1523.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-26 13:15:00 | 1565.00 | 1545.72 | 1527.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-26 13:30:00 | 1524.80 | 1545.72 | 1527.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 09:15:00 | 1519.00 | 1539.18 | 1528.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-27 09:30:00 | 1500.00 | 1539.18 | 1528.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 10:15:00 | 1487.00 | 1528.74 | 1524.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-27 11:00:00 | 1487.00 | 1528.74 | 1524.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 7 — SELL (started 2024-06-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-27 12:15:00 | 1511.00 | 1522.23 | 1522.35 | EMA200 below EMA400 |

### Cycle 8 — BUY (started 2024-06-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-28 09:15:00 | 1545.50 | 1523.66 | 1522.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-28 14:15:00 | 1550.00 | 1532.02 | 1527.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-01 10:15:00 | 1526.60 | 1535.33 | 1530.43 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-01 10:15:00 | 1526.60 | 1535.33 | 1530.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-01 10:15:00 | 1526.60 | 1535.33 | 1530.43 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-01 15:00:00 | 1552.00 | 1542.57 | 1535.70 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-02 12:15:00 | 1515.00 | 1535.64 | 1534.92 | SL hit (close<static) qty=1.00 sl=1520.05 alert=retest2 |

### Cycle 9 — SELL (started 2024-07-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-04 11:15:00 | 1525.00 | 1541.26 | 1541.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-04 13:15:00 | 1480.00 | 1526.57 | 1534.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-05 13:15:00 | 1485.00 | 1483.05 | 1503.57 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-07-05 14:00:00 | 1485.00 | 1483.05 | 1503.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-08 09:15:00 | 1499.00 | 1484.49 | 1498.93 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-08 11:15:00 | 1484.95 | 1487.58 | 1499.02 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-08 12:00:00 | 1484.05 | 1486.88 | 1497.66 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-08 13:00:00 | 1490.00 | 1487.50 | 1496.96 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-08 14:15:00 | 1485.00 | 1489.40 | 1496.97 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-08 14:15:00 | 1499.00 | 1491.32 | 1497.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-08 15:00:00 | 1499.00 | 1491.32 | 1497.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-08 15:15:00 | 1498.95 | 1492.85 | 1497.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-09 09:15:00 | 1507.90 | 1492.85 | 1497.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-09 09:15:00 | 1507.95 | 1495.87 | 1498.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-09 10:15:00 | 1524.80 | 1495.87 | 1498.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-07-09 10:15:00 | 1535.00 | 1503.69 | 1501.62 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 10 — BUY (started 2024-07-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-09 10:15:00 | 1535.00 | 1503.69 | 1501.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-09 11:15:00 | 1544.70 | 1511.89 | 1505.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-10 09:15:00 | 1517.95 | 1531.41 | 1519.72 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-10 09:15:00 | 1517.95 | 1531.41 | 1519.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-10 09:15:00 | 1517.95 | 1531.41 | 1519.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-10 10:00:00 | 1517.95 | 1531.41 | 1519.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-10 10:15:00 | 1497.00 | 1524.53 | 1517.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-10 10:30:00 | 1473.45 | 1524.53 | 1517.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-10 11:15:00 | 1519.00 | 1523.42 | 1517.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-10 11:30:00 | 1520.00 | 1523.42 | 1517.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 11 — SELL (started 2024-07-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-10 12:15:00 | 1473.45 | 1513.43 | 1513.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-11 12:15:00 | 1462.90 | 1485.92 | 1497.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-11 15:15:00 | 1490.00 | 1485.03 | 1493.95 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-07-12 09:15:00 | 1478.60 | 1485.03 | 1493.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-12 09:15:00 | 1449.00 | 1477.83 | 1489.87 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-12 14:00:00 | 1391.15 | 1448.34 | 1471.19 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-15 12:30:00 | 1400.00 | 1432.35 | 1452.93 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-16 09:15:00 | 1494.00 | 1462.87 | 1462.28 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 12 — BUY (started 2024-07-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-16 09:15:00 | 1494.00 | 1462.87 | 1462.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-18 11:15:00 | 1530.40 | 1506.75 | 1488.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-19 09:15:00 | 1495.00 | 1510.04 | 1498.19 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-19 09:15:00 | 1495.00 | 1510.04 | 1498.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-19 09:15:00 | 1495.00 | 1510.04 | 1498.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-19 10:15:00 | 1495.00 | 1510.04 | 1498.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-19 10:15:00 | 1484.00 | 1504.83 | 1496.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-19 11:15:00 | 1506.00 | 1504.83 | 1496.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-19 11:15:00 | 1489.50 | 1501.77 | 1496.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-19 11:30:00 | 1451.00 | 1501.77 | 1496.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-19 12:15:00 | 1498.00 | 1501.01 | 1496.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-19 12:45:00 | 1490.00 | 1501.01 | 1496.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-19 13:15:00 | 1495.00 | 1499.81 | 1496.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-19 13:45:00 | 1499.00 | 1499.81 | 1496.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-19 14:15:00 | 1479.40 | 1495.73 | 1494.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-19 15:00:00 | 1479.40 | 1495.73 | 1494.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-19 15:15:00 | 1488.90 | 1494.36 | 1494.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-22 09:15:00 | 1494.00 | 1494.36 | 1494.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 09:15:00 | 1525.00 | 1535.46 | 1521.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-23 09:45:00 | 1515.00 | 1535.46 | 1521.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 10:15:00 | 1559.90 | 1540.35 | 1525.19 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-24 09:15:00 | 1561.50 | 1541.24 | 1531.27 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-24 13:15:00 | 1569.00 | 1541.06 | 1534.19 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-24 13:45:00 | 1576.00 | 1550.04 | 1538.90 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Target hit | 2024-07-26 09:15:00 | 1717.65 | 1650.54 | 1612.75 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 13 — SELL (started 2024-07-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-30 13:15:00 | 1645.00 | 1645.24 | 1645.26 | EMA200 below EMA400 |

### Cycle 14 — BUY (started 2024-07-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-31 09:15:00 | 1648.90 | 1645.72 | 1645.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-31 12:15:00 | 1670.00 | 1653.89 | 1649.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-01 09:15:00 | 1664.00 | 1667.70 | 1658.54 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-01 09:15:00 | 1664.00 | 1667.70 | 1658.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-01 09:15:00 | 1664.00 | 1667.70 | 1658.54 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-02 12:45:00 | 1750.00 | 1708.59 | 1689.99 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-02 14:45:00 | 1724.00 | 1719.23 | 1698.45 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-05 10:15:00 | 1625.80 | 1690.92 | 1690.24 | SL hit (close<static) qty=1.00 sl=1640.55 alert=retest2 |

### Cycle 15 — SELL (started 2024-08-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-05 11:15:00 | 1624.95 | 1677.72 | 1684.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-06 10:15:00 | 1600.00 | 1634.44 | 1656.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-07 10:15:00 | 1623.95 | 1616.78 | 1635.24 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-07 10:15:00 | 1623.95 | 1616.78 | 1635.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 10:15:00 | 1623.95 | 1616.78 | 1635.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-07 10:30:00 | 1640.00 | 1616.78 | 1635.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 12:15:00 | 1620.00 | 1618.54 | 1632.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-07 12:45:00 | 1630.00 | 1618.54 | 1632.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 13:15:00 | 1617.85 | 1618.40 | 1631.53 | EMA400 retest candle locked (from downside) |

### Cycle 16 — BUY (started 2024-08-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-09 11:15:00 | 1665.00 | 1638.45 | 1634.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-12 09:15:00 | 1709.00 | 1660.10 | 1647.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-13 09:15:00 | 1680.25 | 1688.41 | 1672.46 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-13 09:15:00 | 1680.25 | 1688.41 | 1672.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-13 09:15:00 | 1680.25 | 1688.41 | 1672.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-13 10:00:00 | 1680.25 | 1688.41 | 1672.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-13 10:15:00 | 1656.20 | 1681.97 | 1670.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-13 10:45:00 | 1668.00 | 1681.97 | 1670.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-13 11:15:00 | 1655.00 | 1676.58 | 1669.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-13 11:30:00 | 1668.00 | 1676.58 | 1669.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 17 — SELL (started 2024-08-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-13 13:15:00 | 1628.20 | 1661.85 | 1663.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-13 14:15:00 | 1620.80 | 1653.64 | 1659.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-16 09:15:00 | 1647.25 | 1606.70 | 1625.15 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-16 09:15:00 | 1647.25 | 1606.70 | 1625.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-16 09:15:00 | 1647.25 | 1606.70 | 1625.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-16 09:30:00 | 1647.00 | 1606.70 | 1625.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-16 10:15:00 | 1647.25 | 1614.81 | 1627.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-16 10:30:00 | 1647.25 | 1614.81 | 1627.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 18 — BUY (started 2024-08-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-16 14:15:00 | 1647.25 | 1633.96 | 1633.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-19 09:15:00 | 1729.60 | 1655.22 | 1643.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-20 11:15:00 | 1708.15 | 1714.62 | 1690.53 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-20 12:00:00 | 1708.15 | 1714.62 | 1690.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-20 15:15:00 | 1700.00 | 1709.24 | 1695.42 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-21 09:15:00 | 1726.15 | 1709.24 | 1695.42 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-21 11:15:00 | 1676.10 | 1699.91 | 1694.31 | SL hit (close<static) qty=1.00 sl=1690.00 alert=retest2 |

### Cycle 19 — SELL (started 2024-08-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-22 09:15:00 | 1657.35 | 1687.50 | 1690.07 | EMA200 below EMA400 |

### Cycle 20 — BUY (started 2024-08-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-23 11:15:00 | 1700.80 | 1688.00 | 1687.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-26 09:15:00 | 1713.40 | 1695.93 | 1691.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-26 12:15:00 | 1697.30 | 1697.34 | 1693.55 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-26 12:15:00 | 1697.30 | 1697.34 | 1693.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-26 12:15:00 | 1697.30 | 1697.34 | 1693.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-26 13:00:00 | 1697.30 | 1697.34 | 1693.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-26 13:15:00 | 1700.00 | 1697.87 | 1694.14 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-26 14:30:00 | 1710.00 | 1699.68 | 1695.30 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-27 12:15:00 | 1704.45 | 1702.44 | 1698.54 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-27 13:15:00 | 1703.00 | 1701.95 | 1698.68 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-27 15:15:00 | 1710.00 | 1702.57 | 1699.58 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-27 15:15:00 | 1710.00 | 1704.06 | 1700.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-28 09:15:00 | 1715.40 | 1704.06 | 1700.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-28 09:15:00 | 1703.20 | 1703.89 | 1700.77 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-08-28 10:15:00 | 1674.50 | 1698.01 | 1698.38 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 21 — SELL (started 2024-08-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-28 10:15:00 | 1674.50 | 1698.01 | 1698.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-29 10:15:00 | 1610.00 | 1660.36 | 1677.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-30 10:15:00 | 1633.70 | 1630.27 | 1649.74 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-30 10:30:00 | 1635.05 | 1630.27 | 1649.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-30 11:15:00 | 1649.90 | 1634.20 | 1649.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-30 11:45:00 | 1645.30 | 1634.20 | 1649.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-30 12:15:00 | 1635.00 | 1634.36 | 1648.41 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-30 13:30:00 | 1621.90 | 1635.49 | 1647.65 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-30 14:15:00 | 1620.10 | 1635.49 | 1647.65 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-30 15:15:00 | 1630.00 | 1637.39 | 1647.41 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-02 12:15:00 | 1669.90 | 1638.37 | 1642.98 | SL hit (close>static) qty=1.00 sl=1650.00 alert=retest2 |

### Cycle 22 — BUY (started 2024-09-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-02 15:15:00 | 1655.00 | 1647.18 | 1646.33 | EMA200 above EMA400 |

### Cycle 23 — SELL (started 2024-09-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-03 15:15:00 | 1641.00 | 1647.32 | 1647.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-04 09:15:00 | 1616.00 | 1641.06 | 1644.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-05 15:15:00 | 1595.00 | 1593.63 | 1608.55 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-06 09:15:00 | 1600.00 | 1593.63 | 1608.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-09 09:15:00 | 1599.65 | 1580.68 | 1592.64 | EMA400 retest candle locked (from downside) |

### Cycle 24 — BUY (started 2024-09-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-09 14:15:00 | 1624.15 | 1599.38 | 1598.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-10 09:15:00 | 1645.25 | 1611.13 | 1603.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-10 11:15:00 | 1604.00 | 1612.10 | 1605.68 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-10 11:15:00 | 1604.00 | 1612.10 | 1605.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-10 11:15:00 | 1604.00 | 1612.10 | 1605.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-10 11:45:00 | 1603.60 | 1612.10 | 1605.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-10 12:15:00 | 1613.00 | 1612.28 | 1606.35 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-10 15:00:00 | 1640.65 | 1620.79 | 1611.42 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-11 09:45:00 | 1648.85 | 1626.21 | 1615.66 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-12 14:15:00 | 1602.05 | 1613.61 | 1615.13 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 25 — SELL (started 2024-09-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-12 14:15:00 | 1602.05 | 1613.61 | 1615.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-12 15:15:00 | 1600.00 | 1610.89 | 1613.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-13 13:15:00 | 1606.85 | 1595.08 | 1603.44 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-13 13:15:00 | 1606.85 | 1595.08 | 1603.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-13 13:15:00 | 1606.85 | 1595.08 | 1603.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-13 14:00:00 | 1606.85 | 1595.08 | 1603.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-13 14:15:00 | 1604.05 | 1596.87 | 1603.49 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-16 09:15:00 | 1589.10 | 1597.90 | 1603.36 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-16 11:45:00 | 1591.05 | 1596.59 | 1601.32 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-16 12:45:00 | 1596.20 | 1596.84 | 1601.00 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-16 14:15:00 | 1585.00 | 1597.43 | 1600.89 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-16 14:15:00 | 1582.75 | 1594.49 | 1599.24 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-17 11:00:00 | 1567.55 | 1583.61 | 1592.60 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-17 12:30:00 | 1562.05 | 1576.83 | 1587.76 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-18 12:15:00 | 1569.95 | 1587.19 | 1589.55 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-18 13:00:00 | 1569.45 | 1583.64 | 1587.72 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 09:15:00 | 1583.00 | 1581.48 | 1585.14 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-19 10:15:00 | 1568.65 | 1581.48 | 1585.14 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-19 12:15:00 | 1511.50 | 1560.56 | 1574.10 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-19 12:15:00 | 1516.39 | 1560.56 | 1574.10 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-19 14:15:00 | 1509.64 | 1547.87 | 1565.61 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-19 14:15:00 | 1505.75 | 1547.87 | 1565.61 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-09-20 11:15:00 | 1541.30 | 1540.66 | 1555.88 | SL hit (close>ema200) qty=0.50 sl=1540.66 alert=retest2 |

### Cycle 26 — BUY (started 2024-09-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-24 10:15:00 | 1602.40 | 1553.61 | 1550.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-25 10:15:00 | 1614.85 | 1584.41 | 1568.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-25 13:15:00 | 1589.95 | 1590.83 | 1575.96 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-25 13:30:00 | 1591.20 | 1590.83 | 1575.96 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-26 09:15:00 | 1588.05 | 1593.50 | 1581.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-26 10:00:00 | 1588.05 | 1593.50 | 1581.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-26 10:15:00 | 1611.55 | 1597.11 | 1583.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-26 10:30:00 | 1576.50 | 1597.11 | 1583.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-26 11:15:00 | 1598.75 | 1597.44 | 1585.26 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-27 13:30:00 | 1616.15 | 1601.61 | 1593.69 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-30 09:15:00 | 1615.30 | 1608.80 | 1598.61 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-30 09:45:00 | 1616.95 | 1609.66 | 1599.93 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-01 12:15:00 | 1584.00 | 1597.19 | 1598.90 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 27 — SELL (started 2024-10-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-01 12:15:00 | 1584.00 | 1597.19 | 1598.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-01 13:15:00 | 1577.70 | 1593.29 | 1596.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-03 14:15:00 | 1560.75 | 1560.40 | 1574.88 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-03 15:00:00 | 1560.75 | 1560.40 | 1574.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-07 09:15:00 | 1650.00 | 1564.96 | 1566.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-07 09:30:00 | 1635.50 | 1564.96 | 1566.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 28 — BUY (started 2024-10-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-07 10:15:00 | 1583.55 | 1568.68 | 1567.68 | EMA200 above EMA400 |

### Cycle 29 — SELL (started 2024-10-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-07 12:15:00 | 1526.00 | 1560.76 | 1564.30 | EMA200 below EMA400 |

### Cycle 30 — BUY (started 2024-10-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-08 10:15:00 | 1587.85 | 1565.73 | 1564.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-08 13:15:00 | 1591.00 | 1575.53 | 1569.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-11 14:15:00 | 1717.25 | 1722.91 | 1697.89 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-11 15:00:00 | 1717.25 | 1722.91 | 1697.89 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-15 12:15:00 | 1747.65 | 1763.18 | 1745.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-15 13:00:00 | 1747.65 | 1763.18 | 1745.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-15 13:15:00 | 1772.75 | 1765.09 | 1748.12 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-16 09:30:00 | 1785.00 | 1765.52 | 1752.46 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-17 09:15:00 | 1734.90 | 1757.41 | 1755.34 | SL hit (close<static) qty=1.00 sl=1746.85 alert=retest2 |

### Cycle 31 — SELL (started 2024-10-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-18 09:15:00 | 1722.40 | 1754.85 | 1756.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-21 13:15:00 | 1672.30 | 1711.44 | 1728.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-23 13:15:00 | 1599.70 | 1586.82 | 1622.81 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-23 14:00:00 | 1599.70 | 1586.82 | 1622.81 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-28 11:15:00 | 1482.80 | 1489.29 | 1521.72 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-28 12:45:00 | 1474.00 | 1485.77 | 1517.18 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-29 09:45:00 | 1463.75 | 1469.80 | 1498.62 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-29 14:45:00 | 1472.75 | 1471.62 | 1488.26 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-30 12:15:00 | 1539.55 | 1500.09 | 1496.98 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 32 — BUY (started 2024-10-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-30 12:15:00 | 1539.55 | 1500.09 | 1496.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-31 11:15:00 | 1571.90 | 1523.99 | 1510.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-31 15:15:00 | 1548.00 | 1548.71 | 1528.44 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-01 18:00:00 | 1599.00 | 1558.77 | 1534.86 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-01 18:30:00 | 1597.90 | 1566.01 | 1540.33 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 09:15:00 | 1538.30 | 1560.47 | 1540.14 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-11-04 09:15:00 | 1538.30 | 1560.47 | 1540.14 | SL hit (close<ema400) qty=1.00 sl=1540.14 alert=retest1 |

### Cycle 33 — SELL (started 2024-11-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-11 09:15:00 | 1591.80 | 1627.88 | 1629.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-11 13:15:00 | 1568.45 | 1600.85 | 1614.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-12 10:15:00 | 1608.00 | 1593.18 | 1605.74 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-12 10:15:00 | 1608.00 | 1593.18 | 1605.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-12 10:15:00 | 1608.00 | 1593.18 | 1605.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-12 11:00:00 | 1608.00 | 1593.18 | 1605.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-12 11:15:00 | 1588.85 | 1592.31 | 1604.20 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-12 12:30:00 | 1583.50 | 1589.16 | 1601.69 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2024-11-13 09:15:00 | 1425.15 | 1552.25 | 1579.84 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 34 — BUY (started 2024-11-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-21 11:15:00 | 1484.30 | 1479.18 | 1479.01 | EMA200 above EMA400 |

### Cycle 35 — SELL (started 2024-11-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-22 09:15:00 | 1455.70 | 1474.83 | 1477.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-25 12:15:00 | 1441.55 | 1462.12 | 1469.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-25 14:15:00 | 1484.00 | 1462.69 | 1468.30 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-25 14:15:00 | 1484.00 | 1462.69 | 1468.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-25 14:15:00 | 1484.00 | 1462.69 | 1468.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-25 15:00:00 | 1484.00 | 1462.69 | 1468.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-25 15:15:00 | 1477.00 | 1465.55 | 1469.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-26 09:15:00 | 1473.20 | 1465.55 | 1469.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-26 10:15:00 | 1474.75 | 1466.50 | 1468.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-26 10:45:00 | 1469.40 | 1466.50 | 1468.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-26 11:15:00 | 1460.90 | 1465.38 | 1468.13 | EMA400 retest candle locked (from downside) |

### Cycle 36 — BUY (started 2024-11-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-27 14:15:00 | 1487.30 | 1469.25 | 1467.73 | EMA200 above EMA400 |

### Cycle 37 — SELL (started 2024-11-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-28 14:15:00 | 1458.00 | 1467.22 | 1467.90 | EMA200 below EMA400 |

### Cycle 38 — BUY (started 2024-11-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-29 11:15:00 | 1470.45 | 1467.97 | 1467.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-29 13:15:00 | 1488.25 | 1472.58 | 1469.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-02 09:15:00 | 1460.25 | 1471.40 | 1470.19 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-02 09:15:00 | 1460.25 | 1471.40 | 1470.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-02 09:15:00 | 1460.25 | 1471.40 | 1470.19 | EMA400 retest candle locked (from upside) |

### Cycle 39 — SELL (started 2024-12-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-02 10:15:00 | 1454.90 | 1468.10 | 1468.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-02 11:15:00 | 1443.65 | 1463.21 | 1466.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-02 14:15:00 | 1461.40 | 1461.23 | 1464.62 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-02 15:00:00 | 1461.40 | 1461.23 | 1464.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-02 15:15:00 | 1465.00 | 1461.99 | 1464.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-03 09:15:00 | 1493.50 | 1461.99 | 1464.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 40 — BUY (started 2024-12-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-03 09:15:00 | 1499.90 | 1469.57 | 1467.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-04 10:15:00 | 1548.05 | 1505.23 | 1489.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-06 09:15:00 | 1514.50 | 1523.75 | 1515.88 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-06 09:15:00 | 1514.50 | 1523.75 | 1515.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-06 09:15:00 | 1514.50 | 1523.75 | 1515.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-06 10:00:00 | 1514.50 | 1523.75 | 1515.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-06 10:15:00 | 1508.25 | 1520.65 | 1515.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-06 11:00:00 | 1508.25 | 1520.65 | 1515.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-06 11:15:00 | 1490.90 | 1514.70 | 1512.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-06 11:45:00 | 1497.35 | 1514.70 | 1512.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 41 — SELL (started 2024-12-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-06 13:15:00 | 1507.95 | 1511.43 | 1511.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-09 09:15:00 | 1466.05 | 1501.07 | 1506.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-12 12:15:00 | 1432.30 | 1429.27 | 1442.88 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-12 13:00:00 | 1432.30 | 1429.27 | 1442.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-12 14:15:00 | 1440.55 | 1431.74 | 1441.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-12 15:00:00 | 1440.55 | 1431.74 | 1441.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-12 15:15:00 | 1438.95 | 1433.19 | 1441.41 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-13 09:15:00 | 1425.50 | 1433.19 | 1441.41 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-13 12:15:00 | 1449.95 | 1432.12 | 1437.62 | SL hit (close>static) qty=1.00 sl=1442.15 alert=retest2 |

### Cycle 42 — BUY (started 2024-12-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-16 09:15:00 | 1449.70 | 1441.94 | 1441.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-16 11:15:00 | 1506.95 | 1457.16 | 1448.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-17 12:15:00 | 1490.00 | 1502.43 | 1482.68 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-17 13:00:00 | 1490.00 | 1502.43 | 1482.68 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-17 13:15:00 | 1479.90 | 1497.92 | 1482.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-17 14:00:00 | 1479.90 | 1497.92 | 1482.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-17 14:15:00 | 1484.30 | 1495.20 | 1482.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-17 14:30:00 | 1482.05 | 1495.20 | 1482.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-17 15:15:00 | 1490.15 | 1494.19 | 1483.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-18 09:15:00 | 1481.00 | 1494.19 | 1483.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-18 09:15:00 | 1491.70 | 1493.69 | 1484.05 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-18 13:00:00 | 1544.05 | 1504.08 | 1491.07 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-18 13:30:00 | 1544.95 | 1513.66 | 1496.60 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-18 14:00:00 | 1551.95 | 1513.66 | 1496.60 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-19 11:15:00 | 1569.00 | 1541.18 | 1517.53 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 09:15:00 | 1577.70 | 1549.77 | 1532.15 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-20 10:30:00 | 1595.00 | 1562.55 | 1539.56 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-20 15:00:00 | 1620.00 | 1581.52 | 1556.50 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2024-12-23 09:15:00 | 1698.46 | 1600.74 | 1570.11 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 43 — SELL (started 2024-12-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-26 11:15:00 | 1564.45 | 1593.99 | 1596.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-26 13:15:00 | 1558.80 | 1582.92 | 1591.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-27 11:15:00 | 1581.00 | 1574.49 | 1582.93 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-27 11:15:00 | 1581.00 | 1574.49 | 1582.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-27 11:15:00 | 1581.00 | 1574.49 | 1582.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-27 12:00:00 | 1581.00 | 1574.49 | 1582.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-27 12:15:00 | 1563.90 | 1572.37 | 1581.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-27 12:30:00 | 1563.10 | 1572.37 | 1581.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-30 09:15:00 | 1573.60 | 1566.70 | 1575.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-30 09:45:00 | 1571.80 | 1566.70 | 1575.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-30 10:15:00 | 1561.95 | 1565.75 | 1573.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-30 10:30:00 | 1569.15 | 1565.75 | 1573.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-30 14:15:00 | 1582.25 | 1565.52 | 1570.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-30 15:00:00 | 1582.25 | 1565.52 | 1570.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-30 15:15:00 | 1585.05 | 1569.43 | 1572.01 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-31 09:15:00 | 1574.10 | 1569.43 | 1572.01 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-01 09:15:00 | 1632.95 | 1579.14 | 1573.50 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 44 — BUY (started 2025-01-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-01 09:15:00 | 1632.95 | 1579.14 | 1573.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-01 10:15:00 | 1634.70 | 1590.25 | 1579.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-03 09:15:00 | 1661.45 | 1675.38 | 1651.51 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-03 10:00:00 | 1661.45 | 1675.38 | 1651.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-03 10:15:00 | 1653.80 | 1671.07 | 1651.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-03 10:30:00 | 1652.10 | 1671.07 | 1651.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-03 11:15:00 | 1657.05 | 1668.26 | 1652.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-03 11:30:00 | 1653.70 | 1668.26 | 1652.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-03 12:15:00 | 1635.70 | 1661.75 | 1650.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-03 13:00:00 | 1635.70 | 1661.75 | 1650.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-03 13:15:00 | 1640.00 | 1657.40 | 1649.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-03 13:30:00 | 1644.10 | 1657.40 | 1649.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 45 — SELL (started 2025-01-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-06 09:15:00 | 1602.55 | 1639.91 | 1643.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-06 10:15:00 | 1567.45 | 1625.42 | 1636.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-07 09:15:00 | 1589.40 | 1579.33 | 1604.07 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-07 10:15:00 | 1588.00 | 1581.06 | 1602.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 10:15:00 | 1588.00 | 1581.06 | 1602.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-07 11:00:00 | 1588.00 | 1581.06 | 1602.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 11:15:00 | 1596.00 | 1584.05 | 1602.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-07 11:45:00 | 1593.20 | 1584.05 | 1602.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-16 09:15:00 | 1336.45 | 1294.56 | 1320.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-16 10:00:00 | 1336.45 | 1294.56 | 1320.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-16 10:15:00 | 1334.65 | 1302.57 | 1322.03 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-16 11:15:00 | 1328.50 | 1302.57 | 1322.03 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-20 09:15:00 | 1262.08 | 1288.89 | 1301.55 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-01-20 11:15:00 | 1318.00 | 1294.62 | 1301.95 | SL hit (close>ema200) qty=0.50 sl=1294.62 alert=retest2 |

### Cycle 46 — BUY (started 2025-01-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-20 14:15:00 | 1324.85 | 1308.41 | 1307.10 | EMA200 above EMA400 |

### Cycle 47 — SELL (started 2025-01-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-21 10:15:00 | 1282.00 | 1301.64 | 1304.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-21 14:15:00 | 1275.80 | 1290.13 | 1297.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-23 09:15:00 | 1233.10 | 1224.21 | 1250.73 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-23 10:00:00 | 1233.10 | 1224.21 | 1250.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 09:15:00 | 1082.10 | 1052.39 | 1082.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-29 10:00:00 | 1082.10 | 1052.39 | 1082.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 10:15:00 | 1079.10 | 1057.73 | 1082.29 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-29 11:15:00 | 1071.75 | 1057.73 | 1082.29 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-29 15:15:00 | 1109.95 | 1076.96 | 1082.72 | SL hit (close>static) qty=1.00 sl=1087.90 alert=retest2 |

### Cycle 48 — BUY (started 2025-02-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-01 09:15:00 | 1091.00 | 1076.89 | 1075.26 | EMA200 above EMA400 |

### Cycle 49 — SELL (started 2025-02-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-01 12:15:00 | 1037.00 | 1068.10 | 1071.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-03 09:15:00 | 1005.40 | 1045.53 | 1058.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-04 09:15:00 | 1043.30 | 1016.17 | 1033.07 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-04 09:15:00 | 1043.30 | 1016.17 | 1033.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-04 09:15:00 | 1043.30 | 1016.17 | 1033.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-04 10:00:00 | 1043.30 | 1016.17 | 1033.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-04 10:15:00 | 1034.00 | 1019.73 | 1033.16 | EMA400 retest candle locked (from downside) |

### Cycle 50 — BUY (started 2025-02-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-04 13:15:00 | 1088.00 | 1043.70 | 1041.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-04 14:15:00 | 1114.85 | 1057.93 | 1048.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-06 13:15:00 | 1100.85 | 1101.89 | 1088.91 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-06 13:45:00 | 1102.00 | 1101.89 | 1088.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 09:15:00 | 1103.95 | 1105.95 | 1094.26 | EMA400 retest candle locked (from upside) |

### Cycle 51 — SELL (started 2025-02-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-10 09:15:00 | 1043.85 | 1087.99 | 1090.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-10 10:15:00 | 1031.90 | 1076.77 | 1085.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-13 09:15:00 | 987.00 | 977.74 | 998.97 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-13 09:15:00 | 987.00 | 977.74 | 998.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-13 09:15:00 | 987.00 | 977.74 | 998.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-13 09:45:00 | 997.85 | 977.74 | 998.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-14 09:15:00 | 931.85 | 964.71 | 982.25 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-14 10:15:00 | 927.20 | 964.71 | 982.25 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-14 15:00:00 | 925.10 | 935.04 | 958.85 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-18 09:15:00 | 880.84 | 914.48 | 931.57 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-18 09:15:00 | 878.85 | 914.48 | 931.57 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-02-18 14:15:00 | 907.25 | 906.10 | 920.26 | SL hit (close>ema200) qty=0.50 sl=906.10 alert=retest2 |

### Cycle 52 — BUY (started 2025-02-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-19 10:15:00 | 981.35 | 928.27 | 927.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-19 14:15:00 | 992.85 | 960.19 | 944.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-20 15:15:00 | 994.50 | 995.12 | 975.92 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-21 09:15:00 | 1027.35 | 995.12 | 975.92 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-21 11:30:00 | 1006.45 | 996.47 | 981.24 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-21 14:45:00 | 1003.35 | 997.62 | 985.66 | BUY ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-24 09:15:00 | 986.40 | 995.14 | 986.59 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-02-24 09:15:00 | 986.40 | 995.14 | 986.59 | SL hit (close<ema400) qty=1.00 sl=986.59 alert=retest1 |

### Cycle 53 — SELL (started 2025-02-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-25 10:15:00 | 969.50 | 983.29 | 984.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-25 11:15:00 | 960.15 | 978.66 | 982.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-27 14:15:00 | 952.35 | 947.64 | 959.35 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-27 14:30:00 | 950.40 | 947.64 | 959.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-28 09:15:00 | 924.25 | 943.02 | 955.22 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-28 11:00:00 | 916.55 | 937.73 | 951.71 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-28 14:15:00 | 979.15 | 939.57 | 947.27 | SL hit (close>static) qty=1.00 sl=961.95 alert=retest2 |

### Cycle 54 — BUY (started 2025-03-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-05 10:15:00 | 960.00 | 943.90 | 942.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-06 15:15:00 | 973.50 | 961.67 | 954.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-07 11:15:00 | 963.20 | 966.61 | 959.18 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-07 12:00:00 | 963.20 | 966.61 | 959.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-07 12:15:00 | 960.95 | 965.48 | 959.34 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-07 14:30:00 | 969.55 | 963.06 | 959.25 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-07 15:15:00 | 947.45 | 959.94 | 958.17 | SL hit (close<static) qty=1.00 sl=956.95 alert=retest2 |

### Cycle 55 — SELL (started 2025-03-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-10 09:15:00 | 928.95 | 953.74 | 955.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-10 10:15:00 | 919.95 | 946.98 | 952.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-11 13:15:00 | 912.00 | 904.69 | 920.52 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-11 14:00:00 | 912.00 | 904.69 | 920.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-11 14:15:00 | 916.15 | 906.98 | 920.12 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-12 10:45:00 | 899.35 | 907.21 | 917.13 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-12 15:15:00 | 935.00 | 907.29 | 912.44 | SL hit (close>static) qty=1.00 sl=922.00 alert=retest2 |

### Cycle 56 — BUY (started 2025-03-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-17 10:15:00 | 931.60 | 912.17 | 911.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-18 09:15:00 | 953.05 | 927.60 | 920.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-18 14:15:00 | 936.85 | 938.41 | 929.31 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-18 14:15:00 | 936.85 | 938.41 | 929.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-18 14:15:00 | 936.85 | 938.41 | 929.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-18 15:00:00 | 936.85 | 938.41 | 929.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 09:15:00 | 1024.00 | 1043.51 | 1028.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-25 10:00:00 | 1024.00 | 1043.51 | 1028.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 10:15:00 | 1016.15 | 1038.04 | 1027.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-25 11:00:00 | 1016.15 | 1038.04 | 1027.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 11:15:00 | 1015.80 | 1033.59 | 1026.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-25 11:30:00 | 1009.30 | 1033.59 | 1026.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 57 — SELL (started 2025-03-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-25 15:15:00 | 1006.90 | 1019.89 | 1021.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-26 13:15:00 | 1001.10 | 1011.28 | 1016.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-27 10:15:00 | 1013.95 | 1008.75 | 1013.17 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-27 10:15:00 | 1013.95 | 1008.75 | 1013.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 10:15:00 | 1013.95 | 1008.75 | 1013.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-27 11:00:00 | 1013.95 | 1008.75 | 1013.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 11:15:00 | 1006.45 | 1008.29 | 1012.56 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-27 12:15:00 | 1003.40 | 1008.29 | 1012.56 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-27 14:15:00 | 1016.50 | 1010.76 | 1012.67 | SL hit (close>static) qty=1.00 sl=1013.95 alert=retest2 |

### Cycle 58 — BUY (started 2025-03-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-28 09:15:00 | 1039.60 | 1018.33 | 1015.89 | EMA200 above EMA400 |

### Cycle 59 — SELL (started 2025-03-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-28 14:15:00 | 1004.05 | 1015.14 | 1015.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-28 15:15:00 | 1001.95 | 1012.51 | 1014.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-01 09:15:00 | 1027.30 | 1015.46 | 1015.61 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-01 09:15:00 | 1027.30 | 1015.46 | 1015.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-01 09:15:00 | 1027.30 | 1015.46 | 1015.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-01 10:00:00 | 1027.30 | 1015.46 | 1015.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 60 — BUY (started 2025-04-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-01 10:15:00 | 1020.65 | 1016.50 | 1016.07 | EMA200 above EMA400 |

### Cycle 61 — SELL (started 2025-04-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-04 09:15:00 | 974.30 | 1012.42 | 1017.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-07 09:15:00 | 909.25 | 977.06 | 996.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-08 10:15:00 | 938.95 | 938.07 | 959.88 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-08 11:00:00 | 938.95 | 938.07 | 959.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-11 09:15:00 | 966.05 | 933.35 | 939.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-11 10:00:00 | 966.05 | 933.35 | 939.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 62 — BUY (started 2025-04-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-11 11:15:00 | 960.15 | 945.38 | 944.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-15 09:15:00 | 1026.20 | 979.55 | 963.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-17 14:15:00 | 1053.80 | 1053.87 | 1032.94 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-17 14:45:00 | 1050.00 | 1053.87 | 1032.94 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-22 13:15:00 | 1107.00 | 1119.51 | 1095.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-22 13:30:00 | 1106.10 | 1119.51 | 1095.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 09:15:00 | 1090.90 | 1114.31 | 1099.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-23 10:00:00 | 1090.90 | 1114.31 | 1099.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 10:15:00 | 1111.10 | 1113.66 | 1100.28 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-23 12:45:00 | 1116.50 | 1113.83 | 1102.65 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-25 09:15:00 | 1076.50 | 1103.91 | 1106.87 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 63 — SELL (started 2025-04-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-25 09:15:00 | 1076.50 | 1103.91 | 1106.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-30 14:15:00 | 1059.20 | 1078.96 | 1086.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-05 09:15:00 | 1082.50 | 1071.73 | 1076.58 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-05 09:15:00 | 1082.50 | 1071.73 | 1076.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 09:15:00 | 1082.50 | 1071.73 | 1076.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-05 10:00:00 | 1082.50 | 1071.73 | 1076.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 10:15:00 | 1080.60 | 1073.51 | 1076.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-05 10:30:00 | 1082.70 | 1073.51 | 1076.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 64 — BUY (started 2025-05-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-05 15:15:00 | 1085.00 | 1078.84 | 1078.54 | EMA200 above EMA400 |

### Cycle 65 — SELL (started 2025-05-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-06 09:15:00 | 1056.80 | 1074.43 | 1076.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-06 14:15:00 | 1049.70 | 1064.15 | 1070.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-07 11:15:00 | 1069.50 | 1055.76 | 1063.11 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-07 11:15:00 | 1069.50 | 1055.76 | 1063.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 11:15:00 | 1069.50 | 1055.76 | 1063.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-07 11:30:00 | 1069.10 | 1055.76 | 1063.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 12:15:00 | 1078.50 | 1060.31 | 1064.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-07 13:00:00 | 1078.50 | 1060.31 | 1064.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 66 — BUY (started 2025-05-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-07 15:15:00 | 1081.00 | 1069.10 | 1067.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-08 09:15:00 | 1102.10 | 1075.70 | 1071.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-08 13:15:00 | 1081.00 | 1081.30 | 1075.72 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-08 13:15:00 | 1081.00 | 1081.30 | 1075.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 13:15:00 | 1081.00 | 1081.30 | 1075.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-08 13:30:00 | 1072.10 | 1081.30 | 1075.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 14:15:00 | 1068.20 | 1078.68 | 1075.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-08 15:00:00 | 1068.20 | 1078.68 | 1075.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 15:15:00 | 1070.00 | 1076.95 | 1074.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-09 09:15:00 | 1045.40 | 1076.95 | 1074.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 67 — SELL (started 2025-05-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-09 10:15:00 | 1058.40 | 1070.64 | 1071.96 | EMA200 below EMA400 |

### Cycle 68 — BUY (started 2025-05-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-09 14:15:00 | 1082.60 | 1073.70 | 1072.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 09:15:00 | 1118.60 | 1083.85 | 1077.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-13 12:15:00 | 1115.70 | 1117.55 | 1104.90 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-13 13:00:00 | 1115.70 | 1117.55 | 1104.90 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 09:15:00 | 1252.40 | 1266.87 | 1247.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 09:30:00 | 1254.90 | 1266.87 | 1247.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 10:15:00 | 1250.70 | 1263.64 | 1247.56 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-20 11:15:00 | 1259.20 | 1263.64 | 1247.56 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-20 13:15:00 | 1241.80 | 1256.24 | 1247.99 | SL hit (close<static) qty=1.00 sl=1246.00 alert=retest2 |

### Cycle 69 — SELL (started 2025-05-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-21 11:15:00 | 1230.00 | 1243.84 | 1244.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-22 12:15:00 | 1228.00 | 1236.93 | 1239.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-26 09:15:00 | 1273.10 | 1231.80 | 1232.21 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-26 09:15:00 | 1273.10 | 1231.80 | 1232.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 09:15:00 | 1273.10 | 1231.80 | 1232.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-26 09:30:00 | 1274.80 | 1231.80 | 1232.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 70 — BUY (started 2025-05-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-26 10:15:00 | 1270.40 | 1239.52 | 1235.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-26 12:15:00 | 1293.90 | 1256.07 | 1244.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-27 11:15:00 | 1245.20 | 1266.91 | 1256.40 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-27 11:15:00 | 1245.20 | 1266.91 | 1256.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 11:15:00 | 1245.20 | 1266.91 | 1256.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-27 12:00:00 | 1245.20 | 1266.91 | 1256.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 12:15:00 | 1249.50 | 1263.43 | 1255.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-27 12:30:00 | 1236.00 | 1263.43 | 1255.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 14:15:00 | 1245.60 | 1256.21 | 1253.59 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-28 09:15:00 | 1359.00 | 1257.19 | 1254.28 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-05-30 09:15:00 | 1494.90 | 1427.11 | 1387.84 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 71 — SELL (started 2025-06-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-12 09:15:00 | 1516.10 | 1519.34 | 1519.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-12 10:15:00 | 1477.20 | 1510.91 | 1515.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-16 09:15:00 | 1463.10 | 1461.53 | 1475.89 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-16 09:15:00 | 1463.10 | 1461.53 | 1475.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 09:15:00 | 1463.10 | 1461.53 | 1475.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 10:15:00 | 1479.40 | 1461.53 | 1475.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 10:15:00 | 1471.30 | 1463.48 | 1475.48 | EMA400 retest candle locked (from downside) |

### Cycle 72 — BUY (started 2025-06-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-17 09:15:00 | 1527.90 | 1484.57 | 1480.89 | EMA200 above EMA400 |

### Cycle 73 — SELL (started 2025-06-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-19 11:15:00 | 1483.10 | 1496.17 | 1496.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-19 12:15:00 | 1461.50 | 1489.24 | 1493.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-20 09:15:00 | 1479.00 | 1478.21 | 1485.89 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-20 09:15:00 | 1479.00 | 1478.21 | 1485.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 09:15:00 | 1479.00 | 1478.21 | 1485.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 09:45:00 | 1472.10 | 1478.21 | 1485.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 14:15:00 | 1486.50 | 1478.00 | 1482.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 15:00:00 | 1486.50 | 1478.00 | 1482.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 15:15:00 | 1485.00 | 1479.40 | 1482.93 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-23 09:15:00 | 1475.10 | 1479.40 | 1482.93 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-23 11:15:00 | 1490.90 | 1482.59 | 1483.58 | SL hit (close>static) qty=1.00 sl=1488.00 alert=retest2 |

### Cycle 74 — BUY (started 2025-06-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-23 12:15:00 | 1492.40 | 1484.55 | 1484.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-23 13:15:00 | 1515.50 | 1490.74 | 1487.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-24 15:15:00 | 1523.00 | 1523.25 | 1510.99 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-25 09:15:00 | 1536.70 | 1523.25 | 1510.99 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-25 11:30:00 | 1535.00 | 1526.32 | 1515.65 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-25 12:15:00 | 1531.70 | 1526.32 | 1515.65 | BUY ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-25 12:15:00 | 1504.00 | 1521.85 | 1514.59 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-06-25 12:15:00 | 1504.00 | 1521.85 | 1514.59 | SL hit (close<ema400) qty=1.00 sl=1514.59 alert=retest1 |

### Cycle 75 — SELL (started 2025-06-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-26 09:15:00 | 1482.80 | 1506.93 | 1509.15 | EMA200 below EMA400 |

### Cycle 76 — BUY (started 2025-06-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-26 15:15:00 | 1533.00 | 1512.90 | 1510.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-27 09:15:00 | 1606.20 | 1531.56 | 1518.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-30 12:15:00 | 1588.00 | 1590.05 | 1566.61 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-30 13:00:00 | 1588.00 | 1590.05 | 1566.61 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 10:15:00 | 1578.50 | 1589.34 | 1575.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-01 10:45:00 | 1568.10 | 1589.34 | 1575.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 11:15:00 | 1586.80 | 1588.83 | 1576.74 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-01 12:45:00 | 1596.80 | 1589.34 | 1578.07 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-01 13:15:00 | 1602.00 | 1589.34 | 1578.07 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-03 09:15:00 | 1575.00 | 1591.51 | 1591.39 | SL hit (close<static) qty=1.00 sl=1576.00 alert=retest2 |

### Cycle 77 — SELL (started 2025-07-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-03 10:15:00 | 1578.30 | 1588.87 | 1590.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-03 12:15:00 | 1569.70 | 1582.70 | 1587.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-04 09:15:00 | 1575.00 | 1572.98 | 1580.27 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-04 09:15:00 | 1575.00 | 1572.98 | 1580.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 09:15:00 | 1575.00 | 1572.98 | 1580.27 | EMA400 retest candle locked (from downside) |

### Cycle 78 — BUY (started 2025-07-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-08 09:15:00 | 1609.50 | 1581.57 | 1578.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-09 10:15:00 | 1627.90 | 1608.65 | 1595.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-10 10:15:00 | 1619.00 | 1620.34 | 1609.58 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-10 11:00:00 | 1619.00 | 1620.34 | 1609.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 12:15:00 | 1608.40 | 1617.92 | 1610.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-10 13:00:00 | 1608.40 | 1617.92 | 1610.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 13:15:00 | 1608.90 | 1616.11 | 1610.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-10 13:45:00 | 1607.50 | 1616.11 | 1610.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 14:15:00 | 1612.50 | 1615.39 | 1610.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-10 14:30:00 | 1607.60 | 1615.39 | 1610.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 15:15:00 | 1605.00 | 1613.31 | 1609.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-11 09:15:00 | 1600.60 | 1613.31 | 1609.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 09:15:00 | 1598.00 | 1610.25 | 1608.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-11 09:45:00 | 1601.70 | 1610.25 | 1608.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 79 — SELL (started 2025-07-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-11 10:15:00 | 1592.20 | 1606.64 | 1607.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-14 09:15:00 | 1541.80 | 1587.69 | 1597.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-15 10:15:00 | 1553.20 | 1552.67 | 1569.86 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-15 11:00:00 | 1553.20 | 1552.67 | 1569.86 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 12:15:00 | 1565.30 | 1554.85 | 1567.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-15 13:00:00 | 1565.30 | 1554.85 | 1567.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 13:15:00 | 1562.50 | 1556.38 | 1567.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-15 14:15:00 | 1572.50 | 1556.38 | 1567.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 14:15:00 | 1574.50 | 1560.00 | 1568.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-15 15:00:00 | 1574.50 | 1560.00 | 1568.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 15:15:00 | 1571.00 | 1562.20 | 1568.28 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-16 09:15:00 | 1561.80 | 1562.20 | 1568.28 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-23 13:15:00 | 1483.71 | 1503.01 | 1518.27 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-07-28 13:15:00 | 1405.62 | 1432.95 | 1453.32 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 80 — BUY (started 2025-07-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-30 11:15:00 | 1457.10 | 1446.15 | 1445.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-31 12:15:00 | 1477.00 | 1458.09 | 1452.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-31 14:15:00 | 1461.10 | 1461.32 | 1454.96 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-31 14:15:00 | 1461.10 | 1461.32 | 1454.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 14:15:00 | 1461.10 | 1461.32 | 1454.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-31 15:00:00 | 1461.10 | 1461.32 | 1454.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 15:15:00 | 1451.00 | 1459.26 | 1454.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-01 09:15:00 | 1439.80 | 1459.26 | 1454.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 09:15:00 | 1432.60 | 1453.93 | 1452.60 | EMA400 retest candle locked (from upside) |

### Cycle 81 — SELL (started 2025-08-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-01 10:15:00 | 1437.20 | 1450.58 | 1451.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-01 13:15:00 | 1426.60 | 1441.73 | 1446.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-04 14:15:00 | 1438.90 | 1430.40 | 1436.04 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-04 14:15:00 | 1438.90 | 1430.40 | 1436.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 14:15:00 | 1438.90 | 1430.40 | 1436.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 15:00:00 | 1438.90 | 1430.40 | 1436.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 15:15:00 | 1425.00 | 1429.32 | 1435.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-05 09:15:00 | 1453.00 | 1429.32 | 1435.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 09:15:00 | 1435.50 | 1430.56 | 1435.07 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-06 10:00:00 | 1409.00 | 1425.86 | 1430.92 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-13 09:15:00 | 1445.70 | 1391.34 | 1390.70 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 82 — BUY (started 2025-08-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-13 09:15:00 | 1445.70 | 1391.34 | 1390.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-14 11:15:00 | 1489.10 | 1451.08 | 1427.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-18 10:15:00 | 1473.10 | 1476.66 | 1453.70 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-18 10:45:00 | 1471.40 | 1476.66 | 1453.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 12:15:00 | 1447.30 | 1468.49 | 1453.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-18 13:00:00 | 1447.30 | 1468.49 | 1453.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 13:15:00 | 1442.90 | 1463.37 | 1452.84 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-19 10:15:00 | 1450.40 | 1453.49 | 1450.40 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-19 13:45:00 | 1451.60 | 1450.49 | 1449.88 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-19 15:00:00 | 1453.10 | 1451.02 | 1450.17 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-29 09:15:00 | 1511.10 | 1528.56 | 1530.91 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 83 — SELL (started 2025-08-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-29 09:15:00 | 1511.10 | 1528.56 | 1530.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-01 11:15:00 | 1501.90 | 1514.91 | 1522.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-01 13:15:00 | 1517.50 | 1513.44 | 1520.01 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-01 13:15:00 | 1517.50 | 1513.44 | 1520.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 13:15:00 | 1517.50 | 1513.44 | 1520.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 13:30:00 | 1517.30 | 1513.44 | 1520.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 14:15:00 | 1516.80 | 1514.11 | 1519.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 14:45:00 | 1516.60 | 1514.11 | 1519.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 09:15:00 | 1547.00 | 1520.83 | 1521.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-02 09:45:00 | 1535.00 | 1520.83 | 1521.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 84 — BUY (started 2025-09-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-02 10:15:00 | 1550.10 | 1526.69 | 1524.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-03 09:15:00 | 1560.90 | 1537.48 | 1531.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-04 10:15:00 | 1535.00 | 1546.92 | 1541.58 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-04 10:15:00 | 1535.00 | 1546.92 | 1541.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 10:15:00 | 1535.00 | 1546.92 | 1541.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 10:45:00 | 1535.40 | 1546.92 | 1541.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 11:15:00 | 1528.50 | 1543.23 | 1540.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 11:30:00 | 1529.80 | 1543.23 | 1540.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 85 — SELL (started 2025-09-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-04 12:15:00 | 1518.40 | 1538.27 | 1538.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-04 13:15:00 | 1511.10 | 1532.83 | 1535.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-08 09:15:00 | 1546.90 | 1522.00 | 1525.72 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-08 09:15:00 | 1546.90 | 1522.00 | 1525.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 09:15:00 | 1546.90 | 1522.00 | 1525.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-08 09:45:00 | 1542.10 | 1522.00 | 1525.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 10:15:00 | 1536.30 | 1524.86 | 1526.68 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-08 11:15:00 | 1526.90 | 1524.86 | 1526.68 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-12 09:15:00 | 1450.56 | 1462.80 | 1477.91 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-15 09:15:00 | 1451.80 | 1447.44 | 1460.58 | SL hit (close>ema200) qty=0.50 sl=1447.44 alert=retest2 |

### Cycle 86 — BUY (started 2025-10-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-03 09:15:00 | 1351.00 | 1341.10 | 1340.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-03 14:15:00 | 1371.50 | 1355.03 | 1347.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-06 09:15:00 | 1357.10 | 1359.12 | 1351.01 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-06 10:00:00 | 1357.10 | 1359.12 | 1351.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 10:15:00 | 1351.00 | 1357.50 | 1351.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-06 10:30:00 | 1353.80 | 1357.50 | 1351.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 11:15:00 | 1356.30 | 1357.26 | 1351.49 | EMA400 retest candle locked (from upside) |

### Cycle 87 — SELL (started 2025-10-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-07 11:15:00 | 1334.50 | 1349.35 | 1350.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-07 15:15:00 | 1333.30 | 1342.19 | 1346.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-08 10:15:00 | 1342.90 | 1341.71 | 1345.35 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-08 10:15:00 | 1342.90 | 1341.71 | 1345.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 10:15:00 | 1342.90 | 1341.71 | 1345.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-08 10:45:00 | 1343.90 | 1341.71 | 1345.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 11:15:00 | 1340.00 | 1341.37 | 1344.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-08 11:30:00 | 1339.90 | 1341.37 | 1344.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 12:15:00 | 1344.80 | 1342.06 | 1344.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-08 12:30:00 | 1350.30 | 1342.06 | 1344.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 13:15:00 | 1338.80 | 1341.40 | 1344.31 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-09 09:30:00 | 1335.90 | 1340.82 | 1343.42 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-09 10:00:00 | 1333.80 | 1340.82 | 1343.42 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-10 10:15:00 | 1349.00 | 1329.77 | 1333.72 | SL hit (close>static) qty=1.00 sl=1346.10 alert=retest2 |

### Cycle 88 — BUY (started 2025-10-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-10 12:15:00 | 1372.20 | 1342.18 | 1338.92 | EMA200 above EMA400 |

### Cycle 89 — SELL (started 2025-10-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-15 10:15:00 | 1340.30 | 1349.08 | 1349.66 | EMA200 below EMA400 |

### Cycle 90 — BUY (started 2025-10-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-15 14:15:00 | 1360.00 | 1349.86 | 1349.59 | EMA200 above EMA400 |

### Cycle 91 — SELL (started 2025-10-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-16 14:15:00 | 1341.40 | 1349.20 | 1350.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-17 12:15:00 | 1319.70 | 1339.70 | 1344.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-20 12:15:00 | 1321.50 | 1320.20 | 1330.52 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-20 12:30:00 | 1322.40 | 1320.20 | 1330.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 14:15:00 | 1329.50 | 1322.83 | 1329.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-20 14:30:00 | 1330.00 | 1322.83 | 1329.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 15:15:00 | 1334.00 | 1325.06 | 1330.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-21 13:45:00 | 1350.50 | 1331.25 | 1332.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 92 — BUY (started 2025-10-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-21 14:15:00 | 1354.00 | 1335.80 | 1334.61 | EMA200 above EMA400 |

### Cycle 93 — SELL (started 2025-10-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-24 09:15:00 | 1317.00 | 1334.47 | 1336.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-24 14:15:00 | 1311.70 | 1321.06 | 1328.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-27 13:15:00 | 1320.70 | 1312.41 | 1319.81 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-27 13:15:00 | 1320.70 | 1312.41 | 1319.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 13:15:00 | 1320.70 | 1312.41 | 1319.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-27 14:00:00 | 1320.70 | 1312.41 | 1319.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 14:15:00 | 1322.90 | 1314.51 | 1320.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-27 15:00:00 | 1322.90 | 1314.51 | 1320.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 15:15:00 | 1324.70 | 1316.55 | 1320.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-28 09:15:00 | 1328.80 | 1316.55 | 1320.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 09:15:00 | 1328.90 | 1319.02 | 1321.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-28 09:30:00 | 1329.70 | 1319.02 | 1321.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 10:15:00 | 1333.00 | 1321.81 | 1322.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-28 10:45:00 | 1333.50 | 1321.81 | 1322.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 94 — BUY (started 2025-10-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-28 11:15:00 | 1331.20 | 1323.69 | 1323.14 | EMA200 above EMA400 |

### Cycle 95 — SELL (started 2025-10-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-28 13:15:00 | 1314.00 | 1321.77 | 1322.37 | EMA200 below EMA400 |

### Cycle 96 — BUY (started 2025-10-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-28 15:15:00 | 1330.00 | 1323.57 | 1323.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-29 10:15:00 | 1342.20 | 1326.80 | 1324.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-30 09:15:00 | 1336.80 | 1338.88 | 1332.78 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-30 10:00:00 | 1336.80 | 1338.88 | 1332.78 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 10:15:00 | 1333.80 | 1337.86 | 1332.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-30 11:00:00 | 1333.80 | 1337.86 | 1332.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 11:15:00 | 1335.00 | 1337.29 | 1333.06 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-30 14:30:00 | 1348.70 | 1337.46 | 1334.05 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-31 11:15:00 | 1341.10 | 1339.39 | 1335.97 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-31 11:45:00 | 1341.90 | 1340.56 | 1336.81 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-31 14:15:00 | 1316.50 | 1335.04 | 1335.20 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 97 — SELL (started 2025-10-31 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-31 14:15:00 | 1316.50 | 1335.04 | 1335.20 | EMA200 below EMA400 |

### Cycle 98 — BUY (started 2025-11-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-03 09:15:00 | 1343.90 | 1336.00 | 1335.57 | EMA200 above EMA400 |

### Cycle 99 — SELL (started 2025-11-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-04 10:15:00 | 1329.00 | 1336.44 | 1336.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-04 11:15:00 | 1316.50 | 1332.45 | 1334.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-07 13:15:00 | 1275.80 | 1270.99 | 1287.02 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-07 13:15:00 | 1275.80 | 1270.99 | 1287.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 13:15:00 | 1275.80 | 1270.99 | 1287.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-07 13:45:00 | 1277.90 | 1270.99 | 1287.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 15:15:00 | 1283.40 | 1275.54 | 1286.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-10 09:15:00 | 1290.50 | 1275.54 | 1286.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 09:15:00 | 1289.70 | 1278.37 | 1286.73 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-10 11:45:00 | 1282.50 | 1280.82 | 1286.53 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-11 09:15:00 | 1272.90 | 1280.38 | 1284.25 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-13 09:15:00 | 1218.38 | 1267.38 | 1273.41 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-13 09:15:00 | 1209.26 | 1267.38 | 1273.41 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-14 12:15:00 | 1241.50 | 1240.58 | 1250.65 | SL hit (close>ema200) qty=0.50 sl=1240.58 alert=retest2 |

### Cycle 100 — BUY (started 2025-11-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-26 13:15:00 | 1185.80 | 1179.07 | 1178.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-27 10:15:00 | 1194.00 | 1184.15 | 1181.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-28 09:15:00 | 1194.40 | 1195.91 | 1189.81 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-28 10:15:00 | 1210.20 | 1195.91 | 1189.81 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-28 11:15:00 | 1209.50 | 1198.15 | 1191.38 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-28 11:45:00 | 1211.50 | 1200.32 | 1192.98 | BUY ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-28 15:00:00 | 1208.80 | 1206.06 | 1197.78 | BUY ENTRY1 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 09:15:00 | 1200.90 | 1204.22 | 1198.33 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-12-01 10:15:00 | 1191.50 | 1201.68 | 1197.71 | SL hit (close<ema400) qty=1.00 sl=1197.71 alert=retest1 |

### Cycle 101 — SELL (started 2025-12-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-02 09:15:00 | 1174.20 | 1192.01 | 1194.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-02 10:15:00 | 1166.60 | 1186.93 | 1191.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-03 15:15:00 | 1147.00 | 1146.50 | 1160.38 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-04 09:15:00 | 1129.00 | 1146.50 | 1160.38 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-08 12:15:00 | 1072.55 | 1087.44 | 1105.83 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-12-09 10:15:00 | 1091.00 | 1077.87 | 1092.86 | SL hit (close>ema200) qty=0.50 sl=1077.87 alert=retest1 |

### Cycle 102 — BUY (started 2025-12-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-11 14:15:00 | 1102.70 | 1092.68 | 1091.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-12 09:15:00 | 1106.20 | 1097.32 | 1093.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-15 09:15:00 | 1100.30 | 1102.42 | 1098.78 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-15 09:15:00 | 1100.30 | 1102.42 | 1098.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 09:15:00 | 1100.30 | 1102.42 | 1098.78 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-15 10:15:00 | 1104.80 | 1102.42 | 1098.78 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-16 11:15:00 | 1088.60 | 1099.95 | 1100.88 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 103 — SELL (started 2025-12-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-16 11:15:00 | 1088.60 | 1099.95 | 1100.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-17 09:15:00 | 1084.40 | 1096.47 | 1098.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-19 10:15:00 | 1072.20 | 1064.63 | 1073.28 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-19 10:15:00 | 1072.20 | 1064.63 | 1073.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 10:15:00 | 1072.20 | 1064.63 | 1073.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 11:00:00 | 1072.20 | 1064.63 | 1073.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 11:15:00 | 1075.00 | 1066.70 | 1073.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 11:30:00 | 1075.00 | 1066.70 | 1073.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 12:15:00 | 1075.20 | 1068.40 | 1073.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 12:30:00 | 1074.90 | 1068.40 | 1073.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 13:15:00 | 1070.80 | 1068.88 | 1073.35 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-19 14:15:00 | 1066.80 | 1068.88 | 1073.35 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-19 14:15:00 | 1090.00 | 1073.11 | 1074.86 | SL hit (close>static) qty=1.00 sl=1075.20 alert=retest2 |

### Cycle 104 — BUY (started 2025-12-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-19 15:15:00 | 1096.00 | 1077.68 | 1076.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-22 15:15:00 | 1104.00 | 1086.58 | 1082.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-24 09:15:00 | 1116.80 | 1117.15 | 1104.86 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-24 10:00:00 | 1116.80 | 1117.15 | 1104.86 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 13:15:00 | 1105.80 | 1113.25 | 1106.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-24 14:00:00 | 1105.80 | 1113.25 | 1106.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 14:15:00 | 1107.20 | 1112.04 | 1106.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-24 15:00:00 | 1107.20 | 1112.04 | 1106.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 15:15:00 | 1102.80 | 1110.20 | 1106.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-26 09:15:00 | 1102.00 | 1110.20 | 1106.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 09:15:00 | 1104.90 | 1109.14 | 1106.34 | EMA400 retest candle locked (from upside) |

### Cycle 105 — SELL (started 2025-12-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-26 12:15:00 | 1096.60 | 1103.90 | 1104.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-26 13:15:00 | 1092.30 | 1101.58 | 1103.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-31 09:15:00 | 1067.90 | 1061.20 | 1072.47 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-31 09:15:00 | 1067.90 | 1061.20 | 1072.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 09:15:00 | 1067.90 | 1061.20 | 1072.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 09:30:00 | 1064.00 | 1061.20 | 1072.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 10:15:00 | 1074.70 | 1063.90 | 1072.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 10:45:00 | 1078.60 | 1063.90 | 1072.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 11:15:00 | 1080.00 | 1067.12 | 1073.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 12:00:00 | 1080.00 | 1067.12 | 1073.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 12:15:00 | 1078.10 | 1069.32 | 1073.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 12:45:00 | 1082.00 | 1069.32 | 1073.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 10:15:00 | 1074.80 | 1075.06 | 1075.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-01 10:45:00 | 1075.00 | 1075.06 | 1075.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 11:15:00 | 1068.10 | 1073.67 | 1074.86 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-01 12:15:00 | 1065.10 | 1073.67 | 1074.86 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-01 13:00:00 | 1062.10 | 1071.35 | 1073.70 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-02 10:15:00 | 1097.10 | 1074.50 | 1073.81 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 106 — BUY (started 2026-01-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-02 10:15:00 | 1097.10 | 1074.50 | 1073.81 | EMA200 above EMA400 |

### Cycle 107 — SELL (started 2026-01-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-06 11:15:00 | 1075.90 | 1090.48 | 1090.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-06 13:15:00 | 1073.00 | 1085.50 | 1088.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-12 15:15:00 | 1005.00 | 1004.37 | 1018.44 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-13 09:15:00 | 1006.10 | 1004.37 | 1018.44 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 09:15:00 | 991.40 | 994.37 | 999.67 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-16 12:15:00 | 982.80 | 992.18 | 997.69 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-20 11:15:00 | 933.66 | 950.73 | 965.76 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-22 09:15:00 | 919.80 | 917.93 | 933.22 | SL hit (close>ema200) qty=0.50 sl=917.93 alert=retest2 |

### Cycle 108 — BUY (started 2026-01-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-28 11:15:00 | 939.70 | 914.97 | 911.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-30 09:15:00 | 986.00 | 946.12 | 933.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-01 14:15:00 | 982.25 | 995.43 | 976.43 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-01 14:15:00 | 982.25 | 995.43 | 976.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 14:15:00 | 982.25 | 995.43 | 976.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 15:00:00 | 982.25 | 995.43 | 976.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 15:15:00 | 1005.00 | 997.34 | 979.02 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-02 09:45:00 | 1022.70 | 1000.68 | 982.20 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-03 09:15:00 | 1029.20 | 993.62 | 985.76 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-03 09:45:00 | 1027.15 | 1000.17 | 989.45 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-06 10:15:00 | 1018.80 | 1028.29 | 1028.90 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 109 — SELL (started 2026-02-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-06 10:15:00 | 1018.80 | 1028.29 | 1028.90 | EMA200 below EMA400 |

### Cycle 110 — BUY (started 2026-02-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-06 14:15:00 | 1039.55 | 1028.85 | 1028.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-09 09:15:00 | 1074.80 | 1038.35 | 1033.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-11 12:15:00 | 1108.40 | 1109.63 | 1093.49 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-11 12:30:00 | 1107.95 | 1109.63 | 1093.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 09:15:00 | 1078.15 | 1105.12 | 1096.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-12 10:00:00 | 1078.15 | 1105.12 | 1096.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 10:15:00 | 1084.70 | 1101.04 | 1095.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-12 10:45:00 | 1071.70 | 1101.04 | 1095.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 111 — SELL (started 2026-02-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-12 14:15:00 | 1081.95 | 1091.12 | 1092.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-13 09:15:00 | 1042.80 | 1079.68 | 1086.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-16 14:15:00 | 1050.40 | 1041.32 | 1054.27 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-16 14:15:00 | 1050.40 | 1041.32 | 1054.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 14:15:00 | 1050.40 | 1041.32 | 1054.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-16 15:00:00 | 1050.40 | 1041.32 | 1054.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 15:15:00 | 1055.00 | 1044.06 | 1054.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 09:15:00 | 1051.20 | 1044.06 | 1054.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 09:15:00 | 1077.50 | 1050.75 | 1056.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 09:45:00 | 1079.50 | 1050.75 | 1056.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 10:15:00 | 1071.85 | 1054.97 | 1057.84 | EMA400 retest candle locked (from downside) |

### Cycle 112 — BUY (started 2026-02-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-17 12:15:00 | 1068.45 | 1060.77 | 1060.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-18 09:15:00 | 1091.55 | 1072.36 | 1066.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-19 15:15:00 | 1131.90 | 1135.51 | 1118.33 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-20 09:15:00 | 1121.00 | 1135.51 | 1118.33 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 09:15:00 | 1189.30 | 1162.07 | 1147.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-24 09:30:00 | 1155.75 | 1162.07 | 1147.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 12:15:00 | 1160.05 | 1168.93 | 1161.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-25 13:00:00 | 1160.05 | 1168.93 | 1161.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 13:15:00 | 1153.35 | 1165.82 | 1160.63 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-26 09:15:00 | 1180.00 | 1163.38 | 1160.34 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-02 09:15:00 | 1147.30 | 1170.87 | 1171.94 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 113 — SELL (started 2026-03-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-02 09:15:00 | 1147.30 | 1170.87 | 1171.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-09 09:15:00 | 1063.00 | 1113.95 | 1125.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-09 14:15:00 | 1083.30 | 1081.39 | 1102.39 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-09 15:00:00 | 1083.30 | 1081.39 | 1102.39 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 09:15:00 | 1100.20 | 1085.96 | 1100.87 | EMA400 retest candle locked (from downside) |

### Cycle 114 — BUY (started 2026-03-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-11 10:15:00 | 1126.60 | 1107.48 | 1104.95 | EMA200 above EMA400 |

### Cycle 115 — SELL (started 2026-03-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-12 09:15:00 | 1083.70 | 1104.45 | 1105.53 | EMA200 below EMA400 |

### Cycle 116 — BUY (started 2026-03-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-12 13:15:00 | 1115.10 | 1106.68 | 1106.13 | EMA200 above EMA400 |

### Cycle 117 — SELL (started 2026-03-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-13 09:15:00 | 1078.40 | 1101.49 | 1103.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-16 09:15:00 | 1075.90 | 1086.13 | 1093.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-16 14:15:00 | 1081.50 | 1076.35 | 1084.91 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-16 14:15:00 | 1081.50 | 1076.35 | 1084.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 14:15:00 | 1081.50 | 1076.35 | 1084.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-16 14:45:00 | 1086.20 | 1076.35 | 1084.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 09:15:00 | 1131.20 | 1087.27 | 1088.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 09:45:00 | 1133.60 | 1087.27 | 1088.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 118 — BUY (started 2026-03-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-17 10:15:00 | 1127.80 | 1095.37 | 1091.96 | EMA200 above EMA400 |

### Cycle 119 — SELL (started 2026-03-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 15:15:00 | 1095.90 | 1109.25 | 1109.99 | EMA200 below EMA400 |

### Cycle 120 — BUY (started 2026-03-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-20 09:15:00 | 1127.30 | 1112.86 | 1111.56 | EMA200 above EMA400 |

### Cycle 121 — SELL (started 2026-03-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-20 12:15:00 | 1092.80 | 1108.50 | 1109.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-23 09:15:00 | 1033.70 | 1086.96 | 1098.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 12:15:00 | 1057.10 | 1040.26 | 1057.41 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-24 12:15:00 | 1057.10 | 1040.26 | 1057.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 12:15:00 | 1057.10 | 1040.26 | 1057.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 12:45:00 | 1056.80 | 1040.26 | 1057.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 13:15:00 | 1050.70 | 1042.35 | 1056.80 | EMA400 retest candle locked (from downside) |

### Cycle 122 — BUY (started 2026-03-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 11:15:00 | 1087.40 | 1066.06 | 1063.61 | EMA200 above EMA400 |

### Cycle 123 — SELL (started 2026-03-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 10:15:00 | 1045.60 | 1063.68 | 1064.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-27 14:15:00 | 1032.50 | 1050.76 | 1057.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 1044.70 | 1011.05 | 1026.25 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 1044.70 | 1011.05 | 1026.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 1044.70 | 1011.05 | 1026.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 10:00:00 | 1044.70 | 1011.05 | 1026.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 10:15:00 | 1022.55 | 1013.35 | 1025.91 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-02 09:15:00 | 999.65 | 1025.87 | 1028.27 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-02 13:15:00 | 1055.00 | 1031.20 | 1029.09 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 124 — BUY (started 2026-04-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-02 13:15:00 | 1055.00 | 1031.20 | 1029.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-06 11:15:00 | 1059.75 | 1042.97 | 1036.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-07 09:15:00 | 1042.85 | 1051.20 | 1043.77 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-07 09:15:00 | 1042.85 | 1051.20 | 1043.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-07 09:15:00 | 1042.85 | 1051.20 | 1043.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-07 09:45:00 | 1039.70 | 1051.20 | 1043.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-07 10:15:00 | 1036.10 | 1048.18 | 1043.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-07 11:00:00 | 1036.10 | 1048.18 | 1043.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-07 11:15:00 | 1048.50 | 1048.25 | 1043.56 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-08 09:15:00 | 1076.30 | 1045.74 | 1043.74 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2026-04-15 13:15:00 | 1183.93 | 1158.41 | 1130.84 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 125 — SELL (started 2026-04-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-24 10:15:00 | 1221.45 | 1246.54 | 1249.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-24 12:15:00 | 1220.00 | 1237.83 | 1244.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-27 09:15:00 | 1269.20 | 1237.96 | 1241.90 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-27 09:15:00 | 1269.20 | 1237.96 | 1241.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 09:15:00 | 1269.20 | 1237.96 | 1241.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 09:45:00 | 1276.50 | 1237.96 | 1241.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 126 — BUY (started 2026-04-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 10:15:00 | 1277.00 | 1245.77 | 1245.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-29 11:15:00 | 1297.20 | 1274.90 | 1268.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-30 09:15:00 | 1274.20 | 1288.23 | 1279.41 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-30 09:15:00 | 1274.20 | 1288.23 | 1279.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 09:15:00 | 1274.20 | 1288.23 | 1279.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-30 10:00:00 | 1274.20 | 1288.23 | 1279.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 10:15:00 | 1265.75 | 1283.74 | 1278.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-30 11:00:00 | 1265.75 | 1283.74 | 1278.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 13:15:00 | 1281.75 | 1281.59 | 1278.40 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-30 14:30:00 | 1289.15 | 1281.82 | 1278.79 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-04 09:15:00 | 1325.20 | 1281.86 | 1279.09 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-05 11:00:00 | 1289.30 | 1299.31 | 1293.58 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-05 12:15:00 | 1289.00 | 1297.23 | 1293.15 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-05-05 14:15:00 | 1271.70 | 1287.74 | 1289.49 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 127 — SELL (started 2026-05-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-05 14:15:00 | 1271.70 | 1287.74 | 1289.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-05-06 12:15:00 | 1263.60 | 1277.40 | 1283.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-05-06 14:15:00 | 1286.90 | 1277.75 | 1282.46 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-06 14:15:00 | 1286.90 | 1277.75 | 1282.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 14:15:00 | 1286.90 | 1277.75 | 1282.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-06 15:00:00 | 1286.90 | 1277.75 | 1282.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 15:15:00 | 1292.00 | 1280.60 | 1283.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-07 09:15:00 | 1291.60 | 1280.60 | 1283.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-07 10:15:00 | 1289.00 | 1282.65 | 1283.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-07 11:00:00 | 1289.00 | 1282.65 | 1283.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-07 11:15:00 | 1288.10 | 1283.74 | 1284.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-07 11:30:00 | 1286.00 | 1283.74 | 1284.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-07 12:15:00 | 1279.90 | 1282.97 | 1283.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-07 12:30:00 | 1289.90 | 1282.97 | 1283.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-07 13:15:00 | 1280.70 | 1282.52 | 1283.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-07 13:45:00 | 1283.80 | 1282.52 | 1283.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-07 14:15:00 | 1278.90 | 1281.79 | 1283.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-07 14:30:00 | 1283.00 | 1281.79 | 1283.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-07 15:15:00 | 1288.50 | 1283.13 | 1283.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-08 09:15:00 | 1274.70 | 1283.13 | 1283.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 09:15:00 | 1260.50 | 1278.61 | 1281.49 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-08 11:15:00 | 1251.90 | 1275.09 | 1279.63 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-08 15:00:00 | 1253.90 | 1266.05 | 1273.45 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-05-13 11:30:00 | 1030.20 | 2024-05-16 10:15:00 | 1133.22 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-05-13 12:00:00 | 1033.45 | 2024-05-21 09:15:00 | 1136.80 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-06-18 09:30:00 | 1498.00 | 2024-06-19 15:15:00 | 1430.00 | STOP_HIT | 1.00 | -4.54% |
| BUY | retest2 | 2024-06-18 10:15:00 | 1500.00 | 2024-06-19 15:15:00 | 1430.00 | STOP_HIT | 1.00 | -4.67% |
| BUY | retest2 | 2024-07-01 15:00:00 | 1552.00 | 2024-07-02 12:15:00 | 1515.00 | STOP_HIT | 1.00 | -2.38% |
| BUY | retest2 | 2024-07-02 14:45:00 | 1563.55 | 2024-07-04 11:15:00 | 1525.00 | STOP_HIT | 1.00 | -2.47% |
| BUY | retest2 | 2024-07-03 09:30:00 | 1555.00 | 2024-07-04 11:15:00 | 1525.00 | STOP_HIT | 1.00 | -1.93% |
| BUY | retest2 | 2024-07-03 15:15:00 | 1554.95 | 2024-07-04 11:15:00 | 1525.00 | STOP_HIT | 1.00 | -1.93% |
| SELL | retest2 | 2024-07-08 11:15:00 | 1484.95 | 2024-07-09 10:15:00 | 1535.00 | STOP_HIT | 1.00 | -3.37% |
| SELL | retest2 | 2024-07-08 12:00:00 | 1484.05 | 2024-07-09 10:15:00 | 1535.00 | STOP_HIT | 1.00 | -3.43% |
| SELL | retest2 | 2024-07-08 13:00:00 | 1490.00 | 2024-07-09 10:15:00 | 1535.00 | STOP_HIT | 1.00 | -3.02% |
| SELL | retest2 | 2024-07-08 14:15:00 | 1485.00 | 2024-07-09 10:15:00 | 1535.00 | STOP_HIT | 1.00 | -3.37% |
| SELL | retest2 | 2024-07-12 14:00:00 | 1391.15 | 2024-07-16 09:15:00 | 1494.00 | STOP_HIT | 1.00 | -7.39% |
| SELL | retest2 | 2024-07-15 12:30:00 | 1400.00 | 2024-07-16 09:15:00 | 1494.00 | STOP_HIT | 1.00 | -6.71% |
| BUY | retest2 | 2024-07-24 09:15:00 | 1561.50 | 2024-07-26 09:15:00 | 1717.65 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-07-24 13:15:00 | 1569.00 | 2024-07-30 13:15:00 | 1645.00 | STOP_HIT | 1.00 | 4.84% |
| BUY | retest2 | 2024-07-24 13:45:00 | 1576.00 | 2024-07-30 13:15:00 | 1645.00 | STOP_HIT | 1.00 | 4.38% |
| BUY | retest2 | 2024-08-02 12:45:00 | 1750.00 | 2024-08-05 10:15:00 | 1625.80 | STOP_HIT | 1.00 | -7.10% |
| BUY | retest2 | 2024-08-02 14:45:00 | 1724.00 | 2024-08-05 10:15:00 | 1625.80 | STOP_HIT | 1.00 | -5.70% |
| BUY | retest2 | 2024-08-21 09:15:00 | 1726.15 | 2024-08-21 11:15:00 | 1676.10 | STOP_HIT | 1.00 | -2.90% |
| BUY | retest2 | 2024-08-26 14:30:00 | 1710.00 | 2024-08-28 10:15:00 | 1674.50 | STOP_HIT | 1.00 | -2.08% |
| BUY | retest2 | 2024-08-27 12:15:00 | 1704.45 | 2024-08-28 10:15:00 | 1674.50 | STOP_HIT | 1.00 | -1.76% |
| BUY | retest2 | 2024-08-27 13:15:00 | 1703.00 | 2024-08-28 10:15:00 | 1674.50 | STOP_HIT | 1.00 | -1.67% |
| BUY | retest2 | 2024-08-27 15:15:00 | 1710.00 | 2024-08-28 10:15:00 | 1674.50 | STOP_HIT | 1.00 | -2.08% |
| SELL | retest2 | 2024-08-30 13:30:00 | 1621.90 | 2024-09-02 12:15:00 | 1669.90 | STOP_HIT | 1.00 | -2.96% |
| SELL | retest2 | 2024-08-30 14:15:00 | 1620.10 | 2024-09-02 12:15:00 | 1669.90 | STOP_HIT | 1.00 | -3.07% |
| SELL | retest2 | 2024-08-30 15:15:00 | 1630.00 | 2024-09-02 12:15:00 | 1669.90 | STOP_HIT | 1.00 | -2.45% |
| BUY | retest2 | 2024-09-10 15:00:00 | 1640.65 | 2024-09-12 14:15:00 | 1602.05 | STOP_HIT | 1.00 | -2.35% |
| BUY | retest2 | 2024-09-11 09:45:00 | 1648.85 | 2024-09-12 14:15:00 | 1602.05 | STOP_HIT | 1.00 | -2.84% |
| SELL | retest2 | 2024-09-16 09:15:00 | 1589.10 | 2024-09-19 12:15:00 | 1511.50 | PARTIAL | 0.50 | 4.88% |
| SELL | retest2 | 2024-09-16 11:45:00 | 1591.05 | 2024-09-19 12:15:00 | 1516.39 | PARTIAL | 0.50 | 4.69% |
| SELL | retest2 | 2024-09-16 12:45:00 | 1596.20 | 2024-09-19 14:15:00 | 1509.64 | PARTIAL | 0.50 | 5.42% |
| SELL | retest2 | 2024-09-16 14:15:00 | 1585.00 | 2024-09-19 14:15:00 | 1505.75 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-09-16 09:15:00 | 1589.10 | 2024-09-20 11:15:00 | 1541.30 | STOP_HIT | 0.50 | 3.01% |
| SELL | retest2 | 2024-09-16 11:45:00 | 1591.05 | 2024-09-20 11:15:00 | 1541.30 | STOP_HIT | 0.50 | 3.13% |
| SELL | retest2 | 2024-09-16 12:45:00 | 1596.20 | 2024-09-20 11:15:00 | 1541.30 | STOP_HIT | 0.50 | 3.44% |
| SELL | retest2 | 2024-09-16 14:15:00 | 1585.00 | 2024-09-20 11:15:00 | 1541.30 | STOP_HIT | 0.50 | 2.76% |
| SELL | retest2 | 2024-09-17 11:00:00 | 1567.55 | 2024-09-20 14:15:00 | 1489.17 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-09-17 12:30:00 | 1562.05 | 2024-09-20 14:15:00 | 1483.95 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-09-18 12:15:00 | 1569.95 | 2024-09-20 14:15:00 | 1491.45 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-09-18 13:00:00 | 1569.45 | 2024-09-20 14:15:00 | 1490.98 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-09-19 10:15:00 | 1568.65 | 2024-09-20 14:15:00 | 1490.22 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-09-17 11:00:00 | 1567.55 | 2024-09-20 15:15:00 | 1570.00 | STOP_HIT | 0.50 | -0.16% |
| SELL | retest2 | 2024-09-17 12:30:00 | 1562.05 | 2024-09-20 15:15:00 | 1570.00 | STOP_HIT | 0.50 | -0.51% |
| SELL | retest2 | 2024-09-18 12:15:00 | 1569.95 | 2024-09-20 15:15:00 | 1570.00 | STOP_HIT | 0.50 | -0.00% |
| SELL | retest2 | 2024-09-18 13:00:00 | 1569.45 | 2024-09-20 15:15:00 | 1570.00 | STOP_HIT | 0.50 | -0.04% |
| SELL | retest2 | 2024-09-19 10:15:00 | 1568.65 | 2024-09-20 15:15:00 | 1570.00 | STOP_HIT | 0.50 | -0.09% |
| BUY | retest2 | 2024-09-27 13:30:00 | 1616.15 | 2024-10-01 12:15:00 | 1584.00 | STOP_HIT | 1.00 | -1.99% |
| BUY | retest2 | 2024-09-30 09:15:00 | 1615.30 | 2024-10-01 12:15:00 | 1584.00 | STOP_HIT | 1.00 | -1.94% |
| BUY | retest2 | 2024-09-30 09:45:00 | 1616.95 | 2024-10-01 12:15:00 | 1584.00 | STOP_HIT | 1.00 | -2.04% |
| BUY | retest2 | 2024-10-16 09:30:00 | 1785.00 | 2024-10-17 09:15:00 | 1734.90 | STOP_HIT | 1.00 | -2.81% |
| BUY | retest2 | 2024-10-17 10:30:00 | 1780.15 | 2024-10-18 09:15:00 | 1722.40 | STOP_HIT | 1.00 | -3.24% |
| SELL | retest2 | 2024-10-28 12:45:00 | 1474.00 | 2024-10-30 12:15:00 | 1539.55 | STOP_HIT | 1.00 | -4.45% |
| SELL | retest2 | 2024-10-29 09:45:00 | 1463.75 | 2024-10-30 12:15:00 | 1539.55 | STOP_HIT | 1.00 | -5.18% |
| SELL | retest2 | 2024-10-29 14:45:00 | 1472.75 | 2024-10-30 12:15:00 | 1539.55 | STOP_HIT | 1.00 | -4.54% |
| BUY | retest1 | 2024-11-01 18:00:00 | 1599.00 | 2024-11-04 09:15:00 | 1538.30 | STOP_HIT | 1.00 | -3.80% |
| BUY | retest1 | 2024-11-01 18:30:00 | 1597.90 | 2024-11-04 09:15:00 | 1538.30 | STOP_HIT | 1.00 | -3.73% |
| BUY | retest2 | 2024-11-04 11:45:00 | 1558.75 | 2024-11-11 09:15:00 | 1591.80 | STOP_HIT | 1.00 | 2.12% |
| BUY | retest2 | 2024-11-05 12:15:00 | 1573.60 | 2024-11-11 09:15:00 | 1591.80 | STOP_HIT | 1.00 | 1.16% |
| SELL | retest2 | 2024-11-12 12:30:00 | 1583.50 | 2024-11-13 09:15:00 | 1425.15 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-12-13 09:15:00 | 1425.50 | 2024-12-13 12:15:00 | 1449.95 | STOP_HIT | 1.00 | -1.72% |
| BUY | retest2 | 2024-12-18 13:00:00 | 1544.05 | 2024-12-23 09:15:00 | 1698.46 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-12-18 13:30:00 | 1544.95 | 2024-12-23 09:15:00 | 1699.45 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-12-18 14:00:00 | 1551.95 | 2024-12-23 09:15:00 | 1707.15 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-12-19 11:15:00 | 1569.00 | 2024-12-26 11:15:00 | 1564.45 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest2 | 2024-12-20 10:30:00 | 1595.00 | 2024-12-26 11:15:00 | 1564.45 | STOP_HIT | 1.00 | -1.92% |
| BUY | retest2 | 2024-12-20 15:00:00 | 1620.00 | 2024-12-26 11:15:00 | 1564.45 | STOP_HIT | 1.00 | -3.43% |
| SELL | retest2 | 2024-12-31 09:15:00 | 1574.10 | 2025-01-01 09:15:00 | 1632.95 | STOP_HIT | 1.00 | -3.74% |
| SELL | retest2 | 2025-01-16 11:15:00 | 1328.50 | 2025-01-20 09:15:00 | 1262.08 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-16 11:15:00 | 1328.50 | 2025-01-20 11:15:00 | 1318.00 | STOP_HIT | 0.50 | 0.79% |
| SELL | retest2 | 2025-01-29 11:15:00 | 1071.75 | 2025-01-29 15:15:00 | 1109.95 | STOP_HIT | 1.00 | -3.56% |
| SELL | retest2 | 2025-01-30 09:15:00 | 1008.40 | 2025-01-30 09:15:00 | 957.98 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-30 09:15:00 | 1008.40 | 2025-01-30 12:15:00 | 1074.15 | STOP_HIT | 0.50 | -6.52% |
| SELL | retest2 | 2025-01-30 13:15:00 | 1070.20 | 2025-02-01 09:15:00 | 1091.00 | STOP_HIT | 1.00 | -1.94% |
| SELL | retest2 | 2025-01-31 09:15:00 | 1052.90 | 2025-02-01 09:15:00 | 1091.00 | STOP_HIT | 1.00 | -3.62% |
| SELL | retest2 | 2025-02-14 10:15:00 | 927.20 | 2025-02-18 09:15:00 | 880.84 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-14 15:00:00 | 925.10 | 2025-02-18 09:15:00 | 878.85 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-14 10:15:00 | 927.20 | 2025-02-18 14:15:00 | 907.25 | STOP_HIT | 0.50 | 2.15% |
| SELL | retest2 | 2025-02-14 15:00:00 | 925.10 | 2025-02-18 14:15:00 | 907.25 | STOP_HIT | 0.50 | 1.93% |
| BUY | retest1 | 2025-02-21 09:15:00 | 1027.35 | 2025-02-24 09:15:00 | 986.40 | STOP_HIT | 1.00 | -3.99% |
| BUY | retest1 | 2025-02-21 11:30:00 | 1006.45 | 2025-02-24 09:15:00 | 986.40 | STOP_HIT | 1.00 | -1.99% |
| BUY | retest1 | 2025-02-21 14:45:00 | 1003.35 | 2025-02-24 09:15:00 | 986.40 | STOP_HIT | 1.00 | -1.69% |
| SELL | retest2 | 2025-02-28 11:00:00 | 916.55 | 2025-02-28 14:15:00 | 979.15 | STOP_HIT | 1.00 | -6.83% |
| SELL | retest2 | 2025-03-03 09:45:00 | 918.00 | 2025-03-05 10:15:00 | 960.00 | STOP_HIT | 1.00 | -4.58% |
| SELL | retest2 | 2025-03-03 11:45:00 | 914.25 | 2025-03-05 10:15:00 | 960.00 | STOP_HIT | 1.00 | -5.00% |
| BUY | retest2 | 2025-03-07 14:30:00 | 969.55 | 2025-03-07 15:15:00 | 947.45 | STOP_HIT | 1.00 | -2.28% |
| SELL | retest2 | 2025-03-12 10:45:00 | 899.35 | 2025-03-12 15:15:00 | 935.00 | STOP_HIT | 1.00 | -3.96% |
| SELL | retest2 | 2025-03-13 09:15:00 | 898.05 | 2025-03-17 10:15:00 | 931.60 | STOP_HIT | 1.00 | -3.74% |
| SELL | retest2 | 2025-03-13 13:15:00 | 900.00 | 2025-03-17 10:15:00 | 931.60 | STOP_HIT | 1.00 | -3.51% |
| SELL | retest2 | 2025-03-13 15:15:00 | 900.00 | 2025-03-17 10:15:00 | 931.60 | STOP_HIT | 1.00 | -3.51% |
| SELL | retest2 | 2025-03-27 12:15:00 | 1003.40 | 2025-03-27 14:15:00 | 1016.50 | STOP_HIT | 1.00 | -1.31% |
| BUY | retest2 | 2025-04-23 12:45:00 | 1116.50 | 2025-04-25 09:15:00 | 1076.50 | STOP_HIT | 1.00 | -3.58% |
| BUY | retest2 | 2025-05-20 11:15:00 | 1259.20 | 2025-05-20 13:15:00 | 1241.80 | STOP_HIT | 1.00 | -1.38% |
| BUY | retest2 | 2025-05-28 09:15:00 | 1359.00 | 2025-05-30 09:15:00 | 1494.90 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-06-23 09:15:00 | 1475.10 | 2025-06-23 11:15:00 | 1490.90 | STOP_HIT | 1.00 | -1.07% |
| BUY | retest1 | 2025-06-25 09:15:00 | 1536.70 | 2025-06-25 12:15:00 | 1504.00 | STOP_HIT | 1.00 | -2.13% |
| BUY | retest1 | 2025-06-25 11:30:00 | 1535.00 | 2025-06-25 12:15:00 | 1504.00 | STOP_HIT | 1.00 | -2.02% |
| BUY | retest1 | 2025-06-25 12:15:00 | 1531.70 | 2025-06-25 12:15:00 | 1504.00 | STOP_HIT | 1.00 | -1.81% |
| BUY | retest2 | 2025-07-01 12:45:00 | 1596.80 | 2025-07-03 09:15:00 | 1575.00 | STOP_HIT | 1.00 | -1.37% |
| BUY | retest2 | 2025-07-01 13:15:00 | 1602.00 | 2025-07-03 09:15:00 | 1575.00 | STOP_HIT | 1.00 | -1.69% |
| SELL | retest2 | 2025-07-16 09:15:00 | 1561.80 | 2025-07-23 13:15:00 | 1483.71 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-16 09:15:00 | 1561.80 | 2025-07-28 13:15:00 | 1405.62 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-08-06 10:00:00 | 1409.00 | 2025-08-13 09:15:00 | 1445.70 | STOP_HIT | 1.00 | -2.60% |
| BUY | retest2 | 2025-08-19 10:15:00 | 1450.40 | 2025-08-29 09:15:00 | 1511.10 | STOP_HIT | 1.00 | 4.19% |
| BUY | retest2 | 2025-08-19 13:45:00 | 1451.60 | 2025-08-29 09:15:00 | 1511.10 | STOP_HIT | 1.00 | 4.10% |
| BUY | retest2 | 2025-08-19 15:00:00 | 1453.10 | 2025-08-29 09:15:00 | 1511.10 | STOP_HIT | 1.00 | 3.99% |
| SELL | retest2 | 2025-09-08 11:15:00 | 1526.90 | 2025-09-12 09:15:00 | 1450.56 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-08 11:15:00 | 1526.90 | 2025-09-15 09:15:00 | 1451.80 | STOP_HIT | 0.50 | 4.92% |
| SELL | retest2 | 2025-10-09 09:30:00 | 1335.90 | 2025-10-10 10:15:00 | 1349.00 | STOP_HIT | 1.00 | -0.98% |
| SELL | retest2 | 2025-10-09 10:00:00 | 1333.80 | 2025-10-10 10:15:00 | 1349.00 | STOP_HIT | 1.00 | -1.14% |
| BUY | retest2 | 2025-10-30 14:30:00 | 1348.70 | 2025-10-31 14:15:00 | 1316.50 | STOP_HIT | 1.00 | -2.39% |
| BUY | retest2 | 2025-10-31 11:15:00 | 1341.10 | 2025-10-31 14:15:00 | 1316.50 | STOP_HIT | 1.00 | -1.83% |
| BUY | retest2 | 2025-10-31 11:45:00 | 1341.90 | 2025-10-31 14:15:00 | 1316.50 | STOP_HIT | 1.00 | -1.89% |
| SELL | retest2 | 2025-11-10 11:45:00 | 1282.50 | 2025-11-13 09:15:00 | 1218.38 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-11 09:15:00 | 1272.90 | 2025-11-13 09:15:00 | 1209.26 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-10 11:45:00 | 1282.50 | 2025-11-14 12:15:00 | 1241.50 | STOP_HIT | 0.50 | 3.20% |
| SELL | retest2 | 2025-11-11 09:15:00 | 1272.90 | 2025-11-14 12:15:00 | 1241.50 | STOP_HIT | 0.50 | 2.47% |
| BUY | retest1 | 2025-11-28 10:15:00 | 1210.20 | 2025-12-01 10:15:00 | 1191.50 | STOP_HIT | 1.00 | -1.55% |
| BUY | retest1 | 2025-11-28 11:15:00 | 1209.50 | 2025-12-01 10:15:00 | 1191.50 | STOP_HIT | 1.00 | -1.49% |
| BUY | retest1 | 2025-11-28 11:45:00 | 1211.50 | 2025-12-01 10:15:00 | 1191.50 | STOP_HIT | 1.00 | -1.65% |
| BUY | retest1 | 2025-11-28 15:00:00 | 1208.80 | 2025-12-01 10:15:00 | 1191.50 | STOP_HIT | 1.00 | -1.43% |
| SELL | retest1 | 2025-12-04 09:15:00 | 1129.00 | 2025-12-08 12:15:00 | 1072.55 | PARTIAL | 0.50 | 5.00% |
| SELL | retest1 | 2025-12-04 09:15:00 | 1129.00 | 2025-12-09 10:15:00 | 1091.00 | STOP_HIT | 0.50 | 3.37% |
| SELL | retest2 | 2025-12-09 14:15:00 | 1082.30 | 2025-12-09 15:15:00 | 1098.90 | STOP_HIT | 1.00 | -1.53% |
| SELL | retest2 | 2025-12-10 12:30:00 | 1082.50 | 2025-12-11 13:15:00 | 1100.00 | STOP_HIT | 1.00 | -1.62% |
| SELL | retest2 | 2025-12-10 13:45:00 | 1080.00 | 2025-12-11 13:15:00 | 1100.00 | STOP_HIT | 1.00 | -1.85% |
| SELL | retest2 | 2025-12-11 09:30:00 | 1080.00 | 2025-12-11 13:15:00 | 1100.00 | STOP_HIT | 1.00 | -1.85% |
| BUY | retest2 | 2025-12-15 10:15:00 | 1104.80 | 2025-12-16 11:15:00 | 1088.60 | STOP_HIT | 1.00 | -1.47% |
| SELL | retest2 | 2025-12-19 14:15:00 | 1066.80 | 2025-12-19 14:15:00 | 1090.00 | STOP_HIT | 1.00 | -2.17% |
| SELL | retest2 | 2026-01-01 12:15:00 | 1065.10 | 2026-01-02 10:15:00 | 1097.10 | STOP_HIT | 1.00 | -3.00% |
| SELL | retest2 | 2026-01-01 13:00:00 | 1062.10 | 2026-01-02 10:15:00 | 1097.10 | STOP_HIT | 1.00 | -3.30% |
| SELL | retest2 | 2026-01-16 12:15:00 | 982.80 | 2026-01-20 11:15:00 | 933.66 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-16 12:15:00 | 982.80 | 2026-01-22 09:15:00 | 919.80 | STOP_HIT | 0.50 | 6.41% |
| BUY | retest2 | 2026-02-02 09:45:00 | 1022.70 | 2026-02-06 10:15:00 | 1018.80 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest2 | 2026-02-03 09:15:00 | 1029.20 | 2026-02-06 10:15:00 | 1018.80 | STOP_HIT | 1.00 | -1.01% |
| BUY | retest2 | 2026-02-03 09:45:00 | 1027.15 | 2026-02-06 10:15:00 | 1018.80 | STOP_HIT | 1.00 | -0.81% |
| BUY | retest2 | 2026-02-26 09:15:00 | 1180.00 | 2026-03-02 09:15:00 | 1147.30 | STOP_HIT | 1.00 | -2.77% |
| SELL | retest2 | 2026-04-02 09:15:00 | 999.65 | 2026-04-02 13:15:00 | 1055.00 | STOP_HIT | 1.00 | -5.54% |
| BUY | retest2 | 2026-04-08 09:15:00 | 1076.30 | 2026-04-15 13:15:00 | 1183.93 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-30 14:30:00 | 1289.15 | 2026-05-05 14:15:00 | 1271.70 | STOP_HIT | 1.00 | -1.35% |
| BUY | retest2 | 2026-05-04 09:15:00 | 1325.20 | 2026-05-05 14:15:00 | 1271.70 | STOP_HIT | 1.00 | -4.04% |
| BUY | retest2 | 2026-05-05 11:00:00 | 1289.30 | 2026-05-05 14:15:00 | 1271.70 | STOP_HIT | 1.00 | -1.37% |
| BUY | retest2 | 2026-05-05 12:15:00 | 1289.00 | 2026-05-05 14:15:00 | 1271.70 | STOP_HIT | 1.00 | -1.34% |
