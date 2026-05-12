# Concord Biotech Ltd. (CONCORDBIO)

## Backtest Summary

- **Window:** 2024-03-13 09:15:00 → 2026-05-11 15:15:00 (3717 bars)
- **Last close:** 1205.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 133 |
| ALERT1 | 89 |
| ALERT2 | 89 |
| ALERT2_SKIP | 47 |
| ALERT3 | 236 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 4 |
| ENTRY2 | 102 |
| PARTIAL | 30 |
| TARGET_HIT | 9 |
| STOP_HIT | 99 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 134 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 73 / 61
- **Target hits / Stop hits / Partials:** 9 / 97 / 28
- **Avg / median % per leg:** 1.74% / 1.94%
- **Sum % (uncompounded):** 233.69%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 41 | 14 | 34.1% | 5 | 36 | 0 | 0.71% | 29.3% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 41 | 14 | 34.1% | 5 | 36 | 0 | 0.71% | 29.3% |
| SELL (all) | 93 | 59 | 63.4% | 4 | 61 | 28 | 2.20% | 204.4% |
| SELL @ 2nd Alert (retest1) | 4 | 1 | 25.0% | 0 | 4 | 0 | -0.90% | -3.6% |
| SELL @ 3rd Alert (retest2) | 89 | 58 | 65.2% | 4 | 57 | 28 | 2.34% | 208.0% |
| retest1 (combined) | 4 | 1 | 25.0% | 0 | 4 | 0 | -0.90% | -3.6% |
| retest2 (combined) | 130 | 72 | 55.4% | 9 | 93 | 28 | 1.83% | 237.3% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-05-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-23 14:15:00 | 1470.00 | 1459.42 | 1458.99 | EMA200 above EMA400 |

### Cycle 2 — SELL (started 2024-05-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-23 15:15:00 | 1440.00 | 1455.53 | 1457.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-27 09:15:00 | 1418.00 | 1442.64 | 1449.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-27 10:15:00 | 1443.90 | 1442.89 | 1448.57 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-27 11:00:00 | 1443.90 | 1442.89 | 1448.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-27 11:15:00 | 1454.70 | 1445.25 | 1449.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-27 12:00:00 | 1454.70 | 1445.25 | 1449.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-27 12:15:00 | 1438.35 | 1443.87 | 1448.14 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-27 14:15:00 | 1431.85 | 1442.49 | 1447.13 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-28 09:15:00 | 1392.00 | 1442.14 | 1446.15 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-28 09:15:00 | 1360.26 | 1431.29 | 1440.85 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-05-29 09:15:00 | 1493.00 | 1426.44 | 1430.59 | SL hit (close>ema200) qty=0.50 sl=1426.44 alert=retest2 |

### Cycle 3 — BUY (started 2024-05-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-29 10:15:00 | 1493.10 | 1439.77 | 1436.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-29 11:15:00 | 1501.20 | 1452.06 | 1442.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-29 14:15:00 | 1445.95 | 1466.87 | 1452.95 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-29 14:15:00 | 1445.95 | 1466.87 | 1452.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-29 14:15:00 | 1445.95 | 1466.87 | 1452.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-29 14:45:00 | 1447.25 | 1466.87 | 1452.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-29 15:15:00 | 1450.00 | 1463.50 | 1452.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-30 09:15:00 | 1466.25 | 1463.50 | 1452.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-30 09:15:00 | 1446.00 | 1460.00 | 1452.07 | EMA400 retest candle locked (from upside) |

### Cycle 4 — SELL (started 2024-05-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-31 10:15:00 | 1429.30 | 1446.08 | 1448.33 | EMA200 below EMA400 |

### Cycle 5 — BUY (started 2024-06-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-03 09:15:00 | 1471.40 | 1452.34 | 1449.94 | EMA200 above EMA400 |

### Cycle 6 — SELL (started 2024-06-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-03 13:15:00 | 1437.10 | 1447.20 | 1448.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-04 10:15:00 | 1424.80 | 1443.00 | 1445.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-04 13:15:00 | 1427.20 | 1425.45 | 1435.66 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-04 13:15:00 | 1427.20 | 1425.45 | 1435.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 13:15:00 | 1427.20 | 1425.45 | 1435.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-04 13:45:00 | 1429.75 | 1425.45 | 1435.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 09:15:00 | 1438.00 | 1427.71 | 1434.08 | EMA400 retest candle locked (from downside) |

### Cycle 7 — BUY (started 2024-06-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-05 12:15:00 | 1458.25 | 1440.92 | 1439.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-05 14:15:00 | 1487.00 | 1451.94 | 1444.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-06 11:15:00 | 1465.95 | 1467.76 | 1455.56 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-06 12:00:00 | 1465.95 | 1467.76 | 1455.56 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 14:15:00 | 1469.45 | 1467.55 | 1458.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-06 14:30:00 | 1457.50 | 1467.55 | 1458.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 15:15:00 | 1465.00 | 1467.04 | 1459.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-07 09:15:00 | 1469.95 | 1467.04 | 1459.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-07 09:15:00 | 1469.95 | 1467.62 | 1460.02 | EMA400 retest candle locked (from upside) |

### Cycle 8 — SELL (started 2024-06-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-10 12:15:00 | 1448.05 | 1458.74 | 1459.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-11 14:15:00 | 1445.05 | 1449.81 | 1453.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-11 15:15:00 | 1449.95 | 1449.84 | 1452.75 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-11 15:15:00 | 1449.95 | 1449.84 | 1452.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-11 15:15:00 | 1449.95 | 1449.84 | 1452.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-12 09:15:00 | 1450.10 | 1449.84 | 1452.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-12 09:15:00 | 1448.55 | 1449.58 | 1452.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-12 09:30:00 | 1452.00 | 1449.58 | 1452.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-13 09:15:00 | 1444.50 | 1439.62 | 1444.83 | EMA400 retest candle locked (from downside) |

### Cycle 9 — BUY (started 2024-06-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-14 10:15:00 | 1449.95 | 1446.18 | 1445.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-14 11:15:00 | 1470.95 | 1451.13 | 1448.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-19 09:15:00 | 1554.80 | 1555.58 | 1521.35 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-19 09:30:00 | 1559.75 | 1555.58 | 1521.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-21 14:15:00 | 1556.25 | 1568.25 | 1561.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-21 15:00:00 | 1556.25 | 1568.25 | 1561.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-21 15:15:00 | 1562.50 | 1567.10 | 1561.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-24 09:15:00 | 1545.15 | 1567.10 | 1561.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-24 09:15:00 | 1541.00 | 1561.88 | 1559.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-24 09:30:00 | 1544.35 | 1561.88 | 1559.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 10 — SELL (started 2024-06-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-24 10:15:00 | 1540.10 | 1557.52 | 1558.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-24 11:15:00 | 1532.00 | 1552.42 | 1555.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-25 14:15:00 | 1537.15 | 1520.65 | 1532.06 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-25 14:15:00 | 1537.15 | 1520.65 | 1532.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-25 14:15:00 | 1537.15 | 1520.65 | 1532.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-25 15:00:00 | 1537.15 | 1520.65 | 1532.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-25 15:15:00 | 1538.60 | 1524.24 | 1532.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-26 09:15:00 | 1540.70 | 1524.24 | 1532.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-26 10:15:00 | 1531.95 | 1527.18 | 1532.61 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-26 12:00:00 | 1528.90 | 1527.52 | 1532.28 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-27 14:15:00 | 1559.05 | 1522.74 | 1524.27 | SL hit (close>static) qty=1.00 sl=1540.00 alert=retest2 |

### Cycle 11 — BUY (started 2024-06-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-27 15:15:00 | 1566.00 | 1531.40 | 1528.06 | EMA200 above EMA400 |

### Cycle 12 — SELL (started 2024-07-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-01 12:15:00 | 1527.70 | 1532.13 | 1532.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-02 10:15:00 | 1519.80 | 1527.81 | 1530.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-02 13:15:00 | 1542.05 | 1528.20 | 1529.51 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-02 13:15:00 | 1542.05 | 1528.20 | 1529.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-02 13:15:00 | 1542.05 | 1528.20 | 1529.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-02 14:00:00 | 1542.05 | 1528.20 | 1529.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 13 — BUY (started 2024-07-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-02 14:15:00 | 1541.95 | 1530.95 | 1530.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-02 15:15:00 | 1545.00 | 1533.76 | 1531.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-05 13:15:00 | 1675.90 | 1676.31 | 1637.77 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-05 14:00:00 | 1675.90 | 1676.31 | 1637.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-09 15:15:00 | 1699.00 | 1694.74 | 1679.11 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-10 09:15:00 | 1711.35 | 1694.74 | 1679.11 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-10 11:15:00 | 1708.60 | 1697.46 | 1683.18 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-12 09:15:00 | 1702.70 | 1693.58 | 1690.90 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-12 11:15:00 | 1673.15 | 1691.10 | 1690.58 | SL hit (close<static) qty=1.00 sl=1675.00 alert=retest2 |

### Cycle 14 — SELL (started 2024-07-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-16 13:15:00 | 1688.85 | 1698.09 | 1699.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-19 09:15:00 | 1668.05 | 1690.12 | 1693.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-19 11:15:00 | 1688.10 | 1686.49 | 1691.46 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-19 11:15:00 | 1688.10 | 1686.49 | 1691.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-19 11:15:00 | 1688.10 | 1686.49 | 1691.46 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-19 15:15:00 | 1662.95 | 1683.70 | 1688.78 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-22 12:15:00 | 1702.50 | 1688.18 | 1688.71 | SL hit (close>static) qty=1.00 sl=1694.30 alert=retest2 |

### Cycle 15 — BUY (started 2024-07-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-23 13:15:00 | 1711.15 | 1686.73 | 1684.79 | EMA200 above EMA400 |

### Cycle 16 — SELL (started 2024-07-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-29 11:15:00 | 1696.10 | 1700.99 | 1701.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-29 12:15:00 | 1692.60 | 1699.31 | 1700.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-31 13:15:00 | 1654.45 | 1650.23 | 1663.78 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-07-31 14:00:00 | 1654.45 | 1650.23 | 1663.78 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-06 15:15:00 | 1542.00 | 1533.80 | 1554.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-07 09:15:00 | 1561.60 | 1533.80 | 1554.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 09:15:00 | 1571.65 | 1541.37 | 1555.93 | EMA400 retest candle locked (from downside) |

### Cycle 17 — BUY (started 2024-08-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-07 14:15:00 | 1580.00 | 1566.00 | 1564.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-08 09:15:00 | 1597.60 | 1574.54 | 1568.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-09 10:15:00 | 1589.50 | 1590.39 | 1581.76 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-09 11:00:00 | 1589.50 | 1590.39 | 1581.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-09 12:15:00 | 1575.80 | 1587.14 | 1581.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-09 13:00:00 | 1575.80 | 1587.14 | 1581.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-09 13:15:00 | 1568.30 | 1583.37 | 1580.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-09 13:30:00 | 1572.15 | 1583.37 | 1580.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 18 — SELL (started 2024-08-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-09 14:15:00 | 1555.10 | 1577.72 | 1578.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-12 09:15:00 | 1501.05 | 1559.07 | 1569.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-13 13:15:00 | 1537.35 | 1521.38 | 1534.59 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-13 13:15:00 | 1537.35 | 1521.38 | 1534.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-13 13:15:00 | 1537.35 | 1521.38 | 1534.59 | EMA400 retest candle locked (from downside) |

### Cycle 19 — BUY (started 2024-08-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-14 10:15:00 | 1559.00 | 1544.51 | 1542.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-16 09:15:00 | 1595.70 | 1567.36 | 1556.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-20 11:15:00 | 1627.10 | 1633.51 | 1617.91 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-20 12:00:00 | 1627.10 | 1633.51 | 1617.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-21 11:15:00 | 1631.25 | 1636.27 | 1627.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-21 11:45:00 | 1622.00 | 1636.27 | 1627.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-21 12:15:00 | 1630.00 | 1635.01 | 1627.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-21 12:30:00 | 1626.40 | 1635.01 | 1627.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-21 13:15:00 | 1621.35 | 1632.28 | 1627.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-21 14:00:00 | 1621.35 | 1632.28 | 1627.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-21 14:15:00 | 1608.00 | 1627.42 | 1625.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-21 15:00:00 | 1608.00 | 1627.42 | 1625.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 20 — SELL (started 2024-08-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-22 09:15:00 | 1608.20 | 1622.36 | 1623.36 | EMA200 below EMA400 |

### Cycle 21 — BUY (started 2024-08-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-23 14:15:00 | 1652.80 | 1617.24 | 1616.54 | EMA200 above EMA400 |

### Cycle 22 — SELL (started 2024-08-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-26 09:15:00 | 1599.30 | 1616.23 | 1616.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-27 13:15:00 | 1585.80 | 1598.94 | 1605.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-28 09:15:00 | 1612.00 | 1600.34 | 1604.21 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-28 09:15:00 | 1612.00 | 1600.34 | 1604.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-28 09:15:00 | 1612.00 | 1600.34 | 1604.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-28 09:30:00 | 1610.10 | 1600.34 | 1604.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-28 10:15:00 | 1606.00 | 1601.47 | 1604.37 | EMA400 retest candle locked (from downside) |

### Cycle 23 — BUY (started 2024-08-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-29 09:15:00 | 1707.65 | 1624.99 | 1614.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-05 09:15:00 | 1876.30 | 1799.41 | 1769.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-06 12:15:00 | 1849.95 | 1850.31 | 1822.91 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-06 13:00:00 | 1849.95 | 1850.31 | 1822.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-06 14:15:00 | 1833.05 | 1846.55 | 1825.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-06 14:30:00 | 1826.20 | 1846.55 | 1825.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-09 09:15:00 | 1841.40 | 1845.13 | 1828.84 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-10 12:15:00 | 1863.95 | 1850.97 | 1840.84 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-10 13:45:00 | 1863.30 | 1853.83 | 1843.99 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-10 14:45:00 | 1863.55 | 1861.99 | 1848.60 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Target hit | 2024-09-12 15:15:00 | 2050.35 | 1962.43 | 1930.50 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 24 — SELL (started 2024-09-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-24 13:15:00 | 2225.00 | 2307.74 | 2314.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-25 13:15:00 | 2202.20 | 2249.99 | 2276.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-26 15:15:00 | 2176.45 | 2170.36 | 2211.23 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-27 09:15:00 | 2142.50 | 2170.36 | 2211.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-04 10:15:00 | 1964.40 | 1951.04 | 1980.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-04 10:45:00 | 1974.35 | 1951.04 | 1980.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-04 11:15:00 | 1982.50 | 1957.33 | 1980.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-04 12:00:00 | 1982.50 | 1957.33 | 1980.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-04 12:15:00 | 1969.00 | 1959.67 | 1979.84 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-04 13:15:00 | 1956.00 | 1959.67 | 1979.84 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-07 10:15:00 | 1858.20 | 1921.47 | 1952.45 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-10-08 10:15:00 | 1869.90 | 1869.12 | 1904.55 | SL hit (close>ema200) qty=0.50 sl=1869.12 alert=retest2 |

### Cycle 25 — BUY (started 2024-10-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-09 10:15:00 | 1977.35 | 1913.08 | 1910.58 | EMA200 above EMA400 |

### Cycle 26 — SELL (started 2024-10-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-10 12:15:00 | 1890.00 | 1921.56 | 1922.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-10 13:15:00 | 1883.00 | 1913.84 | 1918.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-11 14:15:00 | 1896.50 | 1889.09 | 1899.31 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-11 14:15:00 | 1896.50 | 1889.09 | 1899.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-11 14:15:00 | 1896.50 | 1889.09 | 1899.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-11 14:30:00 | 1896.45 | 1889.09 | 1899.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-11 15:15:00 | 1890.00 | 1889.27 | 1898.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-14 09:15:00 | 1891.65 | 1889.27 | 1898.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-14 09:15:00 | 1906.95 | 1892.81 | 1899.24 | EMA400 retest candle locked (from downside) |

### Cycle 27 — BUY (started 2024-10-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-14 11:15:00 | 1927.70 | 1905.46 | 1904.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-15 12:15:00 | 1940.95 | 1926.62 | 1917.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-16 11:15:00 | 1931.00 | 1931.69 | 1924.49 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-16 11:15:00 | 1931.00 | 1931.69 | 1924.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-16 11:15:00 | 1931.00 | 1931.69 | 1924.49 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-16 12:15:00 | 1935.65 | 1931.69 | 1924.49 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-16 13:00:00 | 1937.90 | 1932.93 | 1925.71 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-16 15:00:00 | 2000.25 | 1946.89 | 1933.34 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-21 12:15:00 | 1960.75 | 1986.83 | 1988.37 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 28 — SELL (started 2024-10-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-21 12:15:00 | 1960.75 | 1986.83 | 1988.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-21 13:15:00 | 1951.15 | 1979.70 | 1984.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-24 09:15:00 | 1859.65 | 1855.12 | 1882.83 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-24 11:15:00 | 1829.30 | 1852.39 | 1879.07 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-24 11:45:00 | 1823.10 | 1845.95 | 1873.72 | SELL ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-25 15:15:00 | 1818.45 | 1808.93 | 1829.68 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-28 09:15:00 | 1787.75 | 1808.93 | 1829.68 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-28 13:15:00 | 1824.60 | 1806.05 | 1819.00 | SL hit (close>ema400) qty=1.00 sl=1819.00 alert=retest1 |

### Cycle 29 — BUY (started 2024-10-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-29 15:15:00 | 1844.00 | 1821.66 | 1821.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-30 09:15:00 | 1849.70 | 1827.26 | 1824.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-30 10:15:00 | 1815.10 | 1824.83 | 1823.22 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-30 10:15:00 | 1815.10 | 1824.83 | 1823.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-30 10:15:00 | 1815.10 | 1824.83 | 1823.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-30 11:00:00 | 1815.10 | 1824.83 | 1823.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-30 11:15:00 | 1812.05 | 1822.28 | 1822.20 | EMA400 retest candle locked (from upside) |

### Cycle 30 — SELL (started 2024-10-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-30 12:15:00 | 1809.75 | 1819.77 | 1821.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-30 13:15:00 | 1798.25 | 1815.47 | 1818.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-31 09:15:00 | 1841.70 | 1817.95 | 1819.00 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-31 09:15:00 | 1841.70 | 1817.95 | 1819.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-31 09:15:00 | 1841.70 | 1817.95 | 1819.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-31 09:45:00 | 1856.80 | 1817.95 | 1819.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 31 — BUY (started 2024-10-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-31 10:15:00 | 1832.50 | 1820.86 | 1820.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-31 15:15:00 | 1848.00 | 1828.27 | 1824.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-04 09:15:00 | 1799.65 | 1833.18 | 1828.65 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-04 09:15:00 | 1799.65 | 1833.18 | 1828.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 09:15:00 | 1799.65 | 1833.18 | 1828.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-04 10:00:00 | 1799.65 | 1833.18 | 1828.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 10:15:00 | 1807.55 | 1828.05 | 1826.73 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-04 11:15:00 | 1819.50 | 1828.05 | 1826.73 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-04 12:15:00 | 1822.15 | 1825.69 | 1825.81 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 32 — SELL (started 2024-11-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-04 12:15:00 | 1822.15 | 1825.69 | 1825.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-04 14:15:00 | 1815.45 | 1822.84 | 1824.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-05 13:15:00 | 1823.85 | 1816.14 | 1819.54 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-05 13:15:00 | 1823.85 | 1816.14 | 1819.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-05 13:15:00 | 1823.85 | 1816.14 | 1819.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-05 14:00:00 | 1823.85 | 1816.14 | 1819.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-05 14:15:00 | 1815.45 | 1816.00 | 1819.17 | EMA400 retest candle locked (from downside) |

### Cycle 33 — BUY (started 2024-11-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-06 10:15:00 | 1839.75 | 1822.87 | 1821.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-07 10:15:00 | 1844.15 | 1836.36 | 1830.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-07 13:15:00 | 1832.00 | 1836.78 | 1832.17 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-07 13:15:00 | 1832.00 | 1836.78 | 1832.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 13:15:00 | 1832.00 | 1836.78 | 1832.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-07 13:45:00 | 1834.55 | 1836.78 | 1832.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 14:15:00 | 1826.05 | 1834.63 | 1831.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-07 15:00:00 | 1826.05 | 1834.63 | 1831.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 15:15:00 | 1826.05 | 1832.92 | 1831.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-08 09:15:00 | 1820.00 | 1832.92 | 1831.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 34 — SELL (started 2024-11-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-08 09:15:00 | 1816.40 | 1829.61 | 1829.77 | EMA200 below EMA400 |

### Cycle 35 — BUY (started 2024-11-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-08 12:15:00 | 1853.05 | 1834.39 | 1831.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-11 11:15:00 | 1863.00 | 1847.34 | 1840.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-12 12:15:00 | 1862.45 | 1871.31 | 1860.34 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-12 12:15:00 | 1862.45 | 1871.31 | 1860.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-12 12:15:00 | 1862.45 | 1871.31 | 1860.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-12 12:45:00 | 1866.25 | 1871.31 | 1860.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-12 13:15:00 | 1863.00 | 1869.65 | 1860.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-12 13:30:00 | 1862.85 | 1869.65 | 1860.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-12 14:15:00 | 1860.00 | 1867.72 | 1860.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-12 14:30:00 | 1863.40 | 1867.72 | 1860.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-12 15:15:00 | 1873.00 | 1868.77 | 1861.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-13 09:30:00 | 1844.95 | 1865.61 | 1860.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-13 10:15:00 | 1861.00 | 1864.69 | 1860.88 | EMA400 retest candle locked (from upside) |

### Cycle 36 — SELL (started 2024-11-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-13 14:15:00 | 1853.80 | 1858.16 | 1858.58 | EMA200 below EMA400 |

### Cycle 37 — BUY (started 2024-11-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-14 09:15:00 | 1872.50 | 1859.72 | 1859.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-19 09:15:00 | 1924.90 | 1900.44 | 1886.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-19 14:15:00 | 1903.45 | 1906.03 | 1895.23 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-19 15:00:00 | 1903.45 | 1906.03 | 1895.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 15:15:00 | 1903.05 | 1905.44 | 1895.94 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-21 09:15:00 | 1953.70 | 1905.44 | 1895.94 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-28 15:15:00 | 2035.95 | 2040.51 | 2040.70 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 38 — SELL (started 2024-11-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-28 15:15:00 | 2035.95 | 2040.51 | 2040.70 | EMA200 below EMA400 |

### Cycle 39 — BUY (started 2024-11-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-29 09:15:00 | 2059.95 | 2044.40 | 2042.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-29 10:15:00 | 2072.85 | 2050.09 | 2045.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-04 10:15:00 | 2159.25 | 2179.70 | 2159.46 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-04 10:15:00 | 2159.25 | 2179.70 | 2159.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-04 10:15:00 | 2159.25 | 2179.70 | 2159.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-04 10:30:00 | 2156.80 | 2179.70 | 2159.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-04 11:15:00 | 2145.10 | 2172.78 | 2158.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-04 11:45:00 | 2145.20 | 2172.78 | 2158.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-04 12:15:00 | 2173.05 | 2172.84 | 2159.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-04 12:30:00 | 2147.30 | 2172.84 | 2159.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-05 09:15:00 | 2161.85 | 2171.29 | 2163.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-05 09:45:00 | 2151.05 | 2171.29 | 2163.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-05 10:15:00 | 2178.00 | 2172.63 | 2164.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-05 10:45:00 | 2145.80 | 2172.63 | 2164.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-05 14:15:00 | 2202.90 | 2183.94 | 2172.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-05 14:45:00 | 2193.05 | 2183.94 | 2172.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-05 15:15:00 | 2172.00 | 2181.55 | 2172.90 | EMA400 retest candle locked (from upside) |

### Cycle 40 — SELL (started 2024-12-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-06 13:15:00 | 2145.80 | 2168.79 | 2169.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-09 11:15:00 | 2133.70 | 2154.71 | 2162.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-10 15:15:00 | 2095.95 | 2093.36 | 2115.24 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-11 09:15:00 | 2111.00 | 2093.36 | 2115.24 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-11 09:15:00 | 2115.95 | 2097.88 | 2115.30 | EMA400 retest candle locked (from downside) |

### Cycle 41 — BUY (started 2024-12-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-11 12:15:00 | 2187.75 | 2136.06 | 2130.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-11 15:15:00 | 2199.55 | 2163.90 | 2145.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-12 11:15:00 | 2157.00 | 2166.53 | 2151.60 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-12 12:00:00 | 2157.00 | 2166.53 | 2151.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-12 12:15:00 | 2154.50 | 2164.12 | 2151.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-12 13:00:00 | 2154.50 | 2164.12 | 2151.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-12 13:15:00 | 2142.20 | 2159.74 | 2150.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-12 14:00:00 | 2142.20 | 2159.74 | 2150.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-12 14:15:00 | 2133.05 | 2154.40 | 2149.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-12 15:00:00 | 2133.05 | 2154.40 | 2149.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-12 15:15:00 | 2141.00 | 2151.72 | 2148.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-13 09:15:00 | 2129.20 | 2151.72 | 2148.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 42 — SELL (started 2024-12-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-13 09:15:00 | 2109.80 | 2143.34 | 2145.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-13 15:15:00 | 2101.50 | 2121.14 | 2131.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-16 15:15:00 | 2104.95 | 2104.61 | 2116.07 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-17 09:15:00 | 2096.00 | 2104.61 | 2116.07 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-18 10:15:00 | 2083.10 | 2069.36 | 2084.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-18 10:30:00 | 2081.55 | 2069.36 | 2084.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-18 11:15:00 | 2143.50 | 2084.19 | 2090.20 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-12-18 11:15:00 | 2143.50 | 2084.19 | 2090.20 | SL hit (close>ema400) qty=1.00 sl=2090.20 alert=retest1 |

### Cycle 43 — BUY (started 2024-12-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-18 12:15:00 | 2153.10 | 2097.97 | 2095.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-18 14:15:00 | 2180.00 | 2121.74 | 2107.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-19 13:15:00 | 2116.80 | 2136.06 | 2123.69 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-19 13:15:00 | 2116.80 | 2136.06 | 2123.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-19 13:15:00 | 2116.80 | 2136.06 | 2123.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-19 14:00:00 | 2116.80 | 2136.06 | 2123.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-19 14:15:00 | 2120.55 | 2132.95 | 2123.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-19 15:15:00 | 2175.10 | 2132.95 | 2123.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-19 15:15:00 | 2175.10 | 2141.38 | 2128.10 | EMA400 retest candle locked (from upside) |

### Cycle 44 — SELL (started 2024-12-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-23 09:15:00 | 2102.35 | 2125.03 | 2126.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-23 10:15:00 | 2076.10 | 2115.25 | 2121.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-23 13:15:00 | 2110.95 | 2108.58 | 2116.44 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-23 14:00:00 | 2110.95 | 2108.58 | 2116.44 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-23 14:15:00 | 2144.70 | 2115.80 | 2119.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-23 14:45:00 | 2129.10 | 2115.80 | 2119.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-23 15:15:00 | 2112.35 | 2115.11 | 2118.41 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-24 09:30:00 | 2105.05 | 2114.99 | 2118.05 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-24 11:15:00 | 2111.00 | 2114.78 | 2117.68 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-24 13:15:00 | 2136.95 | 2121.62 | 2120.27 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 45 — BUY (started 2024-12-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-24 13:15:00 | 2136.95 | 2121.62 | 2120.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-24 14:15:00 | 2152.50 | 2127.80 | 2123.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-27 14:15:00 | 2165.85 | 2172.08 | 2158.60 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-27 14:45:00 | 2168.10 | 2172.08 | 2158.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-27 15:15:00 | 2160.00 | 2169.66 | 2158.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-30 09:15:00 | 2164.35 | 2169.66 | 2158.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-30 09:15:00 | 2189.45 | 2173.62 | 2161.52 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-30 11:30:00 | 2215.50 | 2183.13 | 2168.15 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-30 12:15:00 | 2212.10 | 2183.13 | 2168.15 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-30 13:45:00 | 2214.95 | 2193.41 | 2175.66 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-31 09:15:00 | 2239.45 | 2189.73 | 2177.01 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-31 14:15:00 | 2216.00 | 2236.67 | 2214.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-31 14:30:00 | 2266.15 | 2236.67 | 2214.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-31 15:15:00 | 2212.00 | 2231.74 | 2214.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-01 09:15:00 | 2182.00 | 2231.74 | 2214.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-01 09:15:00 | 2173.00 | 2219.99 | 2210.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-01 09:30:00 | 2178.45 | 2219.99 | 2210.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-01 10:15:00 | 2180.05 | 2212.00 | 2207.58 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-01 11:15:00 | 2193.00 | 2212.00 | 2207.58 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-01 12:15:00 | 2170.80 | 2200.19 | 2202.76 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 46 — SELL (started 2025-01-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-01 12:15:00 | 2170.80 | 2200.19 | 2202.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-01 14:15:00 | 2154.00 | 2186.60 | 2195.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-06 09:15:00 | 2135.70 | 2107.84 | 2127.08 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-06 09:15:00 | 2135.70 | 2107.84 | 2127.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 09:15:00 | 2135.70 | 2107.84 | 2127.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-06 09:45:00 | 2147.40 | 2107.84 | 2127.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 10:15:00 | 2115.40 | 2109.35 | 2126.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-06 10:30:00 | 2126.20 | 2109.35 | 2126.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 12:15:00 | 2128.00 | 2114.73 | 2125.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-06 12:45:00 | 2134.45 | 2114.73 | 2125.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 13:15:00 | 2127.35 | 2117.25 | 2125.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-06 14:00:00 | 2127.35 | 2117.25 | 2125.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 14:15:00 | 2142.70 | 2122.34 | 2127.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-06 15:00:00 | 2142.70 | 2122.34 | 2127.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 15:15:00 | 2140.00 | 2125.87 | 2128.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-07 09:15:00 | 2153.75 | 2125.87 | 2128.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 47 — BUY (started 2025-01-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-07 09:15:00 | 2164.25 | 2133.55 | 2131.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-07 11:15:00 | 2174.00 | 2146.66 | 2138.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-10 09:15:00 | 2223.70 | 2282.31 | 2247.42 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-10 09:15:00 | 2223.70 | 2282.31 | 2247.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-10 09:15:00 | 2223.70 | 2282.31 | 2247.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-10 10:00:00 | 2223.70 | 2282.31 | 2247.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-10 10:15:00 | 2229.05 | 2271.66 | 2245.75 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-10 11:15:00 | 2247.95 | 2271.66 | 2245.75 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-10 12:30:00 | 2244.65 | 2264.90 | 2246.89 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-13 09:15:00 | 2285.00 | 2256.13 | 2247.10 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-13 13:45:00 | 2253.00 | 2254.66 | 2250.26 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-13 14:15:00 | 2263.30 | 2256.39 | 2251.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-13 14:30:00 | 2253.65 | 2256.39 | 2251.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-15 09:15:00 | 2288.30 | 2302.07 | 2284.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-15 10:00:00 | 2288.30 | 2302.07 | 2284.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-15 10:15:00 | 2289.30 | 2299.52 | 2285.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-15 10:30:00 | 2284.35 | 2299.52 | 2285.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-15 11:15:00 | 2288.50 | 2297.32 | 2285.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-15 12:00:00 | 2288.50 | 2297.32 | 2285.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-15 12:15:00 | 2281.15 | 2294.08 | 2285.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-15 12:45:00 | 2277.40 | 2294.08 | 2285.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-15 13:15:00 | 2257.00 | 2286.67 | 2282.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-15 13:45:00 | 2268.00 | 2286.67 | 2282.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-01-15 14:15:00 | 2238.25 | 2276.98 | 2278.62 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 48 — SELL (started 2025-01-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-15 14:15:00 | 2238.25 | 2276.98 | 2278.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-17 09:15:00 | 2213.75 | 2239.58 | 2255.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-20 13:15:00 | 2195.75 | 2195.16 | 2213.99 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-20 13:30:00 | 2196.35 | 2195.16 | 2213.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 09:15:00 | 2198.50 | 2190.14 | 2206.43 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-21 14:15:00 | 2167.95 | 2185.88 | 2199.48 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-27 09:15:00 | 2059.55 | 2094.71 | 2119.45 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-01-27 10:15:00 | 1951.15 | 2075.61 | 2108.52 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 49 — BUY (started 2025-01-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-30 09:15:00 | 2060.05 | 2038.32 | 2037.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-31 09:15:00 | 2144.15 | 2069.74 | 2054.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-01 09:15:00 | 2122.40 | 2130.02 | 2100.57 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-01 09:15:00 | 2122.40 | 2130.02 | 2100.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 09:15:00 | 2122.40 | 2130.02 | 2100.57 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-04 09:15:00 | 2160.80 | 2125.99 | 2113.54 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-02-10 09:15:00 | 2376.88 | 2352.26 | 2319.51 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 50 — SELL (started 2025-02-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-10 15:15:00 | 2231.00 | 2302.27 | 2310.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-11 09:15:00 | 2115.55 | 2264.93 | 2292.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-12 12:15:00 | 2135.10 | 2134.96 | 2184.60 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-12 13:00:00 | 2135.10 | 2134.96 | 2184.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-13 09:15:00 | 2171.00 | 2137.16 | 2169.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-13 10:00:00 | 2171.00 | 2137.16 | 2169.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-13 10:15:00 | 2160.00 | 2141.73 | 2168.42 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-13 11:30:00 | 2145.85 | 2140.36 | 2165.37 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-02-14 09:15:00 | 1931.26 | 2069.83 | 2122.26 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 51 — BUY (started 2025-03-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-04 11:15:00 | 1650.65 | 1575.41 | 1565.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-04 13:15:00 | 1732.45 | 1619.90 | 1588.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-06 13:15:00 | 1742.30 | 1745.12 | 1708.75 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-06 14:00:00 | 1742.30 | 1745.12 | 1708.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-07 09:15:00 | 1734.55 | 1743.53 | 1717.22 | EMA400 retest candle locked (from upside) |

### Cycle 52 — SELL (started 2025-03-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-10 12:15:00 | 1693.30 | 1715.36 | 1716.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-10 14:15:00 | 1679.85 | 1704.46 | 1710.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-12 14:15:00 | 1628.80 | 1617.52 | 1645.32 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-12 15:00:00 | 1628.80 | 1617.52 | 1645.32 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-13 09:15:00 | 1637.40 | 1621.89 | 1642.51 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-17 13:45:00 | 1600.05 | 1620.94 | 1631.06 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-17 15:00:00 | 1596.35 | 1616.02 | 1627.90 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-18 15:15:00 | 1651.05 | 1629.54 | 1628.20 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 53 — BUY (started 2025-03-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-18 15:15:00 | 1651.05 | 1629.54 | 1628.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-19 09:15:00 | 1675.75 | 1638.78 | 1632.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-19 12:15:00 | 1639.15 | 1651.81 | 1641.26 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-19 12:15:00 | 1639.15 | 1651.81 | 1641.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-19 12:15:00 | 1639.15 | 1651.81 | 1641.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-19 13:00:00 | 1639.15 | 1651.81 | 1641.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-19 13:15:00 | 1652.55 | 1651.96 | 1642.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-19 13:30:00 | 1625.55 | 1651.96 | 1642.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-19 14:15:00 | 1645.15 | 1650.60 | 1642.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-19 15:00:00 | 1645.15 | 1650.60 | 1642.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-19 15:15:00 | 1643.05 | 1649.09 | 1642.59 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-20 09:15:00 | 1670.00 | 1649.09 | 1642.59 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-20 10:15:00 | 1658.60 | 1648.73 | 1643.02 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-20 10:45:00 | 1656.25 | 1652.13 | 1645.08 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-20 13:45:00 | 1662.70 | 1652.32 | 1646.92 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-20 14:15:00 | 1649.00 | 1651.66 | 1647.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-20 14:45:00 | 1646.90 | 1651.66 | 1647.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-20 15:15:00 | 1645.90 | 1650.51 | 1647.00 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-21 09:15:00 | 1672.00 | 1650.51 | 1647.00 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-21 13:15:00 | 1650.65 | 1654.38 | 1650.58 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-25 09:30:00 | 1653.05 | 1664.68 | 1661.55 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-25 12:15:00 | 1636.85 | 1658.73 | 1659.73 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 54 — SELL (started 2025-03-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-25 12:15:00 | 1636.85 | 1658.73 | 1659.73 | EMA200 below EMA400 |

### Cycle 55 — BUY (started 2025-03-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-25 14:15:00 | 1689.75 | 1662.31 | 1661.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-27 11:15:00 | 1702.35 | 1679.25 | 1672.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-28 09:15:00 | 1689.60 | 1694.64 | 1684.04 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-28 09:15:00 | 1689.60 | 1694.64 | 1684.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-28 09:15:00 | 1689.60 | 1694.64 | 1684.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-28 09:30:00 | 1695.50 | 1694.64 | 1684.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-28 10:15:00 | 1685.65 | 1692.84 | 1684.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-28 11:15:00 | 1682.85 | 1692.84 | 1684.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-28 11:15:00 | 1683.80 | 1691.03 | 1684.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-28 11:45:00 | 1680.40 | 1691.03 | 1684.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-28 12:15:00 | 1674.85 | 1687.80 | 1683.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-28 13:00:00 | 1674.85 | 1687.80 | 1683.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-28 13:15:00 | 1684.00 | 1687.04 | 1683.37 | EMA400 retest candle locked (from upside) |

### Cycle 56 — SELL (started 2025-04-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-01 09:15:00 | 1663.95 | 1678.41 | 1680.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-01 10:15:00 | 1651.40 | 1673.01 | 1677.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-01 15:15:00 | 1679.95 | 1668.31 | 1672.58 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-01 15:15:00 | 1679.95 | 1668.31 | 1672.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-01 15:15:00 | 1679.95 | 1668.31 | 1672.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-02 09:15:00 | 1649.25 | 1668.31 | 1672.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 09:15:00 | 1654.05 | 1665.45 | 1670.90 | EMA400 retest candle locked (from downside) |

### Cycle 57 — BUY (started 2025-04-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-02 15:15:00 | 1694.60 | 1674.41 | 1672.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-03 09:15:00 | 1721.85 | 1683.90 | 1677.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-04 09:15:00 | 1646.90 | 1694.06 | 1688.67 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-04 09:15:00 | 1646.90 | 1694.06 | 1688.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 09:15:00 | 1646.90 | 1694.06 | 1688.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-04 10:00:00 | 1646.90 | 1694.06 | 1688.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 10:15:00 | 1673.75 | 1690.00 | 1687.31 | EMA400 retest candle locked (from upside) |

### Cycle 58 — SELL (started 2025-04-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-04 11:15:00 | 1658.25 | 1683.65 | 1684.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-04 12:15:00 | 1652.05 | 1677.33 | 1681.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-08 09:15:00 | 1612.50 | 1602.90 | 1627.56 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-08 13:15:00 | 1608.35 | 1608.98 | 1623.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 13:15:00 | 1608.35 | 1608.98 | 1623.15 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-08 15:00:00 | 1591.70 | 1605.52 | 1620.29 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-09 11:15:00 | 1637.70 | 1613.43 | 1619.12 | SL hit (close>static) qty=1.00 sl=1623.90 alert=retest2 |

### Cycle 59 — BUY (started 2025-04-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-11 09:15:00 | 1652.00 | 1620.07 | 1619.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-11 11:15:00 | 1657.70 | 1632.63 | 1626.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-16 12:15:00 | 1731.20 | 1733.35 | 1703.07 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-16 13:00:00 | 1731.20 | 1733.35 | 1703.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-17 13:15:00 | 1717.60 | 1726.76 | 1716.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-17 13:45:00 | 1716.30 | 1726.76 | 1716.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-17 14:15:00 | 1722.40 | 1725.89 | 1716.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-17 14:45:00 | 1721.80 | 1725.89 | 1716.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-17 15:15:00 | 1716.90 | 1724.09 | 1716.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-21 09:15:00 | 1714.50 | 1724.09 | 1716.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-21 09:15:00 | 1719.60 | 1723.19 | 1716.90 | EMA400 retest candle locked (from upside) |

### Cycle 60 — SELL (started 2025-04-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-22 09:15:00 | 1682.40 | 1712.69 | 1714.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-22 15:15:00 | 1677.70 | 1691.02 | 1701.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-24 10:15:00 | 1676.60 | 1657.49 | 1673.37 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-24 10:15:00 | 1676.60 | 1657.49 | 1673.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-24 10:15:00 | 1676.60 | 1657.49 | 1673.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-24 11:00:00 | 1676.60 | 1657.49 | 1673.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-24 11:15:00 | 1691.40 | 1664.27 | 1675.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-24 11:30:00 | 1689.60 | 1664.27 | 1675.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-24 12:15:00 | 1662.50 | 1663.91 | 1673.87 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-24 13:15:00 | 1655.10 | 1663.91 | 1673.87 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-24 13:45:00 | 1648.90 | 1661.53 | 1671.88 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-28 09:15:00 | 1572.34 | 1599.35 | 1626.35 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-28 09:15:00 | 1566.45 | 1599.35 | 1626.35 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-04-29 10:15:00 | 1585.50 | 1570.88 | 1593.26 | SL hit (close>ema200) qty=0.50 sl=1570.88 alert=retest2 |

### Cycle 61 — BUY (started 2025-05-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-08 11:15:00 | 1508.00 | 1434.23 | 1433.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-08 12:15:00 | 1527.10 | 1452.80 | 1442.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-08 15:15:00 | 1482.30 | 1482.87 | 1460.82 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-09 09:15:00 | 1472.70 | 1482.87 | 1460.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-09 09:15:00 | 1488.90 | 1484.08 | 1463.37 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-09 12:15:00 | 1504.10 | 1486.36 | 1468.05 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-16 14:15:00 | 1497.50 | 1513.94 | 1514.16 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 62 — SELL (started 2025-05-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-16 14:15:00 | 1497.50 | 1513.94 | 1514.16 | EMA200 below EMA400 |

### Cycle 63 — BUY (started 2025-05-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-19 09:15:00 | 1531.80 | 1515.73 | 1514.83 | EMA200 above EMA400 |

### Cycle 64 — SELL (started 2025-05-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-20 12:15:00 | 1511.90 | 1519.30 | 1519.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-21 13:15:00 | 1508.40 | 1514.38 | 1516.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-22 10:15:00 | 1519.40 | 1512.26 | 1514.49 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-22 10:15:00 | 1519.40 | 1512.26 | 1514.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 10:15:00 | 1519.40 | 1512.26 | 1514.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-22 11:00:00 | 1519.40 | 1512.26 | 1514.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 11:15:00 | 1523.00 | 1514.41 | 1515.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-22 12:00:00 | 1523.00 | 1514.41 | 1515.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 65 — BUY (started 2025-05-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-22 13:15:00 | 1536.80 | 1519.64 | 1517.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-23 14:15:00 | 1538.30 | 1531.58 | 1525.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-26 12:15:00 | 1536.00 | 1536.04 | 1530.62 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-26 13:00:00 | 1536.00 | 1536.04 | 1530.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 13:15:00 | 1530.80 | 1534.99 | 1530.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-26 14:00:00 | 1530.80 | 1534.99 | 1530.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 14:15:00 | 1535.70 | 1535.13 | 1531.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-26 15:15:00 | 1534.00 | 1535.13 | 1531.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 15:15:00 | 1534.00 | 1534.90 | 1531.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-27 09:15:00 | 1534.00 | 1534.90 | 1531.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 09:15:00 | 1540.10 | 1535.94 | 1532.16 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-28 09:30:00 | 1557.00 | 1536.91 | 1534.40 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-05-30 09:15:00 | 1712.70 | 1679.78 | 1634.32 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 66 — SELL (started 2025-06-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-17 14:15:00 | 2074.50 | 2093.01 | 2093.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-18 09:15:00 | 2019.30 | 2074.56 | 2084.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-18 15:15:00 | 2035.00 | 2026.63 | 2051.79 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-19 09:15:00 | 2036.70 | 2028.64 | 2050.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 09:15:00 | 2036.70 | 2028.64 | 2050.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-19 10:00:00 | 2036.70 | 2028.64 | 2050.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 09:15:00 | 1940.90 | 1984.46 | 2015.22 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-20 10:15:00 | 1935.10 | 1984.46 | 2015.22 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-20 11:00:00 | 1935.70 | 1974.71 | 2007.99 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-20 14:15:00 | 1838.34 | 1918.85 | 1968.91 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-20 14:15:00 | 1838.91 | 1918.85 | 1968.91 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-25 09:15:00 | 1821.70 | 1799.40 | 1836.93 | SL hit (close>ema200) qty=0.50 sl=1799.40 alert=retest2 |

### Cycle 67 — BUY (started 2025-06-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-30 14:15:00 | 1839.10 | 1823.45 | 1822.37 | EMA200 above EMA400 |

### Cycle 68 — SELL (started 2025-07-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-01 11:15:00 | 1817.30 | 1822.51 | 1822.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-01 14:15:00 | 1800.00 | 1815.14 | 1818.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-03 13:15:00 | 1760.90 | 1760.57 | 1777.15 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-03 13:30:00 | 1760.60 | 1760.57 | 1777.15 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 09:15:00 | 1778.60 | 1751.84 | 1760.09 | EMA400 retest candle locked (from downside) |

### Cycle 69 — BUY (started 2025-07-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-07 13:15:00 | 1774.00 | 1765.05 | 1764.59 | EMA200 above EMA400 |

### Cycle 70 — SELL (started 2025-07-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-07 14:15:00 | 1760.90 | 1764.22 | 1764.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-07 15:15:00 | 1756.60 | 1762.69 | 1763.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-08 12:15:00 | 1753.30 | 1749.70 | 1755.77 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-08 12:15:00 | 1753.30 | 1749.70 | 1755.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 12:15:00 | 1753.30 | 1749.70 | 1755.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-08 12:45:00 | 1755.80 | 1749.70 | 1755.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 13:15:00 | 1777.70 | 1755.30 | 1757.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-08 14:00:00 | 1777.70 | 1755.30 | 1757.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 71 — BUY (started 2025-07-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-08 14:15:00 | 1789.90 | 1762.22 | 1760.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-09 10:15:00 | 1827.30 | 1788.20 | 1774.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-10 09:15:00 | 1813.60 | 1818.27 | 1799.13 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-10 09:30:00 | 1804.30 | 1818.27 | 1799.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 10:15:00 | 1796.90 | 1813.99 | 1798.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-10 11:00:00 | 1796.90 | 1813.99 | 1798.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 11:15:00 | 1800.00 | 1811.19 | 1799.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-10 11:30:00 | 1801.90 | 1811.19 | 1799.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 12:15:00 | 1811.60 | 1811.28 | 1800.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-10 12:45:00 | 1816.10 | 1811.28 | 1800.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 13:15:00 | 1803.00 | 1809.62 | 1800.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-10 13:45:00 | 1804.90 | 1809.62 | 1800.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 09:15:00 | 1798.80 | 1806.76 | 1801.38 | EMA400 retest candle locked (from upside) |

### Cycle 72 — SELL (started 2025-07-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-11 12:15:00 | 1790.80 | 1797.84 | 1798.15 | EMA200 below EMA400 |

### Cycle 73 — BUY (started 2025-07-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-11 13:15:00 | 1805.30 | 1799.34 | 1798.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-14 09:15:00 | 1837.70 | 1808.17 | 1803.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-15 12:15:00 | 1833.00 | 1835.74 | 1824.67 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-15 13:00:00 | 1833.00 | 1835.74 | 1824.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 14:15:00 | 1828.20 | 1834.58 | 1826.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-15 15:00:00 | 1828.20 | 1834.58 | 1826.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 15:15:00 | 1846.70 | 1837.00 | 1827.96 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-16 11:15:00 | 1849.00 | 1837.44 | 1829.72 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-16 12:30:00 | 1849.90 | 1840.45 | 1832.51 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-16 14:15:00 | 1848.10 | 1841.48 | 1833.70 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-17 09:15:00 | 1848.90 | 1839.95 | 1834.35 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 09:15:00 | 1847.80 | 1841.52 | 1835.57 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-17 10:45:00 | 1876.70 | 1845.44 | 1837.89 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-17 12:00:00 | 1898.30 | 1856.01 | 1843.39 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-25 11:15:00 | 1898.10 | 1924.61 | 1926.30 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 74 — SELL (started 2025-07-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-25 11:15:00 | 1898.10 | 1924.61 | 1926.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-25 12:15:00 | 1896.00 | 1918.89 | 1923.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-05 13:15:00 | 1698.70 | 1698.00 | 1724.17 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-05 13:45:00 | 1706.20 | 1698.00 | 1724.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 09:15:00 | 1656.50 | 1689.40 | 1713.79 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-06 10:15:00 | 1654.10 | 1689.40 | 1713.79 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-06 15:15:00 | 1650.00 | 1663.54 | 1689.64 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2025-08-11 09:15:00 | 1488.69 | 1599.52 | 1626.98 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 75 — BUY (started 2025-08-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-12 12:15:00 | 1671.10 | 1627.13 | 1623.26 | EMA200 above EMA400 |

### Cycle 76 — SELL (started 2025-08-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-18 13:15:00 | 1631.70 | 1636.16 | 1636.34 | EMA200 below EMA400 |

### Cycle 77 — BUY (started 2025-08-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-19 09:15:00 | 1647.00 | 1637.99 | 1637.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-20 09:15:00 | 1653.30 | 1645.88 | 1642.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-25 09:15:00 | 1743.00 | 1757.54 | 1735.47 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-25 10:15:00 | 1734.90 | 1757.54 | 1735.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 10:15:00 | 1731.00 | 1752.23 | 1735.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-25 10:45:00 | 1730.90 | 1752.23 | 1735.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 11:15:00 | 1728.20 | 1747.43 | 1734.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-25 12:00:00 | 1728.20 | 1747.43 | 1734.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 12:15:00 | 1734.00 | 1744.74 | 1734.40 | EMA400 retest candle locked (from upside) |

### Cycle 78 — SELL (started 2025-08-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-26 10:15:00 | 1703.60 | 1726.58 | 1728.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-28 09:15:00 | 1689.20 | 1709.65 | 1718.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-28 14:15:00 | 1700.20 | 1697.96 | 1708.09 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-28 14:45:00 | 1701.00 | 1697.96 | 1708.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 09:15:00 | 1702.20 | 1695.11 | 1704.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-29 09:45:00 | 1713.60 | 1695.11 | 1704.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 10:15:00 | 1708.30 | 1697.74 | 1705.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-29 11:00:00 | 1708.30 | 1697.74 | 1705.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 11:15:00 | 1701.70 | 1698.54 | 1704.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-29 11:30:00 | 1703.40 | 1698.54 | 1704.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 12:15:00 | 1698.70 | 1698.57 | 1704.26 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-29 14:00:00 | 1692.90 | 1697.43 | 1703.22 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-01 10:00:00 | 1692.00 | 1691.98 | 1698.90 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-01 11:15:00 | 1685.50 | 1692.82 | 1698.66 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-02 09:15:00 | 1682.00 | 1689.71 | 1694.46 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 09:15:00 | 1695.80 | 1690.93 | 1694.58 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-02 12:45:00 | 1674.10 | 1686.55 | 1691.65 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-03 09:15:00 | 1657.30 | 1687.25 | 1690.71 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-03 12:15:00 | 1671.40 | 1675.18 | 1683.41 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-04 10:30:00 | 1672.50 | 1681.97 | 1684.23 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 12:15:00 | 1672.60 | 1678.55 | 1682.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-04 13:00:00 | 1672.60 | 1678.55 | 1682.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 09:15:00 | 1687.00 | 1677.07 | 1679.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-05 10:00:00 | 1687.00 | 1677.07 | 1679.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 10:15:00 | 1679.30 | 1677.51 | 1679.93 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-05 12:00:00 | 1665.90 | 1675.19 | 1678.66 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-08 09:45:00 | 1663.30 | 1664.86 | 1671.28 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-08 15:15:00 | 1608.26 | 1633.11 | 1650.47 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-08 15:15:00 | 1607.40 | 1633.11 | 1650.47 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-09 14:15:00 | 1601.22 | 1614.71 | 1633.18 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-09 14:15:00 | 1597.90 | 1614.71 | 1633.18 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-09 14:15:00 | 1590.39 | 1614.71 | 1633.18 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-09 14:15:00 | 1574.43 | 1614.71 | 1633.18 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-09 14:15:00 | 1587.83 | 1614.71 | 1633.18 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-09 14:15:00 | 1588.88 | 1614.71 | 1633.18 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-09 14:15:00 | 1582.61 | 1614.71 | 1633.18 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-09 14:15:00 | 1580.13 | 1614.71 | 1633.18 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-10 09:15:00 | 1631.10 | 1614.64 | 1629.74 | SL hit (close>ema200) qty=0.50 sl=1614.64 alert=retest2 |

### Cycle 79 — BUY (started 2025-09-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-10 14:15:00 | 1673.50 | 1640.79 | 1637.71 | EMA200 above EMA400 |

### Cycle 80 — SELL (started 2025-09-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-12 12:15:00 | 1630.00 | 1637.85 | 1638.74 | EMA200 below EMA400 |

### Cycle 81 — BUY (started 2025-09-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-15 09:15:00 | 1655.50 | 1642.43 | 1640.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-15 10:15:00 | 1664.00 | 1646.74 | 1642.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-16 09:15:00 | 1646.70 | 1651.56 | 1647.51 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-16 09:15:00 | 1646.70 | 1651.56 | 1647.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 09:15:00 | 1646.70 | 1651.56 | 1647.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-16 09:45:00 | 1644.40 | 1651.56 | 1647.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 10:15:00 | 1654.00 | 1652.05 | 1648.10 | EMA400 retest candle locked (from upside) |

### Cycle 82 — SELL (started 2025-09-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-16 14:15:00 | 1624.00 | 1642.08 | 1644.45 | EMA200 below EMA400 |

### Cycle 83 — BUY (started 2025-09-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-18 15:15:00 | 1670.00 | 1642.23 | 1638.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-19 09:15:00 | 1679.30 | 1649.65 | 1642.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-22 09:15:00 | 1660.70 | 1674.40 | 1661.97 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-22 09:15:00 | 1660.70 | 1674.40 | 1661.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 09:15:00 | 1660.70 | 1674.40 | 1661.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-22 10:15:00 | 1656.00 | 1674.40 | 1661.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 10:15:00 | 1647.00 | 1668.92 | 1660.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-22 11:00:00 | 1647.00 | 1668.92 | 1660.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 11:15:00 | 1630.30 | 1661.20 | 1657.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-22 12:00:00 | 1630.30 | 1661.20 | 1657.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 84 — SELL (started 2025-09-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-22 12:15:00 | 1625.00 | 1653.96 | 1654.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-22 14:15:00 | 1620.60 | 1642.86 | 1649.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-23 11:15:00 | 1631.40 | 1630.35 | 1640.38 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-23 11:45:00 | 1629.80 | 1630.35 | 1640.38 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 09:15:00 | 1630.90 | 1629.18 | 1635.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-24 10:00:00 | 1630.90 | 1629.18 | 1635.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 13:15:00 | 1634.40 | 1627.99 | 1632.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-24 13:30:00 | 1631.00 | 1627.99 | 1632.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 14:15:00 | 1626.60 | 1627.72 | 1632.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-24 14:45:00 | 1632.30 | 1627.72 | 1632.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 15:15:00 | 1630.60 | 1628.29 | 1632.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-25 09:15:00 | 1618.80 | 1628.29 | 1632.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 09:15:00 | 1618.70 | 1626.37 | 1630.98 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-25 14:45:00 | 1589.50 | 1615.51 | 1624.09 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-30 10:15:00 | 1599.60 | 1590.09 | 1589.53 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 85 — BUY (started 2025-09-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-30 10:15:00 | 1599.60 | 1590.09 | 1589.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-30 11:15:00 | 1610.70 | 1594.22 | 1591.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-01 15:15:00 | 1630.30 | 1631.65 | 1620.59 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-03 09:15:00 | 1627.20 | 1631.65 | 1620.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 13:15:00 | 1604.40 | 1624.52 | 1621.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-03 14:00:00 | 1604.40 | 1624.52 | 1621.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 86 — SELL (started 2025-10-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-03 14:15:00 | 1592.90 | 1618.20 | 1618.78 | EMA200 below EMA400 |

### Cycle 87 — BUY (started 2025-10-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-07 09:15:00 | 1630.70 | 1615.46 | 1614.73 | EMA200 above EMA400 |

### Cycle 88 — SELL (started 2025-10-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-08 14:15:00 | 1605.90 | 1618.69 | 1619.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-09 09:15:00 | 1593.20 | 1611.72 | 1616.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-10 09:15:00 | 1609.80 | 1601.83 | 1607.79 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-10 09:15:00 | 1609.80 | 1601.83 | 1607.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 09:15:00 | 1609.80 | 1601.83 | 1607.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-10 10:00:00 | 1609.80 | 1601.83 | 1607.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 10:15:00 | 1590.10 | 1599.49 | 1606.18 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-10 11:30:00 | 1585.50 | 1597.29 | 1604.58 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-10 15:00:00 | 1587.30 | 1595.00 | 1601.67 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-13 10:00:00 | 1586.00 | 1594.72 | 1600.48 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-15 09:15:00 | 1507.93 | 1530.73 | 1553.90 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-15 10:15:00 | 1506.22 | 1524.76 | 1549.08 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-15 10:15:00 | 1506.70 | 1524.76 | 1549.08 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-16 09:15:00 | 1509.60 | 1509.43 | 1529.26 | SL hit (close>ema200) qty=0.50 sl=1509.43 alert=retest2 |

### Cycle 89 — BUY (started 2025-10-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-23 13:15:00 | 1516.80 | 1514.43 | 1514.41 | EMA200 above EMA400 |

### Cycle 90 — SELL (started 2025-10-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-23 14:15:00 | 1505.90 | 1512.73 | 1513.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-24 09:15:00 | 1497.00 | 1509.47 | 1511.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-29 09:15:00 | 1444.40 | 1442.73 | 1457.67 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-29 10:15:00 | 1435.20 | 1442.73 | 1457.67 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 14:15:00 | 1450.00 | 1447.58 | 1454.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-29 14:30:00 | 1453.30 | 1447.58 | 1454.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 15:15:00 | 1456.80 | 1449.42 | 1454.91 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-10-29 15:15:00 | 1456.80 | 1449.42 | 1454.91 | SL hit (close>ema400) qty=1.00 sl=1454.91 alert=retest1 |

### Cycle 91 — BUY (started 2025-11-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-03 11:15:00 | 1463.20 | 1454.99 | 1454.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-03 13:15:00 | 1467.00 | 1458.51 | 1455.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-04 09:15:00 | 1458.00 | 1461.19 | 1458.08 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-04 09:15:00 | 1458.00 | 1461.19 | 1458.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 09:15:00 | 1458.00 | 1461.19 | 1458.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-04 10:00:00 | 1458.00 | 1461.19 | 1458.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 10:15:00 | 1459.30 | 1460.82 | 1458.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-04 10:45:00 | 1458.00 | 1460.82 | 1458.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 11:15:00 | 1465.00 | 1461.65 | 1458.81 | EMA400 retest candle locked (from upside) |

### Cycle 92 — SELL (started 2025-11-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-06 09:15:00 | 1432.30 | 1453.26 | 1455.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-06 10:15:00 | 1424.00 | 1447.41 | 1452.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-07 15:15:00 | 1402.20 | 1399.38 | 1415.56 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-07 15:15:00 | 1402.20 | 1399.38 | 1415.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 15:15:00 | 1402.20 | 1399.38 | 1415.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-10 10:00:00 | 1421.00 | 1403.70 | 1416.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 10:15:00 | 1432.00 | 1409.36 | 1417.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-10 10:30:00 | 1432.00 | 1409.36 | 1417.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 93 — BUY (started 2025-11-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-10 13:15:00 | 1435.20 | 1421.96 | 1421.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-10 14:15:00 | 1439.50 | 1425.47 | 1423.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-11 09:15:00 | 1422.30 | 1426.41 | 1424.35 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-11 09:15:00 | 1422.30 | 1426.41 | 1424.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 09:15:00 | 1422.30 | 1426.41 | 1424.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-11 09:30:00 | 1416.00 | 1426.41 | 1424.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 10:15:00 | 1416.90 | 1424.51 | 1423.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-11 10:30:00 | 1415.00 | 1424.51 | 1423.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 14:15:00 | 1457.40 | 1471.67 | 1460.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-13 15:00:00 | 1457.40 | 1471.67 | 1460.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 15:15:00 | 1463.80 | 1470.09 | 1461.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-14 09:15:00 | 1464.30 | 1470.09 | 1461.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 09:15:00 | 1460.90 | 1468.26 | 1461.09 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-14 12:00:00 | 1488.10 | 1471.54 | 1463.79 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-19 10:15:00 | 1455.20 | 1485.67 | 1487.84 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 94 — SELL (started 2025-11-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-19 10:15:00 | 1455.20 | 1485.67 | 1487.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-19 14:15:00 | 1448.50 | 1466.87 | 1477.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-24 10:15:00 | 1438.10 | 1432.69 | 1441.85 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-24 10:30:00 | 1437.80 | 1432.69 | 1441.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 09:15:00 | 1440.20 | 1413.84 | 1421.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-26 10:00:00 | 1440.20 | 1413.84 | 1421.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 10:15:00 | 1436.10 | 1418.29 | 1422.61 | EMA400 retest candle locked (from downside) |

### Cycle 95 — BUY (started 2025-11-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-26 12:15:00 | 1439.90 | 1426.25 | 1425.71 | EMA200 above EMA400 |

### Cycle 96 — SELL (started 2025-11-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-28 10:15:00 | 1413.60 | 1425.49 | 1426.95 | EMA200 below EMA400 |

### Cycle 97 — BUY (started 2025-12-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-01 11:15:00 | 1440.00 | 1425.96 | 1425.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-01 13:15:00 | 1442.00 | 1431.06 | 1428.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-01 15:15:00 | 1430.00 | 1431.05 | 1428.71 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-02 09:15:00 | 1425.70 | 1431.05 | 1428.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 09:15:00 | 1431.90 | 1431.22 | 1429.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-02 09:45:00 | 1426.10 | 1431.22 | 1429.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 12:15:00 | 1439.90 | 1432.62 | 1430.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-02 12:30:00 | 1432.10 | 1432.62 | 1430.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 09:15:00 | 1419.60 | 1431.75 | 1430.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-03 10:00:00 | 1419.60 | 1431.75 | 1430.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 98 — SELL (started 2025-12-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-03 10:15:00 | 1421.00 | 1429.60 | 1429.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-03 12:15:00 | 1407.00 | 1423.61 | 1427.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-04 09:15:00 | 1417.10 | 1415.42 | 1421.40 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-04 09:15:00 | 1417.10 | 1415.42 | 1421.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 09:15:00 | 1417.10 | 1415.42 | 1421.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-04 09:45:00 | 1417.10 | 1415.42 | 1421.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 10:15:00 | 1415.20 | 1415.38 | 1420.83 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-04 14:15:00 | 1410.30 | 1415.28 | 1419.42 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-05 10:15:00 | 1426.90 | 1414.53 | 1417.32 | SL hit (close>static) qty=1.00 sl=1421.90 alert=retest2 |

### Cycle 99 — BUY (started 2025-12-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-10 12:15:00 | 1394.60 | 1385.05 | 1384.42 | EMA200 above EMA400 |

### Cycle 100 — SELL (started 2025-12-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-10 15:15:00 | 1379.90 | 1383.79 | 1384.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-11 09:15:00 | 1365.80 | 1380.19 | 1382.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-12 09:15:00 | 1374.70 | 1373.77 | 1377.20 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-12 09:15:00 | 1374.70 | 1373.77 | 1377.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 09:15:00 | 1374.70 | 1373.77 | 1377.20 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-12 10:45:00 | 1366.00 | 1372.75 | 1376.42 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-12 11:45:00 | 1367.60 | 1371.58 | 1375.56 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-15 09:15:00 | 1359.10 | 1370.79 | 1373.71 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-15 12:15:00 | 1380.00 | 1375.74 | 1375.34 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 101 — BUY (started 2025-12-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-15 12:15:00 | 1380.00 | 1375.74 | 1375.34 | EMA200 above EMA400 |

### Cycle 102 — SELL (started 2025-12-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-15 14:15:00 | 1365.00 | 1373.52 | 1374.39 | EMA200 below EMA400 |

### Cycle 103 — BUY (started 2025-12-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-16 14:15:00 | 1384.00 | 1374.27 | 1373.66 | EMA200 above EMA400 |

### Cycle 104 — SELL (started 2025-12-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-17 09:15:00 | 1357.20 | 1370.17 | 1371.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-17 12:15:00 | 1348.00 | 1362.64 | 1367.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-18 14:15:00 | 1340.30 | 1337.20 | 1347.45 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-18 15:00:00 | 1340.30 | 1337.20 | 1347.45 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 15:15:00 | 1348.00 | 1339.36 | 1347.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 09:15:00 | 1361.00 | 1339.36 | 1347.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 09:15:00 | 1352.40 | 1341.97 | 1347.94 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-19 11:00:00 | 1343.80 | 1342.34 | 1347.57 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-22 12:15:00 | 1336.00 | 1345.38 | 1345.38 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-24 09:30:00 | 1343.00 | 1339.97 | 1339.98 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-24 10:15:00 | 1351.60 | 1342.29 | 1341.04 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 105 — BUY (started 2025-12-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-24 10:15:00 | 1351.60 | 1342.29 | 1341.04 | EMA200 above EMA400 |

### Cycle 106 — SELL (started 2025-12-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-26 14:15:00 | 1340.00 | 1342.58 | 1342.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-29 09:15:00 | 1331.20 | 1340.20 | 1341.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-29 13:15:00 | 1344.60 | 1338.23 | 1339.87 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-29 13:15:00 | 1344.60 | 1338.23 | 1339.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 13:15:00 | 1344.60 | 1338.23 | 1339.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-29 14:00:00 | 1344.60 | 1338.23 | 1339.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 14:15:00 | 1336.00 | 1337.78 | 1339.52 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-29 15:15:00 | 1334.00 | 1337.78 | 1339.52 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-30 09:15:00 | 1368.00 | 1343.22 | 1341.65 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 107 — BUY (started 2025-12-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-30 09:15:00 | 1368.00 | 1343.22 | 1341.65 | EMA200 above EMA400 |

### Cycle 108 — SELL (started 2026-01-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-01 12:15:00 | 1339.20 | 1345.03 | 1345.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-02 09:15:00 | 1324.50 | 1338.85 | 1342.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-05 10:15:00 | 1329.60 | 1319.71 | 1327.95 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-05 10:15:00 | 1329.60 | 1319.71 | 1327.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 10:15:00 | 1329.60 | 1319.71 | 1327.95 | EMA400 retest candle locked (from downside) |

### Cycle 109 — BUY (started 2026-01-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-05 14:15:00 | 1339.20 | 1332.80 | 1332.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-06 12:15:00 | 1347.20 | 1336.61 | 1334.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-07 14:15:00 | 1367.90 | 1377.46 | 1361.95 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-07 15:00:00 | 1367.90 | 1377.46 | 1361.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 15:15:00 | 1370.00 | 1375.96 | 1362.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-08 09:15:00 | 1353.30 | 1375.96 | 1362.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 09:15:00 | 1343.30 | 1369.43 | 1360.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-08 10:00:00 | 1343.30 | 1369.43 | 1360.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 10:15:00 | 1331.20 | 1361.79 | 1358.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-08 11:00:00 | 1331.20 | 1361.79 | 1358.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 110 — SELL (started 2026-01-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-08 12:15:00 | 1337.20 | 1353.14 | 1354.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-09 09:15:00 | 1332.10 | 1345.34 | 1350.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-12 11:15:00 | 1332.00 | 1322.17 | 1331.87 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-12 11:15:00 | 1332.00 | 1322.17 | 1331.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 11:15:00 | 1332.00 | 1322.17 | 1331.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-12 12:00:00 | 1332.00 | 1322.17 | 1331.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 12:15:00 | 1330.40 | 1323.82 | 1331.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-12 12:30:00 | 1335.80 | 1323.82 | 1331.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 13:15:00 | 1331.70 | 1325.40 | 1331.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-12 13:45:00 | 1331.90 | 1325.40 | 1331.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 14:15:00 | 1336.90 | 1327.70 | 1332.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-12 14:45:00 | 1341.00 | 1327.70 | 1332.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 15:15:00 | 1334.00 | 1328.96 | 1332.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-13 09:15:00 | 1341.90 | 1328.96 | 1332.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 111 — BUY (started 2026-01-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-13 10:15:00 | 1356.60 | 1337.28 | 1335.73 | EMA200 above EMA400 |

### Cycle 112 — SELL (started 2026-01-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-16 12:15:00 | 1326.60 | 1340.77 | 1342.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-16 13:15:00 | 1324.60 | 1337.53 | 1340.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-16 14:15:00 | 1346.20 | 1339.27 | 1341.45 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-16 14:15:00 | 1346.20 | 1339.27 | 1341.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 14:15:00 | 1346.20 | 1339.27 | 1341.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-16 15:00:00 | 1346.20 | 1339.27 | 1341.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 15:15:00 | 1356.30 | 1342.67 | 1342.80 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-19 09:15:00 | 1342.50 | 1342.67 | 1342.80 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-19 09:15:00 | 1346.20 | 1343.38 | 1343.11 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 113 — BUY (started 2026-01-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-19 09:15:00 | 1346.20 | 1343.38 | 1343.11 | EMA200 above EMA400 |

### Cycle 114 — SELL (started 2026-01-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-19 10:15:00 | 1325.10 | 1339.72 | 1341.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-19 11:15:00 | 1313.00 | 1334.38 | 1338.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-22 09:15:00 | 1253.50 | 1246.40 | 1268.05 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-22 10:00:00 | 1253.50 | 1246.40 | 1268.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 13:15:00 | 1259.20 | 1251.50 | 1263.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 14:00:00 | 1259.20 | 1251.50 | 1263.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 14:15:00 | 1262.20 | 1253.64 | 1263.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 14:45:00 | 1263.50 | 1253.64 | 1263.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 15:15:00 | 1250.40 | 1252.99 | 1262.41 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-23 10:45:00 | 1249.80 | 1252.54 | 1260.57 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-23 11:15:00 | 1247.20 | 1252.54 | 1260.57 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-28 09:15:00 | 1187.31 | 1226.86 | 1240.20 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-28 09:15:00 | 1184.84 | 1226.86 | 1240.20 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-29 10:15:00 | 1208.50 | 1205.12 | 1219.26 | SL hit (close>ema200) qty=0.50 sl=1205.12 alert=retest2 |

### Cycle 115 — BUY (started 2026-02-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-04 09:15:00 | 1169.60 | 1154.62 | 1153.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-04 13:15:00 | 1190.60 | 1164.92 | 1158.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-05 09:15:00 | 1164.40 | 1169.18 | 1162.63 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-05 09:15:00 | 1164.40 | 1169.18 | 1162.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 09:15:00 | 1164.40 | 1169.18 | 1162.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-05 09:45:00 | 1165.70 | 1169.18 | 1162.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 10:15:00 | 1163.40 | 1168.02 | 1162.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-05 10:30:00 | 1162.50 | 1168.02 | 1162.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 11:15:00 | 1152.00 | 1164.82 | 1161.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-05 12:00:00 | 1152.00 | 1164.82 | 1161.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 12:15:00 | 1149.20 | 1161.69 | 1160.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-05 12:30:00 | 1147.30 | 1161.69 | 1160.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 116 — SELL (started 2026-02-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-05 13:15:00 | 1147.20 | 1158.79 | 1159.37 | EMA200 below EMA400 |

### Cycle 117 — BUY (started 2026-02-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-05 14:15:00 | 1177.70 | 1162.58 | 1161.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-06 09:15:00 | 1243.00 | 1177.69 | 1168.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-09 15:15:00 | 1267.50 | 1275.71 | 1250.26 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-10 09:15:00 | 1265.00 | 1275.71 | 1250.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 09:15:00 | 1257.10 | 1274.75 | 1263.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-11 09:30:00 | 1257.50 | 1274.75 | 1263.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 10:15:00 | 1252.60 | 1270.32 | 1262.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-11 10:45:00 | 1252.70 | 1270.32 | 1262.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 11:15:00 | 1246.50 | 1265.56 | 1261.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-11 11:30:00 | 1249.00 | 1265.56 | 1261.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 118 — SELL (started 2026-02-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-11 13:15:00 | 1240.80 | 1257.68 | 1258.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-12 09:15:00 | 1235.00 | 1249.55 | 1254.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-18 12:15:00 | 1150.90 | 1149.55 | 1162.61 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-18 13:00:00 | 1150.90 | 1149.55 | 1162.61 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 09:15:00 | 1146.60 | 1148.70 | 1158.07 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-19 10:15:00 | 1143.60 | 1148.70 | 1158.07 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-19 11:30:00 | 1143.00 | 1146.47 | 1155.42 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-19 13:30:00 | 1143.60 | 1143.71 | 1152.62 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-23 11:15:00 | 1086.42 | 1102.13 | 1118.27 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-23 11:15:00 | 1085.85 | 1102.13 | 1118.27 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-23 11:15:00 | 1086.42 | 1102.13 | 1118.27 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-23 14:15:00 | 1097.80 | 1096.18 | 1111.09 | SL hit (close>ema200) qty=0.50 sl=1096.18 alert=retest2 |

### Cycle 119 — BUY (started 2026-02-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-25 11:15:00 | 1209.50 | 1121.61 | 1110.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-25 12:15:00 | 1229.90 | 1143.27 | 1121.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-27 11:15:00 | 1226.10 | 1232.83 | 1204.20 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-27 11:45:00 | 1228.10 | 1232.83 | 1204.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 09:15:00 | 1213.10 | 1223.11 | 1209.88 | EMA400 retest candle locked (from upside) |

### Cycle 120 — SELL (started 2026-03-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-04 09:15:00 | 1175.60 | 1201.97 | 1204.47 | EMA200 below EMA400 |

### Cycle 121 — BUY (started 2026-03-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-05 12:15:00 | 1213.20 | 1203.12 | 1202.30 | EMA200 above EMA400 |

### Cycle 122 — SELL (started 2026-03-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-09 09:15:00 | 1169.00 | 1198.86 | 1201.48 | EMA200 below EMA400 |

### Cycle 123 — BUY (started 2026-03-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-10 13:15:00 | 1207.40 | 1193.79 | 1192.80 | EMA200 above EMA400 |

### Cycle 124 — SELL (started 2026-03-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-11 13:15:00 | 1165.90 | 1188.00 | 1190.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-12 09:15:00 | 1127.10 | 1168.70 | 1180.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-12 12:15:00 | 1173.80 | 1167.08 | 1176.76 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-12 13:00:00 | 1173.80 | 1167.08 | 1176.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 13:15:00 | 1173.20 | 1168.30 | 1176.43 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-13 09:15:00 | 1156.00 | 1171.38 | 1176.49 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-13 13:15:00 | 1188.50 | 1170.17 | 1173.19 | SL hit (close>static) qty=1.00 sl=1182.40 alert=retest2 |

### Cycle 125 — BUY (started 2026-04-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-06 13:15:00 | 1036.50 | 1025.55 | 1024.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-08 09:15:00 | 1057.80 | 1034.73 | 1029.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-09 14:15:00 | 1058.90 | 1065.52 | 1054.61 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-09 14:15:00 | 1058.90 | 1065.52 | 1054.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 14:15:00 | 1058.90 | 1065.52 | 1054.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-09 15:00:00 | 1058.90 | 1065.52 | 1054.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 15:15:00 | 1060.00 | 1064.42 | 1055.10 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-10 09:15:00 | 1069.50 | 1064.42 | 1055.10 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-13 09:15:00 | 1046.50 | 1060.86 | 1059.08 | SL hit (close<static) qty=1.00 sl=1052.10 alert=retest2 |

### Cycle 126 — SELL (started 2026-04-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-13 12:15:00 | 1048.00 | 1056.43 | 1057.36 | EMA200 below EMA400 |

### Cycle 127 — BUY (started 2026-04-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-15 10:15:00 | 1060.40 | 1057.60 | 1057.40 | EMA200 above EMA400 |

### Cycle 128 — SELL (started 2026-04-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-15 11:15:00 | 1049.00 | 1055.88 | 1056.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-15 12:15:00 | 1044.60 | 1053.62 | 1055.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-16 09:15:00 | 1048.00 | 1047.54 | 1051.61 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-16 09:15:00 | 1048.00 | 1047.54 | 1051.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-16 09:15:00 | 1048.00 | 1047.54 | 1051.61 | EMA400 retest candle locked (from downside) |

### Cycle 129 — BUY (started 2026-04-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-17 11:15:00 | 1094.50 | 1057.62 | 1053.11 | EMA200 above EMA400 |

### Cycle 130 — SELL (started 2026-04-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-21 13:15:00 | 1056.70 | 1062.36 | 1062.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-21 14:15:00 | 1048.90 | 1059.67 | 1061.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-22 14:15:00 | 1052.00 | 1050.46 | 1054.57 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-22 14:15:00 | 1052.00 | 1050.46 | 1054.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-22 14:15:00 | 1052.00 | 1050.46 | 1054.57 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-23 09:15:00 | 1046.90 | 1050.89 | 1054.39 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-23 09:15:00 | 1065.30 | 1053.77 | 1055.39 | SL hit (close>static) qty=1.00 sl=1056.00 alert=retest2 |

### Cycle 131 — BUY (started 2026-04-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 11:15:00 | 1067.60 | 1046.36 | 1046.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-28 09:15:00 | 1071.70 | 1057.77 | 1052.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-28 14:15:00 | 1071.30 | 1075.36 | 1064.12 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-28 15:00:00 | 1071.30 | 1075.36 | 1064.12 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 10:15:00 | 1173.60 | 1179.71 | 1159.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-05 10:45:00 | 1153.00 | 1179.71 | 1159.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-07 10:15:00 | 1200.00 | 1210.22 | 1195.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-07 10:30:00 | 1198.00 | 1210.22 | 1195.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-07 12:15:00 | 1197.90 | 1205.93 | 1195.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-07 13:00:00 | 1197.90 | 1205.93 | 1195.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-07 13:15:00 | 1199.90 | 1204.72 | 1196.16 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-08 09:15:00 | 1201.50 | 1201.53 | 1196.14 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-05-08 12:15:00 | 1183.50 | 1195.21 | 1194.60 | SL hit (close<static) qty=1.00 sl=1191.60 alert=retest2 |

### Cycle 132 — SELL (started 2026-05-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-08 13:15:00 | 1174.50 | 1191.07 | 1192.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-05-08 15:15:00 | 1168.40 | 1184.25 | 1189.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-05-11 10:15:00 | 1193.60 | 1183.89 | 1188.07 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-11 10:15:00 | 1193.60 | 1183.89 | 1188.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-11 10:15:00 | 1193.60 | 1183.89 | 1188.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-11 11:00:00 | 1193.60 | 1183.89 | 1188.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-11 11:15:00 | 1195.90 | 1186.29 | 1188.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-11 11:45:00 | 1197.20 | 1186.29 | 1188.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 133 — BUY (started 2026-05-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-11 12:15:00 | 1212.00 | 1191.43 | 1190.89 | EMA200 above EMA400 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2024-05-13 12:45:00 | 1520.80 | 2024-05-17 09:15:00 | 1444.71 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-05-13 12:45:00 | 1520.80 | 2024-05-18 09:15:00 | 1460.40 | STOP_HIT | 0.50 | 3.97% |
| SELL | retest2 | 2024-05-27 14:15:00 | 1431.85 | 2024-05-28 09:15:00 | 1360.26 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-05-27 14:15:00 | 1431.85 | 2024-05-29 09:15:00 | 1493.00 | STOP_HIT | 0.50 | -4.27% |
| SELL | retest2 | 2024-05-28 09:15:00 | 1392.00 | 2024-05-29 09:15:00 | 1493.00 | STOP_HIT | 1.00 | -7.26% |
| SELL | retest2 | 2024-06-26 12:00:00 | 1528.90 | 2024-06-27 14:15:00 | 1559.05 | STOP_HIT | 1.00 | -1.97% |
| BUY | retest2 | 2024-07-10 09:15:00 | 1711.35 | 2024-07-12 11:15:00 | 1673.15 | STOP_HIT | 1.00 | -2.23% |
| BUY | retest2 | 2024-07-10 11:15:00 | 1708.60 | 2024-07-12 11:15:00 | 1673.15 | STOP_HIT | 1.00 | -2.07% |
| BUY | retest2 | 2024-07-12 09:15:00 | 1702.70 | 2024-07-12 11:15:00 | 1673.15 | STOP_HIT | 1.00 | -1.74% |
| BUY | retest2 | 2024-07-12 13:30:00 | 1708.05 | 2024-07-16 13:15:00 | 1688.85 | STOP_HIT | 1.00 | -1.12% |
| BUY | retest2 | 2024-07-15 09:45:00 | 1697.85 | 2024-07-16 13:15:00 | 1688.85 | STOP_HIT | 1.00 | -0.53% |
| SELL | retest2 | 2024-07-19 15:15:00 | 1662.95 | 2024-07-22 12:15:00 | 1702.50 | STOP_HIT | 1.00 | -2.38% |
| SELL | retest2 | 2024-07-22 13:45:00 | 1667.55 | 2024-07-23 13:15:00 | 1711.15 | STOP_HIT | 1.00 | -2.61% |
| SELL | retest2 | 2024-07-23 12:15:00 | 1666.40 | 2024-07-23 13:15:00 | 1711.15 | STOP_HIT | 1.00 | -2.69% |
| BUY | retest2 | 2024-09-10 12:15:00 | 1863.95 | 2024-09-12 15:15:00 | 2050.35 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-09-10 13:45:00 | 1863.30 | 2024-09-12 15:15:00 | 2049.63 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-09-10 14:45:00 | 1863.55 | 2024-09-12 15:15:00 | 2049.91 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-10-04 13:15:00 | 1956.00 | 2024-10-07 10:15:00 | 1858.20 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-04 13:15:00 | 1956.00 | 2024-10-08 10:15:00 | 1869.90 | STOP_HIT | 0.50 | 4.40% |
| BUY | retest2 | 2024-10-16 12:15:00 | 1935.65 | 2024-10-21 12:15:00 | 1960.75 | STOP_HIT | 1.00 | 1.30% |
| BUY | retest2 | 2024-10-16 13:00:00 | 1937.90 | 2024-10-21 12:15:00 | 1960.75 | STOP_HIT | 1.00 | 1.18% |
| BUY | retest2 | 2024-10-16 15:00:00 | 2000.25 | 2024-10-21 12:15:00 | 1960.75 | STOP_HIT | 1.00 | -1.97% |
| SELL | retest1 | 2024-10-24 11:15:00 | 1829.30 | 2024-10-28 13:15:00 | 1824.60 | STOP_HIT | 1.00 | 0.26% |
| SELL | retest1 | 2024-10-24 11:45:00 | 1823.10 | 2024-10-28 13:15:00 | 1824.60 | STOP_HIT | 1.00 | -0.08% |
| SELL | retest2 | 2024-10-28 09:15:00 | 1787.75 | 2024-10-28 15:15:00 | 1846.50 | STOP_HIT | 1.00 | -3.29% |
| BUY | retest2 | 2024-11-04 11:15:00 | 1819.50 | 2024-11-04 12:15:00 | 1822.15 | STOP_HIT | 1.00 | 0.15% |
| BUY | retest2 | 2024-11-21 09:15:00 | 1953.70 | 2024-11-28 15:15:00 | 2035.95 | STOP_HIT | 1.00 | 4.21% |
| SELL | retest1 | 2024-12-17 09:15:00 | 2096.00 | 2024-12-18 11:15:00 | 2143.50 | STOP_HIT | 1.00 | -2.27% |
| SELL | retest2 | 2024-12-24 09:30:00 | 2105.05 | 2024-12-24 13:15:00 | 2136.95 | STOP_HIT | 1.00 | -1.52% |
| SELL | retest2 | 2024-12-24 11:15:00 | 2111.00 | 2024-12-24 13:15:00 | 2136.95 | STOP_HIT | 1.00 | -1.23% |
| BUY | retest2 | 2024-12-30 11:30:00 | 2215.50 | 2025-01-01 12:15:00 | 2170.80 | STOP_HIT | 1.00 | -2.02% |
| BUY | retest2 | 2024-12-30 12:15:00 | 2212.10 | 2025-01-01 12:15:00 | 2170.80 | STOP_HIT | 1.00 | -1.87% |
| BUY | retest2 | 2024-12-30 13:45:00 | 2214.95 | 2025-01-01 12:15:00 | 2170.80 | STOP_HIT | 1.00 | -1.99% |
| BUY | retest2 | 2024-12-31 09:15:00 | 2239.45 | 2025-01-01 12:15:00 | 2170.80 | STOP_HIT | 1.00 | -3.07% |
| BUY | retest2 | 2025-01-01 11:15:00 | 2193.00 | 2025-01-01 12:15:00 | 2170.80 | STOP_HIT | 1.00 | -1.01% |
| BUY | retest2 | 2025-01-10 11:15:00 | 2247.95 | 2025-01-15 14:15:00 | 2238.25 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest2 | 2025-01-10 12:30:00 | 2244.65 | 2025-01-15 14:15:00 | 2238.25 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest2 | 2025-01-13 09:15:00 | 2285.00 | 2025-01-15 14:15:00 | 2238.25 | STOP_HIT | 1.00 | -2.05% |
| BUY | retest2 | 2025-01-13 13:45:00 | 2253.00 | 2025-01-15 14:15:00 | 2238.25 | STOP_HIT | 1.00 | -0.65% |
| SELL | retest2 | 2025-01-21 14:15:00 | 2167.95 | 2025-01-27 09:15:00 | 2059.55 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-21 14:15:00 | 2167.95 | 2025-01-27 10:15:00 | 1951.15 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2025-02-04 09:15:00 | 2160.80 | 2025-02-10 09:15:00 | 2376.88 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-02-13 11:30:00 | 2145.85 | 2025-02-14 09:15:00 | 1931.26 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-03-17 13:45:00 | 1600.05 | 2025-03-18 15:15:00 | 1651.05 | STOP_HIT | 1.00 | -3.19% |
| SELL | retest2 | 2025-03-17 15:00:00 | 1596.35 | 2025-03-18 15:15:00 | 1651.05 | STOP_HIT | 1.00 | -3.43% |
| BUY | retest2 | 2025-03-20 09:15:00 | 1670.00 | 2025-03-25 12:15:00 | 1636.85 | STOP_HIT | 1.00 | -1.99% |
| BUY | retest2 | 2025-03-20 10:15:00 | 1658.60 | 2025-03-25 12:15:00 | 1636.85 | STOP_HIT | 1.00 | -1.31% |
| BUY | retest2 | 2025-03-20 10:45:00 | 1656.25 | 2025-03-25 12:15:00 | 1636.85 | STOP_HIT | 1.00 | -1.17% |
| BUY | retest2 | 2025-03-20 13:45:00 | 1662.70 | 2025-03-25 12:15:00 | 1636.85 | STOP_HIT | 1.00 | -1.55% |
| BUY | retest2 | 2025-03-21 09:15:00 | 1672.00 | 2025-03-25 12:15:00 | 1636.85 | STOP_HIT | 1.00 | -2.10% |
| BUY | retest2 | 2025-03-21 13:15:00 | 1650.65 | 2025-03-25 12:15:00 | 1636.85 | STOP_HIT | 1.00 | -0.84% |
| BUY | retest2 | 2025-03-25 09:30:00 | 1653.05 | 2025-03-25 12:15:00 | 1636.85 | STOP_HIT | 1.00 | -0.98% |
| SELL | retest2 | 2025-04-08 15:00:00 | 1591.70 | 2025-04-09 11:15:00 | 1637.70 | STOP_HIT | 1.00 | -2.89% |
| SELL | retest2 | 2025-04-09 13:45:00 | 1605.95 | 2025-04-11 09:15:00 | 1652.00 | STOP_HIT | 1.00 | -2.87% |
| SELL | retest2 | 2025-04-09 14:45:00 | 1599.50 | 2025-04-11 09:15:00 | 1652.00 | STOP_HIT | 1.00 | -3.28% |
| SELL | retest2 | 2025-04-09 15:15:00 | 1605.00 | 2025-04-11 09:15:00 | 1652.00 | STOP_HIT | 1.00 | -2.93% |
| SELL | retest2 | 2025-04-24 13:15:00 | 1655.10 | 2025-04-28 09:15:00 | 1572.34 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-04-24 13:45:00 | 1648.90 | 2025-04-28 09:15:00 | 1566.45 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-04-24 13:15:00 | 1655.10 | 2025-04-29 10:15:00 | 1585.50 | STOP_HIT | 0.50 | 4.21% |
| SELL | retest2 | 2025-04-24 13:45:00 | 1648.90 | 2025-04-29 10:15:00 | 1585.50 | STOP_HIT | 0.50 | 3.84% |
| BUY | retest2 | 2025-05-09 12:15:00 | 1504.10 | 2025-05-16 14:15:00 | 1497.50 | STOP_HIT | 1.00 | -0.44% |
| BUY | retest2 | 2025-05-28 09:30:00 | 1557.00 | 2025-05-30 09:15:00 | 1712.70 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-06-20 10:15:00 | 1935.10 | 2025-06-20 14:15:00 | 1838.34 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-06-20 11:00:00 | 1935.70 | 2025-06-20 14:15:00 | 1838.91 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-06-20 10:15:00 | 1935.10 | 2025-06-25 09:15:00 | 1821.70 | STOP_HIT | 0.50 | 5.86% |
| SELL | retest2 | 2025-06-20 11:00:00 | 1935.70 | 2025-06-25 09:15:00 | 1821.70 | STOP_HIT | 0.50 | 5.89% |
| BUY | retest2 | 2025-07-16 11:15:00 | 1849.00 | 2025-07-25 11:15:00 | 1898.10 | STOP_HIT | 1.00 | 2.66% |
| BUY | retest2 | 2025-07-16 12:30:00 | 1849.90 | 2025-07-25 11:15:00 | 1898.10 | STOP_HIT | 1.00 | 2.61% |
| BUY | retest2 | 2025-07-16 14:15:00 | 1848.10 | 2025-07-25 11:15:00 | 1898.10 | STOP_HIT | 1.00 | 2.71% |
| BUY | retest2 | 2025-07-17 09:15:00 | 1848.90 | 2025-07-25 11:15:00 | 1898.10 | STOP_HIT | 1.00 | 2.66% |
| BUY | retest2 | 2025-07-17 10:45:00 | 1876.70 | 2025-07-25 11:15:00 | 1898.10 | STOP_HIT | 1.00 | 1.14% |
| BUY | retest2 | 2025-07-17 12:00:00 | 1898.30 | 2025-07-25 11:15:00 | 1898.10 | STOP_HIT | 1.00 | -0.01% |
| SELL | retest2 | 2025-08-06 10:15:00 | 1654.10 | 2025-08-11 09:15:00 | 1488.69 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-08-06 15:15:00 | 1650.00 | 2025-08-11 09:15:00 | 1485.00 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-08-29 14:00:00 | 1692.90 | 2025-09-08 15:15:00 | 1608.26 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-01 10:00:00 | 1692.00 | 2025-09-08 15:15:00 | 1607.40 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-01 11:15:00 | 1685.50 | 2025-09-09 14:15:00 | 1601.22 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-02 09:15:00 | 1682.00 | 2025-09-09 14:15:00 | 1597.90 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-02 12:45:00 | 1674.10 | 2025-09-09 14:15:00 | 1590.39 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-03 09:15:00 | 1657.30 | 2025-09-09 14:15:00 | 1574.43 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-03 12:15:00 | 1671.40 | 2025-09-09 14:15:00 | 1587.83 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-04 10:30:00 | 1672.50 | 2025-09-09 14:15:00 | 1588.88 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-05 12:00:00 | 1665.90 | 2025-09-09 14:15:00 | 1582.61 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-08 09:45:00 | 1663.30 | 2025-09-09 14:15:00 | 1580.13 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-29 14:00:00 | 1692.90 | 2025-09-10 09:15:00 | 1631.10 | STOP_HIT | 0.50 | 3.65% |
| SELL | retest2 | 2025-09-01 10:00:00 | 1692.00 | 2025-09-10 09:15:00 | 1631.10 | STOP_HIT | 0.50 | 3.60% |
| SELL | retest2 | 2025-09-01 11:15:00 | 1685.50 | 2025-09-10 09:15:00 | 1631.10 | STOP_HIT | 0.50 | 3.23% |
| SELL | retest2 | 2025-09-02 09:15:00 | 1682.00 | 2025-09-10 09:15:00 | 1631.10 | STOP_HIT | 0.50 | 3.03% |
| SELL | retest2 | 2025-09-02 12:45:00 | 1674.10 | 2025-09-10 09:15:00 | 1631.10 | STOP_HIT | 0.50 | 2.57% |
| SELL | retest2 | 2025-09-03 09:15:00 | 1657.30 | 2025-09-10 09:15:00 | 1631.10 | STOP_HIT | 0.50 | 1.58% |
| SELL | retest2 | 2025-09-03 12:15:00 | 1671.40 | 2025-09-10 09:15:00 | 1631.10 | STOP_HIT | 0.50 | 2.41% |
| SELL | retest2 | 2025-09-04 10:30:00 | 1672.50 | 2025-09-10 09:15:00 | 1631.10 | STOP_HIT | 0.50 | 2.48% |
| SELL | retest2 | 2025-09-05 12:00:00 | 1665.90 | 2025-09-10 09:15:00 | 1631.10 | STOP_HIT | 0.50 | 2.09% |
| SELL | retest2 | 2025-09-08 09:45:00 | 1663.30 | 2025-09-10 09:15:00 | 1631.10 | STOP_HIT | 0.50 | 1.94% |
| SELL | retest2 | 2025-09-25 14:45:00 | 1589.50 | 2025-09-30 10:15:00 | 1599.60 | STOP_HIT | 1.00 | -0.64% |
| SELL | retest2 | 2025-10-10 11:30:00 | 1585.50 | 2025-10-15 09:15:00 | 1507.93 | PARTIAL | 0.50 | 4.89% |
| SELL | retest2 | 2025-10-10 15:00:00 | 1587.30 | 2025-10-15 10:15:00 | 1506.22 | PARTIAL | 0.50 | 5.11% |
| SELL | retest2 | 2025-10-13 10:00:00 | 1586.00 | 2025-10-15 10:15:00 | 1506.70 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-10 11:30:00 | 1585.50 | 2025-10-16 09:15:00 | 1509.60 | STOP_HIT | 0.50 | 4.79% |
| SELL | retest2 | 2025-10-10 15:00:00 | 1587.30 | 2025-10-16 09:15:00 | 1509.60 | STOP_HIT | 0.50 | 4.90% |
| SELL | retest2 | 2025-10-13 10:00:00 | 1586.00 | 2025-10-16 09:15:00 | 1509.60 | STOP_HIT | 0.50 | 4.82% |
| SELL | retest1 | 2025-10-29 10:15:00 | 1435.20 | 2025-10-29 15:15:00 | 1456.80 | STOP_HIT | 1.00 | -1.51% |
| SELL | retest2 | 2025-10-31 15:00:00 | 1437.50 | 2025-11-03 09:15:00 | 1468.00 | STOP_HIT | 1.00 | -2.12% |
| BUY | retest2 | 2025-11-14 12:00:00 | 1488.10 | 2025-11-19 10:15:00 | 1455.20 | STOP_HIT | 1.00 | -2.21% |
| SELL | retest2 | 2025-12-04 14:15:00 | 1410.30 | 2025-12-05 10:15:00 | 1426.90 | STOP_HIT | 1.00 | -1.18% |
| SELL | retest2 | 2025-12-05 14:45:00 | 1406.90 | 2025-12-09 10:15:00 | 1336.56 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-05 14:45:00 | 1406.90 | 2025-12-09 13:15:00 | 1365.30 | STOP_HIT | 0.50 | 2.96% |
| SELL | retest2 | 2025-12-12 10:45:00 | 1366.00 | 2025-12-15 12:15:00 | 1380.00 | STOP_HIT | 1.00 | -1.02% |
| SELL | retest2 | 2025-12-12 11:45:00 | 1367.60 | 2025-12-15 12:15:00 | 1380.00 | STOP_HIT | 1.00 | -0.91% |
| SELL | retest2 | 2025-12-15 09:15:00 | 1359.10 | 2025-12-15 12:15:00 | 1380.00 | STOP_HIT | 1.00 | -1.54% |
| SELL | retest2 | 2025-12-19 11:00:00 | 1343.80 | 2025-12-24 10:15:00 | 1351.60 | STOP_HIT | 1.00 | -0.58% |
| SELL | retest2 | 2025-12-22 12:15:00 | 1336.00 | 2025-12-24 10:15:00 | 1351.60 | STOP_HIT | 1.00 | -1.17% |
| SELL | retest2 | 2025-12-24 09:30:00 | 1343.00 | 2025-12-24 10:15:00 | 1351.60 | STOP_HIT | 1.00 | -0.64% |
| SELL | retest2 | 2025-12-29 15:15:00 | 1334.00 | 2025-12-30 09:15:00 | 1368.00 | STOP_HIT | 1.00 | -2.55% |
| SELL | retest2 | 2026-01-19 09:15:00 | 1342.50 | 2026-01-19 09:15:00 | 1346.20 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest2 | 2026-01-23 10:45:00 | 1249.80 | 2026-01-28 09:15:00 | 1187.31 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-23 11:15:00 | 1247.20 | 2026-01-28 09:15:00 | 1184.84 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-23 10:45:00 | 1249.80 | 2026-01-29 10:15:00 | 1208.50 | STOP_HIT | 0.50 | 3.30% |
| SELL | retest2 | 2026-01-23 11:15:00 | 1247.20 | 2026-01-29 10:15:00 | 1208.50 | STOP_HIT | 0.50 | 3.10% |
| SELL | retest2 | 2026-02-19 10:15:00 | 1143.60 | 2026-02-23 11:15:00 | 1086.42 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-19 11:30:00 | 1143.00 | 2026-02-23 11:15:00 | 1085.85 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-19 13:30:00 | 1143.60 | 2026-02-23 11:15:00 | 1086.42 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-19 10:15:00 | 1143.60 | 2026-02-23 14:15:00 | 1097.80 | STOP_HIT | 0.50 | 4.00% |
| SELL | retest2 | 2026-02-19 11:30:00 | 1143.00 | 2026-02-23 14:15:00 | 1097.80 | STOP_HIT | 0.50 | 3.95% |
| SELL | retest2 | 2026-02-19 13:30:00 | 1143.60 | 2026-02-23 14:15:00 | 1097.80 | STOP_HIT | 0.50 | 4.00% |
| SELL | retest2 | 2026-03-13 09:15:00 | 1156.00 | 2026-03-13 13:15:00 | 1188.50 | STOP_HIT | 1.00 | -2.81% |
| SELL | retest2 | 2026-03-16 09:30:00 | 1167.80 | 2026-03-19 14:15:00 | 1109.41 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-16 09:30:00 | 1167.80 | 2026-03-20 15:15:00 | 1112.00 | STOP_HIT | 0.50 | 4.78% |
| BUY | retest2 | 2026-04-10 09:15:00 | 1069.50 | 2026-04-13 09:15:00 | 1046.50 | STOP_HIT | 1.00 | -2.15% |
| SELL | retest2 | 2026-04-23 09:15:00 | 1046.90 | 2026-04-23 09:15:00 | 1065.30 | STOP_HIT | 1.00 | -1.76% |
| SELL | retest2 | 2026-04-23 12:15:00 | 1048.00 | 2026-04-23 13:15:00 | 1060.10 | STOP_HIT | 1.00 | -1.15% |
| SELL | retest2 | 2026-04-24 09:45:00 | 1044.90 | 2026-04-27 09:15:00 | 1061.50 | STOP_HIT | 1.00 | -1.59% |
| SELL | retest2 | 2026-04-24 10:15:00 | 1044.30 | 2026-04-27 09:15:00 | 1061.50 | STOP_HIT | 1.00 | -1.65% |
| BUY | retest2 | 2026-05-08 09:15:00 | 1201.50 | 2026-05-08 12:15:00 | 1183.50 | STOP_HIT | 1.00 | -1.50% |
