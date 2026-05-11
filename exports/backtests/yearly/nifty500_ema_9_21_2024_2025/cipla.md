# Cipla Ltd. (CIPLA)

## Backtest Summary

- **Window:** 2024-03-12 15:15:00 → 2026-05-08 15:15:00 (3711 bars)
- **Last close:** 1348.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 155 |
| ALERT1 | 113 |
| ALERT2 | 110 |
| ALERT2_SKIP | 55 |
| ALERT3 | 293 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 1 |
| ENTRY2 | 110 |
| PARTIAL | 4 |
| TARGET_HIT | 1 |
| STOP_HIT | 108 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 113 (incl. partial bookings)
- **Trades open at end:** 2
- **Winners / losers:** 32 / 81
- **Target hits / Stop hits / Partials:** 1 / 108 / 4
- **Avg / median % per leg:** 0.05% / -0.62%
- **Sum % (uncompounded):** 5.94%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 53 | 9 | 17.0% | 0 | 53 | 0 | -0.55% | -29.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 53 | 9 | 17.0% | 0 | 53 | 0 | -0.55% | -29.0% |
| SELL (all) | 60 | 23 | 38.3% | 1 | 55 | 4 | 0.58% | 35.0% |
| SELL @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -1.71% | -1.7% |
| SELL @ 3rd Alert (retest2) | 59 | 23 | 39.0% | 1 | 54 | 4 | 0.62% | 36.7% |
| retest1 (combined) | 1 | 0 | 0.0% | 0 | 1 | 0 | -1.71% | -1.7% |
| retest2 (combined) | 112 | 32 | 28.6% | 1 | 107 | 4 | 0.07% | 7.6% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-05-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-13 10:15:00 | 1419.95 | 1384.21 | 1382.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-13 15:15:00 | 1421.00 | 1406.79 | 1395.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-14 09:15:00 | 1383.00 | 1402.03 | 1394.59 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-14 09:15:00 | 1383.00 | 1402.03 | 1394.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-14 09:15:00 | 1383.00 | 1402.03 | 1394.59 | EMA400 retest candle locked (from upside) |

### Cycle 2 — SELL (started 2024-05-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-14 12:15:00 | 1375.50 | 1388.15 | 1389.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-14 14:15:00 | 1359.25 | 1379.39 | 1384.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-15 09:15:00 | 1419.65 | 1384.05 | 1385.90 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-15 09:15:00 | 1419.65 | 1384.05 | 1385.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-15 09:15:00 | 1419.65 | 1384.05 | 1385.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-15 10:15:00 | 1407.90 | 1384.05 | 1385.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 3 — BUY (started 2024-05-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-15 10:15:00 | 1417.55 | 1390.75 | 1388.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-21 13:15:00 | 1436.70 | 1418.30 | 1410.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-24 10:15:00 | 1484.45 | 1486.55 | 1471.61 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-24 11:00:00 | 1484.45 | 1486.55 | 1471.61 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-27 09:15:00 | 1479.25 | 1485.84 | 1477.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-27 10:00:00 | 1479.25 | 1485.84 | 1477.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-27 10:15:00 | 1481.95 | 1485.06 | 1478.32 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-27 11:45:00 | 1486.00 | 1485.41 | 1479.09 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-28 09:30:00 | 1485.55 | 1482.85 | 1479.86 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-28 10:30:00 | 1487.60 | 1483.20 | 1480.29 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-29 11:15:00 | 1485.35 | 1481.41 | 1480.51 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-29 12:15:00 | 1490.00 | 1483.52 | 1481.64 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-29 13:15:00 | 1493.80 | 1483.52 | 1481.64 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-29 14:45:00 | 1492.40 | 1486.50 | 1483.39 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-30 09:15:00 | 1479.95 | 1486.30 | 1483.91 | SL hit (close<static) qty=1.00 sl=1480.00 alert=retest2 |

### Cycle 4 — SELL (started 2024-05-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-30 11:15:00 | 1473.35 | 1481.98 | 1482.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-30 13:15:00 | 1469.95 | 1477.93 | 1480.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-03 15:15:00 | 1458.00 | 1453.69 | 1459.23 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-03 15:15:00 | 1458.00 | 1453.69 | 1459.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-03 15:15:00 | 1458.00 | 1453.69 | 1459.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-04 09:15:00 | 1448.80 | 1453.69 | 1459.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 09:15:00 | 1472.05 | 1457.36 | 1460.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-04 10:00:00 | 1472.05 | 1457.36 | 1460.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 10:15:00 | 1445.65 | 1455.02 | 1459.05 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-04 12:00:00 | 1437.80 | 1451.57 | 1457.12 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-05 09:15:00 | 1500.90 | 1465.61 | 1461.80 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 5 — BUY (started 2024-06-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-05 09:15:00 | 1500.90 | 1465.61 | 1461.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-05 11:15:00 | 1503.65 | 1478.69 | 1468.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-05 15:15:00 | 1487.50 | 1489.78 | 1478.18 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-06 09:15:00 | 1470.20 | 1489.78 | 1478.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 09:15:00 | 1477.65 | 1487.35 | 1478.13 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-07 11:15:00 | 1502.35 | 1481.02 | 1478.27 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-10 09:15:00 | 1529.40 | 1490.57 | 1484.60 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-20 09:15:00 | 1543.05 | 1558.27 | 1558.33 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 6 — SELL (started 2024-06-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-20 09:15:00 | 1543.05 | 1558.27 | 1558.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-21 15:15:00 | 1537.25 | 1545.79 | 1549.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-27 14:15:00 | 1482.55 | 1478.44 | 1489.22 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-06-27 15:00:00 | 1482.55 | 1478.44 | 1489.22 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-28 09:15:00 | 1480.40 | 1479.88 | 1488.07 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-01 12:15:00 | 1475.35 | 1482.36 | 1484.84 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-01 13:00:00 | 1475.80 | 1481.04 | 1484.02 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-02 12:15:00 | 1485.95 | 1484.29 | 1484.24 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 7 — BUY (started 2024-07-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-02 12:15:00 | 1485.95 | 1484.29 | 1484.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-02 13:15:00 | 1490.30 | 1485.50 | 1484.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-03 09:15:00 | 1486.00 | 1486.64 | 1485.57 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-03 09:15:00 | 1486.00 | 1486.64 | 1485.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-03 09:15:00 | 1486.00 | 1486.64 | 1485.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-03 10:00:00 | 1486.00 | 1486.64 | 1485.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-03 10:15:00 | 1479.00 | 1485.11 | 1484.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-03 11:00:00 | 1479.00 | 1485.11 | 1484.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 8 — SELL (started 2024-07-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-03 11:15:00 | 1482.25 | 1484.54 | 1484.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-04 12:15:00 | 1474.55 | 1479.99 | 1482.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-04 14:15:00 | 1481.55 | 1479.81 | 1481.72 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-04 14:15:00 | 1481.55 | 1479.81 | 1481.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-04 14:15:00 | 1481.55 | 1479.81 | 1481.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-04 15:00:00 | 1481.55 | 1479.81 | 1481.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-04 15:15:00 | 1485.00 | 1480.85 | 1482.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-05 09:15:00 | 1506.15 | 1480.85 | 1482.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 9 — BUY (started 2024-07-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-05 09:15:00 | 1505.20 | 1485.72 | 1484.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-05 11:15:00 | 1513.95 | 1495.11 | 1488.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-08 10:15:00 | 1496.05 | 1503.30 | 1496.68 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-08 10:15:00 | 1496.05 | 1503.30 | 1496.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-08 10:15:00 | 1496.05 | 1503.30 | 1496.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-08 11:00:00 | 1496.05 | 1503.30 | 1496.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-08 11:15:00 | 1492.40 | 1501.12 | 1496.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-08 12:00:00 | 1492.40 | 1501.12 | 1496.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-08 12:15:00 | 1489.50 | 1498.80 | 1495.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-08 12:45:00 | 1489.00 | 1498.80 | 1495.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 10 — SELL (started 2024-07-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-08 15:15:00 | 1491.00 | 1493.33 | 1493.62 | EMA200 below EMA400 |

### Cycle 11 — BUY (started 2024-07-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-09 09:15:00 | 1510.20 | 1496.71 | 1495.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-09 14:15:00 | 1512.05 | 1504.09 | 1499.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-10 09:15:00 | 1504.50 | 1505.75 | 1501.27 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-10 09:15:00 | 1504.50 | 1505.75 | 1501.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-10 09:15:00 | 1504.50 | 1505.75 | 1501.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-10 10:00:00 | 1504.50 | 1505.75 | 1501.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-10 10:15:00 | 1498.60 | 1504.32 | 1501.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-10 10:30:00 | 1494.90 | 1504.32 | 1501.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-10 11:15:00 | 1501.95 | 1503.85 | 1501.11 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-10 12:45:00 | 1504.90 | 1503.95 | 1501.40 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-10 14:30:00 | 1507.85 | 1505.48 | 1502.52 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-11 14:30:00 | 1506.25 | 1503.35 | 1502.64 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-12 13:00:00 | 1506.00 | 1505.91 | 1504.40 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-15 15:15:00 | 1514.90 | 1518.29 | 1513.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-16 09:15:00 | 1517.40 | 1518.29 | 1513.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-16 09:15:00 | 1519.20 | 1518.47 | 1514.32 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-07-16 15:15:00 | 1508.40 | 1513.05 | 1513.21 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 12 — SELL (started 2024-07-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-16 15:15:00 | 1508.40 | 1513.05 | 1513.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-18 10:15:00 | 1496.95 | 1509.59 | 1511.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-18 13:15:00 | 1506.80 | 1506.37 | 1509.34 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-07-18 14:15:00 | 1507.20 | 1506.37 | 1509.34 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-18 14:15:00 | 1506.80 | 1506.46 | 1509.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-18 14:45:00 | 1506.15 | 1506.46 | 1509.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 10:15:00 | 1503.35 | 1493.88 | 1498.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 11:00:00 | 1503.35 | 1493.88 | 1498.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 11:15:00 | 1501.35 | 1495.38 | 1498.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 11:45:00 | 1501.70 | 1495.38 | 1498.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 12:15:00 | 1494.60 | 1495.22 | 1498.44 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-22 14:45:00 | 1493.60 | 1494.04 | 1497.35 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-23 10:00:00 | 1489.45 | 1492.66 | 1496.12 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-23 10:45:00 | 1493.65 | 1494.53 | 1496.66 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-23 12:00:00 | 1494.05 | 1494.43 | 1496.42 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 13:15:00 | 1493.55 | 1492.82 | 1495.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-23 14:00:00 | 1493.55 | 1492.82 | 1495.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 14:15:00 | 1497.95 | 1493.84 | 1495.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-23 15:00:00 | 1497.95 | 1493.84 | 1495.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 15:15:00 | 1495.00 | 1494.08 | 1495.45 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-24 09:15:00 | 1491.40 | 1494.08 | 1495.45 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-24 10:15:00 | 1499.15 | 1495.73 | 1496.01 | SL hit (close>static) qty=1.00 sl=1498.80 alert=retest2 |

### Cycle 13 — BUY (started 2024-07-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-24 11:15:00 | 1499.20 | 1496.43 | 1496.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-26 09:15:00 | 1521.00 | 1505.52 | 1502.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-29 09:15:00 | 1534.70 | 1544.03 | 1526.60 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-29 09:30:00 | 1537.35 | 1544.03 | 1526.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-30 09:15:00 | 1535.95 | 1545.71 | 1536.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-30 10:00:00 | 1535.95 | 1545.71 | 1536.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-30 10:15:00 | 1536.10 | 1543.79 | 1536.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-30 10:30:00 | 1534.80 | 1543.79 | 1536.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-30 11:15:00 | 1539.65 | 1542.96 | 1536.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-30 12:15:00 | 1535.50 | 1542.96 | 1536.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-30 12:15:00 | 1535.00 | 1541.37 | 1536.41 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-31 11:30:00 | 1541.35 | 1536.42 | 1535.30 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-01 11:30:00 | 1543.70 | 1541.52 | 1539.12 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-01 14:00:00 | 1540.95 | 1541.37 | 1539.47 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-02 10:15:00 | 1528.60 | 1538.48 | 1538.85 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 14 — SELL (started 2024-08-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-02 10:15:00 | 1528.60 | 1538.48 | 1538.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-02 15:15:00 | 1526.00 | 1533.08 | 1535.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-05 13:15:00 | 1519.05 | 1518.99 | 1526.86 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-05 14:00:00 | 1519.05 | 1518.99 | 1526.86 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-06 09:15:00 | 1533.85 | 1521.10 | 1525.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-06 09:45:00 | 1534.60 | 1521.10 | 1525.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-06 10:15:00 | 1538.10 | 1524.50 | 1526.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-06 11:00:00 | 1538.10 | 1524.50 | 1526.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 15 — BUY (started 2024-08-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-06 12:15:00 | 1535.10 | 1528.56 | 1528.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-07 09:15:00 | 1540.55 | 1532.47 | 1530.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-09 10:15:00 | 1563.00 | 1566.43 | 1557.44 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-09 10:30:00 | 1563.60 | 1566.43 | 1557.44 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-13 13:15:00 | 1584.45 | 1590.53 | 1582.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-13 14:00:00 | 1584.45 | 1590.53 | 1582.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-13 14:15:00 | 1583.05 | 1589.04 | 1582.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-13 15:00:00 | 1583.05 | 1589.04 | 1582.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-13 15:15:00 | 1585.00 | 1588.23 | 1582.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-14 09:15:00 | 1577.15 | 1588.23 | 1582.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-14 09:15:00 | 1576.45 | 1585.87 | 1582.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-14 09:30:00 | 1569.85 | 1585.87 | 1582.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-14 10:15:00 | 1577.25 | 1584.15 | 1581.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-14 11:00:00 | 1577.25 | 1584.15 | 1581.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-14 11:15:00 | 1573.55 | 1582.03 | 1581.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-14 12:00:00 | 1573.55 | 1582.03 | 1581.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 16 — SELL (started 2024-08-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-14 13:15:00 | 1563.85 | 1577.24 | 1578.93 | EMA200 below EMA400 |

### Cycle 17 — BUY (started 2024-08-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-19 11:15:00 | 1583.70 | 1576.56 | 1575.98 | EMA200 above EMA400 |

### Cycle 18 — SELL (started 2024-08-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-19 15:15:00 | 1570.00 | 1575.89 | 1575.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-20 09:15:00 | 1564.45 | 1573.60 | 1574.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-21 09:15:00 | 1582.00 | 1568.62 | 1570.52 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-21 09:15:00 | 1582.00 | 1568.62 | 1570.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-21 09:15:00 | 1582.00 | 1568.62 | 1570.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-21 10:00:00 | 1582.00 | 1568.62 | 1570.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-21 10:15:00 | 1583.95 | 1571.69 | 1571.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-21 10:45:00 | 1586.85 | 1571.69 | 1571.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 19 — BUY (started 2024-08-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-21 11:15:00 | 1586.95 | 1574.74 | 1573.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-21 12:15:00 | 1587.65 | 1577.32 | 1574.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-22 09:15:00 | 1583.40 | 1584.73 | 1579.45 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-22 10:00:00 | 1583.40 | 1584.73 | 1579.45 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-22 10:15:00 | 1581.40 | 1584.07 | 1579.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-22 11:00:00 | 1581.40 | 1584.07 | 1579.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-22 11:15:00 | 1582.10 | 1583.67 | 1579.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-22 11:30:00 | 1579.15 | 1583.67 | 1579.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-22 12:15:00 | 1585.70 | 1584.08 | 1580.39 | EMA400 retest candle locked (from upside) |

### Cycle 20 — SELL (started 2024-08-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-23 13:15:00 | 1577.00 | 1579.73 | 1579.96 | EMA200 below EMA400 |

### Cycle 21 — BUY (started 2024-08-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-26 10:15:00 | 1584.55 | 1580.62 | 1580.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-26 14:15:00 | 1594.55 | 1585.62 | 1582.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-27 14:15:00 | 1597.35 | 1600.40 | 1593.62 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-27 15:00:00 | 1597.35 | 1600.40 | 1593.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-04 09:15:00 | 1646.40 | 1649.95 | 1645.16 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-04 10:45:00 | 1650.00 | 1649.96 | 1645.60 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-04 14:30:00 | 1650.50 | 1650.23 | 1647.17 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-05 09:30:00 | 1651.30 | 1651.34 | 1648.24 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-05 12:30:00 | 1649.40 | 1648.96 | 1647.83 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-05 13:15:00 | 1638.50 | 1646.87 | 1646.98 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 22 — SELL (started 2024-09-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-05 13:15:00 | 1638.50 | 1646.87 | 1646.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-05 14:15:00 | 1628.45 | 1643.18 | 1645.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-09 09:15:00 | 1622.25 | 1620.90 | 1629.57 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-09 10:00:00 | 1622.25 | 1620.90 | 1629.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-09 10:15:00 | 1634.80 | 1623.68 | 1630.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-09 11:00:00 | 1634.80 | 1623.68 | 1630.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-09 11:15:00 | 1632.65 | 1625.48 | 1630.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-09 11:30:00 | 1635.05 | 1625.48 | 1630.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-09 12:15:00 | 1625.00 | 1625.38 | 1629.80 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-09 13:45:00 | 1619.40 | 1624.10 | 1628.82 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-10 11:15:00 | 1638.00 | 1627.77 | 1628.66 | SL hit (close>static) qty=1.00 sl=1632.80 alert=retest2 |

### Cycle 23 — BUY (started 2024-09-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-10 12:15:00 | 1640.70 | 1630.36 | 1629.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-11 12:15:00 | 1648.65 | 1638.06 | 1634.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-11 14:15:00 | 1627.60 | 1636.18 | 1634.15 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-11 14:15:00 | 1627.60 | 1636.18 | 1634.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-11 14:15:00 | 1627.60 | 1636.18 | 1634.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-11 15:00:00 | 1627.60 | 1636.18 | 1634.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-11 15:15:00 | 1635.00 | 1635.94 | 1634.23 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-12 09:15:00 | 1644.10 | 1635.94 | 1634.23 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-18 12:15:00 | 1639.60 | 1659.57 | 1660.08 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 24 — SELL (started 2024-09-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-18 12:15:00 | 1639.60 | 1659.57 | 1660.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-20 09:15:00 | 1623.20 | 1642.34 | 1649.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-20 12:15:00 | 1638.85 | 1638.83 | 1645.77 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-20 12:45:00 | 1638.25 | 1638.83 | 1645.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 13:15:00 | 1647.35 | 1640.53 | 1645.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-20 13:30:00 | 1643.20 | 1640.53 | 1645.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 14:15:00 | 1637.90 | 1640.01 | 1645.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-20 14:30:00 | 1643.60 | 1640.01 | 1645.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 15:15:00 | 1644.05 | 1640.82 | 1645.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-23 09:15:00 | 1657.00 | 1640.82 | 1645.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-23 09:15:00 | 1655.45 | 1643.74 | 1646.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-23 09:45:00 | 1659.55 | 1643.74 | 1646.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-23 10:15:00 | 1650.70 | 1645.13 | 1646.45 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-23 11:15:00 | 1642.15 | 1645.13 | 1646.45 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-23 12:15:00 | 1655.75 | 1647.55 | 1647.34 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 25 — BUY (started 2024-09-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-23 12:15:00 | 1655.75 | 1647.55 | 1647.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-23 14:15:00 | 1658.50 | 1651.03 | 1649.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-24 09:15:00 | 1649.35 | 1651.90 | 1649.85 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-24 09:15:00 | 1649.35 | 1651.90 | 1649.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-24 09:15:00 | 1649.35 | 1651.90 | 1649.85 | EMA400 retest candle locked (from upside) |

### Cycle 26 — SELL (started 2024-09-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-24 13:15:00 | 1641.80 | 1648.99 | 1649.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-24 14:15:00 | 1637.95 | 1646.78 | 1648.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-25 14:15:00 | 1644.00 | 1634.33 | 1639.46 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-25 14:15:00 | 1644.00 | 1634.33 | 1639.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 14:15:00 | 1644.00 | 1634.33 | 1639.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-25 15:00:00 | 1644.00 | 1634.33 | 1639.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 15:15:00 | 1648.00 | 1637.07 | 1640.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-26 09:15:00 | 1640.85 | 1637.07 | 1640.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-26 10:15:00 | 1639.20 | 1638.31 | 1640.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-26 10:30:00 | 1642.30 | 1638.31 | 1640.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-26 11:15:00 | 1618.40 | 1634.33 | 1638.31 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-26 12:45:00 | 1615.60 | 1628.62 | 1635.35 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-27 10:15:00 | 1641.40 | 1628.78 | 1632.17 | SL hit (close>static) qty=1.00 sl=1639.80 alert=retest2 |

### Cycle 27 — BUY (started 2024-09-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-27 11:15:00 | 1658.50 | 1634.72 | 1634.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-27 12:15:00 | 1662.90 | 1640.36 | 1637.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-30 10:15:00 | 1647.65 | 1653.89 | 1646.45 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-30 10:15:00 | 1647.65 | 1653.89 | 1646.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-30 10:15:00 | 1647.65 | 1653.89 | 1646.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-30 11:00:00 | 1647.65 | 1653.89 | 1646.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-01 11:15:00 | 1662.70 | 1657.19 | 1652.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-01 11:30:00 | 1654.80 | 1657.19 | 1652.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-03 09:15:00 | 1661.70 | 1659.83 | 1655.38 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-03 10:15:00 | 1668.60 | 1659.83 | 1655.38 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-04 09:15:00 | 1632.45 | 1654.63 | 1655.57 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 28 — SELL (started 2024-10-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-04 09:15:00 | 1632.45 | 1654.63 | 1655.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-04 13:15:00 | 1617.95 | 1638.53 | 1646.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-07 09:15:00 | 1640.25 | 1634.73 | 1642.70 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-07 09:15:00 | 1640.25 | 1634.73 | 1642.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-07 09:15:00 | 1640.25 | 1634.73 | 1642.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-07 10:00:00 | 1640.25 | 1634.73 | 1642.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-07 10:15:00 | 1633.55 | 1634.49 | 1641.87 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-07 13:45:00 | 1624.05 | 1631.09 | 1638.43 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-08 10:15:00 | 1648.85 | 1634.56 | 1637.52 | SL hit (close>static) qty=1.00 sl=1643.50 alert=retest2 |

### Cycle 29 — BUY (started 2024-10-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-08 12:15:00 | 1649.80 | 1640.05 | 1639.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-09 09:15:00 | 1667.60 | 1646.27 | 1642.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-10 09:15:00 | 1663.60 | 1669.63 | 1658.93 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-10 09:15:00 | 1663.60 | 1669.63 | 1658.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-10 09:15:00 | 1663.60 | 1669.63 | 1658.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-10 09:45:00 | 1659.00 | 1669.63 | 1658.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-10 10:15:00 | 1657.10 | 1667.13 | 1658.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-10 10:30:00 | 1658.40 | 1667.13 | 1658.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-10 11:15:00 | 1647.20 | 1663.14 | 1657.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-10 11:30:00 | 1646.55 | 1663.14 | 1657.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-10 12:15:00 | 1641.35 | 1658.78 | 1656.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-10 12:30:00 | 1642.85 | 1658.78 | 1656.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 30 — SELL (started 2024-10-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-10 13:15:00 | 1620.15 | 1651.06 | 1652.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-11 09:15:00 | 1599.00 | 1632.25 | 1643.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-14 14:15:00 | 1600.30 | 1593.59 | 1607.69 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-14 14:30:00 | 1596.95 | 1593.59 | 1607.69 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-17 09:15:00 | 1569.45 | 1565.87 | 1576.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-17 09:45:00 | 1577.00 | 1565.87 | 1576.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 14:15:00 | 1551.40 | 1554.27 | 1561.45 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-21 09:30:00 | 1544.95 | 1551.07 | 1558.73 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-28 15:15:00 | 1507.10 | 1498.46 | 1497.44 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 31 — BUY (started 2024-10-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-28 15:15:00 | 1507.10 | 1498.46 | 1497.44 | EMA200 above EMA400 |

### Cycle 32 — SELL (started 2024-10-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-29 09:15:00 | 1457.85 | 1490.34 | 1493.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-30 09:15:00 | 1418.75 | 1466.11 | 1478.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-31 09:15:00 | 1525.00 | 1450.68 | 1459.75 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-31 09:15:00 | 1525.00 | 1450.68 | 1459.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-31 09:15:00 | 1525.00 | 1450.68 | 1459.75 | EMA400 retest candle locked (from downside) |

### Cycle 33 — BUY (started 2024-10-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-31 10:15:00 | 1554.60 | 1471.47 | 1468.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-31 11:15:00 | 1559.80 | 1489.13 | 1476.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-06 10:15:00 | 1595.40 | 1596.47 | 1577.40 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-06 10:30:00 | 1592.70 | 1596.47 | 1577.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 09:15:00 | 1568.10 | 1588.58 | 1581.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-07 10:00:00 | 1568.10 | 1588.58 | 1581.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 10:15:00 | 1571.30 | 1585.12 | 1580.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-07 10:30:00 | 1568.40 | 1585.12 | 1580.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-08 09:15:00 | 1580.20 | 1579.50 | 1578.93 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-08 13:00:00 | 1588.00 | 1580.59 | 1579.52 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-11 10:15:00 | 1565.05 | 1579.39 | 1579.97 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 34 — SELL (started 2024-11-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-11 10:15:00 | 1565.05 | 1579.39 | 1579.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-11 11:15:00 | 1553.00 | 1574.11 | 1577.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-19 11:15:00 | 1481.25 | 1478.21 | 1491.55 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-19 11:45:00 | 1480.95 | 1478.21 | 1491.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-22 09:15:00 | 1487.20 | 1469.90 | 1476.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-22 10:00:00 | 1487.20 | 1469.90 | 1476.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-22 10:15:00 | 1478.60 | 1471.64 | 1476.50 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-22 11:15:00 | 1477.65 | 1471.64 | 1476.50 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-22 12:15:00 | 1488.85 | 1477.32 | 1478.37 | SL hit (close>static) qty=1.00 sl=1487.70 alert=retest2 |

### Cycle 35 — BUY (started 2024-11-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-22 13:15:00 | 1494.65 | 1480.79 | 1479.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-25 10:15:00 | 1497.30 | 1487.53 | 1483.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-25 12:15:00 | 1489.00 | 1489.69 | 1485.35 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-25 13:00:00 | 1489.00 | 1489.69 | 1485.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-26 13:15:00 | 1495.25 | 1499.05 | 1493.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-26 14:00:00 | 1495.25 | 1499.05 | 1493.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-26 14:15:00 | 1491.95 | 1497.63 | 1493.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-26 14:30:00 | 1498.20 | 1497.63 | 1493.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-26 15:15:00 | 1497.00 | 1497.50 | 1494.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-27 09:15:00 | 1476.10 | 1497.50 | 1494.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-27 09:15:00 | 1472.90 | 1492.58 | 1492.13 | EMA400 retest candle locked (from upside) |

### Cycle 36 — SELL (started 2024-11-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-27 10:15:00 | 1477.25 | 1489.52 | 1490.78 | EMA200 below EMA400 |

### Cycle 37 — BUY (started 2024-11-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-28 14:15:00 | 1494.40 | 1489.05 | 1488.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-29 09:15:00 | 1519.20 | 1496.53 | 1492.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-02 09:15:00 | 1520.60 | 1524.41 | 1512.30 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-02 12:15:00 | 1510.30 | 1520.93 | 1513.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-02 12:15:00 | 1510.30 | 1520.93 | 1513.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-02 13:00:00 | 1510.30 | 1520.93 | 1513.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-02 13:15:00 | 1508.50 | 1518.44 | 1513.21 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-02 14:30:00 | 1513.00 | 1516.03 | 1512.60 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-02 15:15:00 | 1513.00 | 1516.03 | 1512.60 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-04 12:15:00 | 1511.50 | 1521.87 | 1521.13 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-04 12:15:00 | 1507.00 | 1518.89 | 1519.85 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 38 — SELL (started 2024-12-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-04 12:15:00 | 1507.00 | 1518.89 | 1519.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-04 13:15:00 | 1503.40 | 1515.79 | 1518.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-05 13:15:00 | 1496.45 | 1495.77 | 1504.84 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-05 14:00:00 | 1496.45 | 1495.77 | 1504.84 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-05 14:15:00 | 1501.50 | 1496.92 | 1504.54 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-06 09:45:00 | 1484.45 | 1494.52 | 1502.14 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-17 09:15:00 | 1471.30 | 1451.92 | 1449.79 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 39 — BUY (started 2024-12-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-17 09:15:00 | 1471.30 | 1451.92 | 1449.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-19 10:15:00 | 1486.50 | 1473.35 | 1465.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-20 11:15:00 | 1494.25 | 1495.12 | 1484.14 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-20 11:30:00 | 1493.45 | 1495.12 | 1484.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 12:15:00 | 1482.95 | 1492.69 | 1484.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-20 13:00:00 | 1482.95 | 1492.69 | 1484.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 13:15:00 | 1476.90 | 1489.53 | 1483.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-20 14:00:00 | 1476.90 | 1489.53 | 1483.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 15:15:00 | 1475.40 | 1483.59 | 1481.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-23 09:15:00 | 1465.85 | 1483.59 | 1481.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-23 10:15:00 | 1482.90 | 1481.28 | 1480.72 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-23 11:15:00 | 1487.45 | 1481.28 | 1480.72 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-23 13:15:00 | 1473.00 | 1479.03 | 1479.82 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 40 — SELL (started 2024-12-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-23 13:15:00 | 1473.00 | 1479.03 | 1479.82 | EMA200 below EMA400 |

### Cycle 41 — BUY (started 2024-12-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-26 12:15:00 | 1481.30 | 1478.58 | 1478.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-26 14:15:00 | 1491.85 | 1481.70 | 1479.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-01 14:15:00 | 1531.25 | 1536.53 | 1527.47 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-01 15:00:00 | 1531.25 | 1536.53 | 1527.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-01 15:15:00 | 1525.00 | 1534.22 | 1527.25 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-02 10:30:00 | 1536.00 | 1534.21 | 1528.42 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-02 12:15:00 | 1534.50 | 1533.37 | 1528.56 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-02 14:00:00 | 1535.10 | 1533.65 | 1529.52 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-03 09:15:00 | 1519.35 | 1531.63 | 1529.72 | SL hit (close<static) qty=1.00 sl=1520.00 alert=retest2 |

### Cycle 42 — SELL (started 2025-01-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-03 11:15:00 | 1508.05 | 1525.03 | 1526.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-06 09:15:00 | 1492.80 | 1512.16 | 1519.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-07 09:15:00 | 1502.55 | 1495.81 | 1505.43 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-07 10:00:00 | 1502.55 | 1495.81 | 1505.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 10:15:00 | 1506.90 | 1498.03 | 1505.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-07 10:45:00 | 1505.30 | 1498.03 | 1505.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 11:15:00 | 1497.55 | 1497.93 | 1504.84 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-07 13:15:00 | 1497.00 | 1497.99 | 1504.23 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-07 15:15:00 | 1496.25 | 1497.70 | 1503.01 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-08 10:15:00 | 1495.90 | 1498.13 | 1502.26 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-08 12:00:00 | 1497.00 | 1498.14 | 1501.57 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-09 09:15:00 | 1486.50 | 1494.24 | 1498.23 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-10 09:45:00 | 1476.55 | 1488.60 | 1493.38 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-10 11:30:00 | 1480.30 | 1483.87 | 1490.29 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-20 15:15:00 | 1447.95 | 1445.18 | 1444.81 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 43 — BUY (started 2025-01-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-20 15:15:00 | 1447.95 | 1445.18 | 1444.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-21 09:15:00 | 1448.00 | 1445.74 | 1445.10 | Break + close above crossover candle high |

### Cycle 44 — SELL (started 2025-01-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-21 10:15:00 | 1439.05 | 1444.40 | 1444.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-21 14:15:00 | 1428.10 | 1439.40 | 1442.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-22 11:15:00 | 1436.75 | 1435.35 | 1438.86 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-22 11:15:00 | 1436.75 | 1435.35 | 1438.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-22 11:15:00 | 1436.75 | 1435.35 | 1438.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-22 11:45:00 | 1436.00 | 1435.35 | 1438.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-22 12:15:00 | 1438.15 | 1435.91 | 1438.79 | EMA400 retest candle locked (from downside) |

### Cycle 45 — BUY (started 2025-01-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-23 10:15:00 | 1456.30 | 1443.01 | 1441.26 | EMA200 above EMA400 |

### Cycle 46 — SELL (started 2025-01-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-24 12:15:00 | 1426.05 | 1440.07 | 1441.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-24 13:15:00 | 1419.00 | 1435.86 | 1439.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-28 13:15:00 | 1463.25 | 1407.58 | 1412.24 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-28 13:15:00 | 1463.25 | 1407.58 | 1412.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-28 13:15:00 | 1463.25 | 1407.58 | 1412.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-28 14:00:00 | 1463.25 | 1407.58 | 1412.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-28 14:15:00 | 1433.50 | 1412.77 | 1414.17 | EMA400 retest candle locked (from downside) |

### Cycle 47 — BUY (started 2025-01-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-28 15:15:00 | 1427.00 | 1415.61 | 1415.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-30 09:15:00 | 1447.50 | 1426.85 | 1421.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-01 09:15:00 | 1471.20 | 1471.29 | 1457.38 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-01 09:15:00 | 1471.20 | 1471.29 | 1457.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 09:15:00 | 1471.20 | 1471.29 | 1457.38 | EMA400 retest candle locked (from upside) |

### Cycle 48 — SELL (started 2025-02-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-01 15:15:00 | 1437.90 | 1450.44 | 1452.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-03 09:15:00 | 1414.50 | 1443.25 | 1448.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-03 15:15:00 | 1425.90 | 1425.49 | 1435.46 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-04 09:15:00 | 1441.35 | 1425.49 | 1435.46 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-04 09:15:00 | 1445.25 | 1429.44 | 1436.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-04 09:45:00 | 1450.30 | 1429.44 | 1436.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-04 10:15:00 | 1449.65 | 1433.48 | 1437.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-04 11:00:00 | 1449.65 | 1433.48 | 1437.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 49 — BUY (started 2025-02-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-04 12:15:00 | 1458.25 | 1442.72 | 1441.31 | EMA200 above EMA400 |

### Cycle 50 — SELL (started 2025-02-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-05 12:15:00 | 1432.85 | 1442.67 | 1442.72 | EMA200 below EMA400 |

### Cycle 51 — BUY (started 2025-02-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-06 09:15:00 | 1456.80 | 1443.39 | 1442.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-06 11:15:00 | 1468.95 | 1451.27 | 1446.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-07 09:15:00 | 1459.20 | 1462.72 | 1455.11 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-07 09:15:00 | 1459.20 | 1462.72 | 1455.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 09:15:00 | 1459.20 | 1462.72 | 1455.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-07 09:45:00 | 1456.00 | 1462.72 | 1455.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-10 09:15:00 | 1446.80 | 1464.28 | 1460.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-10 09:30:00 | 1433.90 | 1464.28 | 1460.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-10 10:15:00 | 1451.55 | 1461.74 | 1459.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-10 10:30:00 | 1449.90 | 1461.74 | 1459.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 52 — SELL (started 2025-02-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-10 13:15:00 | 1455.20 | 1458.37 | 1458.39 | EMA200 below EMA400 |

### Cycle 53 — BUY (started 2025-02-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-10 14:15:00 | 1458.60 | 1458.41 | 1458.41 | EMA200 above EMA400 |

### Cycle 54 — SELL (started 2025-02-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-10 15:15:00 | 1455.60 | 1457.85 | 1458.16 | EMA200 below EMA400 |

### Cycle 55 — BUY (started 2025-02-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-11 09:15:00 | 1463.50 | 1458.98 | 1458.64 | EMA200 above EMA400 |

### Cycle 56 — SELL (started 2025-02-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-11 12:15:00 | 1444.90 | 1457.34 | 1458.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-12 10:15:00 | 1441.55 | 1449.75 | 1453.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-12 11:15:00 | 1453.00 | 1450.40 | 1453.67 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-12 11:15:00 | 1453.00 | 1450.40 | 1453.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 11:15:00 | 1453.00 | 1450.40 | 1453.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-12 12:00:00 | 1453.00 | 1450.40 | 1453.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 12:15:00 | 1454.65 | 1451.25 | 1453.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-12 12:30:00 | 1452.70 | 1451.25 | 1453.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 13:15:00 | 1441.85 | 1449.37 | 1452.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-12 13:30:00 | 1450.70 | 1449.37 | 1452.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 57 — BUY (started 2025-02-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-13 09:15:00 | 1490.00 | 1457.90 | 1455.76 | EMA200 above EMA400 |

### Cycle 58 — SELL (started 2025-02-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-14 13:15:00 | 1456.55 | 1462.92 | 1463.03 | EMA200 below EMA400 |

### Cycle 59 — BUY (started 2025-02-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-17 10:15:00 | 1472.55 | 1464.56 | 1463.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-18 14:15:00 | 1482.15 | 1475.40 | 1471.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-19 09:15:00 | 1471.45 | 1475.66 | 1472.21 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-19 09:15:00 | 1471.45 | 1475.66 | 1472.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-19 09:15:00 | 1471.45 | 1475.66 | 1472.21 | EMA400 retest candle locked (from upside) |

### Cycle 60 — SELL (started 2025-02-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-19 13:15:00 | 1466.00 | 1470.52 | 1470.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-19 14:15:00 | 1465.00 | 1469.41 | 1470.08 | Break + close below crossover candle low |

### Cycle 61 — BUY (started 2025-02-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-20 09:15:00 | 1480.00 | 1470.98 | 1470.64 | EMA200 above EMA400 |

### Cycle 62 — SELL (started 2025-02-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-25 12:15:00 | 1473.05 | 1475.68 | 1475.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-25 13:15:00 | 1466.90 | 1473.92 | 1475.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-04 14:15:00 | 1402.25 | 1401.20 | 1412.33 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-04 15:00:00 | 1402.25 | 1401.20 | 1412.33 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-05 09:15:00 | 1412.30 | 1403.92 | 1411.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-05 10:00:00 | 1412.30 | 1403.92 | 1411.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-05 10:15:00 | 1411.60 | 1405.46 | 1411.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-05 11:15:00 | 1416.00 | 1405.46 | 1411.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-05 11:15:00 | 1420.25 | 1408.42 | 1412.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-05 12:00:00 | 1420.25 | 1408.42 | 1412.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-05 12:15:00 | 1425.90 | 1411.91 | 1413.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-05 13:00:00 | 1425.90 | 1411.91 | 1413.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 63 — BUY (started 2025-03-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-05 13:15:00 | 1428.15 | 1415.16 | 1414.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-06 09:15:00 | 1444.90 | 1422.58 | 1418.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-10 13:15:00 | 1468.45 | 1468.67 | 1457.79 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-10 13:45:00 | 1467.80 | 1468.67 | 1457.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 14:15:00 | 1462.40 | 1467.42 | 1458.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-10 14:45:00 | 1461.85 | 1467.42 | 1458.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 15:15:00 | 1460.60 | 1466.05 | 1458.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-11 09:15:00 | 1444.65 | 1466.05 | 1458.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-11 09:15:00 | 1455.55 | 1463.95 | 1458.17 | EMA400 retest candle locked (from upside) |

### Cycle 64 — SELL (started 2025-03-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-11 13:15:00 | 1446.55 | 1453.87 | 1454.75 | EMA200 below EMA400 |

### Cycle 65 — BUY (started 2025-03-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-13 09:15:00 | 1465.00 | 1454.01 | 1453.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-17 09:15:00 | 1490.95 | 1465.99 | 1460.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-19 09:15:00 | 1502.80 | 1502.86 | 1491.45 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-19 09:30:00 | 1501.85 | 1502.86 | 1491.45 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-20 09:15:00 | 1511.00 | 1503.25 | 1497.02 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-20 10:15:00 | 1514.35 | 1503.25 | 1497.02 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-21 09:15:00 | 1516.15 | 1507.97 | 1502.62 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-25 09:30:00 | 1519.65 | 1522.42 | 1518.20 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-25 14:45:00 | 1513.75 | 1518.39 | 1517.70 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-25 15:15:00 | 1506.00 | 1515.91 | 1516.63 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 66 — SELL (started 2025-03-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-25 15:15:00 | 1506.00 | 1515.91 | 1516.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-26 09:15:00 | 1504.90 | 1513.71 | 1515.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-27 14:15:00 | 1484.00 | 1479.19 | 1489.89 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-27 15:00:00 | 1484.00 | 1479.19 | 1489.89 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 09:15:00 | 1438.20 | 1447.36 | 1457.90 | EMA400 retest candle locked (from downside) |

### Cycle 67 — BUY (started 2025-04-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-03 10:15:00 | 1494.85 | 1464.70 | 1460.70 | EMA200 above EMA400 |

### Cycle 68 — SELL (started 2025-04-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-04 10:15:00 | 1410.00 | 1458.48 | 1463.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-07 09:15:00 | 1381.20 | 1419.59 | 1439.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-07 15:15:00 | 1393.40 | 1392.95 | 1414.73 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-08 09:15:00 | 1393.70 | 1392.95 | 1414.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 11:15:00 | 1418.55 | 1400.18 | 1412.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-08 12:00:00 | 1418.55 | 1400.18 | 1412.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 12:15:00 | 1426.25 | 1405.40 | 1413.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-08 12:30:00 | 1427.40 | 1405.40 | 1413.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-09 11:15:00 | 1410.00 | 1415.14 | 1416.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-09 11:30:00 | 1416.10 | 1415.14 | 1416.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-09 12:15:00 | 1415.30 | 1415.17 | 1416.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-09 12:30:00 | 1417.50 | 1415.17 | 1416.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-09 13:15:00 | 1414.70 | 1415.08 | 1416.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-09 14:00:00 | 1414.70 | 1415.08 | 1416.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-09 14:15:00 | 1417.55 | 1415.57 | 1416.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-09 14:45:00 | 1419.60 | 1415.57 | 1416.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-09 15:15:00 | 1416.00 | 1415.66 | 1416.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-11 09:15:00 | 1474.85 | 1415.66 | 1416.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 69 — BUY (started 2025-04-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-11 09:15:00 | 1468.00 | 1426.13 | 1421.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-16 09:15:00 | 1494.60 | 1475.40 | 1459.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-21 14:15:00 | 1510.10 | 1513.07 | 1502.51 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-21 15:00:00 | 1510.10 | 1513.07 | 1502.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-22 09:15:00 | 1513.00 | 1513.12 | 1504.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-22 09:30:00 | 1507.60 | 1513.12 | 1504.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-25 09:15:00 | 1539.10 | 1543.82 | 1534.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-25 10:00:00 | 1539.10 | 1543.82 | 1534.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-25 10:15:00 | 1524.50 | 1539.96 | 1533.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-25 10:45:00 | 1524.00 | 1539.96 | 1533.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-25 11:15:00 | 1522.90 | 1536.55 | 1532.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-25 11:45:00 | 1519.00 | 1536.55 | 1532.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-25 14:15:00 | 1525.40 | 1531.87 | 1531.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-25 14:30:00 | 1527.90 | 1531.87 | 1531.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 70 — SELL (started 2025-04-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-25 15:15:00 | 1525.10 | 1530.52 | 1530.91 | EMA200 below EMA400 |

### Cycle 71 — BUY (started 2025-04-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-28 09:15:00 | 1552.90 | 1534.99 | 1532.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-28 11:15:00 | 1553.60 | 1541.02 | 1536.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-29 09:15:00 | 1538.50 | 1547.89 | 1542.36 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-29 09:15:00 | 1538.50 | 1547.89 | 1542.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-29 09:15:00 | 1538.50 | 1547.89 | 1542.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-29 09:30:00 | 1536.90 | 1547.89 | 1542.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-29 10:15:00 | 1545.40 | 1547.39 | 1542.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-29 10:30:00 | 1542.90 | 1547.39 | 1542.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-29 11:15:00 | 1548.90 | 1547.69 | 1543.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-29 12:00:00 | 1548.90 | 1547.69 | 1543.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-29 13:15:00 | 1543.70 | 1547.01 | 1543.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-29 13:45:00 | 1544.10 | 1547.01 | 1543.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-29 14:15:00 | 1542.10 | 1546.03 | 1543.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-29 14:30:00 | 1543.20 | 1546.03 | 1543.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-29 15:15:00 | 1542.00 | 1545.22 | 1543.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-30 09:15:00 | 1556.20 | 1545.22 | 1543.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-30 12:15:00 | 1547.00 | 1551.62 | 1547.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-30 12:45:00 | 1545.80 | 1551.62 | 1547.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-30 13:15:00 | 1551.80 | 1551.66 | 1548.11 | EMA400 retest candle locked (from upside) |

### Cycle 72 — SELL (started 2025-05-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-02 10:15:00 | 1529.00 | 1545.87 | 1546.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-02 11:15:00 | 1526.10 | 1541.91 | 1544.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-05 09:15:00 | 1536.50 | 1533.46 | 1538.46 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-05 09:15:00 | 1536.50 | 1533.46 | 1538.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 09:15:00 | 1536.50 | 1533.46 | 1538.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-05 09:30:00 | 1535.30 | 1533.46 | 1538.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 10:15:00 | 1539.30 | 1534.63 | 1538.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-05 11:15:00 | 1542.40 | 1534.63 | 1538.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 11:15:00 | 1541.90 | 1536.08 | 1538.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-05 12:00:00 | 1541.90 | 1536.08 | 1538.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 12:15:00 | 1538.00 | 1536.47 | 1538.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-05 12:30:00 | 1538.00 | 1536.47 | 1538.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 13:15:00 | 1538.70 | 1536.91 | 1538.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-05 13:45:00 | 1538.20 | 1536.91 | 1538.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 14:15:00 | 1542.50 | 1538.03 | 1539.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-05 15:00:00 | 1542.50 | 1538.03 | 1539.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 15:15:00 | 1542.70 | 1538.96 | 1539.43 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-06 09:15:00 | 1511.30 | 1538.96 | 1539.43 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-12 13:15:00 | 1508.00 | 1496.50 | 1495.05 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 73 — BUY (started 2025-05-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 13:15:00 | 1508.00 | 1496.50 | 1495.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 14:15:00 | 1511.10 | 1499.42 | 1496.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-14 09:15:00 | 1503.60 | 1513.99 | 1508.57 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-14 09:15:00 | 1503.60 | 1513.99 | 1508.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-14 09:15:00 | 1503.60 | 1513.99 | 1508.57 | EMA400 retest candle locked (from upside) |

### Cycle 74 — SELL (started 2025-05-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-14 13:15:00 | 1490.50 | 1502.99 | 1504.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-15 09:15:00 | 1477.60 | 1496.26 | 1500.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-15 13:15:00 | 1506.00 | 1492.78 | 1497.06 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-15 13:15:00 | 1506.00 | 1492.78 | 1497.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-15 13:15:00 | 1506.00 | 1492.78 | 1497.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-15 14:00:00 | 1506.00 | 1492.78 | 1497.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-15 14:15:00 | 1499.70 | 1494.16 | 1497.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-15 14:30:00 | 1506.70 | 1494.16 | 1497.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-16 10:15:00 | 1494.90 | 1495.41 | 1497.21 | EMA400 retest candle locked (from downside) |

### Cycle 75 — BUY (started 2025-05-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-16 13:15:00 | 1500.40 | 1498.19 | 1498.15 | EMA200 above EMA400 |

### Cycle 76 — SELL (started 2025-05-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-19 11:15:00 | 1493.90 | 1498.21 | 1498.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-20 09:15:00 | 1471.40 | 1489.88 | 1494.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-21 09:15:00 | 1480.40 | 1471.38 | 1480.61 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-21 09:15:00 | 1480.40 | 1471.38 | 1480.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 09:15:00 | 1480.40 | 1471.38 | 1480.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-21 09:30:00 | 1478.20 | 1471.38 | 1480.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 10:15:00 | 1487.70 | 1474.65 | 1481.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-21 10:45:00 | 1487.60 | 1474.65 | 1481.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 11:15:00 | 1485.20 | 1476.76 | 1481.61 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-21 13:45:00 | 1478.20 | 1478.66 | 1481.75 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-23 11:15:00 | 1482.20 | 1476.04 | 1477.33 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-23 12:15:00 | 1487.50 | 1479.16 | 1478.57 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 77 — BUY (started 2025-05-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-23 12:15:00 | 1487.50 | 1479.16 | 1478.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-27 11:15:00 | 1488.80 | 1484.33 | 1482.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-27 13:15:00 | 1482.60 | 1484.36 | 1482.94 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-27 13:15:00 | 1482.60 | 1484.36 | 1482.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 13:15:00 | 1482.60 | 1484.36 | 1482.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-27 14:00:00 | 1482.60 | 1484.36 | 1482.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 14:15:00 | 1480.50 | 1483.59 | 1482.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-27 14:45:00 | 1481.30 | 1483.59 | 1482.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 15:15:00 | 1480.90 | 1483.05 | 1482.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-28 09:15:00 | 1477.20 | 1483.05 | 1482.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 78 — SELL (started 2025-05-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-28 09:15:00 | 1478.10 | 1482.06 | 1482.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-28 10:15:00 | 1475.20 | 1480.69 | 1481.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-29 11:15:00 | 1473.90 | 1472.65 | 1476.01 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-29 11:45:00 | 1474.20 | 1472.65 | 1476.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 12:15:00 | 1473.30 | 1472.78 | 1475.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-29 13:00:00 | 1473.30 | 1472.78 | 1475.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 14:15:00 | 1475.60 | 1473.05 | 1475.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-29 15:00:00 | 1475.60 | 1473.05 | 1475.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 15:15:00 | 1475.10 | 1473.46 | 1475.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-30 09:15:00 | 1486.10 | 1473.46 | 1475.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 09:15:00 | 1479.80 | 1474.72 | 1475.73 | EMA400 retest candle locked (from downside) |

### Cycle 79 — BUY (started 2025-05-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-30 11:15:00 | 1483.10 | 1476.46 | 1476.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-30 12:15:00 | 1485.40 | 1478.25 | 1477.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-30 14:15:00 | 1465.10 | 1476.81 | 1476.78 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-30 14:15:00 | 1465.10 | 1476.81 | 1476.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 14:15:00 | 1465.10 | 1476.81 | 1476.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-30 15:00:00 | 1465.10 | 1476.81 | 1476.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 80 — SELL (started 2025-05-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-30 15:15:00 | 1463.90 | 1474.23 | 1475.61 | EMA200 below EMA400 |

### Cycle 81 — BUY (started 2025-06-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-04 09:15:00 | 1480.90 | 1471.25 | 1470.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-05 09:15:00 | 1491.80 | 1476.62 | 1474.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-12 12:15:00 | 1513.10 | 1519.71 | 1514.74 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-12 12:15:00 | 1513.10 | 1519.71 | 1514.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 12:15:00 | 1513.10 | 1519.71 | 1514.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-12 13:00:00 | 1513.10 | 1519.71 | 1514.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 13:15:00 | 1497.70 | 1515.30 | 1513.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-12 14:00:00 | 1497.70 | 1515.30 | 1513.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 14:15:00 | 1501.70 | 1512.58 | 1512.15 | EMA400 retest candle locked (from upside) |

### Cycle 82 — SELL (started 2025-06-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-12 15:15:00 | 1502.80 | 1510.63 | 1511.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-13 10:15:00 | 1498.10 | 1506.98 | 1509.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-13 14:15:00 | 1504.70 | 1504.46 | 1507.23 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-13 15:00:00 | 1504.70 | 1504.46 | 1507.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 15:15:00 | 1507.80 | 1505.12 | 1507.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 09:15:00 | 1526.70 | 1505.12 | 1507.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 83 — BUY (started 2025-06-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-16 09:15:00 | 1535.50 | 1511.20 | 1509.85 | EMA200 above EMA400 |

### Cycle 84 — SELL (started 2025-06-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-17 12:15:00 | 1504.00 | 1514.42 | 1515.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-18 14:15:00 | 1497.00 | 1503.65 | 1508.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-20 09:15:00 | 1498.60 | 1491.54 | 1497.73 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-20 09:15:00 | 1498.60 | 1491.54 | 1497.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 09:15:00 | 1498.60 | 1491.54 | 1497.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 10:00:00 | 1498.60 | 1491.54 | 1497.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 10:15:00 | 1498.80 | 1493.00 | 1497.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 10:30:00 | 1501.30 | 1493.00 | 1497.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 11:15:00 | 1505.40 | 1495.48 | 1498.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 12:00:00 | 1505.40 | 1495.48 | 1498.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 12:15:00 | 1500.70 | 1496.52 | 1498.71 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-20 15:15:00 | 1493.00 | 1497.68 | 1498.92 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-23 09:30:00 | 1490.80 | 1494.67 | 1497.29 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-23 12:45:00 | 1495.90 | 1494.53 | 1496.51 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-24 09:15:00 | 1512.60 | 1500.24 | 1498.74 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 85 — BUY (started 2025-06-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-24 09:15:00 | 1512.60 | 1500.24 | 1498.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-26 09:15:00 | 1521.50 | 1510.50 | 1506.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-26 10:15:00 | 1510.30 | 1510.46 | 1506.81 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-26 10:45:00 | 1512.00 | 1510.46 | 1506.81 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 11:15:00 | 1505.90 | 1509.55 | 1506.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-26 11:45:00 | 1505.50 | 1509.55 | 1506.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 12:15:00 | 1506.20 | 1508.88 | 1506.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-26 12:45:00 | 1505.20 | 1508.88 | 1506.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 13:15:00 | 1510.00 | 1509.10 | 1506.98 | EMA400 retest candle locked (from upside) |

### Cycle 86 — SELL (started 2025-06-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-27 10:15:00 | 1496.90 | 1506.01 | 1506.31 | EMA200 below EMA400 |

### Cycle 87 — BUY (started 2025-06-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-30 13:15:00 | 1506.10 | 1504.73 | 1504.73 | EMA200 above EMA400 |

### Cycle 88 — SELL (started 2025-06-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-30 15:15:00 | 1503.10 | 1504.57 | 1504.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-01 09:15:00 | 1501.20 | 1503.90 | 1504.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-01 11:15:00 | 1504.00 | 1503.73 | 1504.18 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-01 11:15:00 | 1504.00 | 1503.73 | 1504.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 11:15:00 | 1504.00 | 1503.73 | 1504.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-01 11:30:00 | 1504.50 | 1503.73 | 1504.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 89 — BUY (started 2025-07-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-01 12:15:00 | 1507.90 | 1504.56 | 1504.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-01 13:15:00 | 1514.80 | 1506.61 | 1505.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-02 10:15:00 | 1507.00 | 1509.65 | 1507.62 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-02 10:15:00 | 1507.00 | 1509.65 | 1507.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 10:15:00 | 1507.00 | 1509.65 | 1507.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-02 10:45:00 | 1505.20 | 1509.65 | 1507.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 11:15:00 | 1506.30 | 1508.98 | 1507.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-02 11:30:00 | 1506.00 | 1508.98 | 1507.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 12:15:00 | 1507.70 | 1508.72 | 1507.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-02 13:00:00 | 1507.70 | 1508.72 | 1507.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 13:15:00 | 1499.10 | 1506.80 | 1506.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-02 13:30:00 | 1496.10 | 1506.80 | 1506.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 90 — SELL (started 2025-07-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-02 14:15:00 | 1497.20 | 1504.88 | 1505.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-02 15:15:00 | 1495.40 | 1502.98 | 1504.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-03 09:15:00 | 1505.30 | 1503.45 | 1504.96 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-03 09:15:00 | 1505.30 | 1503.45 | 1504.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 09:15:00 | 1505.30 | 1503.45 | 1504.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-03 10:15:00 | 1504.90 | 1503.45 | 1504.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 10:15:00 | 1503.50 | 1503.46 | 1504.83 | EMA400 retest candle locked (from downside) |

### Cycle 91 — BUY (started 2025-07-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-03 13:15:00 | 1510.40 | 1506.49 | 1506.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-04 09:15:00 | 1515.00 | 1508.77 | 1507.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-04 13:15:00 | 1510.40 | 1511.86 | 1509.53 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-04 13:15:00 | 1510.40 | 1511.86 | 1509.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 13:15:00 | 1510.40 | 1511.86 | 1509.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-04 13:45:00 | 1511.50 | 1511.86 | 1509.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 14:15:00 | 1513.60 | 1512.21 | 1509.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-04 14:30:00 | 1513.60 | 1512.21 | 1509.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 09:15:00 | 1510.60 | 1511.86 | 1510.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-07 09:45:00 | 1510.00 | 1511.86 | 1510.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 10:15:00 | 1515.00 | 1512.48 | 1510.58 | EMA400 retest candle locked (from upside) |

### Cycle 92 — SELL (started 2025-07-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-08 09:15:00 | 1493.20 | 1506.70 | 1508.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-08 10:15:00 | 1481.00 | 1501.56 | 1506.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-09 11:15:00 | 1495.00 | 1493.11 | 1497.89 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-09 12:00:00 | 1495.00 | 1493.11 | 1497.89 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 13:15:00 | 1483.90 | 1479.27 | 1483.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-11 14:00:00 | 1483.90 | 1479.27 | 1483.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 14:15:00 | 1485.20 | 1480.45 | 1483.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-11 14:45:00 | 1485.90 | 1480.45 | 1483.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 15:15:00 | 1486.00 | 1481.56 | 1483.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-14 09:15:00 | 1478.90 | 1481.56 | 1483.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 09:15:00 | 1483.90 | 1482.03 | 1483.70 | EMA400 retest candle locked (from downside) |

### Cycle 93 — BUY (started 2025-07-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-14 13:15:00 | 1488.20 | 1485.05 | 1484.72 | EMA200 above EMA400 |

### Cycle 94 — SELL (started 2025-07-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-14 15:15:00 | 1482.90 | 1484.48 | 1484.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-15 09:15:00 | 1472.20 | 1482.03 | 1483.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-15 10:15:00 | 1482.50 | 1482.12 | 1483.31 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-15 11:00:00 | 1482.50 | 1482.12 | 1483.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 11:15:00 | 1488.50 | 1483.40 | 1483.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-15 12:00:00 | 1488.50 | 1483.40 | 1483.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 95 — BUY (started 2025-07-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-15 12:15:00 | 1490.30 | 1484.78 | 1484.38 | EMA200 above EMA400 |

### Cycle 96 — SELL (started 2025-07-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-16 10:15:00 | 1478.30 | 1483.88 | 1484.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-16 15:15:00 | 1470.70 | 1478.68 | 1481.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-17 09:15:00 | 1479.40 | 1478.83 | 1481.27 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-17 09:15:00 | 1479.40 | 1478.83 | 1481.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 09:15:00 | 1479.40 | 1478.83 | 1481.27 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-18 10:15:00 | 1471.50 | 1479.84 | 1481.06 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-21 13:45:00 | 1469.00 | 1475.88 | 1477.74 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-22 09:15:00 | 1468.70 | 1474.79 | 1476.89 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-24 09:15:00 | 1477.30 | 1471.01 | 1470.70 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 97 — BUY (started 2025-07-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-24 09:15:00 | 1477.30 | 1471.01 | 1470.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-24 11:15:00 | 1482.20 | 1473.45 | 1471.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-30 10:15:00 | 1564.80 | 1567.63 | 1552.53 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-30 10:45:00 | 1562.70 | 1567.63 | 1552.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 13:15:00 | 1553.10 | 1562.81 | 1553.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-30 14:00:00 | 1553.10 | 1562.81 | 1553.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 14:15:00 | 1560.00 | 1562.25 | 1554.50 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-31 14:00:00 | 1562.90 | 1553.75 | 1552.46 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-01 09:15:00 | 1517.30 | 1546.38 | 1549.44 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 98 — SELL (started 2025-08-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-01 09:15:00 | 1517.30 | 1546.38 | 1549.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-01 11:15:00 | 1506.70 | 1533.60 | 1542.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-04 11:15:00 | 1515.40 | 1514.04 | 1525.65 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-04 11:45:00 | 1516.70 | 1514.04 | 1525.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 09:15:00 | 1481.90 | 1486.09 | 1495.99 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-07 11:00:00 | 1470.40 | 1482.95 | 1493.67 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-11 10:15:00 | 1502.90 | 1490.16 | 1489.45 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 99 — BUY (started 2025-08-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-11 10:15:00 | 1502.90 | 1490.16 | 1489.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-11 13:15:00 | 1505.00 | 1496.04 | 1492.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-18 11:15:00 | 1559.30 | 1560.08 | 1549.30 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-18 11:45:00 | 1559.00 | 1560.08 | 1549.30 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 13:15:00 | 1554.20 | 1561.67 | 1557.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-19 14:00:00 | 1554.20 | 1561.67 | 1557.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 14:15:00 | 1547.80 | 1558.90 | 1556.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-19 15:00:00 | 1547.80 | 1558.90 | 1556.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 100 — SELL (started 2025-08-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-20 09:15:00 | 1537.20 | 1552.82 | 1553.99 | EMA200 below EMA400 |

### Cycle 101 — BUY (started 2025-08-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-21 11:15:00 | 1566.50 | 1554.99 | 1553.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-21 13:15:00 | 1573.40 | 1560.75 | 1556.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-26 09:15:00 | 1584.60 | 1596.39 | 1589.00 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-26 09:15:00 | 1584.60 | 1596.39 | 1589.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 09:15:00 | 1584.60 | 1596.39 | 1589.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-26 10:00:00 | 1584.60 | 1596.39 | 1589.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 10:15:00 | 1587.30 | 1594.57 | 1588.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-26 10:45:00 | 1586.50 | 1594.57 | 1588.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 11:15:00 | 1587.60 | 1593.18 | 1588.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-26 12:00:00 | 1587.60 | 1593.18 | 1588.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 102 — SELL (started 2025-08-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-26 15:15:00 | 1576.60 | 1585.68 | 1586.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-28 09:15:00 | 1571.10 | 1582.76 | 1584.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-28 10:15:00 | 1583.40 | 1582.89 | 1584.77 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-28 11:00:00 | 1583.40 | 1582.89 | 1584.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 14:15:00 | 1578.90 | 1581.70 | 1583.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-28 14:30:00 | 1582.20 | 1581.70 | 1583.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 09:15:00 | 1569.40 | 1578.97 | 1581.99 | EMA400 retest candle locked (from downside) |

### Cycle 103 — BUY (started 2025-08-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-29 13:15:00 | 1592.00 | 1584.10 | 1583.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-01 09:15:00 | 1597.40 | 1588.04 | 1585.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-01 13:15:00 | 1587.50 | 1589.24 | 1587.20 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-01 13:15:00 | 1587.50 | 1589.24 | 1587.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 13:15:00 | 1587.50 | 1589.24 | 1587.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-01 14:00:00 | 1587.50 | 1589.24 | 1587.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 14:15:00 | 1586.70 | 1588.73 | 1587.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-01 14:30:00 | 1588.00 | 1588.73 | 1587.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 15:15:00 | 1585.10 | 1588.00 | 1586.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-02 09:15:00 | 1578.30 | 1588.00 | 1586.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 09:15:00 | 1587.70 | 1587.94 | 1587.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-02 09:30:00 | 1582.00 | 1587.94 | 1587.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 10:15:00 | 1588.10 | 1587.97 | 1587.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-02 11:00:00 | 1588.10 | 1587.97 | 1587.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 11:15:00 | 1584.00 | 1587.18 | 1586.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-02 11:30:00 | 1584.80 | 1587.18 | 1586.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 104 — SELL (started 2025-09-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-02 12:15:00 | 1580.20 | 1585.78 | 1586.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-02 13:15:00 | 1567.70 | 1582.17 | 1584.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-03 09:15:00 | 1589.10 | 1579.44 | 1582.33 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-03 09:15:00 | 1589.10 | 1579.44 | 1582.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 09:15:00 | 1589.10 | 1579.44 | 1582.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-03 10:00:00 | 1589.10 | 1579.44 | 1582.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 10:15:00 | 1592.00 | 1581.95 | 1583.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-03 11:00:00 | 1592.00 | 1581.95 | 1583.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 105 — BUY (started 2025-09-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-03 12:15:00 | 1587.60 | 1584.41 | 1584.20 | EMA200 above EMA400 |

### Cycle 106 — SELL (started 2025-09-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-03 14:15:00 | 1578.40 | 1583.26 | 1583.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-04 13:15:00 | 1575.00 | 1578.41 | 1580.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-04 15:15:00 | 1578.90 | 1578.15 | 1580.28 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-05 09:15:00 | 1580.10 | 1578.15 | 1580.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 09:15:00 | 1577.60 | 1578.04 | 1580.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-05 09:45:00 | 1581.80 | 1578.04 | 1580.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 10:15:00 | 1568.40 | 1576.11 | 1578.98 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-05 11:15:00 | 1567.60 | 1576.11 | 1578.98 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-10 10:30:00 | 1565.30 | 1554.87 | 1556.40 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-10 13:15:00 | 1559.60 | 1557.22 | 1557.22 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 107 — BUY (started 2025-09-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-10 13:15:00 | 1559.60 | 1557.22 | 1557.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-10 14:15:00 | 1564.60 | 1558.69 | 1557.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-11 10:15:00 | 1558.00 | 1558.98 | 1558.26 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-11 10:15:00 | 1558.00 | 1558.98 | 1558.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 10:15:00 | 1558.00 | 1558.98 | 1558.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-11 11:00:00 | 1558.00 | 1558.98 | 1558.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 11:15:00 | 1559.70 | 1559.13 | 1558.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-11 11:30:00 | 1558.70 | 1559.13 | 1558.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 13:15:00 | 1561.10 | 1559.92 | 1558.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-11 13:45:00 | 1560.10 | 1559.92 | 1558.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 09:15:00 | 1565.10 | 1570.19 | 1566.51 | EMA400 retest candle locked (from upside) |

### Cycle 108 — SELL (started 2025-09-15 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-15 13:15:00 | 1554.70 | 1563.88 | 1564.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-15 14:15:00 | 1548.70 | 1560.84 | 1563.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-16 13:15:00 | 1557.40 | 1556.97 | 1559.57 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-16 14:00:00 | 1557.40 | 1556.97 | 1559.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 14:15:00 | 1559.40 | 1557.46 | 1559.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-16 15:00:00 | 1559.40 | 1557.46 | 1559.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 15:15:00 | 1555.40 | 1557.05 | 1559.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-17 09:15:00 | 1554.00 | 1557.05 | 1559.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 09:15:00 | 1553.40 | 1556.32 | 1558.65 | EMA400 retest candle locked (from downside) |

### Cycle 109 — BUY (started 2025-09-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-18 09:15:00 | 1569.50 | 1559.80 | 1559.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-18 13:15:00 | 1572.90 | 1565.96 | 1562.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-19 09:15:00 | 1560.00 | 1568.59 | 1564.95 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-19 09:15:00 | 1560.00 | 1568.59 | 1564.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 09:15:00 | 1560.00 | 1568.59 | 1564.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-19 10:00:00 | 1560.00 | 1568.59 | 1564.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 10:15:00 | 1564.90 | 1567.85 | 1564.94 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-19 13:15:00 | 1567.30 | 1565.76 | 1564.44 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-19 13:45:00 | 1566.60 | 1566.11 | 1564.72 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-22 11:15:00 | 1555.10 | 1566.64 | 1566.07 | SL hit (close<static) qty=1.00 sl=1558.00 alert=retest2 |

### Cycle 110 — SELL (started 2025-09-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-22 12:15:00 | 1550.60 | 1563.43 | 1564.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-22 14:15:00 | 1541.20 | 1557.30 | 1561.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-24 11:15:00 | 1532.10 | 1531.67 | 1540.42 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-24 11:30:00 | 1533.60 | 1531.67 | 1540.42 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 13:15:00 | 1540.10 | 1534.37 | 1540.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-24 14:00:00 | 1540.10 | 1534.37 | 1540.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 14:15:00 | 1538.10 | 1535.12 | 1540.00 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-24 15:15:00 | 1535.00 | 1535.12 | 1540.00 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-25 10:30:00 | 1530.20 | 1534.51 | 1538.53 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-01 13:15:00 | 1515.40 | 1504.69 | 1504.46 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 111 — BUY (started 2025-10-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-01 13:15:00 | 1515.40 | 1504.69 | 1504.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-03 09:15:00 | 1518.30 | 1509.67 | 1507.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-06 09:15:00 | 1503.00 | 1512.51 | 1510.38 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-06 09:15:00 | 1503.00 | 1512.51 | 1510.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 09:15:00 | 1503.00 | 1512.51 | 1510.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-06 09:30:00 | 1502.60 | 1512.51 | 1510.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 10:15:00 | 1509.40 | 1511.88 | 1510.29 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-06 11:45:00 | 1510.10 | 1511.51 | 1510.27 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-07 13:00:00 | 1518.80 | 1514.48 | 1512.72 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-08 09:15:00 | 1500.70 | 1511.41 | 1511.89 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 112 — SELL (started 2025-10-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-08 09:15:00 | 1500.70 | 1511.41 | 1511.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-08 14:15:00 | 1495.00 | 1502.60 | 1506.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-09 09:15:00 | 1508.30 | 1502.44 | 1505.95 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-09 09:15:00 | 1508.30 | 1502.44 | 1505.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 09:15:00 | 1508.30 | 1502.44 | 1505.95 | EMA400 retest candle locked (from downside) |

### Cycle 113 — BUY (started 2025-10-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-09 13:15:00 | 1512.00 | 1508.49 | 1508.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-10 09:15:00 | 1520.20 | 1512.26 | 1510.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-14 09:15:00 | 1548.20 | 1556.06 | 1545.33 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-14 10:00:00 | 1548.20 | 1556.06 | 1545.33 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 10:15:00 | 1551.90 | 1555.23 | 1545.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-14 11:00:00 | 1551.90 | 1555.23 | 1545.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 11:15:00 | 1545.80 | 1553.34 | 1545.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-14 11:30:00 | 1539.00 | 1553.34 | 1545.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 12:15:00 | 1541.70 | 1551.01 | 1545.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-14 13:00:00 | 1541.70 | 1551.01 | 1545.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 13:15:00 | 1542.10 | 1549.23 | 1545.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-14 13:45:00 | 1541.20 | 1549.23 | 1545.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 14:15:00 | 1553.90 | 1550.16 | 1546.01 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-15 09:15:00 | 1560.60 | 1549.79 | 1546.22 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-15 10:45:00 | 1556.30 | 1550.93 | 1547.39 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-15 13:15:00 | 1554.90 | 1551.24 | 1548.14 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-24 12:15:00 | 1592.00 | 1615.98 | 1616.51 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 114 — SELL (started 2025-10-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-24 12:15:00 | 1592.00 | 1615.98 | 1616.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-24 13:15:00 | 1587.20 | 1610.22 | 1613.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-29 09:15:00 | 1575.60 | 1574.59 | 1583.23 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-29 10:15:00 | 1584.40 | 1576.55 | 1583.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 10:15:00 | 1584.40 | 1576.55 | 1583.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-29 11:00:00 | 1584.40 | 1576.55 | 1583.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 11:15:00 | 1581.60 | 1577.56 | 1583.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-29 11:30:00 | 1583.50 | 1577.56 | 1583.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 12:15:00 | 1583.80 | 1578.81 | 1583.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-29 12:30:00 | 1582.20 | 1578.81 | 1583.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 13:15:00 | 1577.00 | 1578.45 | 1582.67 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-30 09:15:00 | 1568.70 | 1579.27 | 1582.33 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-30 13:45:00 | 1543.50 | 1570.56 | 1576.58 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-10 12:15:00 | 1512.00 | 1507.02 | 1506.84 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 115 — BUY (started 2025-11-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-10 12:15:00 | 1512.00 | 1507.02 | 1506.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-12 09:15:00 | 1519.10 | 1514.03 | 1511.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-12 13:15:00 | 1518.00 | 1518.10 | 1514.62 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-12 14:00:00 | 1518.00 | 1518.10 | 1514.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 09:15:00 | 1520.60 | 1519.31 | 1516.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-13 09:30:00 | 1516.10 | 1519.31 | 1516.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 09:15:00 | 1522.80 | 1531.46 | 1528.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-18 10:00:00 | 1522.80 | 1531.46 | 1528.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 10:15:00 | 1526.60 | 1530.49 | 1528.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-18 10:30:00 | 1521.90 | 1530.49 | 1528.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 12:15:00 | 1520.10 | 1528.01 | 1527.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-18 13:00:00 | 1520.10 | 1528.01 | 1527.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 116 — SELL (started 2025-11-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-18 13:15:00 | 1526.00 | 1527.61 | 1527.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-18 14:15:00 | 1514.40 | 1524.97 | 1526.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-19 11:15:00 | 1524.00 | 1520.37 | 1523.38 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-19 11:15:00 | 1524.00 | 1520.37 | 1523.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 11:15:00 | 1524.00 | 1520.37 | 1523.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-19 12:00:00 | 1524.00 | 1520.37 | 1523.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 12:15:00 | 1526.50 | 1521.60 | 1523.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-19 13:00:00 | 1526.50 | 1521.60 | 1523.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 13:15:00 | 1526.90 | 1522.66 | 1523.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-19 14:00:00 | 1526.90 | 1522.66 | 1523.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 14:15:00 | 1528.10 | 1523.75 | 1524.34 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-19 15:15:00 | 1526.00 | 1523.75 | 1524.34 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-20 10:15:00 | 1531.20 | 1525.84 | 1525.20 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 117 — BUY (started 2025-11-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-20 10:15:00 | 1531.20 | 1525.84 | 1525.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-20 11:15:00 | 1532.30 | 1527.13 | 1525.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-21 09:15:00 | 1517.70 | 1526.35 | 1526.16 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-21 09:15:00 | 1517.70 | 1526.35 | 1526.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 09:15:00 | 1517.70 | 1526.35 | 1526.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-21 10:00:00 | 1517.70 | 1526.35 | 1526.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 118 — SELL (started 2025-11-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-21 10:15:00 | 1521.60 | 1525.40 | 1525.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-21 14:15:00 | 1512.10 | 1520.73 | 1523.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-24 09:15:00 | 1523.40 | 1519.47 | 1522.11 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-24 09:15:00 | 1523.40 | 1519.47 | 1522.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 09:15:00 | 1523.40 | 1519.47 | 1522.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-24 10:00:00 | 1523.40 | 1519.47 | 1522.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 10:15:00 | 1521.00 | 1519.77 | 1522.01 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-24 11:45:00 | 1518.40 | 1518.66 | 1521.30 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-25 12:45:00 | 1516.80 | 1515.48 | 1517.23 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-26 14:15:00 | 1525.00 | 1517.74 | 1516.89 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 119 — BUY (started 2025-11-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-26 14:15:00 | 1525.00 | 1517.74 | 1516.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-27 09:15:00 | 1527.10 | 1520.77 | 1518.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-28 09:15:00 | 1522.70 | 1524.09 | 1521.77 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-28 09:15:00 | 1522.70 | 1524.09 | 1521.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 09:15:00 | 1522.70 | 1524.09 | 1521.77 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-28 13:00:00 | 1530.00 | 1525.98 | 1523.26 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-01 12:15:00 | 1529.40 | 1527.35 | 1525.38 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-02 09:15:00 | 1520.80 | 1523.96 | 1524.29 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 120 — SELL (started 2025-12-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-02 09:15:00 | 1520.80 | 1523.96 | 1524.29 | EMA200 below EMA400 |

### Cycle 121 — BUY (started 2025-12-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-02 10:15:00 | 1527.70 | 1524.71 | 1524.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-02 11:15:00 | 1530.90 | 1525.95 | 1525.17 | Break + close above crossover candle high |

### Cycle 122 — SELL (started 2025-12-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-02 12:15:00 | 1517.00 | 1524.16 | 1524.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-03 09:15:00 | 1509.90 | 1519.27 | 1521.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-04 11:15:00 | 1510.00 | 1509.74 | 1513.63 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-04 12:00:00 | 1510.00 | 1509.74 | 1513.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 12:15:00 | 1514.50 | 1510.69 | 1513.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-04 13:00:00 | 1514.50 | 1510.69 | 1513.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 13:15:00 | 1518.30 | 1512.21 | 1514.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-04 14:00:00 | 1518.30 | 1512.21 | 1514.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 14:15:00 | 1519.80 | 1513.73 | 1514.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-04 15:00:00 | 1519.80 | 1513.73 | 1514.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 123 — BUY (started 2025-12-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-05 09:15:00 | 1517.60 | 1515.19 | 1515.19 | EMA200 above EMA400 |

### Cycle 124 — SELL (started 2025-12-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-08 10:15:00 | 1511.50 | 1516.46 | 1516.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-08 13:15:00 | 1501.80 | 1511.61 | 1514.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-10 09:15:00 | 1502.00 | 1497.92 | 1503.30 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-10 10:00:00 | 1502.00 | 1497.92 | 1503.30 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 10:15:00 | 1496.80 | 1497.69 | 1502.71 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-10 11:30:00 | 1494.90 | 1496.66 | 1501.78 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-10 13:30:00 | 1494.90 | 1495.00 | 1500.10 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-11 12:15:00 | 1510.40 | 1500.89 | 1500.72 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 125 — BUY (started 2025-12-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-11 12:15:00 | 1510.40 | 1500.89 | 1500.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-11 14:15:00 | 1512.60 | 1504.71 | 1502.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-15 09:15:00 | 1502.80 | 1513.04 | 1509.92 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-15 09:15:00 | 1502.80 | 1513.04 | 1509.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 09:15:00 | 1502.80 | 1513.04 | 1509.92 | EMA400 retest candle locked (from upside) |

### Cycle 126 — SELL (started 2025-12-15 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-15 15:15:00 | 1504.50 | 1508.27 | 1508.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-16 09:15:00 | 1501.30 | 1506.87 | 1507.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-16 10:15:00 | 1507.00 | 1506.90 | 1507.90 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-16 10:15:00 | 1507.00 | 1506.90 | 1507.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 10:15:00 | 1507.00 | 1506.90 | 1507.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-16 11:00:00 | 1507.00 | 1506.90 | 1507.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 11:15:00 | 1502.30 | 1505.98 | 1507.39 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-16 12:45:00 | 1500.50 | 1504.82 | 1506.74 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-16 14:00:00 | 1500.00 | 1503.86 | 1506.12 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-18 13:15:00 | 1500.40 | 1499.69 | 1500.59 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-19 10:15:00 | 1509.20 | 1501.98 | 1501.19 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 127 — BUY (started 2025-12-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-19 10:15:00 | 1509.20 | 1501.98 | 1501.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-19 14:15:00 | 1516.90 | 1507.26 | 1504.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-22 11:15:00 | 1511.20 | 1512.52 | 1507.99 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-22 12:00:00 | 1511.20 | 1512.52 | 1507.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-22 14:15:00 | 1512.70 | 1512.21 | 1508.94 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-23 09:15:00 | 1517.80 | 1511.81 | 1509.06 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-23 11:15:00 | 1508.00 | 1512.05 | 1509.99 | SL hit (close<static) qty=1.00 sl=1508.20 alert=retest2 |

### Cycle 128 — SELL (started 2025-12-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-23 14:15:00 | 1500.80 | 1507.37 | 1508.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-24 10:15:00 | 1496.60 | 1503.00 | 1505.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-26 09:15:00 | 1508.60 | 1499.82 | 1502.30 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-26 09:15:00 | 1508.60 | 1499.82 | 1502.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 09:15:00 | 1508.60 | 1499.82 | 1502.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-26 10:00:00 | 1508.60 | 1499.82 | 1502.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 10:15:00 | 1506.00 | 1501.06 | 1502.64 | EMA400 retest candle locked (from downside) |

### Cycle 129 — BUY (started 2025-12-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-26 14:15:00 | 1506.70 | 1503.89 | 1503.62 | EMA200 above EMA400 |

### Cycle 130 — SELL (started 2025-12-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-29 11:15:00 | 1500.50 | 1502.94 | 1503.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-29 14:15:00 | 1494.20 | 1499.66 | 1501.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-30 14:15:00 | 1493.00 | 1489.70 | 1494.46 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-30 14:15:00 | 1493.00 | 1489.70 | 1494.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 14:15:00 | 1493.00 | 1489.70 | 1494.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-30 15:00:00 | 1493.00 | 1489.70 | 1494.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 15:15:00 | 1488.90 | 1489.54 | 1493.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 09:15:00 | 1496.40 | 1489.54 | 1493.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 09:15:00 | 1493.60 | 1490.35 | 1493.92 | EMA400 retest candle locked (from downside) |

### Cycle 131 — BUY (started 2025-12-31 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-31 12:15:00 | 1505.30 | 1496.36 | 1496.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-31 13:15:00 | 1507.10 | 1498.50 | 1497.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-01 11:15:00 | 1502.00 | 1503.05 | 1500.37 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-01 11:45:00 | 1503.60 | 1503.05 | 1500.37 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 12:15:00 | 1501.50 | 1502.74 | 1500.48 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-02 09:30:00 | 1504.30 | 1503.15 | 1501.30 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-07 10:00:00 | 1503.10 | 1523.13 | 1519.77 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-07 10:15:00 | 1480.10 | 1514.52 | 1516.17 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 132 — SELL (started 2026-01-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-07 10:15:00 | 1480.10 | 1514.52 | 1516.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-07 11:15:00 | 1460.30 | 1503.68 | 1511.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-09 13:15:00 | 1464.50 | 1463.95 | 1473.71 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-09 13:45:00 | 1464.70 | 1463.95 | 1473.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 14:15:00 | 1466.50 | 1461.85 | 1466.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-12 14:45:00 | 1467.00 | 1461.85 | 1466.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 15:15:00 | 1464.00 | 1462.28 | 1466.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-13 09:15:00 | 1449.20 | 1462.28 | 1466.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 09:15:00 | 1443.50 | 1458.52 | 1464.57 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 12:00:00 | 1442.10 | 1453.20 | 1460.97 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 14:15:00 | 1442.30 | 1450.03 | 1458.11 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-14 09:30:00 | 1433.00 | 1445.17 | 1453.83 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-16 09:15:00 | 1369.99 | 1428.34 | 1440.40 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-16 09:15:00 | 1370.18 | 1428.34 | 1440.40 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-20 10:15:00 | 1390.00 | 1389.81 | 1401.97 | SL hit (close>ema200) qty=0.50 sl=1389.81 alert=retest2 |

### Cycle 133 — BUY (started 2026-02-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-01 15:15:00 | 1342.90 | 1326.09 | 1324.67 | EMA200 above EMA400 |

### Cycle 134 — SELL (started 2026-02-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-02 09:15:00 | 1306.70 | 1322.21 | 1323.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-02 10:15:00 | 1295.20 | 1316.81 | 1320.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-02 14:15:00 | 1313.00 | 1309.50 | 1315.19 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-02 15:00:00 | 1313.00 | 1309.50 | 1315.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 09:15:00 | 1332.60 | 1314.92 | 1316.72 | EMA400 retest candle locked (from downside) |

### Cycle 135 — BUY (started 2026-02-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 10:15:00 | 1334.60 | 1318.86 | 1318.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-09 14:15:00 | 1341.90 | 1333.61 | 1330.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-10 14:15:00 | 1341.60 | 1341.94 | 1337.44 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-10 14:30:00 | 1340.20 | 1341.94 | 1337.44 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 09:15:00 | 1337.00 | 1340.80 | 1337.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-11 10:00:00 | 1337.00 | 1340.80 | 1337.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 10:15:00 | 1346.30 | 1341.90 | 1338.48 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-11 11:15:00 | 1347.30 | 1341.90 | 1338.48 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-11 12:45:00 | 1348.40 | 1343.89 | 1340.03 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-12 10:15:00 | 1332.40 | 1342.23 | 1341.02 | SL hit (close<static) qty=1.00 sl=1336.70 alert=retest2 |

### Cycle 136 — SELL (started 2026-02-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-12 11:15:00 | 1332.00 | 1340.18 | 1340.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-12 12:15:00 | 1328.80 | 1337.90 | 1339.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-13 12:15:00 | 1330.10 | 1328.98 | 1333.06 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-13 12:15:00 | 1330.10 | 1328.98 | 1333.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 12:15:00 | 1330.10 | 1328.98 | 1333.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-13 12:30:00 | 1331.90 | 1328.98 | 1333.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 13:15:00 | 1334.20 | 1330.03 | 1333.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-13 14:00:00 | 1334.20 | 1330.03 | 1333.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 14:15:00 | 1330.50 | 1330.12 | 1332.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-13 14:45:00 | 1334.70 | 1330.12 | 1332.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 09:15:00 | 1337.70 | 1331.78 | 1333.20 | EMA400 retest candle locked (from downside) |

### Cycle 137 — BUY (started 2026-02-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-16 10:15:00 | 1344.60 | 1334.34 | 1334.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-16 11:15:00 | 1348.10 | 1337.09 | 1335.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-17 10:15:00 | 1345.70 | 1346.84 | 1342.16 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-17 11:00:00 | 1345.70 | 1346.84 | 1342.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 11:15:00 | 1346.10 | 1346.69 | 1342.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-17 11:30:00 | 1343.30 | 1346.69 | 1342.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 13:15:00 | 1343.20 | 1346.17 | 1343.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-17 14:00:00 | 1343.20 | 1346.17 | 1343.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 14:15:00 | 1344.10 | 1345.76 | 1343.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-17 14:30:00 | 1342.80 | 1345.76 | 1343.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 15:15:00 | 1343.20 | 1345.25 | 1343.12 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-18 09:15:00 | 1349.30 | 1345.25 | 1343.12 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-18 11:15:00 | 1340.60 | 1343.37 | 1342.71 | SL hit (close<static) qty=1.00 sl=1341.50 alert=retest2 |

### Cycle 138 — SELL (started 2026-02-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-19 13:15:00 | 1335.10 | 1343.28 | 1343.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-19 14:15:00 | 1331.00 | 1340.82 | 1342.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-20 11:15:00 | 1337.40 | 1335.37 | 1338.93 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-20 12:00:00 | 1337.40 | 1335.37 | 1338.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 12:15:00 | 1336.40 | 1335.58 | 1338.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-20 12:30:00 | 1337.40 | 1335.58 | 1338.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 14:15:00 | 1341.50 | 1336.70 | 1338.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-20 15:00:00 | 1341.50 | 1336.70 | 1338.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 15:15:00 | 1338.00 | 1336.96 | 1338.60 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-23 09:15:00 | 1320.00 | 1336.96 | 1338.60 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-25 10:15:00 | 1341.30 | 1331.71 | 1331.35 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 139 — BUY (started 2026-02-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-25 10:15:00 | 1341.30 | 1331.71 | 1331.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-25 14:15:00 | 1346.80 | 1338.49 | 1335.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-27 09:15:00 | 1343.80 | 1350.08 | 1344.85 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-27 09:15:00 | 1343.80 | 1350.08 | 1344.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 09:15:00 | 1343.80 | 1350.08 | 1344.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-27 09:30:00 | 1343.90 | 1350.08 | 1344.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 10:15:00 | 1344.30 | 1348.92 | 1344.80 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-27 14:00:00 | 1347.40 | 1347.53 | 1345.08 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-02 09:15:00 | 1330.90 | 1343.98 | 1344.07 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 140 — SELL (started 2026-03-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-02 09:15:00 | 1330.90 | 1343.98 | 1344.07 | EMA200 below EMA400 |

### Cycle 141 — BUY (started 2026-03-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-02 14:15:00 | 1353.20 | 1344.87 | 1344.18 | EMA200 above EMA400 |

### Cycle 142 — SELL (started 2026-03-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-04 09:15:00 | 1331.20 | 1343.11 | 1343.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-04 11:15:00 | 1323.70 | 1337.20 | 1340.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-05 12:15:00 | 1324.00 | 1323.08 | 1329.55 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-05 12:30:00 | 1323.50 | 1323.08 | 1329.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 14:15:00 | 1323.80 | 1322.44 | 1328.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-05 15:00:00 | 1323.80 | 1322.44 | 1328.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 09:15:00 | 1328.40 | 1324.44 | 1328.07 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-06 10:45:00 | 1318.40 | 1324.40 | 1327.72 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-09 09:15:00 | 1304.50 | 1324.76 | 1326.81 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-09 12:45:00 | 1321.00 | 1317.13 | 1321.73 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-09 14:30:00 | 1321.00 | 1319.32 | 1321.98 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-09 15:15:00 | 1325.00 | 1320.45 | 1322.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-10 09:15:00 | 1327.60 | 1320.45 | 1322.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 09:15:00 | 1322.50 | 1320.86 | 1322.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-10 09:30:00 | 1325.50 | 1320.86 | 1322.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 10:15:00 | 1326.20 | 1321.93 | 1322.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-10 11:00:00 | 1326.20 | 1321.93 | 1322.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2026-03-10 11:15:00 | 1331.90 | 1323.92 | 1323.47 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 143 — BUY (started 2026-03-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-10 11:15:00 | 1331.90 | 1323.92 | 1323.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-10 12:15:00 | 1337.70 | 1326.68 | 1324.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-11 09:15:00 | 1330.00 | 1330.07 | 1327.22 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-11 09:15:00 | 1330.00 | 1330.07 | 1327.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 09:15:00 | 1330.00 | 1330.07 | 1327.22 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-12 10:15:00 | 1335.60 | 1330.17 | 1328.84 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-12 10:45:00 | 1335.60 | 1330.94 | 1329.31 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-12 14:15:00 | 1325.10 | 1328.37 | 1328.51 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 144 — SELL (started 2026-03-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-12 14:15:00 | 1325.10 | 1328.37 | 1328.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-13 09:15:00 | 1310.30 | 1324.66 | 1326.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-13 15:15:00 | 1316.90 | 1316.29 | 1320.77 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-16 09:15:00 | 1319.30 | 1316.29 | 1320.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 09:15:00 | 1310.30 | 1315.10 | 1319.82 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-16 10:15:00 | 1298.60 | 1315.10 | 1319.82 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-23 13:15:00 | 1233.67 | 1243.71 | 1252.97 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-25 09:15:00 | 1235.80 | 1228.19 | 1236.16 | SL hit (close>ema200) qty=0.50 sl=1228.19 alert=retest2 |

### Cycle 145 — BUY (started 2026-03-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 14:15:00 | 1243.60 | 1240.06 | 1239.70 | EMA200 above EMA400 |

### Cycle 146 — SELL (started 2026-03-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 13:15:00 | 1235.30 | 1240.15 | 1240.17 | EMA200 below EMA400 |

### Cycle 147 — BUY (started 2026-03-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-27 14:15:00 | 1242.30 | 1240.58 | 1240.37 | EMA200 above EMA400 |

### Cycle 148 — SELL (started 2026-03-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-30 09:15:00 | 1226.70 | 1238.03 | 1239.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-01 10:15:00 | 1209.90 | 1226.11 | 1231.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-02 15:15:00 | 1195.00 | 1193.80 | 1204.66 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-06 09:15:00 | 1177.40 | 1193.80 | 1204.66 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-06 13:15:00 | 1197.50 | 1189.29 | 1197.35 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-04-06 13:15:00 | 1197.50 | 1189.29 | 1197.35 | SL hit (close>ema400) qty=1.00 sl=1197.35 alert=retest1 |

### Cycle 149 — BUY (started 2026-04-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-07 14:15:00 | 1202.00 | 1199.94 | 1199.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-08 09:15:00 | 1206.60 | 1201.60 | 1200.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-10 14:15:00 | 1230.10 | 1230.26 | 1223.57 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-10 14:30:00 | 1228.20 | 1230.26 | 1223.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 1211.70 | 1226.47 | 1223.01 | EMA400 retest candle locked (from upside) |

### Cycle 150 — SELL (started 2026-04-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-13 12:15:00 | 1211.90 | 1219.98 | 1220.60 | EMA200 below EMA400 |

### Cycle 151 — BUY (started 2026-04-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-15 11:15:00 | 1223.10 | 1219.89 | 1219.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-15 13:15:00 | 1225.60 | 1221.77 | 1220.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-20 10:15:00 | 1235.90 | 1237.50 | 1233.38 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-20 11:00:00 | 1235.90 | 1237.50 | 1233.38 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-20 11:15:00 | 1234.70 | 1236.94 | 1233.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-20 12:00:00 | 1234.70 | 1236.94 | 1233.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-20 12:15:00 | 1231.00 | 1235.75 | 1233.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-20 13:00:00 | 1231.00 | 1235.75 | 1233.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-20 13:15:00 | 1231.70 | 1234.94 | 1233.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-20 14:00:00 | 1231.70 | 1234.94 | 1233.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-20 15:15:00 | 1228.00 | 1232.78 | 1232.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-21 09:15:00 | 1229.10 | 1232.78 | 1232.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 10:15:00 | 1230.20 | 1232.57 | 1232.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-21 10:30:00 | 1230.90 | 1232.57 | 1232.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 152 — SELL (started 2026-04-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-21 11:15:00 | 1230.80 | 1232.22 | 1232.25 | EMA200 below EMA400 |

### Cycle 153 — BUY (started 2026-04-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-21 13:15:00 | 1233.70 | 1232.41 | 1232.33 | EMA200 above EMA400 |

### Cycle 154 — SELL (started 2026-04-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-22 09:15:00 | 1226.80 | 1231.37 | 1231.88 | EMA200 below EMA400 |

### Cycle 155 — BUY (started 2026-04-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-22 12:15:00 | 1235.10 | 1232.37 | 1232.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-23 09:15:00 | 1289.30 | 1244.99 | 1238.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-28 10:15:00 | 1312.20 | 1312.47 | 1299.58 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-28 11:00:00 | 1312.20 | 1312.47 | 1299.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 13:15:00 | 1300.30 | 1309.03 | 1301.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-28 13:30:00 | 1300.10 | 1309.03 | 1301.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 14:15:00 | 1307.40 | 1308.71 | 1301.72 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-28 15:15:00 | 1310.90 | 1308.71 | 1301.72 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-30 15:15:00 | 1312.00 | 1313.44 | 1311.15 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-05-27 11:45:00 | 1486.00 | 2024-05-30 09:15:00 | 1479.95 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest2 | 2024-05-28 09:30:00 | 1485.55 | 2024-05-30 09:15:00 | 1479.95 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest2 | 2024-05-28 10:30:00 | 1487.60 | 2024-05-30 11:15:00 | 1473.35 | STOP_HIT | 1.00 | -0.96% |
| BUY | retest2 | 2024-05-29 11:15:00 | 1485.35 | 2024-05-30 11:15:00 | 1473.35 | STOP_HIT | 1.00 | -0.81% |
| BUY | retest2 | 2024-05-29 13:15:00 | 1493.80 | 2024-05-30 11:15:00 | 1473.35 | STOP_HIT | 1.00 | -1.37% |
| BUY | retest2 | 2024-05-29 14:45:00 | 1492.40 | 2024-05-30 11:15:00 | 1473.35 | STOP_HIT | 1.00 | -1.28% |
| SELL | retest2 | 2024-06-04 12:00:00 | 1437.80 | 2024-06-05 09:15:00 | 1500.90 | STOP_HIT | 1.00 | -4.39% |
| BUY | retest2 | 2024-06-07 11:15:00 | 1502.35 | 2024-06-20 09:15:00 | 1543.05 | STOP_HIT | 1.00 | 2.71% |
| BUY | retest2 | 2024-06-10 09:15:00 | 1529.40 | 2024-06-20 09:15:00 | 1543.05 | STOP_HIT | 1.00 | 0.89% |
| SELL | retest2 | 2024-07-01 12:15:00 | 1475.35 | 2024-07-02 12:15:00 | 1485.95 | STOP_HIT | 1.00 | -0.72% |
| SELL | retest2 | 2024-07-01 13:00:00 | 1475.80 | 2024-07-02 12:15:00 | 1485.95 | STOP_HIT | 1.00 | -0.69% |
| BUY | retest2 | 2024-07-10 12:45:00 | 1504.90 | 2024-07-16 15:15:00 | 1508.40 | STOP_HIT | 1.00 | 0.23% |
| BUY | retest2 | 2024-07-10 14:30:00 | 1507.85 | 2024-07-16 15:15:00 | 1508.40 | STOP_HIT | 1.00 | 0.04% |
| BUY | retest2 | 2024-07-11 14:30:00 | 1506.25 | 2024-07-16 15:15:00 | 1508.40 | STOP_HIT | 1.00 | 0.14% |
| BUY | retest2 | 2024-07-12 13:00:00 | 1506.00 | 2024-07-16 15:15:00 | 1508.40 | STOP_HIT | 1.00 | 0.16% |
| SELL | retest2 | 2024-07-22 14:45:00 | 1493.60 | 2024-07-24 10:15:00 | 1499.15 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest2 | 2024-07-23 10:00:00 | 1489.45 | 2024-07-24 11:15:00 | 1499.20 | STOP_HIT | 1.00 | -0.65% |
| SELL | retest2 | 2024-07-23 10:45:00 | 1493.65 | 2024-07-24 11:15:00 | 1499.20 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest2 | 2024-07-23 12:00:00 | 1494.05 | 2024-07-24 11:15:00 | 1499.20 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest2 | 2024-07-24 09:15:00 | 1491.40 | 2024-07-24 11:15:00 | 1499.20 | STOP_HIT | 1.00 | -0.52% |
| BUY | retest2 | 2024-07-31 11:30:00 | 1541.35 | 2024-08-02 10:15:00 | 1528.60 | STOP_HIT | 1.00 | -0.83% |
| BUY | retest2 | 2024-08-01 11:30:00 | 1543.70 | 2024-08-02 10:15:00 | 1528.60 | STOP_HIT | 1.00 | -0.98% |
| BUY | retest2 | 2024-08-01 14:00:00 | 1540.95 | 2024-08-02 10:15:00 | 1528.60 | STOP_HIT | 1.00 | -0.80% |
| BUY | retest2 | 2024-09-04 10:45:00 | 1650.00 | 2024-09-05 13:15:00 | 1638.50 | STOP_HIT | 1.00 | -0.70% |
| BUY | retest2 | 2024-09-04 14:30:00 | 1650.50 | 2024-09-05 13:15:00 | 1638.50 | STOP_HIT | 1.00 | -0.73% |
| BUY | retest2 | 2024-09-05 09:30:00 | 1651.30 | 2024-09-05 13:15:00 | 1638.50 | STOP_HIT | 1.00 | -0.78% |
| BUY | retest2 | 2024-09-05 12:30:00 | 1649.40 | 2024-09-05 13:15:00 | 1638.50 | STOP_HIT | 1.00 | -0.66% |
| SELL | retest2 | 2024-09-09 13:45:00 | 1619.40 | 2024-09-10 11:15:00 | 1638.00 | STOP_HIT | 1.00 | -1.15% |
| BUY | retest2 | 2024-09-12 09:15:00 | 1644.10 | 2024-09-18 12:15:00 | 1639.60 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest2 | 2024-09-23 11:15:00 | 1642.15 | 2024-09-23 12:15:00 | 1655.75 | STOP_HIT | 1.00 | -0.83% |
| SELL | retest2 | 2024-09-26 12:45:00 | 1615.60 | 2024-09-27 10:15:00 | 1641.40 | STOP_HIT | 1.00 | -1.60% |
| BUY | retest2 | 2024-10-03 10:15:00 | 1668.60 | 2024-10-04 09:15:00 | 1632.45 | STOP_HIT | 1.00 | -2.17% |
| SELL | retest2 | 2024-10-07 13:45:00 | 1624.05 | 2024-10-08 10:15:00 | 1648.85 | STOP_HIT | 1.00 | -1.53% |
| SELL | retest2 | 2024-10-21 09:30:00 | 1544.95 | 2024-10-28 15:15:00 | 1507.10 | STOP_HIT | 1.00 | 2.45% |
| BUY | retest2 | 2024-11-08 13:00:00 | 1588.00 | 2024-11-11 10:15:00 | 1565.05 | STOP_HIT | 1.00 | -1.45% |
| SELL | retest2 | 2024-11-22 11:15:00 | 1477.65 | 2024-11-22 12:15:00 | 1488.85 | STOP_HIT | 1.00 | -0.76% |
| BUY | retest2 | 2024-12-02 14:30:00 | 1513.00 | 2024-12-04 12:15:00 | 1507.00 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest2 | 2024-12-02 15:15:00 | 1513.00 | 2024-12-04 12:15:00 | 1507.00 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest2 | 2024-12-04 12:15:00 | 1511.50 | 2024-12-04 12:15:00 | 1507.00 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest2 | 2024-12-06 09:45:00 | 1484.45 | 2024-12-17 09:15:00 | 1471.30 | STOP_HIT | 1.00 | 0.89% |
| BUY | retest2 | 2024-12-23 11:15:00 | 1487.45 | 2024-12-23 13:15:00 | 1473.00 | STOP_HIT | 1.00 | -0.97% |
| BUY | retest2 | 2025-01-02 10:30:00 | 1536.00 | 2025-01-03 09:15:00 | 1519.35 | STOP_HIT | 1.00 | -1.08% |
| BUY | retest2 | 2025-01-02 12:15:00 | 1534.50 | 2025-01-03 09:15:00 | 1519.35 | STOP_HIT | 1.00 | -0.99% |
| BUY | retest2 | 2025-01-02 14:00:00 | 1535.10 | 2025-01-03 09:15:00 | 1519.35 | STOP_HIT | 1.00 | -1.03% |
| SELL | retest2 | 2025-01-07 13:15:00 | 1497.00 | 2025-01-20 15:15:00 | 1447.95 | STOP_HIT | 1.00 | 3.28% |
| SELL | retest2 | 2025-01-07 15:15:00 | 1496.25 | 2025-01-20 15:15:00 | 1447.95 | STOP_HIT | 1.00 | 3.23% |
| SELL | retest2 | 2025-01-08 10:15:00 | 1495.90 | 2025-01-20 15:15:00 | 1447.95 | STOP_HIT | 1.00 | 3.21% |
| SELL | retest2 | 2025-01-08 12:00:00 | 1497.00 | 2025-01-20 15:15:00 | 1447.95 | STOP_HIT | 1.00 | 3.28% |
| SELL | retest2 | 2025-01-10 09:45:00 | 1476.55 | 2025-01-20 15:15:00 | 1447.95 | STOP_HIT | 1.00 | 1.94% |
| SELL | retest2 | 2025-01-10 11:30:00 | 1480.30 | 2025-01-20 15:15:00 | 1447.95 | STOP_HIT | 1.00 | 2.19% |
| BUY | retest2 | 2025-03-20 10:15:00 | 1514.35 | 2025-03-25 15:15:00 | 1506.00 | STOP_HIT | 1.00 | -0.55% |
| BUY | retest2 | 2025-03-21 09:15:00 | 1516.15 | 2025-03-25 15:15:00 | 1506.00 | STOP_HIT | 1.00 | -0.67% |
| BUY | retest2 | 2025-03-25 09:30:00 | 1519.65 | 2025-03-25 15:15:00 | 1506.00 | STOP_HIT | 1.00 | -0.90% |
| BUY | retest2 | 2025-03-25 14:45:00 | 1513.75 | 2025-03-25 15:15:00 | 1506.00 | STOP_HIT | 1.00 | -0.51% |
| SELL | retest2 | 2025-05-06 09:15:00 | 1511.30 | 2025-05-12 13:15:00 | 1508.00 | STOP_HIT | 1.00 | 0.22% |
| SELL | retest2 | 2025-05-21 13:45:00 | 1478.20 | 2025-05-23 12:15:00 | 1487.50 | STOP_HIT | 1.00 | -0.63% |
| SELL | retest2 | 2025-05-23 11:15:00 | 1482.20 | 2025-05-23 12:15:00 | 1487.50 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest2 | 2025-06-20 15:15:00 | 1493.00 | 2025-06-24 09:15:00 | 1512.60 | STOP_HIT | 1.00 | -1.31% |
| SELL | retest2 | 2025-06-23 09:30:00 | 1490.80 | 2025-06-24 09:15:00 | 1512.60 | STOP_HIT | 1.00 | -1.46% |
| SELL | retest2 | 2025-06-23 12:45:00 | 1495.90 | 2025-06-24 09:15:00 | 1512.60 | STOP_HIT | 1.00 | -1.12% |
| SELL | retest2 | 2025-07-18 10:15:00 | 1471.50 | 2025-07-24 09:15:00 | 1477.30 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest2 | 2025-07-21 13:45:00 | 1469.00 | 2025-07-24 09:15:00 | 1477.30 | STOP_HIT | 1.00 | -0.57% |
| SELL | retest2 | 2025-07-22 09:15:00 | 1468.70 | 2025-07-24 09:15:00 | 1477.30 | STOP_HIT | 1.00 | -0.59% |
| BUY | retest2 | 2025-07-31 14:00:00 | 1562.90 | 2025-08-01 09:15:00 | 1517.30 | STOP_HIT | 1.00 | -2.92% |
| SELL | retest2 | 2025-08-07 11:00:00 | 1470.40 | 2025-08-11 10:15:00 | 1502.90 | STOP_HIT | 1.00 | -2.21% |
| SELL | retest2 | 2025-09-05 11:15:00 | 1567.60 | 2025-09-10 13:15:00 | 1559.60 | STOP_HIT | 1.00 | 0.51% |
| SELL | retest2 | 2025-09-10 10:30:00 | 1565.30 | 2025-09-10 13:15:00 | 1559.60 | STOP_HIT | 1.00 | 0.36% |
| BUY | retest2 | 2025-09-19 13:15:00 | 1567.30 | 2025-09-22 11:15:00 | 1555.10 | STOP_HIT | 1.00 | -0.78% |
| BUY | retest2 | 2025-09-19 13:45:00 | 1566.60 | 2025-09-22 11:15:00 | 1555.10 | STOP_HIT | 1.00 | -0.73% |
| SELL | retest2 | 2025-09-24 15:15:00 | 1535.00 | 2025-10-01 13:15:00 | 1515.40 | STOP_HIT | 1.00 | 1.28% |
| SELL | retest2 | 2025-09-25 10:30:00 | 1530.20 | 2025-10-01 13:15:00 | 1515.40 | STOP_HIT | 1.00 | 0.97% |
| BUY | retest2 | 2025-10-06 11:45:00 | 1510.10 | 2025-10-08 09:15:00 | 1500.70 | STOP_HIT | 1.00 | -0.62% |
| BUY | retest2 | 2025-10-07 13:00:00 | 1518.80 | 2025-10-08 09:15:00 | 1500.70 | STOP_HIT | 1.00 | -1.19% |
| BUY | retest2 | 2025-10-15 09:15:00 | 1560.60 | 2025-10-24 12:15:00 | 1592.00 | STOP_HIT | 1.00 | 2.01% |
| BUY | retest2 | 2025-10-15 10:45:00 | 1556.30 | 2025-10-24 12:15:00 | 1592.00 | STOP_HIT | 1.00 | 2.29% |
| BUY | retest2 | 2025-10-15 13:15:00 | 1554.90 | 2025-10-24 12:15:00 | 1592.00 | STOP_HIT | 1.00 | 2.39% |
| SELL | retest2 | 2025-10-30 09:15:00 | 1568.70 | 2025-11-10 12:15:00 | 1512.00 | STOP_HIT | 1.00 | 3.61% |
| SELL | retest2 | 2025-10-30 13:45:00 | 1543.50 | 2025-11-10 12:15:00 | 1512.00 | STOP_HIT | 1.00 | 2.04% |
| SELL | retest2 | 2025-11-19 15:15:00 | 1526.00 | 2025-11-20 10:15:00 | 1531.20 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest2 | 2025-11-24 11:45:00 | 1518.40 | 2025-11-26 14:15:00 | 1525.00 | STOP_HIT | 1.00 | -0.43% |
| SELL | retest2 | 2025-11-25 12:45:00 | 1516.80 | 2025-11-26 14:15:00 | 1525.00 | STOP_HIT | 1.00 | -0.54% |
| BUY | retest2 | 2025-11-28 13:00:00 | 1530.00 | 2025-12-02 09:15:00 | 1520.80 | STOP_HIT | 1.00 | -0.60% |
| BUY | retest2 | 2025-12-01 12:15:00 | 1529.40 | 2025-12-02 09:15:00 | 1520.80 | STOP_HIT | 1.00 | -0.56% |
| SELL | retest2 | 2025-12-10 11:30:00 | 1494.90 | 2025-12-11 12:15:00 | 1510.40 | STOP_HIT | 1.00 | -1.04% |
| SELL | retest2 | 2025-12-10 13:30:00 | 1494.90 | 2025-12-11 12:15:00 | 1510.40 | STOP_HIT | 1.00 | -1.04% |
| SELL | retest2 | 2025-12-16 12:45:00 | 1500.50 | 2025-12-19 10:15:00 | 1509.20 | STOP_HIT | 1.00 | -0.58% |
| SELL | retest2 | 2025-12-16 14:00:00 | 1500.00 | 2025-12-19 10:15:00 | 1509.20 | STOP_HIT | 1.00 | -0.61% |
| SELL | retest2 | 2025-12-18 13:15:00 | 1500.40 | 2025-12-19 10:15:00 | 1509.20 | STOP_HIT | 1.00 | -0.59% |
| BUY | retest2 | 2025-12-23 09:15:00 | 1517.80 | 2025-12-23 11:15:00 | 1508.00 | STOP_HIT | 1.00 | -0.65% |
| BUY | retest2 | 2026-01-02 09:30:00 | 1504.30 | 2026-01-07 10:15:00 | 1480.10 | STOP_HIT | 1.00 | -1.61% |
| BUY | retest2 | 2026-01-07 10:00:00 | 1503.10 | 2026-01-07 10:15:00 | 1480.10 | STOP_HIT | 1.00 | -1.53% |
| SELL | retest2 | 2026-01-13 12:00:00 | 1442.10 | 2026-01-16 09:15:00 | 1369.99 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-13 14:15:00 | 1442.30 | 2026-01-16 09:15:00 | 1370.18 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-13 12:00:00 | 1442.10 | 2026-01-20 10:15:00 | 1390.00 | STOP_HIT | 0.50 | 3.61% |
| SELL | retest2 | 2026-01-13 14:15:00 | 1442.30 | 2026-01-20 10:15:00 | 1390.00 | STOP_HIT | 0.50 | 3.63% |
| SELL | retest2 | 2026-01-14 09:30:00 | 1433.00 | 2026-01-23 13:15:00 | 1361.35 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-14 09:30:00 | 1433.00 | 2026-01-27 09:15:00 | 1289.70 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2026-02-11 11:15:00 | 1347.30 | 2026-02-12 10:15:00 | 1332.40 | STOP_HIT | 1.00 | -1.11% |
| BUY | retest2 | 2026-02-11 12:45:00 | 1348.40 | 2026-02-12 10:15:00 | 1332.40 | STOP_HIT | 1.00 | -1.19% |
| BUY | retest2 | 2026-02-18 09:15:00 | 1349.30 | 2026-02-18 11:15:00 | 1340.60 | STOP_HIT | 1.00 | -0.64% |
| BUY | retest2 | 2026-02-18 14:30:00 | 1346.50 | 2026-02-19 12:15:00 | 1338.00 | STOP_HIT | 1.00 | -0.63% |
| SELL | retest2 | 2026-02-23 09:15:00 | 1320.00 | 2026-02-25 10:15:00 | 1341.30 | STOP_HIT | 1.00 | -1.61% |
| BUY | retest2 | 2026-02-27 14:00:00 | 1347.40 | 2026-03-02 09:15:00 | 1330.90 | STOP_HIT | 1.00 | -1.22% |
| SELL | retest2 | 2026-03-06 10:45:00 | 1318.40 | 2026-03-10 11:15:00 | 1331.90 | STOP_HIT | 1.00 | -1.02% |
| SELL | retest2 | 2026-03-09 09:15:00 | 1304.50 | 2026-03-10 11:15:00 | 1331.90 | STOP_HIT | 1.00 | -2.10% |
| SELL | retest2 | 2026-03-09 12:45:00 | 1321.00 | 2026-03-10 11:15:00 | 1331.90 | STOP_HIT | 1.00 | -0.83% |
| SELL | retest2 | 2026-03-09 14:30:00 | 1321.00 | 2026-03-10 11:15:00 | 1331.90 | STOP_HIT | 1.00 | -0.83% |
| BUY | retest2 | 2026-03-12 10:15:00 | 1335.60 | 2026-03-12 14:15:00 | 1325.10 | STOP_HIT | 1.00 | -0.79% |
| BUY | retest2 | 2026-03-12 10:45:00 | 1335.60 | 2026-03-12 14:15:00 | 1325.10 | STOP_HIT | 1.00 | -0.79% |
| SELL | retest2 | 2026-03-16 10:15:00 | 1298.60 | 2026-03-23 13:15:00 | 1233.67 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-16 10:15:00 | 1298.60 | 2026-03-25 09:15:00 | 1235.80 | STOP_HIT | 0.50 | 4.84% |
| SELL | retest1 | 2026-04-06 09:15:00 | 1177.40 | 2026-04-06 13:15:00 | 1197.50 | STOP_HIT | 1.00 | -1.71% |
| SELL | retest2 | 2026-04-07 09:15:00 | 1193.40 | 2026-04-07 14:15:00 | 1202.00 | STOP_HIT | 1.00 | -0.72% |
