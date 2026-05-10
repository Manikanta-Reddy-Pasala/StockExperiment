# Clean Science and Technology Ltd. (CLEAN)

## Backtest Summary

- **Window:** 2025-04-11 09:15:00 → 2026-05-08 15:15:00 (1850 bars)
- **Last close:** 891.90
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 84 |
| ALERT1 | 50 |
| ALERT2 | 50 |
| ALERT2_SKIP | 25 |
| ALERT3 | 129 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 1 |
| ENTRY2 | 60 |
| PARTIAL | 8 |
| TARGET_HIT | 2 |
| STOP_HIT | 59 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 69 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 32 / 37
- **Target hits / Stop hits / Partials:** 2 / 59 / 8
- **Avg / median % per leg:** 0.77% / -0.25%
- **Sum % (uncompounded):** 53.11%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 15 | 7 | 46.7% | 2 | 13 | 0 | 1.05% | 15.7% |
| BUY @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -0.17% | -0.2% |
| BUY @ 3rd Alert (retest2) | 14 | 7 | 50.0% | 2 | 12 | 0 | 1.13% | 15.9% |
| SELL (all) | 54 | 25 | 46.3% | 0 | 46 | 8 | 0.69% | 37.4% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 54 | 25 | 46.3% | 0 | 46 | 8 | 0.69% | 37.4% |
| retest1 (combined) | 1 | 0 | 0.0% | 0 | 1 | 0 | -0.17% | -0.2% |
| retest2 (combined) | 68 | 32 | 47.1% | 2 | 58 | 8 | 0.78% | 53.3% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-05-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-20 14:15:00 | 1268.20 | 1276.79 | 1276.92 | EMA200 below EMA400 |

### Cycle 2 — BUY (started 2025-05-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-21 10:15:00 | 1279.10 | 1277.19 | 1277.00 | EMA200 above EMA400 |

### Cycle 3 — SELL (started 2025-05-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-21 11:15:00 | 1272.70 | 1276.29 | 1276.61 | EMA200 below EMA400 |

### Cycle 4 — BUY (started 2025-05-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-22 09:15:00 | 1281.00 | 1276.72 | 1276.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-22 13:15:00 | 1290.00 | 1282.51 | 1279.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-28 14:15:00 | 1423.20 | 1429.12 | 1416.27 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-28 15:00:00 | 1423.20 | 1429.12 | 1416.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 09:15:00 | 1481.40 | 1439.62 | 1423.28 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-30 15:00:00 | 1548.10 | 1481.30 | 1459.99 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-03 13:15:00 | 1491.70 | 1503.50 | 1495.31 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-03 14:00:00 | 1491.20 | 1501.04 | 1494.94 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-04 09:15:00 | 1496.10 | 1493.11 | 1492.22 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 10:15:00 | 1493.50 | 1493.48 | 1492.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-04 11:00:00 | 1493.50 | 1493.48 | 1492.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 14:15:00 | 1489.30 | 1496.18 | 1494.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-04 15:00:00 | 1489.30 | 1496.18 | 1494.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 15:15:00 | 1492.00 | 1495.35 | 1494.25 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-05 09:30:00 | 1496.80 | 1500.52 | 1496.70 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-06 14:15:00 | 1492.00 | 1508.91 | 1509.89 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-06 14:15:00 | 1492.00 | 1508.91 | 1509.89 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-06 14:15:00 | 1492.00 | 1508.91 | 1509.89 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-06 14:15:00 | 1492.00 | 1508.91 | 1509.89 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-06 14:15:00 | 1492.00 | 1508.91 | 1509.89 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 5 — SELL (started 2025-06-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-06 14:15:00 | 1492.00 | 1508.91 | 1509.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-10 14:15:00 | 1478.20 | 1493.85 | 1499.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-11 09:15:00 | 1492.90 | 1492.09 | 1497.37 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-11 09:15:00 | 1492.90 | 1492.09 | 1497.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 09:15:00 | 1492.90 | 1492.09 | 1497.37 | EMA400 retest candle locked (from downside) |

### Cycle 6 — BUY (started 2025-06-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-11 15:15:00 | 1509.80 | 1498.70 | 1498.35 | EMA200 above EMA400 |

### Cycle 7 — SELL (started 2025-06-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-12 09:15:00 | 1491.70 | 1497.30 | 1497.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-12 10:15:00 | 1477.40 | 1493.32 | 1495.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-13 13:15:00 | 1471.20 | 1466.83 | 1476.59 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-13 14:00:00 | 1471.20 | 1466.83 | 1476.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 14:15:00 | 1469.10 | 1467.28 | 1475.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-13 15:00:00 | 1469.10 | 1467.28 | 1475.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 09:15:00 | 1447.20 | 1462.72 | 1472.30 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-16 10:30:00 | 1440.10 | 1460.58 | 1470.46 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-17 14:45:00 | 1440.80 | 1454.87 | 1461.89 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-18 10:15:00 | 1441.80 | 1450.94 | 1458.76 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-18 15:15:00 | 1439.50 | 1449.93 | 1455.35 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 09:15:00 | 1448.00 | 1447.87 | 1453.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-19 09:45:00 | 1451.70 | 1447.87 | 1453.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 10:15:00 | 1442.10 | 1446.72 | 1452.35 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-19 11:45:00 | 1435.50 | 1444.82 | 1450.97 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-24 10:15:00 | 1444.40 | 1430.40 | 1429.59 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-24 10:15:00 | 1444.40 | 1430.40 | 1429.59 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-24 10:15:00 | 1444.40 | 1430.40 | 1429.59 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-24 10:15:00 | 1444.40 | 1430.40 | 1429.59 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-24 10:15:00 | 1444.40 | 1430.40 | 1429.59 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 8 — BUY (started 2025-06-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-24 10:15:00 | 1444.40 | 1430.40 | 1429.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-25 09:15:00 | 1481.00 | 1449.39 | 1440.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-26 09:15:00 | 1482.10 | 1484.43 | 1466.52 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-26 10:00:00 | 1482.10 | 1484.43 | 1466.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 11:15:00 | 1481.00 | 1483.24 | 1469.06 | EMA400 retest candle locked (from upside) |

### Cycle 9 — SELL (started 2025-06-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-27 15:15:00 | 1463.80 | 1467.90 | 1468.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-30 10:15:00 | 1451.00 | 1464.47 | 1466.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-30 13:15:00 | 1462.30 | 1462.05 | 1464.82 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-30 13:45:00 | 1462.90 | 1462.05 | 1464.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 14:15:00 | 1468.10 | 1463.26 | 1465.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-30 14:45:00 | 1466.90 | 1463.26 | 1465.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 15:15:00 | 1462.30 | 1463.07 | 1464.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-01 09:15:00 | 1471.60 | 1463.07 | 1464.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 09:15:00 | 1456.80 | 1461.82 | 1464.13 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-01 10:45:00 | 1454.20 | 1460.63 | 1463.38 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-03 12:00:00 | 1455.00 | 1452.58 | 1453.34 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-03 12:15:00 | 1473.80 | 1456.82 | 1455.20 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-03 12:15:00 | 1473.80 | 1456.82 | 1455.20 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 10 — BUY (started 2025-07-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-03 12:15:00 | 1473.80 | 1456.82 | 1455.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-03 13:15:00 | 1483.00 | 1462.06 | 1457.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-04 09:15:00 | 1469.60 | 1470.30 | 1463.17 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-04 10:00:00 | 1469.60 | 1470.30 | 1463.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 09:15:00 | 1471.30 | 1477.27 | 1471.47 | EMA400 retest candle locked (from upside) |

### Cycle 11 — SELL (started 2025-07-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-08 09:15:00 | 1454.90 | 1467.32 | 1468.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-08 10:15:00 | 1446.10 | 1463.08 | 1466.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-08 14:15:00 | 1452.40 | 1452.30 | 1459.52 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-08 15:00:00 | 1452.40 | 1452.30 | 1459.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 10:15:00 | 1455.10 | 1452.28 | 1457.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-09 10:45:00 | 1460.80 | 1452.28 | 1457.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 11:15:00 | 1455.80 | 1452.98 | 1457.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-09 11:30:00 | 1456.60 | 1452.98 | 1457.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 12:15:00 | 1467.70 | 1455.93 | 1458.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-09 12:45:00 | 1470.10 | 1455.93 | 1458.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 13:15:00 | 1466.30 | 1458.00 | 1459.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-09 13:30:00 | 1467.80 | 1458.00 | 1459.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 12 — BUY (started 2025-07-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-09 15:15:00 | 1470.00 | 1461.31 | 1460.49 | EMA200 above EMA400 |

### Cycle 13 — SELL (started 2025-07-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-10 11:15:00 | 1457.00 | 1459.89 | 1460.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-10 12:15:00 | 1452.00 | 1458.31 | 1459.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-14 09:15:00 | 1467.00 | 1439.86 | 1445.31 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-14 09:15:00 | 1467.00 | 1439.86 | 1445.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 09:15:00 | 1467.00 | 1439.86 | 1445.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-14 10:00:00 | 1467.00 | 1439.86 | 1445.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 10:15:00 | 1466.20 | 1445.13 | 1447.21 | EMA400 retest candle locked (from downside) |

### Cycle 14 — BUY (started 2025-07-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-14 11:15:00 | 1463.00 | 1448.70 | 1448.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-15 10:15:00 | 1478.30 | 1460.16 | 1454.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-15 11:15:00 | 1458.00 | 1459.73 | 1455.15 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-15 12:00:00 | 1458.00 | 1459.73 | 1455.15 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 12:15:00 | 1450.00 | 1457.78 | 1454.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-15 12:45:00 | 1449.90 | 1457.78 | 1454.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 13:15:00 | 1447.50 | 1455.73 | 1454.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-15 13:30:00 | 1448.30 | 1455.73 | 1454.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 15 — SELL (started 2025-07-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-15 14:15:00 | 1440.00 | 1452.58 | 1452.75 | EMA200 below EMA400 |

### Cycle 16 — BUY (started 2025-07-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-16 13:15:00 | 1456.00 | 1452.04 | 1451.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-16 15:15:00 | 1464.90 | 1455.25 | 1453.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-17 11:15:00 | 1450.20 | 1457.89 | 1455.43 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-17 11:15:00 | 1450.20 | 1457.89 | 1455.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 11:15:00 | 1450.20 | 1457.89 | 1455.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-17 12:00:00 | 1450.20 | 1457.89 | 1455.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 12:15:00 | 1442.00 | 1454.71 | 1454.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-17 12:45:00 | 1445.50 | 1454.71 | 1454.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 17 — SELL (started 2025-07-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-17 13:15:00 | 1443.30 | 1452.43 | 1453.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-18 09:15:00 | 1350.60 | 1429.54 | 1442.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-22 09:15:00 | 1300.80 | 1298.32 | 1336.26 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-22 10:00:00 | 1300.80 | 1298.32 | 1336.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 09:15:00 | 1243.10 | 1240.02 | 1248.07 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-31 13:45:00 | 1224.90 | 1235.73 | 1241.49 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-01 11:45:00 | 1214.70 | 1226.89 | 1234.49 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-06 12:00:00 | 1222.00 | 1212.47 | 1213.22 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-06 12:15:00 | 1218.90 | 1213.76 | 1213.74 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-06 12:15:00 | 1218.90 | 1213.76 | 1213.74 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-06 12:15:00 | 1218.90 | 1213.76 | 1213.74 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 18 — BUY (started 2025-08-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-06 12:15:00 | 1218.90 | 1213.76 | 1213.74 | EMA200 above EMA400 |

### Cycle 19 — SELL (started 2025-08-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-06 13:15:00 | 1193.90 | 1209.79 | 1211.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-07 10:15:00 | 1187.30 | 1200.28 | 1206.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-07 13:15:00 | 1196.40 | 1195.52 | 1202.31 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-07 14:00:00 | 1196.40 | 1195.52 | 1202.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 11:15:00 | 1198.10 | 1195.95 | 1199.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-08 11:45:00 | 1199.90 | 1195.95 | 1199.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 12:15:00 | 1197.50 | 1196.26 | 1199.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-08 13:15:00 | 1199.00 | 1196.26 | 1199.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 13:15:00 | 1196.20 | 1196.25 | 1199.39 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-08 14:45:00 | 1187.40 | 1192.70 | 1197.49 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-13 10:00:00 | 1180.90 | 1168.37 | 1175.97 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-13 13:15:00 | 1213.50 | 1185.54 | 1182.27 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-13 13:15:00 | 1213.50 | 1185.54 | 1182.27 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 20 — BUY (started 2025-08-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-13 13:15:00 | 1213.50 | 1185.54 | 1182.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-13 14:15:00 | 1232.10 | 1194.86 | 1186.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-14 11:15:00 | 1195.60 | 1203.96 | 1194.81 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-14 11:15:00 | 1195.60 | 1203.96 | 1194.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 11:15:00 | 1195.60 | 1203.96 | 1194.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-14 11:45:00 | 1194.00 | 1203.96 | 1194.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 12:15:00 | 1194.00 | 1201.97 | 1194.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-14 13:15:00 | 1193.10 | 1201.97 | 1194.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 13:15:00 | 1193.60 | 1200.30 | 1194.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-14 14:15:00 | 1191.40 | 1200.30 | 1194.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 15:15:00 | 1173.90 | 1193.93 | 1192.65 | EMA400 retest candle locked (from upside) |

### Cycle 21 — SELL (started 2025-08-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-18 10:15:00 | 1181.00 | 1189.98 | 1190.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-18 11:15:00 | 1179.60 | 1187.90 | 1189.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-19 09:15:00 | 1182.60 | 1177.27 | 1183.08 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-19 09:15:00 | 1182.60 | 1177.27 | 1183.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 09:15:00 | 1182.60 | 1177.27 | 1183.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-19 10:00:00 | 1182.60 | 1177.27 | 1183.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 10:15:00 | 1189.90 | 1179.80 | 1183.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-19 11:00:00 | 1189.90 | 1179.80 | 1183.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 11:15:00 | 1191.20 | 1182.08 | 1184.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-19 12:00:00 | 1191.20 | 1182.08 | 1184.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 22 — BUY (started 2025-08-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-19 14:15:00 | 1189.60 | 1185.79 | 1185.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-20 09:15:00 | 1195.00 | 1188.29 | 1186.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-20 13:15:00 | 1181.90 | 1190.60 | 1188.90 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-20 13:15:00 | 1181.90 | 1190.60 | 1188.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 13:15:00 | 1181.90 | 1190.60 | 1188.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-20 14:00:00 | 1181.90 | 1190.60 | 1188.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 14:15:00 | 1179.90 | 1188.46 | 1188.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-20 15:00:00 | 1179.90 | 1188.46 | 1188.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 23 — SELL (started 2025-08-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-20 15:15:00 | 1180.00 | 1186.77 | 1187.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-21 09:15:00 | 1139.40 | 1177.29 | 1182.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-22 12:15:00 | 1156.60 | 1153.56 | 1163.77 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-22 13:00:00 | 1156.60 | 1153.56 | 1163.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 09:15:00 | 1144.00 | 1151.39 | 1159.47 | EMA400 retest candle locked (from downside) |

### Cycle 24 — BUY (started 2025-08-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-25 14:15:00 | 1170.00 | 1163.10 | 1162.85 | EMA200 above EMA400 |

### Cycle 25 — SELL (started 2025-08-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-26 09:15:00 | 1154.60 | 1162.10 | 1162.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-26 12:15:00 | 1145.80 | 1155.50 | 1159.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-28 13:15:00 | 1153.30 | 1145.36 | 1150.03 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-28 13:15:00 | 1153.30 | 1145.36 | 1150.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 13:15:00 | 1153.30 | 1145.36 | 1150.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-28 13:30:00 | 1154.20 | 1145.36 | 1150.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 14:15:00 | 1155.50 | 1147.39 | 1150.53 | EMA400 retest candle locked (from downside) |

### Cycle 26 — BUY (started 2025-08-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-29 09:15:00 | 1174.90 | 1154.45 | 1153.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-01 13:15:00 | 1192.80 | 1179.07 | 1170.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-02 09:15:00 | 1176.90 | 1181.17 | 1173.76 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-02 09:15:00 | 1176.90 | 1181.17 | 1173.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 09:15:00 | 1176.90 | 1181.17 | 1173.76 | EMA400 retest candle locked (from upside) |

### Cycle 27 — SELL (started 2025-09-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-03 09:15:00 | 1167.50 | 1171.00 | 1171.33 | EMA200 below EMA400 |

### Cycle 28 — BUY (started 2025-09-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-03 15:15:00 | 1177.00 | 1170.97 | 1170.94 | EMA200 above EMA400 |

### Cycle 29 — SELL (started 2025-09-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-04 09:15:00 | 1166.50 | 1170.07 | 1170.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-04 10:15:00 | 1164.70 | 1169.00 | 1170.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-05 09:15:00 | 1181.30 | 1166.50 | 1167.68 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-05 09:15:00 | 1181.30 | 1166.50 | 1167.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 09:15:00 | 1181.30 | 1166.50 | 1167.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-05 10:00:00 | 1181.30 | 1166.50 | 1167.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 30 — BUY (started 2025-09-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-05 10:15:00 | 1189.20 | 1171.04 | 1169.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-05 11:15:00 | 1190.30 | 1174.89 | 1171.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-05 15:15:00 | 1184.20 | 1184.62 | 1178.16 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-08 09:15:00 | 1172.90 | 1184.62 | 1178.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 09:15:00 | 1174.80 | 1182.66 | 1177.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-08 09:30:00 | 1171.20 | 1182.66 | 1177.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 10:15:00 | 1170.00 | 1180.13 | 1177.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-08 10:45:00 | 1169.40 | 1180.13 | 1177.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 12:15:00 | 1182.80 | 1180.18 | 1177.65 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-08 13:15:00 | 1185.90 | 1180.18 | 1177.65 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-09 10:15:00 | 1185.80 | 1184.11 | 1180.73 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-09 12:15:00 | 1174.00 | 1182.00 | 1180.65 | SL hit (close<static) qty=1.00 sl=1175.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-09 12:15:00 | 1174.00 | 1182.00 | 1180.65 | SL hit (close<static) qty=1.00 sl=1175.00 alert=retest2 |

### Cycle 31 — SELL (started 2025-09-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-09 14:15:00 | 1174.60 | 1179.42 | 1179.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-09 15:15:00 | 1167.00 | 1176.93 | 1178.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-11 10:15:00 | 1169.90 | 1164.50 | 1168.90 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-11 10:15:00 | 1169.90 | 1164.50 | 1168.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 10:15:00 | 1169.90 | 1164.50 | 1168.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-11 10:30:00 | 1172.00 | 1164.50 | 1168.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 11:15:00 | 1159.50 | 1163.50 | 1168.05 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-15 15:15:00 | 1151.70 | 1163.25 | 1164.72 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-16 09:30:00 | 1151.50 | 1159.57 | 1162.68 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-16 12:15:00 | 1177.50 | 1163.37 | 1163.59 | SL hit (close>static) qty=1.00 sl=1172.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-16 12:15:00 | 1177.50 | 1163.37 | 1163.59 | SL hit (close>static) qty=1.00 sl=1172.00 alert=retest2 |

### Cycle 32 — BUY (started 2025-09-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-16 13:15:00 | 1172.90 | 1165.28 | 1164.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-17 09:15:00 | 1195.70 | 1174.00 | 1168.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-18 12:15:00 | 1188.50 | 1193.21 | 1185.48 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-18 13:00:00 | 1188.50 | 1193.21 | 1185.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 13:15:00 | 1192.20 | 1193.01 | 1186.09 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-18 14:15:00 | 1194.00 | 1193.01 | 1186.09 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-19 10:15:00 | 1193.70 | 1192.46 | 1187.59 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-19 14:15:00 | 1178.00 | 1186.72 | 1186.47 | SL hit (close<static) qty=1.00 sl=1182.20 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-19 14:15:00 | 1178.00 | 1186.72 | 1186.47 | SL hit (close<static) qty=1.00 sl=1182.20 alert=retest2 |

### Cycle 33 — SELL (started 2025-09-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-19 15:15:00 | 1178.00 | 1184.98 | 1185.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-22 09:15:00 | 1156.30 | 1179.24 | 1183.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-24 09:15:00 | 1153.80 | 1140.66 | 1150.30 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-24 09:15:00 | 1153.80 | 1140.66 | 1150.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 09:15:00 | 1153.80 | 1140.66 | 1150.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-24 10:00:00 | 1153.80 | 1140.66 | 1150.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 10:15:00 | 1147.00 | 1141.93 | 1150.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-24 10:45:00 | 1151.20 | 1141.93 | 1150.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 14:15:00 | 1102.60 | 1096.62 | 1106.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-29 14:45:00 | 1100.40 | 1096.62 | 1106.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 15:15:00 | 1101.20 | 1097.54 | 1106.38 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-30 09:30:00 | 1098.90 | 1098.21 | 1105.88 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-30 10:45:00 | 1098.50 | 1098.65 | 1105.38 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-30 11:15:00 | 1094.40 | 1098.65 | 1105.38 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-01 09:15:00 | 1092.70 | 1094.42 | 1099.99 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 09:15:00 | 1093.60 | 1094.25 | 1099.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-01 09:30:00 | 1099.30 | 1094.25 | 1099.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 11:15:00 | 1097.80 | 1094.57 | 1098.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-01 11:30:00 | 1096.90 | 1094.57 | 1098.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 12:15:00 | 1094.90 | 1094.64 | 1098.30 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-03 10:30:00 | 1092.90 | 1096.01 | 1097.84 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-03 11:15:00 | 1092.70 | 1096.01 | 1097.84 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-03 12:00:00 | 1088.80 | 1094.57 | 1097.02 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-09 11:15:00 | 1043.95 | 1051.29 | 1059.52 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-09 11:15:00 | 1043.58 | 1051.29 | 1059.52 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-09 11:15:00 | 1039.68 | 1051.29 | 1059.52 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-09 12:15:00 | 1038.07 | 1048.83 | 1057.65 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-09 12:15:00 | 1038.26 | 1048.83 | 1057.65 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-09 12:15:00 | 1038.07 | 1048.83 | 1057.65 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-10 09:15:00 | 1054.70 | 1045.75 | 1052.90 | SL hit (close>ema200) qty=0.50 sl=1045.75 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-10 09:15:00 | 1054.70 | 1045.75 | 1052.90 | SL hit (close>ema200) qty=0.50 sl=1045.75 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-10 09:15:00 | 1054.70 | 1045.75 | 1052.90 | SL hit (close>ema200) qty=0.50 sl=1045.75 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-10 09:15:00 | 1054.70 | 1045.75 | 1052.90 | SL hit (close>ema200) qty=0.50 sl=1045.75 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-10 09:15:00 | 1054.70 | 1045.75 | 1052.90 | SL hit (close>ema200) qty=0.50 sl=1045.75 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-10 09:15:00 | 1054.70 | 1045.75 | 1052.90 | SL hit (close>ema200) qty=0.50 sl=1045.75 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-10 09:15:00 | 1034.36 | 1045.75 | 1052.90 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-10 09:15:00 | 1054.70 | 1045.75 | 1052.90 | SL hit (close>static) qty=0.50 sl=1045.75 alert=retest2 |

### Cycle 34 — BUY (started 2025-10-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-10 13:15:00 | 1068.60 | 1057.15 | 1056.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-10 14:15:00 | 1076.80 | 1061.08 | 1058.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-13 12:15:00 | 1066.90 | 1069.61 | 1064.56 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-13 12:15:00 | 1066.90 | 1069.61 | 1064.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 12:15:00 | 1066.90 | 1069.61 | 1064.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-13 13:00:00 | 1066.90 | 1069.61 | 1064.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 13:15:00 | 1065.50 | 1068.79 | 1064.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-13 14:00:00 | 1065.50 | 1068.79 | 1064.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 14:15:00 | 1065.00 | 1068.03 | 1064.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-13 15:15:00 | 1064.00 | 1068.03 | 1064.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 15:15:00 | 1064.00 | 1067.23 | 1064.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-14 09:15:00 | 1059.10 | 1067.23 | 1064.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 09:15:00 | 1053.00 | 1064.38 | 1063.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-14 10:00:00 | 1053.00 | 1064.38 | 1063.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 35 — SELL (started 2025-10-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-14 10:15:00 | 1043.50 | 1060.20 | 1061.74 | EMA200 below EMA400 |

### Cycle 36 — BUY (started 2025-10-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-16 13:15:00 | 1056.40 | 1051.99 | 1051.85 | EMA200 above EMA400 |

### Cycle 37 — SELL (started 2025-10-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-17 10:15:00 | 1048.20 | 1051.72 | 1051.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-17 12:15:00 | 1045.00 | 1049.93 | 1051.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-17 14:15:00 | 1049.60 | 1049.06 | 1050.37 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-17 14:15:00 | 1049.60 | 1049.06 | 1050.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 14:15:00 | 1049.60 | 1049.06 | 1050.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-17 14:45:00 | 1050.70 | 1049.06 | 1050.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 15:15:00 | 1053.60 | 1049.96 | 1050.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-20 09:15:00 | 1044.60 | 1049.96 | 1050.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 09:15:00 | 1040.70 | 1048.11 | 1049.76 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-20 10:45:00 | 1039.70 | 1046.17 | 1048.72 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-20 13:15:00 | 1039.70 | 1044.03 | 1047.25 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-20 14:00:00 | 1038.60 | 1042.95 | 1046.46 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-20 14:45:00 | 1039.00 | 1041.42 | 1045.45 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-21 13:15:00 | 1049.60 | 1042.11 | 1045.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-21 14:00:00 | 1049.60 | 1042.11 | 1045.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-21 14:15:00 | 1053.50 | 1044.39 | 1045.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-23 09:15:00 | 1052.80 | 1044.39 | 1045.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-10-23 10:15:00 | 1056.90 | 1048.68 | 1047.60 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-23 10:15:00 | 1056.90 | 1048.68 | 1047.60 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-23 10:15:00 | 1056.90 | 1048.68 | 1047.60 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-23 10:15:00 | 1056.90 | 1048.68 | 1047.60 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 38 — BUY (started 2025-10-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-23 10:15:00 | 1056.90 | 1048.68 | 1047.60 | EMA200 above EMA400 |

### Cycle 39 — SELL (started 2025-10-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-27 12:15:00 | 1046.50 | 1053.10 | 1053.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-27 13:15:00 | 1044.20 | 1051.32 | 1052.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-28 12:15:00 | 1047.20 | 1046.89 | 1049.50 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-28 12:15:00 | 1047.20 | 1046.89 | 1049.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 12:15:00 | 1047.20 | 1046.89 | 1049.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-28 12:30:00 | 1045.20 | 1046.89 | 1049.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 13:15:00 | 1047.00 | 1046.91 | 1049.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-28 13:45:00 | 1050.50 | 1046.91 | 1049.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 14:15:00 | 1047.90 | 1047.11 | 1049.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-28 14:45:00 | 1048.00 | 1047.11 | 1049.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 15:15:00 | 1049.90 | 1047.67 | 1049.22 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-29 09:15:00 | 1046.70 | 1047.67 | 1049.22 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-29 11:15:00 | 1052.00 | 1047.75 | 1048.79 | SL hit (close>static) qty=1.00 sl=1050.00 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-29 12:45:00 | 1045.20 | 1047.26 | 1048.47 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-30 11:15:00 | 1051.50 | 1049.01 | 1048.87 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 40 — BUY (started 2025-10-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-30 11:15:00 | 1051.50 | 1049.01 | 1048.87 | EMA200 above EMA400 |

### Cycle 41 — SELL (started 2025-10-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-30 12:15:00 | 1029.10 | 1045.03 | 1047.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-30 13:15:00 | 1025.00 | 1041.02 | 1045.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-03 11:15:00 | 1010.90 | 1010.48 | 1020.65 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-03 12:00:00 | 1010.90 | 1010.48 | 1020.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 12:15:00 | 1025.30 | 1013.44 | 1021.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-03 13:00:00 | 1025.30 | 1013.44 | 1021.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 13:15:00 | 1021.00 | 1014.95 | 1021.07 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-04 10:00:00 | 1007.60 | 1013.32 | 1018.80 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-07 09:15:00 | 957.22 | 976.56 | 989.91 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-10 09:15:00 | 950.80 | 949.45 | 966.60 | SL hit (close>ema200) qty=0.50 sl=949.45 alert=retest2 |

### Cycle 42 — BUY (started 2025-11-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-13 11:15:00 | 951.00 | 944.21 | 943.51 | EMA200 above EMA400 |

### Cycle 43 — SELL (started 2025-11-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-14 09:15:00 | 931.50 | 941.20 | 942.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-14 11:15:00 | 925.90 | 936.64 | 940.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-17 09:15:00 | 933.40 | 929.01 | 934.22 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-17 09:15:00 | 933.40 | 929.01 | 934.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 09:15:00 | 933.40 | 929.01 | 934.22 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-19 12:30:00 | 923.50 | 928.20 | 930.71 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-20 11:15:00 | 952.90 | 932.92 | 931.28 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 44 — BUY (started 2025-11-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-20 11:15:00 | 952.90 | 932.92 | 931.28 | EMA200 above EMA400 |

### Cycle 45 — SELL (started 2025-11-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-20 15:15:00 | 926.70 | 929.77 | 930.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-21 09:15:00 | 918.30 | 927.48 | 929.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-21 11:15:00 | 927.70 | 926.65 | 928.38 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-21 11:15:00 | 927.70 | 926.65 | 928.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 11:15:00 | 927.70 | 926.65 | 928.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-21 12:00:00 | 927.70 | 926.65 | 928.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 12:15:00 | 924.10 | 926.14 | 927.99 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-21 13:45:00 | 920.10 | 925.11 | 927.36 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-24 14:15:00 | 954.20 | 924.76 | 924.52 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 46 — BUY (started 2025-11-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-24 14:15:00 | 954.20 | 924.76 | 924.52 | EMA200 above EMA400 |

### Cycle 47 — SELL (started 2025-11-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-26 09:15:00 | 920.10 | 926.23 | 926.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-27 09:15:00 | 916.10 | 920.27 | 922.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-27 12:15:00 | 919.30 | 918.43 | 921.29 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-27 13:00:00 | 919.30 | 918.43 | 921.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 13:15:00 | 921.30 | 919.00 | 921.29 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-28 09:15:00 | 916.60 | 918.81 | 920.80 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-28 09:45:00 | 915.00 | 918.15 | 920.32 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-10 09:15:00 | 907.55 | 893.12 | 891.63 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-10 09:15:00 | 907.55 | 893.12 | 891.63 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 48 — BUY (started 2025-12-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-10 09:15:00 | 907.55 | 893.12 | 891.63 | EMA200 above EMA400 |

### Cycle 49 — SELL (started 2025-12-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-10 15:15:00 | 882.95 | 891.13 | 891.69 | EMA200 below EMA400 |

### Cycle 50 — BUY (started 2025-12-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-11 09:15:00 | 896.10 | 892.12 | 892.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-11 14:15:00 | 900.00 | 895.48 | 893.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-11 15:15:00 | 895.20 | 895.42 | 894.01 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-12 09:15:00 | 906.15 | 895.42 | 894.01 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 15:15:00 | 908.00 | 910.32 | 906.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 09:15:00 | 908.45 | 910.32 | 906.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 09:15:00 | 909.70 | 910.19 | 907.12 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-12-16 12:15:00 | 904.60 | 908.70 | 907.19 | SL hit (close<ema400) qty=1.00 sl=907.19 alert=retest1 |

### Cycle 51 — SELL (started 2025-12-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-16 14:15:00 | 898.10 | 904.77 | 905.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-17 10:15:00 | 891.15 | 900.62 | 903.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-18 14:15:00 | 905.70 | 889.53 | 893.18 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-18 14:15:00 | 905.70 | 889.53 | 893.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 14:15:00 | 905.70 | 889.53 | 893.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-18 15:00:00 | 905.70 | 889.53 | 893.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 15:15:00 | 890.10 | 889.64 | 892.90 | EMA400 retest candle locked (from downside) |

### Cycle 52 — BUY (started 2025-12-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-19 14:15:00 | 902.30 | 895.96 | 895.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-22 09:15:00 | 910.15 | 899.52 | 896.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-23 11:15:00 | 899.65 | 904.23 | 901.66 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-23 11:15:00 | 899.65 | 904.23 | 901.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 11:15:00 | 899.65 | 904.23 | 901.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-23 11:30:00 | 898.05 | 904.23 | 901.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 12:15:00 | 912.25 | 905.84 | 902.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-23 12:30:00 | 899.00 | 905.84 | 902.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 09:15:00 | 901.35 | 905.63 | 903.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-24 10:15:00 | 903.60 | 905.63 | 903.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 10:15:00 | 900.55 | 904.61 | 903.41 | EMA400 retest candle locked (from upside) |

### Cycle 53 — SELL (started 2025-12-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-24 12:15:00 | 898.15 | 902.62 | 902.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-24 13:15:00 | 897.45 | 901.58 | 902.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-29 13:15:00 | 879.85 | 876.51 | 884.68 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-29 14:00:00 | 879.85 | 876.51 | 884.68 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 14:15:00 | 869.45 | 875.10 | 883.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-29 14:45:00 | 884.60 | 875.10 | 883.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 10:15:00 | 892.95 | 877.87 | 882.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-30 11:00:00 | 892.95 | 877.87 | 882.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 11:15:00 | 891.15 | 880.53 | 883.24 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-30 15:00:00 | 884.55 | 883.42 | 884.07 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-07 11:15:00 | 869.20 | 866.55 | 866.42 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 54 — BUY (started 2026-01-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-07 11:15:00 | 869.20 | 866.55 | 866.42 | EMA200 above EMA400 |

### Cycle 55 — SELL (started 2026-01-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-08 11:15:00 | 858.95 | 865.51 | 866.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-09 09:15:00 | 854.45 | 860.25 | 863.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-12 12:15:00 | 855.00 | 853.40 | 856.51 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-12 12:15:00 | 855.00 | 853.40 | 856.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 12:15:00 | 855.00 | 853.40 | 856.51 | EMA400 retest candle locked (from downside) |

### Cycle 56 — BUY (started 2026-01-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-14 11:15:00 | 861.05 | 857.64 | 857.19 | EMA200 above EMA400 |

### Cycle 57 — SELL (started 2026-01-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-19 10:15:00 | 855.10 | 858.95 | 859.01 | EMA200 below EMA400 |

### Cycle 58 — BUY (started 2026-01-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-21 10:15:00 | 858.95 | 856.96 | 856.95 | EMA200 above EMA400 |

### Cycle 59 — SELL (started 2026-01-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-21 11:15:00 | 856.80 | 856.93 | 856.94 | EMA200 below EMA400 |

### Cycle 60 — BUY (started 2026-01-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-21 14:15:00 | 862.95 | 857.91 | 857.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-22 11:15:00 | 873.80 | 862.58 | 859.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-23 10:15:00 | 867.35 | 870.55 | 866.00 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-23 10:30:00 | 869.15 | 870.55 | 866.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 11:15:00 | 853.90 | 867.22 | 864.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-23 12:00:00 | 853.90 | 867.22 | 864.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 12:15:00 | 853.25 | 864.43 | 863.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-23 13:00:00 | 853.25 | 864.43 | 863.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 61 — SELL (started 2026-01-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-23 13:15:00 | 854.95 | 862.53 | 863.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-27 11:15:00 | 850.10 | 855.90 | 859.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-27 13:15:00 | 856.35 | 855.10 | 858.25 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-27 13:15:00 | 856.35 | 855.10 | 858.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 13:15:00 | 856.35 | 855.10 | 858.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-27 13:45:00 | 856.00 | 855.10 | 858.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 15:15:00 | 857.00 | 855.54 | 857.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-28 09:15:00 | 855.05 | 855.54 | 857.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 09:15:00 | 856.85 | 855.80 | 857.81 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-29 09:30:00 | 852.00 | 855.99 | 857.17 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-30 12:15:00 | 861.00 | 851.53 | 852.04 | SL hit (close>static) qty=1.00 sl=858.95 alert=retest2 |

### Cycle 62 — BUY (started 2026-01-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-30 13:15:00 | 861.00 | 853.43 | 852.85 | EMA200 above EMA400 |

### Cycle 63 — SELL (started 2026-02-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-01 09:15:00 | 831.50 | 851.64 | 852.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-01 12:15:00 | 807.80 | 836.91 | 845.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-03 09:15:00 | 825.00 | 804.91 | 815.38 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-03 09:15:00 | 825.00 | 804.91 | 815.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 09:15:00 | 825.00 | 804.91 | 815.38 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-03 10:45:00 | 820.45 | 808.83 | 816.21 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-03 12:30:00 | 822.05 | 814.57 | 817.70 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-03 15:00:00 | 822.00 | 817.72 | 818.69 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-04 09:15:00 | 816.10 | 818.72 | 819.05 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 14:15:00 | 809.75 | 808.70 | 811.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-05 14:30:00 | 809.70 | 808.70 | 811.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 09:15:00 | 800.60 | 800.32 | 804.37 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-09 11:30:00 | 799.80 | 799.99 | 803.55 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-09 13:15:00 | 799.35 | 800.01 | 803.24 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-09 14:30:00 | 799.55 | 799.84 | 802.60 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-10 09:15:00 | 816.75 | 803.50 | 803.80 | SL hit (close>static) qty=1.00 sl=805.20 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-10 09:15:00 | 816.75 | 803.50 | 803.80 | SL hit (close>static) qty=1.00 sl=805.20 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-10 09:15:00 | 816.75 | 803.50 | 803.80 | SL hit (close>static) qty=1.00 sl=805.20 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-10 10:15:00 | 810.05 | 804.81 | 804.37 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-10 10:15:00 | 810.05 | 804.81 | 804.37 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-10 10:15:00 | 810.05 | 804.81 | 804.37 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-10 10:15:00 | 810.05 | 804.81 | 804.37 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 64 — BUY (started 2026-02-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-10 10:15:00 | 810.05 | 804.81 | 804.37 | EMA200 above EMA400 |

### Cycle 65 — SELL (started 2026-02-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-10 14:15:00 | 799.50 | 803.68 | 804.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-10 15:15:00 | 796.00 | 802.15 | 803.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-13 12:15:00 | 763.50 | 763.24 | 773.40 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-13 12:30:00 | 763.65 | 763.24 | 773.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 09:15:00 | 744.30 | 746.41 | 751.75 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-18 10:15:00 | 741.75 | 746.41 | 751.75 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-18 13:15:00 | 762.20 | 750.48 | 751.97 | SL hit (close>static) qty=1.00 sl=755.25 alert=retest2 |

### Cycle 66 — BUY (started 2026-02-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-18 15:15:00 | 761.00 | 754.01 | 753.40 | EMA200 above EMA400 |

### Cycle 67 — SELL (started 2026-02-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-19 10:15:00 | 746.45 | 752.11 | 752.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-19 11:15:00 | 744.10 | 750.51 | 751.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-23 10:15:00 | 722.05 | 719.87 | 729.96 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-23 10:30:00 | 721.25 | 719.87 | 729.96 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 12:15:00 | 725.60 | 721.28 | 728.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-23 12:45:00 | 727.85 | 721.28 | 728.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 13:15:00 | 725.85 | 722.19 | 728.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-23 13:30:00 | 728.75 | 722.19 | 728.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 14:15:00 | 729.00 | 723.55 | 728.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-23 15:00:00 | 729.00 | 723.55 | 728.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 15:15:00 | 728.70 | 724.58 | 728.64 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-24 09:15:00 | 718.05 | 724.58 | 728.64 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-24 10:45:00 | 726.40 | 724.42 | 727.83 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-25 11:00:00 | 726.20 | 725.02 | 726.56 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-26 09:15:00 | 731.05 | 724.26 | 725.27 | SL hit (close>static) qty=1.00 sl=729.40 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-26 09:15:00 | 731.05 | 724.26 | 725.27 | SL hit (close>static) qty=1.00 sl=729.40 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-26 09:15:00 | 731.05 | 724.26 | 725.27 | SL hit (close>static) qty=1.00 sl=729.40 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-26 10:45:00 | 726.80 | 724.71 | 725.38 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 14:15:00 | 724.65 | 722.59 | 723.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-26 15:00:00 | 724.65 | 722.59 | 723.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 15:15:00 | 722.25 | 722.52 | 723.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-27 09:15:00 | 715.30 | 722.52 | 723.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 09:15:00 | 718.45 | 721.71 | 723.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-27 10:15:00 | 722.85 | 721.71 | 723.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 10:15:00 | 725.85 | 722.53 | 723.57 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-02-27 12:15:00 | 729.85 | 724.42 | 724.27 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 68 — BUY (started 2026-02-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-27 12:15:00 | 729.85 | 724.42 | 724.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-27 13:15:00 | 735.60 | 726.65 | 725.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-02 09:15:00 | 724.30 | 728.46 | 726.67 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-02 09:15:00 | 724.30 | 728.46 | 726.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 09:15:00 | 724.30 | 728.46 | 726.67 | EMA400 retest candle locked (from upside) |

### Cycle 69 — SELL (started 2026-03-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-02 11:15:00 | 716.00 | 724.82 | 725.25 | EMA200 below EMA400 |

### Cycle 70 — BUY (started 2026-03-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-02 14:15:00 | 742.00 | 727.54 | 726.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-05 09:15:00 | 754.10 | 740.30 | 734.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-06 10:15:00 | 761.80 | 763.29 | 752.24 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-06 11:00:00 | 761.80 | 763.29 | 752.24 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 15:15:00 | 752.80 | 759.06 | 754.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-09 09:15:00 | 734.30 | 759.06 | 754.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-09 09:15:00 | 736.50 | 754.55 | 752.62 | EMA400 retest candle locked (from upside) |

### Cycle 71 — SELL (started 2026-03-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-09 10:15:00 | 737.40 | 751.12 | 751.23 | EMA200 below EMA400 |

### Cycle 72 — BUY (started 2026-03-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-11 09:15:00 | 751.80 | 747.54 | 747.39 | EMA200 above EMA400 |

### Cycle 73 — SELL (started 2026-03-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-11 14:15:00 | 742.80 | 747.51 | 747.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-12 09:15:00 | 727.05 | 742.14 | 745.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-12 13:15:00 | 740.70 | 738.65 | 742.16 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-12 14:00:00 | 740.70 | 738.65 | 742.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 09:15:00 | 702.30 | 694.79 | 700.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-18 09:45:00 | 704.95 | 694.79 | 700.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 10:15:00 | 709.95 | 697.82 | 701.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-18 10:45:00 | 710.60 | 697.82 | 701.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 11:15:00 | 705.70 | 699.40 | 701.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-18 12:15:00 | 711.60 | 699.40 | 701.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 12:15:00 | 716.50 | 702.82 | 703.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-18 13:00:00 | 716.50 | 702.82 | 703.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 74 — BUY (started 2026-03-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 13:15:00 | 718.85 | 706.02 | 704.71 | EMA200 above EMA400 |

### Cycle 75 — SELL (started 2026-03-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 14:15:00 | 696.25 | 704.48 | 705.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-23 09:15:00 | 687.35 | 697.58 | 700.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 10:15:00 | 684.35 | 683.61 | 690.01 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-24 11:00:00 | 684.35 | 683.61 | 690.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 11:15:00 | 690.50 | 684.99 | 690.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 11:45:00 | 692.90 | 684.99 | 690.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 12:15:00 | 704.15 | 688.82 | 691.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 12:45:00 | 703.55 | 688.82 | 691.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 13:15:00 | 707.75 | 692.61 | 692.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 14:00:00 | 707.75 | 692.61 | 692.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 76 — BUY (started 2026-03-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-24 14:15:00 | 699.65 | 694.01 | 693.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 09:15:00 | 720.85 | 699.65 | 696.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-25 14:15:00 | 702.85 | 707.60 | 702.39 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-25 14:15:00 | 702.85 | 707.60 | 702.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 14:15:00 | 702.85 | 707.60 | 702.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-25 15:00:00 | 702.85 | 707.60 | 702.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 15:15:00 | 703.00 | 706.68 | 702.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 09:15:00 | 701.00 | 706.68 | 702.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 09:15:00 | 696.50 | 704.65 | 701.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 09:45:00 | 693.65 | 704.65 | 701.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 10:15:00 | 690.15 | 701.75 | 700.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 10:30:00 | 691.40 | 701.75 | 700.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 77 — SELL (started 2026-03-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 11:15:00 | 682.75 | 697.95 | 699.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-27 14:15:00 | 680.90 | 690.67 | 695.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 695.35 | 673.31 | 680.53 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 695.35 | 673.31 | 680.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 695.35 | 673.31 | 680.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 10:00:00 | 695.35 | 673.31 | 680.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 10:15:00 | 700.05 | 678.66 | 682.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 11:00:00 | 700.05 | 678.66 | 682.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 78 — BUY (started 2026-04-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 11:15:00 | 709.95 | 684.92 | 684.82 | EMA200 above EMA400 |

### Cycle 79 — SELL (started 2026-04-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-07 10:15:00 | 690.75 | 694.81 | 695.21 | EMA200 below EMA400 |

### Cycle 80 — BUY (started 2026-04-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-08 09:15:00 | 716.05 | 698.10 | 696.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-08 10:15:00 | 722.25 | 702.93 | 698.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-09 09:15:00 | 720.45 | 720.67 | 711.12 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-09 10:00:00 | 720.45 | 720.67 | 711.12 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 13:15:00 | 716.50 | 721.21 | 714.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-09 13:45:00 | 716.90 | 721.21 | 714.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 14:15:00 | 717.95 | 720.56 | 714.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-09 14:30:00 | 715.75 | 720.56 | 714.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 722.30 | 724.20 | 720.82 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 10:15:00 | 726.50 | 724.20 | 720.82 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 13:30:00 | 725.95 | 723.20 | 721.36 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-15 09:15:00 | 733.50 | 722.91 | 721.54 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-20 15:15:00 | 740.00 | 747.89 | 748.61 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-20 15:15:00 | 740.00 | 747.89 | 748.61 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-20 15:15:00 | 740.00 | 747.89 | 748.61 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 81 — SELL (started 2026-04-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-20 15:15:00 | 740.00 | 747.89 | 748.61 | EMA200 below EMA400 |

### Cycle 82 — BUY (started 2026-04-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-21 10:15:00 | 755.85 | 750.30 | 749.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-22 12:15:00 | 766.45 | 755.66 | 752.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-24 12:15:00 | 802.55 | 802.90 | 790.01 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-24 12:30:00 | 803.00 | 802.90 | 790.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 15:15:00 | 807.00 | 810.46 | 803.37 | EMA400 retest candle locked (from upside) |

### Cycle 83 — SELL (started 2026-04-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-28 15:15:00 | 799.00 | 801.41 | 801.56 | EMA200 below EMA400 |

### Cycle 84 — BUY (started 2026-04-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-29 09:15:00 | 811.45 | 803.42 | 802.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-29 14:15:00 | 818.65 | 810.85 | 806.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-30 09:15:00 | 809.55 | 813.18 | 808.83 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-30 10:00:00 | 809.55 | 813.18 | 808.83 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 10:15:00 | 806.30 | 811.80 | 808.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-30 10:45:00 | 803.85 | 811.80 | 808.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 11:15:00 | 812.10 | 811.86 | 808.92 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-30 12:15:00 | 814.80 | 811.86 | 808.92 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-05 11:00:00 | 813.70 | 819.41 | 817.62 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2026-05-08 13:15:00 | 896.28 | 882.25 | 866.44 | Target hit (10%) qty=1.00 alert=retest2 |
| Target hit | 2026-05-08 13:15:00 | 895.07 | 882.25 | 866.44 | Target hit (10%) qty=1.00 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-05-30 15:00:00 | 1548.10 | 2025-06-06 14:15:00 | 1492.00 | STOP_HIT | 1.00 | -3.62% |
| BUY | retest2 | 2025-06-03 13:15:00 | 1491.70 | 2025-06-06 14:15:00 | 1492.00 | STOP_HIT | 1.00 | 0.02% |
| BUY | retest2 | 2025-06-03 14:00:00 | 1491.20 | 2025-06-06 14:15:00 | 1492.00 | STOP_HIT | 1.00 | 0.05% |
| BUY | retest2 | 2025-06-04 09:15:00 | 1496.10 | 2025-06-06 14:15:00 | 1492.00 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest2 | 2025-06-05 09:30:00 | 1496.80 | 2025-06-06 14:15:00 | 1492.00 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest2 | 2025-06-16 10:30:00 | 1440.10 | 2025-06-24 10:15:00 | 1444.40 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest2 | 2025-06-17 14:45:00 | 1440.80 | 2025-06-24 10:15:00 | 1444.40 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest2 | 2025-06-18 10:15:00 | 1441.80 | 2025-06-24 10:15:00 | 1444.40 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest2 | 2025-06-18 15:15:00 | 1439.50 | 2025-06-24 10:15:00 | 1444.40 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest2 | 2025-06-19 11:45:00 | 1435.50 | 2025-06-24 10:15:00 | 1444.40 | STOP_HIT | 1.00 | -0.62% |
| SELL | retest2 | 2025-07-01 10:45:00 | 1454.20 | 2025-07-03 12:15:00 | 1473.80 | STOP_HIT | 1.00 | -1.35% |
| SELL | retest2 | 2025-07-03 12:00:00 | 1455.00 | 2025-07-03 12:15:00 | 1473.80 | STOP_HIT | 1.00 | -1.29% |
| SELL | retest2 | 2025-07-31 13:45:00 | 1224.90 | 2025-08-06 12:15:00 | 1218.90 | STOP_HIT | 1.00 | 0.49% |
| SELL | retest2 | 2025-08-01 11:45:00 | 1214.70 | 2025-08-06 12:15:00 | 1218.90 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest2 | 2025-08-06 12:00:00 | 1222.00 | 2025-08-06 12:15:00 | 1218.90 | STOP_HIT | 1.00 | 0.25% |
| SELL | retest2 | 2025-08-08 14:45:00 | 1187.40 | 2025-08-13 13:15:00 | 1213.50 | STOP_HIT | 1.00 | -2.20% |
| SELL | retest2 | 2025-08-13 10:00:00 | 1180.90 | 2025-08-13 13:15:00 | 1213.50 | STOP_HIT | 1.00 | -2.76% |
| BUY | retest2 | 2025-09-08 13:15:00 | 1185.90 | 2025-09-09 12:15:00 | 1174.00 | STOP_HIT | 1.00 | -1.00% |
| BUY | retest2 | 2025-09-09 10:15:00 | 1185.80 | 2025-09-09 12:15:00 | 1174.00 | STOP_HIT | 1.00 | -1.00% |
| SELL | retest2 | 2025-09-15 15:15:00 | 1151.70 | 2025-09-16 12:15:00 | 1177.50 | STOP_HIT | 1.00 | -2.24% |
| SELL | retest2 | 2025-09-16 09:30:00 | 1151.50 | 2025-09-16 12:15:00 | 1177.50 | STOP_HIT | 1.00 | -2.26% |
| BUY | retest2 | 2025-09-18 14:15:00 | 1194.00 | 2025-09-19 14:15:00 | 1178.00 | STOP_HIT | 1.00 | -1.34% |
| BUY | retest2 | 2025-09-19 10:15:00 | 1193.70 | 2025-09-19 14:15:00 | 1178.00 | STOP_HIT | 1.00 | -1.32% |
| SELL | retest2 | 2025-09-30 09:30:00 | 1098.90 | 2025-10-09 11:15:00 | 1043.95 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-30 10:45:00 | 1098.50 | 2025-10-09 11:15:00 | 1043.58 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-30 11:15:00 | 1094.40 | 2025-10-09 11:15:00 | 1039.68 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-01 09:15:00 | 1092.70 | 2025-10-09 12:15:00 | 1038.07 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-03 10:30:00 | 1092.90 | 2025-10-09 12:15:00 | 1038.26 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-03 11:15:00 | 1092.70 | 2025-10-09 12:15:00 | 1038.07 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-30 09:30:00 | 1098.90 | 2025-10-10 09:15:00 | 1054.70 | STOP_HIT | 0.50 | 4.02% |
| SELL | retest2 | 2025-09-30 10:45:00 | 1098.50 | 2025-10-10 09:15:00 | 1054.70 | STOP_HIT | 0.50 | 3.99% |
| SELL | retest2 | 2025-09-30 11:15:00 | 1094.40 | 2025-10-10 09:15:00 | 1054.70 | STOP_HIT | 0.50 | 3.63% |
| SELL | retest2 | 2025-10-01 09:15:00 | 1092.70 | 2025-10-10 09:15:00 | 1054.70 | STOP_HIT | 0.50 | 3.48% |
| SELL | retest2 | 2025-10-03 10:30:00 | 1092.90 | 2025-10-10 09:15:00 | 1054.70 | STOP_HIT | 0.50 | 3.50% |
| SELL | retest2 | 2025-10-03 11:15:00 | 1092.70 | 2025-10-10 09:15:00 | 1054.70 | STOP_HIT | 0.50 | 3.48% |
| SELL | retest2 | 2025-10-03 12:00:00 | 1088.80 | 2025-10-10 09:15:00 | 1034.36 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-03 12:00:00 | 1088.80 | 2025-10-10 09:15:00 | 1054.70 | STOP_HIT | 0.50 | 3.13% |
| SELL | retest2 | 2025-10-20 10:45:00 | 1039.70 | 2025-10-23 10:15:00 | 1056.90 | STOP_HIT | 1.00 | -1.65% |
| SELL | retest2 | 2025-10-20 13:15:00 | 1039.70 | 2025-10-23 10:15:00 | 1056.90 | STOP_HIT | 1.00 | -1.65% |
| SELL | retest2 | 2025-10-20 14:00:00 | 1038.60 | 2025-10-23 10:15:00 | 1056.90 | STOP_HIT | 1.00 | -1.76% |
| SELL | retest2 | 2025-10-20 14:45:00 | 1039.00 | 2025-10-23 10:15:00 | 1056.90 | STOP_HIT | 1.00 | -1.72% |
| SELL | retest2 | 2025-10-29 09:15:00 | 1046.70 | 2025-10-29 11:15:00 | 1052.00 | STOP_HIT | 1.00 | -0.51% |
| SELL | retest2 | 2025-10-29 12:45:00 | 1045.20 | 2025-10-30 11:15:00 | 1051.50 | STOP_HIT | 1.00 | -0.60% |
| SELL | retest2 | 2025-11-04 10:00:00 | 1007.60 | 2025-11-07 09:15:00 | 957.22 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-04 10:00:00 | 1007.60 | 2025-11-10 09:15:00 | 950.80 | STOP_HIT | 0.50 | 5.64% |
| SELL | retest2 | 2025-11-19 12:30:00 | 923.50 | 2025-11-20 11:15:00 | 952.90 | STOP_HIT | 1.00 | -3.18% |
| SELL | retest2 | 2025-11-21 13:45:00 | 920.10 | 2025-11-24 14:15:00 | 954.20 | STOP_HIT | 1.00 | -3.71% |
| SELL | retest2 | 2025-11-28 09:15:00 | 916.60 | 2025-12-10 09:15:00 | 907.55 | STOP_HIT | 1.00 | 0.99% |
| SELL | retest2 | 2025-11-28 09:45:00 | 915.00 | 2025-12-10 09:15:00 | 907.55 | STOP_HIT | 1.00 | 0.81% |
| BUY | retest1 | 2025-12-12 09:15:00 | 906.15 | 2025-12-16 12:15:00 | 904.60 | STOP_HIT | 1.00 | -0.17% |
| SELL | retest2 | 2025-12-30 15:00:00 | 884.55 | 2026-01-07 11:15:00 | 869.20 | STOP_HIT | 1.00 | 1.74% |
| SELL | retest2 | 2026-01-29 09:30:00 | 852.00 | 2026-01-30 12:15:00 | 861.00 | STOP_HIT | 1.00 | -1.06% |
| SELL | retest2 | 2026-02-03 10:45:00 | 820.45 | 2026-02-10 09:15:00 | 816.75 | STOP_HIT | 1.00 | 0.45% |
| SELL | retest2 | 2026-02-03 12:30:00 | 822.05 | 2026-02-10 09:15:00 | 816.75 | STOP_HIT | 1.00 | 0.64% |
| SELL | retest2 | 2026-02-03 15:00:00 | 822.00 | 2026-02-10 09:15:00 | 816.75 | STOP_HIT | 1.00 | 0.64% |
| SELL | retest2 | 2026-02-04 09:15:00 | 816.10 | 2026-02-10 10:15:00 | 810.05 | STOP_HIT | 1.00 | 0.74% |
| SELL | retest2 | 2026-02-09 11:30:00 | 799.80 | 2026-02-10 10:15:00 | 810.05 | STOP_HIT | 1.00 | -1.28% |
| SELL | retest2 | 2026-02-09 13:15:00 | 799.35 | 2026-02-10 10:15:00 | 810.05 | STOP_HIT | 1.00 | -1.34% |
| SELL | retest2 | 2026-02-09 14:30:00 | 799.55 | 2026-02-10 10:15:00 | 810.05 | STOP_HIT | 1.00 | -1.31% |
| SELL | retest2 | 2026-02-18 10:15:00 | 741.75 | 2026-02-18 13:15:00 | 762.20 | STOP_HIT | 1.00 | -2.76% |
| SELL | retest2 | 2026-02-24 09:15:00 | 718.05 | 2026-02-26 09:15:00 | 731.05 | STOP_HIT | 1.00 | -1.81% |
| SELL | retest2 | 2026-02-24 10:45:00 | 726.40 | 2026-02-26 09:15:00 | 731.05 | STOP_HIT | 1.00 | -0.64% |
| SELL | retest2 | 2026-02-25 11:00:00 | 726.20 | 2026-02-26 09:15:00 | 731.05 | STOP_HIT | 1.00 | -0.67% |
| SELL | retest2 | 2026-02-26 10:45:00 | 726.80 | 2026-02-27 12:15:00 | 729.85 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest2 | 2026-04-13 10:15:00 | 726.50 | 2026-04-20 15:15:00 | 740.00 | STOP_HIT | 1.00 | 1.86% |
| BUY | retest2 | 2026-04-13 13:30:00 | 725.95 | 2026-04-20 15:15:00 | 740.00 | STOP_HIT | 1.00 | 1.94% |
| BUY | retest2 | 2026-04-15 09:15:00 | 733.50 | 2026-04-20 15:15:00 | 740.00 | STOP_HIT | 1.00 | 0.89% |
| BUY | retest2 | 2026-04-30 12:15:00 | 814.80 | 2026-05-08 13:15:00 | 896.28 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-05-05 11:00:00 | 813.70 | 2026-05-08 13:15:00 | 895.07 | TARGET_HIT | 1.00 | 10.00% |
