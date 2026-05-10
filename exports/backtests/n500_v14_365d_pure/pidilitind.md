# Pidilite Industries Ltd. (PIDILITIND)

## Backtest Summary

- **Window:** 2024-04-05 09:15:00 → 2026-05-08 15:15:00 (3605 bars)
- **Last close:** 1472.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 5 |
| ALERT1 | 5 |
| ALERT2 | 5 |
| ALERT2_SKIP | 1 |
| ALERT3 | 35 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 27 |
| PARTIAL | 2 |
| TARGET_HIT | 0 |
| STOP_HIT | 27 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 29 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 4 / 25
- **Target hits / Stop hits / Partials:** 0 / 27 / 2
- **Avg / median % per leg:** -0.76% / -0.85%
- **Sum % (uncompounded):** -21.98%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 8 | 0 | 0.0% | 0 | 8 | 0 | -2.09% | -16.7% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 8 | 0 | 0.0% | 0 | 8 | 0 | -2.09% | -16.7% |
| SELL (all) | 21 | 4 | 19.0% | 0 | 19 | 2 | -0.25% | -5.3% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 21 | 4 | 19.0% | 0 | 19 | 2 | -0.25% | -5.3% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 29 | 4 | 13.8% | 0 | 27 | 2 | -0.76% | -22.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-07-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-24 13:15:00 | 1452.20 | 1500.28 | 1500.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-25 09:15:00 | 1442.70 | 1498.79 | 1499.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-04 11:15:00 | 1482.95 | 1478.34 | 1488.20 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-04 11:45:00 | 1483.50 | 1478.34 | 1488.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 09:15:00 | 1482.05 | 1478.31 | 1487.94 | EMA400 retest candle locked (from downside) |

### Cycle 2 — BUY (started 2025-08-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-13 09:15:00 | 1540.65 | 1495.59 | 1495.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-25 10:15:00 | 1564.35 | 1514.12 | 1505.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-29 09:15:00 | 1513.90 | 1520.05 | 1509.79 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-29 10:00:00 | 1513.90 | 1520.05 | 1509.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 13:15:00 | 1526.70 | 1536.07 | 1524.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-18 13:45:00 | 1524.05 | 1536.07 | 1524.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 10:15:00 | 1524.95 | 1535.78 | 1524.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-19 11:00:00 | 1524.95 | 1535.78 | 1524.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 11:15:00 | 1524.10 | 1535.67 | 1524.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-19 11:45:00 | 1522.30 | 1535.67 | 1524.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 12:15:00 | 1522.40 | 1535.54 | 1524.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-19 13:00:00 | 1522.40 | 1535.54 | 1524.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 13:15:00 | 1521.55 | 1535.40 | 1524.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-19 14:00:00 | 1521.55 | 1535.40 | 1524.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 15:15:00 | 1529.95 | 1535.24 | 1524.42 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-22 10:30:00 | 1531.20 | 1535.14 | 1524.47 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-22 14:15:00 | 1519.35 | 1534.82 | 1524.53 | SL hit (close<static) qty=1.00 sl=1522.75 alert=retest2 |

### Cycle 3 — SELL (started 2025-10-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-01 09:15:00 | 1451.70 | 1516.54 | 1516.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-31 13:15:00 | 1445.00 | 1505.99 | 1509.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-13 10:15:00 | 1489.30 | 1487.56 | 1498.03 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-13 10:45:00 | 1491.30 | 1487.56 | 1498.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 13:15:00 | 1491.70 | 1485.53 | 1495.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-18 13:30:00 | 1495.70 | 1485.53 | 1495.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 09:15:00 | 1492.10 | 1485.72 | 1495.70 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-19 10:30:00 | 1484.50 | 1485.68 | 1495.63 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-20 11:30:00 | 1488.40 | 1485.49 | 1495.14 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-21 09:15:00 | 1475.00 | 1485.73 | 1495.07 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-26 13:15:00 | 1487.20 | 1482.47 | 1492.24 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 09:15:00 | 1484.30 | 1474.66 | 1483.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-16 09:45:00 | 1486.40 | 1474.66 | 1483.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 10:15:00 | 1486.10 | 1474.77 | 1483.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-16 11:00:00 | 1486.10 | 1474.77 | 1483.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 11:15:00 | 1482.50 | 1474.85 | 1483.77 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-16 12:30:00 | 1477.70 | 1474.85 | 1483.73 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-01 09:15:00 | 1476.80 | 1465.25 | 1475.28 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-02 13:30:00 | 1478.40 | 1466.41 | 1475.29 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-02 14:30:00 | 1477.60 | 1466.53 | 1475.31 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 15:15:00 | 1477.40 | 1466.64 | 1475.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-05 09:15:00 | 1475.90 | 1466.64 | 1475.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 10:15:00 | 1497.50 | 1467.07 | 1475.45 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-01-05 10:15:00 | 1497.50 | 1467.07 | 1475.45 | SL hit (close>static) qty=1.00 sl=1497.20 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-05 10:15:00 | 1497.50 | 1467.07 | 1475.45 | SL hit (close>static) qty=1.00 sl=1497.20 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-05 10:15:00 | 1497.50 | 1467.07 | 1475.45 | SL hit (close>static) qty=1.00 sl=1497.20 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-05 10:15:00 | 1497.50 | 1467.07 | 1475.45 | SL hit (close>static) qty=1.00 sl=1497.20 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-05 10:15:00 | 1497.50 | 1467.07 | 1475.45 | SL hit (close>static) qty=1.00 sl=1487.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-05 10:15:00 | 1497.50 | 1467.07 | 1475.45 | SL hit (close>static) qty=1.00 sl=1487.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-05 10:15:00 | 1497.50 | 1467.07 | 1475.45 | SL hit (close>static) qty=1.00 sl=1487.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-05 10:15:00 | 1497.50 | 1467.07 | 1475.45 | SL hit (close>static) qty=1.00 sl=1487.00 alert=retest2 |
| ALERT3_SIDEWAYS | 2026-01-05 11:00:00 | 1497.50 | 1467.07 | 1475.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 15:15:00 | 1483.50 | 1476.20 | 1479.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-12 09:30:00 | 1487.20 | 1476.38 | 1479.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 10:15:00 | 1491.70 | 1476.54 | 1479.29 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 14:15:00 | 1486.40 | 1478.34 | 1480.08 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-13 14:15:00 | 1498.60 | 1478.54 | 1480.17 | SL hit (close>static) qty=1.00 sl=1498.00 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-14 12:45:00 | 1487.70 | 1479.37 | 1480.55 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-16 11:00:00 | 1484.10 | 1480.03 | 1480.85 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-21 10:15:00 | 1413.32 | 1476.50 | 1478.97 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-02 09:15:00 | 1409.89 | 1459.05 | 1468.57 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-04 09:15:00 | 1469.50 | 1454.38 | 1465.49 | SL hit (close>ema200) qty=0.50 sl=1454.38 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-04 09:15:00 | 1469.50 | 1454.38 | 1465.49 | SL hit (close>ema200) qty=0.50 sl=1454.38 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-06 14:30:00 | 1488.60 | 1456.89 | 1465.77 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-12 14:15:00 | 1498.20 | 1463.31 | 1468.03 | SL hit (close>static) qty=1.00 sl=1498.00 alert=retest2 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 10:15:00 | 1479.30 | 1464.94 | 1468.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-16 11:15:00 | 1484.90 | 1464.94 | 1468.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 10:15:00 | 1471.80 | 1469.60 | 1470.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-19 10:30:00 | 1473.50 | 1469.60 | 1470.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 11:15:00 | 1464.60 | 1469.55 | 1470.70 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-19 14:15:00 | 1461.40 | 1469.45 | 1470.64 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-20 09:15:00 | 1462.80 | 1469.39 | 1470.59 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-20 09:45:00 | 1462.00 | 1469.31 | 1470.55 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-20 15:15:00 | 1463.00 | 1469.16 | 1470.44 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 09:15:00 | 1475.20 | 1469.16 | 1470.43 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-02-23 09:15:00 | 1475.20 | 1469.16 | 1470.43 | SL hit (close>static) qty=1.00 sl=1474.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-23 09:15:00 | 1475.20 | 1469.16 | 1470.43 | SL hit (close>static) qty=1.00 sl=1474.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-23 09:15:00 | 1475.20 | 1469.16 | 1470.43 | SL hit (close>static) qty=1.00 sl=1474.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-23 09:15:00 | 1475.20 | 1469.16 | 1470.43 | SL hit (close>static) qty=1.00 sl=1474.00 alert=retest2 |
| ALERT3_SIDEWAYS | 2026-02-23 09:30:00 | 1477.10 | 1469.16 | 1470.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 10:15:00 | 1473.70 | 1469.21 | 1470.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-23 11:15:00 | 1475.70 | 1469.21 | 1470.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 11:15:00 | 1478.70 | 1469.30 | 1470.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-23 12:00:00 | 1478.70 | 1469.30 | 1470.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 10:15:00 | 1480.60 | 1469.89 | 1470.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-24 11:00:00 | 1480.60 | 1469.89 | 1470.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 4 — BUY (started 2026-02-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-25 14:15:00 | 1497.70 | 1471.73 | 1471.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-26 10:15:00 | 1508.30 | 1472.60 | 1472.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-02 09:15:00 | 1464.50 | 1476.06 | 1473.91 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-02 09:15:00 | 1464.50 | 1476.06 | 1473.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 09:15:00 | 1464.50 | 1476.06 | 1473.91 | EMA400 retest candle locked (from upside) |

### Cycle 5 — SELL (started 2026-03-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-05 10:15:00 | 1426.50 | 1471.86 | 1471.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-09 09:15:00 | 1374.20 | 1467.48 | 1469.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 10:15:00 | 1371.00 | 1364.17 | 1403.23 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-08 10:45:00 | 1372.00 | 1364.17 | 1403.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-17 11:15:00 | 1390.00 | 1356.65 | 1391.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-17 12:00:00 | 1390.00 | 1356.65 | 1391.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-17 12:15:00 | 1391.10 | 1356.99 | 1391.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-17 13:00:00 | 1391.10 | 1356.99 | 1391.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-17 13:15:00 | 1392.60 | 1357.35 | 1391.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-17 14:00:00 | 1392.60 | 1357.35 | 1391.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-17 14:15:00 | 1394.30 | 1357.72 | 1391.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-17 14:30:00 | 1394.90 | 1357.72 | 1391.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-17 15:15:00 | 1398.00 | 1358.12 | 1391.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-20 09:15:00 | 1391.00 | 1358.12 | 1391.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-20 10:15:00 | 1396.70 | 1358.85 | 1391.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-20 10:30:00 | 1397.30 | 1358.85 | 1391.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-20 12:15:00 | 1391.80 | 1359.50 | 1391.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-20 13:15:00 | 1396.20 | 1359.50 | 1391.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-20 13:15:00 | 1393.00 | 1359.84 | 1391.66 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-20 15:15:00 | 1389.00 | 1360.16 | 1391.67 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-21 09:15:00 | 1409.80 | 1360.94 | 1391.74 | SL hit (close>static) qty=1.00 sl=1396.20 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-24 12:15:00 | 1390.60 | 1370.07 | 1393.23 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-24 15:15:00 | 1398.00 | 1371.02 | 1393.25 | SL hit (close>static) qty=1.00 sl=1396.20 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-29 14:45:00 | 1390.00 | 1375.75 | 1393.63 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-05-06 10:15:00 | 1409.00 | 1374.74 | 1391.06 | SL hit (close>static) qty=1.00 sl=1396.20 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-06-13 10:15:00 | 1496.35 | 2025-07-14 09:15:00 | 1490.00 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest2 | 2025-06-13 13:30:00 | 1500.85 | 2025-07-14 09:15:00 | 1490.00 | STOP_HIT | 1.00 | -0.72% |
| BUY | retest2 | 2025-06-24 09:15:00 | 1498.30 | 2025-07-14 09:15:00 | 1490.00 | STOP_HIT | 1.00 | -0.55% |
| BUY | retest2 | 2025-06-25 14:00:00 | 1501.75 | 2025-07-24 13:15:00 | 1452.20 | STOP_HIT | 1.00 | -3.30% |
| BUY | retest2 | 2025-06-26 09:30:00 | 1509.10 | 2025-07-24 13:15:00 | 1452.20 | STOP_HIT | 1.00 | -3.77% |
| BUY | retest2 | 2025-06-26 13:00:00 | 1506.25 | 2025-07-24 13:15:00 | 1452.20 | STOP_HIT | 1.00 | -3.59% |
| BUY | retest2 | 2025-07-11 12:30:00 | 1506.00 | 2025-07-24 13:15:00 | 1452.20 | STOP_HIT | 1.00 | -3.57% |
| BUY | retest2 | 2025-09-22 10:30:00 | 1531.20 | 2025-09-22 14:15:00 | 1519.35 | STOP_HIT | 1.00 | -0.77% |
| SELL | retest2 | 2025-11-19 10:30:00 | 1484.50 | 2026-01-05 10:15:00 | 1497.50 | STOP_HIT | 1.00 | -0.88% |
| SELL | retest2 | 2025-11-20 11:30:00 | 1488.40 | 2026-01-05 10:15:00 | 1497.50 | STOP_HIT | 1.00 | -0.61% |
| SELL | retest2 | 2025-11-21 09:15:00 | 1475.00 | 2026-01-05 10:15:00 | 1497.50 | STOP_HIT | 1.00 | -1.53% |
| SELL | retest2 | 2025-11-26 13:15:00 | 1487.20 | 2026-01-05 10:15:00 | 1497.50 | STOP_HIT | 1.00 | -0.69% |
| SELL | retest2 | 2025-12-16 12:30:00 | 1477.70 | 2026-01-05 10:15:00 | 1497.50 | STOP_HIT | 1.00 | -1.34% |
| SELL | retest2 | 2026-01-01 09:15:00 | 1476.80 | 2026-01-05 10:15:00 | 1497.50 | STOP_HIT | 1.00 | -1.40% |
| SELL | retest2 | 2026-01-02 13:30:00 | 1478.40 | 2026-01-05 10:15:00 | 1497.50 | STOP_HIT | 1.00 | -1.29% |
| SELL | retest2 | 2026-01-02 14:30:00 | 1477.60 | 2026-01-05 10:15:00 | 1497.50 | STOP_HIT | 1.00 | -1.35% |
| SELL | retest2 | 2026-01-13 14:15:00 | 1486.40 | 2026-01-13 14:15:00 | 1498.60 | STOP_HIT | 1.00 | -0.82% |
| SELL | retest2 | 2026-01-14 12:45:00 | 1487.70 | 2026-01-21 10:15:00 | 1413.32 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-16 11:00:00 | 1484.10 | 2026-02-02 09:15:00 | 1409.89 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-14 12:45:00 | 1487.70 | 2026-02-04 09:15:00 | 1469.50 | STOP_HIT | 0.50 | 1.22% |
| SELL | retest2 | 2026-01-16 11:00:00 | 1484.10 | 2026-02-04 09:15:00 | 1469.50 | STOP_HIT | 0.50 | 0.98% |
| SELL | retest2 | 2026-02-06 14:30:00 | 1488.60 | 2026-02-12 14:15:00 | 1498.20 | STOP_HIT | 1.00 | -0.64% |
| SELL | retest2 | 2026-02-19 14:15:00 | 1461.40 | 2026-02-23 09:15:00 | 1475.20 | STOP_HIT | 1.00 | -0.94% |
| SELL | retest2 | 2026-02-20 09:15:00 | 1462.80 | 2026-02-23 09:15:00 | 1475.20 | STOP_HIT | 1.00 | -0.85% |
| SELL | retest2 | 2026-02-20 09:45:00 | 1462.00 | 2026-02-23 09:15:00 | 1475.20 | STOP_HIT | 1.00 | -0.90% |
| SELL | retest2 | 2026-02-20 15:15:00 | 1463.00 | 2026-02-23 09:15:00 | 1475.20 | STOP_HIT | 1.00 | -0.83% |
| SELL | retest2 | 2026-04-20 15:15:00 | 1389.00 | 2026-04-21 09:15:00 | 1409.80 | STOP_HIT | 1.00 | -1.50% |
| SELL | retest2 | 2026-04-24 12:15:00 | 1390.60 | 2026-04-24 15:15:00 | 1398.00 | STOP_HIT | 1.00 | -0.53% |
| SELL | retest2 | 2026-04-29 14:45:00 | 1390.00 | 2026-05-06 10:15:00 | 1409.00 | STOP_HIT | 1.00 | -1.37% |
