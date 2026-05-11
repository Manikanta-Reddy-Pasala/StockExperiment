# Astral Ltd. (ASTRAL)

## Backtest Summary

- **Window:** 2024-04-08 09:15:00 → 2026-05-08 15:15:00 (3164 bars)
- **Last close:** 1567.40
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 9 |
| ALERT1 | 7 |
| ALERT2 | 6 |
| ALERT2_SKIP | 3 |
| ALERT3 | 44 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 22 |
| PARTIAL | 0 |
| TARGET_HIT | 0 |
| STOP_HIT | 23 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 22 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 0 / 22
- **Target hits / Stop hits / Partials:** 0 / 22 / 0
- **Avg / median % per leg:** -1.63% / -1.50%
- **Sum % (uncompounded):** -35.83%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 4 | 0 | 0.0% | 0 | 4 | 0 | -1.30% | -5.2% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 4 | 0 | 0.0% | 0 | 4 | 0 | -1.30% | -5.2% |
| SELL (all) | 18 | 0 | 0.0% | 0 | 18 | 0 | -1.70% | -30.6% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 18 | 0 | 0.0% | 0 | 18 | 0 | -1.70% | -30.6% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 22 | 0 | 0.0% | 0 | 22 | 0 | -1.63% | -35.8% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-28 10:15:00 | 1511.80 | 1377.05 | 1376.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-28 14:15:00 | 1528.50 | 1382.68 | 1379.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-02 09:15:00 | 1486.30 | 1493.86 | 1459.64 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-02 10:00:00 | 1486.30 | 1493.86 | 1459.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 11:15:00 | 1469.40 | 1490.45 | 1466.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-14 11:45:00 | 1463.80 | 1490.45 | 1466.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 14:15:00 | 1478.30 | 1493.85 | 1473.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-22 15:00:00 | 1478.30 | 1493.85 | 1473.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 15:15:00 | 1473.50 | 1493.65 | 1473.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-23 09:15:00 | 1463.10 | 1493.65 | 1473.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 09:15:00 | 1459.70 | 1493.31 | 1473.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-23 10:00:00 | 1459.70 | 1493.31 | 1473.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 10:15:00 | 1455.00 | 1492.93 | 1472.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-23 10:45:00 | 1456.60 | 1492.93 | 1472.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 09:15:00 | 1467.70 | 1491.26 | 1472.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-24 09:30:00 | 1469.70 | 1491.26 | 1472.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 10:15:00 | 1461.50 | 1490.96 | 1472.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-24 11:00:00 | 1461.50 | 1490.96 | 1472.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 14:15:00 | 1471.70 | 1490.14 | 1472.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-24 14:30:00 | 1474.10 | 1490.14 | 1472.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 15:15:00 | 1471.50 | 1489.95 | 1472.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-25 09:15:00 | 1472.10 | 1489.95 | 1472.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 09:15:00 | 1459.70 | 1489.65 | 1472.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-25 10:00:00 | 1459.70 | 1489.65 | 1472.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 10:15:00 | 1456.00 | 1489.32 | 1472.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-25 10:45:00 | 1458.00 | 1489.32 | 1472.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 2 — SELL (started 2025-08-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-06 10:15:00 | 1419.60 | 1459.50 | 1459.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-07 10:15:00 | 1403.90 | 1456.51 | 1458.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-21 10:15:00 | 1438.60 | 1402.69 | 1426.59 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-21 10:15:00 | 1438.60 | 1402.69 | 1426.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 10:15:00 | 1438.60 | 1402.69 | 1426.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-21 11:00:00 | 1438.60 | 1402.69 | 1426.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 11:15:00 | 1435.30 | 1403.01 | 1426.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-21 11:30:00 | 1436.10 | 1403.01 | 1426.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 13:15:00 | 1427.20 | 1403.47 | 1426.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-21 13:45:00 | 1426.20 | 1403.47 | 1426.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 14:15:00 | 1420.90 | 1403.65 | 1426.60 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-22 09:15:00 | 1417.70 | 1403.81 | 1426.57 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-22 12:15:00 | 1432.10 | 1404.79 | 1426.61 | SL hit (close>static) qty=1.00 sl=1431.80 alert=retest2 |

### Cycle 3 — BUY (started 2025-09-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-19 12:15:00 | 1451.50 | 1431.35 | 1431.33 | EMA200 above EMA400 |

### Cycle 4 — SELL (started 2025-09-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-25 15:15:00 | 1404.00 | 1431.45 | 1431.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-26 09:15:00 | 1400.50 | 1431.15 | 1431.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-07 13:15:00 | 1417.50 | 1411.74 | 1420.40 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-07 14:15:00 | 1417.80 | 1411.80 | 1420.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 14:15:00 | 1417.80 | 1411.80 | 1420.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-08 09:15:00 | 1415.90 | 1411.89 | 1420.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 09:15:00 | 1411.00 | 1411.88 | 1420.34 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-08 10:45:00 | 1405.80 | 1411.81 | 1420.26 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-08 12:00:00 | 1404.80 | 1411.74 | 1420.19 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-08 14:00:00 | 1402.90 | 1411.60 | 1420.03 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-09 15:15:00 | 1398.00 | 1410.85 | 1419.32 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 11:15:00 | 1422.40 | 1410.91 | 1419.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-10 11:45:00 | 1422.30 | 1410.91 | 1419.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 12:15:00 | 1431.00 | 1411.11 | 1419.24 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-10-10 12:15:00 | 1431.00 | 1411.11 | 1419.24 | SL hit (close>static) qty=1.00 sl=1427.00 alert=retest2 |

### Cycle 5 — BUY (started 2025-10-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-28 10:15:00 | 1453.00 | 1424.47 | 1424.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-29 09:15:00 | 1457.30 | 1425.97 | 1425.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-17 09:15:00 | 1468.50 | 1482.73 | 1458.44 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-17 10:00:00 | 1468.50 | 1482.73 | 1458.44 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 10:15:00 | 1465.20 | 1482.56 | 1458.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-17 10:30:00 | 1460.90 | 1482.56 | 1458.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 11:15:00 | 1466.40 | 1482.40 | 1458.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-17 11:30:00 | 1457.30 | 1482.40 | 1458.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 09:15:00 | 1443.00 | 1481.40 | 1458.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-18 10:00:00 | 1443.00 | 1481.40 | 1458.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 10:15:00 | 1450.00 | 1481.08 | 1458.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-18 10:30:00 | 1438.70 | 1481.08 | 1458.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 10:15:00 | 1453.80 | 1476.74 | 1457.83 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-20 11:30:00 | 1458.10 | 1476.59 | 1457.84 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-21 10:15:00 | 1450.00 | 1475.59 | 1457.89 | SL hit (close<static) qty=1.00 sl=1450.30 alert=retest2 |

### Cycle 6 — SELL (started 2025-12-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-11 14:15:00 | 1405.70 | 1449.37 | 1449.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-24 13:15:00 | 1395.90 | 1436.24 | 1442.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-01 13:15:00 | 1426.70 | 1420.72 | 1432.67 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-01 13:45:00 | 1426.40 | 1420.72 | 1432.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 14:15:00 | 1435.30 | 1420.86 | 1432.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-01 15:00:00 | 1435.30 | 1420.86 | 1432.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 15:15:00 | 1435.00 | 1421.00 | 1432.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-02 09:15:00 | 1444.40 | 1421.00 | 1432.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 15:15:00 | 1440.30 | 1439.64 | 1440.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-12 09:15:00 | 1437.00 | 1439.64 | 1440.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 09:15:00 | 1441.00 | 1439.65 | 1440.80 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-12 10:30:00 | 1434.40 | 1439.54 | 1440.73 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-12 12:45:00 | 1436.30 | 1439.36 | 1440.64 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-12 14:15:00 | 1432.40 | 1439.34 | 1440.62 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 09:30:00 | 1428.40 | 1439.31 | 1440.58 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 10:15:00 | 1430.40 | 1439.23 | 1440.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-13 10:30:00 | 1434.00 | 1439.23 | 1440.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 11:15:00 | 1433.30 | 1439.17 | 1440.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-13 11:30:00 | 1440.70 | 1439.17 | 1440.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 12:15:00 | 1431.50 | 1439.09 | 1440.45 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 13:45:00 | 1428.10 | 1438.98 | 1440.39 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-14 09:15:00 | 1447.30 | 1439.03 | 1440.39 | SL hit (close>static) qty=1.00 sl=1442.70 alert=retest2 |

### Cycle 7 — BUY (started 2026-01-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-16 12:15:00 | 1472.60 | 1441.94 | 1441.82 | EMA200 above EMA400 |

### Cycle 8 — SELL (started 2026-01-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-20 11:15:00 | 1416.50 | 1441.52 | 1441.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-20 12:15:00 | 1404.10 | 1441.15 | 1441.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-28 09:15:00 | 1428.80 | 1426.65 | 1433.50 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-28 09:15:00 | 1428.80 | 1426.65 | 1433.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 09:15:00 | 1428.80 | 1426.65 | 1433.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-28 10:00:00 | 1428.80 | 1426.65 | 1433.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 10:15:00 | 1445.50 | 1426.84 | 1433.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-28 11:00:00 | 1445.50 | 1426.84 | 1433.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 11:15:00 | 1445.00 | 1427.02 | 1433.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-28 12:15:00 | 1443.40 | 1427.02 | 1433.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 9 — BUY (started 2026-05-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-06 14:15:00 | 1574.20 | 1440.03 | 1439.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-06 15:15:00 | 1588.70 | 1441.51 | 1440.35 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-08-22 09:15:00 | 1417.70 | 2025-08-22 12:15:00 | 1432.10 | STOP_HIT | 1.00 | -1.02% |
| SELL | retest2 | 2025-08-25 09:15:00 | 1417.30 | 2025-09-05 10:15:00 | 1435.00 | STOP_HIT | 1.00 | -1.25% |
| SELL | retest2 | 2025-08-25 14:15:00 | 1416.40 | 2025-09-05 10:15:00 | 1435.00 | STOP_HIT | 1.00 | -1.31% |
| SELL | retest2 | 2025-09-02 11:00:00 | 1418.60 | 2025-09-05 10:15:00 | 1435.00 | STOP_HIT | 1.00 | -1.16% |
| SELL | retest2 | 2025-09-04 09:15:00 | 1413.80 | 2025-09-05 10:15:00 | 1435.00 | STOP_HIT | 1.00 | -1.50% |
| SELL | retest2 | 2025-09-04 10:00:00 | 1411.00 | 2025-09-05 10:15:00 | 1435.00 | STOP_HIT | 1.00 | -1.70% |
| SELL | retest2 | 2025-10-08 10:45:00 | 1405.80 | 2025-10-10 12:15:00 | 1431.00 | STOP_HIT | 1.00 | -1.79% |
| SELL | retest2 | 2025-10-08 12:00:00 | 1404.80 | 2025-10-10 12:15:00 | 1431.00 | STOP_HIT | 1.00 | -1.87% |
| SELL | retest2 | 2025-10-08 14:00:00 | 1402.90 | 2025-10-10 12:15:00 | 1431.00 | STOP_HIT | 1.00 | -2.00% |
| SELL | retest2 | 2025-10-09 15:15:00 | 1398.00 | 2025-10-10 12:15:00 | 1431.00 | STOP_HIT | 1.00 | -2.36% |
| SELL | retest2 | 2025-10-14 09:30:00 | 1419.10 | 2025-10-15 09:15:00 | 1436.00 | STOP_HIT | 1.00 | -1.19% |
| SELL | retest2 | 2025-10-14 15:00:00 | 1417.80 | 2025-10-15 09:15:00 | 1436.00 | STOP_HIT | 1.00 | -1.28% |
| SELL | retest2 | 2025-10-27 13:30:00 | 1418.10 | 2025-10-27 14:15:00 | 1435.70 | STOP_HIT | 1.00 | -1.24% |
| BUY | retest2 | 2025-11-20 11:30:00 | 1458.10 | 2025-11-21 10:15:00 | 1450.00 | STOP_HIT | 1.00 | -0.56% |
| BUY | retest2 | 2025-11-24 15:00:00 | 1474.30 | 2025-11-28 09:15:00 | 1446.30 | STOP_HIT | 1.00 | -1.90% |
| BUY | retest2 | 2025-11-25 11:00:00 | 1460.90 | 2025-11-28 09:15:00 | 1446.30 | STOP_HIT | 1.00 | -1.00% |
| BUY | retest2 | 2025-12-05 13:30:00 | 1458.40 | 2025-12-08 09:15:00 | 1433.00 | STOP_HIT | 1.00 | -1.74% |
| SELL | retest2 | 2026-01-12 10:30:00 | 1434.40 | 2026-01-14 09:15:00 | 1447.30 | STOP_HIT | 1.00 | -0.90% |
| SELL | retest2 | 2026-01-12 12:45:00 | 1436.30 | 2026-01-14 10:15:00 | 1467.30 | STOP_HIT | 1.00 | -2.16% |
| SELL | retest2 | 2026-01-12 14:15:00 | 1432.40 | 2026-01-14 10:15:00 | 1467.30 | STOP_HIT | 1.00 | -2.44% |
| SELL | retest2 | 2026-01-13 09:30:00 | 1428.40 | 2026-01-14 10:15:00 | 1467.30 | STOP_HIT | 1.00 | -2.72% |
| SELL | retest2 | 2026-01-13 13:45:00 | 1428.10 | 2026-01-14 10:15:00 | 1467.30 | STOP_HIT | 1.00 | -2.74% |
