# Reliance Industries Ltd. (RELIANCE)

## Backtest Summary

- **Window:** 2025-03-13 09:15:00 → 2026-05-08 15:15:00 (1976 bars)
- **Last close:** 1436.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 76 |
| ALERT1 | 49 |
| ALERT2 | 50 |
| ALERT2_SKIP | 29 |
| ALERT3 | 136 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 2 |
| ENTRY2 | 58 |
| PARTIAL | 5 |
| TARGET_HIT | 0 |
| STOP_HIT | 60 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 65 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 33 / 32
- **Target hits / Stop hits / Partials:** 0 / 60 / 5
- **Avg / median % per leg:** 1.00% / 0.14%
- **Sum % (uncompounded):** 64.74%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 30 | 18 | 60.0% | 0 | 30 | 0 | 0.83% | 24.8% |
| BUY @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -0.33% | -0.3% |
| BUY @ 3rd Alert (retest2) | 29 | 18 | 62.1% | 0 | 29 | 0 | 0.87% | 25.1% |
| SELL (all) | 35 | 15 | 42.9% | 0 | 30 | 5 | 1.14% | 40.0% |
| SELL @ 2nd Alert (retest1) | 2 | 2 | 100.0% | 0 | 1 | 1 | 2.80% | 5.6% |
| SELL @ 3rd Alert (retest2) | 33 | 13 | 39.4% | 0 | 29 | 4 | 1.04% | 34.4% |
| retest1 (combined) | 3 | 2 | 66.7% | 0 | 2 | 1 | 1.76% | 5.3% |
| retest2 (combined) | 62 | 31 | 50.0% | 0 | 58 | 4 | 0.96% | 59.5% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 11:15:00 | 1422.10 | 1404.15 | 1403.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 13:15:00 | 1430.70 | 1412.45 | 1407.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-13 12:15:00 | 1422.60 | 1424.45 | 1417.24 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-13 13:00:00 | 1422.60 | 1424.45 | 1417.24 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-13 13:15:00 | 1420.00 | 1423.56 | 1417.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-13 13:45:00 | 1416.60 | 1423.56 | 1417.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-13 14:15:00 | 1413.80 | 1421.61 | 1417.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-13 15:00:00 | 1413.80 | 1421.61 | 1417.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-13 15:15:00 | 1416.20 | 1420.53 | 1417.07 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-14 09:15:00 | 1426.20 | 1420.53 | 1417.07 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-15 10:30:00 | 1429.90 | 1421.93 | 1420.25 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-20 11:15:00 | 1435.70 | 1441.48 | 1441.63 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 2 — SELL (started 2025-05-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-20 11:15:00 | 1435.70 | 1441.48 | 1441.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-20 12:15:00 | 1432.40 | 1439.66 | 1440.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-21 10:15:00 | 1434.00 | 1432.43 | 1436.12 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-21 10:30:00 | 1434.10 | 1432.43 | 1436.12 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 11:15:00 | 1422.20 | 1430.39 | 1434.86 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-22 09:15:00 | 1415.00 | 1428.76 | 1432.53 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-23 14:15:00 | 1426.00 | 1423.44 | 1423.40 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 3 — BUY (started 2025-05-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-23 14:15:00 | 1426.00 | 1423.44 | 1423.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-26 09:15:00 | 1438.50 | 1426.86 | 1424.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-27 09:15:00 | 1418.30 | 1430.81 | 1429.01 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-27 09:15:00 | 1418.30 | 1430.81 | 1429.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 09:15:00 | 1418.30 | 1430.81 | 1429.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-27 10:00:00 | 1418.30 | 1430.81 | 1429.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 10:15:00 | 1424.80 | 1429.61 | 1428.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-27 10:30:00 | 1416.10 | 1429.61 | 1428.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 11:15:00 | 1430.30 | 1429.75 | 1428.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-27 12:15:00 | 1423.60 | 1429.75 | 1428.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 4 — SELL (started 2025-05-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-27 12:15:00 | 1421.10 | 1428.02 | 1428.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-27 13:15:00 | 1417.70 | 1425.95 | 1427.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-29 14:15:00 | 1419.00 | 1414.97 | 1417.70 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-29 14:15:00 | 1419.00 | 1414.97 | 1417.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 14:15:00 | 1419.00 | 1414.97 | 1417.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-29 15:00:00 | 1419.00 | 1414.97 | 1417.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 15:15:00 | 1418.40 | 1415.66 | 1417.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-30 09:15:00 | 1421.10 | 1415.66 | 1417.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 09:15:00 | 1422.00 | 1416.93 | 1418.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-30 09:45:00 | 1427.80 | 1416.93 | 1418.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 10:15:00 | 1417.50 | 1417.04 | 1418.09 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-30 11:30:00 | 1413.60 | 1416.17 | 1417.60 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-30 14:00:00 | 1413.40 | 1415.58 | 1417.08 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-02 09:15:00 | 1398.90 | 1417.02 | 1417.54 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-02 12:45:00 | 1411.80 | 1411.77 | 1414.34 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 13:15:00 | 1412.30 | 1411.87 | 1414.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-02 13:30:00 | 1415.60 | 1411.87 | 1414.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 14:15:00 | 1414.00 | 1412.30 | 1414.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-02 15:00:00 | 1414.00 | 1412.30 | 1414.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 15:15:00 | 1412.60 | 1412.36 | 1414.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-03 09:15:00 | 1411.90 | 1412.36 | 1414.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 09:15:00 | 1415.20 | 1412.93 | 1414.11 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-03 14:45:00 | 1408.40 | 1410.87 | 1412.75 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-04 10:00:00 | 1408.10 | 1409.44 | 1411.72 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-04 11:00:00 | 1408.00 | 1409.16 | 1411.38 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-04 13:15:00 | 1418.40 | 1412.46 | 1412.46 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 5 — BUY (started 2025-06-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-04 13:15:00 | 1418.40 | 1412.46 | 1412.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-04 14:15:00 | 1424.00 | 1414.77 | 1413.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-06 13:15:00 | 1442.10 | 1443.02 | 1435.25 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-06 14:00:00 | 1442.10 | 1443.02 | 1435.25 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 11:15:00 | 1442.90 | 1446.63 | 1443.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-10 12:00:00 | 1442.90 | 1446.63 | 1443.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 12:15:00 | 1440.60 | 1445.42 | 1443.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-10 13:00:00 | 1440.60 | 1445.42 | 1443.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 13:15:00 | 1440.60 | 1444.46 | 1442.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-10 13:30:00 | 1442.10 | 1444.46 | 1442.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 14:15:00 | 1438.10 | 1443.19 | 1442.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-10 15:00:00 | 1438.10 | 1443.19 | 1442.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 15:15:00 | 1437.30 | 1442.01 | 1441.98 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-11 09:15:00 | 1456.70 | 1442.01 | 1441.98 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-12 14:15:00 | 1442.40 | 1446.78 | 1447.25 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 6 — SELL (started 2025-06-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-12 14:15:00 | 1442.40 | 1446.78 | 1447.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-13 09:15:00 | 1417.30 | 1439.72 | 1443.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-16 10:15:00 | 1438.50 | 1432.53 | 1436.62 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-16 10:15:00 | 1438.50 | 1432.53 | 1436.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 10:15:00 | 1438.50 | 1432.53 | 1436.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 11:00:00 | 1438.50 | 1432.53 | 1436.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 11:15:00 | 1436.50 | 1433.33 | 1436.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 11:45:00 | 1435.30 | 1433.33 | 1436.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 12:15:00 | 1438.20 | 1434.30 | 1436.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 12:30:00 | 1439.70 | 1434.30 | 1436.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 13:15:00 | 1440.00 | 1435.44 | 1437.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 13:30:00 | 1441.20 | 1435.44 | 1437.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 14:15:00 | 1438.00 | 1435.95 | 1437.14 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-16 15:15:00 | 1436.90 | 1435.95 | 1437.14 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-18 09:45:00 | 1437.00 | 1432.92 | 1434.24 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-18 10:15:00 | 1436.00 | 1432.92 | 1434.24 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-19 13:00:00 | 1434.40 | 1432.67 | 1432.83 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-19 13:15:00 | 1437.30 | 1433.60 | 1433.23 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 7 — BUY (started 2025-06-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-19 13:15:00 | 1437.30 | 1433.60 | 1433.23 | EMA200 above EMA400 |

### Cycle 8 — SELL (started 2025-06-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-19 15:15:00 | 1428.00 | 1432.48 | 1432.79 | EMA200 below EMA400 |

### Cycle 9 — BUY (started 2025-06-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-20 09:15:00 | 1447.80 | 1435.54 | 1434.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-20 10:15:00 | 1454.70 | 1439.37 | 1436.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-23 09:15:00 | 1446.70 | 1451.75 | 1444.78 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-23 09:15:00 | 1446.70 | 1451.75 | 1444.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 09:15:00 | 1446.70 | 1451.75 | 1444.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-23 10:00:00 | 1446.70 | 1451.75 | 1444.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 13:15:00 | 1449.80 | 1459.50 | 1454.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-24 14:00:00 | 1449.80 | 1459.50 | 1454.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 14:15:00 | 1452.80 | 1458.16 | 1454.51 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-24 15:15:00 | 1456.80 | 1458.16 | 1454.51 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-09 15:15:00 | 1520.50 | 1535.50 | 1535.56 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 10 — SELL (started 2025-07-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-09 15:15:00 | 1520.50 | 1535.50 | 1535.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-10 09:15:00 | 1515.20 | 1531.44 | 1533.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-15 10:15:00 | 1493.40 | 1489.94 | 1497.72 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-15 11:00:00 | 1493.40 | 1489.94 | 1497.72 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 14:15:00 | 1485.40 | 1484.33 | 1488.98 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-17 09:30:00 | 1483.40 | 1483.74 | 1487.93 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-17 11:30:00 | 1483.60 | 1483.24 | 1486.96 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-17 12:45:00 | 1483.50 | 1483.03 | 1486.52 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-24 11:15:00 | 1409.23 | 1415.55 | 1424.46 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-24 11:15:00 | 1409.42 | 1415.55 | 1424.46 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-24 11:15:00 | 1409.33 | 1415.55 | 1424.46 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-28 09:15:00 | 1399.10 | 1397.03 | 1405.42 | SL hit (close>ema200) qty=0.50 sl=1397.03 alert=retest2 |

### Cycle 11 — BUY (started 2025-07-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-29 15:15:00 | 1416.60 | 1402.56 | 1401.26 | EMA200 above EMA400 |

### Cycle 12 — SELL (started 2025-07-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-31 10:15:00 | 1392.90 | 1402.33 | 1403.08 | EMA200 below EMA400 |

### Cycle 13 — BUY (started 2025-08-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-04 10:15:00 | 1413.30 | 1402.04 | 1400.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-04 13:15:00 | 1414.80 | 1407.89 | 1404.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-05 09:15:00 | 1397.20 | 1406.51 | 1404.48 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-05 09:15:00 | 1397.20 | 1406.51 | 1404.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 09:15:00 | 1397.20 | 1406.51 | 1404.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-05 10:00:00 | 1397.20 | 1406.51 | 1404.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 10:15:00 | 1397.50 | 1404.70 | 1403.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-05 10:45:00 | 1396.80 | 1404.70 | 1403.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 14 — SELL (started 2025-08-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-05 11:15:00 | 1396.40 | 1403.04 | 1403.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-05 12:15:00 | 1392.50 | 1400.94 | 1402.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-06 14:15:00 | 1393.00 | 1392.39 | 1395.72 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-06 14:15:00 | 1393.00 | 1392.39 | 1395.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 14:15:00 | 1393.00 | 1392.39 | 1395.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-06 14:45:00 | 1394.50 | 1392.39 | 1395.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 14:15:00 | 1388.10 | 1383.37 | 1388.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-07 14:45:00 | 1386.70 | 1383.37 | 1388.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 15:15:00 | 1391.00 | 1384.90 | 1388.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-08 09:15:00 | 1388.50 | 1384.90 | 1388.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 09:15:00 | 1383.30 | 1384.58 | 1388.35 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-08 10:30:00 | 1376.70 | 1382.46 | 1387.05 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-12 09:15:00 | 1397.20 | 1382.05 | 1380.96 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 15 — BUY (started 2025-08-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-12 09:15:00 | 1397.20 | 1382.05 | 1380.96 | EMA200 above EMA400 |

### Cycle 16 — SELL (started 2025-08-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-14 09:15:00 | 1375.00 | 1382.12 | 1382.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-14 14:15:00 | 1373.90 | 1377.35 | 1379.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-18 09:15:00 | 1387.40 | 1378.64 | 1379.79 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-18 09:15:00 | 1387.40 | 1378.64 | 1379.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 09:15:00 | 1387.40 | 1378.64 | 1379.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-18 09:45:00 | 1389.90 | 1378.64 | 1379.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 17 — BUY (started 2025-08-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-18 12:15:00 | 1385.30 | 1380.91 | 1380.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-19 09:15:00 | 1413.30 | 1388.11 | 1384.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-20 14:15:00 | 1413.00 | 1416.18 | 1407.22 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-20 15:00:00 | 1413.00 | 1416.18 | 1407.22 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 09:15:00 | 1420.50 | 1423.51 | 1417.39 | EMA400 retest candle locked (from upside) |

### Cycle 18 — SELL (started 2025-08-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-22 15:15:00 | 1410.00 | 1414.87 | 1415.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-25 09:15:00 | 1408.70 | 1413.63 | 1414.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-25 12:15:00 | 1416.10 | 1413.32 | 1414.09 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-25 12:15:00 | 1416.10 | 1413.32 | 1414.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 12:15:00 | 1416.10 | 1413.32 | 1414.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-25 13:00:00 | 1416.10 | 1413.32 | 1414.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 13:15:00 | 1415.50 | 1413.75 | 1414.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-25 13:45:00 | 1417.80 | 1413.75 | 1414.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 15:15:00 | 1413.70 | 1413.17 | 1413.84 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-26 09:15:00 | 1408.00 | 1413.17 | 1413.84 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-05 13:15:00 | 1374.40 | 1367.55 | 1367.51 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 19 — BUY (started 2025-09-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-05 13:15:00 | 1374.40 | 1367.55 | 1367.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-08 12:15:00 | 1383.90 | 1375.70 | 1372.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-09 09:15:00 | 1373.80 | 1377.00 | 1374.08 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-09 09:15:00 | 1373.80 | 1377.00 | 1374.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 09:15:00 | 1373.80 | 1377.00 | 1374.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-09 09:30:00 | 1376.00 | 1377.00 | 1374.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 10:15:00 | 1373.80 | 1376.36 | 1374.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-09 10:30:00 | 1371.60 | 1376.36 | 1374.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 11:15:00 | 1372.80 | 1375.65 | 1373.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-09 12:00:00 | 1372.80 | 1375.65 | 1373.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 12:15:00 | 1377.50 | 1376.02 | 1374.26 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-10 09:15:00 | 1381.40 | 1376.44 | 1374.92 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-11 09:15:00 | 1380.70 | 1378.74 | 1377.32 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-22 10:15:00 | 1404.70 | 1408.56 | 1408.65 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 20 — SELL (started 2025-09-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-22 10:15:00 | 1404.70 | 1408.56 | 1408.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-22 13:15:00 | 1403.50 | 1406.77 | 1407.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-23 13:15:00 | 1392.20 | 1391.09 | 1397.53 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-23 14:00:00 | 1392.20 | 1391.09 | 1397.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 11:15:00 | 1395.00 | 1390.39 | 1394.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-24 11:45:00 | 1393.80 | 1390.39 | 1394.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 12:15:00 | 1393.70 | 1391.05 | 1394.51 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-24 13:45:00 | 1389.70 | 1390.50 | 1393.94 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-06 14:15:00 | 1374.40 | 1369.02 | 1368.31 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 21 — BUY (started 2025-10-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-06 14:15:00 | 1374.40 | 1369.02 | 1368.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-07 09:15:00 | 1387.90 | 1373.74 | 1370.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-08 09:15:00 | 1381.00 | 1382.60 | 1377.82 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-08 10:15:00 | 1380.30 | 1382.60 | 1377.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 10:15:00 | 1371.00 | 1380.28 | 1377.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-08 11:00:00 | 1371.00 | 1380.28 | 1377.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 11:15:00 | 1369.30 | 1378.09 | 1376.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-08 12:15:00 | 1366.70 | 1378.09 | 1376.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 22 — SELL (started 2025-10-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-08 13:15:00 | 1369.00 | 1374.90 | 1375.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-08 14:15:00 | 1366.50 | 1373.22 | 1374.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-09 09:15:00 | 1374.40 | 1372.51 | 1373.84 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-09 09:15:00 | 1374.40 | 1372.51 | 1373.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 09:15:00 | 1374.40 | 1372.51 | 1373.84 | EMA400 retest candle locked (from downside) |

### Cycle 23 — BUY (started 2025-10-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-09 13:15:00 | 1377.00 | 1375.03 | 1374.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-10 09:15:00 | 1382.60 | 1377.17 | 1375.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-10 15:15:00 | 1380.70 | 1381.57 | 1379.11 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-13 09:15:00 | 1372.90 | 1381.57 | 1379.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 09:15:00 | 1371.50 | 1379.56 | 1378.42 | EMA400 retest candle locked (from upside) |

### Cycle 24 — SELL (started 2025-10-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-13 10:15:00 | 1370.00 | 1377.65 | 1377.65 | EMA200 below EMA400 |

### Cycle 25 — BUY (started 2025-10-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-14 10:15:00 | 1380.70 | 1376.78 | 1376.78 | EMA200 above EMA400 |

### Cycle 26 — SELL (started 2025-10-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-14 12:15:00 | 1373.60 | 1376.28 | 1376.56 | EMA200 below EMA400 |

### Cycle 27 — BUY (started 2025-10-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-15 12:15:00 | 1378.30 | 1376.74 | 1376.56 | EMA200 above EMA400 |

### Cycle 28 — SELL (started 2025-10-15 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-15 13:15:00 | 1374.70 | 1376.33 | 1376.39 | EMA200 below EMA400 |

### Cycle 29 — BUY (started 2025-10-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-16 10:15:00 | 1377.90 | 1376.35 | 1376.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-16 11:15:00 | 1383.40 | 1377.76 | 1376.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-23 13:15:00 | 1460.60 | 1463.66 | 1448.59 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-23 14:00:00 | 1460.60 | 1463.66 | 1448.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 14:15:00 | 1446.30 | 1460.18 | 1448.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-23 15:00:00 | 1446.30 | 1460.18 | 1448.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 15:15:00 | 1447.00 | 1457.55 | 1448.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-24 09:15:00 | 1445.40 | 1457.55 | 1448.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 09:15:00 | 1448.00 | 1455.64 | 1448.23 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-24 11:30:00 | 1454.60 | 1454.61 | 1448.98 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-24 15:00:00 | 1452.40 | 1453.85 | 1450.06 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-31 14:15:00 | 1486.50 | 1489.10 | 1489.35 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 30 — SELL (started 2025-10-31 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-31 14:15:00 | 1486.50 | 1489.10 | 1489.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-03 14:15:00 | 1484.60 | 1487.74 | 1488.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-04 09:15:00 | 1488.60 | 1487.84 | 1488.42 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-04 09:15:00 | 1488.60 | 1487.84 | 1488.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 09:15:00 | 1488.60 | 1487.84 | 1488.42 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-04 13:00:00 | 1480.50 | 1486.21 | 1487.54 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-04 14:00:00 | 1479.80 | 1484.93 | 1486.84 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-06 09:15:00 | 1493.20 | 1482.70 | 1484.99 | SL hit (close>static) qty=1.00 sl=1492.80 alert=retest2 |

### Cycle 31 — BUY (started 2025-11-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-06 11:15:00 | 1494.80 | 1487.41 | 1486.88 | EMA200 above EMA400 |

### Cycle 32 — SELL (started 2025-11-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-07 13:15:00 | 1480.40 | 1487.54 | 1488.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-07 14:15:00 | 1476.90 | 1485.41 | 1487.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-10 09:15:00 | 1492.50 | 1486.04 | 1487.03 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-10 09:15:00 | 1492.50 | 1486.04 | 1487.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 09:15:00 | 1492.50 | 1486.04 | 1487.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-10 09:45:00 | 1492.40 | 1486.04 | 1487.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 10:15:00 | 1492.60 | 1487.35 | 1487.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-10 11:00:00 | 1492.60 | 1487.35 | 1487.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 33 — BUY (started 2025-11-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-10 11:15:00 | 1494.90 | 1488.86 | 1488.20 | EMA200 above EMA400 |

### Cycle 34 — SELL (started 2025-11-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-11 10:15:00 | 1481.80 | 1487.58 | 1488.20 | EMA200 below EMA400 |

### Cycle 35 — BUY (started 2025-11-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-11 14:15:00 | 1494.00 | 1488.86 | 1488.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-12 09:15:00 | 1504.00 | 1492.68 | 1490.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-13 13:15:00 | 1509.90 | 1512.84 | 1506.51 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-13 14:00:00 | 1509.90 | 1512.84 | 1506.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 09:15:00 | 1515.00 | 1512.98 | 1508.12 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-14 10:45:00 | 1517.90 | 1513.55 | 1508.81 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-14 12:15:00 | 1517.50 | 1513.88 | 1509.39 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-14 13:15:00 | 1516.80 | 1514.28 | 1509.99 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-14 15:00:00 | 1520.10 | 1515.38 | 1511.23 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 11:15:00 | 1514.50 | 1516.27 | 1513.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-17 11:30:00 | 1513.50 | 1516.27 | 1513.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 12:15:00 | 1513.00 | 1515.61 | 1513.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-17 13:00:00 | 1513.00 | 1515.61 | 1513.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 13:15:00 | 1516.00 | 1515.69 | 1513.35 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-17 15:00:00 | 1518.00 | 1516.15 | 1513.77 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-18 11:45:00 | 1521.40 | 1516.44 | 1514.64 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-18 15:15:00 | 1520.00 | 1518.66 | 1516.29 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-19 09:45:00 | 1517.70 | 1519.18 | 1516.95 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 10:15:00 | 1514.50 | 1518.25 | 1516.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-19 11:00:00 | 1514.50 | 1518.25 | 1516.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 11:15:00 | 1517.20 | 1518.04 | 1516.77 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-19 12:15:00 | 1518.70 | 1518.04 | 1516.77 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-19 13:45:00 | 1520.00 | 1518.76 | 1517.34 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-02 10:15:00 | 1553.40 | 1564.42 | 1565.24 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 36 — SELL (started 2025-12-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-02 10:15:00 | 1553.40 | 1564.42 | 1565.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-02 12:15:00 | 1548.20 | 1559.30 | 1562.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-04 09:15:00 | 1550.80 | 1543.78 | 1549.27 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-04 09:15:00 | 1550.80 | 1543.78 | 1549.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 09:15:00 | 1550.80 | 1543.78 | 1549.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-04 10:00:00 | 1550.80 | 1543.78 | 1549.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 10:15:00 | 1559.90 | 1547.00 | 1550.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-04 11:00:00 | 1559.90 | 1547.00 | 1550.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 11:15:00 | 1549.20 | 1547.44 | 1550.14 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-04 12:15:00 | 1548.60 | 1547.44 | 1550.14 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-11 12:15:00 | 1546.40 | 1539.24 | 1538.46 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 37 — BUY (started 2025-12-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-11 12:15:00 | 1546.40 | 1539.24 | 1538.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-12 09:15:00 | 1551.30 | 1543.36 | 1540.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-15 09:15:00 | 1552.10 | 1552.75 | 1547.97 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-15 09:15:00 | 1552.10 | 1552.75 | 1547.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 09:15:00 | 1552.10 | 1552.75 | 1547.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-15 09:30:00 | 1548.00 | 1552.75 | 1547.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 09:15:00 | 1540.80 | 1551.78 | 1550.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 10:00:00 | 1540.80 | 1551.78 | 1550.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 10:15:00 | 1541.30 | 1549.68 | 1549.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 10:30:00 | 1540.70 | 1549.68 | 1549.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 38 — SELL (started 2025-12-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-16 11:15:00 | 1543.20 | 1548.39 | 1548.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-18 09:15:00 | 1539.30 | 1543.04 | 1544.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-18 11:15:00 | 1549.70 | 1544.35 | 1545.02 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-18 11:15:00 | 1549.70 | 1544.35 | 1545.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 11:15:00 | 1549.70 | 1544.35 | 1545.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-18 12:00:00 | 1549.70 | 1544.35 | 1545.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 12:15:00 | 1548.50 | 1545.18 | 1545.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-18 12:30:00 | 1551.00 | 1545.18 | 1545.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 14:15:00 | 1544.70 | 1544.94 | 1545.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-18 14:30:00 | 1543.10 | 1544.94 | 1545.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 39 — BUY (started 2025-12-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-19 09:15:00 | 1564.90 | 1548.70 | 1546.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-19 10:15:00 | 1572.00 | 1553.36 | 1549.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-23 14:15:00 | 1571.80 | 1572.94 | 1568.24 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-23 14:15:00 | 1571.80 | 1572.94 | 1568.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 14:15:00 | 1571.80 | 1572.94 | 1568.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-23 15:00:00 | 1571.80 | 1572.94 | 1568.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 09:15:00 | 1569.10 | 1571.54 | 1568.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-24 10:00:00 | 1569.10 | 1571.54 | 1568.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 10:15:00 | 1566.60 | 1570.56 | 1568.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-24 11:00:00 | 1566.60 | 1570.56 | 1568.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 11:15:00 | 1563.70 | 1569.18 | 1567.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-24 12:00:00 | 1563.70 | 1569.18 | 1567.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 40 — SELL (started 2025-12-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-24 12:15:00 | 1557.50 | 1566.85 | 1566.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-29 09:15:00 | 1551.40 | 1558.38 | 1561.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-30 09:15:00 | 1550.50 | 1549.98 | 1554.64 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-30 09:30:00 | 1551.20 | 1549.98 | 1554.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 09:15:00 | 1548.00 | 1544.93 | 1549.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 09:30:00 | 1546.90 | 1544.93 | 1549.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 10:15:00 | 1555.00 | 1546.94 | 1549.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 11:00:00 | 1555.00 | 1546.94 | 1549.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 11:15:00 | 1567.80 | 1551.11 | 1551.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 12:00:00 | 1567.80 | 1551.11 | 1551.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 41 — BUY (started 2025-12-31 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-31 12:15:00 | 1572.50 | 1555.39 | 1553.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-01 09:15:00 | 1582.20 | 1566.33 | 1559.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-01 15:15:00 | 1572.90 | 1573.43 | 1566.95 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-02 09:15:00 | 1585.00 | 1573.43 | 1566.95 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 13:15:00 | 1579.70 | 1588.49 | 1582.80 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-01-05 13:15:00 | 1579.70 | 1588.49 | 1582.80 | SL hit (close<ema400) qty=1.00 sl=1582.80 alert=retest1 |

### Cycle 42 — SELL (started 2026-01-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-06 09:15:00 | 1526.20 | 1572.79 | 1576.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-06 10:15:00 | 1510.50 | 1560.33 | 1570.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-09 14:15:00 | 1476.00 | 1474.68 | 1488.86 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-09 15:00:00 | 1476.00 | 1474.68 | 1488.86 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 13:15:00 | 1478.70 | 1470.54 | 1479.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-12 14:00:00 | 1478.70 | 1470.54 | 1479.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 14:15:00 | 1483.40 | 1473.11 | 1480.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-12 15:15:00 | 1484.00 | 1473.11 | 1480.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 15:15:00 | 1484.00 | 1475.29 | 1480.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-13 09:15:00 | 1468.60 | 1475.29 | 1480.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 12:15:00 | 1463.50 | 1461.18 | 1466.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-14 12:30:00 | 1464.50 | 1461.18 | 1466.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 09:15:00 | 1466.80 | 1461.13 | 1464.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-16 10:15:00 | 1470.30 | 1461.13 | 1464.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 10:15:00 | 1473.10 | 1463.52 | 1465.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-16 10:45:00 | 1476.00 | 1463.52 | 1465.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 12:15:00 | 1458.70 | 1463.71 | 1465.40 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-19 09:15:00 | 1421.60 | 1461.34 | 1463.74 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-28 15:15:00 | 1398.50 | 1394.30 | 1393.99 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 43 — BUY (started 2026-01-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-28 15:15:00 | 1398.50 | 1394.30 | 1393.99 | EMA200 above EMA400 |

### Cycle 44 — SELL (started 2026-01-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-29 09:15:00 | 1391.30 | 1393.70 | 1393.74 | EMA200 below EMA400 |

### Cycle 45 — BUY (started 2026-01-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-29 10:15:00 | 1395.60 | 1394.08 | 1393.91 | EMA200 above EMA400 |

### Cycle 46 — SELL (started 2026-01-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-29 11:15:00 | 1392.40 | 1393.75 | 1393.78 | EMA200 below EMA400 |

### Cycle 47 — BUY (started 2026-01-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-29 12:15:00 | 1394.80 | 1393.96 | 1393.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-29 13:15:00 | 1398.60 | 1394.89 | 1394.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-29 14:15:00 | 1391.10 | 1394.13 | 1394.01 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-29 14:15:00 | 1391.10 | 1394.13 | 1394.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 14:15:00 | 1391.10 | 1394.13 | 1394.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-29 15:00:00 | 1391.10 | 1394.13 | 1394.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 48 — SELL (started 2026-01-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-29 15:15:00 | 1388.80 | 1393.06 | 1393.53 | EMA200 below EMA400 |

### Cycle 49 — BUY (started 2026-01-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-30 15:15:00 | 1397.00 | 1393.05 | 1392.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-01 09:15:00 | 1404.80 | 1395.40 | 1393.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-01 12:15:00 | 1380.00 | 1394.86 | 1394.32 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-01 12:15:00 | 1380.00 | 1394.86 | 1394.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 12:15:00 | 1380.00 | 1394.86 | 1394.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 12:30:00 | 1378.40 | 1394.86 | 1394.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 50 — SELL (started 2026-02-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-01 13:15:00 | 1370.70 | 1390.03 | 1392.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-01 14:15:00 | 1349.80 | 1381.98 | 1388.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-02 11:15:00 | 1378.30 | 1371.60 | 1380.22 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-02 12:00:00 | 1378.30 | 1371.60 | 1380.22 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 12:15:00 | 1380.20 | 1373.32 | 1380.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-02 13:00:00 | 1380.20 | 1373.32 | 1380.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 13:15:00 | 1379.20 | 1374.50 | 1380.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-02 13:45:00 | 1379.80 | 1374.50 | 1380.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 14:15:00 | 1391.00 | 1377.80 | 1381.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-02 15:00:00 | 1391.00 | 1377.80 | 1381.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 15:15:00 | 1388.00 | 1379.84 | 1381.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-03 09:15:00 | 1445.20 | 1379.84 | 1381.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 51 — BUY (started 2026-02-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 09:15:00 | 1443.90 | 1392.65 | 1387.39 | EMA200 above EMA400 |

### Cycle 52 — SELL (started 2026-02-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-12 13:15:00 | 1450.00 | 1458.91 | 1459.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-12 14:15:00 | 1447.70 | 1456.67 | 1458.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-16 13:15:00 | 1426.00 | 1425.08 | 1433.77 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-16 13:45:00 | 1424.10 | 1425.08 | 1433.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 14:15:00 | 1438.20 | 1427.71 | 1434.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-16 14:30:00 | 1436.00 | 1427.71 | 1434.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 15:15:00 | 1435.80 | 1429.32 | 1434.32 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-17 09:15:00 | 1424.10 | 1429.32 | 1434.32 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-18 13:15:00 | 1433.00 | 1426.49 | 1428.25 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-18 14:15:00 | 1441.90 | 1430.87 | 1430.02 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 53 — BUY (started 2026-02-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-18 14:15:00 | 1441.90 | 1430.87 | 1430.02 | EMA200 above EMA400 |

### Cycle 54 — SELL (started 2026-02-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-19 11:15:00 | 1421.10 | 1429.25 | 1429.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-19 12:15:00 | 1418.30 | 1427.06 | 1428.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-20 10:15:00 | 1420.30 | 1417.66 | 1422.58 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-20 11:00:00 | 1420.30 | 1417.66 | 1422.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 11:15:00 | 1422.40 | 1418.61 | 1422.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-20 11:45:00 | 1422.20 | 1418.61 | 1422.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 12:15:00 | 1421.90 | 1419.27 | 1422.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-20 12:45:00 | 1425.20 | 1419.27 | 1422.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 13:15:00 | 1422.40 | 1419.89 | 1422.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-20 13:30:00 | 1420.00 | 1419.89 | 1422.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 14:15:00 | 1418.70 | 1419.66 | 1422.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-20 15:00:00 | 1418.70 | 1419.66 | 1422.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 09:15:00 | 1426.00 | 1420.84 | 1422.25 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-23 11:30:00 | 1422.70 | 1422.34 | 1422.76 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-23 13:45:00 | 1419.80 | 1422.02 | 1422.56 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-23 14:15:00 | 1428.80 | 1423.38 | 1423.13 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 55 — BUY (started 2026-02-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-23 14:15:00 | 1428.80 | 1423.38 | 1423.13 | EMA200 above EMA400 |

### Cycle 56 — SELL (started 2026-02-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-24 09:15:00 | 1416.70 | 1422.30 | 1422.70 | EMA200 below EMA400 |

### Cycle 57 — BUY (started 2026-02-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-24 13:15:00 | 1426.70 | 1423.26 | 1422.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-24 14:15:00 | 1429.30 | 1424.47 | 1423.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-25 12:15:00 | 1412.40 | 1425.09 | 1424.71 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-25 12:15:00 | 1412.40 | 1425.09 | 1424.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 12:15:00 | 1412.40 | 1425.09 | 1424.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-25 13:00:00 | 1412.40 | 1425.09 | 1424.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 58 — SELL (started 2026-02-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-25 13:15:00 | 1410.90 | 1422.26 | 1423.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-25 14:15:00 | 1395.40 | 1416.88 | 1420.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-26 15:15:00 | 1409.00 | 1406.83 | 1412.01 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-27 09:15:00 | 1392.40 | 1406.83 | 1412.01 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 12:15:00 | 1405.50 | 1404.91 | 1409.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-27 12:30:00 | 1405.20 | 1404.91 | 1409.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 13:15:00 | 1407.80 | 1405.48 | 1409.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-27 14:00:00 | 1407.80 | 1405.48 | 1409.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 14:15:00 | 1391.20 | 1402.63 | 1407.45 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-02 09:15:00 | 1367.50 | 1401.06 | 1406.30 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-04 09:15:00 | 1322.78 | 1360.44 | 1378.61 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Stop hit — per-position SL triggered | 2026-03-05 09:15:00 | 1384.00 | 1355.59 | 1365.44 | SL hit (close>ema200) qty=0.50 sl=1355.59 alert=retest1 |

### Cycle 59 — BUY (started 2026-03-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-05 13:15:00 | 1377.90 | 1370.82 | 1370.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-05 14:15:00 | 1387.80 | 1374.22 | 1372.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-09 09:15:00 | 1402.00 | 1403.50 | 1393.19 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-09 09:15:00 | 1402.00 | 1403.50 | 1393.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-09 09:15:00 | 1402.00 | 1403.50 | 1393.19 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-09 14:30:00 | 1414.50 | 1407.57 | 1398.72 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-10 10:15:00 | 1414.30 | 1410.48 | 1401.69 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-10 13:30:00 | 1413.40 | 1411.17 | 1404.85 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-10 14:00:00 | 1414.00 | 1411.17 | 1404.85 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 15:15:00 | 1410.90 | 1410.69 | 1405.71 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-11 09:15:00 | 1419.80 | 1410.69 | 1405.71 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-11 09:15:00 | 1403.70 | 1409.29 | 1405.52 | SL hit (close<static) qty=1.00 sl=1405.70 alert=retest2 |

### Cycle 60 — SELL (started 2026-03-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-11 13:15:00 | 1398.30 | 1403.27 | 1403.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-11 14:15:00 | 1389.60 | 1400.54 | 1402.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-12 10:15:00 | 1406.90 | 1398.97 | 1400.79 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-12 10:15:00 | 1406.90 | 1398.97 | 1400.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 10:15:00 | 1406.90 | 1398.97 | 1400.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-12 11:00:00 | 1406.90 | 1398.97 | 1400.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 11:15:00 | 1409.70 | 1401.11 | 1401.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-12 12:00:00 | 1409.70 | 1401.11 | 1401.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 61 — BUY (started 2026-03-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-12 12:15:00 | 1407.50 | 1402.39 | 1402.14 | EMA200 above EMA400 |

### Cycle 62 — SELL (started 2026-03-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-12 14:15:00 | 1390.60 | 1400.32 | 1401.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-13 12:15:00 | 1380.20 | 1392.66 | 1396.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-16 13:15:00 | 1383.70 | 1381.69 | 1387.66 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-16 14:00:00 | 1383.70 | 1381.69 | 1387.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 14:15:00 | 1396.80 | 1384.71 | 1388.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-16 15:00:00 | 1396.80 | 1384.71 | 1388.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 15:15:00 | 1394.00 | 1386.57 | 1388.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 09:15:00 | 1398.20 | 1386.57 | 1388.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 63 — BUY (started 2026-03-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-17 10:15:00 | 1397.90 | 1391.03 | 1390.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-17 13:15:00 | 1403.60 | 1395.34 | 1392.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-17 15:15:00 | 1394.80 | 1396.15 | 1393.75 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-17 15:15:00 | 1394.80 | 1396.15 | 1393.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 15:15:00 | 1394.80 | 1396.15 | 1393.75 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-18 09:15:00 | 1406.50 | 1396.15 | 1393.75 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-19 09:45:00 | 1404.10 | 1406.13 | 1401.57 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-19 13:15:00 | 1391.10 | 1400.86 | 1400.42 | SL hit (close<static) qty=1.00 sl=1392.30 alert=retest2 |

### Cycle 64 — SELL (started 2026-03-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 14:15:00 | 1385.70 | 1397.83 | 1399.08 | EMA200 below EMA400 |

### Cycle 65 — BUY (started 2026-03-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-20 09:15:00 | 1422.30 | 1401.79 | 1400.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-20 10:15:00 | 1428.90 | 1407.21 | 1403.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-20 15:15:00 | 1414.00 | 1415.45 | 1409.65 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-23 09:15:00 | 1404.50 | 1415.45 | 1409.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-23 09:15:00 | 1400.70 | 1412.50 | 1408.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-23 10:15:00 | 1393.20 | 1412.50 | 1408.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-23 10:15:00 | 1396.00 | 1409.20 | 1407.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-23 10:45:00 | 1396.50 | 1409.20 | 1407.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 66 — SELL (started 2026-03-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-23 12:15:00 | 1400.40 | 1405.95 | 1406.37 | EMA200 below EMA400 |

### Cycle 67 — BUY (started 2026-03-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-23 14:15:00 | 1407.20 | 1406.72 | 1406.68 | EMA200 above EMA400 |

### Cycle 68 — SELL (started 2026-03-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-23 15:15:00 | 1404.00 | 1406.18 | 1406.43 | EMA200 below EMA400 |

### Cycle 69 — BUY (started 2026-03-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-24 09:15:00 | 1413.80 | 1407.70 | 1407.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 10:15:00 | 1427.60 | 1416.25 | 1412.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-25 14:15:00 | 1412.00 | 1417.71 | 1414.51 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-25 14:15:00 | 1412.00 | 1417.71 | 1414.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 14:15:00 | 1412.00 | 1417.71 | 1414.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-25 15:00:00 | 1412.00 | 1417.71 | 1414.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 15:15:00 | 1413.80 | 1416.93 | 1414.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 09:15:00 | 1386.50 | 1416.93 | 1414.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 70 — SELL (started 2026-03-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 09:15:00 | 1375.20 | 1408.58 | 1410.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-27 10:15:00 | 1370.90 | 1401.05 | 1407.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 1375.00 | 1356.17 | 1368.03 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 1375.00 | 1356.17 | 1368.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 1375.00 | 1356.17 | 1368.03 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-01 14:45:00 | 1364.20 | 1367.05 | 1369.82 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-02 09:15:00 | 1350.80 | 1367.62 | 1369.83 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-06 10:15:00 | 1295.99 | 1336.47 | 1349.98 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-07 14:15:00 | 1303.90 | 1302.64 | 1317.37 | SL hit (close>ema200) qty=0.50 sl=1302.64 alert=retest2 |

### Cycle 71 — BUY (started 2026-04-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-08 11:15:00 | 1345.00 | 1323.67 | 1323.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-08 15:15:00 | 1349.20 | 1337.39 | 1330.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-09 09:15:00 | 1336.70 | 1337.25 | 1331.32 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-09 10:00:00 | 1336.70 | 1337.25 | 1331.32 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 10:15:00 | 1336.90 | 1337.18 | 1331.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-09 10:45:00 | 1334.80 | 1337.18 | 1331.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 13:15:00 | 1332.60 | 1335.77 | 1332.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-09 13:30:00 | 1332.30 | 1335.77 | 1332.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 14:15:00 | 1330.70 | 1334.76 | 1332.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-09 15:00:00 | 1330.70 | 1334.76 | 1332.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 15:15:00 | 1329.10 | 1333.62 | 1332.03 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-10 09:15:00 | 1340.00 | 1333.62 | 1332.03 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-13 09:15:00 | 1321.00 | 1340.72 | 1338.38 | SL hit (close<static) qty=1.00 sl=1327.30 alert=retest2 |

### Cycle 72 — SELL (started 2026-04-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-13 10:15:00 | 1320.00 | 1336.57 | 1336.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-13 11:15:00 | 1316.00 | 1332.46 | 1334.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-15 09:15:00 | 1338.40 | 1325.17 | 1329.29 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-15 09:15:00 | 1338.40 | 1325.17 | 1329.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-15 09:15:00 | 1338.40 | 1325.17 | 1329.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-15 09:30:00 | 1338.90 | 1325.17 | 1329.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 73 — BUY (started 2026-04-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-15 12:15:00 | 1339.40 | 1331.89 | 1331.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-15 14:15:00 | 1344.40 | 1335.48 | 1333.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-16 09:15:00 | 1334.00 | 1336.37 | 1334.25 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-16 09:15:00 | 1334.00 | 1336.37 | 1334.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-16 09:15:00 | 1334.00 | 1336.37 | 1334.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-16 10:00:00 | 1334.00 | 1336.37 | 1334.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-16 10:15:00 | 1338.10 | 1336.72 | 1334.60 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-16 14:30:00 | 1341.00 | 1337.32 | 1335.44 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-23 10:15:00 | 1346.70 | 1356.93 | 1357.71 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 74 — SELL (started 2026-04-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-23 10:15:00 | 1346.70 | 1356.93 | 1357.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-23 14:15:00 | 1341.90 | 1351.56 | 1354.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-27 09:15:00 | 1339.00 | 1335.35 | 1342.25 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-27 09:15:00 | 1339.00 | 1335.35 | 1342.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 09:15:00 | 1339.00 | 1335.35 | 1342.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 10:00:00 | 1339.00 | 1335.35 | 1342.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 11:15:00 | 1341.60 | 1336.83 | 1341.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 12:00:00 | 1341.60 | 1336.83 | 1341.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 12:15:00 | 1347.60 | 1338.99 | 1342.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 12:30:00 | 1349.60 | 1338.99 | 1342.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 13:15:00 | 1356.00 | 1342.39 | 1343.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 14:00:00 | 1356.00 | 1342.39 | 1343.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 75 — BUY (started 2026-04-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 14:15:00 | 1365.00 | 1346.91 | 1345.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-27 15:15:00 | 1370.00 | 1351.53 | 1347.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-30 09:15:00 | 1398.60 | 1410.06 | 1392.91 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-30 10:00:00 | 1398.60 | 1410.06 | 1392.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 10:15:00 | 1400.20 | 1408.09 | 1393.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-30 10:45:00 | 1398.80 | 1408.09 | 1393.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 10:15:00 | 1447.80 | 1456.70 | 1446.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-06 10:45:00 | 1445.10 | 1456.70 | 1446.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 11:15:00 | 1446.40 | 1454.64 | 1446.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-06 12:00:00 | 1446.40 | 1454.64 | 1446.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 12:15:00 | 1433.00 | 1450.31 | 1445.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-06 12:45:00 | 1432.80 | 1450.31 | 1445.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 13:15:00 | 1436.80 | 1447.61 | 1444.70 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-06 14:15:00 | 1450.00 | 1447.61 | 1444.70 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-07 09:30:00 | 1442.30 | 1443.16 | 1443.15 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-05-07 10:15:00 | 1435.30 | 1441.59 | 1442.44 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 76 — SELL (started 2026-05-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-07 10:15:00 | 1435.30 | 1441.59 | 1442.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-05-08 09:15:00 | 1428.40 | 1436.89 | 1439.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-05-08 14:15:00 | 1434.60 | 1430.91 | 1435.11 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-08 14:15:00 | 1434.60 | 1430.91 | 1435.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 14:15:00 | 1434.60 | 1430.91 | 1435.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-08 14:45:00 | 1438.50 | 1430.91 | 1435.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 15:15:00 | 1436.00 | 1431.93 | 1435.19 | EMA400 retest candle locked (from downside) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-05-14 09:15:00 | 1426.20 | 2025-05-20 11:15:00 | 1435.70 | STOP_HIT | 1.00 | 0.67% |
| BUY | retest2 | 2025-05-15 10:30:00 | 1429.90 | 2025-05-20 11:15:00 | 1435.70 | STOP_HIT | 1.00 | 0.41% |
| SELL | retest2 | 2025-05-22 09:15:00 | 1415.00 | 2025-05-23 14:15:00 | 1426.00 | STOP_HIT | 1.00 | -0.78% |
| SELL | retest2 | 2025-05-30 11:30:00 | 1413.60 | 2025-06-04 13:15:00 | 1418.40 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest2 | 2025-05-30 14:00:00 | 1413.40 | 2025-06-04 13:15:00 | 1418.40 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest2 | 2025-06-02 09:15:00 | 1398.90 | 2025-06-04 13:15:00 | 1418.40 | STOP_HIT | 1.00 | -1.39% |
| SELL | retest2 | 2025-06-02 12:45:00 | 1411.80 | 2025-06-04 13:15:00 | 1418.40 | STOP_HIT | 1.00 | -0.47% |
| SELL | retest2 | 2025-06-03 14:45:00 | 1408.40 | 2025-06-04 13:15:00 | 1418.40 | STOP_HIT | 1.00 | -0.71% |
| SELL | retest2 | 2025-06-04 10:00:00 | 1408.10 | 2025-06-04 13:15:00 | 1418.40 | STOP_HIT | 1.00 | -0.73% |
| SELL | retest2 | 2025-06-04 11:00:00 | 1408.00 | 2025-06-04 13:15:00 | 1418.40 | STOP_HIT | 1.00 | -0.74% |
| BUY | retest2 | 2025-06-11 09:15:00 | 1456.70 | 2025-06-12 14:15:00 | 1442.40 | STOP_HIT | 1.00 | -0.98% |
| SELL | retest2 | 2025-06-16 15:15:00 | 1436.90 | 2025-06-19 13:15:00 | 1437.30 | STOP_HIT | 1.00 | -0.03% |
| SELL | retest2 | 2025-06-18 09:45:00 | 1437.00 | 2025-06-19 13:15:00 | 1437.30 | STOP_HIT | 1.00 | -0.02% |
| SELL | retest2 | 2025-06-18 10:15:00 | 1436.00 | 2025-06-19 13:15:00 | 1437.30 | STOP_HIT | 1.00 | -0.09% |
| SELL | retest2 | 2025-06-19 13:00:00 | 1434.40 | 2025-06-19 13:15:00 | 1437.30 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest2 | 2025-06-24 15:15:00 | 1456.80 | 2025-07-09 15:15:00 | 1520.50 | STOP_HIT | 1.00 | 4.37% |
| SELL | retest2 | 2025-07-17 09:30:00 | 1483.40 | 2025-07-24 11:15:00 | 1409.23 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-17 11:30:00 | 1483.60 | 2025-07-24 11:15:00 | 1409.42 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-17 12:45:00 | 1483.50 | 2025-07-24 11:15:00 | 1409.33 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-17 09:30:00 | 1483.40 | 2025-07-28 09:15:00 | 1399.10 | STOP_HIT | 0.50 | 5.68% |
| SELL | retest2 | 2025-07-17 11:30:00 | 1483.60 | 2025-07-28 09:15:00 | 1399.10 | STOP_HIT | 0.50 | 5.70% |
| SELL | retest2 | 2025-07-17 12:45:00 | 1483.50 | 2025-07-28 09:15:00 | 1399.10 | STOP_HIT | 0.50 | 5.69% |
| SELL | retest2 | 2025-08-08 10:30:00 | 1376.70 | 2025-08-12 09:15:00 | 1397.20 | STOP_HIT | 1.00 | -1.49% |
| SELL | retest2 | 2025-08-26 09:15:00 | 1408.00 | 2025-09-05 13:15:00 | 1374.40 | STOP_HIT | 1.00 | 2.39% |
| BUY | retest2 | 2025-09-10 09:15:00 | 1381.40 | 2025-09-22 10:15:00 | 1404.70 | STOP_HIT | 1.00 | 1.69% |
| BUY | retest2 | 2025-09-11 09:15:00 | 1380.70 | 2025-09-22 10:15:00 | 1404.70 | STOP_HIT | 1.00 | 1.74% |
| SELL | retest2 | 2025-09-24 13:45:00 | 1389.70 | 2025-10-06 14:15:00 | 1374.40 | STOP_HIT | 1.00 | 1.10% |
| BUY | retest2 | 2025-10-24 11:30:00 | 1454.60 | 2025-10-31 14:15:00 | 1486.50 | STOP_HIT | 1.00 | 2.19% |
| BUY | retest2 | 2025-10-24 15:00:00 | 1452.40 | 2025-10-31 14:15:00 | 1486.50 | STOP_HIT | 1.00 | 2.35% |
| SELL | retest2 | 2025-11-04 13:00:00 | 1480.50 | 2025-11-06 09:15:00 | 1493.20 | STOP_HIT | 1.00 | -0.86% |
| SELL | retest2 | 2025-11-04 14:00:00 | 1479.80 | 2025-11-06 09:15:00 | 1493.20 | STOP_HIT | 1.00 | -0.91% |
| BUY | retest2 | 2025-11-14 10:45:00 | 1517.90 | 2025-12-02 10:15:00 | 1553.40 | STOP_HIT | 1.00 | 2.34% |
| BUY | retest2 | 2025-11-14 12:15:00 | 1517.50 | 2025-12-02 10:15:00 | 1553.40 | STOP_HIT | 1.00 | 2.37% |
| BUY | retest2 | 2025-11-14 13:15:00 | 1516.80 | 2025-12-02 10:15:00 | 1553.40 | STOP_HIT | 1.00 | 2.41% |
| BUY | retest2 | 2025-11-14 15:00:00 | 1520.10 | 2025-12-02 10:15:00 | 1553.40 | STOP_HIT | 1.00 | 2.19% |
| BUY | retest2 | 2025-11-17 15:00:00 | 1518.00 | 2025-12-02 10:15:00 | 1553.40 | STOP_HIT | 1.00 | 2.33% |
| BUY | retest2 | 2025-11-18 11:45:00 | 1521.40 | 2025-12-02 10:15:00 | 1553.40 | STOP_HIT | 1.00 | 2.10% |
| BUY | retest2 | 2025-11-18 15:15:00 | 1520.00 | 2025-12-02 10:15:00 | 1553.40 | STOP_HIT | 1.00 | 2.20% |
| BUY | retest2 | 2025-11-19 09:45:00 | 1517.70 | 2025-12-02 10:15:00 | 1553.40 | STOP_HIT | 1.00 | 2.35% |
| BUY | retest2 | 2025-11-19 12:15:00 | 1518.70 | 2025-12-02 10:15:00 | 1553.40 | STOP_HIT | 1.00 | 2.28% |
| BUY | retest2 | 2025-11-19 13:45:00 | 1520.00 | 2025-12-02 10:15:00 | 1553.40 | STOP_HIT | 1.00 | 2.20% |
| SELL | retest2 | 2025-12-04 12:15:00 | 1548.60 | 2025-12-11 12:15:00 | 1546.40 | STOP_HIT | 1.00 | 0.14% |
| BUY | retest1 | 2026-01-02 09:15:00 | 1585.00 | 2026-01-05 13:15:00 | 1579.70 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest2 | 2026-01-19 09:15:00 | 1421.60 | 2026-01-28 15:15:00 | 1398.50 | STOP_HIT | 1.00 | 1.62% |
| SELL | retest2 | 2026-02-17 09:15:00 | 1424.10 | 2026-02-18 14:15:00 | 1441.90 | STOP_HIT | 1.00 | -1.25% |
| SELL | retest2 | 2026-02-18 13:15:00 | 1433.00 | 2026-02-18 14:15:00 | 1441.90 | STOP_HIT | 1.00 | -0.62% |
| SELL | retest2 | 2026-02-23 11:30:00 | 1422.70 | 2026-02-23 14:15:00 | 1428.80 | STOP_HIT | 1.00 | -0.43% |
| SELL | retest2 | 2026-02-23 13:45:00 | 1419.80 | 2026-02-23 14:15:00 | 1428.80 | STOP_HIT | 1.00 | -0.63% |
| SELL | retest1 | 2026-02-27 09:15:00 | 1392.40 | 2026-03-04 09:15:00 | 1322.78 | PARTIAL | 0.50 | 5.00% |
| SELL | retest1 | 2026-02-27 09:15:00 | 1392.40 | 2026-03-05 09:15:00 | 1384.00 | STOP_HIT | 0.50 | 0.60% |
| SELL | retest2 | 2026-03-02 09:15:00 | 1367.50 | 2026-03-05 13:15:00 | 1377.90 | STOP_HIT | 1.00 | -0.76% |
| BUY | retest2 | 2026-03-09 14:30:00 | 1414.50 | 2026-03-11 09:15:00 | 1403.70 | STOP_HIT | 1.00 | -0.76% |
| BUY | retest2 | 2026-03-10 10:15:00 | 1414.30 | 2026-03-11 13:15:00 | 1398.30 | STOP_HIT | 1.00 | -1.13% |
| BUY | retest2 | 2026-03-10 13:30:00 | 1413.40 | 2026-03-11 13:15:00 | 1398.30 | STOP_HIT | 1.00 | -1.07% |
| BUY | retest2 | 2026-03-10 14:00:00 | 1414.00 | 2026-03-11 13:15:00 | 1398.30 | STOP_HIT | 1.00 | -1.11% |
| BUY | retest2 | 2026-03-11 09:15:00 | 1419.80 | 2026-03-11 13:15:00 | 1398.30 | STOP_HIT | 1.00 | -1.51% |
| BUY | retest2 | 2026-03-18 09:15:00 | 1406.50 | 2026-03-19 13:15:00 | 1391.10 | STOP_HIT | 1.00 | -1.09% |
| BUY | retest2 | 2026-03-19 09:45:00 | 1404.10 | 2026-03-19 13:15:00 | 1391.10 | STOP_HIT | 1.00 | -0.93% |
| SELL | retest2 | 2026-04-01 14:45:00 | 1364.20 | 2026-04-06 10:15:00 | 1295.99 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-04-01 14:45:00 | 1364.20 | 2026-04-07 14:15:00 | 1303.90 | STOP_HIT | 0.50 | 4.42% |
| SELL | retest2 | 2026-04-02 09:15:00 | 1350.80 | 2026-04-08 11:15:00 | 1345.00 | STOP_HIT | 1.00 | 0.43% |
| BUY | retest2 | 2026-04-10 09:15:00 | 1340.00 | 2026-04-13 09:15:00 | 1321.00 | STOP_HIT | 1.00 | -1.42% |
| BUY | retest2 | 2026-04-16 14:30:00 | 1341.00 | 2026-04-23 10:15:00 | 1346.70 | STOP_HIT | 1.00 | 0.43% |
| BUY | retest2 | 2026-05-06 14:15:00 | 1450.00 | 2026-05-07 10:15:00 | 1435.30 | STOP_HIT | 1.00 | -1.01% |
| BUY | retest2 | 2026-05-07 09:30:00 | 1442.30 | 2026-05-07 10:15:00 | 1435.30 | STOP_HIT | 1.00 | -0.49% |
