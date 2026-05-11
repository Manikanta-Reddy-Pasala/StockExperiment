# SBI Life Insurance Company Ltd. (SBILIFE)

## Backtest Summary

- **Window:** 2024-03-13 09:15:00 → 2026-05-08 15:15:00 (3710 bars)
- **Last close:** 1871.10
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 155 |
| ALERT1 | 104 |
| ALERT2 | 103 |
| ALERT2_SKIP | 58 |
| ALERT3 | 267 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 4 |
| ENTRY2 | 129 |
| PARTIAL | 6 |
| TARGET_HIT | 9 |
| STOP_HIT | 124 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 139 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 59 / 80
- **Target hits / Stop hits / Partials:** 9 / 124 / 6
- **Avg / median % per leg:** 0.70% / -0.56%
- **Sum % (uncompounded):** 97.10%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 59 | 32 | 54.2% | 9 | 50 | 0 | 1.54% | 91.1% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 59 | 32 | 54.2% | 9 | 50 | 0 | 1.54% | 91.1% |
| SELL (all) | 80 | 27 | 33.8% | 0 | 74 | 6 | 0.07% | 6.0% |
| SELL @ 2nd Alert (retest1) | 4 | 4 | 100.0% | 0 | 4 | 0 | 1.18% | 4.7% |
| SELL @ 3rd Alert (retest2) | 76 | 23 | 30.3% | 0 | 70 | 6 | 0.02% | 1.3% |
| retest1 (combined) | 4 | 4 | 100.0% | 0 | 4 | 0 | 1.18% | 4.7% |
| retest2 (combined) | 135 | 55 | 40.7% | 9 | 120 | 6 | 0.68% | 92.4% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-05-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-14 15:15:00 | 1436.50 | 1430.20 | 1429.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-15 10:15:00 | 1440.00 | 1432.74 | 1431.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-15 12:15:00 | 1430.10 | 1432.60 | 1431.33 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-15 12:15:00 | 1430.10 | 1432.60 | 1431.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-15 12:15:00 | 1430.10 | 1432.60 | 1431.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-15 12:45:00 | 1429.70 | 1432.60 | 1431.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-15 13:15:00 | 1434.90 | 1433.06 | 1431.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-15 13:45:00 | 1433.10 | 1433.06 | 1431.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-15 14:15:00 | 1429.20 | 1432.29 | 1431.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-15 15:00:00 | 1429.20 | 1432.29 | 1431.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-15 15:15:00 | 1428.80 | 1431.59 | 1431.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-16 09:15:00 | 1426.00 | 1431.59 | 1431.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 2 — SELL (started 2024-05-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-16 09:15:00 | 1426.00 | 1430.47 | 1430.72 | EMA200 below EMA400 |

### Cycle 3 — BUY (started 2024-05-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-16 11:15:00 | 1440.60 | 1431.46 | 1431.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-16 14:15:00 | 1453.10 | 1436.66 | 1433.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-17 09:15:00 | 1437.25 | 1438.95 | 1435.32 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-17 09:15:00 | 1437.25 | 1438.95 | 1435.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-17 09:15:00 | 1437.25 | 1438.95 | 1435.32 | EMA400 retest candle locked (from upside) |

### Cycle 4 — SELL (started 2024-05-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-21 09:15:00 | 1428.35 | 1435.34 | 1435.73 | EMA200 below EMA400 |

### Cycle 5 — BUY (started 2024-05-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-23 10:15:00 | 1438.75 | 1432.23 | 1431.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-23 12:15:00 | 1446.40 | 1436.82 | 1434.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-24 09:15:00 | 1437.90 | 1440.59 | 1437.21 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-24 09:15:00 | 1437.90 | 1440.59 | 1437.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-24 09:15:00 | 1437.90 | 1440.59 | 1437.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-24 09:30:00 | 1439.05 | 1440.59 | 1437.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-24 10:15:00 | 1444.75 | 1441.42 | 1437.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-24 10:30:00 | 1437.90 | 1441.42 | 1437.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-24 11:15:00 | 1442.10 | 1441.56 | 1438.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-24 12:00:00 | 1442.10 | 1441.56 | 1438.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-24 13:15:00 | 1439.10 | 1441.39 | 1438.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-24 14:00:00 | 1439.10 | 1441.39 | 1438.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-24 14:15:00 | 1436.90 | 1440.49 | 1438.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-24 15:00:00 | 1436.90 | 1440.49 | 1438.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-24 15:15:00 | 1437.00 | 1439.79 | 1438.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-27 09:15:00 | 1437.25 | 1439.79 | 1438.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-27 09:15:00 | 1436.00 | 1439.04 | 1438.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-27 09:30:00 | 1438.10 | 1439.04 | 1438.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-27 10:15:00 | 1434.25 | 1438.08 | 1437.88 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-27 11:45:00 | 1439.85 | 1438.75 | 1438.21 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-27 13:15:00 | 1431.00 | 1437.05 | 1437.52 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 6 — SELL (started 2024-05-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-27 13:15:00 | 1431.00 | 1437.05 | 1437.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-27 14:15:00 | 1409.20 | 1431.48 | 1434.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-28 09:15:00 | 1426.70 | 1426.61 | 1431.89 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-28 10:00:00 | 1426.70 | 1426.61 | 1431.89 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-28 10:15:00 | 1430.20 | 1427.33 | 1431.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-28 10:45:00 | 1429.45 | 1427.33 | 1431.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-28 11:15:00 | 1432.45 | 1428.35 | 1431.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-28 12:00:00 | 1432.45 | 1428.35 | 1431.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-28 12:15:00 | 1434.10 | 1429.50 | 1432.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-28 12:45:00 | 1434.90 | 1429.50 | 1432.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-28 13:15:00 | 1434.00 | 1430.40 | 1432.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-28 13:45:00 | 1437.90 | 1430.40 | 1432.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 7 — BUY (started 2024-05-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-28 14:15:00 | 1450.20 | 1434.36 | 1433.83 | EMA200 above EMA400 |

### Cycle 8 — SELL (started 2024-05-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-29 10:15:00 | 1417.05 | 1431.05 | 1432.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-29 11:15:00 | 1408.15 | 1426.47 | 1430.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-31 12:15:00 | 1395.30 | 1388.73 | 1399.47 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-31 12:45:00 | 1395.05 | 1388.73 | 1399.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 13:15:00 | 1398.05 | 1390.59 | 1399.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-31 13:45:00 | 1400.00 | 1390.59 | 1399.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 14:15:00 | 1382.60 | 1388.99 | 1397.82 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-04 09:15:00 | 1355.05 | 1393.02 | 1396.02 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-04 09:15:00 | 1401.30 | 1394.68 | 1396.50 | SL hit (close>static) qty=1.00 sl=1398.75 alert=retest2 |

### Cycle 9 — BUY (started 2024-06-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-05 15:15:00 | 1385.00 | 1378.37 | 1378.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-06 09:15:00 | 1421.10 | 1386.91 | 1381.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-07 13:15:00 | 1421.05 | 1422.06 | 1410.53 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-07 13:30:00 | 1420.70 | 1422.06 | 1410.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-11 09:15:00 | 1436.50 | 1432.62 | 1424.29 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-12 10:30:00 | 1442.35 | 1433.81 | 1428.98 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-13 14:15:00 | 1440.35 | 1445.48 | 1440.91 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-19 14:15:00 | 1448.70 | 1458.23 | 1458.32 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 10 — SELL (started 2024-06-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-19 14:15:00 | 1448.70 | 1458.23 | 1458.32 | EMA200 below EMA400 |

### Cycle 11 — BUY (started 2024-06-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-21 09:15:00 | 1465.95 | 1457.81 | 1457.36 | EMA200 above EMA400 |

### Cycle 12 — SELL (started 2024-06-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-24 11:15:00 | 1456.95 | 1459.13 | 1459.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-24 12:15:00 | 1451.05 | 1457.51 | 1458.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-25 12:15:00 | 1453.80 | 1448.19 | 1452.54 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-25 12:15:00 | 1453.80 | 1448.19 | 1452.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-25 12:15:00 | 1453.80 | 1448.19 | 1452.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-25 12:45:00 | 1454.95 | 1448.19 | 1452.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-25 13:15:00 | 1455.75 | 1449.70 | 1452.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-25 14:00:00 | 1455.75 | 1449.70 | 1452.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-25 14:15:00 | 1464.30 | 1452.62 | 1453.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-25 15:00:00 | 1464.30 | 1452.62 | 1453.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-25 15:15:00 | 1461.00 | 1454.30 | 1454.52 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-26 09:15:00 | 1457.60 | 1454.30 | 1454.52 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-26 09:15:00 | 1465.70 | 1456.58 | 1455.54 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 13 — BUY (started 2024-06-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-26 09:15:00 | 1465.70 | 1456.58 | 1455.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-26 10:15:00 | 1467.55 | 1458.77 | 1456.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-26 11:15:00 | 1457.80 | 1458.58 | 1456.73 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-26 12:00:00 | 1457.80 | 1458.58 | 1456.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-26 12:15:00 | 1450.25 | 1456.91 | 1456.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-26 13:00:00 | 1450.25 | 1456.91 | 1456.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-26 13:15:00 | 1453.10 | 1456.15 | 1455.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-26 13:45:00 | 1450.40 | 1456.15 | 1455.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 14 — SELL (started 2024-06-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-26 14:15:00 | 1451.60 | 1455.24 | 1455.48 | EMA200 below EMA400 |

### Cycle 15 — BUY (started 2024-06-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-27 09:15:00 | 1461.35 | 1455.75 | 1455.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-28 09:15:00 | 1491.75 | 1465.72 | 1460.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-02 09:15:00 | 1473.95 | 1492.34 | 1484.81 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-02 09:15:00 | 1473.95 | 1492.34 | 1484.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-02 09:15:00 | 1473.95 | 1492.34 | 1484.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-02 09:45:00 | 1469.80 | 1492.34 | 1484.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-02 10:15:00 | 1490.65 | 1492.01 | 1485.34 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-02 11:45:00 | 1496.55 | 1492.15 | 1486.02 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-02 14:00:00 | 1496.95 | 1492.35 | 1487.12 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-03 09:15:00 | 1498.60 | 1492.56 | 1488.15 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Target hit | 2024-07-18 09:15:00 | 1646.21 | 1614.83 | 1596.34 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 16 — SELL (started 2024-07-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-22 15:15:00 | 1619.10 | 1628.86 | 1629.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-23 09:15:00 | 1610.35 | 1625.16 | 1627.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-24 09:15:00 | 1625.15 | 1609.61 | 1615.88 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-24 09:15:00 | 1625.15 | 1609.61 | 1615.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-24 09:15:00 | 1625.15 | 1609.61 | 1615.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-24 10:00:00 | 1625.15 | 1609.61 | 1615.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-24 10:15:00 | 1643.65 | 1616.42 | 1618.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-24 10:45:00 | 1649.70 | 1616.42 | 1618.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-24 13:15:00 | 1610.00 | 1616.49 | 1618.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-24 14:15:00 | 1625.00 | 1616.49 | 1618.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 17 — BUY (started 2024-07-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-24 14:15:00 | 1635.40 | 1620.27 | 1619.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-25 09:15:00 | 1662.55 | 1630.71 | 1624.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-30 09:15:00 | 1710.90 | 1737.64 | 1721.43 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-30 09:15:00 | 1710.90 | 1737.64 | 1721.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-30 09:15:00 | 1710.90 | 1737.64 | 1721.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-30 10:00:00 | 1710.90 | 1737.64 | 1721.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-30 10:15:00 | 1725.00 | 1735.11 | 1721.76 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-30 11:15:00 | 1727.00 | 1735.11 | 1721.76 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-31 09:45:00 | 1733.25 | 1732.01 | 1725.94 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-05 09:15:00 | 1737.90 | 1750.03 | 1750.68 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 18 — SELL (started 2024-08-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-05 09:15:00 | 1737.90 | 1750.03 | 1750.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-06 14:15:00 | 1674.30 | 1698.31 | 1716.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-08 11:15:00 | 1691.60 | 1685.09 | 1695.13 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-08 12:00:00 | 1691.60 | 1685.09 | 1695.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-08 12:15:00 | 1683.75 | 1684.82 | 1694.10 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-08 13:15:00 | 1679.25 | 1684.82 | 1694.10 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-08 14:15:00 | 1705.30 | 1690.63 | 1695.23 | SL hit (close>static) qty=1.00 sl=1698.00 alert=retest2 |

### Cycle 19 — BUY (started 2024-08-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-09 10:15:00 | 1728.10 | 1703.35 | 1700.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-09 13:15:00 | 1730.05 | 1715.97 | 1707.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-12 09:15:00 | 1700.70 | 1716.16 | 1709.97 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-12 09:15:00 | 1700.70 | 1716.16 | 1709.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-12 09:15:00 | 1700.70 | 1716.16 | 1709.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-12 09:45:00 | 1700.20 | 1716.16 | 1709.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-12 10:15:00 | 1717.00 | 1716.32 | 1710.61 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-12 11:45:00 | 1727.80 | 1717.11 | 1711.49 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-13 09:15:00 | 1681.10 | 1704.72 | 1707.40 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 20 — SELL (started 2024-08-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-13 09:15:00 | 1681.10 | 1704.72 | 1707.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-13 13:15:00 | 1676.45 | 1692.20 | 1700.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-14 10:15:00 | 1694.40 | 1689.23 | 1695.75 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-14 10:15:00 | 1694.40 | 1689.23 | 1695.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-14 10:15:00 | 1694.40 | 1689.23 | 1695.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-14 11:00:00 | 1694.40 | 1689.23 | 1695.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-14 11:15:00 | 1687.20 | 1688.83 | 1694.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-14 11:30:00 | 1693.90 | 1688.83 | 1694.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-14 12:15:00 | 1700.45 | 1691.15 | 1695.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-14 13:00:00 | 1700.45 | 1691.15 | 1695.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-14 13:15:00 | 1696.95 | 1692.31 | 1695.61 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-14 15:15:00 | 1688.75 | 1691.90 | 1695.12 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-16 09:30:00 | 1680.65 | 1687.81 | 1692.67 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-16 14:30:00 | 1687.25 | 1687.60 | 1690.43 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-16 15:00:00 | 1687.65 | 1687.60 | 1690.43 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-16 15:15:00 | 1690.90 | 1688.26 | 1690.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-19 09:15:00 | 1684.10 | 1688.26 | 1690.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-19 09:15:00 | 1681.20 | 1686.85 | 1689.63 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-19 10:15:00 | 1676.50 | 1686.85 | 1689.63 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-20 09:15:00 | 1705.85 | 1680.92 | 1683.37 | SL hit (close>static) qty=1.00 sl=1699.90 alert=retest2 |

### Cycle 21 — BUY (started 2024-08-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-20 10:15:00 | 1716.80 | 1688.09 | 1686.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-20 11:15:00 | 1719.90 | 1694.45 | 1689.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-22 14:15:00 | 1794.75 | 1795.22 | 1772.42 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-22 15:00:00 | 1794.75 | 1795.22 | 1772.42 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-26 09:15:00 | 1796.60 | 1793.52 | 1784.23 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-26 13:15:00 | 1799.55 | 1793.70 | 1786.57 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-27 09:45:00 | 1804.45 | 1796.93 | 1790.41 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-27 10:45:00 | 1806.10 | 1799.83 | 1792.32 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-06 14:15:00 | 1893.45 | 1900.57 | 1900.86 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 22 — SELL (started 2024-09-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-06 14:15:00 | 1893.45 | 1900.57 | 1900.86 | EMA200 below EMA400 |

### Cycle 23 — BUY (started 2024-09-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-09 09:15:00 | 1910.50 | 1902.94 | 1901.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-09 11:15:00 | 1928.95 | 1911.54 | 1906.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-09 13:15:00 | 1898.35 | 1911.05 | 1907.04 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-09 13:15:00 | 1898.35 | 1911.05 | 1907.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-09 13:15:00 | 1898.35 | 1911.05 | 1907.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-09 13:45:00 | 1896.20 | 1911.05 | 1907.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-09 14:15:00 | 1901.95 | 1909.23 | 1906.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-09 14:45:00 | 1898.10 | 1909.23 | 1906.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 24 — SELL (started 2024-09-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-10 09:15:00 | 1843.00 | 1894.03 | 1900.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-16 09:15:00 | 1832.00 | 1848.00 | 1856.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-17 10:15:00 | 1831.90 | 1828.50 | 1838.83 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-17 11:00:00 | 1831.90 | 1828.50 | 1838.83 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-18 09:15:00 | 1840.25 | 1826.57 | 1832.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-18 09:45:00 | 1841.10 | 1826.57 | 1832.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-18 10:15:00 | 1847.25 | 1830.71 | 1834.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-18 11:00:00 | 1847.25 | 1830.71 | 1834.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 25 — BUY (started 2024-09-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-18 14:15:00 | 1842.15 | 1836.34 | 1836.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-19 09:15:00 | 1854.55 | 1841.05 | 1838.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-19 11:15:00 | 1838.90 | 1841.52 | 1839.06 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-19 11:15:00 | 1838.90 | 1841.52 | 1839.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 11:15:00 | 1838.90 | 1841.52 | 1839.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-19 12:00:00 | 1838.90 | 1841.52 | 1839.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 12:15:00 | 1829.70 | 1839.16 | 1838.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-19 13:00:00 | 1829.70 | 1839.16 | 1838.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 13:15:00 | 1835.40 | 1838.41 | 1837.95 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-19 14:45:00 | 1840.50 | 1839.53 | 1838.50 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-25 09:15:00 | 1859.05 | 1876.77 | 1877.86 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 26 — SELL (started 2024-09-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-25 09:15:00 | 1859.05 | 1876.77 | 1877.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-25 10:15:00 | 1845.60 | 1870.54 | 1874.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-25 14:15:00 | 1865.05 | 1863.52 | 1869.60 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-25 15:00:00 | 1865.05 | 1863.52 | 1869.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 15:15:00 | 1871.00 | 1865.02 | 1869.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-26 09:15:00 | 1888.85 | 1865.02 | 1869.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-26 09:15:00 | 1887.05 | 1869.43 | 1871.30 | EMA400 retest candle locked (from downside) |

### Cycle 27 — BUY (started 2024-09-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-26 10:15:00 | 1889.20 | 1873.38 | 1872.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-26 14:15:00 | 1898.00 | 1882.72 | 1877.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-27 14:15:00 | 1883.90 | 1895.62 | 1888.77 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-27 14:15:00 | 1883.90 | 1895.62 | 1888.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-27 14:15:00 | 1883.90 | 1895.62 | 1888.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-27 15:00:00 | 1883.90 | 1895.62 | 1888.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-27 15:15:00 | 1893.00 | 1895.09 | 1889.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-30 09:15:00 | 1890.95 | 1895.09 | 1889.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-30 09:15:00 | 1881.75 | 1892.43 | 1888.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-30 10:00:00 | 1881.75 | 1892.43 | 1888.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-30 10:15:00 | 1870.10 | 1887.96 | 1886.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-30 11:00:00 | 1870.10 | 1887.96 | 1886.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 28 — SELL (started 2024-09-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-30 11:15:00 | 1860.30 | 1882.43 | 1884.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-30 13:15:00 | 1848.70 | 1872.18 | 1879.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-04 09:15:00 | 1828.30 | 1817.79 | 1832.20 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-04 09:15:00 | 1828.30 | 1817.79 | 1832.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-04 09:15:00 | 1828.30 | 1817.79 | 1832.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-04 09:45:00 | 1840.65 | 1817.79 | 1832.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-07 09:15:00 | 1797.85 | 1805.05 | 1818.10 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-07 10:45:00 | 1787.40 | 1802.05 | 1815.55 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-07 11:45:00 | 1788.35 | 1799.61 | 1813.22 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-15 13:15:00 | 1746.40 | 1740.45 | 1739.76 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 29 — BUY (started 2024-10-15 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-15 13:15:00 | 1746.40 | 1740.45 | 1739.76 | EMA200 above EMA400 |

### Cycle 30 — SELL (started 2024-10-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-15 14:15:00 | 1722.15 | 1736.79 | 1738.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-17 09:15:00 | 1707.85 | 1727.93 | 1732.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-18 13:15:00 | 1707.45 | 1706.62 | 1714.44 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-18 13:45:00 | 1708.85 | 1706.62 | 1714.44 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-21 09:15:00 | 1713.70 | 1707.46 | 1712.82 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-21 10:30:00 | 1710.00 | 1708.96 | 1713.02 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-21 12:00:00 | 1710.00 | 1709.17 | 1712.74 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-21 14:00:00 | 1709.50 | 1707.69 | 1711.35 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-21 15:15:00 | 1707.00 | 1708.75 | 1711.50 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-21 15:15:00 | 1707.00 | 1708.40 | 1711.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-22 09:30:00 | 1719.95 | 1711.11 | 1712.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-22 10:15:00 | 1713.80 | 1711.65 | 1712.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-22 10:30:00 | 1720.85 | 1711.65 | 1712.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-22 11:15:00 | 1696.35 | 1708.59 | 1710.79 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-10-23 12:15:00 | 1718.95 | 1709.68 | 1709.28 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 31 — BUY (started 2024-10-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-23 12:15:00 | 1718.95 | 1709.68 | 1709.28 | EMA200 above EMA400 |

### Cycle 32 — SELL (started 2024-10-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-24 09:15:00 | 1641.85 | 1698.09 | 1704.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-25 10:15:00 | 1616.80 | 1641.50 | 1665.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-28 15:15:00 | 1615.00 | 1610.55 | 1626.61 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-29 09:15:00 | 1610.70 | 1610.55 | 1626.61 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-29 10:15:00 | 1625.70 | 1613.01 | 1624.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-29 11:00:00 | 1625.70 | 1613.01 | 1624.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-29 11:15:00 | 1638.50 | 1618.11 | 1626.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-29 12:00:00 | 1638.50 | 1618.11 | 1626.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-29 12:15:00 | 1657.00 | 1625.89 | 1628.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-29 12:45:00 | 1658.80 | 1625.89 | 1628.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 33 — BUY (started 2024-10-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-29 13:15:00 | 1660.50 | 1632.81 | 1631.82 | EMA200 above EMA400 |

### Cycle 34 — SELL (started 2024-10-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-30 14:15:00 | 1625.35 | 1633.29 | 1634.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-31 12:15:00 | 1616.50 | 1625.58 | 1629.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-01 17:15:00 | 1626.05 | 1623.94 | 1627.48 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-01 17:15:00 | 1626.05 | 1623.94 | 1627.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-01 17:15:00 | 1626.05 | 1623.94 | 1627.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-01 18:00:00 | 1626.05 | 1623.94 | 1627.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-01 18:15:00 | 1633.65 | 1625.88 | 1628.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-01 18:45:00 | 1633.65 | 1625.88 | 1628.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 09:15:00 | 1604.30 | 1621.56 | 1625.89 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-04 10:30:00 | 1601.45 | 1615.93 | 1622.93 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-04 11:30:00 | 1601.85 | 1613.98 | 1621.41 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-04 12:15:00 | 1600.55 | 1613.98 | 1621.41 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-04 12:45:00 | 1600.25 | 1611.93 | 1619.80 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-05 09:15:00 | 1609.60 | 1610.03 | 1616.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-05 09:30:00 | 1608.95 | 1610.03 | 1616.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-05 10:15:00 | 1612.50 | 1610.52 | 1615.94 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-05 11:30:00 | 1602.00 | 1609.08 | 1614.79 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-05 12:15:00 | 1601.40 | 1609.08 | 1614.79 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-05 13:15:00 | 1639.30 | 1615.17 | 1616.57 | SL hit (close>static) qty=1.00 sl=1619.05 alert=retest2 |

### Cycle 35 — BUY (started 2024-11-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-05 14:15:00 | 1631.00 | 1618.34 | 1617.88 | EMA200 above EMA400 |

### Cycle 36 — SELL (started 2024-11-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-06 10:15:00 | 1614.40 | 1617.47 | 1617.66 | EMA200 below EMA400 |

### Cycle 37 — BUY (started 2024-11-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-06 12:15:00 | 1618.55 | 1617.82 | 1617.79 | EMA200 above EMA400 |

### Cycle 38 — SELL (started 2024-11-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-06 13:15:00 | 1614.35 | 1617.13 | 1617.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-06 14:15:00 | 1602.00 | 1614.10 | 1616.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-08 09:15:00 | 1599.00 | 1593.14 | 1600.52 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-08 09:15:00 | 1599.00 | 1593.14 | 1600.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-08 09:15:00 | 1599.00 | 1593.14 | 1600.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-08 09:45:00 | 1594.60 | 1593.14 | 1600.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-08 10:15:00 | 1590.35 | 1592.58 | 1599.59 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-08 11:15:00 | 1584.15 | 1592.58 | 1599.59 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-19 12:15:00 | 1504.94 | 1539.95 | 1551.29 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-11-22 11:15:00 | 1490.20 | 1489.61 | 1507.10 | SL hit (close>ema200) qty=0.50 sl=1489.61 alert=retest2 |

### Cycle 39 — BUY (started 2024-11-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-26 11:15:00 | 1514.95 | 1501.58 | 1501.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-28 09:15:00 | 1515.40 | 1506.98 | 1504.98 | Break + close above crossover candle high |

### Cycle 40 — SELL (started 2024-11-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-28 12:15:00 | 1448.15 | 1496.42 | 1500.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-28 13:15:00 | 1415.80 | 1480.30 | 1493.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-29 14:15:00 | 1438.45 | 1426.23 | 1451.25 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-29 15:00:00 | 1438.45 | 1426.23 | 1451.25 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-02 09:15:00 | 1432.75 | 1429.10 | 1448.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-02 09:30:00 | 1440.55 | 1429.10 | 1448.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-03 09:15:00 | 1422.95 | 1424.09 | 1436.02 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-03 10:30:00 | 1419.10 | 1423.55 | 1434.69 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-03 13:15:00 | 1444.55 | 1429.68 | 1434.85 | SL hit (close>static) qty=1.00 sl=1438.45 alert=retest2 |

### Cycle 41 — BUY (started 2024-12-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-04 09:15:00 | 1463.10 | 1439.21 | 1438.21 | EMA200 above EMA400 |

### Cycle 42 — SELL (started 2024-12-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-05 11:15:00 | 1430.00 | 1440.94 | 1442.37 | EMA200 below EMA400 |

### Cycle 43 — BUY (started 2024-12-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-06 14:15:00 | 1449.20 | 1440.89 | 1440.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-09 09:15:00 | 1462.95 | 1446.17 | 1442.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-10 11:15:00 | 1467.95 | 1469.65 | 1460.50 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-10 12:00:00 | 1467.95 | 1469.65 | 1460.50 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-10 13:15:00 | 1459.55 | 1467.26 | 1460.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-10 14:00:00 | 1459.55 | 1467.26 | 1460.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-10 14:15:00 | 1462.50 | 1466.31 | 1461.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-10 14:45:00 | 1456.50 | 1466.31 | 1461.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-10 15:15:00 | 1462.00 | 1465.44 | 1461.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-11 09:15:00 | 1461.30 | 1465.44 | 1461.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-11 09:15:00 | 1458.85 | 1464.13 | 1460.98 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-11 10:30:00 | 1467.50 | 1464.16 | 1461.28 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-11 11:30:00 | 1464.20 | 1464.33 | 1461.62 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-12 09:15:00 | 1438.20 | 1457.17 | 1459.28 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 44 — SELL (started 2024-12-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-12 09:15:00 | 1438.20 | 1457.17 | 1459.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-12 10:15:00 | 1434.50 | 1452.64 | 1457.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-13 14:15:00 | 1428.70 | 1428.45 | 1437.75 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-16 09:15:00 | 1426.85 | 1428.28 | 1436.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 09:15:00 | 1426.85 | 1428.28 | 1436.07 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-16 10:15:00 | 1420.75 | 1428.28 | 1436.07 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-16 12:00:00 | 1421.65 | 1425.22 | 1433.22 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-16 15:00:00 | 1421.95 | 1424.62 | 1430.96 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-17 10:45:00 | 1420.75 | 1422.62 | 1428.37 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-18 09:15:00 | 1407.10 | 1412.99 | 1420.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-18 09:30:00 | 1419.60 | 1412.99 | 1420.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-19 12:15:00 | 1402.90 | 1398.90 | 1406.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-19 12:45:00 | 1403.30 | 1398.90 | 1406.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-19 13:15:00 | 1406.45 | 1400.41 | 1406.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-19 14:00:00 | 1406.45 | 1400.41 | 1406.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-19 14:15:00 | 1406.45 | 1401.61 | 1406.24 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-19 15:15:00 | 1402.30 | 1401.61 | 1406.24 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-20 09:30:00 | 1398.75 | 1400.04 | 1404.73 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-20 12:45:00 | 1403.40 | 1403.31 | 1405.23 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-20 15:00:00 | 1394.30 | 1402.22 | 1404.44 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-23 10:15:00 | 1403.05 | 1401.98 | 1403.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-23 11:00:00 | 1403.05 | 1401.98 | 1403.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-23 11:15:00 | 1401.35 | 1401.85 | 1403.53 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-23 13:00:00 | 1396.55 | 1400.79 | 1402.89 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-24 09:15:00 | 1407.90 | 1403.77 | 1403.73 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 45 — BUY (started 2024-12-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-24 09:15:00 | 1407.90 | 1403.77 | 1403.73 | EMA200 above EMA400 |

### Cycle 46 — SELL (started 2024-12-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-24 10:15:00 | 1403.15 | 1403.65 | 1403.68 | EMA200 below EMA400 |

### Cycle 47 — BUY (started 2024-12-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-24 11:15:00 | 1406.40 | 1404.20 | 1403.93 | EMA200 above EMA400 |

### Cycle 48 — SELL (started 2024-12-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-24 14:15:00 | 1386.00 | 1401.20 | 1402.70 | EMA200 below EMA400 |

### Cycle 49 — BUY (started 2024-12-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-26 12:15:00 | 1408.50 | 1403.29 | 1403.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-26 14:15:00 | 1409.50 | 1405.30 | 1404.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-27 13:15:00 | 1406.00 | 1408.12 | 1406.41 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-27 13:15:00 | 1406.00 | 1408.12 | 1406.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-27 13:15:00 | 1406.00 | 1408.12 | 1406.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-27 13:45:00 | 1407.15 | 1408.12 | 1406.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-27 14:15:00 | 1404.85 | 1407.46 | 1406.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-27 14:45:00 | 1404.55 | 1407.46 | 1406.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-27 15:15:00 | 1400.80 | 1406.13 | 1405.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-30 09:15:00 | 1402.35 | 1406.13 | 1405.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-30 09:15:00 | 1404.55 | 1405.81 | 1405.66 | EMA400 retest candle locked (from upside) |

### Cycle 50 — SELL (started 2024-12-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-30 11:15:00 | 1403.70 | 1405.54 | 1405.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-30 12:15:00 | 1396.20 | 1403.67 | 1404.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-30 14:15:00 | 1404.40 | 1402.11 | 1403.72 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-30 14:15:00 | 1404.40 | 1402.11 | 1403.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-30 14:15:00 | 1404.40 | 1402.11 | 1403.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-30 15:00:00 | 1404.40 | 1402.11 | 1403.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-30 15:15:00 | 1401.95 | 1402.08 | 1403.56 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-31 09:15:00 | 1387.20 | 1402.08 | 1403.56 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-31 12:30:00 | 1390.00 | 1395.09 | 1399.27 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-31 13:30:00 | 1390.10 | 1394.27 | 1398.52 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-31 14:15:00 | 1390.00 | 1394.27 | 1398.52 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-31 14:15:00 | 1390.05 | 1393.43 | 1397.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-31 14:45:00 | 1397.80 | 1393.43 | 1397.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-01 09:15:00 | 1391.55 | 1392.66 | 1396.63 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-01-02 09:15:00 | 1402.00 | 1398.35 | 1397.93 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 51 — BUY (started 2025-01-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-02 09:15:00 | 1402.00 | 1398.35 | 1397.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-02 14:15:00 | 1423.20 | 1409.16 | 1403.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-06 10:15:00 | 1437.35 | 1439.83 | 1427.34 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-06 11:00:00 | 1437.35 | 1439.83 | 1427.34 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 11:15:00 | 1435.55 | 1438.97 | 1428.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 11:45:00 | 1427.60 | 1438.97 | 1428.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 14:15:00 | 1429.95 | 1436.01 | 1429.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 14:45:00 | 1433.80 | 1436.01 | 1429.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 15:15:00 | 1437.50 | 1436.31 | 1430.09 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-07 09:15:00 | 1446.65 | 1436.31 | 1430.09 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-20 13:15:00 | 1497.70 | 1508.33 | 1508.74 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 52 — SELL (started 2025-01-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-20 13:15:00 | 1497.70 | 1508.33 | 1508.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-21 09:15:00 | 1482.15 | 1500.24 | 1504.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-22 15:15:00 | 1464.95 | 1464.46 | 1475.89 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-23 09:15:00 | 1450.50 | 1464.46 | 1475.89 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-23 11:15:00 | 1457.00 | 1462.50 | 1472.95 | SELL ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-23 12:15:00 | 1457.10 | 1462.00 | 1471.77 | SELL ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-23 14:15:00 | 1456.85 | 1460.83 | 1469.50 | SELL ENTRY1 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-28 13:15:00 | 1424.30 | 1427.40 | 1435.11 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-28 15:00:00 | 1420.20 | 1425.96 | 1433.75 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-29 09:15:00 | 1438.25 | 1427.23 | 1432.90 | SL hit (close>ema400) qty=1.00 sl=1432.90 alert=retest1 |

### Cycle 53 — BUY (started 2025-01-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-29 12:15:00 | 1449.95 | 1437.26 | 1436.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-29 13:15:00 | 1457.55 | 1441.31 | 1438.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-01 10:15:00 | 1466.65 | 1477.91 | 1469.72 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-01 10:15:00 | 1466.65 | 1477.91 | 1469.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 10:15:00 | 1466.65 | 1477.91 | 1469.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-01 11:00:00 | 1466.65 | 1477.91 | 1469.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 11:15:00 | 1480.45 | 1478.42 | 1470.69 | EMA400 retest candle locked (from upside) |

### Cycle 54 — SELL (started 2025-02-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-01 12:15:00 | 1395.05 | 1461.74 | 1463.82 | EMA200 below EMA400 |

### Cycle 55 — BUY (started 2025-02-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-04 11:15:00 | 1465.45 | 1460.23 | 1459.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-04 13:15:00 | 1472.10 | 1463.19 | 1461.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-05 14:15:00 | 1470.50 | 1472.13 | 1468.07 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-05 14:15:00 | 1470.50 | 1472.13 | 1468.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-05 14:15:00 | 1470.50 | 1472.13 | 1468.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-05 15:00:00 | 1470.50 | 1472.13 | 1468.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-05 15:15:00 | 1466.05 | 1470.91 | 1467.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-06 09:15:00 | 1461.85 | 1470.91 | 1467.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 09:15:00 | 1465.15 | 1469.76 | 1467.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-06 09:30:00 | 1461.45 | 1469.76 | 1467.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 10:15:00 | 1469.35 | 1469.68 | 1467.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-06 10:30:00 | 1466.15 | 1469.68 | 1467.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 11:15:00 | 1476.30 | 1471.00 | 1468.57 | EMA400 retest candle locked (from upside) |

### Cycle 56 — SELL (started 2025-02-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-07 13:15:00 | 1465.00 | 1468.20 | 1468.35 | EMA200 below EMA400 |

### Cycle 57 — BUY (started 2025-02-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-07 14:15:00 | 1471.85 | 1468.93 | 1468.67 | EMA200 above EMA400 |

### Cycle 58 — SELL (started 2025-02-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-10 09:15:00 | 1465.00 | 1468.32 | 1468.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-10 10:15:00 | 1453.00 | 1465.25 | 1467.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-12 10:15:00 | 1446.05 | 1426.77 | 1437.27 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-12 10:15:00 | 1446.05 | 1426.77 | 1437.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 10:15:00 | 1446.05 | 1426.77 | 1437.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-12 11:00:00 | 1446.05 | 1426.77 | 1437.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 11:15:00 | 1458.25 | 1433.06 | 1439.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-12 11:45:00 | 1456.65 | 1433.06 | 1439.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 59 — BUY (started 2025-02-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-12 13:15:00 | 1457.55 | 1443.05 | 1442.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-13 10:15:00 | 1471.80 | 1453.89 | 1448.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-14 09:15:00 | 1463.65 | 1464.96 | 1457.67 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-14 10:00:00 | 1463.65 | 1464.96 | 1457.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-14 10:15:00 | 1458.80 | 1463.73 | 1457.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-14 11:00:00 | 1458.80 | 1463.73 | 1457.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-14 11:15:00 | 1456.50 | 1462.28 | 1457.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-14 12:00:00 | 1456.50 | 1462.28 | 1457.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-14 12:15:00 | 1452.00 | 1460.23 | 1457.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-14 12:30:00 | 1454.70 | 1460.23 | 1457.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-17 09:15:00 | 1472.50 | 1463.07 | 1459.38 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-17 10:30:00 | 1479.60 | 1465.09 | 1460.63 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-17 11:30:00 | 1479.80 | 1467.49 | 1462.13 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-17 13:15:00 | 1478.85 | 1468.99 | 1463.30 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-18 13:30:00 | 1479.55 | 1472.07 | 1468.18 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-19 09:15:00 | 1474.30 | 1473.41 | 1469.84 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-20 09:15:00 | 1483.00 | 1476.18 | 1472.78 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-20 10:15:00 | 1482.55 | 1477.00 | 1473.46 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-20 15:15:00 | 1470.00 | 1471.90 | 1472.06 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 60 — SELL (started 2025-02-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-20 15:15:00 | 1470.00 | 1471.90 | 1472.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-21 09:15:00 | 1466.65 | 1470.85 | 1471.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-21 10:15:00 | 1470.95 | 1470.87 | 1471.51 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-21 10:15:00 | 1470.95 | 1470.87 | 1471.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 10:15:00 | 1470.95 | 1470.87 | 1471.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-21 10:30:00 | 1473.00 | 1470.87 | 1471.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 61 — BUY (started 2025-02-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-21 11:15:00 | 1476.50 | 1471.99 | 1471.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-21 12:15:00 | 1480.45 | 1473.69 | 1472.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-21 15:15:00 | 1480.55 | 1480.80 | 1476.67 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-24 09:15:00 | 1482.30 | 1480.80 | 1476.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-24 09:15:00 | 1488.40 | 1482.32 | 1477.74 | EMA400 retest candle locked (from upside) |

### Cycle 62 — SELL (started 2025-02-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-25 11:15:00 | 1467.00 | 1477.18 | 1478.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-27 15:15:00 | 1462.05 | 1468.24 | 1471.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-03 09:15:00 | 1444.50 | 1439.24 | 1450.77 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-03 09:30:00 | 1446.60 | 1439.24 | 1450.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-03 10:15:00 | 1440.00 | 1439.39 | 1449.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-03 10:30:00 | 1445.90 | 1439.39 | 1449.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-05 13:15:00 | 1421.30 | 1407.40 | 1414.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-05 14:00:00 | 1421.30 | 1407.40 | 1414.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-05 14:15:00 | 1419.85 | 1409.89 | 1415.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-05 15:00:00 | 1419.85 | 1409.89 | 1415.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-05 15:15:00 | 1416.60 | 1411.23 | 1415.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-06 09:15:00 | 1402.00 | 1411.23 | 1415.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-06 09:15:00 | 1406.85 | 1410.36 | 1414.70 | EMA400 retest candle locked (from downside) |

### Cycle 63 — BUY (started 2025-03-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-06 14:15:00 | 1421.60 | 1416.33 | 1416.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-07 09:15:00 | 1434.10 | 1420.46 | 1418.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-07 11:15:00 | 1415.00 | 1421.35 | 1419.07 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-07 11:15:00 | 1415.00 | 1421.35 | 1419.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-07 11:15:00 | 1415.00 | 1421.35 | 1419.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-07 12:00:00 | 1415.00 | 1421.35 | 1419.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-07 12:15:00 | 1409.05 | 1418.89 | 1418.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-07 13:00:00 | 1409.05 | 1418.89 | 1418.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 64 — SELL (started 2025-03-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-07 13:15:00 | 1410.20 | 1417.15 | 1417.43 | EMA200 below EMA400 |

### Cycle 65 — BUY (started 2025-03-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-10 12:15:00 | 1425.30 | 1417.84 | 1417.32 | EMA200 above EMA400 |

### Cycle 66 — SELL (started 2025-03-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-11 12:15:00 | 1413.25 | 1417.00 | 1417.26 | EMA200 below EMA400 |

### Cycle 67 — BUY (started 2025-03-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-11 13:15:00 | 1419.75 | 1417.55 | 1417.49 | EMA200 above EMA400 |

### Cycle 68 — SELL (started 2025-03-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-12 09:15:00 | 1409.45 | 1415.87 | 1416.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-13 10:15:00 | 1402.25 | 1409.13 | 1412.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-17 09:15:00 | 1423.05 | 1401.04 | 1405.31 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-17 09:15:00 | 1423.05 | 1401.04 | 1405.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 09:15:00 | 1423.05 | 1401.04 | 1405.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-17 09:30:00 | 1428.35 | 1401.04 | 1405.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 10:15:00 | 1425.00 | 1405.84 | 1407.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-17 10:30:00 | 1424.95 | 1405.84 | 1407.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 69 — BUY (started 2025-03-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-17 11:15:00 | 1436.45 | 1411.96 | 1409.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-18 09:15:00 | 1448.95 | 1430.52 | 1420.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-25 11:15:00 | 1563.65 | 1567.58 | 1546.93 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-25 12:00:00 | 1563.65 | 1567.58 | 1546.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-26 09:15:00 | 1557.55 | 1561.34 | 1551.39 | EMA400 retest candle locked (from upside) |

### Cycle 70 — SELL (started 2025-03-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-26 15:15:00 | 1545.00 | 1546.90 | 1547.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-27 09:15:00 | 1538.50 | 1545.22 | 1546.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-27 13:15:00 | 1551.00 | 1544.04 | 1545.17 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-27 13:15:00 | 1551.00 | 1544.04 | 1545.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 13:15:00 | 1551.00 | 1544.04 | 1545.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-27 14:00:00 | 1551.00 | 1544.04 | 1545.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 14:15:00 | 1547.20 | 1544.67 | 1545.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-27 14:30:00 | 1548.95 | 1544.67 | 1545.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 15:15:00 | 1548.25 | 1545.39 | 1545.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-28 09:15:00 | 1558.70 | 1545.39 | 1545.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 71 — BUY (started 2025-03-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-28 09:15:00 | 1557.10 | 1547.73 | 1546.66 | EMA200 above EMA400 |

### Cycle 72 — SELL (started 2025-03-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-28 11:15:00 | 1537.05 | 1544.69 | 1545.41 | EMA200 below EMA400 |

### Cycle 73 — BUY (started 2025-03-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-28 13:15:00 | 1554.75 | 1547.31 | 1546.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-01 09:15:00 | 1559.60 | 1550.25 | 1548.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-01 10:15:00 | 1543.45 | 1548.89 | 1547.66 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-01 10:15:00 | 1543.45 | 1548.89 | 1547.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-01 10:15:00 | 1543.45 | 1548.89 | 1547.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-01 11:00:00 | 1543.45 | 1548.89 | 1547.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-01 11:15:00 | 1544.65 | 1548.04 | 1547.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-01 11:45:00 | 1542.75 | 1548.04 | 1547.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-01 12:15:00 | 1548.05 | 1548.04 | 1547.45 | EMA400 retest candle locked (from upside) |

### Cycle 74 — SELL (started 2025-04-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-01 13:15:00 | 1543.00 | 1547.03 | 1547.04 | EMA200 below EMA400 |

### Cycle 75 — BUY (started 2025-04-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-02 09:15:00 | 1554.75 | 1547.50 | 1547.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-02 14:15:00 | 1561.45 | 1555.02 | 1551.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-03 09:15:00 | 1547.00 | 1553.62 | 1551.48 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-03 09:15:00 | 1547.00 | 1553.62 | 1551.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-03 09:15:00 | 1547.00 | 1553.62 | 1551.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-03 10:00:00 | 1547.00 | 1553.62 | 1551.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-03 10:15:00 | 1546.25 | 1552.14 | 1551.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-03 10:45:00 | 1544.70 | 1552.14 | 1551.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-03 12:15:00 | 1550.95 | 1551.02 | 1550.64 | EMA400 retest candle locked (from upside) |

### Cycle 76 — SELL (started 2025-04-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-03 14:15:00 | 1542.15 | 1549.12 | 1549.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-04 09:15:00 | 1515.50 | 1541.42 | 1546.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-08 13:15:00 | 1470.10 | 1469.51 | 1486.05 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-08 14:00:00 | 1470.10 | 1469.51 | 1486.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 14:15:00 | 1487.90 | 1473.19 | 1486.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-08 14:45:00 | 1492.15 | 1473.19 | 1486.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 15:15:00 | 1493.50 | 1477.25 | 1486.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-09 09:15:00 | 1494.35 | 1477.25 | 1486.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-09 10:15:00 | 1484.70 | 1480.54 | 1486.82 | EMA400 retest candle locked (from downside) |

### Cycle 77 — BUY (started 2025-04-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-11 09:15:00 | 1523.45 | 1491.72 | 1489.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-15 09:15:00 | 1553.80 | 1523.24 | 1509.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-17 09:15:00 | 1555.00 | 1562.17 | 1550.14 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-17 09:15:00 | 1555.00 | 1562.17 | 1550.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-17 09:15:00 | 1555.00 | 1562.17 | 1550.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-17 09:30:00 | 1559.20 | 1562.17 | 1550.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-17 10:15:00 | 1555.50 | 1560.84 | 1550.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-17 11:00:00 | 1555.50 | 1560.84 | 1550.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 09:15:00 | 1608.00 | 1617.01 | 1607.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-23 09:45:00 | 1606.40 | 1617.01 | 1607.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 10:15:00 | 1605.70 | 1614.75 | 1607.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-23 10:30:00 | 1605.80 | 1614.75 | 1607.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 11:15:00 | 1610.20 | 1613.84 | 1607.75 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-23 12:45:00 | 1613.60 | 1613.35 | 1608.08 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-23 13:30:00 | 1614.20 | 1613.48 | 1608.62 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-23 15:00:00 | 1614.20 | 1613.62 | 1609.13 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-24 15:15:00 | 1615.00 | 1615.63 | 1613.32 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-24 15:15:00 | 1615.00 | 1615.50 | 1613.47 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-25 09:15:00 | 1697.70 | 1615.50 | 1613.47 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-05-02 09:15:00 | 1774.96 | 1755.76 | 1733.46 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 78 — SELL (started 2025-05-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-06 13:15:00 | 1731.00 | 1750.78 | 1753.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-09 10:15:00 | 1709.10 | 1727.62 | 1735.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-12 09:15:00 | 1745.20 | 1716.75 | 1724.66 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-12 09:15:00 | 1745.20 | 1716.75 | 1724.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-12 09:15:00 | 1745.20 | 1716.75 | 1724.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-12 10:00:00 | 1745.20 | 1716.75 | 1724.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-12 10:15:00 | 1747.50 | 1722.90 | 1726.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-12 11:00:00 | 1747.50 | 1722.90 | 1726.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 79 — BUY (started 2025-05-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 12:15:00 | 1747.10 | 1731.40 | 1730.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 14:15:00 | 1750.90 | 1738.28 | 1733.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-13 10:15:00 | 1734.20 | 1739.53 | 1735.60 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-13 10:15:00 | 1734.20 | 1739.53 | 1735.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-13 10:15:00 | 1734.20 | 1739.53 | 1735.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-13 11:00:00 | 1734.20 | 1739.53 | 1735.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-13 11:15:00 | 1749.80 | 1741.58 | 1736.89 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-13 12:15:00 | 1753.90 | 1741.58 | 1736.89 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-14 11:15:00 | 1753.10 | 1745.73 | 1741.30 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-14 12:15:00 | 1752.20 | 1746.75 | 1742.17 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-15 10:30:00 | 1755.60 | 1749.23 | 1745.96 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-15 11:15:00 | 1758.90 | 1751.16 | 1747.14 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-15 13:00:00 | 1772.40 | 1755.41 | 1749.43 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-20 14:15:00 | 1759.60 | 1766.43 | 1767.22 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 80 — SELL (started 2025-05-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-20 14:15:00 | 1759.60 | 1766.43 | 1767.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-22 09:15:00 | 1752.50 | 1763.11 | 1764.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-22 14:15:00 | 1758.30 | 1754.84 | 1759.36 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-22 14:15:00 | 1758.30 | 1754.84 | 1759.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 14:15:00 | 1758.30 | 1754.84 | 1759.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-22 15:00:00 | 1758.30 | 1754.84 | 1759.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 15:15:00 | 1762.00 | 1756.27 | 1759.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-23 09:15:00 | 1782.70 | 1756.27 | 1759.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 09:15:00 | 1783.40 | 1761.70 | 1761.76 | EMA400 retest candle locked (from downside) |

### Cycle 81 — BUY (started 2025-05-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-23 10:15:00 | 1792.00 | 1767.76 | 1764.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-23 11:15:00 | 1800.00 | 1774.21 | 1767.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-29 10:15:00 | 1806.40 | 1807.14 | 1802.76 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-29 10:30:00 | 1804.70 | 1807.14 | 1802.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 13:15:00 | 1807.50 | 1817.64 | 1813.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-30 14:00:00 | 1807.50 | 1817.64 | 1813.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 14:15:00 | 1815.10 | 1817.13 | 1813.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-30 15:00:00 | 1815.10 | 1817.13 | 1813.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 15:15:00 | 1809.00 | 1815.50 | 1813.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-02 09:15:00 | 1824.10 | 1815.50 | 1813.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 09:15:00 | 1815.10 | 1815.42 | 1813.23 | EMA400 retest candle locked (from upside) |

### Cycle 82 — SELL (started 2025-06-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-02 13:15:00 | 1809.00 | 1811.67 | 1812.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-03 09:15:00 | 1786.80 | 1803.87 | 1808.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-04 14:15:00 | 1776.60 | 1775.65 | 1785.63 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-04 15:00:00 | 1776.60 | 1775.65 | 1785.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 11:15:00 | 1787.40 | 1777.12 | 1783.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-05 12:00:00 | 1787.40 | 1777.12 | 1783.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 12:15:00 | 1785.70 | 1778.84 | 1783.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-05 12:45:00 | 1785.80 | 1778.84 | 1783.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 10:15:00 | 1770.60 | 1773.42 | 1778.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-06 10:30:00 | 1777.00 | 1773.42 | 1778.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 11:15:00 | 1774.90 | 1773.71 | 1778.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-06 11:45:00 | 1779.90 | 1773.71 | 1778.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 14:15:00 | 1781.00 | 1774.03 | 1777.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-06 15:00:00 | 1781.00 | 1774.03 | 1777.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 15:15:00 | 1777.70 | 1774.76 | 1777.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-09 09:15:00 | 1768.00 | 1774.76 | 1777.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-09 10:15:00 | 1792.00 | 1778.49 | 1778.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-09 11:00:00 | 1792.00 | 1778.49 | 1778.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 83 — BUY (started 2025-06-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-09 11:15:00 | 1794.70 | 1781.73 | 1780.08 | EMA200 above EMA400 |

### Cycle 84 — SELL (started 2025-06-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-10 11:15:00 | 1773.00 | 1781.11 | 1781.35 | EMA200 below EMA400 |

### Cycle 85 — BUY (started 2025-06-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-11 09:15:00 | 1785.10 | 1781.34 | 1781.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-11 10:15:00 | 1791.20 | 1783.31 | 1782.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-12 11:15:00 | 1788.80 | 1793.59 | 1789.56 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-12 11:15:00 | 1788.80 | 1793.59 | 1789.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 11:15:00 | 1788.80 | 1793.59 | 1789.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-12 12:00:00 | 1788.80 | 1793.59 | 1789.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 12:15:00 | 1787.90 | 1792.45 | 1789.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-12 13:00:00 | 1787.90 | 1792.45 | 1789.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 13:15:00 | 1769.50 | 1787.86 | 1787.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-12 14:00:00 | 1769.50 | 1787.86 | 1787.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 86 — SELL (started 2025-06-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-12 14:15:00 | 1765.50 | 1783.39 | 1785.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-12 15:15:00 | 1761.30 | 1778.97 | 1783.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-13 13:15:00 | 1761.50 | 1756.71 | 1768.58 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-13 14:00:00 | 1761.50 | 1756.71 | 1768.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 09:15:00 | 1772.00 | 1759.74 | 1767.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 10:15:00 | 1781.00 | 1759.74 | 1767.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 10:15:00 | 1793.30 | 1766.45 | 1769.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 11:00:00 | 1793.30 | 1766.45 | 1769.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 87 — BUY (started 2025-06-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-16 11:15:00 | 1795.50 | 1772.26 | 1771.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-16 14:15:00 | 1799.10 | 1784.55 | 1778.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-18 11:15:00 | 1798.50 | 1798.71 | 1792.35 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-18 11:30:00 | 1796.00 | 1798.71 | 1792.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 15:15:00 | 1791.50 | 1796.82 | 1793.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-19 09:15:00 | 1800.00 | 1796.82 | 1793.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 09:15:00 | 1801.00 | 1797.65 | 1794.10 | EMA400 retest candle locked (from upside) |

### Cycle 88 — SELL (started 2025-06-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-19 14:15:00 | 1789.20 | 1792.21 | 1792.53 | EMA200 below EMA400 |

### Cycle 89 — BUY (started 2025-06-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-20 09:15:00 | 1807.70 | 1794.96 | 1793.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-20 10:15:00 | 1816.30 | 1799.23 | 1795.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-23 09:15:00 | 1801.50 | 1806.68 | 1802.07 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-23 09:15:00 | 1801.50 | 1806.68 | 1802.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 09:15:00 | 1801.50 | 1806.68 | 1802.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-23 09:45:00 | 1801.30 | 1806.68 | 1802.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 10:15:00 | 1815.90 | 1808.53 | 1803.33 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-23 11:45:00 | 1818.60 | 1810.26 | 1804.59 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-23 12:15:00 | 1819.40 | 1810.26 | 1804.59 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-24 09:15:00 | 1830.40 | 1815.96 | 1809.52 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-27 10:00:00 | 1822.70 | 1848.62 | 1843.72 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 10:15:00 | 1847.90 | 1848.48 | 1844.10 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-27 14:30:00 | 1862.80 | 1849.05 | 1845.95 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-30 14:15:00 | 1838.80 | 1844.56 | 1845.12 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 90 — SELL (started 2025-06-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-30 14:15:00 | 1838.80 | 1844.56 | 1845.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-30 15:15:00 | 1833.90 | 1842.43 | 1844.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-01 09:15:00 | 1843.30 | 1842.61 | 1844.03 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-01 09:15:00 | 1843.30 | 1842.61 | 1844.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 09:15:00 | 1843.30 | 1842.61 | 1844.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-01 09:45:00 | 1852.90 | 1842.61 | 1844.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 10:15:00 | 1834.70 | 1841.02 | 1843.18 | EMA400 retest candle locked (from downside) |

### Cycle 91 — BUY (started 2025-07-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-01 13:15:00 | 1859.00 | 1846.90 | 1845.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-01 14:15:00 | 1863.60 | 1850.24 | 1847.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-02 14:15:00 | 1857.90 | 1860.46 | 1855.20 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-02 14:15:00 | 1857.90 | 1860.46 | 1855.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 14:15:00 | 1857.90 | 1860.46 | 1855.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-02 14:30:00 | 1855.90 | 1860.46 | 1855.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 15:15:00 | 1853.10 | 1858.98 | 1855.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-03 09:15:00 | 1841.80 | 1858.98 | 1855.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 09:15:00 | 1837.00 | 1854.59 | 1853.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-03 10:00:00 | 1837.00 | 1854.59 | 1853.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 92 — SELL (started 2025-07-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-03 10:15:00 | 1831.00 | 1849.87 | 1851.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-03 11:15:00 | 1817.60 | 1843.42 | 1848.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-07 12:15:00 | 1803.30 | 1799.75 | 1810.36 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-07 13:00:00 | 1803.30 | 1799.75 | 1810.36 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 15:15:00 | 1808.90 | 1803.49 | 1809.57 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-08 09:15:00 | 1805.20 | 1803.49 | 1809.57 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-08 10:15:00 | 1816.20 | 1806.55 | 1809.94 | SL hit (close>static) qty=1.00 sl=1811.80 alert=retest2 |

### Cycle 93 — BUY (started 2025-07-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-08 15:15:00 | 1816.20 | 1811.74 | 1811.42 | EMA200 above EMA400 |

### Cycle 94 — SELL (started 2025-07-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-10 09:15:00 | 1807.00 | 1812.16 | 1812.23 | EMA200 below EMA400 |

### Cycle 95 — BUY (started 2025-07-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-10 13:15:00 | 1818.30 | 1813.03 | 1812.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-11 09:15:00 | 1835.90 | 1817.29 | 1814.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-15 11:15:00 | 1836.10 | 1841.04 | 1834.44 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-15 12:00:00 | 1836.10 | 1841.04 | 1834.44 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 12:15:00 | 1828.40 | 1838.51 | 1833.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-15 13:00:00 | 1828.40 | 1838.51 | 1833.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 13:15:00 | 1838.60 | 1838.53 | 1834.32 | EMA400 retest candle locked (from upside) |

### Cycle 96 — SELL (started 2025-07-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-16 10:15:00 | 1826.80 | 1831.59 | 1831.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-16 11:15:00 | 1823.80 | 1830.03 | 1831.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-16 12:15:00 | 1830.20 | 1830.07 | 1831.02 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-16 12:15:00 | 1830.20 | 1830.07 | 1831.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 12:15:00 | 1830.20 | 1830.07 | 1831.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-16 12:45:00 | 1830.10 | 1830.07 | 1831.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 13:15:00 | 1837.10 | 1831.47 | 1831.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-16 14:00:00 | 1837.10 | 1831.47 | 1831.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 14:15:00 | 1829.60 | 1831.10 | 1831.39 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-17 09:15:00 | 1817.00 | 1830.64 | 1831.16 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-22 12:15:00 | 1805.50 | 1799.91 | 1799.43 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 97 — BUY (started 2025-07-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-22 12:15:00 | 1805.50 | 1799.91 | 1799.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-23 12:15:00 | 1813.10 | 1806.53 | 1803.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-23 15:15:00 | 1805.80 | 1808.33 | 1805.31 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-24 09:15:00 | 1803.50 | 1808.33 | 1805.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 09:15:00 | 1794.10 | 1805.49 | 1804.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-24 10:00:00 | 1794.10 | 1805.49 | 1804.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 98 — SELL (started 2025-07-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-24 10:15:00 | 1792.50 | 1802.89 | 1803.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-24 11:15:00 | 1785.70 | 1799.45 | 1801.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-24 15:15:00 | 1819.90 | 1798.77 | 1800.11 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-24 15:15:00 | 1819.90 | 1798.77 | 1800.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 15:15:00 | 1819.90 | 1798.77 | 1800.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-25 09:15:00 | 1836.00 | 1798.77 | 1800.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 99 — BUY (started 2025-07-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-25 09:15:00 | 1837.10 | 1806.43 | 1803.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-28 10:15:00 | 1848.50 | 1834.36 | 1822.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-29 09:15:00 | 1834.00 | 1842.03 | 1832.20 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-29 09:15:00 | 1834.00 | 1842.03 | 1832.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 09:15:00 | 1834.00 | 1842.03 | 1832.20 | EMA400 retest candle locked (from upside) |

### Cycle 100 — SELL (started 2025-08-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-01 11:15:00 | 1808.40 | 1832.02 | 1834.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-01 12:15:00 | 1802.10 | 1826.03 | 1831.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-04 10:15:00 | 1813.50 | 1809.99 | 1820.31 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-04 11:00:00 | 1813.50 | 1809.99 | 1820.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 12:15:00 | 1826.50 | 1813.87 | 1820.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 13:00:00 | 1826.50 | 1813.87 | 1820.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 13:15:00 | 1832.50 | 1817.60 | 1821.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 14:00:00 | 1832.50 | 1817.60 | 1821.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 101 — BUY (started 2025-08-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-05 10:15:00 | 1836.40 | 1825.42 | 1824.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-05 11:15:00 | 1841.30 | 1828.60 | 1825.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-07 09:15:00 | 1847.00 | 1852.41 | 1845.45 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-07 09:15:00 | 1847.00 | 1852.41 | 1845.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 09:15:00 | 1847.00 | 1852.41 | 1845.45 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-07 14:30:00 | 1860.40 | 1851.25 | 1847.02 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-08 10:15:00 | 1856.40 | 1853.24 | 1848.74 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-08 13:15:00 | 1834.80 | 1846.13 | 1846.60 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 102 — SELL (started 2025-08-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-08 13:15:00 | 1834.80 | 1846.13 | 1846.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-11 11:15:00 | 1829.60 | 1838.79 | 1842.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-11 13:15:00 | 1848.60 | 1840.16 | 1842.35 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-11 13:15:00 | 1848.60 | 1840.16 | 1842.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 13:15:00 | 1848.60 | 1840.16 | 1842.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-11 14:00:00 | 1848.60 | 1840.16 | 1842.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 14:15:00 | 1850.50 | 1842.23 | 1843.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-11 15:00:00 | 1850.50 | 1842.23 | 1843.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 103 — BUY (started 2025-08-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-11 15:15:00 | 1849.50 | 1843.68 | 1843.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-12 09:15:00 | 1855.10 | 1845.97 | 1844.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-12 10:15:00 | 1840.80 | 1844.93 | 1844.36 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-12 10:15:00 | 1840.80 | 1844.93 | 1844.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 10:15:00 | 1840.80 | 1844.93 | 1844.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-12 11:00:00 | 1840.80 | 1844.93 | 1844.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 11:15:00 | 1843.90 | 1844.73 | 1844.31 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-12 12:15:00 | 1848.50 | 1844.73 | 1844.31 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-12 14:15:00 | 1837.20 | 1842.83 | 1843.54 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 104 — SELL (started 2025-08-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-12 14:15:00 | 1837.20 | 1842.83 | 1843.54 | EMA200 below EMA400 |

### Cycle 105 — BUY (started 2025-08-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-13 09:15:00 | 1852.00 | 1844.84 | 1844.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-13 11:15:00 | 1856.50 | 1847.91 | 1845.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-13 13:15:00 | 1848.70 | 1849.24 | 1846.90 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-13 13:15:00 | 1848.70 | 1849.24 | 1846.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 13:15:00 | 1848.70 | 1849.24 | 1846.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-13 14:00:00 | 1848.70 | 1849.24 | 1846.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 14:15:00 | 1840.90 | 1847.57 | 1846.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-13 15:00:00 | 1840.90 | 1847.57 | 1846.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 15:15:00 | 1846.10 | 1847.28 | 1846.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-14 09:15:00 | 1847.90 | 1847.28 | 1846.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 09:15:00 | 1842.60 | 1846.34 | 1845.99 | EMA400 retest candle locked (from upside) |

### Cycle 106 — SELL (started 2025-08-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-14 10:15:00 | 1840.40 | 1845.15 | 1845.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-14 11:15:00 | 1836.10 | 1843.34 | 1844.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-14 13:15:00 | 1843.40 | 1842.91 | 1844.19 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-14 13:15:00 | 1843.40 | 1842.91 | 1844.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 13:15:00 | 1843.40 | 1842.91 | 1844.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-14 13:45:00 | 1844.90 | 1842.91 | 1844.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 14:15:00 | 1841.30 | 1842.59 | 1843.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-14 14:45:00 | 1842.60 | 1842.59 | 1843.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 107 — BUY (started 2025-08-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-18 09:15:00 | 1905.50 | 1855.29 | 1849.47 | EMA200 above EMA400 |

### Cycle 108 — SELL (started 2025-08-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-19 12:15:00 | 1852.80 | 1857.55 | 1857.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-20 10:15:00 | 1849.00 | 1853.63 | 1855.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-20 14:15:00 | 1856.90 | 1852.90 | 1854.53 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-20 14:15:00 | 1856.90 | 1852.90 | 1854.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 14:15:00 | 1856.90 | 1852.90 | 1854.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-20 15:00:00 | 1856.90 | 1852.90 | 1854.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 15:15:00 | 1862.00 | 1854.72 | 1855.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-21 09:15:00 | 1880.00 | 1854.72 | 1855.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 109 — BUY (started 2025-08-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-21 09:15:00 | 1885.50 | 1860.88 | 1857.96 | EMA200 above EMA400 |

### Cycle 110 — SELL (started 2025-08-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-22 15:15:00 | 1857.00 | 1863.44 | 1864.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-25 09:15:00 | 1846.00 | 1859.95 | 1862.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-28 09:15:00 | 1833.40 | 1830.51 | 1840.96 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-28 10:00:00 | 1833.40 | 1830.51 | 1840.96 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 11:15:00 | 1832.30 | 1832.19 | 1840.00 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-28 12:15:00 | 1829.60 | 1832.19 | 1840.00 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-28 14:30:00 | 1824.70 | 1828.44 | 1836.34 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-04 09:15:00 | 1830.50 | 1814.03 | 1813.35 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 111 — BUY (started 2025-09-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-04 09:15:00 | 1830.50 | 1814.03 | 1813.35 | EMA200 above EMA400 |

### Cycle 112 — SELL (started 2025-09-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-04 11:15:00 | 1785.00 | 1807.69 | 1810.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-04 13:15:00 | 1781.70 | 1801.20 | 1807.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-05 09:15:00 | 1806.60 | 1799.69 | 1804.62 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-05 09:15:00 | 1806.60 | 1799.69 | 1804.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 09:15:00 | 1806.60 | 1799.69 | 1804.62 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-08 09:15:00 | 1779.70 | 1806.36 | 1806.51 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-09 11:45:00 | 1799.00 | 1790.21 | 1793.95 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-09 13:15:00 | 1792.10 | 1792.53 | 1794.67 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-09 15:15:00 | 1810.00 | 1798.19 | 1796.86 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 113 — BUY (started 2025-09-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-09 15:15:00 | 1810.00 | 1798.19 | 1796.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-10 09:15:00 | 1826.50 | 1803.85 | 1799.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-11 09:15:00 | 1821.60 | 1824.57 | 1814.80 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-11 09:15:00 | 1821.60 | 1824.57 | 1814.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 09:15:00 | 1821.60 | 1824.57 | 1814.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-11 09:45:00 | 1820.90 | 1824.57 | 1814.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 11:15:00 | 1815.50 | 1822.03 | 1815.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-11 11:30:00 | 1813.60 | 1822.03 | 1815.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 12:15:00 | 1806.70 | 1818.96 | 1814.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-11 13:00:00 | 1806.70 | 1818.96 | 1814.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 13:15:00 | 1813.40 | 1817.85 | 1814.41 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-11 14:30:00 | 1817.20 | 1817.04 | 1814.35 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-12 10:45:00 | 1819.00 | 1815.97 | 1814.40 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-16 14:45:00 | 1816.50 | 1820.97 | 1820.78 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-16 15:15:00 | 1819.10 | 1820.59 | 1820.63 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 114 — SELL (started 2025-09-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-16 15:15:00 | 1819.10 | 1820.59 | 1820.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-17 10:15:00 | 1814.60 | 1819.43 | 1820.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-18 10:15:00 | 1821.10 | 1811.39 | 1814.44 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-18 10:15:00 | 1821.10 | 1811.39 | 1814.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 10:15:00 | 1821.10 | 1811.39 | 1814.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-18 11:00:00 | 1821.10 | 1811.39 | 1814.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 11:15:00 | 1827.90 | 1814.69 | 1815.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-18 12:00:00 | 1827.90 | 1814.69 | 1815.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 12:15:00 | 1818.50 | 1815.45 | 1815.92 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-18 13:45:00 | 1816.00 | 1816.06 | 1816.16 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-18 14:15:00 | 1822.80 | 1817.41 | 1816.76 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 115 — BUY (started 2025-09-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-18 14:15:00 | 1822.80 | 1817.41 | 1816.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-19 09:15:00 | 1840.00 | 1822.82 | 1819.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-22 15:15:00 | 1855.60 | 1855.83 | 1844.74 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-23 09:15:00 | 1847.10 | 1855.83 | 1844.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 09:15:00 | 1855.50 | 1855.76 | 1845.72 | EMA400 retest candle locked (from upside) |

### Cycle 116 — SELL (started 2025-09-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-23 14:15:00 | 1822.00 | 1837.84 | 1839.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-24 09:15:00 | 1810.70 | 1829.60 | 1835.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-26 09:15:00 | 1813.40 | 1813.14 | 1818.88 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-26 09:15:00 | 1813.40 | 1813.14 | 1818.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 09:15:00 | 1813.40 | 1813.14 | 1818.88 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-26 12:15:00 | 1803.00 | 1811.55 | 1817.16 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-07 14:15:00 | 1782.30 | 1779.19 | 1779.08 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 117 — BUY (started 2025-10-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-07 14:15:00 | 1782.30 | 1779.19 | 1779.08 | EMA200 above EMA400 |

### Cycle 118 — SELL (started 2025-10-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-08 10:15:00 | 1772.10 | 1778.54 | 1778.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-08 11:15:00 | 1762.30 | 1775.29 | 1777.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-08 14:15:00 | 1773.10 | 1771.81 | 1775.05 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-08 14:45:00 | 1772.90 | 1771.81 | 1775.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 15:15:00 | 1772.00 | 1771.85 | 1774.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-09 09:15:00 | 1777.50 | 1771.85 | 1774.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 09:15:00 | 1771.90 | 1771.86 | 1774.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-09 09:30:00 | 1772.80 | 1771.86 | 1774.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 10:15:00 | 1782.60 | 1774.01 | 1775.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-09 11:00:00 | 1782.60 | 1774.01 | 1775.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 119 — BUY (started 2025-10-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-09 11:15:00 | 1790.50 | 1777.31 | 1776.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-09 12:15:00 | 1799.70 | 1781.79 | 1778.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-10 13:15:00 | 1812.70 | 1815.81 | 1801.91 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-10 14:00:00 | 1812.70 | 1815.81 | 1801.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 09:15:00 | 1821.20 | 1814.98 | 1804.82 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-13 10:15:00 | 1822.30 | 1814.98 | 1804.82 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-15 09:15:00 | 1841.10 | 1814.05 | 1811.40 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-16 11:00:00 | 1825.90 | 1831.12 | 1825.89 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Target hit | 2025-11-10 09:15:00 | 2004.53 | 1992.45 | 1982.96 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 120 — SELL (started 2025-11-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-13 10:15:00 | 1981.20 | 1992.01 | 1992.09 | EMA200 below EMA400 |

### Cycle 121 — BUY (started 2025-11-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-14 14:15:00 | 2001.30 | 1992.05 | 1991.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-19 12:15:00 | 2005.50 | 1999.68 | 1997.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-19 15:15:00 | 2000.20 | 2001.25 | 1998.86 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-20 09:15:00 | 2004.30 | 2001.25 | 1998.86 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 09:15:00 | 2018.40 | 2020.26 | 2015.65 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-24 15:15:00 | 2075.00 | 2016.84 | 2015.49 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-25 15:00:00 | 2029.30 | 2025.84 | 2022.48 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-26 14:30:00 | 2029.90 | 2030.45 | 2027.22 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-27 10:15:00 | 2014.20 | 2023.94 | 2024.83 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 122 — SELL (started 2025-11-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-27 10:15:00 | 2014.20 | 2023.94 | 2024.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-27 12:15:00 | 2007.20 | 2018.62 | 2022.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-01 14:15:00 | 1971.40 | 1968.08 | 1981.78 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-01 15:00:00 | 1971.40 | 1968.08 | 1981.78 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 09:15:00 | 1960.80 | 1967.57 | 1979.23 | EMA400 retest candle locked (from downside) |

### Cycle 123 — BUY (started 2025-12-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-04 10:15:00 | 2000.60 | 1979.14 | 1977.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-04 11:15:00 | 2007.30 | 1984.77 | 1980.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-05 15:15:00 | 2018.80 | 2020.54 | 2007.17 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-08 09:15:00 | 2025.90 | 2020.54 | 2007.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 09:15:00 | 2028.00 | 2022.03 | 2009.06 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-08 11:30:00 | 2035.90 | 2024.67 | 2012.58 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-09 09:15:00 | 1997.20 | 2018.92 | 2014.57 | SL hit (close<static) qty=1.00 sl=2005.00 alert=retest2 |

### Cycle 124 — SELL (started 2025-12-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-09 12:15:00 | 1999.70 | 2011.19 | 2011.75 | EMA200 below EMA400 |

### Cycle 125 — BUY (started 2025-12-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-10 11:15:00 | 2023.80 | 2010.38 | 2010.32 | EMA200 above EMA400 |

### Cycle 126 — SELL (started 2025-12-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-11 12:15:00 | 2007.00 | 2010.98 | 2011.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-11 15:15:00 | 1997.10 | 2008.45 | 2010.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-12 09:15:00 | 2012.60 | 2009.28 | 2010.32 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-12 09:15:00 | 2012.60 | 2009.28 | 2010.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 09:15:00 | 2012.60 | 2009.28 | 2010.32 | EMA400 retest candle locked (from downside) |

### Cycle 127 — BUY (started 2025-12-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-12 12:15:00 | 2013.30 | 2011.36 | 2011.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-12 13:15:00 | 2025.50 | 2014.19 | 2012.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-16 14:15:00 | 2035.90 | 2038.95 | 2031.36 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-16 15:00:00 | 2035.90 | 2038.95 | 2031.36 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 15:15:00 | 2040.00 | 2039.16 | 2032.15 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-17 09:15:00 | 2043.10 | 2039.16 | 2032.15 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-17 09:15:00 | 2012.60 | 2033.85 | 2030.37 | SL hit (close<static) qty=1.00 sl=2021.00 alert=retest2 |

### Cycle 128 — SELL (started 2025-12-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-17 11:15:00 | 1999.60 | 2024.13 | 2026.38 | EMA200 below EMA400 |

### Cycle 129 — BUY (started 2025-12-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-19 12:15:00 | 2027.20 | 2019.60 | 2018.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-22 09:15:00 | 2029.90 | 2024.61 | 2021.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-22 10:15:00 | 2020.00 | 2023.69 | 2021.60 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-22 10:15:00 | 2020.00 | 2023.69 | 2021.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-22 10:15:00 | 2020.00 | 2023.69 | 2021.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-22 11:00:00 | 2020.00 | 2023.69 | 2021.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-22 11:15:00 | 2016.90 | 2022.33 | 2021.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-22 11:45:00 | 2017.80 | 2022.33 | 2021.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-22 14:15:00 | 2021.90 | 2020.93 | 2020.72 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-22 15:15:00 | 2024.80 | 2020.93 | 2020.72 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-23 14:45:00 | 2023.40 | 2025.88 | 2023.84 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-26 09:15:00 | 2023.20 | 2025.76 | 2024.96 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-26 10:15:00 | 2006.50 | 2021.27 | 2023.02 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 130 — SELL (started 2025-12-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-26 10:15:00 | 2006.50 | 2021.27 | 2023.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-30 09:15:00 | 2000.00 | 2011.60 | 2015.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-30 14:15:00 | 1996.80 | 1995.54 | 2004.96 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-30 15:00:00 | 1996.80 | 1995.54 | 2004.96 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 15:15:00 | 2019.70 | 2000.37 | 2006.30 | EMA400 retest candle locked (from downside) |

### Cycle 131 — BUY (started 2025-12-31 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-31 12:15:00 | 2038.40 | 2013.02 | 2010.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-01 15:15:00 | 2044.00 | 2035.64 | 2027.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-05 13:15:00 | 2073.20 | 2076.25 | 2061.10 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-05 14:00:00 | 2073.20 | 2076.25 | 2061.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 09:15:00 | 2087.90 | 2079.19 | 2066.25 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-06 12:00:00 | 2097.10 | 2083.81 | 2070.66 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-07 09:15:00 | 2093.30 | 2091.01 | 2078.89 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-09 12:15:00 | 2071.40 | 2079.80 | 2080.51 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 132 — SELL (started 2026-01-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-09 12:15:00 | 2071.40 | 2079.80 | 2080.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-09 13:15:00 | 2069.80 | 2077.80 | 2079.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-12 09:15:00 | 2091.70 | 2079.14 | 2079.55 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-12 09:15:00 | 2091.70 | 2079.14 | 2079.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 09:15:00 | 2091.70 | 2079.14 | 2079.55 | EMA400 retest candle locked (from downside) |

### Cycle 133 — BUY (started 2026-01-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-12 10:15:00 | 2085.00 | 2080.31 | 2080.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-12 12:15:00 | 2099.60 | 2086.20 | 2082.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-13 09:15:00 | 2089.90 | 2091.20 | 2086.83 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-13 10:15:00 | 2088.60 | 2091.20 | 2086.83 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 10:15:00 | 2089.90 | 2090.94 | 2087.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-13 10:30:00 | 2086.00 | 2090.94 | 2087.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 11:15:00 | 2083.00 | 2089.35 | 2086.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-13 12:15:00 | 2084.90 | 2089.35 | 2086.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 12:15:00 | 2080.80 | 2087.64 | 2086.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-13 13:00:00 | 2080.80 | 2087.64 | 2086.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 14:15:00 | 2081.10 | 2086.71 | 2086.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-13 14:30:00 | 2083.70 | 2086.71 | 2086.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 134 — SELL (started 2026-01-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-13 15:15:00 | 2076.90 | 2084.75 | 2085.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-14 09:15:00 | 2073.20 | 2082.44 | 2084.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-16 10:15:00 | 2095.60 | 2078.26 | 2079.81 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-16 10:15:00 | 2095.60 | 2078.26 | 2079.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 10:15:00 | 2095.60 | 2078.26 | 2079.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-16 11:00:00 | 2095.60 | 2078.26 | 2079.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 11:15:00 | 2084.90 | 2079.58 | 2080.27 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-16 12:15:00 | 2080.60 | 2079.58 | 2080.27 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-16 13:00:00 | 2079.60 | 2079.59 | 2080.21 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-19 09:15:00 | 2087.00 | 2080.74 | 2080.45 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 135 — BUY (started 2026-01-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-19 09:15:00 | 2087.00 | 2080.74 | 2080.45 | EMA200 above EMA400 |

### Cycle 136 — SELL (started 2026-01-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-19 11:15:00 | 2077.10 | 2080.22 | 2080.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-19 12:15:00 | 2074.80 | 2079.13 | 2079.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-19 15:15:00 | 2079.00 | 2077.81 | 2078.90 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-19 15:15:00 | 2079.00 | 2077.81 | 2078.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 15:15:00 | 2079.00 | 2077.81 | 2078.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-20 09:15:00 | 2073.00 | 2077.81 | 2078.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 09:15:00 | 2062.30 | 2074.71 | 2077.39 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-20 10:15:00 | 2061.00 | 2074.71 | 2077.39 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-20 13:15:00 | 2061.60 | 2068.28 | 2073.39 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-21 10:15:00 | 2057.20 | 2059.89 | 2067.13 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-27 15:15:00 | 2035.00 | 2027.63 | 2027.50 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 137 — BUY (started 2026-01-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-27 15:15:00 | 2035.00 | 2027.63 | 2027.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-28 10:15:00 | 2064.00 | 2036.74 | 2031.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-29 09:15:00 | 2018.80 | 2043.13 | 2038.77 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-29 09:15:00 | 2018.80 | 2043.13 | 2038.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 09:15:00 | 2018.80 | 2043.13 | 2038.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-29 09:45:00 | 2007.00 | 2043.13 | 2038.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 138 — SELL (started 2026-01-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-29 10:15:00 | 1999.90 | 2034.48 | 2035.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-29 11:15:00 | 1981.10 | 2023.81 | 2030.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-30 11:15:00 | 2000.30 | 1995.85 | 2009.48 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-30 12:00:00 | 2000.30 | 1995.85 | 2009.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 12:15:00 | 1994.40 | 1995.56 | 2008.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-30 12:45:00 | 2005.00 | 1995.56 | 2008.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 13:15:00 | 2008.10 | 1998.07 | 2008.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-30 14:00:00 | 2008.10 | 1998.07 | 2008.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 14:15:00 | 1995.20 | 1997.49 | 2006.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-30 14:45:00 | 2012.90 | 1997.49 | 2006.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 15:15:00 | 2009.90 | 1999.98 | 2007.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-01 09:15:00 | 2003.90 | 1999.98 | 2007.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 09:15:00 | 1990.10 | 1998.00 | 2005.65 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-01 11:30:00 | 1980.70 | 1990.60 | 2000.93 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-01 12:00:00 | 1963.00 | 1990.60 | 2000.93 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-01 14:45:00 | 1976.20 | 1984.45 | 1995.05 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-02 10:45:00 | 1980.50 | 1979.47 | 1989.95 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 11:15:00 | 1992.20 | 1982.02 | 1990.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-02 12:00:00 | 1992.20 | 1982.02 | 1990.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 12:15:00 | 1996.30 | 1984.87 | 1990.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-02 12:45:00 | 1997.70 | 1984.87 | 1990.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 13:15:00 | 1991.50 | 1986.20 | 1990.79 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-02-03 09:15:00 | 2051.70 | 2003.57 | 1997.92 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 139 — BUY (started 2026-02-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 09:15:00 | 2051.70 | 2003.57 | 1997.92 | EMA200 above EMA400 |

### Cycle 140 — SELL (started 2026-02-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-06 09:15:00 | 2007.90 | 2017.60 | 2018.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-06 10:15:00 | 1985.70 | 2011.22 | 2015.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-06 14:15:00 | 1997.50 | 1996.81 | 2005.85 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-06 15:00:00 | 1997.50 | 1996.81 | 2005.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 09:15:00 | 2028.40 | 2002.62 | 2006.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-09 10:00:00 | 2028.40 | 2002.62 | 2006.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 10:15:00 | 2033.40 | 2008.77 | 2009.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-09 10:30:00 | 2034.20 | 2008.77 | 2009.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 141 — BUY (started 2026-02-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-09 11:15:00 | 2025.40 | 2012.10 | 2010.76 | EMA200 above EMA400 |

### Cycle 142 — SELL (started 2026-02-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-10 11:15:00 | 2004.90 | 2011.80 | 2012.24 | EMA200 below EMA400 |

### Cycle 143 — BUY (started 2026-02-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-10 13:15:00 | 2020.90 | 2013.67 | 2013.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-10 15:15:00 | 2022.10 | 2016.05 | 2014.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-12 09:15:00 | 2005.00 | 2019.50 | 2017.95 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-12 09:15:00 | 2005.00 | 2019.50 | 2017.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 09:15:00 | 2005.00 | 2019.50 | 2017.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-12 09:45:00 | 2002.70 | 2019.50 | 2017.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 10:15:00 | 2010.10 | 2017.62 | 2017.24 | EMA400 retest candle locked (from upside) |

### Cycle 144 — SELL (started 2026-02-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-12 11:15:00 | 2012.20 | 2016.53 | 2016.78 | EMA200 below EMA400 |

### Cycle 145 — BUY (started 2026-02-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-12 14:15:00 | 2020.40 | 2017.20 | 2016.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-13 09:15:00 | 2031.30 | 2020.47 | 2018.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-13 11:15:00 | 2017.00 | 2019.81 | 2018.57 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-13 11:15:00 | 2017.00 | 2019.81 | 2018.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 11:15:00 | 2017.00 | 2019.81 | 2018.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-13 12:00:00 | 2017.00 | 2019.81 | 2018.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 12:15:00 | 2022.50 | 2020.35 | 2018.92 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-13 14:00:00 | 2031.90 | 2022.66 | 2020.10 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-25 14:15:00 | 2072.10 | 2084.61 | 2085.85 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 146 — SELL (started 2026-02-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-25 14:15:00 | 2072.10 | 2084.61 | 2085.85 | EMA200 below EMA400 |

### Cycle 147 — BUY (started 2026-02-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-26 10:15:00 | 2091.70 | 2086.29 | 2086.23 | EMA200 above EMA400 |

### Cycle 148 — SELL (started 2026-02-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-26 11:15:00 | 2077.00 | 2084.43 | 2085.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-26 12:15:00 | 2074.20 | 2082.38 | 2084.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-26 14:15:00 | 2081.60 | 2080.04 | 2082.83 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-26 14:15:00 | 2081.60 | 2080.04 | 2082.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 14:15:00 | 2081.60 | 2080.04 | 2082.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-26 15:00:00 | 2081.60 | 2080.04 | 2082.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 15:15:00 | 2084.90 | 2081.01 | 2083.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-27 09:15:00 | 2064.00 | 2081.01 | 2083.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 09:15:00 | 2052.60 | 2075.33 | 2080.25 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-27 11:00:00 | 2048.20 | 2069.90 | 2077.34 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-27 11:45:00 | 2044.50 | 2063.88 | 2073.93 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-04 11:15:00 | 1945.79 | 1990.64 | 2019.82 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-04 11:15:00 | 1942.27 | 1990.64 | 2019.82 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-05 14:15:00 | 1941.50 | 1935.39 | 1964.59 | SL hit (close>ema200) qty=0.50 sl=1935.39 alert=retest2 |

### Cycle 149 — BUY (started 2026-03-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-10 13:15:00 | 1970.90 | 1945.06 | 1941.55 | EMA200 above EMA400 |

### Cycle 150 — SELL (started 2026-03-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-11 14:15:00 | 1938.30 | 1942.34 | 1942.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-11 15:15:00 | 1935.00 | 1940.88 | 1942.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-12 12:15:00 | 1931.90 | 1930.48 | 1935.76 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-12 13:00:00 | 1931.90 | 1930.48 | 1935.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 13:15:00 | 1943.30 | 1933.04 | 1936.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-12 13:45:00 | 1941.70 | 1933.04 | 1936.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 14:15:00 | 1942.40 | 1934.92 | 1936.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-12 14:30:00 | 1940.40 | 1934.92 | 1936.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 15:15:00 | 1940.90 | 1936.11 | 1937.34 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-13 09:15:00 | 1922.00 | 1936.11 | 1937.34 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-17 13:15:00 | 1933.60 | 1917.24 | 1916.08 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 151 — BUY (started 2026-03-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-17 13:15:00 | 1933.60 | 1917.24 | 1916.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-17 14:15:00 | 1935.20 | 1920.84 | 1917.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 09:15:00 | 1934.00 | 1950.06 | 1939.08 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-19 09:15:00 | 1934.00 | 1950.06 | 1939.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 09:15:00 | 1934.00 | 1950.06 | 1939.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-19 09:30:00 | 1931.00 | 1950.06 | 1939.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 10:15:00 | 1926.50 | 1945.35 | 1937.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-19 11:00:00 | 1926.50 | 1945.35 | 1937.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 152 — SELL (started 2026-03-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 13:15:00 | 1907.30 | 1931.77 | 1933.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 14:15:00 | 1902.40 | 1925.90 | 1930.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 12:15:00 | 1844.60 | 1835.81 | 1858.53 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-24 12:45:00 | 1848.70 | 1835.81 | 1858.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 09:15:00 | 1863.20 | 1842.44 | 1854.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-25 09:30:00 | 1865.30 | 1842.44 | 1854.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 10:15:00 | 1864.80 | 1846.91 | 1855.51 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-25 12:45:00 | 1858.00 | 1851.69 | 1856.34 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-25 13:15:00 | 1857.40 | 1851.69 | 1856.34 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-25 14:00:00 | 1857.50 | 1852.85 | 1856.45 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-01 11:15:00 | 1765.10 | 1787.58 | 1807.84 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-01 11:15:00 | 1764.53 | 1787.58 | 1807.84 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-01 11:15:00 | 1764.62 | 1787.58 | 1807.84 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-01 14:15:00 | 1791.60 | 1786.20 | 1801.97 | SL hit (close>ema200) qty=0.50 sl=1786.20 alert=retest2 |

### Cycle 153 — BUY (started 2026-04-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-06 13:15:00 | 1829.60 | 1793.04 | 1788.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-06 14:15:00 | 1838.50 | 1802.13 | 1792.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-09 12:15:00 | 1900.00 | 1903.58 | 1876.80 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-09 13:00:00 | 1900.00 | 1903.58 | 1876.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 1909.70 | 1922.05 | 1907.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-13 09:30:00 | 1895.40 | 1922.05 | 1907.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 10:15:00 | 1918.60 | 1921.36 | 1908.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-13 10:30:00 | 1907.00 | 1921.36 | 1908.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 14:15:00 | 1914.90 | 1921.13 | 1912.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-13 15:00:00 | 1914.90 | 1921.13 | 1912.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-17 09:15:00 | 1971.70 | 1971.13 | 1957.91 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-17 14:30:00 | 1973.50 | 1970.49 | 1962.46 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-20 10:00:00 | 1980.00 | 1972.01 | 1964.52 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-21 09:15:00 | 1900.00 | 1962.84 | 1965.08 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 154 — SELL (started 2026-04-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-21 09:15:00 | 1900.00 | 1962.84 | 1965.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-22 09:15:00 | 1896.00 | 1917.94 | 1936.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-27 09:15:00 | 1795.30 | 1793.28 | 1826.88 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-27 09:30:00 | 1784.90 | 1793.28 | 1826.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 09:15:00 | 1816.40 | 1808.79 | 1819.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-28 10:00:00 | 1816.40 | 1808.79 | 1819.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 10:15:00 | 1822.10 | 1811.45 | 1819.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-28 11:00:00 | 1822.10 | 1811.45 | 1819.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 11:15:00 | 1817.70 | 1812.70 | 1819.69 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-28 15:00:00 | 1808.30 | 1813.07 | 1818.26 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-29 09:15:00 | 1826.10 | 1814.96 | 1818.18 | SL hit (close>static) qty=1.00 sl=1823.70 alert=retest2 |

### Cycle 155 — BUY (started 2026-05-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-04 09:15:00 | 1837.10 | 1816.99 | 1815.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-06 11:15:00 | 1853.00 | 1835.04 | 1827.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-08 14:15:00 | 1871.10 | 1872.22 | 1862.87 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-05-08 15:00:00 | 1871.10 | 1872.22 | 1862.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2024-05-13 14:45:00 | 1424.10 | 2024-05-14 15:15:00 | 1436.50 | STOP_HIT | 1.00 | -0.87% |
| SELL | retest2 | 2024-05-14 12:15:00 | 1424.45 | 2024-05-14 15:15:00 | 1436.50 | STOP_HIT | 1.00 | -0.85% |
| BUY | retest2 | 2024-05-27 11:45:00 | 1439.85 | 2024-05-27 13:15:00 | 1431.00 | STOP_HIT | 1.00 | -0.61% |
| SELL | retest2 | 2024-06-04 09:15:00 | 1355.05 | 2024-06-04 09:15:00 | 1401.30 | STOP_HIT | 1.00 | -3.41% |
| SELL | retest2 | 2024-06-04 10:30:00 | 1374.90 | 2024-06-05 15:15:00 | 1385.00 | STOP_HIT | 1.00 | -0.73% |
| SELL | retest2 | 2024-06-05 12:30:00 | 1375.05 | 2024-06-05 15:15:00 | 1385.00 | STOP_HIT | 1.00 | -0.72% |
| SELL | retest2 | 2024-06-05 13:30:00 | 1373.70 | 2024-06-05 15:15:00 | 1385.00 | STOP_HIT | 1.00 | -0.82% |
| BUY | retest2 | 2024-06-12 10:30:00 | 1442.35 | 2024-06-19 14:15:00 | 1448.70 | STOP_HIT | 1.00 | 0.44% |
| BUY | retest2 | 2024-06-13 14:15:00 | 1440.35 | 2024-06-19 14:15:00 | 1448.70 | STOP_HIT | 1.00 | 0.58% |
| SELL | retest2 | 2024-06-26 09:15:00 | 1457.60 | 2024-06-26 09:15:00 | 1465.70 | STOP_HIT | 1.00 | -0.56% |
| BUY | retest2 | 2024-07-02 11:45:00 | 1496.55 | 2024-07-18 09:15:00 | 1646.21 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-07-02 14:00:00 | 1496.95 | 2024-07-18 09:15:00 | 1646.65 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-07-03 09:15:00 | 1498.60 | 2024-07-18 09:15:00 | 1648.46 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-07-30 11:15:00 | 1727.00 | 2024-08-05 09:15:00 | 1737.90 | STOP_HIT | 1.00 | 0.63% |
| BUY | retest2 | 2024-07-31 09:45:00 | 1733.25 | 2024-08-05 09:15:00 | 1737.90 | STOP_HIT | 1.00 | 0.27% |
| SELL | retest2 | 2024-08-08 13:15:00 | 1679.25 | 2024-08-08 14:15:00 | 1705.30 | STOP_HIT | 1.00 | -1.55% |
| BUY | retest2 | 2024-08-12 11:45:00 | 1727.80 | 2024-08-13 09:15:00 | 1681.10 | STOP_HIT | 1.00 | -2.70% |
| SELL | retest2 | 2024-08-14 15:15:00 | 1688.75 | 2024-08-20 09:15:00 | 1705.85 | STOP_HIT | 1.00 | -1.01% |
| SELL | retest2 | 2024-08-16 09:30:00 | 1680.65 | 2024-08-20 09:15:00 | 1705.85 | STOP_HIT | 1.00 | -1.50% |
| SELL | retest2 | 2024-08-16 14:30:00 | 1687.25 | 2024-08-20 09:15:00 | 1705.85 | STOP_HIT | 1.00 | -1.10% |
| SELL | retest2 | 2024-08-16 15:00:00 | 1687.65 | 2024-08-20 09:15:00 | 1705.85 | STOP_HIT | 1.00 | -1.08% |
| SELL | retest2 | 2024-08-19 10:15:00 | 1676.50 | 2024-08-20 09:15:00 | 1705.85 | STOP_HIT | 1.00 | -1.75% |
| BUY | retest2 | 2024-08-26 13:15:00 | 1799.55 | 2024-09-06 14:15:00 | 1893.45 | STOP_HIT | 1.00 | 5.22% |
| BUY | retest2 | 2024-08-27 09:45:00 | 1804.45 | 2024-09-06 14:15:00 | 1893.45 | STOP_HIT | 1.00 | 4.93% |
| BUY | retest2 | 2024-08-27 10:45:00 | 1806.10 | 2024-09-06 14:15:00 | 1893.45 | STOP_HIT | 1.00 | 4.84% |
| BUY | retest2 | 2024-09-19 14:45:00 | 1840.50 | 2024-09-25 09:15:00 | 1859.05 | STOP_HIT | 1.00 | 1.01% |
| SELL | retest2 | 2024-10-07 10:45:00 | 1787.40 | 2024-10-15 13:15:00 | 1746.40 | STOP_HIT | 1.00 | 2.29% |
| SELL | retest2 | 2024-10-07 11:45:00 | 1788.35 | 2024-10-15 13:15:00 | 1746.40 | STOP_HIT | 1.00 | 2.35% |
| SELL | retest2 | 2024-10-21 10:30:00 | 1710.00 | 2024-10-23 12:15:00 | 1718.95 | STOP_HIT | 1.00 | -0.52% |
| SELL | retest2 | 2024-10-21 12:00:00 | 1710.00 | 2024-10-23 12:15:00 | 1718.95 | STOP_HIT | 1.00 | -0.52% |
| SELL | retest2 | 2024-10-21 14:00:00 | 1709.50 | 2024-10-23 12:15:00 | 1718.95 | STOP_HIT | 1.00 | -0.55% |
| SELL | retest2 | 2024-10-21 15:15:00 | 1707.00 | 2024-10-23 12:15:00 | 1718.95 | STOP_HIT | 1.00 | -0.70% |
| SELL | retest2 | 2024-11-04 10:30:00 | 1601.45 | 2024-11-05 13:15:00 | 1639.30 | STOP_HIT | 1.00 | -2.36% |
| SELL | retest2 | 2024-11-04 11:30:00 | 1601.85 | 2024-11-05 13:15:00 | 1639.30 | STOP_HIT | 1.00 | -2.34% |
| SELL | retest2 | 2024-11-04 12:15:00 | 1600.55 | 2024-11-05 14:15:00 | 1631.00 | STOP_HIT | 1.00 | -1.90% |
| SELL | retest2 | 2024-11-04 12:45:00 | 1600.25 | 2024-11-05 14:15:00 | 1631.00 | STOP_HIT | 1.00 | -1.92% |
| SELL | retest2 | 2024-11-05 11:30:00 | 1602.00 | 2024-11-05 14:15:00 | 1631.00 | STOP_HIT | 1.00 | -1.81% |
| SELL | retest2 | 2024-11-05 12:15:00 | 1601.40 | 2024-11-05 14:15:00 | 1631.00 | STOP_HIT | 1.00 | -1.85% |
| SELL | retest2 | 2024-11-08 11:15:00 | 1584.15 | 2024-11-19 12:15:00 | 1504.94 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-11-08 11:15:00 | 1584.15 | 2024-11-22 11:15:00 | 1490.20 | STOP_HIT | 0.50 | 5.93% |
| SELL | retest2 | 2024-12-03 10:30:00 | 1419.10 | 2024-12-03 13:15:00 | 1444.55 | STOP_HIT | 1.00 | -1.79% |
| BUY | retest2 | 2024-12-11 10:30:00 | 1467.50 | 2024-12-12 09:15:00 | 1438.20 | STOP_HIT | 1.00 | -2.00% |
| BUY | retest2 | 2024-12-11 11:30:00 | 1464.20 | 2024-12-12 09:15:00 | 1438.20 | STOP_HIT | 1.00 | -1.78% |
| SELL | retest2 | 2024-12-16 10:15:00 | 1420.75 | 2024-12-24 09:15:00 | 1407.90 | STOP_HIT | 1.00 | 0.90% |
| SELL | retest2 | 2024-12-16 12:00:00 | 1421.65 | 2024-12-24 09:15:00 | 1407.90 | STOP_HIT | 1.00 | 0.97% |
| SELL | retest2 | 2024-12-16 15:00:00 | 1421.95 | 2024-12-24 09:15:00 | 1407.90 | STOP_HIT | 1.00 | 0.99% |
| SELL | retest2 | 2024-12-17 10:45:00 | 1420.75 | 2024-12-24 09:15:00 | 1407.90 | STOP_HIT | 1.00 | 0.90% |
| SELL | retest2 | 2024-12-19 15:15:00 | 1402.30 | 2024-12-24 09:15:00 | 1407.90 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest2 | 2024-12-20 09:30:00 | 1398.75 | 2024-12-24 09:15:00 | 1407.90 | STOP_HIT | 1.00 | -0.65% |
| SELL | retest2 | 2024-12-20 12:45:00 | 1403.40 | 2024-12-24 09:15:00 | 1407.90 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest2 | 2024-12-20 15:00:00 | 1394.30 | 2024-12-24 09:15:00 | 1407.90 | STOP_HIT | 1.00 | -0.98% |
| SELL | retest2 | 2024-12-23 13:00:00 | 1396.55 | 2024-12-24 09:15:00 | 1407.90 | STOP_HIT | 1.00 | -0.81% |
| SELL | retest2 | 2024-12-31 09:15:00 | 1387.20 | 2025-01-02 09:15:00 | 1402.00 | STOP_HIT | 1.00 | -1.07% |
| SELL | retest2 | 2024-12-31 12:30:00 | 1390.00 | 2025-01-02 09:15:00 | 1402.00 | STOP_HIT | 1.00 | -0.86% |
| SELL | retest2 | 2024-12-31 13:30:00 | 1390.10 | 2025-01-02 09:15:00 | 1402.00 | STOP_HIT | 1.00 | -0.86% |
| SELL | retest2 | 2024-12-31 14:15:00 | 1390.00 | 2025-01-02 09:15:00 | 1402.00 | STOP_HIT | 1.00 | -0.86% |
| BUY | retest2 | 2025-01-07 09:15:00 | 1446.65 | 2025-01-20 13:15:00 | 1497.70 | STOP_HIT | 1.00 | 3.53% |
| SELL | retest1 | 2025-01-23 09:15:00 | 1450.50 | 2025-01-29 09:15:00 | 1438.25 | STOP_HIT | 1.00 | 0.84% |
| SELL | retest1 | 2025-01-23 11:15:00 | 1457.00 | 2025-01-29 09:15:00 | 1438.25 | STOP_HIT | 1.00 | 1.29% |
| SELL | retest1 | 2025-01-23 12:15:00 | 1457.10 | 2025-01-29 09:15:00 | 1438.25 | STOP_HIT | 1.00 | 1.29% |
| SELL | retest1 | 2025-01-23 14:15:00 | 1456.85 | 2025-01-29 09:15:00 | 1438.25 | STOP_HIT | 1.00 | 1.28% |
| SELL | retest2 | 2025-01-28 15:00:00 | 1420.20 | 2025-01-29 09:15:00 | 1438.25 | STOP_HIT | 1.00 | -1.27% |
| BUY | retest2 | 2025-02-17 10:30:00 | 1479.60 | 2025-02-20 15:15:00 | 1470.00 | STOP_HIT | 1.00 | -0.65% |
| BUY | retest2 | 2025-02-17 11:30:00 | 1479.80 | 2025-02-20 15:15:00 | 1470.00 | STOP_HIT | 1.00 | -0.66% |
| BUY | retest2 | 2025-02-17 13:15:00 | 1478.85 | 2025-02-20 15:15:00 | 1470.00 | STOP_HIT | 1.00 | -0.60% |
| BUY | retest2 | 2025-02-18 13:30:00 | 1479.55 | 2025-02-20 15:15:00 | 1470.00 | STOP_HIT | 1.00 | -0.65% |
| BUY | retest2 | 2025-02-20 09:15:00 | 1483.00 | 2025-02-20 15:15:00 | 1470.00 | STOP_HIT | 1.00 | -0.88% |
| BUY | retest2 | 2025-02-20 10:15:00 | 1482.55 | 2025-02-20 15:15:00 | 1470.00 | STOP_HIT | 1.00 | -0.85% |
| BUY | retest2 | 2025-04-23 12:45:00 | 1613.60 | 2025-05-02 09:15:00 | 1774.96 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-04-23 13:30:00 | 1614.20 | 2025-05-02 09:15:00 | 1775.62 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-04-23 15:00:00 | 1614.20 | 2025-05-02 09:15:00 | 1775.62 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-04-24 15:15:00 | 1615.00 | 2025-05-02 09:15:00 | 1776.50 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-04-25 09:15:00 | 1697.70 | 2025-05-06 13:15:00 | 1731.00 | STOP_HIT | 1.00 | 1.96% |
| BUY | retest2 | 2025-05-13 12:15:00 | 1753.90 | 2025-05-20 14:15:00 | 1759.60 | STOP_HIT | 1.00 | 0.32% |
| BUY | retest2 | 2025-05-14 11:15:00 | 1753.10 | 2025-05-20 14:15:00 | 1759.60 | STOP_HIT | 1.00 | 0.37% |
| BUY | retest2 | 2025-05-14 12:15:00 | 1752.20 | 2025-05-20 14:15:00 | 1759.60 | STOP_HIT | 1.00 | 0.42% |
| BUY | retest2 | 2025-05-15 10:30:00 | 1755.60 | 2025-05-20 14:15:00 | 1759.60 | STOP_HIT | 1.00 | 0.23% |
| BUY | retest2 | 2025-05-15 13:00:00 | 1772.40 | 2025-05-20 14:15:00 | 1759.60 | STOP_HIT | 1.00 | -0.72% |
| BUY | retest2 | 2025-06-23 11:45:00 | 1818.60 | 2025-06-30 14:15:00 | 1838.80 | STOP_HIT | 1.00 | 1.11% |
| BUY | retest2 | 2025-06-23 12:15:00 | 1819.40 | 2025-06-30 14:15:00 | 1838.80 | STOP_HIT | 1.00 | 1.07% |
| BUY | retest2 | 2025-06-24 09:15:00 | 1830.40 | 2025-06-30 14:15:00 | 1838.80 | STOP_HIT | 1.00 | 0.46% |
| BUY | retest2 | 2025-06-27 10:00:00 | 1822.70 | 2025-06-30 14:15:00 | 1838.80 | STOP_HIT | 1.00 | 0.88% |
| BUY | retest2 | 2025-06-27 14:30:00 | 1862.80 | 2025-06-30 14:15:00 | 1838.80 | STOP_HIT | 1.00 | -1.29% |
| SELL | retest2 | 2025-07-08 09:15:00 | 1805.20 | 2025-07-08 10:15:00 | 1816.20 | STOP_HIT | 1.00 | -0.61% |
| SELL | retest2 | 2025-07-17 09:15:00 | 1817.00 | 2025-07-22 12:15:00 | 1805.50 | STOP_HIT | 1.00 | 0.63% |
| BUY | retest2 | 2025-08-07 14:30:00 | 1860.40 | 2025-08-08 13:15:00 | 1834.80 | STOP_HIT | 1.00 | -1.38% |
| BUY | retest2 | 2025-08-08 10:15:00 | 1856.40 | 2025-08-08 13:15:00 | 1834.80 | STOP_HIT | 1.00 | -1.16% |
| BUY | retest2 | 2025-08-12 12:15:00 | 1848.50 | 2025-08-12 14:15:00 | 1837.20 | STOP_HIT | 1.00 | -0.61% |
| SELL | retest2 | 2025-08-28 12:15:00 | 1829.60 | 2025-09-04 09:15:00 | 1830.50 | STOP_HIT | 1.00 | -0.05% |
| SELL | retest2 | 2025-08-28 14:30:00 | 1824.70 | 2025-09-04 09:15:00 | 1830.50 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest2 | 2025-09-08 09:15:00 | 1779.70 | 2025-09-09 15:15:00 | 1810.00 | STOP_HIT | 1.00 | -1.70% |
| SELL | retest2 | 2025-09-09 11:45:00 | 1799.00 | 2025-09-09 15:15:00 | 1810.00 | STOP_HIT | 1.00 | -0.61% |
| SELL | retest2 | 2025-09-09 13:15:00 | 1792.10 | 2025-09-09 15:15:00 | 1810.00 | STOP_HIT | 1.00 | -1.00% |
| BUY | retest2 | 2025-09-11 14:30:00 | 1817.20 | 2025-09-16 15:15:00 | 1819.10 | STOP_HIT | 1.00 | 0.10% |
| BUY | retest2 | 2025-09-12 10:45:00 | 1819.00 | 2025-09-16 15:15:00 | 1819.10 | STOP_HIT | 1.00 | 0.01% |
| BUY | retest2 | 2025-09-16 14:45:00 | 1816.50 | 2025-09-16 15:15:00 | 1819.10 | STOP_HIT | 1.00 | 0.14% |
| SELL | retest2 | 2025-09-18 13:45:00 | 1816.00 | 2025-09-18 14:15:00 | 1822.80 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest2 | 2025-09-26 12:15:00 | 1803.00 | 2025-10-07 14:15:00 | 1782.30 | STOP_HIT | 1.00 | 1.15% |
| BUY | retest2 | 2025-10-13 10:15:00 | 1822.30 | 2025-11-10 09:15:00 | 2004.53 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-10-15 09:15:00 | 1841.10 | 2025-11-10 09:15:00 | 2008.49 | TARGET_HIT | 1.00 | 9.09% |
| BUY | retest2 | 2025-10-16 11:00:00 | 1825.90 | 2025-11-13 10:15:00 | 1981.20 | STOP_HIT | 1.00 | 8.51% |
| BUY | retest2 | 2025-11-24 15:15:00 | 2075.00 | 2025-11-27 10:15:00 | 2014.20 | STOP_HIT | 1.00 | -2.93% |
| BUY | retest2 | 2025-11-25 15:00:00 | 2029.30 | 2025-11-27 10:15:00 | 2014.20 | STOP_HIT | 1.00 | -0.74% |
| BUY | retest2 | 2025-11-26 14:30:00 | 2029.90 | 2025-11-27 10:15:00 | 2014.20 | STOP_HIT | 1.00 | -0.77% |
| BUY | retest2 | 2025-12-08 11:30:00 | 2035.90 | 2025-12-09 09:15:00 | 1997.20 | STOP_HIT | 1.00 | -1.90% |
| BUY | retest2 | 2025-12-17 09:15:00 | 2043.10 | 2025-12-17 09:15:00 | 2012.60 | STOP_HIT | 1.00 | -1.49% |
| BUY | retest2 | 2025-12-22 15:15:00 | 2024.80 | 2025-12-26 10:15:00 | 2006.50 | STOP_HIT | 1.00 | -0.90% |
| BUY | retest2 | 2025-12-23 14:45:00 | 2023.40 | 2025-12-26 10:15:00 | 2006.50 | STOP_HIT | 1.00 | -0.84% |
| BUY | retest2 | 2025-12-26 09:15:00 | 2023.20 | 2025-12-26 10:15:00 | 2006.50 | STOP_HIT | 1.00 | -0.83% |
| BUY | retest2 | 2026-01-06 12:00:00 | 2097.10 | 2026-01-09 12:15:00 | 2071.40 | STOP_HIT | 1.00 | -1.23% |
| BUY | retest2 | 2026-01-07 09:15:00 | 2093.30 | 2026-01-09 12:15:00 | 2071.40 | STOP_HIT | 1.00 | -1.05% |
| SELL | retest2 | 2026-01-16 12:15:00 | 2080.60 | 2026-01-19 09:15:00 | 2087.00 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest2 | 2026-01-16 13:00:00 | 2079.60 | 2026-01-19 09:15:00 | 2087.00 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest2 | 2026-01-20 10:15:00 | 2061.00 | 2026-01-27 15:15:00 | 2035.00 | STOP_HIT | 1.00 | 1.26% |
| SELL | retest2 | 2026-01-20 13:15:00 | 2061.60 | 2026-01-27 15:15:00 | 2035.00 | STOP_HIT | 1.00 | 1.29% |
| SELL | retest2 | 2026-01-21 10:15:00 | 2057.20 | 2026-01-27 15:15:00 | 2035.00 | STOP_HIT | 1.00 | 1.08% |
| SELL | retest2 | 2026-02-01 11:30:00 | 1980.70 | 2026-02-03 09:15:00 | 2051.70 | STOP_HIT | 1.00 | -3.58% |
| SELL | retest2 | 2026-02-01 12:00:00 | 1963.00 | 2026-02-03 09:15:00 | 2051.70 | STOP_HIT | 1.00 | -4.52% |
| SELL | retest2 | 2026-02-01 14:45:00 | 1976.20 | 2026-02-03 09:15:00 | 2051.70 | STOP_HIT | 1.00 | -3.82% |
| SELL | retest2 | 2026-02-02 10:45:00 | 1980.50 | 2026-02-03 09:15:00 | 2051.70 | STOP_HIT | 1.00 | -3.60% |
| BUY | retest2 | 2026-02-13 14:00:00 | 2031.90 | 2026-02-25 14:15:00 | 2072.10 | STOP_HIT | 1.00 | 1.98% |
| SELL | retest2 | 2026-02-27 11:00:00 | 2048.20 | 2026-03-04 11:15:00 | 1945.79 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-27 11:45:00 | 2044.50 | 2026-03-04 11:15:00 | 1942.27 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-27 11:00:00 | 2048.20 | 2026-03-05 14:15:00 | 1941.50 | STOP_HIT | 0.50 | 5.21% |
| SELL | retest2 | 2026-02-27 11:45:00 | 2044.50 | 2026-03-05 14:15:00 | 1941.50 | STOP_HIT | 0.50 | 5.04% |
| SELL | retest2 | 2026-03-13 09:15:00 | 1922.00 | 2026-03-17 13:15:00 | 1933.60 | STOP_HIT | 1.00 | -0.60% |
| SELL | retest2 | 2026-03-25 12:45:00 | 1858.00 | 2026-04-01 11:15:00 | 1765.10 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-25 13:15:00 | 1857.40 | 2026-04-01 11:15:00 | 1764.53 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-25 14:00:00 | 1857.50 | 2026-04-01 11:15:00 | 1764.62 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-25 12:45:00 | 1858.00 | 2026-04-01 14:15:00 | 1791.60 | STOP_HIT | 0.50 | 3.57% |
| SELL | retest2 | 2026-03-25 13:15:00 | 1857.40 | 2026-04-01 14:15:00 | 1791.60 | STOP_HIT | 0.50 | 3.54% |
| SELL | retest2 | 2026-03-25 14:00:00 | 1857.50 | 2026-04-01 14:15:00 | 1791.60 | STOP_HIT | 0.50 | 3.55% |
| BUY | retest2 | 2026-04-17 14:30:00 | 1973.50 | 2026-04-21 09:15:00 | 1900.00 | STOP_HIT | 1.00 | -3.72% |
| BUY | retest2 | 2026-04-20 10:00:00 | 1980.00 | 2026-04-21 09:15:00 | 1900.00 | STOP_HIT | 1.00 | -4.04% |
| SELL | retest2 | 2026-04-28 15:00:00 | 1808.30 | 2026-04-29 09:15:00 | 1826.10 | STOP_HIT | 1.00 | -0.98% |
| SELL | retest2 | 2026-04-29 12:15:00 | 1811.80 | 2026-05-04 09:15:00 | 1837.10 | STOP_HIT | 1.00 | -1.40% |
| SELL | retest2 | 2026-04-29 14:15:00 | 1811.20 | 2026-05-04 09:15:00 | 1837.10 | STOP_HIT | 1.00 | -1.43% |
| SELL | retest2 | 2026-04-29 14:45:00 | 1810.70 | 2026-05-04 09:15:00 | 1837.10 | STOP_HIT | 1.00 | -1.46% |
| SELL | retest2 | 2026-04-30 09:15:00 | 1795.00 | 2026-05-04 09:15:00 | 1837.10 | STOP_HIT | 1.00 | -2.35% |
