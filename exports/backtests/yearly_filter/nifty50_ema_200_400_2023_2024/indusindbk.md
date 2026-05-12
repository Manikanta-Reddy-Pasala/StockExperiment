# INDUSINDBK (INDUSINDBK)

## Backtest Summary

- **Window:** 2022-04-08 09:15:00 → 2026-05-08 15:15:00 (7047 bars)
- **Last close:** 948.45
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 12 |
| ALERT1 | 11 |
| ALERT2 | 11 |
| ALERT2_SKIP | 1 |
| ALERT3 | 67 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 56 |
| PARTIAL | 12 |
| TARGET_HIT | 6 |
| STOP_HIT | 50 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 68 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 29 / 39
- **Target hits / Stop hits / Partials:** 6 / 50 / 12
- **Avg / median % per leg:** 0.29% / -1.21%
- **Sum % (uncompounded):** 19.60%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 10 | 5 | 50.0% | 5 | 5 | 0 | 2.54% | 25.4% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 10 | 5 | 50.0% | 5 | 5 | 0 | 2.54% | 25.4% |
| SELL (all) | 58 | 24 | 41.4% | 1 | 45 | 12 | -0.10% | -5.8% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 58 | 24 | 41.4% | 1 | 45 | 12 | -0.10% | -5.8% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 68 | 29 | 42.6% | 6 | 50 | 12 | 0.29% | 19.6% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-02-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-14 11:15:00 | 1466.05 | 1532.46 | 1532.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-28 14:15:00 | 1450.00 | 1507.32 | 1517.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-01 11:15:00 | 1507.75 | 1503.62 | 1515.35 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-03-01 11:45:00 | 1506.80 | 1503.62 | 1515.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-01 13:15:00 | 1516.15 | 1503.80 | 1515.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-01 14:00:00 | 1516.15 | 1503.80 | 1515.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-01 14:15:00 | 1526.95 | 1504.03 | 1515.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-01 15:00:00 | 1526.95 | 1504.03 | 1515.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-13 10:15:00 | 1520.65 | 1518.85 | 1521.21 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-13 11:15:00 | 1514.45 | 1518.85 | 1521.21 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-19 14:15:00 | 1438.73 | 1507.26 | 1514.71 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-03-22 12:15:00 | 1501.85 | 1500.89 | 1510.65 | SL hit (close>ema200) qty=0.50 sl=1500.89 alert=retest2 |

### Cycle 2 — BUY (started 2024-04-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-05 14:15:00 | 1553.60 | 1518.07 | 1517.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-08 09:15:00 | 1558.50 | 1518.81 | 1518.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-16 09:15:00 | 1513.00 | 1529.25 | 1524.11 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-16 09:15:00 | 1513.00 | 1529.25 | 1524.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-16 09:15:00 | 1513.00 | 1529.25 | 1524.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-16 09:45:00 | 1510.95 | 1529.25 | 1524.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-16 10:15:00 | 1509.80 | 1529.05 | 1524.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-16 11:15:00 | 1510.00 | 1529.05 | 1524.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 3 — SELL (started 2024-04-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-22 13:15:00 | 1471.25 | 1519.47 | 1519.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-24 09:15:00 | 1469.00 | 1515.43 | 1517.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-30 12:15:00 | 1507.90 | 1506.42 | 1512.31 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-04-30 12:45:00 | 1505.95 | 1506.42 | 1512.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-30 13:15:00 | 1531.20 | 1506.67 | 1512.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-30 14:00:00 | 1531.20 | 1506.67 | 1512.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-30 14:15:00 | 1517.85 | 1506.78 | 1512.43 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-30 15:15:00 | 1509.00 | 1506.78 | 1512.43 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-08 09:15:00 | 1433.55 | 1501.59 | 1508.95 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-05-27 09:15:00 | 1455.40 | 1452.70 | 1476.48 | SL hit (close>ema200) qty=0.50 sl=1452.70 alert=retest2 |

### Cycle 4 — BUY (started 2024-06-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-21 14:15:00 | 1525.60 | 1482.89 | 1482.81 | EMA200 above EMA400 |

### Cycle 5 — SELL (started 2024-07-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-02 09:15:00 | 1440.55 | 1482.93 | 1483.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-02 11:15:00 | 1435.90 | 1482.09 | 1482.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-18 14:15:00 | 1458.80 | 1457.51 | 1467.61 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-07-18 14:45:00 | 1456.90 | 1457.51 | 1467.61 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-28 11:15:00 | 1419.00 | 1391.59 | 1415.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-28 12:00:00 | 1419.00 | 1391.59 | 1415.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-28 12:15:00 | 1418.60 | 1391.86 | 1415.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-28 12:30:00 | 1420.00 | 1391.86 | 1415.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-28 15:15:00 | 1409.00 | 1392.54 | 1415.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-29 09:30:00 | 1411.35 | 1392.81 | 1415.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-29 10:15:00 | 1416.15 | 1393.04 | 1415.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-29 10:30:00 | 1417.90 | 1393.04 | 1415.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-29 11:15:00 | 1420.15 | 1393.31 | 1415.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-29 12:00:00 | 1420.15 | 1393.31 | 1415.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-29 12:15:00 | 1420.85 | 1393.59 | 1415.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-29 12:30:00 | 1419.40 | 1393.59 | 1415.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-29 14:15:00 | 1416.85 | 1393.98 | 1415.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-29 15:00:00 | 1416.85 | 1393.98 | 1415.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-29 15:15:00 | 1415.50 | 1394.20 | 1415.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-30 09:15:00 | 1421.30 | 1394.20 | 1415.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-30 09:15:00 | 1420.00 | 1394.45 | 1415.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-30 09:45:00 | 1420.30 | 1394.45 | 1415.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-04 10:15:00 | 1430.00 | 1402.71 | 1417.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-04 11:00:00 | 1430.00 | 1402.71 | 1417.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-09 09:15:00 | 1409.00 | 1406.38 | 1418.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-09 09:30:00 | 1418.15 | 1406.38 | 1418.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-09 10:15:00 | 1418.40 | 1406.50 | 1418.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-09 11:00:00 | 1418.40 | 1406.50 | 1418.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-09 11:15:00 | 1416.45 | 1406.60 | 1418.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-09 11:45:00 | 1420.10 | 1406.60 | 1418.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-09 12:15:00 | 1419.85 | 1406.73 | 1418.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-09 13:00:00 | 1419.85 | 1406.73 | 1418.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-09 13:15:00 | 1422.70 | 1406.89 | 1418.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-09 13:30:00 | 1420.40 | 1406.89 | 1418.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-11 15:15:00 | 1420.20 | 1410.34 | 1419.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-12 09:15:00 | 1426.95 | 1410.34 | 1419.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 6 — BUY (started 2024-09-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-18 15:15:00 | 1474.50 | 1426.25 | 1426.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-19 09:15:00 | 1491.00 | 1426.90 | 1426.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-25 12:15:00 | 1434.55 | 1437.15 | 1432.19 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-25 13:00:00 | 1434.55 | 1437.15 | 1432.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-01 09:15:00 | 1441.15 | 1440.01 | 1434.28 | EMA400 retest candle locked (from upside) |

### Cycle 7 — SELL (started 2024-10-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-07 14:15:00 | 1349.20 | 1428.87 | 1429.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-09 14:15:00 | 1341.85 | 1419.78 | 1424.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-02 15:15:00 | 996.00 | 992.56 | 1068.65 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-03 09:15:00 | 997.60 | 992.56 | 1068.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 09:15:00 | 1018.75 | 970.01 | 1014.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-01 09:30:00 | 1021.00 | 970.01 | 1014.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 10:15:00 | 1014.50 | 970.45 | 1014.20 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-01 12:30:00 | 1003.10 | 971.13 | 1014.11 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-01 13:00:00 | 1002.50 | 971.13 | 1014.11 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-03 09:15:00 | 1003.75 | 972.32 | 1014.07 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-04 09:15:00 | 1031.80 | 975.66 | 1014.13 | SL hit (close>static) qty=1.00 sl=1020.35 alert=retest2 |

### Cycle 8 — BUY (started 2025-06-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-27 11:15:00 | 866.20 | 822.24 | 822.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-01 11:15:00 | 882.00 | 828.17 | 825.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-21 10:15:00 | 845.10 | 850.17 | 839.76 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-21 10:45:00 | 841.85 | 850.17 | 839.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 09:15:00 | 835.75 | 849.79 | 840.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-25 10:00:00 | 835.75 | 849.79 | 840.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 10:15:00 | 830.25 | 849.60 | 840.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-25 11:00:00 | 830.25 | 849.60 | 840.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 9 — SELL (started 2025-08-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-04 09:15:00 | 795.95 | 833.84 | 833.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-08 15:15:00 | 780.55 | 824.59 | 828.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-06 09:15:00 | 752.70 | 750.34 | 770.40 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-06 09:30:00 | 753.50 | 750.34 | 770.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 09:15:00 | 765.60 | 750.20 | 767.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-13 10:00:00 | 765.60 | 750.20 | 767.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 10:15:00 | 762.50 | 750.32 | 767.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-13 10:30:00 | 765.25 | 750.32 | 767.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 10:15:00 | 768.20 | 749.69 | 764.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-20 11:00:00 | 768.20 | 749.69 | 764.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 11:15:00 | 765.65 | 749.85 | 764.02 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-20 15:00:00 | 760.10 | 750.23 | 764.00 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-23 11:15:00 | 761.85 | 750.71 | 763.91 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-23 14:45:00 | 760.45 | 751.11 | 763.85 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-27 11:00:00 | 760.60 | 751.57 | 763.46 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 11:15:00 | 761.65 | 751.67 | 763.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-27 11:45:00 | 763.50 | 751.67 | 763.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 13:15:00 | 765.00 | 751.90 | 763.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-27 13:45:00 | 766.30 | 751.90 | 763.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 14:15:00 | 769.65 | 752.08 | 763.48 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-10-27 14:15:00 | 769.65 | 752.08 | 763.48 | SL hit (close>static) qty=1.00 sl=769.45 alert=retest2 |

### Cycle 10 — BUY (started 2025-11-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-10 13:15:00 | 799.00 | 771.97 | 771.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-11 09:15:00 | 820.50 | 772.97 | 772.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-10 15:15:00 | 832.00 | 833.00 | 813.70 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-11 09:45:00 | 831.95 | 833.05 | 813.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 09:15:00 | 886.55 | 888.15 | 862.21 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-27 11:15:00 | 891.20 | 888.15 | 862.34 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-27 14:30:00 | 889.90 | 888.14 | 862.84 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-28 11:15:00 | 892.05 | 888.16 | 863.23 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-01 12:30:00 | 891.70 | 890.08 | 866.95 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-09 09:15:00 | 882.85 | 924.05 | 903.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-09 10:00:00 | 882.85 | 924.05 | 903.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 12:15:00 | 894.35 | 920.29 | 902.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-10 12:30:00 | 892.50 | 920.29 | 902.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 10:15:00 | 890.80 | 919.14 | 902.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-11 11:00:00 | 890.80 | 919.14 | 902.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2026-03-12 09:15:00 | 837.45 | 916.51 | 901.28 | SL hit (close<static) qty=1.00 sl=851.15 alert=retest2 |

### Cycle 11 — SELL (started 2026-03-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 10:15:00 | 820.75 | 888.47 | 888.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 14:15:00 | 814.65 | 885.78 | 887.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 10:15:00 | 838.55 | 835.26 | 857.16 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-08 10:45:00 | 838.05 | 835.26 | 857.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-16 09:15:00 | 853.15 | 833.29 | 852.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-16 09:30:00 | 851.15 | 833.29 | 852.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-16 10:15:00 | 842.90 | 833.38 | 852.60 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-17 09:30:00 | 840.85 | 834.20 | 852.44 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-17 13:15:00 | 854.20 | 834.77 | 852.37 | SL hit (close>static) qty=1.00 sl=853.30 alert=retest2 |

### Cycle 12 — BUY (started 2026-05-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-05 11:15:00 | 908.20 | 862.52 | 862.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-05 12:15:00 | 913.70 | 863.03 | 862.75 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2023-10-05 09:15:00 | 1405.65 | 2023-10-05 11:15:00 | 1398.60 | STOP_HIT | 1.00 | -0.50% |
| BUY | retest2 | 2023-10-06 09:15:00 | 1412.00 | 2023-12-14 10:15:00 | 1547.92 | TARGET_HIT | 1.00 | 9.63% |
| BUY | retest2 | 2023-10-26 10:45:00 | 1407.20 | 2023-12-14 10:15:00 | 1544.73 | TARGET_HIT | 1.00 | 9.77% |
| BUY | retest2 | 2023-10-26 11:45:00 | 1404.30 | 2023-12-14 13:15:00 | 1553.20 | TARGET_HIT | 1.00 | 10.60% |
| BUY | retest2 | 2023-10-26 14:30:00 | 1419.50 | 2023-12-15 09:15:00 | 1561.45 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2023-10-27 09:15:00 | 1419.60 | 2023-12-15 09:15:00 | 1561.56 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-03-13 11:15:00 | 1514.45 | 2024-03-19 14:15:00 | 1438.73 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-03-13 11:15:00 | 1514.45 | 2024-03-22 12:15:00 | 1501.85 | STOP_HIT | 0.50 | 0.83% |
| SELL | retest2 | 2024-03-26 10:00:00 | 1512.30 | 2024-03-27 11:15:00 | 1531.80 | STOP_HIT | 1.00 | -1.29% |
| SELL | retest2 | 2024-04-04 10:30:00 | 1514.20 | 2024-04-04 12:15:00 | 1530.70 | STOP_HIT | 1.00 | -1.09% |
| SELL | retest2 | 2024-04-30 15:15:00 | 1509.00 | 2024-05-08 09:15:00 | 1433.55 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-04-30 15:15:00 | 1509.00 | 2024-05-27 09:15:00 | 1455.40 | STOP_HIT | 0.50 | 3.55% |
| SELL | retest2 | 2024-06-03 09:30:00 | 1504.75 | 2024-06-04 10:15:00 | 1429.51 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-06-04 09:15:00 | 1465.85 | 2024-06-04 10:15:00 | 1392.56 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-06-03 09:30:00 | 1504.75 | 2024-06-05 12:15:00 | 1464.95 | STOP_HIT | 0.50 | 2.64% |
| SELL | retest2 | 2024-06-04 09:15:00 | 1465.85 | 2024-06-05 12:15:00 | 1464.95 | STOP_HIT | 0.50 | 0.06% |
| SELL | retest2 | 2024-06-18 13:45:00 | 1512.05 | 2024-06-19 09:15:00 | 1542.80 | STOP_HIT | 1.00 | -2.03% |
| SELL | retest2 | 2025-02-01 12:30:00 | 1003.10 | 2025-02-04 09:15:00 | 1031.80 | STOP_HIT | 1.00 | -2.86% |
| SELL | retest2 | 2025-02-01 13:00:00 | 1002.50 | 2025-02-04 09:15:00 | 1031.80 | STOP_HIT | 1.00 | -2.92% |
| SELL | retest2 | 2025-02-03 09:15:00 | 1003.75 | 2025-02-04 09:15:00 | 1031.80 | STOP_HIT | 1.00 | -2.79% |
| SELL | retest2 | 2025-02-28 09:15:00 | 992.40 | 2025-03-07 09:15:00 | 942.78 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-28 09:15:00 | 992.40 | 2025-03-10 09:15:00 | 893.16 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-04-21 12:15:00 | 829.35 | 2025-04-22 09:15:00 | 787.88 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-04-21 12:15:00 | 829.35 | 2025-04-22 09:15:00 | 804.50 | STOP_HIT | 0.50 | 3.00% |
| SELL | retest2 | 2025-04-21 13:15:00 | 828.80 | 2025-04-22 09:15:00 | 787.36 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-04-21 13:15:00 | 828.80 | 2025-04-22 09:15:00 | 804.50 | STOP_HIT | 0.50 | 2.93% |
| SELL | retest2 | 2025-04-21 15:00:00 | 828.10 | 2025-04-22 09:15:00 | 786.69 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-04-21 15:00:00 | 828.10 | 2025-04-22 09:15:00 | 804.50 | STOP_HIT | 0.50 | 2.85% |
| SELL | retest2 | 2025-04-24 12:45:00 | 820.15 | 2025-04-28 12:15:00 | 837.55 | STOP_HIT | 1.00 | -2.12% |
| SELL | retest2 | 2025-04-30 10:00:00 | 817.50 | 2025-04-30 13:15:00 | 836.25 | STOP_HIT | 1.00 | -2.29% |
| SELL | retest2 | 2025-05-07 10:45:00 | 818.60 | 2025-05-13 10:15:00 | 777.67 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-05-08 15:15:00 | 818.60 | 2025-05-13 10:15:00 | 777.67 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-05-09 10:15:00 | 817.45 | 2025-05-13 12:15:00 | 776.58 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-05-12 11:00:00 | 816.05 | 2025-05-13 12:15:00 | 775.25 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-05-07 10:45:00 | 818.60 | 2025-05-20 09:15:00 | 799.05 | STOP_HIT | 0.50 | 2.39% |
| SELL | retest2 | 2025-05-08 15:15:00 | 818.60 | 2025-05-20 09:15:00 | 799.05 | STOP_HIT | 0.50 | 2.39% |
| SELL | retest2 | 2025-05-09 10:15:00 | 817.45 | 2025-05-20 09:15:00 | 799.05 | STOP_HIT | 0.50 | 2.25% |
| SELL | retest2 | 2025-05-12 11:00:00 | 816.05 | 2025-05-20 09:15:00 | 799.05 | STOP_HIT | 0.50 | 2.08% |
| SELL | retest2 | 2025-05-28 09:45:00 | 814.90 | 2025-06-06 12:15:00 | 841.65 | STOP_HIT | 1.00 | -3.28% |
| SELL | retest2 | 2025-05-28 11:15:00 | 813.90 | 2025-06-06 12:15:00 | 841.65 | STOP_HIT | 1.00 | -3.41% |
| SELL | retest2 | 2025-05-29 12:15:00 | 814.85 | 2025-06-06 12:15:00 | 841.65 | STOP_HIT | 1.00 | -3.29% |
| SELL | retest2 | 2025-06-02 12:30:00 | 811.30 | 2025-06-06 12:15:00 | 841.65 | STOP_HIT | 1.00 | -3.74% |
| SELL | retest2 | 2025-06-02 14:30:00 | 812.05 | 2025-06-06 12:15:00 | 841.65 | STOP_HIT | 1.00 | -3.65% |
| SELL | retest2 | 2025-06-02 15:15:00 | 812.05 | 2025-06-06 12:15:00 | 841.65 | STOP_HIT | 1.00 | -3.65% |
| SELL | retest2 | 2025-06-03 09:30:00 | 812.65 | 2025-06-06 12:15:00 | 841.65 | STOP_HIT | 1.00 | -3.57% |
| SELL | retest2 | 2025-06-05 10:15:00 | 806.55 | 2025-06-06 12:15:00 | 841.65 | STOP_HIT | 1.00 | -4.35% |
| SELL | retest2 | 2025-06-06 09:30:00 | 808.50 | 2025-06-06 12:15:00 | 841.65 | STOP_HIT | 1.00 | -4.10% |
| SELL | retest2 | 2025-06-06 10:15:00 | 810.05 | 2025-06-06 12:15:00 | 841.65 | STOP_HIT | 1.00 | -3.90% |
| SELL | retest2 | 2025-06-06 11:15:00 | 808.05 | 2025-06-06 12:15:00 | 841.65 | STOP_HIT | 1.00 | -4.16% |
| SELL | retest2 | 2025-06-13 10:00:00 | 820.55 | 2025-06-18 09:15:00 | 851.75 | STOP_HIT | 1.00 | -3.80% |
| SELL | retest2 | 2025-06-13 11:15:00 | 820.65 | 2025-06-18 09:15:00 | 851.75 | STOP_HIT | 1.00 | -3.79% |
| SELL | retest2 | 2025-06-16 14:45:00 | 820.70 | 2025-06-18 09:15:00 | 851.75 | STOP_HIT | 1.00 | -3.78% |
| SELL | retest2 | 2025-06-16 15:15:00 | 820.15 | 2025-06-18 09:15:00 | 851.75 | STOP_HIT | 1.00 | -3.85% |
| SELL | retest2 | 2025-06-18 11:15:00 | 835.45 | 2025-06-27 11:15:00 | 866.20 | STOP_HIT | 1.00 | -3.68% |
| SELL | retest2 | 2025-06-19 13:30:00 | 836.15 | 2025-06-27 11:15:00 | 866.20 | STOP_HIT | 1.00 | -3.59% |
| SELL | retest2 | 2025-06-19 14:30:00 | 836.60 | 2025-06-27 11:15:00 | 866.20 | STOP_HIT | 1.00 | -3.54% |
| SELL | retest2 | 2025-06-19 15:15:00 | 836.00 | 2025-06-27 11:15:00 | 866.20 | STOP_HIT | 1.00 | -3.61% |
| SELL | retest2 | 2025-10-20 15:00:00 | 760.10 | 2025-10-27 14:15:00 | 769.65 | STOP_HIT | 1.00 | -1.26% |
| SELL | retest2 | 2025-10-23 11:15:00 | 761.85 | 2025-10-27 14:15:00 | 769.65 | STOP_HIT | 1.00 | -1.02% |
| SELL | retest2 | 2025-10-23 14:45:00 | 760.45 | 2025-10-27 14:15:00 | 769.65 | STOP_HIT | 1.00 | -1.21% |
| SELL | retest2 | 2025-10-27 11:00:00 | 760.60 | 2025-10-27 14:15:00 | 769.65 | STOP_HIT | 1.00 | -1.19% |
| BUY | retest2 | 2026-01-27 11:15:00 | 891.20 | 2026-03-12 09:15:00 | 837.45 | STOP_HIT | 1.00 | -6.03% |
| BUY | retest2 | 2026-01-27 14:30:00 | 889.90 | 2026-03-12 09:15:00 | 837.45 | STOP_HIT | 1.00 | -5.89% |
| BUY | retest2 | 2026-01-28 11:15:00 | 892.05 | 2026-03-12 09:15:00 | 837.45 | STOP_HIT | 1.00 | -6.12% |
| BUY | retest2 | 2026-02-01 12:30:00 | 891.70 | 2026-03-12 09:15:00 | 837.45 | STOP_HIT | 1.00 | -6.08% |
| SELL | retest2 | 2026-04-17 09:30:00 | 840.85 | 2026-04-17 13:15:00 | 854.20 | STOP_HIT | 1.00 | -1.59% |
| SELL | retest2 | 2026-04-20 09:30:00 | 840.40 | 2026-04-20 14:15:00 | 853.45 | STOP_HIT | 1.00 | -1.55% |
| SELL | retest2 | 2026-04-24 14:00:00 | 840.50 | 2026-04-27 09:15:00 | 880.70 | STOP_HIT | 1.00 | -4.78% |
