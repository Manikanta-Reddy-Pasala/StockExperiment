# Adani Ports and Special Economic Zone Ltd. (ADANIPORTS)

## Backtest Summary

- **Window:** 2024-04-08 09:15:00 → 2026-05-08 15:15:00 (3164 bars)
- **Last close:** 1760.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 7 |
| ALERT1 | 7 |
| ALERT2 | 6 |
| ALERT2_SKIP | 3 |
| ALERT3 | 23 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 2 |
| ENTRY2 | 14 |
| PARTIAL | 0 |
| TARGET_HIT | 4 |
| STOP_HIT | 13 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 16 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 4 / 12
- **Target hits / Stop hits / Partials:** 4 / 12 / 0
- **Avg / median % per leg:** 1.14% / -0.85%
- **Sum % (uncompounded):** 18.24%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 16 | 4 | 25.0% | 4 | 12 | 0 | 1.14% | 18.2% |
| BUY @ 2nd Alert (retest1) | 2 | 0 | 0.0% | 0 | 2 | 0 | -2.82% | -5.6% |
| BUY @ 3rd Alert (retest2) | 14 | 4 | 28.6% | 4 | 10 | 0 | 1.71% | 23.9% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 2 | 0 | 0.0% | 0 | 2 | 0 | -2.82% | -5.6% |
| retest2 (combined) | 14 | 4 | 28.6% | 4 | 10 | 0 | 1.71% | 23.9% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-13 15:15:00 | 1369.00 | 1261.00 | 1260.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-14 09:15:00 | 1378.70 | 1262.17 | 1261.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-16 09:15:00 | 1396.70 | 1399.37 | 1353.90 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-17 09:30:00 | 1411.80 | 1399.48 | 1355.53 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 09:15:00 | 1348.30 | 1396.94 | 1357.20 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-06-19 09:15:00 | 1348.30 | 1396.94 | 1357.20 | SL hit (close<ema400) qty=1.00 sl=1357.20 alert=retest1 |

### Cycle 2 — SELL (started 2025-08-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-13 09:15:00 | 1335.60 | 1386.00 | 1386.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-13 10:15:00 | 1325.10 | 1385.40 | 1385.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-20 15:15:00 | 1372.00 | 1371.64 | 1378.12 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-21 09:15:00 | 1372.20 | 1371.64 | 1378.12 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 09:15:00 | 1377.20 | 1350.85 | 1363.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-09 10:00:00 | 1377.20 | 1350.85 | 1363.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 10:15:00 | 1367.00 | 1351.01 | 1363.36 | EMA400 retest candle locked (from downside) |

### Cycle 3 — BUY (started 2025-09-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-19 10:15:00 | 1427.00 | 1372.73 | 1372.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-19 12:15:00 | 1442.90 | 1374.00 | 1373.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-26 15:15:00 | 1391.40 | 1391.40 | 1382.99 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-29 09:15:00 | 1396.00 | 1391.40 | 1382.99 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 11:15:00 | 1380.00 | 1391.30 | 1383.06 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-09-29 11:15:00 | 1380.00 | 1391.30 | 1383.06 | SL hit (close<ema400) qty=1.00 sl=1383.06 alert=retest1 |

### Cycle 4 — SELL (started 2026-01-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-19 15:15:00 | 1403.00 | 1465.11 | 1465.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-20 10:15:00 | 1396.30 | 1463.83 | 1464.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-03 09:15:00 | 1502.40 | 1425.99 | 1442.05 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-03 09:15:00 | 1502.40 | 1425.99 | 1442.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 09:15:00 | 1502.40 | 1425.99 | 1442.05 | EMA400 retest candle locked (from downside) |

### Cycle 5 — BUY (started 2026-02-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-09 10:15:00 | 1562.90 | 1456.70 | 1456.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-09 11:15:00 | 1568.50 | 1457.81 | 1456.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-02 09:15:00 | 1481.00 | 1510.47 | 1490.38 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-02 09:15:00 | 1481.00 | 1510.47 | 1490.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 09:15:00 | 1481.00 | 1510.47 | 1490.38 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-05 14:45:00 | 1497.80 | 1500.63 | 1487.04 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-09 09:15:00 | 1429.60 | 1498.70 | 1486.65 | SL hit (close<static) qty=1.00 sl=1452.20 alert=retest2 |

### Cycle 6 — SELL (started 2026-03-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-13 11:15:00 | 1362.40 | 1476.01 | 1476.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 14:15:00 | 1352.40 | 1448.47 | 1461.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 09:15:00 | 1456.60 | 1403.04 | 1430.25 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-08 09:15:00 | 1456.60 | 1403.04 | 1430.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 09:15:00 | 1456.60 | 1403.04 | 1430.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-08 09:45:00 | 1473.30 | 1403.04 | 1430.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 10:15:00 | 1448.40 | 1406.80 | 1431.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-09 11:00:00 | 1448.40 | 1406.80 | 1431.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 10:15:00 | 1461.70 | 1413.60 | 1432.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-13 11:00:00 | 1461.70 | 1413.60 | 1432.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 7 — BUY (started 2026-04-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-20 15:15:00 | 1572.10 | 1449.50 | 1449.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-21 09:15:00 | 1598.10 | 1450.98 | 1449.92 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-06-17 09:30:00 | 1411.80 | 2025-06-19 09:15:00 | 1348.30 | STOP_HIT | 1.00 | -4.50% |
| BUY | retest2 | 2025-06-24 09:15:00 | 1404.80 | 2025-08-01 14:15:00 | 1346.00 | STOP_HIT | 1.00 | -4.19% |
| BUY | retest2 | 2025-08-01 13:15:00 | 1357.40 | 2025-08-01 14:15:00 | 1346.00 | STOP_HIT | 1.00 | -0.84% |
| BUY | retest2 | 2025-08-04 09:15:00 | 1361.20 | 2025-08-07 09:15:00 | 1345.40 | STOP_HIT | 1.00 | -1.16% |
| BUY | retest2 | 2025-08-04 09:45:00 | 1360.50 | 2025-08-07 09:15:00 | 1345.40 | STOP_HIT | 1.00 | -1.11% |
| BUY | retest1 | 2025-09-29 09:15:00 | 1396.00 | 2025-09-29 11:15:00 | 1380.00 | STOP_HIT | 1.00 | -1.15% |
| BUY | retest2 | 2025-09-30 09:30:00 | 1396.10 | 2025-11-28 13:15:00 | 1531.42 | TARGET_HIT | 1.00 | 9.69% |
| BUY | retest2 | 2025-10-08 12:30:00 | 1392.20 | 2025-11-28 13:15:00 | 1532.30 | TARGET_HIT | 1.00 | 10.06% |
| BUY | retest2 | 2025-10-09 09:15:00 | 1396.00 | 2025-12-01 09:15:00 | 1535.71 | TARGET_HIT | 1.00 | 10.01% |
| BUY | retest2 | 2025-10-09 10:15:00 | 1393.00 | 2025-12-01 09:15:00 | 1535.60 | TARGET_HIT | 1.00 | 10.24% |
| BUY | retest2 | 2025-12-31 13:15:00 | 1474.80 | 2026-01-07 11:15:00 | 1462.20 | STOP_HIT | 1.00 | -0.85% |
| BUY | retest2 | 2026-01-01 09:15:00 | 1473.90 | 2026-01-07 11:15:00 | 1462.20 | STOP_HIT | 1.00 | -0.79% |
| BUY | retest2 | 2026-01-06 14:45:00 | 1474.40 | 2026-01-07 11:15:00 | 1462.20 | STOP_HIT | 1.00 | -0.83% |
| BUY | retest2 | 2026-01-06 15:15:00 | 1476.00 | 2026-01-07 11:15:00 | 1462.20 | STOP_HIT | 1.00 | -0.93% |
| BUY | retest2 | 2026-01-08 09:30:00 | 1482.20 | 2026-01-08 12:15:00 | 1469.50 | STOP_HIT | 1.00 | -0.86% |
| BUY | retest2 | 2026-03-05 14:45:00 | 1497.80 | 2026-03-09 09:15:00 | 1429.60 | STOP_HIT | 1.00 | -4.55% |
