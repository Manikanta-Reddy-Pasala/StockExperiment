# RELIANCE (RELIANCE.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-06-06 09:15:00 → 2026-05-06 15:30:00 (4998 bars)
- **Last close:** 1437.90
- **Strategy:** EMA 200/400 1H crossover (v2 BTC rules)
- **Target:** entry × (1 ± 30%)
- **Partial:** 50% qty booked @ 15%, trail SL → entry
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 3 attempts each at retest1 and retest2

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 4 |
| ALERT1 | 4 |
| ALERT2 | 4 |
| ALERT2_SKIP | 2 |
| ALERT3 | 6 |
| PENDING | 23 |
| PENDING_CANCEL | 7 |
| ENTRY1 | 2 |
| ENTRY2 | 14 |
| PARTIAL | 1 |
| TARGET_HIT | 0 |
| STOP_HIT | 15 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 16 (incl. partial bookings)
- **Trades open at end:** 1
- **Winners / losers:** 2 / 14
- **Target hits / Stop hits / Partials:** 0 / 15 / 1
- **Avg / median % per leg:** 0.47% / -0.59%
- **Sum % (uncompounded):** 7.47%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 1 | 0 | 0.0% | 0 | 1 | 0 | -2.20% | -2.2% |
| BUY @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -2.20% | -2.2% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 15 | 2 | 13.3% | 0 | 14 | 1 | 0.64% | 9.7% |
| SELL @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -2.29% | -2.3% |
| SELL @ 3rd Alert (retest2) | 14 | 2 | 14.3% | 0 | 13 | 1 | 0.85% | 12.0% |
| retest1 (combined) | 2 | 0 | 0.0% | 0 | 2 | 0 | -2.25% | -4.5% |
| retest2 (combined) | 14 | 2 | 14.3% | 0 | 13 | 1 | 0.85% | 12.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-09-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-11 12:15:00 | 1464.88 | 1493.99 | 1494.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-11 13:15:00 | 1453.20 | 1493.59 | 1493.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-20 10:15:00 | 1487.90 | 1484.61 | 1488.78 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-20 11:15:00 | 1489.55 | 1484.66 | 1488.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 11:15:00 | 1489.55 | 1484.66 | 1488.79 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2024-09-20 13:15:00 | 1484.75 | 1484.73 | 1488.79 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-20 14:15:00 | 1485.07 | 1484.74 | 1488.77 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2024-09-20 14:15:00 | 1492.50 | 1484.74 | 1488.77 | SL hit qty=1.00 sl=1492.50 alert=retest2 |
| Cross detected — sustain check pending | 2024-09-25 09:15:00 | 1484.95 | 1485.97 | 1489.10 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-25 10:15:00 | 1483.72 | 1485.95 | 1489.07 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2024-09-25 13:15:00 | 1492.50 | 1486.04 | 1489.08 | SL hit qty=1.00 sl=1492.50 alert=retest2 |
| Cross detected — sustain check pending | 2024-09-30 13:15:00 | 1480.78 | 1488.47 | 1490.05 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-30 14:15:00 | 1475.07 | 1488.33 | 1489.98 | SELL ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Partial book — 50% qty @ 15%, trail SL → entry | 2024-11-13 13:15:00 | 1253.81 | 1343.89 | 1389.39 | Partial book 0.50 @ 15%; trail SL->entry alert=retest2 |
| CROSSOVER_SKIP | 2025-04-24 13:15:00 | 1300.10 | 1249.12 | 1249.00 | HTF filter: close below htf_sma |
| Stop hit — per-position SL triggered | 2025-06-26 09:15:00 | 1475.07 | 1428.93 | 1391.91 | SL hit qty=0.50 sl=1475.07 alert=retest2 |
| Cross detected — sustain check pending | 2025-07-14 12:15:00 | 1483.30 | 1480.61 | 1436.36 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-14 13:15:00 | 1482.60 | 1480.63 | 1436.59 | SELL ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 14:15:00 | 1483.40 | 1480.65 | 1436.82 | EMA400 retest candle locked |
| Stop hit — per-position SL triggered | 2025-07-15 09:15:00 | 1492.50 | 1480.79 | 1437.33 | SL hit qty=1.00 sl=1492.50 alert=retest2 |
| Cross detected — sustain check pending | 2025-07-16 09:15:00 | 1475.70 | 1481.25 | 1439.06 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-16 10:15:00 | 1479.90 | 1481.23 | 1439.26 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-07-16 12:15:00 | 1480.70 | 1481.25 | 1439.68 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-07-16 13:15:00 | 1485.00 | 1481.28 | 1439.91 | ENTRY2 sustain failed after 60m |
| Stop hit — per-position SL triggered | 2025-07-16 14:15:00 | 1487.40 | 1481.32 | 1440.14 | SL hit qty=1.00 sl=1487.40 alert=retest2 |
| Cross detected — sustain check pending | 2025-07-17 09:15:00 | 1479.90 | 1481.36 | 1440.56 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-07-17 10:15:00 | 1481.80 | 1481.36 | 1440.77 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-07-17 13:15:00 | 1479.00 | 1481.35 | 1441.37 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-17 14:15:00 | 1478.10 | 1481.32 | 1441.55 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-08-11 15:15:00 | 1384.70 | 1423.43 | 1423.49 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 2 — SELL (started 2025-08-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-11 15:15:00 | 1384.70 | 1423.43 | 1423.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-12 13:15:00 | 1381.10 | 1421.69 | 1422.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-19 12:15:00 | 1413.60 | 1413.28 | 1417.97 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-19 12:15:00 | 1413.60 | 1413.28 | 1417.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 12:15:00 | 1413.60 | 1413.28 | 1417.97 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-08-20 15:15:00 | 1411.30 | 1413.76 | 1417.99 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-08-21 09:15:00 | 1429.80 | 1413.92 | 1418.05 | ENTRY2 sustain failed after 1080m |
| Cross detected — sustain check pending | 2025-08-22 14:15:00 | 1408.90 | 1414.65 | 1418.19 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-22 15:15:00 | 1410.00 | 1414.60 | 1418.15 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-08-25 13:15:00 | 1418.30 | 1414.49 | 1418.00 | SL hit qty=1.00 sl=1418.30 alert=retest2 |
| Cross detected — sustain check pending | 2025-08-25 14:15:00 | 1410.20 | 1414.45 | 1417.96 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-08-25 15:15:00 | 1412.60 | 1414.43 | 1417.94 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-08-26 09:15:00 | 1402.20 | 1414.31 | 1417.86 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-26 10:15:00 | 1398.70 | 1414.15 | 1417.76 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-09-18 09:15:00 | 1418.30 | 1394.75 | 1403.12 | SL hit qty=1.00 sl=1418.30 alert=retest2 |
| Cross detected — sustain check pending | 2025-09-19 09:15:00 | 1405.80 | 1395.99 | 1403.46 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-19 10:15:00 | 1405.30 | 1396.09 | 1403.47 | SELL ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-09-19 14:15:00 | 1406.30 | 1396.63 | 1403.60 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-19 15:15:00 | 1409.40 | 1396.75 | 1403.63 | SELL ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 09:15:00 | 1409.00 | 1396.88 | 1403.65 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-09-22 14:15:00 | 1390.20 | 1397.12 | 1403.61 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-22 15:15:00 | 1390.90 | 1397.06 | 1403.55 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-10-17 09:15:00 | 1410.30 | 1383.20 | 1391.57 | SL hit qty=1.00 sl=1410.30 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-17 11:15:00 | 1418.30 | 1383.90 | 1391.84 | SL hit qty=1.00 sl=1418.30 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-17 11:15:00 | 1418.30 | 1383.90 | 1391.84 | SL hit qty=1.00 sl=1418.30 alert=retest2 |

### Cycle 3 — BUY (started 2025-10-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-24 12:15:00 | 1459.00 | 1399.06 | 1398.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-27 09:15:00 | 1482.40 | 1401.41 | 1400.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-30 14:15:00 | 1539.30 | 1541.16 | 1511.64 | EMA200 retest candle locked |
| Cross detected — sustain check pending | 2025-12-31 09:15:00 | 1548.00 | 1541.23 | 1511.96 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-31 10:15:00 | 1555.00 | 1541.36 | 1512.18 | BUY ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 09:15:00 | 1526.20 | 1550.52 | 1520.75 | EMA400 retest candle locked |
| Stop hit — per-position SL triggered | 2026-01-06 09:15:00 | 1520.75 | 1550.52 | 1520.75 | SL hit qty=1.00 sl=1520.75 alert=retest1 |

### Cycle 4 — SELL (started 2026-01-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-21 10:15:00 | 1384.00 | 1501.19 | 1501.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-02 09:15:00 | 1363.30 | 1460.02 | 1478.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-04 09:15:00 | 1459.10 | 1454.50 | 1474.41 | EMA200 retest candle locked |
| Cross detected — sustain check pending | 2026-02-05 13:15:00 | 1441.00 | 1454.07 | 1473.12 | ENTRY1 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-02-05 14:15:00 | 1444.50 | 1453.97 | 1472.98 | ENTRY1 sustain failed after 60m |
| Cross detected — sustain check pending | 2026-02-05 15:15:00 | 1441.90 | 1453.85 | 1472.83 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-06 09:15:00 | 1437.90 | 1453.69 | 1472.65 | SELL ENTRY1 attempt 1/4 (retest1 break sustained 1080m) |
| Cross detected — sustain check pending | 2026-02-06 12:15:00 | 1440.50 | 1453.40 | 1472.22 | ENTRY1 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-02-06 13:15:00 | 1442.00 | 1453.29 | 1472.07 | ENTRY1 sustain failed after 60m |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 09:15:00 | 1464.40 | 1455.06 | 1470.87 | EMA400 retest candle locked |
| Stop hit — per-position SL triggered | 2026-02-12 09:15:00 | 1470.87 | 1455.06 | 1470.87 | SL hit qty=1.00 sl=1470.87 alert=retest1 |
| Cross detected — sustain check pending | 2026-02-12 11:15:00 | 1457.20 | 1455.15 | 1470.76 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-12 12:15:00 | 1454.70 | 1455.14 | 1470.68 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2026-05-05 09:15:00 | 1473.00 | 1377.61 | 1392.85 | SL hit qty=1.00 sl=1473.00 alert=retest2 |
| Cross detected — sustain check pending | 2026-05-05 10:15:00 | 1452.30 | 1378.35 | 1393.15 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-05 11:15:00 | 1460.10 | 1379.16 | 1393.48 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2026-05-05 13:15:00 | 1460.40 | 1380.83 | 1394.17 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-05-05 14:15:00 | 1464.40 | 1381.66 | 1394.53 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2026-05-06 09:15:00 | 1455.50 | 1383.19 | 1395.17 | ENTRY2 cross detected — sustain check pending (15m) |
| Stop hit — per-position SL triggered | 2026-05-06 09:15:00 | 1473.00 | 1383.19 | 1395.17 | SL hit qty=1.00 sl=1473.00 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-06 10:15:00 | 1447.30 | 1383.83 | 1395.43 | SELL ENTRY2 attempt 3/4 (retest2 break sustained 60m) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2024-09-20 14:15:00 | 1485.07 | 2024-09-20 14:15:00 | 1492.50 | STOP_HIT | 1.00 | -0.50% |
| SELL | retest2 | 2024-09-25 10:15:00 | 1483.72 | 2024-09-25 13:15:00 | 1492.50 | STOP_HIT | 1.00 | -0.59% |
| SELL | retest2 | 2024-09-30 14:15:00 | 1475.07 | 2024-11-13 13:15:00 | 1253.81 | PARTIAL | 0.50 | 15.00% |
| SELL | retest2 | 2024-09-30 14:15:00 | 1475.07 | 2025-06-26 09:15:00 | 1475.07 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest2 | 2025-07-14 13:15:00 | 1482.60 | 2025-07-15 09:15:00 | 1492.50 | STOP_HIT | 1.00 | -0.67% |
| SELL | retest2 | 2025-07-16 10:15:00 | 1479.90 | 2025-07-16 14:15:00 | 1487.40 | STOP_HIT | 1.00 | -0.51% |
| SELL | retest2 | 2025-07-17 14:15:00 | 1478.10 | 2025-08-11 15:15:00 | 1384.70 | STOP_HIT | 1.00 | 6.32% |
| SELL | retest2 | 2025-08-22 15:15:00 | 1410.00 | 2025-08-25 13:15:00 | 1418.30 | STOP_HIT | 1.00 | -0.59% |
| SELL | retest2 | 2025-08-26 10:15:00 | 1398.70 | 2025-09-18 09:15:00 | 1418.30 | STOP_HIT | 1.00 | -1.40% |
| SELL | retest2 | 2025-09-19 10:15:00 | 1405.30 | 2025-10-17 09:15:00 | 1410.30 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest2 | 2025-09-19 15:15:00 | 1409.40 | 2025-10-17 11:15:00 | 1418.30 | STOP_HIT | 1.00 | -0.63% |
| SELL | retest2 | 2025-09-22 15:15:00 | 1390.90 | 2025-10-17 11:15:00 | 1418.30 | STOP_HIT | 1.00 | -1.97% |
| BUY | retest1 | 2025-12-31 10:15:00 | 1555.00 | 2026-01-06 09:15:00 | 1520.75 | STOP_HIT | 1.00 | -2.20% |
| SELL | retest1 | 2026-02-06 09:15:00 | 1437.90 | 2026-02-12 09:15:00 | 1470.87 | STOP_HIT | 1.00 | -2.29% |
| SELL | retest2 | 2026-02-12 12:15:00 | 1454.70 | 2026-05-05 09:15:00 | 1473.00 | STOP_HIT | 1.00 | -1.26% |
| SELL | retest2 | 2026-05-05 11:15:00 | 1460.10 | 2026-05-06 09:15:00 | 1473.00 | STOP_HIT | 1.00 | -0.88% |
