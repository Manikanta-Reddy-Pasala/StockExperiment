# RELIANCE (RELIANCE)

## Backtest Summary

- **Source:** Fyers history API (1H bars)
- **Window:** 2024-05-18 09:15:00 → 2026-05-07 15:15:00 (3402 bars)
- **Last close:** 1435.50
- **Strategy:** EMA 200/400 1H crossover (v2 BTC rules)
- **Target:** entry × (1 ± 30%)
- **Partial:** 50% qty booked @ 15%, trail SL → entry
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 3 attempts each at retest1 and retest2

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 3 |
| ALERT1 | 3 |
| ALERT2 | 3 |
| ALERT2_SKIP | 3 |
| ALERT3 | 5 |
| PENDING | 23 |
| PENDING_CANCEL | 9 |
| ENTRY1 | 0 |
| ENTRY2 | 14 |
| PARTIAL | 1 |
| TARGET_HIT | 0 |
| STOP_HIT | 11 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 12 (incl. partial bookings)
- **Trades open at end:** 3
- **Winners / losers:** 4 / 8
- **Target hits / Stop hits / Partials:** 0 / 11 / 1
- **Avg / median % per leg:** 2.06% / -0.68%
- **Sum % (uncompounded):** 24.78%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 12 | 4 | 33.3% | 0 | 11 | 1 | 2.06% | 24.8% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 12 | 4 | 33.3% | 0 | 11 | 1 | 2.06% | 24.8% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 12 | 4 | 33.3% | 0 | 11 | 1 | 2.06% | 24.8% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-09-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-11 12:15:00 | 1464.88 | 1493.97 | 1494.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-11 13:15:00 | 1452.93 | 1493.56 | 1493.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-20 10:15:00 | 1487.48 | 1484.58 | 1488.74 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-20 11:15:00 | 1489.55 | 1484.63 | 1488.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 11:15:00 | 1489.55 | 1484.63 | 1488.74 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2024-09-20 13:15:00 | 1484.23 | 1484.71 | 1488.74 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-20 15:15:00 | 1486.03 | 1484.73 | 1488.71 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 120m) |
| Stop hit — per-position SL triggered | 2024-09-23 09:15:00 | 1497.83 | 1484.86 | 1488.76 | SL hit (close>static) qty=1.00 sl=1492.50 alert=retest2 |
| Cross detected — sustain check pending | 2024-09-25 09:15:00 | 1484.98 | 1485.95 | 1489.06 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-25 11:15:00 | 1485.13 | 1485.92 | 1489.01 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 120m) |
| Stop hit — per-position SL triggered | 2024-09-25 13:15:00 | 1495.98 | 1486.02 | 1489.04 | SL hit (close>static) qty=1.00 sl=1492.50 alert=retest2 |
| Cross detected — sustain check pending | 2024-09-30 13:15:00 | 1480.78 | 1488.44 | 1490.01 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-30 15:15:00 | 1479.00 | 1488.21 | 1489.88 | SELL ENTRY2 attempt 3/4 (retest2 break sustained 120m) |
| Partial book — 50% qty @ 15%, trail SL → entry | 2024-11-13 09:15:00 | 1257.15 | 1346.83 | 1391.38 | Partial book 0.50 @ 15%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-12-02 13:15:00 | 1308.55 | 1306.88 | 1352.00 | SL hit (close>ema200) qty=0.50 sl=1306.88 alert=retest2 |
| Cross detected — sustain check pending | 2025-07-14 12:15:00 | 1483.30 | 1480.65 | 1436.39 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-14 14:15:00 | 1483.30 | 1480.70 | 1436.85 | SELL ENTRY2 attempt 4/4 (retest2 break sustained 120m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 15:15:00 | 1486.40 | 1480.76 | 1437.10 | EMA400 retest candle locked |
| Stop hit — per-position SL triggered | 2025-07-15 10:15:00 | 1493.40 | 1480.96 | 1437.63 | SL hit (close>static) qty=1.00 sl=1492.50 alert=retest2 |
| Cross detected — sustain check pending | 2025-07-16 09:15:00 | 1475.70 | 1481.30 | 1439.08 | ENTRY2 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2025-07-16 11:15:00 | 1482.80 | 1481.30 | 1439.51 | ENTRY2 sustain failed after 120m |
| Cross detected — sustain check pending | 2025-07-16 12:15:00 | 1480.70 | 1481.29 | 1439.71 | ENTRY2 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2025-07-16 13:15:00 | 1485.00 | 1481.33 | 1439.94 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-07-17 09:15:00 | 1479.90 | 1481.40 | 1440.59 | ENTRY2 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2025-07-17 11:15:00 | 1482.80 | 1481.42 | 1441.01 | ENTRY2 sustain failed after 120m |
| Cross detected — sustain check pending | 2025-07-17 12:15:00 | 1482.20 | 1481.43 | 1441.21 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-17 14:15:00 | 1478.10 | 1481.37 | 1441.58 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 120m) |
| Stop hit — per-position SL triggered | 2025-08-11 15:15:00 | 1384.70 | 1423.44 | 1423.50 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 2 — SELL (started 2025-08-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-11 15:15:00 | 1384.70 | 1423.44 | 1423.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-12 13:15:00 | 1380.50 | 1421.69 | 1422.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-19 12:15:00 | 1413.60 | 1413.28 | 1417.97 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-19 12:15:00 | 1413.60 | 1413.28 | 1417.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 12:15:00 | 1413.60 | 1413.28 | 1417.97 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-08-20 15:15:00 | 1412.00 | 1413.77 | 1418.00 | ENTRY2 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2025-08-21 09:15:00 | 1429.90 | 1413.93 | 1418.06 | ENTRY2 sustain failed after 1080m |
| Cross detected — sustain check pending | 2025-08-22 14:15:00 | 1409.00 | 1414.65 | 1418.20 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-25 09:15:00 | 1408.70 | 1414.55 | 1418.11 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 4020m) |
| Cross detected — sustain check pending | 2025-08-25 14:15:00 | 1410.20 | 1414.46 | 1417.97 | ENTRY2 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2025-08-25 15:15:00 | 1413.70 | 1414.45 | 1417.95 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-08-26 09:15:00 | 1402.40 | 1414.33 | 1417.87 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-26 11:15:00 | 1395.40 | 1413.99 | 1417.67 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 120m) |
| Cross detected — sustain check pending | 2025-09-19 09:15:00 | 1405.80 | 1396.01 | 1403.48 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-19 11:15:00 | 1408.00 | 1396.22 | 1403.51 | SELL ENTRY2 attempt 3/4 (retest2 break sustained 120m) |
| Cross detected — sustain check pending | 2025-09-19 14:15:00 | 1406.30 | 1396.64 | 1403.62 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-22 09:15:00 | 1409.00 | 1396.87 | 1403.66 | SELL ENTRY2 attempt 4/4 (retest2 break sustained 4020m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 10:15:00 | 1404.70 | 1396.95 | 1403.67 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-09-22 13:15:00 | 1403.50 | 1397.19 | 1403.69 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-22 15:15:00 | 1390.90 | 1397.06 | 1403.56 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 120m) |
| Stop hit — per-position SL triggered | 2025-10-17 10:15:00 | 1416.20 | 1383.54 | 1391.71 | SL hit (close>static) qty=1.00 sl=1409.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-17 11:15:00 | 1421.10 | 1383.92 | 1391.85 | SL hit (close>static) qty=1.00 sl=1418.30 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-17 11:15:00 | 1421.10 | 1383.92 | 1391.85 | SL hit (close>static) qty=1.00 sl=1418.30 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-17 11:15:00 | 1421.10 | 1383.92 | 1391.85 | SL hit (close>static) qty=1.00 sl=1418.30 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-17 11:15:00 | 1421.10 | 1383.92 | 1391.85 | SL hit (close>static) qty=1.00 sl=1418.30 alert=retest2 |
| Cross detected — sustain check pending | 2026-01-20 10:15:00 | 1399.90 | 1508.85 | 1505.40 | ENTRY2 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2026-01-20 12:15:00 | 1404.20 | 1506.76 | 1504.39 | ENTRY2 sustain failed after 120m |
| Cross detected — sustain check pending | 2026-01-20 14:15:00 | 1393.60 | 1504.62 | 1503.34 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-21 09:15:00 | 1390.60 | 1502.38 | 1502.23 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 1140m) |
| Stop hit — per-position SL triggered | 2026-01-21 10:15:00 | 1384.00 | 1501.20 | 1501.64 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 3 — SELL (started 2026-01-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-21 10:15:00 | 1384.00 | 1501.20 | 1501.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-01 13:15:00 | 1370.70 | 1457.67 | 1477.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-04 09:15:00 | 1459.10 | 1449.75 | 1471.23 | EMA200 retest candle locked |
| Cross detected — sustain check pending | 2026-02-06 09:15:00 | 1437.50 | 1449.54 | 1469.68 | ENTRY1 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2026-02-06 10:15:00 | 1448.00 | 1449.53 | 1469.57 | ENTRY1 sustain failed after 60m |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-10 09:15:00 | 1463.40 | 1450.33 | 1468.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 09:15:00 | 1463.40 | 1450.33 | 1468.71 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2026-02-10 11:15:00 | 1459.80 | 1450.57 | 1468.65 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-10 13:15:00 | 1454.90 | 1450.71 | 1468.54 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 120m) |
| Cross detected — sustain check pending | 2026-02-12 10:15:00 | 1461.60 | 1452.02 | 1468.26 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-12 12:15:00 | 1454.90 | 1452.10 | 1468.14 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 120m) |
| Cross detected — sustain check pending | 2026-05-05 10:15:00 | 1452.90 | 1378.28 | 1392.34 | ENTRY2 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2026-05-05 12:15:00 | 1466.40 | 1379.96 | 1393.05 | ENTRY2 sustain failed after 120m |
| Cross detected — sustain check pending | 2026-05-05 13:15:00 | 1460.40 | 1380.76 | 1393.38 | ENTRY2 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2026-05-05 14:15:00 | 1464.40 | 1381.59 | 1393.74 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2026-05-05 15:15:00 | 1462.50 | 1382.40 | 1394.08 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-06 09:15:00 | 1455.50 | 1383.13 | 1394.39 | SELL ENTRY2 attempt 3/4 (retest2 break sustained 1080m) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2024-09-20 15:15:00 | 1486.03 | 2024-09-23 09:15:00 | 1497.83 | STOP_HIT | 1.00 | -0.79% |
| SELL | retest2 | 2024-09-25 11:15:00 | 1485.13 | 2024-09-25 13:15:00 | 1495.98 | STOP_HIT | 1.00 | -0.73% |
| SELL | retest2 | 2024-09-30 15:15:00 | 1479.00 | 2024-11-13 09:15:00 | 1257.15 | PARTIAL | 0.50 | 15.00% |
| SELL | retest2 | 2024-09-30 15:15:00 | 1479.00 | 2024-12-02 13:15:00 | 1308.55 | STOP_HIT | 0.50 | 11.52% |
| SELL | retest2 | 2025-07-14 14:15:00 | 1483.30 | 2025-07-15 10:15:00 | 1493.40 | STOP_HIT | 1.00 | -0.68% |
| SELL | retest2 | 2025-07-17 14:15:00 | 1478.10 | 2025-08-11 15:15:00 | 1384.70 | STOP_HIT | 1.00 | 6.32% |
| SELL | retest2 | 2025-08-25 09:15:00 | 1408.70 | 2025-10-17 10:15:00 | 1416.20 | STOP_HIT | 1.00 | -0.53% |
| SELL | retest2 | 2025-08-26 11:15:00 | 1395.40 | 2025-10-17 11:15:00 | 1421.10 | STOP_HIT | 1.00 | -1.84% |
| SELL | retest2 | 2025-09-19 11:15:00 | 1408.00 | 2025-10-17 11:15:00 | 1421.10 | STOP_HIT | 1.00 | -0.93% |
| SELL | retest2 | 2025-09-22 09:15:00 | 1409.00 | 2025-10-17 11:15:00 | 1421.10 | STOP_HIT | 1.00 | -0.86% |
| SELL | retest2 | 2025-09-22 15:15:00 | 1390.90 | 2025-10-17 11:15:00 | 1421.10 | STOP_HIT | 1.00 | -2.17% |
| SELL | retest2 | 2026-01-21 09:15:00 | 1390.60 | 2026-01-21 10:15:00 | 1384.00 | STOP_HIT | 1.00 | 0.47% |
