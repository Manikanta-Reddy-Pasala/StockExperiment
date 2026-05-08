# ADANIPORTS (ADANIPORTS)

## Backtest Summary

- **Source:** Fyers history API (1H bars)
- **Window:** 2023-06-08 09:15:00 → 2026-05-08 15:30:00 (4997 bars)
- **Last close:** 1760.40
- **Strategy:** EMA 200/400 1H crossover (v2 BTC rules)
- **Target:** entry × (1 ± 30%)
- **Partial:** 50% qty booked @ 15%, trail SL → entry
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 3 attempts each at retest1 and retest2

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 8 |
| ALERT1 | 6 |
| ALERT2 | 5 |
| ALERT2_SKIP | 4 |
| ALERT3 | 5 |
| PENDING | 15 |
| PENDING_CANCEL | 5 |
| ENTRY1 | 1 |
| ENTRY2 | 9 |
| PARTIAL | 1 |
| TARGET_HIT | 3 |
| STOP_HIT | 7 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 11 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 4 / 7
- **Target hits / Stop hits / Partials:** 3 / 7 / 1
- **Avg / median % per leg:** 1.20% / -2.05%
- **Sum % (uncompounded):** 13.21%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 5 | 2 | 40.0% | 2 | 3 | 0 | 2.85% | 14.3% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 5 | 2 | 40.0% | 2 | 3 | 0 | 2.85% | 14.3% |
| SELL (all) | 6 | 2 | 33.3% | 1 | 4 | 1 | -0.17% | -1.0% |
| SELL @ 2nd Alert (retest1) | 2 | 2 | 100.0% | 1 | 0 | 1 | 7.50% | 15.0% |
| SELL @ 3rd Alert (retest2) | 4 | 0 | 0.0% | 0 | 4 | 0 | -4.01% | -16.0% |
| retest1 (combined) | 2 | 2 | 100.0% | 1 | 0 | 1 | 7.50% | 15.0% |
| retest2 (combined) | 9 | 2 | 22.2% | 2 | 7 | 0 | -0.20% | -1.8% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-09-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-19 15:15:00 | 1410.90 | 1463.18 | 1463.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-07 09:15:00 | 1385.90 | 1454.86 | 1458.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-30 10:15:00 | 1411.00 | 1399.66 | 1422.94 | EMA200 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2024-10-31 10:15:00 | 1383.70 | 1399.55 | 1422.08 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-31 11:15:00 | 1384.50 | 1399.40 | 1421.90 | SELL ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Partial book — 50% qty @ 15%, trail SL → entry | 2024-11-05 09:15:00 | 1315.27 | 1394.23 | 1417.94 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Target hit — 30% from entry | 2024-11-21 09:15:00 | 1246.05 | 1353.97 | 1388.20 | Target hit (10%) qty=0.50 alert=retest1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-06 12:15:00 | 1142.10 | 1102.39 | 1141.14 | EMA400 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2025-03-12 11:15:00 | 1109.40 | 1111.87 | 1141.33 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-03-12 12:15:00 | 1122.25 | 1111.98 | 1141.24 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-03-13 14:15:00 | 1119.55 | 1113.20 | 1140.58 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-13 15:15:00 | 1119.20 | 1113.26 | 1140.47 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-03-18 09:15:00 | 1152.65 | 1115.07 | 1140.33 | SL hit (close>static) qty=1.00 sl=1146.40 alert=retest2 |
| Cross detected — sustain check pending | 2025-04-07 09:15:00 | 1085.85 | 1154.72 | 1155.43 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-07 10:15:00 | 1105.15 | 1154.23 | 1155.18 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-04-09 09:15:00 | 1114.20 | 1149.82 | 1152.83 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-09 10:15:00 | 1118.35 | 1149.50 | 1152.66 | SELL ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-04-11 09:15:00 | 1165.05 | 1148.72 | 1152.16 | SL hit (close>static) qty=1.00 sl=1146.40 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-04-11 09:15:00 | 1165.05 | 1148.72 | 1152.16 | SL hit (close>static) qty=1.00 sl=1146.40 alert=retest2 |

### Cycle 2 — BUY (started 2025-04-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-16 11:15:00 | 1215.20 | 1155.55 | 1155.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-16 12:15:00 | 1219.80 | 1156.19 | 1155.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-16 09:15:00 | 1396.70 | 1397.69 | 1333.28 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-19 12:15:00 | 1336.70 | 1394.09 | 1338.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 12:15:00 | 1336.70 | 1394.09 | 1338.72 | EMA400 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2025-06-20 10:15:00 | 1348.30 | 1391.51 | 1338.79 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-06-20 11:15:00 | 1344.20 | 1391.04 | 1338.81 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-06-20 14:15:00 | 1349.40 | 1389.62 | 1338.88 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-20 15:15:00 | 1349.30 | 1389.22 | 1338.93 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-06-23 10:15:00 | 1350.40 | 1388.42 | 1339.03 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-23 11:15:00 | 1358.90 | 1388.13 | 1339.13 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-08-01 15:15:00 | 1349.80 | 1411.35 | 1390.90 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-04 09:15:00 | 1362.80 | 1410.87 | 1390.76 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 3960m) |
| Stop hit — per-position SL triggered | 2025-08-07 10:15:00 | 1331.00 | 1402.28 | 1388.35 | SL hit (close<static) qty=1.00 sl=1332.50 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-07 10:15:00 | 1331.00 | 1402.28 | 1388.35 | SL hit (close<static) qty=1.00 sl=1332.50 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-07 10:15:00 | 1331.00 | 1402.28 | 1388.35 | SL hit (close<static) qty=1.00 sl=1332.50 alert=retest2 |

### Cycle 3 — SELL (started 2025-08-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-14 15:15:00 | 1301.40 | 1376.60 | 1376.85 | EMA200 below EMA400 |

### Cycle 4 — BUY (started 2025-09-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-17 14:15:00 | 1406.90 | 1368.35 | 1368.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-18 09:15:00 | 1409.40 | 1369.13 | 1368.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-29 10:15:00 | 1387.90 | 1391.38 | 1381.44 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-29 11:15:00 | 1380.00 | 1391.27 | 1381.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 11:15:00 | 1380.00 | 1391.27 | 1381.44 | EMA400 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2025-09-29 13:15:00 | 1390.20 | 1391.22 | 1381.51 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-09-29 14:15:00 | 1382.30 | 1391.13 | 1381.51 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-09-30 09:15:00 | 1398.80 | 1391.13 | 1381.61 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-30 10:15:00 | 1398.60 | 1391.20 | 1381.69 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-10-08 11:15:00 | 1390.80 | 1395.58 | 1385.67 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-08 12:15:00 | 1399.50 | 1395.62 | 1385.74 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Target hit — 30% from entry | 2025-12-01 09:15:00 | 1538.46 | 1474.22 | 1447.65 | Target hit (10%) qty=1.00 alert=retest2 |
| Target hit — 30% from entry | 2025-12-01 09:15:00 | 1539.45 | 1474.22 | 1447.65 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 5 — SELL (started 2026-01-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-19 15:15:00 | 1403.00 | 1465.15 | 1465.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-20 10:15:00 | 1395.90 | 1463.86 | 1464.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-03 09:15:00 | 1502.30 | 1428.77 | 1443.91 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-03 09:15:00 | 1502.30 | 1428.77 | 1443.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 09:15:00 | 1502.30 | 1428.77 | 1443.91 | EMA400 retest candle locked (from downside) |

### Cycle 6 — BUY (started 2026-02-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-09 09:15:00 | 1562.70 | 1457.76 | 1457.39 | EMA200 above EMA400 |

### Cycle 7 — SELL (started 2026-03-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-13 11:15:00 | 1362.40 | 1476.40 | 1476.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 14:15:00 | 1352.40 | 1448.69 | 1461.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 09:15:00 | 1457.10 | 1403.09 | 1430.63 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-08 09:15:00 | 1457.10 | 1403.09 | 1430.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 09:15:00 | 1457.10 | 1403.09 | 1430.63 | EMA400 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2026-04-08 11:15:00 | 1452.50 | 1404.14 | 1430.88 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-04-08 12:15:00 | 1456.20 | 1404.65 | 1431.00 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2026-04-08 13:15:00 | 1453.50 | 1405.14 | 1431.12 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-08 14:15:00 | 1452.70 | 1405.61 | 1431.22 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2026-04-13 09:15:00 | 1454.40 | 1413.21 | 1433.19 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-04-13 10:15:00 | 1461.70 | 1413.69 | 1433.33 | ENTRY2 sustain failed after 60m |
| Stop hit — per-position SL triggered | 2026-04-15 09:15:00 | 1503.00 | 1417.30 | 1434.58 | SL hit (close>static) qty=1.00 sl=1485.60 alert=retest2 |

### Cycle 8 — BUY (started 2026-04-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-21 12:15:00 | 1600.80 | 1450.09 | 1449.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-27 09:15:00 | 1629.40 | 1481.25 | 1466.34 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2024-10-31 11:15:00 | 1384.50 | 2024-11-05 09:15:00 | 1315.27 | PARTIAL | 0.50 | 5.00% |
| SELL | retest1 | 2024-10-31 11:15:00 | 1384.50 | 2024-11-21 09:15:00 | 1246.05 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-03-13 15:15:00 | 1119.20 | 2025-03-18 09:15:00 | 1152.65 | STOP_HIT | 1.00 | -2.99% |
| SELL | retest2 | 2025-04-07 10:15:00 | 1105.15 | 2025-04-11 09:15:00 | 1165.05 | STOP_HIT | 1.00 | -5.42% |
| SELL | retest2 | 2025-04-09 10:15:00 | 1118.35 | 2025-04-11 09:15:00 | 1165.05 | STOP_HIT | 1.00 | -4.18% |
| BUY | retest2 | 2025-06-20 15:15:00 | 1349.30 | 2025-08-07 10:15:00 | 1331.00 | STOP_HIT | 1.00 | -1.36% |
| BUY | retest2 | 2025-06-23 11:15:00 | 1358.90 | 2025-08-07 10:15:00 | 1331.00 | STOP_HIT | 1.00 | -2.05% |
| BUY | retest2 | 2025-08-04 09:15:00 | 1362.80 | 2025-08-07 10:15:00 | 1331.00 | STOP_HIT | 1.00 | -2.33% |
| BUY | retest2 | 2025-09-30 10:15:00 | 1398.60 | 2025-12-01 09:15:00 | 1538.46 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-10-08 12:15:00 | 1399.50 | 2025-12-01 09:15:00 | 1539.45 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2026-04-08 14:15:00 | 1452.70 | 2026-04-15 09:15:00 | 1503.00 | STOP_HIT | 1.00 | -3.46% |
