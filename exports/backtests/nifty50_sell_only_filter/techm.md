# TECHM (TECHM.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-06-06 09:15:00 → 2026-05-06 15:30:00 (4997 bars)
- **Last close:** 1466.70
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
| ALERT3 | 5 |
| PENDING | 15 |
| PENDING_CANCEL | 3 |
| ENTRY1 | 3 |
| ENTRY2 | 9 |
| PARTIAL | 6 |
| TARGET_HIT | 1 |
| STOP_HIT | 11 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 18 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 11 / 7
- **Target hits / Stop hits / Partials:** 1 / 11 / 6
- **Avg / median % per leg:** 8.81% / 13.75%
- **Sum % (uncompounded):** 158.64%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 18 | 11 | 61.1% | 1 | 11 | 6 | 8.81% | 158.6% |
| BUY @ 2nd Alert (retest1) | 5 | 3 | 60.0% | 0 | 3 | 2 | 6.42% | 32.1% |
| BUY @ 3rd Alert (retest2) | 13 | 8 | 61.5% | 1 | 8 | 4 | 9.73% | 126.5% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 5 | 3 | 60.0% | 0 | 3 | 2 | 6.42% | 32.1% |
| retest2 (combined) | 13 | 8 | 61.5% | 1 | 8 | 4 | 9.73% | 126.5% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-12-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-04 12:15:00 | 1219.80 | 1190.02 | 1189.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-04 14:15:00 | 1224.00 | 1190.66 | 1190.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-13 11:15:00 | 1199.00 | 1203.45 | 1197.45 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-13 11:15:00 | 1199.00 | 1203.45 | 1197.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-13 11:15:00 | 1199.00 | 1203.45 | 1197.45 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2023-12-13 14:15:00 | 1216.20 | 1203.61 | 1197.62 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-13 15:15:00 | 1216.10 | 1203.73 | 1197.71 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Partial book — 50% qty @ 15%, trail SL → entry | 2024-01-15 09:15:00 | 1398.51 | 1251.56 | 1233.93 | Partial book 0.50 @ 15%; trail SL->entry alert=retest2 |
| CROSSOVER_SKIP | 2024-03-27 11:15:00 | 1255.25 | 1282.56 | 1282.65 | HTF filter: close above htf_sma |
| Stop hit — per-position SL triggered | 2024-04-15 10:15:00 | 1216.10 | 1268.20 | 1274.05 | SL hit qty=0.50 sl=1216.10 alert=retest2 |
| Cross detected — sustain check pending | 2024-04-26 09:15:00 | 1312.30 | 1239.08 | 1256.11 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-26 10:15:00 | 1293.65 | 1239.62 | 1256.29 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2024-05-21 10:15:00 | 1314.00 | 1264.99 | 1264.95 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 2 — BUY (started 2024-05-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-21 10:15:00 | 1314.00 | 1264.99 | 1264.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-21 11:15:00 | 1315.90 | 1265.49 | 1265.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-30 09:15:00 | 1276.00 | 1286.37 | 1277.09 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-30 09:15:00 | 1276.00 | 1286.37 | 1277.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-30 09:15:00 | 1276.00 | 1286.37 | 1277.09 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2024-06-06 09:15:00 | 1294.05 | 1275.82 | 1272.80 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-06 10:15:00 | 1292.10 | 1275.99 | 1272.90 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Partial book — 50% qty @ 15%, trail SL → entry | 2024-07-02 09:15:00 | 1485.91 | 1363.36 | 1327.93 | Partial book 0.50 @ 15%; trail SL->entry alert=retest2 |
| Target hit — 30% from entry | 2024-10-14 11:15:00 | 1679.73 | 1616.61 | 1579.61 | Target hit (30%) qty=0.50 alert=retest2 |
| CROSSOVER_SKIP | 2025-01-28 14:15:00 | 1651.60 | 1690.74 | 1690.87 | HTF filter: close above htf_sma |
| Cross detected — sustain check pending | 2025-04-07 14:15:00 | 1289.15 | 1456.75 | 1529.06 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-07 15:15:00 | 1288.00 | 1455.07 | 1527.86 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-04-09 09:15:00 | 1275.00 | 1443.93 | 1519.33 | SL hit qty=1.00 sl=1275.00 alert=retest2 |
| Cross detected — sustain check pending | 2025-04-11 09:15:00 | 1295.85 | 1432.78 | 1511.06 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-11 10:15:00 | 1301.05 | 1431.47 | 1510.01 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-04-11 13:15:00 | 1275.00 | 1427.26 | 1506.72 | SL hit qty=1.00 sl=1275.00 alert=retest2 |
| Cross detected — sustain check pending | 2025-04-15 09:15:00 | 1295.10 | 1423.11 | 1503.45 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-15 10:15:00 | 1294.10 | 1421.83 | 1502.40 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-15 11:15:00 | 1292.70 | 1420.54 | 1501.36 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-04-15 14:15:00 | 1300.60 | 1416.91 | 1498.32 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-15 15:15:00 | 1304.90 | 1415.79 | 1497.36 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-04-16 09:15:00 | 1286.80 | 1414.68 | 1496.39 | SL hit qty=1.00 sl=1286.80 alert=retest2 |
| Cross detected — sustain check pending | 2025-04-16 13:15:00 | 1306.70 | 1410.10 | 1492.46 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-16 14:15:00 | 1308.30 | 1409.09 | 1491.55 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-04-17 09:15:00 | 1286.80 | 1406.89 | 1489.62 | SL hit qty=1.00 sl=1286.80 alert=retest2 |
| Cross detected — sustain check pending | 2025-04-17 14:15:00 | 1306.90 | 1401.51 | 1484.85 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-17 15:15:00 | 1304.80 | 1400.55 | 1483.96 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Partial book — 50% qty @ 15%, trail SL → entry | 2025-04-29 10:15:00 | 1488.21 | 1411.41 | 1473.09 | Partial book 0.50 @ 15%; trail SL->entry alert=retest2 |
| Partial book — 50% qty @ 15%, trail SL → entry | 2025-04-29 13:15:00 | 1500.52 | 1413.87 | 1473.41 | Partial book 0.50 @ 15%; trail SL->entry alert=retest2 |
| Stop hit — per-position SL triggered | 2025-05-20 14:15:00 | 1575.50 | 1503.54 | 1503.44 | Force close (CROSSOVER_FLIP) qty=0.50 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-05-20 14:15:00 | 1575.50 | 1503.54 | 1503.44 | Force close (CROSSOVER_FLIP) qty=0.50 alert=retest2 |

### Cycle 3 — BUY (started 2025-05-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-20 14:15:00 | 1575.50 | 1503.54 | 1503.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-21 09:15:00 | 1590.00 | 1505.14 | 1504.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-03 09:15:00 | 1536.80 | 1540.67 | 1525.25 | EMA200 retest candle locked |
| Cross detected — sustain check pending | 2025-06-04 13:15:00 | 1565.00 | 1541.83 | 1526.67 | ENTRY1 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-06-04 14:15:00 | 1556.80 | 1541.98 | 1526.82 | ENTRY1 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-06-05 11:15:00 | 1568.20 | 1542.67 | 1527.47 | ENTRY1 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-06-05 12:15:00 | 1561.90 | 1542.86 | 1527.64 | ENTRY1 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-06-06 09:15:00 | 1565.50 | 1543.65 | 1528.34 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-06 10:15:00 | 1571.50 | 1543.93 | 1528.55 | BUY ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 09:15:00 | 1599.80 | 1639.80 | 1604.60 | EMA400 retest candle locked |
| Stop hit — per-position SL triggered | 2025-07-10 09:15:00 | 1604.60 | 1639.80 | 1604.60 | SL hit qty=1.00 sl=1604.60 alert=retest1 |
| CROSSOVER_SKIP | 2025-07-28 10:15:00 | 1460.30 | 1585.46 | 1585.46 | slope filter: EMA200 not falling 0.50% over 350 bars |

### Cycle 4 — BUY (started 2025-12-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-03 11:15:00 | 1545.90 | 1472.95 | 1472.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-04 09:15:00 | 1574.70 | 1476.68 | 1474.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-08 14:15:00 | 1578.30 | 1579.95 | 1546.96 | EMA200 retest candle locked |
| Cross detected — sustain check pending | 2026-01-09 09:15:00 | 1588.60 | 1580.02 | 1547.32 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-09 10:15:00 | 1591.40 | 1580.13 | 1547.54 | BUY ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2026-01-09 14:15:00 | 1584.70 | 1580.28 | 1548.26 | ENTRY1 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-01-09 15:15:00 | 1582.20 | 1580.30 | 1548.43 | ENTRY1 sustain failed after 60m |
| Cross detected — sustain check pending | 2026-01-12 11:15:00 | 1583.50 | 1580.06 | 1548.78 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-12 12:15:00 | 1585.70 | 1580.11 | 1548.97 | BUY ENTRY1 attempt 2/4 (retest1 break sustained 60m) |
| Partial book — 50% qty @ 15%, trail SL → entry | 2026-02-03 09:15:00 | 1830.11 | 1656.59 | 1605.80 | Partial book 0.50 @ 15%; trail SL->entry alert=retest1 |
| Partial book — 50% qty @ 15%, trail SL → entry | 2026-02-03 09:15:00 | 1823.55 | 1656.59 | 1605.80 | Partial book 0.50 @ 15%; trail SL->entry alert=retest1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 09:15:00 | 1628.90 | 1660.54 | 1609.57 | EMA400 retest candle locked |
| Stop hit — per-position SL triggered | 2026-02-12 09:15:00 | 1591.40 | 1650.36 | 1613.60 | SL hit qty=0.50 sl=1591.40 alert=retest1 |
| Stop hit — per-position SL triggered | 2026-02-12 09:15:00 | 1585.70 | 1650.36 | 1613.60 | SL hit qty=0.50 sl=1585.70 alert=retest1 |
| CROSSOVER_SKIP | 2026-02-23 14:15:00 | 1439.60 | 1586.36 | 1586.57 | slope filter: EMA200 not falling 0.50% over 350 bars |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2023-12-13 15:15:00 | 1216.10 | 2024-01-15 09:15:00 | 1398.51 | PARTIAL | 0.50 | 15.00% |
| BUY | retest2 | 2023-12-13 15:15:00 | 1216.10 | 2024-04-15 10:15:00 | 1216.10 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest2 | 2024-04-26 10:15:00 | 1293.65 | 2024-05-21 10:15:00 | 1314.00 | STOP_HIT | 1.00 | 1.57% |
| BUY | retest2 | 2024-06-06 10:15:00 | 1292.10 | 2024-07-02 09:15:00 | 1485.91 | PARTIAL | 0.50 | 15.00% |
| BUY | retest2 | 2024-06-06 10:15:00 | 1292.10 | 2024-10-14 11:15:00 | 1679.73 | TARGET_HIT | 0.50 | 30.00% |
| BUY | retest2 | 2025-04-07 15:15:00 | 1288.00 | 2025-04-09 09:15:00 | 1275.00 | STOP_HIT | 1.00 | -1.01% |
| BUY | retest2 | 2025-04-11 10:15:00 | 1301.05 | 2025-04-11 13:15:00 | 1275.00 | STOP_HIT | 1.00 | -2.00% |
| BUY | retest2 | 2025-04-15 10:15:00 | 1294.10 | 2025-04-16 09:15:00 | 1286.80 | STOP_HIT | 1.00 | -0.56% |
| BUY | retest2 | 2025-04-15 15:15:00 | 1304.90 | 2025-04-17 09:15:00 | 1286.80 | STOP_HIT | 1.00 | -1.39% |
| BUY | retest2 | 2025-04-16 14:15:00 | 1308.30 | 2025-04-29 10:15:00 | 1488.21 | PARTIAL | 0.50 | 13.75% |
| BUY | retest2 | 2025-04-17 15:15:00 | 1304.80 | 2025-04-29 13:15:00 | 1500.52 | PARTIAL | 0.50 | 15.00% |
| BUY | retest2 | 2025-04-16 14:15:00 | 1308.30 | 2025-05-20 14:15:00 | 1575.50 | STOP_HIT | 0.50 | 20.42% |
| BUY | retest2 | 2025-04-17 15:15:00 | 1304.80 | 2025-05-20 14:15:00 | 1575.50 | STOP_HIT | 0.50 | 20.75% |
| BUY | retest1 | 2025-06-06 10:15:00 | 1571.50 | 2025-07-10 09:15:00 | 1604.60 | STOP_HIT | 1.00 | 2.11% |
| BUY | retest1 | 2026-01-09 10:15:00 | 1591.40 | 2026-02-03 09:15:00 | 1830.11 | PARTIAL | 0.50 | 15.00% |
| BUY | retest1 | 2026-01-12 12:15:00 | 1585.70 | 2026-02-03 09:15:00 | 1823.55 | PARTIAL | 0.50 | 15.00% |
| BUY | retest1 | 2026-01-09 10:15:00 | 1591.40 | 2026-02-12 09:15:00 | 1591.40 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-01-12 12:15:00 | 1585.70 | 2026-02-12 09:15:00 | 1585.70 | STOP_HIT | 0.50 | 0.00% |
