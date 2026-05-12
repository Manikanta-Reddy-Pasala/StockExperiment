# Bharat Forge Ltd. (BHARATFORG)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 1984.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 4 |
| ALERT1 | 4 |
| ALERT2 | 5 |
| ALERT2_SKIP | 1 |
| ALERT3 | 32 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 31 |
| PARTIAL | 3 |
| TARGET_HIT | 9 |
| STOP_HIT | 22 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 34 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 12 / 22
- **Target hits / Stop hits / Partials:** 9 / 22 / 3
- **Avg / median % per leg:** 1.11% / -2.16%
- **Sum % (uncompounded):** 37.71%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 18 | 6 | 33.3% | 6 | 12 | 0 | 1.53% | 27.6% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 18 | 6 | 33.3% | 6 | 12 | 0 | 1.53% | 27.6% |
| SELL (all) | 16 | 6 | 37.5% | 3 | 10 | 3 | 0.63% | 10.1% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 16 | 6 | 37.5% | 3 | 10 | 3 | 0.63% | 10.1% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 34 | 12 | 35.3% | 9 | 22 | 3 | 1.11% | 37.7% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-09-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-27 14:15:00 | 1524.30 | 1580.65 | 1580.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-30 13:15:00 | 1514.50 | 1577.25 | 1579.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-05 14:15:00 | 1461.95 | 1460.97 | 1500.84 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-05 15:00:00 | 1461.95 | 1460.97 | 1500.84 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-20 09:15:00 | 1161.15 | 1093.82 | 1149.13 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-27 09:15:00 | 1143.10 | 1120.97 | 1155.22 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-27 10:15:00 | 1193.95 | 1122.20 | 1155.49 | SL hit (close>static) qty=1.00 sl=1173.95 alert=retest2 |

### Cycle 2 — BUY (started 2025-05-15 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-15 13:15:00 | 1248.00 | 1131.84 | 1131.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-16 09:15:00 | 1263.50 | 1135.38 | 1133.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-01 12:15:00 | 1272.40 | 1274.93 | 1237.24 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-01 12:45:00 | 1273.10 | 1274.93 | 1237.24 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 10:15:00 | 1242.60 | 1281.37 | 1248.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-10 11:00:00 | 1242.60 | 1281.37 | 1248.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 11:15:00 | 1232.00 | 1280.88 | 1248.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-10 12:00:00 | 1232.00 | 1280.88 | 1248.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 3 — SELL (started 2025-07-31 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-31 14:15:00 | 1167.90 | 1232.53 | 1232.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-31 15:15:00 | 1161.90 | 1231.82 | 1232.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-08 09:15:00 | 1199.90 | 1160.49 | 1183.68 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-08 09:15:00 | 1199.90 | 1160.49 | 1183.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 09:15:00 | 1199.90 | 1160.49 | 1183.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-08 10:00:00 | 1199.90 | 1160.49 | 1183.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 10:15:00 | 1202.30 | 1160.91 | 1183.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-08 10:30:00 | 1202.80 | 1160.91 | 1183.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 11:15:00 | 1177.30 | 1163.66 | 1184.28 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-09 13:45:00 | 1174.00 | 1163.91 | 1184.20 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-10 09:15:00 | 1231.40 | 1164.93 | 1184.41 | SL hit (close>static) qty=1.00 sl=1187.00 alert=retest2 |

### Cycle 4 — BUY (started 2025-09-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-19 14:15:00 | 1270.90 | 1198.34 | 1198.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-23 09:15:00 | 1302.40 | 1221.22 | 1212.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-08 12:15:00 | 1374.90 | 1375.93 | 1327.93 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-08 12:45:00 | 1374.40 | 1375.93 | 1327.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 09:15:00 | 1396.10 | 1439.32 | 1402.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-20 09:45:00 | 1395.90 | 1439.32 | 1402.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 10:15:00 | 1392.20 | 1438.85 | 1402.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-20 10:45:00 | 1392.00 | 1438.85 | 1402.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 15:15:00 | 1393.10 | 1436.63 | 1402.09 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-21 09:15:00 | 1399.20 | 1436.63 | 1402.09 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-21 10:15:00 | 1379.70 | 1435.64 | 1401.94 | SL hit (close<static) qty=1.00 sl=1385.50 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-07-22 10:15:00 | 1600.70 | 2024-07-29 10:15:00 | 1746.53 | TARGET_HIT | 1.00 | 9.11% |
| BUY | retest2 | 2024-07-23 12:30:00 | 1596.00 | 2024-08-01 09:15:00 | 1760.77 | TARGET_HIT | 1.00 | 10.32% |
| BUY | retest2 | 2024-07-23 14:00:00 | 1587.75 | 2024-08-01 09:15:00 | 1755.60 | TARGET_HIT | 1.00 | 10.57% |
| BUY | retest2 | 2024-07-23 14:30:00 | 1599.55 | 2024-08-01 09:15:00 | 1759.51 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-08-09 09:15:00 | 1649.90 | 2024-08-14 10:15:00 | 1561.00 | STOP_HIT | 1.00 | -5.39% |
| BUY | retest2 | 2024-08-22 10:30:00 | 1619.45 | 2024-08-28 14:15:00 | 1584.10 | STOP_HIT | 1.00 | -2.18% |
| BUY | retest2 | 2024-08-23 09:15:00 | 1628.20 | 2024-08-29 11:15:00 | 1558.15 | STOP_HIT | 1.00 | -4.30% |
| BUY | retest2 | 2024-08-23 10:30:00 | 1622.10 | 2024-08-29 11:15:00 | 1558.15 | STOP_HIT | 1.00 | -3.94% |
| BUY | retest2 | 2024-08-28 11:30:00 | 1600.40 | 2024-08-29 11:15:00 | 1558.15 | STOP_HIT | 1.00 | -2.64% |
| BUY | retest2 | 2024-09-04 15:00:00 | 1608.85 | 2024-09-06 09:15:00 | 1568.95 | STOP_HIT | 1.00 | -2.48% |
| BUY | retest2 | 2024-09-05 11:00:00 | 1600.70 | 2024-09-06 09:15:00 | 1568.95 | STOP_HIT | 1.00 | -1.98% |
| BUY | retest2 | 2024-09-13 09:45:00 | 1599.65 | 2024-09-18 12:15:00 | 1574.95 | STOP_HIT | 1.00 | -1.54% |
| BUY | retest2 | 2024-09-23 09:30:00 | 1605.15 | 2024-09-23 13:15:00 | 1577.75 | STOP_HIT | 1.00 | -1.71% |
| SELL | retest2 | 2025-03-27 09:15:00 | 1143.10 | 2025-03-27 10:15:00 | 1193.95 | STOP_HIT | 1.00 | -4.45% |
| SELL | retest2 | 2025-04-01 11:30:00 | 1145.40 | 2025-04-03 09:15:00 | 1088.13 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-04-01 13:15:00 | 1146.00 | 2025-04-03 09:15:00 | 1088.70 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-04-01 11:30:00 | 1145.40 | 2025-04-04 09:15:00 | 1030.86 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-04-01 13:15:00 | 1146.00 | 2025-04-04 09:15:00 | 1031.40 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-04-03 09:15:00 | 1122.00 | 2025-04-04 09:15:00 | 1065.90 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-04-03 09:15:00 | 1122.00 | 2025-04-07 09:15:00 | 1009.80 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-04-23 10:45:00 | 1100.00 | 2025-04-23 13:15:00 | 1131.90 | STOP_HIT | 1.00 | -2.90% |
| SELL | retest2 | 2025-04-25 09:30:00 | 1097.70 | 2025-04-29 11:15:00 | 1129.70 | STOP_HIT | 1.00 | -2.92% |
| SELL | retest2 | 2025-04-25 14:15:00 | 1098.90 | 2025-04-29 11:15:00 | 1129.70 | STOP_HIT | 1.00 | -2.80% |
| SELL | retest2 | 2025-04-28 10:15:00 | 1093.30 | 2025-04-29 11:15:00 | 1129.70 | STOP_HIT | 1.00 | -3.33% |
| SELL | retest2 | 2025-04-29 10:15:00 | 1122.30 | 2025-05-09 09:15:00 | 1160.60 | STOP_HIT | 1.00 | -3.41% |
| SELL | retest2 | 2025-04-30 09:15:00 | 1120.30 | 2025-05-09 09:15:00 | 1160.60 | STOP_HIT | 1.00 | -3.60% |
| SELL | retest2 | 2025-04-30 11:15:00 | 1123.30 | 2025-05-09 09:15:00 | 1160.60 | STOP_HIT | 1.00 | -3.32% |
| SELL | retest2 | 2025-05-08 09:45:00 | 1123.80 | 2025-05-09 09:15:00 | 1160.60 | STOP_HIT | 1.00 | -3.27% |
| SELL | retest2 | 2025-09-09 13:45:00 | 1174.00 | 2025-09-10 09:15:00 | 1231.40 | STOP_HIT | 1.00 | -4.89% |
| BUY | retest2 | 2026-01-21 09:15:00 | 1399.20 | 2026-01-21 10:15:00 | 1379.70 | STOP_HIT | 1.00 | -1.39% |
| BUY | retest2 | 2026-01-22 09:15:00 | 1417.50 | 2026-02-01 13:15:00 | 1379.40 | STOP_HIT | 1.00 | -2.69% |
| BUY | retest2 | 2026-02-01 12:45:00 | 1409.80 | 2026-02-01 13:15:00 | 1379.40 | STOP_HIT | 1.00 | -2.16% |
| BUY | retest2 | 2026-02-02 09:15:00 | 1401.40 | 2026-02-03 09:15:00 | 1541.54 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-02-02 13:30:00 | 1416.30 | 2026-02-03 09:15:00 | 1557.93 | TARGET_HIT | 1.00 | 10.00% |
