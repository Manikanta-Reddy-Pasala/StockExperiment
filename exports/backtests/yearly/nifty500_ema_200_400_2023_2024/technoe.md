# Techno Electric & Engineering Company Ltd. (TECHNOE)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 1268.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 11 |
| ALERT1 | 9 |
| ALERT2 | 8 |
| ALERT2_SKIP | 2 |
| ALERT3 | 47 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 32 |
| PARTIAL | 1 |
| TARGET_HIT | 6 |
| STOP_HIT | 26 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 33 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 8 / 25
- **Target hits / Stop hits / Partials:** 6 / 26 / 1
- **Avg / median % per leg:** -0.89% / -3.14%
- **Sum % (uncompounded):** -29.41%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 21 | 6 | 28.6% | 6 | 15 | 0 | -0.12% | -2.6% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 21 | 6 | 28.6% | 6 | 15 | 0 | -0.12% | -2.6% |
| SELL (all) | 12 | 2 | 16.7% | 0 | 11 | 1 | -2.23% | -26.8% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 12 | 2 | 16.7% | 0 | 11 | 1 | -2.23% | -26.8% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 33 | 8 | 24.2% | 6 | 26 | 1 | -0.89% | -29.4% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-07-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-06 11:15:00 | 377.20 | 367.06 | 367.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-06 12:15:00 | 382.05 | 367.21 | 367.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-13 09:15:00 | 497.80 | 503.45 | 470.42 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-25 09:15:00 | 501.55 | 514.24 | 498.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-25 09:15:00 | 501.55 | 514.24 | 498.17 | EMA400 retest candle locked (from upside) |

### Cycle 2 — SELL (started 2024-03-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-14 11:15:00 | 643.65 | 743.11 | 743.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-19 10:15:00 | 630.10 | 725.25 | 734.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-26 09:15:00 | 737.50 | 709.45 | 724.38 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-26 09:15:00 | 737.50 | 709.45 | 724.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-26 09:15:00 | 737.50 | 709.45 | 724.38 | EMA400 retest candle locked (from downside) |

### Cycle 3 — BUY (started 2024-04-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-04 11:15:00 | 804.95 | 735.96 | 735.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-05 09:15:00 | 819.95 | 739.55 | 737.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-29 10:15:00 | 1013.70 | 1025.51 | 941.42 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-29 11:00:00 | 1013.70 | 1025.51 | 941.42 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-06 14:15:00 | 1551.00 | 1627.93 | 1545.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-06 14:30:00 | 1540.00 | 1627.93 | 1545.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 11:15:00 | 1538.00 | 1613.70 | 1559.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-19 11:45:00 | 1538.05 | 1613.70 | 1559.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 12:15:00 | 1518.70 | 1612.75 | 1559.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-19 13:00:00 | 1518.70 | 1612.75 | 1559.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 14:15:00 | 1525.00 | 1605.80 | 1557.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-20 15:00:00 | 1525.00 | 1605.80 | 1557.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-23 09:15:00 | 1552.85 | 1604.92 | 1558.00 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-24 10:15:00 | 1576.95 | 1600.33 | 1557.28 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-24 13:00:00 | 1570.00 | 1599.61 | 1557.55 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-03 10:15:00 | 1567.80 | 1598.72 | 1564.55 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-07 09:15:00 | 1626.95 | 1592.37 | 1563.43 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-07 10:15:00 | 1583.55 | 1592.85 | 1563.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-07 11:00:00 | 1583.55 | 1592.85 | 1563.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-07 12:15:00 | 1526.00 | 1591.99 | 1563.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-07 13:00:00 | 1526.00 | 1591.99 | 1563.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-07 13:15:00 | 1543.95 | 1591.51 | 1563.71 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-07 14:45:00 | 1559.40 | 1591.13 | 1563.66 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-08 09:45:00 | 1561.95 | 1590.65 | 1563.69 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2024-10-09 13:15:00 | 1715.34 | 1594.16 | 1566.91 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 4 — SELL (started 2024-11-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-14 13:15:00 | 1439.70 | 1583.30 | 1583.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-12 09:15:00 | 1416.50 | 1507.75 | 1534.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-16 11:15:00 | 1506.95 | 1498.03 | 1527.62 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-16 11:30:00 | 1518.95 | 1498.03 | 1527.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-17 09:15:00 | 1544.30 | 1498.90 | 1527.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-17 09:45:00 | 1546.95 | 1498.90 | 1527.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-17 10:15:00 | 1529.35 | 1499.20 | 1527.34 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-17 12:00:00 | 1508.00 | 1499.29 | 1527.24 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-18 09:45:00 | 1509.40 | 1498.70 | 1526.25 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-18 13:15:00 | 1551.95 | 1499.57 | 1526.14 | SL hit (close>static) qty=1.00 sl=1544.30 alert=retest2 |

### Cycle 5 — BUY (started 2025-01-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-02 09:15:00 | 1692.25 | 1545.34 | 1544.97 | EMA200 above EMA400 |

### Cycle 6 — SELL (started 2025-01-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-10 14:15:00 | 1395.90 | 1546.97 | 1547.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-13 09:15:00 | 1387.80 | 1543.96 | 1545.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-19 09:15:00 | 994.10 | 990.44 | 1108.37 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-19 10:00:00 | 994.10 | 990.44 | 1108.37 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-17 09:15:00 | 1034.40 | 994.45 | 1056.86 | EMA400 retest candle locked (from downside) |

### Cycle 7 — BUY (started 2025-05-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-16 11:15:00 | 1256.60 | 1083.58 | 1082.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-16 14:15:00 | 1274.30 | 1089.03 | 1085.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-22 10:15:00 | 1525.10 | 1525.92 | 1435.23 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-22 11:00:00 | 1525.10 | 1525.92 | 1435.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 10:15:00 | 1441.40 | 1518.48 | 1440.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-25 11:00:00 | 1441.40 | 1518.48 | 1440.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 11:15:00 | 1454.80 | 1517.84 | 1440.49 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-25 12:45:00 | 1457.60 | 1517.19 | 1440.55 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-28 10:15:00 | 1429.50 | 1513.48 | 1440.56 | SL hit (close<static) qty=1.00 sl=1438.50 alert=retest2 |

### Cycle 8 — SELL (started 2025-09-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-26 11:15:00 | 1309.90 | 1454.30 | 1454.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-26 12:15:00 | 1298.10 | 1452.74 | 1453.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-01 09:15:00 | 1039.40 | 1008.26 | 1086.24 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-01 10:00:00 | 1039.40 | 1008.26 | 1086.24 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 09:15:00 | 1074.80 | 1014.88 | 1074.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-09 10:00:00 | 1074.80 | 1014.88 | 1074.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 10:15:00 | 1088.40 | 1015.61 | 1075.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-09 11:00:00 | 1088.40 | 1015.61 | 1075.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 11:15:00 | 1089.10 | 1016.34 | 1075.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-09 12:00:00 | 1089.10 | 1016.34 | 1075.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 10:15:00 | 1084.70 | 1032.45 | 1077.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-12 11:00:00 | 1084.70 | 1032.45 | 1077.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 11:15:00 | 1085.00 | 1032.97 | 1078.02 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-12 15:00:00 | 1081.95 | 1034.49 | 1078.11 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-13 11:15:00 | 1027.85 | 1035.07 | 1077.55 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-13 11:15:00 | 1041.05 | 1035.07 | 1077.55 | SL hit (close>static) qty=0.50 sl=1035.07 alert=retest2 |

### Cycle 9 — BUY (started 2026-03-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-19 09:15:00 | 1117.80 | 1100.19 | 1100.18 | EMA200 above EMA400 |

### Cycle 10 — SELL (started 2026-03-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-23 09:15:00 | 1033.70 | 1100.18 | 1100.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-23 10:15:00 | 1028.00 | 1099.46 | 1099.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-25 10:15:00 | 1092.90 | 1092.08 | 1095.97 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-25 11:00:00 | 1092.90 | 1092.08 | 1095.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 09:15:00 | 1075.60 | 1072.01 | 1083.87 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-09 09:30:00 | 1065.00 | 1072.33 | 1083.62 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-09 13:00:00 | 1064.45 | 1072.36 | 1083.47 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-10 09:15:00 | 1115.50 | 1072.85 | 1083.50 | SL hit (close>static) qty=1.00 sl=1103.30 alert=retest2 |

### Cycle 11 — BUY (started 2026-04-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-16 15:15:00 | 1205.00 | 1092.67 | 1092.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-17 09:15:00 | 1219.00 | 1093.93 | 1093.19 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-09-24 10:15:00 | 1576.95 | 2024-10-09 13:15:00 | 1715.34 | TARGET_HIT | 1.00 | 8.78% |
| BUY | retest2 | 2024-09-24 13:00:00 | 1570.00 | 2024-10-09 13:15:00 | 1718.15 | TARGET_HIT | 1.00 | 9.44% |
| BUY | retest2 | 2024-10-03 10:15:00 | 1567.80 | 2024-10-10 09:15:00 | 1727.00 | TARGET_HIT | 1.00 | 10.15% |
| BUY | retest2 | 2024-10-07 09:15:00 | 1626.95 | 2024-10-10 09:15:00 | 1724.58 | TARGET_HIT | 1.00 | 6.00% |
| BUY | retest2 | 2024-10-07 14:45:00 | 1559.40 | 2024-10-10 10:15:00 | 1734.65 | TARGET_HIT | 1.00 | 11.24% |
| BUY | retest2 | 2024-10-08 09:45:00 | 1561.95 | 2024-10-10 11:15:00 | 1789.65 | TARGET_HIT | 1.00 | 14.58% |
| BUY | retest2 | 2024-10-23 09:45:00 | 1555.00 | 2024-10-25 09:15:00 | 1483.35 | STOP_HIT | 1.00 | -4.61% |
| BUY | retest2 | 2024-10-23 13:00:00 | 1569.25 | 2024-10-25 09:15:00 | 1483.35 | STOP_HIT | 1.00 | -5.47% |
| BUY | retest2 | 2024-11-01 18:00:00 | 1599.00 | 2024-11-04 09:15:00 | 1538.30 | STOP_HIT | 1.00 | -3.80% |
| BUY | retest2 | 2024-11-01 18:30:00 | 1597.90 | 2024-11-04 09:15:00 | 1538.30 | STOP_HIT | 1.00 | -3.73% |
| BUY | retest2 | 2024-11-05 14:45:00 | 1593.05 | 2024-11-11 12:15:00 | 1584.15 | STOP_HIT | 1.00 | -0.56% |
| BUY | retest2 | 2024-11-06 09:15:00 | 1627.90 | 2024-11-13 09:15:00 | 1450.00 | STOP_HIT | 1.00 | -10.93% |
| BUY | retest2 | 2024-11-06 10:15:00 | 1641.35 | 2024-11-13 09:15:00 | 1450.00 | STOP_HIT | 1.00 | -11.66% |
| SELL | retest2 | 2024-12-17 12:00:00 | 1508.00 | 2024-12-18 13:15:00 | 1551.95 | STOP_HIT | 1.00 | -2.91% |
| SELL | retest2 | 2024-12-18 09:45:00 | 1509.40 | 2024-12-18 13:15:00 | 1551.95 | STOP_HIT | 1.00 | -2.82% |
| BUY | retest2 | 2025-07-25 12:45:00 | 1457.60 | 2025-07-28 10:15:00 | 1429.50 | STOP_HIT | 1.00 | -1.93% |
| BUY | retest2 | 2025-07-29 15:00:00 | 1462.80 | 2025-08-01 09:15:00 | 1432.60 | STOP_HIT | 1.00 | -2.06% |
| BUY | retest2 | 2025-07-30 11:15:00 | 1458.90 | 2025-08-01 09:15:00 | 1432.60 | STOP_HIT | 1.00 | -1.80% |
| BUY | retest2 | 2025-07-30 12:30:00 | 1459.40 | 2025-08-01 09:15:00 | 1432.60 | STOP_HIT | 1.00 | -1.84% |
| BUY | retest2 | 2025-07-31 13:00:00 | 1477.00 | 2025-08-01 14:15:00 | 1419.70 | STOP_HIT | 1.00 | -3.88% |
| BUY | retest2 | 2025-08-14 11:00:00 | 1473.90 | 2025-09-19 10:15:00 | 1423.00 | STOP_HIT | 1.00 | -3.45% |
| BUY | retest2 | 2025-08-18 10:30:00 | 1475.50 | 2025-09-19 10:15:00 | 1423.00 | STOP_HIT | 1.00 | -3.56% |
| BUY | retest2 | 2025-08-18 11:15:00 | 1474.80 | 2025-09-19 10:15:00 | 1423.00 | STOP_HIT | 1.00 | -3.51% |
| SELL | retest2 | 2026-02-12 15:00:00 | 1081.95 | 2026-02-13 11:15:00 | 1027.85 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-12 15:00:00 | 1081.95 | 2026-02-13 11:15:00 | 1041.05 | STOP_HIT | 0.50 | 3.78% |
| SELL | retest2 | 2026-03-09 09:15:00 | 1065.70 | 2026-03-10 09:15:00 | 1100.20 | STOP_HIT | 1.00 | -3.24% |
| SELL | retest2 | 2026-03-12 09:30:00 | 1078.10 | 2026-03-12 11:15:00 | 1115.20 | STOP_HIT | 1.00 | -3.44% |
| SELL | retest2 | 2026-03-12 10:15:00 | 1081.30 | 2026-03-12 11:15:00 | 1115.20 | STOP_HIT | 1.00 | -3.14% |
| SELL | retest2 | 2026-03-13 09:15:00 | 1086.10 | 2026-03-17 09:15:00 | 1131.20 | STOP_HIT | 1.00 | -4.15% |
| SELL | retest2 | 2026-03-13 11:45:00 | 1093.20 | 2026-03-17 09:15:00 | 1131.20 | STOP_HIT | 1.00 | -3.48% |
| SELL | retest2 | 2026-03-17 15:00:00 | 1093.20 | 2026-03-18 10:15:00 | 1124.60 | STOP_HIT | 1.00 | -2.87% |
| SELL | retest2 | 2026-04-09 09:30:00 | 1065.00 | 2026-04-10 09:15:00 | 1115.50 | STOP_HIT | 1.00 | -4.74% |
| SELL | retest2 | 2026-04-09 13:00:00 | 1064.45 | 2026-04-10 09:15:00 | 1115.50 | STOP_HIT | 1.00 | -4.80% |
