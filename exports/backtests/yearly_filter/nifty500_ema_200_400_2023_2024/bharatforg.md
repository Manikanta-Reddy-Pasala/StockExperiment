# Bharat Forge Ltd. (BHARATFORG)

## Backtest Summary

- **Window:** 2022-04-08 09:15:00 → 2026-05-08 15:15:00 (7047 bars)
- **Last close:** 1984.00
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
| ALERT2 | 7 |
| ALERT2_SKIP | 2 |
| ALERT3 | 53 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 49 |
| PARTIAL | 6 |
| TARGET_HIT | 13 |
| STOP_HIT | 36 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 55 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 22 / 33
- **Target hits / Stop hits / Partials:** 13 / 36 / 6
- **Avg / median % per leg:** 1.52% / -1.35%
- **Sum % (uncompounded):** 83.43%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 25 | 10 | 40.0% | 10 | 15 | 0 | 2.54% | 63.6% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 25 | 10 | 40.0% | 10 | 15 | 0 | 2.54% | 63.6% |
| SELL (all) | 30 | 12 | 40.0% | 3 | 21 | 6 | 0.66% | 19.8% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 30 | 12 | 40.0% | 3 | 21 | 6 | 0.66% | 19.8% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 55 | 22 | 40.0% | 13 | 36 | 6 | 1.52% | 83.4% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-06-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-14 11:15:00 | 825.80 | 793.23 | 793.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-14 12:15:00 | 826.50 | 793.56 | 793.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-23 15:15:00 | 804.40 | 804.92 | 799.89 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-06-26 09:15:00 | 806.20 | 804.92 | 799.89 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-26 09:15:00 | 804.50 | 804.92 | 799.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-26 10:00:00 | 804.50 | 804.92 | 799.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-25 09:15:00 | 1036.00 | 1085.54 | 1044.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-25 10:00:00 | 1036.00 | 1085.54 | 1044.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-25 10:15:00 | 1033.75 | 1085.03 | 1044.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-25 11:00:00 | 1033.75 | 1085.03 | 1044.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-03 10:15:00 | 1040.25 | 1062.10 | 1040.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-03 10:30:00 | 1040.50 | 1062.10 | 1040.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-03 11:15:00 | 1032.40 | 1061.81 | 1040.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-03 12:00:00 | 1032.40 | 1061.81 | 1040.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-03 12:15:00 | 1031.00 | 1061.50 | 1040.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-03 12:30:00 | 1030.05 | 1061.50 | 1040.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-06 10:15:00 | 1036.40 | 1060.12 | 1040.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-06 10:30:00 | 1036.45 | 1060.12 | 1040.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-07 09:15:00 | 1060.60 | 1060.27 | 1040.78 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-21 10:15:00 | 1075.25 | 1055.37 | 1043.09 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-21 11:30:00 | 1075.65 | 1055.74 | 1043.40 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-21 13:45:00 | 1074.15 | 1056.12 | 1043.71 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-21 14:30:00 | 1073.60 | 1056.33 | 1043.88 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Target hit | 2023-12-05 09:15:00 | 1182.78 | 1085.14 | 1063.04 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 2 — SELL (started 2024-02-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-28 13:15:00 | 1153.10 | 1180.65 | 1180.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-29 11:15:00 | 1140.55 | 1179.04 | 1179.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-01 14:15:00 | 1184.85 | 1177.44 | 1179.01 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-01 14:15:00 | 1184.85 | 1177.44 | 1179.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-01 14:15:00 | 1184.85 | 1177.44 | 1179.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-01 15:00:00 | 1184.85 | 1177.44 | 1179.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-01 15:15:00 | 1185.00 | 1177.52 | 1179.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-02 09:15:00 | 1179.55 | 1177.52 | 1179.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-02 11:15:00 | 1179.35 | 1177.54 | 1179.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-02 11:30:00 | 1177.45 | 1177.54 | 1179.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-02 12:15:00 | 1182.00 | 1177.58 | 1179.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-04 09:15:00 | 1178.40 | 1177.58 | 1179.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-04 09:15:00 | 1179.00 | 1177.60 | 1179.05 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-06 09:45:00 | 1165.00 | 1178.83 | 1179.59 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-11 09:15:00 | 1162.95 | 1178.57 | 1179.41 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-11 13:45:00 | 1168.35 | 1178.01 | 1179.10 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-14 09:15:00 | 1106.75 | 1172.39 | 1176.11 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-14 09:15:00 | 1104.80 | 1172.39 | 1176.11 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-14 09:15:00 | 1109.93 | 1172.39 | 1176.11 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-04-01 10:15:00 | 1148.55 | 1144.33 | 1158.28 | SL hit (close>ema200) qty=0.50 sl=1144.33 alert=retest2 |

### Cycle 3 — BUY (started 2024-04-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-23 14:15:00 | 1206.70 | 1163.86 | 1163.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-24 09:15:00 | 1216.00 | 1164.81 | 1164.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-10 10:15:00 | 1621.90 | 1627.72 | 1529.03 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-10 10:30:00 | 1614.05 | 1627.72 | 1529.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 09:15:00 | 1585.00 | 1627.62 | 1550.28 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-22 10:15:00 | 1600.70 | 1627.62 | 1550.28 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-23 12:30:00 | 1596.00 | 1626.50 | 1553.47 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-23 14:00:00 | 1587.75 | 1626.11 | 1553.64 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-23 14:30:00 | 1599.55 | 1625.84 | 1553.87 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Target hit | 2024-07-29 10:15:00 | 1746.53 | 1627.48 | 1562.70 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 4 — SELL (started 2024-09-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-27 14:15:00 | 1524.30 | 1580.65 | 1580.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-30 13:15:00 | 1514.50 | 1577.25 | 1579.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-05 14:15:00 | 1461.95 | 1460.97 | 1500.84 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-05 15:00:00 | 1461.95 | 1460.97 | 1500.84 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-20 09:15:00 | 1161.15 | 1093.82 | 1149.13 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-27 09:15:00 | 1143.10 | 1120.97 | 1155.22 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-27 10:15:00 | 1193.95 | 1122.20 | 1155.49 | SL hit (close>static) qty=1.00 sl=1173.95 alert=retest2 |

### Cycle 5 — BUY (started 2025-05-15 13:15:00)

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

### Cycle 6 — SELL (started 2025-07-31 14:15:00)

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

### Cycle 7 — BUY (started 2025-09-19 14:15:00)

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
| SELL | retest2 | 2023-06-05 13:00:00 | 800.20 | 2023-06-07 09:15:00 | 806.50 | STOP_HIT | 1.00 | -0.79% |
| SELL | retest2 | 2023-06-06 09:45:00 | 800.15 | 2023-06-07 09:15:00 | 806.50 | STOP_HIT | 1.00 | -0.79% |
| SELL | retest2 | 2023-06-06 10:30:00 | 800.00 | 2023-06-07 09:15:00 | 806.50 | STOP_HIT | 1.00 | -0.81% |
| BUY | retest2 | 2023-11-21 10:15:00 | 1075.25 | 2023-12-05 09:15:00 | 1182.78 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2023-11-21 11:30:00 | 1075.65 | 2023-12-05 09:15:00 | 1183.22 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2023-11-21 13:45:00 | 1074.15 | 2023-12-05 09:15:00 | 1181.57 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2023-11-21 14:30:00 | 1073.60 | 2023-12-05 09:15:00 | 1180.96 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-02-27 09:15:00 | 1178.95 | 2024-02-28 10:15:00 | 1163.00 | STOP_HIT | 1.00 | -1.35% |
| BUY | retest2 | 2024-02-27 12:15:00 | 1179.95 | 2024-02-28 10:15:00 | 1163.00 | STOP_HIT | 1.00 | -1.44% |
| BUY | retest2 | 2024-02-27 14:30:00 | 1177.30 | 2024-02-28 10:15:00 | 1163.00 | STOP_HIT | 1.00 | -1.21% |
| SELL | retest2 | 2024-03-06 09:45:00 | 1165.00 | 2024-03-14 09:15:00 | 1106.75 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-03-11 09:15:00 | 1162.95 | 2024-03-14 09:15:00 | 1104.80 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-03-11 13:45:00 | 1168.35 | 2024-03-14 09:15:00 | 1109.93 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-03-06 09:45:00 | 1165.00 | 2024-04-01 10:15:00 | 1148.55 | STOP_HIT | 0.50 | 1.41% |
| SELL | retest2 | 2024-03-11 09:15:00 | 1162.95 | 2024-04-01 10:15:00 | 1148.55 | STOP_HIT | 0.50 | 1.24% |
| SELL | retest2 | 2024-03-11 13:45:00 | 1168.35 | 2024-04-01 10:15:00 | 1148.55 | STOP_HIT | 0.50 | 1.69% |
| SELL | retest2 | 2024-04-09 10:30:00 | 1169.75 | 2024-04-10 13:15:00 | 1176.35 | STOP_HIT | 1.00 | -0.56% |
| SELL | retest2 | 2024-04-10 11:15:00 | 1169.25 | 2024-04-16 09:15:00 | 1185.65 | STOP_HIT | 1.00 | -1.40% |
| SELL | retest2 | 2024-04-12 14:45:00 | 1169.50 | 2024-04-16 09:15:00 | 1185.65 | STOP_HIT | 1.00 | -1.38% |
| SELL | retest2 | 2024-04-15 10:00:00 | 1159.25 | 2024-04-16 09:15:00 | 1185.65 | STOP_HIT | 1.00 | -2.28% |
| SELL | retest2 | 2024-04-15 13:30:00 | 1169.25 | 2024-04-16 10:15:00 | 1188.00 | STOP_HIT | 1.00 | -1.60% |
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
