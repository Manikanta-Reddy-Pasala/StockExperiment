# Deepak Fertilisers & Petrochemicals Corp. Ltd. (DEEPAKFERT)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 1342.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 8 |
| ALERT1 | 6 |
| ALERT2 | 5 |
| ALERT2_SKIP | 4 |
| ALERT3 | 30 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 4 |
| ENTRY2 | 20 |
| PARTIAL | 11 |
| TARGET_HIT | 7 |
| STOP_HIT | 17 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 35 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 23 / 12
- **Target hits / Stop hits / Partials:** 7 / 17 / 11
- **Avg / median % per leg:** 2.99% / 4.03%
- **Sum % (uncompounded):** 104.76%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 17 | 9 | 52.9% | 5 | 8 | 4 | 2.69% | 45.7% |
| BUY @ 2nd Alert (retest1) | 8 | 8 | 100.0% | 4 | 0 | 4 | 7.50% | 60.0% |
| BUY @ 3rd Alert (retest2) | 9 | 1 | 11.1% | 1 | 8 | 0 | -1.59% | -14.3% |
| SELL (all) | 18 | 14 | 77.8% | 2 | 9 | 7 | 3.28% | 59.1% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 18 | 14 | 77.8% | 2 | 9 | 7 | 3.28% | 59.1% |
| retest1 (combined) | 8 | 8 | 100.0% | 4 | 0 | 4 | 7.50% | 60.0% |
| retest2 (combined) | 27 | 15 | 55.6% | 3 | 17 | 7 | 1.66% | 44.8% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-05-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-29 13:15:00 | 545.50 | 553.82 | 553.86 | EMA200 below EMA400 |

### Cycle 2 — BUY (started 2024-05-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-29 14:15:00 | 574.50 | 554.03 | 553.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-31 12:15:00 | 580.00 | 555.53 | 554.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-04 09:15:00 | 550.45 | 556.63 | 555.35 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-04 09:15:00 | 550.45 | 556.63 | 555.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 09:15:00 | 550.45 | 556.63 | 555.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-04 09:30:00 | 545.60 | 556.63 | 555.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 10:15:00 | 531.50 | 556.38 | 555.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-04 11:00:00 | 531.50 | 556.38 | 555.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 15:15:00 | 553.95 | 554.66 | 554.40 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-06 09:15:00 | 559.05 | 554.66 | 554.40 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2024-06-18 13:15:00 | 614.96 | 572.48 | 564.40 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 3 — SELL (started 2025-01-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-21 12:15:00 | 1155.80 | 1197.05 | 1197.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-22 09:15:00 | 1145.00 | 1195.74 | 1196.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-23 09:15:00 | 1200.65 | 1192.85 | 1195.01 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-23 09:15:00 | 1200.65 | 1192.85 | 1195.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 09:15:00 | 1200.65 | 1192.85 | 1195.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 10:00:00 | 1200.65 | 1192.85 | 1195.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 10:15:00 | 1204.30 | 1192.97 | 1195.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 10:30:00 | 1220.10 | 1192.97 | 1195.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 12:15:00 | 1188.70 | 1192.88 | 1194.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 12:30:00 | 1190.35 | 1192.88 | 1194.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 10:15:00 | 1189.55 | 1182.22 | 1189.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-29 11:00:00 | 1189.55 | 1182.22 | 1189.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 11:15:00 | 1193.15 | 1182.33 | 1189.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-29 11:45:00 | 1197.50 | 1182.33 | 1189.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 12:15:00 | 1191.40 | 1182.42 | 1189.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-29 12:45:00 | 1194.25 | 1182.42 | 1189.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 10:15:00 | 1190.85 | 1168.04 | 1179.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-07 11:00:00 | 1190.85 | 1168.04 | 1179.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 11:15:00 | 1171.20 | 1168.07 | 1179.53 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-10 09:15:00 | 1154.00 | 1168.38 | 1179.46 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-11 09:15:00 | 1096.30 | 1164.95 | 1177.29 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-02-12 09:15:00 | 1038.60 | 1159.11 | 1173.89 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 4 — BUY (started 2025-04-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-15 12:15:00 | 1245.00 | 1122.37 | 1121.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-15 13:15:00 | 1252.40 | 1123.66 | 1122.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-07 09:15:00 | 1221.20 | 1223.41 | 1184.65 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-07 11:00:00 | 1254.00 | 1223.71 | 1184.99 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-07 11:30:00 | 1250.90 | 1224.07 | 1185.36 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-07 12:00:00 | 1259.20 | 1224.07 | 1185.36 | BUY ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-07 13:00:00 | 1259.30 | 1224.42 | 1185.73 | BUY ENTRY1 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-12 10:15:00 | 1316.70 | 1231.17 | 1192.78 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-12 10:15:00 | 1313.45 | 1231.17 | 1192.78 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-12 10:15:00 | 1322.16 | 1231.17 | 1192.78 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-12 10:15:00 | 1322.27 | 1231.17 | 1192.78 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Target hit | 2025-05-15 09:15:00 | 1379.40 | 1245.36 | 1203.84 | Target hit (10%) qty=0.50 alert=retest1 |

### Cycle 5 — SELL (started 2025-09-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-01 12:15:00 | 1394.40 | 1501.64 | 1501.80 | EMA200 below EMA400 |

### Cycle 6 — BUY (started 2025-10-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-03 11:15:00 | 1564.00 | 1485.14 | 1485.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-03 14:15:00 | 1575.70 | 1487.56 | 1486.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-10 10:15:00 | 1496.80 | 1500.23 | 1493.43 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-10 10:15:00 | 1496.80 | 1500.23 | 1493.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 10:15:00 | 1496.80 | 1500.23 | 1493.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-10 10:45:00 | 1500.30 | 1500.23 | 1493.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 14:15:00 | 1499.30 | 1500.25 | 1493.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-10 15:00:00 | 1499.30 | 1500.25 | 1493.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 15:15:00 | 1493.00 | 1500.18 | 1493.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-13 09:15:00 | 1479.80 | 1500.18 | 1493.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 09:15:00 | 1485.00 | 1500.02 | 1493.53 | EMA400 retest candle locked (from upside) |

### Cycle 7 — SELL (started 2025-10-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-16 11:15:00 | 1430.70 | 1487.87 | 1487.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-20 10:15:00 | 1419.60 | 1481.72 | 1484.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-28 09:15:00 | 1521.40 | 1471.77 | 1478.88 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-28 09:15:00 | 1521.40 | 1471.77 | 1478.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 09:15:00 | 1521.40 | 1471.77 | 1478.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-28 09:45:00 | 1531.60 | 1471.77 | 1478.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 10:15:00 | 1522.00 | 1472.27 | 1479.09 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-29 10:30:00 | 1507.50 | 1475.06 | 1480.28 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-29 14:15:00 | 1501.00 | 1475.80 | 1480.58 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-31 11:15:00 | 1505.50 | 1477.88 | 1481.38 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-31 14:00:00 | 1507.80 | 1478.65 | 1481.72 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 09:15:00 | 1494.40 | 1479.04 | 1481.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-03 09:30:00 | 1492.60 | 1479.04 | 1481.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 10:15:00 | 1493.40 | 1479.18 | 1481.93 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-03 11:15:00 | 1493.20 | 1479.18 | 1481.93 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-03 12:30:00 | 1492.90 | 1479.45 | 1482.03 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-04 09:15:00 | 1512.90 | 1479.92 | 1482.22 | SL hit (close>static) qty=1.00 sl=1502.40 alert=retest2 |

### Cycle 8 — BUY (started 2026-04-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-22 13:15:00 | 1267.75 | 1073.09 | 1072.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-22 14:15:00 | 1270.05 | 1075.05 | 1073.23 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-06-06 09:15:00 | 559.05 | 2024-06-18 13:15:00 | 614.96 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-02-10 09:15:00 | 1154.00 | 2025-02-11 09:15:00 | 1096.30 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-10 09:15:00 | 1154.00 | 2025-02-12 09:15:00 | 1038.60 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-04-11 09:30:00 | 1169.40 | 2025-04-11 10:15:00 | 1211.40 | STOP_HIT | 1.00 | -3.59% |
| BUY | retest1 | 2025-05-07 11:00:00 | 1254.00 | 2025-05-12 10:15:00 | 1316.70 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2025-05-07 11:30:00 | 1250.90 | 2025-05-12 10:15:00 | 1313.45 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2025-05-07 12:00:00 | 1259.20 | 2025-05-12 10:15:00 | 1322.16 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2025-05-07 13:00:00 | 1259.30 | 2025-05-12 10:15:00 | 1322.27 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2025-05-07 11:00:00 | 1254.00 | 2025-05-15 09:15:00 | 1379.40 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest1 | 2025-05-07 11:30:00 | 1250.90 | 2025-05-15 09:15:00 | 1375.99 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest1 | 2025-05-07 12:00:00 | 1259.20 | 2025-05-15 09:15:00 | 1385.12 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest1 | 2025-05-07 13:00:00 | 1259.30 | 2025-05-15 09:15:00 | 1385.23 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2025-07-29 11:00:00 | 1571.50 | 2025-08-07 13:15:00 | 1499.30 | STOP_HIT | 1.00 | -4.59% |
| BUY | retest2 | 2025-07-31 09:30:00 | 1576.30 | 2025-08-07 13:15:00 | 1499.30 | STOP_HIT | 1.00 | -4.88% |
| BUY | retest2 | 2025-08-04 12:15:00 | 1555.00 | 2025-08-07 13:15:00 | 1499.30 | STOP_HIT | 1.00 | -3.58% |
| BUY | retest2 | 2025-08-05 09:15:00 | 1575.70 | 2025-08-07 13:15:00 | 1499.30 | STOP_HIT | 1.00 | -4.85% |
| BUY | retest2 | 2025-08-22 10:00:00 | 1538.50 | 2025-08-25 11:15:00 | 1516.40 | STOP_HIT | 1.00 | -1.44% |
| BUY | retest2 | 2025-08-22 12:00:00 | 1547.50 | 2025-08-25 11:15:00 | 1516.40 | STOP_HIT | 1.00 | -2.01% |
| BUY | retest2 | 2025-08-22 14:30:00 | 1539.20 | 2025-08-25 11:15:00 | 1516.40 | STOP_HIT | 1.00 | -1.48% |
| BUY | retest2 | 2025-08-25 09:15:00 | 1539.60 | 2025-08-25 11:15:00 | 1516.40 | STOP_HIT | 1.00 | -1.51% |
| SELL | retest2 | 2025-10-29 10:30:00 | 1507.50 | 2025-11-04 09:15:00 | 1512.90 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest2 | 2025-10-29 14:15:00 | 1501.00 | 2025-11-04 09:15:00 | 1512.90 | STOP_HIT | 1.00 | -0.79% |
| SELL | retest2 | 2025-10-31 11:15:00 | 1505.50 | 2025-11-04 14:15:00 | 1504.60 | STOP_HIT | 1.00 | 0.06% |
| SELL | retest2 | 2025-10-31 14:00:00 | 1507.80 | 2025-11-06 09:15:00 | 1432.12 | PARTIAL | 0.50 | 5.02% |
| SELL | retest2 | 2025-11-03 11:15:00 | 1493.20 | 2025-11-06 09:15:00 | 1425.95 | PARTIAL | 0.50 | 4.50% |
| SELL | retest2 | 2025-11-03 12:30:00 | 1492.90 | 2025-11-06 09:15:00 | 1430.22 | PARTIAL | 0.50 | 4.20% |
| SELL | retest2 | 2025-11-04 13:45:00 | 1492.60 | 2025-11-06 09:15:00 | 1432.41 | PARTIAL | 0.50 | 4.03% |
| SELL | retest2 | 2025-11-06 09:15:00 | 1438.80 | 2025-11-11 10:15:00 | 1366.86 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-31 14:00:00 | 1507.80 | 2025-11-19 14:15:00 | 1451.90 | STOP_HIT | 0.50 | 3.71% |
| SELL | retest2 | 2025-11-03 11:15:00 | 1493.20 | 2025-11-19 14:15:00 | 1451.90 | STOP_HIT | 0.50 | 2.77% |
| SELL | retest2 | 2025-11-03 12:30:00 | 1492.90 | 2025-11-19 14:15:00 | 1451.90 | STOP_HIT | 0.50 | 2.75% |
| SELL | retest2 | 2025-11-04 13:45:00 | 1492.60 | 2025-11-19 14:15:00 | 1451.90 | STOP_HIT | 0.50 | 2.73% |
| SELL | retest2 | 2025-11-06 09:15:00 | 1438.80 | 2025-11-19 14:15:00 | 1451.90 | STOP_HIT | 0.50 | -0.91% |
| SELL | retest2 | 2025-11-21 09:15:00 | 1432.10 | 2025-11-24 15:15:00 | 1360.49 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-21 09:15:00 | 1432.10 | 2025-12-08 09:15:00 | 1288.89 | TARGET_HIT | 0.50 | 10.00% |
