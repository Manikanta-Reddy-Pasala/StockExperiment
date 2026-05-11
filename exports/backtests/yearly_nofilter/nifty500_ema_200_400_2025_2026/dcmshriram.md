# DCM Shriram Ltd. (DCMSHRIRAM)

## Backtest Summary

- **Window:** 2024-04-08 09:15:00 → 2026-05-08 15:15:00 (3598 bars)
- **Last close:** 1237.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 7 |
| ALERT1 | 6 |
| ALERT2 | 5 |
| ALERT2_SKIP | 3 |
| ALERT3 | 21 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 22 |
| PARTIAL | 4 |
| TARGET_HIT | 7 |
| STOP_HIT | 18 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 26 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 11 / 15
- **Target hits / Stop hits / Partials:** 7 / 15 / 4
- **Avg / median % per leg:** 2.39% / -0.18%
- **Sum % (uncompounded):** 62.22%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 5 | 3 | 60.0% | 3 | 2 | 0 | 5.65% | 28.2% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 5 | 3 | 60.0% | 3 | 2 | 0 | 5.65% | 28.2% |
| SELL (all) | 21 | 8 | 38.1% | 4 | 13 | 4 | 1.62% | 34.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 21 | 8 | 38.1% | 4 | 13 | 4 | 1.62% | 34.0% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 26 | 11 | 42.3% | 7 | 15 | 4 | 2.39% | 62.2% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-06-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-03 09:15:00 | 1072.90 | 1047.50 | 1047.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-03 10:15:00 | 1081.10 | 1047.83 | 1047.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-04 14:15:00 | 1048.90 | 1049.67 | 1048.55 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-04 14:15:00 | 1048.90 | 1049.67 | 1048.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 14:15:00 | 1048.90 | 1049.67 | 1048.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-04 14:45:00 | 1047.20 | 1049.67 | 1048.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 15:15:00 | 1050.00 | 1049.67 | 1048.56 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-05 09:15:00 | 1057.10 | 1049.67 | 1048.56 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-05 14:30:00 | 1054.70 | 1050.22 | 1048.87 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-06 09:15:00 | 1066.90 | 1050.25 | 1048.89 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Target hit | 2025-06-18 09:15:00 | 1162.81 | 1077.83 | 1064.41 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 2 — SELL (started 2025-09-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-24 14:15:00 | 1194.10 | 1263.01 | 1263.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-25 12:15:00 | 1181.60 | 1259.54 | 1261.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-15 15:15:00 | 1211.00 | 1209.10 | 1229.44 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-16 09:15:00 | 1202.50 | 1209.10 | 1229.44 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 15:15:00 | 1221.10 | 1208.88 | 1228.63 | EMA400 retest candle locked (from downside) |

### Cycle 3 — BUY (started 2025-10-31 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-31 14:15:00 | 1275.50 | 1241.05 | 1241.00 | EMA200 above EMA400 |

### Cycle 4 — SELL (started 2025-11-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-17 12:15:00 | 1211.60 | 1241.24 | 1241.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-18 09:15:00 | 1197.00 | 1239.92 | 1240.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-21 10:15:00 | 1242.40 | 1229.36 | 1235.04 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-21 10:15:00 | 1242.40 | 1229.36 | 1235.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 10:15:00 | 1242.40 | 1229.36 | 1235.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-21 11:00:00 | 1242.40 | 1229.36 | 1235.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 11:15:00 | 1257.00 | 1229.64 | 1235.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-21 11:30:00 | 1255.50 | 1229.64 | 1235.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 14:15:00 | 1266.50 | 1226.21 | 1232.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-26 15:00:00 | 1266.50 | 1226.21 | 1232.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 15:15:00 | 1246.70 | 1226.42 | 1232.83 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-27 09:15:00 | 1226.70 | 1226.42 | 1232.83 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-11 09:15:00 | 1290.10 | 1224.18 | 1229.57 | SL hit (close>static) qty=1.00 sl=1274.70 alert=retest2 |

### Cycle 5 — BUY (started 2025-12-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-26 12:15:00 | 1259.40 | 1233.86 | 1233.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-01 09:15:00 | 1261.10 | 1236.90 | 1235.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-06 11:15:00 | 1237.20 | 1241.32 | 1237.92 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-06 11:15:00 | 1237.20 | 1241.32 | 1237.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 11:15:00 | 1237.20 | 1241.32 | 1237.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-06 12:00:00 | 1237.20 | 1241.32 | 1237.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 12:15:00 | 1237.80 | 1241.28 | 1237.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-06 12:45:00 | 1235.50 | 1241.28 | 1237.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 13:15:00 | 1226.40 | 1241.13 | 1237.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-06 14:00:00 | 1226.40 | 1241.13 | 1237.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 14:15:00 | 1229.80 | 1241.02 | 1237.82 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-07 10:00:00 | 1233.80 | 1240.84 | 1237.76 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-07 13:45:00 | 1233.10 | 1240.55 | 1237.67 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-07 14:15:00 | 1222.60 | 1240.37 | 1237.60 | SL hit (close<static) qty=1.00 sl=1226.40 alert=retest2 |

### Cycle 6 — SELL (started 2026-01-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-09 12:15:00 | 1180.00 | 1234.75 | 1234.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-16 14:15:00 | 1162.80 | 1220.48 | 1227.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-29 12:15:00 | 1179.20 | 1178.73 | 1201.83 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-29 13:00:00 | 1179.20 | 1178.73 | 1201.83 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 09:15:00 | 1173.40 | 1175.89 | 1197.65 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-04 09:15:00 | 1162.60 | 1175.65 | 1196.89 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-04 09:45:00 | 1154.90 | 1175.42 | 1196.67 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-04 13:00:00 | 1163.10 | 1175.12 | 1196.20 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-04 13:30:00 | 1163.10 | 1174.99 | 1196.03 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-16 09:15:00 | 1104.47 | 1163.13 | 1184.72 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-16 09:15:00 | 1097.15 | 1163.13 | 1184.72 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-16 09:15:00 | 1104.94 | 1163.13 | 1184.72 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-16 09:15:00 | 1104.94 | 1163.13 | 1184.72 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-02-27 10:15:00 | 1046.34 | 1127.31 | 1158.53 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 7 — BUY (started 2026-04-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-20 14:15:00 | 1200.00 | 1119.85 | 1119.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-30 09:15:00 | 1212.00 | 1149.65 | 1136.51 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-05-13 14:45:00 | 1031.55 | 2025-05-14 09:15:00 | 1046.85 | STOP_HIT | 1.00 | -1.48% |
| SELL | retest2 | 2025-05-26 12:00:00 | 1030.00 | 2025-05-26 14:15:00 | 1045.00 | STOP_HIT | 1.00 | -1.46% |
| BUY | retest2 | 2025-06-05 09:15:00 | 1057.10 | 2025-06-18 09:15:00 | 1162.81 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-06-05 14:30:00 | 1054.70 | 2025-06-18 09:15:00 | 1160.17 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-06-06 09:15:00 | 1066.90 | 2025-06-18 09:15:00 | 1173.59 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-11-27 09:15:00 | 1226.70 | 2025-12-11 09:15:00 | 1290.10 | STOP_HIT | 1.00 | -5.17% |
| SELL | retest2 | 2025-12-12 15:00:00 | 1244.50 | 2025-12-22 13:15:00 | 1245.90 | STOP_HIT | 1.00 | -0.11% |
| SELL | retest2 | 2025-12-16 11:00:00 | 1242.20 | 2025-12-22 13:15:00 | 1245.90 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest2 | 2025-12-16 12:00:00 | 1243.70 | 2025-12-22 13:15:00 | 1245.90 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest2 | 2025-12-17 09:15:00 | 1234.90 | 2025-12-26 12:15:00 | 1259.40 | STOP_HIT | 1.00 | -1.98% |
| SELL | retest2 | 2025-12-17 13:15:00 | 1234.20 | 2025-12-26 12:15:00 | 1259.40 | STOP_HIT | 1.00 | -2.04% |
| SELL | retest2 | 2025-12-22 10:00:00 | 1234.60 | 2025-12-26 12:15:00 | 1259.40 | STOP_HIT | 1.00 | -2.01% |
| BUY | retest2 | 2026-01-07 10:00:00 | 1233.80 | 2026-01-07 14:15:00 | 1222.60 | STOP_HIT | 1.00 | -0.91% |
| BUY | retest2 | 2026-01-07 13:45:00 | 1233.10 | 2026-01-07 14:15:00 | 1222.60 | STOP_HIT | 1.00 | -0.85% |
| SELL | retest2 | 2026-02-04 09:15:00 | 1162.60 | 2026-02-16 09:15:00 | 1104.47 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-04 09:45:00 | 1154.90 | 2026-02-16 09:15:00 | 1097.15 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-04 13:00:00 | 1163.10 | 2026-02-16 09:15:00 | 1104.94 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-04 13:30:00 | 1163.10 | 2026-02-16 09:15:00 | 1104.94 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-04 09:15:00 | 1162.60 | 2026-02-27 10:15:00 | 1046.34 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-02-04 09:45:00 | 1154.90 | 2026-02-27 10:15:00 | 1039.41 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-02-04 13:00:00 | 1163.10 | 2026-02-27 10:15:00 | 1046.79 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-02-04 13:30:00 | 1163.10 | 2026-02-27 10:15:00 | 1046.79 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-04-06 13:30:00 | 1114.80 | 2026-04-06 14:15:00 | 1132.40 | STOP_HIT | 1.00 | -1.58% |
| SELL | retest2 | 2026-04-08 14:15:00 | 1118.50 | 2026-04-15 09:15:00 | 1155.00 | STOP_HIT | 1.00 | -3.26% |
| SELL | retest2 | 2026-04-08 15:15:00 | 1117.00 | 2026-04-15 09:15:00 | 1155.00 | STOP_HIT | 1.00 | -3.40% |
| SELL | retest2 | 2026-04-13 11:00:00 | 1120.80 | 2026-04-15 09:15:00 | 1155.00 | STOP_HIT | 1.00 | -3.05% |
