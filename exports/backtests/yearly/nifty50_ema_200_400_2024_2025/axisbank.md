# AXISBANK (AXISBANK)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 1270.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 8 |
| ALERT1 | 8 |
| ALERT2 | 9 |
| ALERT2_SKIP | 3 |
| ALERT3 | 36 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 23 |
| PARTIAL | 3 |
| TARGET_HIT | 3 |
| STOP_HIT | 20 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 26 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 11 / 15
- **Target hits / Stop hits / Partials:** 3 / 20 / 3
- **Avg / median % per leg:** 1.37% / -0.54%
- **Sum % (uncompounded):** 35.58%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 14 | 5 | 35.7% | 3 | 11 | 0 | 1.64% | 22.9% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 14 | 5 | 35.7% | 3 | 11 | 0 | 1.64% | 22.9% |
| SELL (all) | 12 | 6 | 50.0% | 0 | 9 | 3 | 1.05% | 12.7% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 12 | 6 | 50.0% | 0 | 9 | 3 | 1.05% | 12.7% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 26 | 11 | 42.3% | 3 | 20 | 3 | 1.37% | 35.6% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-08-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-13 13:15:00 | 1162.95 | 1194.49 | 1194.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-14 09:15:00 | 1151.20 | 1193.38 | 1194.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-27 12:15:00 | 1182.20 | 1181.78 | 1187.10 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-27 13:00:00 | 1182.20 | 1181.78 | 1187.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-02 10:15:00 | 1183.00 | 1180.50 | 1185.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-02 10:45:00 | 1184.30 | 1180.50 | 1185.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-02 13:15:00 | 1191.40 | 1180.63 | 1185.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-02 14:00:00 | 1191.40 | 1180.63 | 1185.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-02 14:15:00 | 1189.85 | 1180.72 | 1185.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-02 14:30:00 | 1191.10 | 1180.72 | 1185.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-03 14:15:00 | 1192.60 | 1180.98 | 1185.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-03 15:00:00 | 1192.60 | 1180.98 | 1185.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-03 15:15:00 | 1191.00 | 1181.08 | 1185.76 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-04 09:15:00 | 1179.05 | 1181.08 | 1185.76 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-11 09:45:00 | 1189.70 | 1178.94 | 1183.81 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-11 10:45:00 | 1188.60 | 1179.06 | 1183.84 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-11 11:15:00 | 1187.90 | 1179.06 | 1183.84 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-11 12:15:00 | 1184.25 | 1179.19 | 1183.86 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-12 09:45:00 | 1180.45 | 1179.40 | 1183.87 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-12 11:15:00 | 1183.05 | 1179.45 | 1183.87 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-12 13:15:00 | 1199.80 | 1179.77 | 1183.97 | SL hit (close>static) qty=1.00 sl=1193.55 alert=retest2 |

### Cycle 2 — BUY (started 2024-09-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-17 13:15:00 | 1235.45 | 1188.03 | 1187.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-18 11:15:00 | 1242.50 | 1190.32 | 1189.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-03 09:15:00 | 1201.15 | 1219.08 | 1206.22 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-03 09:15:00 | 1201.15 | 1219.08 | 1206.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-03 09:15:00 | 1201.15 | 1219.08 | 1206.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-03 10:00:00 | 1201.15 | 1219.08 | 1206.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-03 10:15:00 | 1197.85 | 1218.87 | 1206.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-03 11:00:00 | 1197.85 | 1218.87 | 1206.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 3 — SELL (started 2024-10-15 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-15 15:15:00 | 1157.70 | 1196.64 | 1196.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-16 11:15:00 | 1150.60 | 1195.40 | 1196.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-18 11:15:00 | 1194.45 | 1189.52 | 1193.08 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-18 11:15:00 | 1194.45 | 1189.52 | 1193.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 11:15:00 | 1194.45 | 1189.52 | 1193.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-18 12:00:00 | 1194.45 | 1189.52 | 1193.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 12:15:00 | 1197.35 | 1189.60 | 1193.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-18 13:00:00 | 1197.35 | 1189.60 | 1193.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 13:15:00 | 1197.55 | 1189.68 | 1193.12 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-21 11:15:00 | 1195.00 | 1190.02 | 1193.23 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-22 10:00:00 | 1193.60 | 1190.08 | 1193.17 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-25 09:30:00 | 1184.80 | 1186.59 | 1191.03 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-05 11:15:00 | 1135.25 | 1179.06 | 1186.09 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-05 11:15:00 | 1133.92 | 1179.06 | 1186.09 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-11-11 12:15:00 | 1176.00 | 1175.33 | 1183.15 | SL hit (close>ema200) qty=0.50 sl=1175.33 alert=retest2 |

### Cycle 4 — BUY (started 2025-03-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-27 15:15:00 | 1098.00 | 1044.57 | 1044.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-15 10:15:00 | 1106.90 | 1060.18 | 1053.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-03 12:15:00 | 1177.10 | 1180.86 | 1148.99 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-03 13:00:00 | 1177.10 | 1180.86 | 1148.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 09:15:00 | 1174.60 | 1205.86 | 1180.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-01 10:00:00 | 1174.60 | 1205.86 | 1180.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 10:15:00 | 1169.90 | 1205.50 | 1180.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-01 11:00:00 | 1169.90 | 1205.50 | 1180.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 10:15:00 | 1178.80 | 1203.34 | 1180.47 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-03 09:45:00 | 1180.20 | 1201.76 | 1180.35 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-03 10:15:00 | 1173.80 | 1201.48 | 1180.32 | SL hit (close<static) qty=1.00 sl=1175.80 alert=retest2 |

### Cycle 5 — SELL (started 2025-07-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-22 13:15:00 | 1101.00 | 1169.99 | 1170.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-22 14:15:00 | 1097.50 | 1169.27 | 1169.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-11 10:15:00 | 1074.20 | 1073.36 | 1097.24 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-11 11:00:00 | 1074.20 | 1073.36 | 1097.24 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 09:15:00 | 1102.30 | 1074.25 | 1096.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-12 09:45:00 | 1103.00 | 1074.25 | 1096.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 10:15:00 | 1105.80 | 1074.57 | 1097.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-12 11:00:00 | 1105.80 | 1074.57 | 1097.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 6 — BUY (started 2025-09-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-29 11:15:00 | 1129.80 | 1110.78 | 1110.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-29 12:15:00 | 1133.60 | 1111.01 | 1110.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-16 09:15:00 | 1237.30 | 1260.94 | 1230.37 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-16 10:00:00 | 1237.30 | 1260.94 | 1230.37 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 11:15:00 | 1236.80 | 1260.43 | 1230.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 11:45:00 | 1234.00 | 1260.43 | 1230.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 12:15:00 | 1230.30 | 1260.13 | 1230.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 13:00:00 | 1230.30 | 1260.13 | 1230.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 13:15:00 | 1224.70 | 1259.78 | 1230.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 14:00:00 | 1224.70 | 1259.78 | 1230.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 14:15:00 | 1220.00 | 1259.38 | 1230.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 15:00:00 | 1220.00 | 1259.38 | 1230.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 10:15:00 | 1234.10 | 1258.43 | 1230.29 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-18 10:30:00 | 1235.90 | 1256.45 | 1230.25 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-18 12:00:00 | 1235.50 | 1256.24 | 1230.27 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-18 12:30:00 | 1235.60 | 1256.02 | 1230.29 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-19 09:15:00 | 1240.20 | 1255.25 | 1230.29 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 14:15:00 | 1230.00 | 1254.01 | 1230.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-19 15:00:00 | 1230.00 | 1254.01 | 1230.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 15:15:00 | 1231.40 | 1253.79 | 1230.40 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-12-22 10:15:00 | 1228.00 | 1253.31 | 1230.40 | SL hit (close<static) qty=1.00 sl=1228.10 alert=retest2 |

### Cycle 7 — SELL (started 2026-03-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 12:15:00 | 1221.40 | 1299.88 | 1300.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 13:15:00 | 1210.80 | 1299.00 | 1299.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-07 14:15:00 | 1250.90 | 1249.64 | 1270.19 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-07 14:45:00 | 1252.10 | 1249.64 | 1270.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 09:15:00 | 1316.00 | 1250.28 | 1270.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-08 10:00:00 | 1316.00 | 1250.28 | 1270.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 8 — BUY (started 2026-04-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-17 14:15:00 | 1357.90 | 1286.29 | 1286.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-17 15:15:00 | 1364.00 | 1287.07 | 1286.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-28 10:15:00 | 1301.30 | 1311.89 | 1300.57 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-28 11:00:00 | 1301.30 | 1311.89 | 1300.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 11:15:00 | 1296.00 | 1311.73 | 1300.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-28 11:30:00 | 1294.70 | 1311.73 | 1300.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 12:15:00 | 1291.60 | 1311.53 | 1300.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-28 12:45:00 | 1291.30 | 1311.53 | 1300.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 13:15:00 | 1296.60 | 1310.11 | 1300.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-29 14:00:00 | 1296.60 | 1310.11 | 1300.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 14:15:00 | 1296.00 | 1309.97 | 1300.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-29 15:15:00 | 1290.60 | 1309.97 | 1300.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 15:15:00 | 1290.60 | 1309.78 | 1300.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-30 09:15:00 | 1265.90 | 1309.78 | 1300.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 15:15:00 | 1300.60 | 1300.04 | 1296.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-07 09:15:00 | 1299.70 | 1300.04 | 1296.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-07 09:15:00 | 1298.50 | 1300.02 | 1296.19 | EMA400 retest candle locked (from upside) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-06-05 11:15:00 | 1149.45 | 2024-06-25 12:15:00 | 1260.93 | TARGET_HIT | 1.00 | 9.70% |
| BUY | retest2 | 2024-06-05 12:15:00 | 1146.30 | 2024-06-25 13:15:00 | 1264.40 | TARGET_HIT | 1.00 | 10.30% |
| BUY | retest2 | 2024-08-09 09:45:00 | 1150.40 | 2024-08-13 13:15:00 | 1162.95 | STOP_HIT | 1.00 | 1.09% |
| BUY | retest2 | 2024-08-09 14:45:00 | 1145.00 | 2024-08-13 13:15:00 | 1162.95 | STOP_HIT | 1.00 | 1.57% |
| SELL | retest2 | 2024-09-04 09:15:00 | 1179.05 | 2024-09-12 13:15:00 | 1199.80 | STOP_HIT | 1.00 | -1.76% |
| SELL | retest2 | 2024-09-11 09:45:00 | 1189.70 | 2024-09-12 13:15:00 | 1199.80 | STOP_HIT | 1.00 | -0.85% |
| SELL | retest2 | 2024-09-11 10:45:00 | 1188.60 | 2024-09-12 13:15:00 | 1199.80 | STOP_HIT | 1.00 | -0.94% |
| SELL | retest2 | 2024-09-11 11:15:00 | 1187.90 | 2024-09-12 13:15:00 | 1199.80 | STOP_HIT | 1.00 | -1.00% |
| SELL | retest2 | 2024-09-12 09:45:00 | 1180.45 | 2024-09-12 13:15:00 | 1199.80 | STOP_HIT | 1.00 | -1.64% |
| SELL | retest2 | 2024-09-12 11:15:00 | 1183.05 | 2024-09-12 13:15:00 | 1199.80 | STOP_HIT | 1.00 | -1.42% |
| SELL | retest2 | 2024-10-21 11:15:00 | 1195.00 | 2024-11-05 11:15:00 | 1135.25 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-22 10:00:00 | 1193.60 | 2024-11-05 11:15:00 | 1133.92 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-21 11:15:00 | 1195.00 | 2024-11-11 12:15:00 | 1176.00 | STOP_HIT | 0.50 | 1.59% |
| SELL | retest2 | 2024-10-22 10:00:00 | 1193.60 | 2024-11-11 12:15:00 | 1176.00 | STOP_HIT | 0.50 | 1.47% |
| SELL | retest2 | 2024-10-25 09:30:00 | 1184.80 | 2024-11-18 09:15:00 | 1125.56 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-25 09:30:00 | 1184.80 | 2024-12-03 11:15:00 | 1158.70 | STOP_HIT | 0.50 | 2.20% |
| BUY | retest2 | 2025-07-03 09:45:00 | 1180.20 | 2025-07-03 10:15:00 | 1173.80 | STOP_HIT | 1.00 | -0.54% |
| BUY | retest2 | 2025-12-18 10:30:00 | 1235.90 | 2025-12-22 10:15:00 | 1228.00 | STOP_HIT | 1.00 | -0.64% |
| BUY | retest2 | 2025-12-18 12:00:00 | 1235.50 | 2025-12-22 10:15:00 | 1228.00 | STOP_HIT | 1.00 | -0.61% |
| BUY | retest2 | 2025-12-18 12:30:00 | 1235.60 | 2025-12-22 10:15:00 | 1228.00 | STOP_HIT | 1.00 | -0.62% |
| BUY | retest2 | 2025-12-19 09:15:00 | 1240.20 | 2025-12-22 10:15:00 | 1228.00 | STOP_HIT | 1.00 | -0.98% |
| BUY | retest2 | 2025-12-22 14:30:00 | 1233.50 | 2025-12-23 14:15:00 | 1225.80 | STOP_HIT | 1.00 | -0.62% |
| BUY | retest2 | 2025-12-22 15:15:00 | 1233.80 | 2025-12-23 14:15:00 | 1225.80 | STOP_HIT | 1.00 | -0.65% |
| BUY | retest2 | 2025-12-24 11:15:00 | 1234.60 | 2025-12-24 12:15:00 | 1228.20 | STOP_HIT | 1.00 | -0.52% |
| BUY | retest2 | 2025-12-30 09:15:00 | 1235.20 | 2026-01-28 09:15:00 | 1358.72 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-01-27 09:15:00 | 1305.30 | 2026-03-12 09:15:00 | 1245.80 | STOP_HIT | 1.00 | -4.56% |
