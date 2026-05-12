# CreditAccess Grameen Ltd. (CREDITACC)

## Backtest Summary

- **Window:** 2025-03-13 09:15:00 → 2026-05-08 15:15:00 (1976 bars)
- **Last close:** 1493.20
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 73 |
| ALERT1 | 52 |
| ALERT2 | 52 |
| ALERT2_SKIP | 26 |
| ALERT3 | 123 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 6 |
| ENTRY2 | 60 |
| PARTIAL | 8 |
| TARGET_HIT | 7 |
| STOP_HIT | 59 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 74 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 30 / 44
- **Target hits / Stop hits / Partials:** 7 / 59 / 8
- **Avg / median % per leg:** 1.00% / -0.50%
- **Sum % (uncompounded):** 74.30%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 39 | 13 | 33.3% | 7 | 32 | 0 | 0.74% | 28.9% |
| BUY @ 2nd Alert (retest1) | 3 | 0 | 0.0% | 0 | 3 | 0 | -5.61% | -16.8% |
| BUY @ 3rd Alert (retest2) | 36 | 13 | 36.1% | 7 | 29 | 0 | 1.27% | 45.7% |
| SELL (all) | 35 | 17 | 48.6% | 0 | 27 | 8 | 1.30% | 45.4% |
| SELL @ 2nd Alert (retest1) | 3 | 0 | 0.0% | 0 | 3 | 0 | -1.44% | -4.3% |
| SELL @ 3rd Alert (retest2) | 32 | 17 | 53.1% | 0 | 24 | 8 | 1.55% | 49.7% |
| retest1 (combined) | 6 | 0 | 0.0% | 0 | 6 | 0 | -3.52% | -21.1% |
| retest2 (combined) | 68 | 30 | 44.1% | 7 | 53 | 8 | 1.40% | 95.4% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 10:15:00 | 1148.50 | 1126.91 | 1126.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 11:15:00 | 1152.30 | 1131.98 | 1128.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-14 15:15:00 | 1180.10 | 1185.77 | 1174.51 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-15 09:15:00 | 1216.00 | 1185.77 | 1174.51 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-15 13:00:00 | 1193.90 | 1193.75 | 1182.60 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-15 13:30:00 | 1191.90 | 1192.80 | 1183.18 | BUY ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-15 14:15:00 | 1198.50 | 1193.94 | 1184.58 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-16 09:30:00 | 1202.80 | 1192.55 | 1185.57 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-16 13:00:00 | 1206.00 | 1193.72 | 1187.71 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-19 09:15:00 | 1133.20 | 1184.44 | 1185.88 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest1 |

### Cycle 2 — SELL (started 2025-05-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-19 09:15:00 | 1133.20 | 1184.44 | 1185.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-19 10:15:00 | 1128.30 | 1173.21 | 1180.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-22 09:15:00 | 1119.00 | 1106.79 | 1117.79 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-22 09:15:00 | 1119.00 | 1106.79 | 1117.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 09:15:00 | 1119.00 | 1106.79 | 1117.79 | EMA400 retest candle locked (from downside) |

### Cycle 3 — BUY (started 2025-05-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-23 14:15:00 | 1118.90 | 1117.50 | 1117.40 | EMA200 above EMA400 |

### Cycle 4 — SELL (started 2025-05-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-23 15:15:00 | 1114.50 | 1116.90 | 1117.14 | EMA200 below EMA400 |

### Cycle 5 — BUY (started 2025-05-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-26 09:15:00 | 1152.00 | 1123.92 | 1120.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-27 09:15:00 | 1195.10 | 1151.19 | 1137.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-27 14:15:00 | 1172.00 | 1178.08 | 1159.10 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-27 15:00:00 | 1172.00 | 1178.08 | 1159.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 15:15:00 | 1170.10 | 1176.48 | 1160.10 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-28 09:15:00 | 1185.50 | 1176.48 | 1160.10 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-30 09:15:00 | 1158.80 | 1168.32 | 1168.47 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 6 — SELL (started 2025-05-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-30 09:15:00 | 1158.80 | 1168.32 | 1168.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-30 10:15:00 | 1157.00 | 1166.06 | 1167.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-02 13:15:00 | 1158.60 | 1153.80 | 1157.68 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-02 13:15:00 | 1158.60 | 1153.80 | 1157.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 13:15:00 | 1158.60 | 1153.80 | 1157.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-02 13:45:00 | 1163.80 | 1153.80 | 1157.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 14:15:00 | 1155.10 | 1154.06 | 1157.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-02 15:15:00 | 1155.00 | 1154.06 | 1157.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 15:15:00 | 1155.00 | 1154.25 | 1157.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-03 09:30:00 | 1157.70 | 1153.82 | 1156.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 11:15:00 | 1163.90 | 1155.96 | 1157.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-03 11:45:00 | 1162.00 | 1155.96 | 1157.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 12:15:00 | 1159.40 | 1156.65 | 1157.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-03 12:30:00 | 1164.60 | 1156.65 | 1157.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 7 — BUY (started 2025-06-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-03 13:15:00 | 1169.20 | 1159.16 | 1158.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-04 14:15:00 | 1171.80 | 1164.40 | 1161.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-06 14:15:00 | 1198.20 | 1207.40 | 1192.17 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-06 15:00:00 | 1198.20 | 1207.40 | 1192.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 09:15:00 | 1205.70 | 1206.07 | 1200.26 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-11 09:15:00 | 1213.70 | 1199.92 | 1199.35 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-12 09:45:00 | 1212.00 | 1212.18 | 1208.40 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-12 10:15:00 | 1211.30 | 1212.18 | 1208.40 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-12 12:15:00 | 1196.40 | 1204.51 | 1205.49 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 8 — SELL (started 2025-06-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-12 12:15:00 | 1196.40 | 1204.51 | 1205.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-12 13:15:00 | 1181.30 | 1199.87 | 1203.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-13 14:15:00 | 1183.40 | 1181.15 | 1189.50 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-13 15:00:00 | 1183.40 | 1181.15 | 1189.50 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 09:15:00 | 1182.10 | 1181.46 | 1188.20 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-17 09:15:00 | 1168.00 | 1189.54 | 1189.70 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-17 11:00:00 | 1168.50 | 1182.22 | 1186.15 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-20 12:15:00 | 1109.60 | 1122.23 | 1134.00 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-20 12:15:00 | 1110.08 | 1122.23 | 1134.00 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-23 09:15:00 | 1124.90 | 1119.59 | 1128.68 | SL hit (close>ema200) qty=0.50 sl=1119.59 alert=retest2 |

### Cycle 9 — BUY (started 2025-06-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-24 10:15:00 | 1163.50 | 1134.66 | 1131.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-24 11:15:00 | 1174.80 | 1142.69 | 1135.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-26 10:15:00 | 1196.30 | 1198.32 | 1179.84 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-26 10:30:00 | 1196.50 | 1198.32 | 1179.84 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 15:15:00 | 1189.70 | 1192.63 | 1183.74 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-27 09:15:00 | 1196.50 | 1192.63 | 1183.74 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-07-04 09:15:00 | 1316.15 | 1264.77 | 1250.88 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 10 — SELL (started 2025-07-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-10 14:15:00 | 1280.40 | 1282.54 | 1282.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-11 10:15:00 | 1273.30 | 1280.34 | 1281.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-11 13:15:00 | 1296.20 | 1279.59 | 1280.57 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-11 13:15:00 | 1296.20 | 1279.59 | 1280.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 13:15:00 | 1296.20 | 1279.59 | 1280.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-11 14:00:00 | 1296.20 | 1279.59 | 1280.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 11 — BUY (started 2025-07-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-11 14:15:00 | 1307.40 | 1285.15 | 1283.01 | EMA200 above EMA400 |

### Cycle 12 — SELL (started 2025-07-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-16 15:15:00 | 1298.00 | 1300.94 | 1301.22 | EMA200 below EMA400 |

### Cycle 13 — BUY (started 2025-07-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-17 09:15:00 | 1309.00 | 1302.55 | 1301.93 | EMA200 above EMA400 |

### Cycle 14 — SELL (started 2025-07-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-17 14:15:00 | 1287.00 | 1301.58 | 1302.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-18 09:15:00 | 1268.30 | 1293.36 | 1298.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-18 15:15:00 | 1272.00 | 1271.86 | 1282.67 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-21 09:15:00 | 1266.10 | 1271.86 | 1282.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 10:15:00 | 1276.10 | 1273.11 | 1281.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-21 10:45:00 | 1277.40 | 1273.11 | 1281.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 12:15:00 | 1291.10 | 1276.34 | 1281.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-21 13:00:00 | 1291.10 | 1276.34 | 1281.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 13:15:00 | 1290.80 | 1279.23 | 1282.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-21 14:00:00 | 1290.80 | 1279.23 | 1282.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 13:15:00 | 1274.80 | 1276.65 | 1279.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-22 13:45:00 | 1281.70 | 1276.65 | 1279.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 14:15:00 | 1281.00 | 1277.52 | 1279.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-22 14:45:00 | 1279.80 | 1277.52 | 1279.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 15:15:00 | 1281.00 | 1278.21 | 1280.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-23 09:15:00 | 1305.70 | 1278.21 | 1280.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 15 — BUY (started 2025-07-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-23 09:15:00 | 1342.10 | 1290.99 | 1285.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-23 11:15:00 | 1369.00 | 1314.10 | 1297.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-25 09:15:00 | 1334.80 | 1354.78 | 1339.65 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-25 09:15:00 | 1334.80 | 1354.78 | 1339.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 09:15:00 | 1334.80 | 1354.78 | 1339.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-25 09:45:00 | 1336.90 | 1354.78 | 1339.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 10:15:00 | 1328.90 | 1349.60 | 1338.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-25 11:00:00 | 1328.90 | 1349.60 | 1338.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 16 — SELL (started 2025-07-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-25 14:15:00 | 1310.40 | 1329.76 | 1331.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-25 15:15:00 | 1301.50 | 1324.11 | 1329.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-31 09:15:00 | 1254.60 | 1246.06 | 1259.36 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-31 09:15:00 | 1254.60 | 1246.06 | 1259.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 09:15:00 | 1254.60 | 1246.06 | 1259.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-31 09:45:00 | 1256.50 | 1246.06 | 1259.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 10:15:00 | 1262.10 | 1249.27 | 1259.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-31 10:30:00 | 1265.90 | 1249.27 | 1259.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 11:15:00 | 1263.90 | 1252.19 | 1260.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-31 11:30:00 | 1264.30 | 1252.19 | 1260.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 13:15:00 | 1260.60 | 1255.59 | 1260.29 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-31 14:15:00 | 1255.20 | 1255.59 | 1260.29 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-31 15:00:00 | 1251.40 | 1254.75 | 1259.49 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-01 09:15:00 | 1267.30 | 1255.25 | 1258.77 | SL hit (close>static) qty=1.00 sl=1265.00 alert=retest2 |

### Cycle 17 — BUY (started 2025-08-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-11 09:15:00 | 1245.00 | 1234.38 | 1233.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-11 14:15:00 | 1256.10 | 1243.85 | 1238.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-13 14:15:00 | 1297.90 | 1298.81 | 1282.90 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-13 15:00:00 | 1297.90 | 1298.81 | 1282.90 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 15:15:00 | 1308.40 | 1300.73 | 1285.22 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-14 09:45:00 | 1316.70 | 1305.20 | 1288.67 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-18 11:30:00 | 1318.80 | 1322.76 | 1311.09 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-26 11:15:00 | 1365.50 | 1381.47 | 1382.04 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 18 — SELL (started 2025-08-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-26 11:15:00 | 1365.50 | 1381.47 | 1382.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-26 12:15:00 | 1362.00 | 1377.58 | 1380.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-28 13:15:00 | 1354.90 | 1353.81 | 1363.95 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-28 14:00:00 | 1354.90 | 1353.81 | 1363.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 11:15:00 | 1355.20 | 1343.13 | 1353.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-29 11:30:00 | 1355.70 | 1343.13 | 1353.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 12:15:00 | 1358.40 | 1346.18 | 1354.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-29 12:45:00 | 1358.00 | 1346.18 | 1354.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 13:15:00 | 1384.00 | 1353.75 | 1356.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-29 14:00:00 | 1384.00 | 1353.75 | 1356.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 19 — BUY (started 2025-08-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-29 14:15:00 | 1391.60 | 1361.32 | 1360.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-01 10:15:00 | 1410.90 | 1382.81 | 1371.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-02 09:15:00 | 1394.00 | 1395.46 | 1384.03 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-02 09:15:00 | 1394.00 | 1395.46 | 1384.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 09:15:00 | 1394.00 | 1395.46 | 1384.03 | EMA400 retest candle locked (from upside) |

### Cycle 20 — SELL (started 2025-09-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-03 15:15:00 | 1378.00 | 1384.57 | 1385.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-04 09:15:00 | 1368.60 | 1381.37 | 1383.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-11 09:15:00 | 1311.80 | 1307.32 | 1318.86 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-11 15:15:00 | 1295.20 | 1304.85 | 1313.32 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-12 11:30:00 | 1292.90 | 1298.55 | 1307.28 | SELL ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-12 14:45:00 | 1295.30 | 1295.74 | 1303.64 | SELL ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 11:15:00 | 1313.10 | 1294.47 | 1300.11 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-09-15 11:15:00 | 1313.10 | 1294.47 | 1300.11 | SL hit (close>ema400) qty=1.00 sl=1300.11 alert=retest1 |

### Cycle 21 — BUY (started 2025-09-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-15 14:15:00 | 1322.00 | 1305.51 | 1304.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-16 10:15:00 | 1335.80 | 1315.52 | 1309.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-19 09:15:00 | 1376.70 | 1381.21 | 1364.93 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-19 09:15:00 | 1376.70 | 1381.21 | 1364.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 09:15:00 | 1376.70 | 1381.21 | 1364.93 | EMA400 retest candle locked (from upside) |

### Cycle 22 — SELL (started 2025-09-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-23 09:15:00 | 1356.70 | 1365.18 | 1365.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-23 11:15:00 | 1343.70 | 1358.61 | 1362.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-23 15:15:00 | 1358.50 | 1356.46 | 1360.13 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-24 09:15:00 | 1356.50 | 1356.46 | 1360.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 09:15:00 | 1366.00 | 1358.37 | 1360.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-24 09:45:00 | 1368.00 | 1358.37 | 1360.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 23 — BUY (started 2025-09-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-24 10:15:00 | 1381.40 | 1362.98 | 1362.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-25 09:15:00 | 1416.00 | 1388.03 | 1376.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-26 09:15:00 | 1404.30 | 1407.49 | 1394.47 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-26 09:15:00 | 1404.30 | 1407.49 | 1394.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 09:15:00 | 1404.30 | 1407.49 | 1394.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-26 09:30:00 | 1403.60 | 1407.49 | 1394.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 13:15:00 | 1389.00 | 1405.14 | 1397.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-26 13:45:00 | 1392.50 | 1405.14 | 1397.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 14:15:00 | 1375.90 | 1399.29 | 1395.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-26 15:00:00 | 1375.90 | 1399.29 | 1395.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 24 — SELL (started 2025-09-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-29 09:15:00 | 1377.10 | 1390.96 | 1392.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-29 10:15:00 | 1353.30 | 1383.43 | 1388.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-30 10:15:00 | 1361.60 | 1355.27 | 1368.44 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-30 10:15:00 | 1361.60 | 1355.27 | 1368.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 10:15:00 | 1361.60 | 1355.27 | 1368.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-30 11:00:00 | 1361.60 | 1355.27 | 1368.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 14:15:00 | 1354.60 | 1354.41 | 1363.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-30 14:45:00 | 1364.80 | 1354.41 | 1363.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 12:15:00 | 1365.90 | 1351.64 | 1358.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-01 13:00:00 | 1365.90 | 1351.64 | 1358.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 13:15:00 | 1380.60 | 1357.43 | 1360.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-01 13:30:00 | 1377.80 | 1357.43 | 1360.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 25 — BUY (started 2025-10-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-01 14:15:00 | 1401.90 | 1366.32 | 1363.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-03 11:15:00 | 1405.50 | 1387.73 | 1376.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-03 15:15:00 | 1398.00 | 1398.18 | 1385.42 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-06 09:15:00 | 1393.50 | 1398.18 | 1385.42 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 10:15:00 | 1392.00 | 1396.78 | 1391.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-07 11:15:00 | 1390.00 | 1396.78 | 1391.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 11:15:00 | 1393.00 | 1396.02 | 1392.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-07 11:30:00 | 1393.90 | 1396.02 | 1392.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 12:15:00 | 1400.00 | 1396.82 | 1392.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-07 12:30:00 | 1387.00 | 1396.82 | 1392.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 13:15:00 | 1384.70 | 1394.40 | 1392.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-07 14:00:00 | 1384.70 | 1394.40 | 1392.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 14:15:00 | 1392.00 | 1393.92 | 1392.06 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-07 15:15:00 | 1395.00 | 1393.92 | 1392.06 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-08 10:15:00 | 1373.10 | 1388.83 | 1390.16 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 26 — SELL (started 2025-10-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-08 10:15:00 | 1373.10 | 1388.83 | 1390.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-08 11:15:00 | 1369.90 | 1385.05 | 1388.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-09 10:15:00 | 1377.40 | 1377.40 | 1382.27 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-09 11:00:00 | 1377.40 | 1377.40 | 1382.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 14:15:00 | 1378.50 | 1376.28 | 1380.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-09 15:00:00 | 1378.50 | 1376.28 | 1380.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 15:15:00 | 1380.30 | 1377.08 | 1380.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-10 09:15:00 | 1381.00 | 1377.08 | 1380.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 09:15:00 | 1376.60 | 1376.99 | 1379.80 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-10 10:45:00 | 1375.00 | 1376.81 | 1379.46 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-10 11:30:00 | 1375.40 | 1376.45 | 1379.06 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-10 15:15:00 | 1385.60 | 1380.15 | 1380.13 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 27 — BUY (started 2025-10-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-10 15:15:00 | 1385.60 | 1380.15 | 1380.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-13 09:15:00 | 1403.30 | 1384.78 | 1382.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-13 15:15:00 | 1398.90 | 1403.51 | 1395.16 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-14 09:15:00 | 1392.30 | 1401.27 | 1394.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 09:15:00 | 1392.30 | 1401.27 | 1394.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-14 09:45:00 | 1392.60 | 1401.27 | 1394.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 10:15:00 | 1382.90 | 1397.59 | 1393.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-14 10:45:00 | 1384.60 | 1397.59 | 1393.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 28 — SELL (started 2025-10-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-14 12:15:00 | 1376.10 | 1389.13 | 1390.36 | EMA200 below EMA400 |

### Cycle 29 — BUY (started 2025-10-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-16 09:15:00 | 1392.90 | 1386.08 | 1385.70 | EMA200 above EMA400 |

### Cycle 30 — SELL (started 2025-10-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-17 09:15:00 | 1354.00 | 1383.07 | 1385.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-17 10:15:00 | 1346.20 | 1375.70 | 1381.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-23 09:15:00 | 1302.20 | 1297.13 | 1315.31 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-23 10:15:00 | 1309.00 | 1297.13 | 1315.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 10:15:00 | 1319.50 | 1301.60 | 1315.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-23 11:00:00 | 1319.50 | 1301.60 | 1315.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 11:15:00 | 1331.40 | 1307.56 | 1317.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-23 12:00:00 | 1331.40 | 1307.56 | 1317.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 31 — BUY (started 2025-10-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-23 15:15:00 | 1329.60 | 1323.10 | 1322.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-24 09:15:00 | 1387.10 | 1335.90 | 1328.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-28 15:15:00 | 1469.90 | 1471.87 | 1446.09 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-29 09:15:00 | 1446.00 | 1471.87 | 1446.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 09:15:00 | 1426.70 | 1462.84 | 1444.32 | EMA400 retest candle locked (from upside) |

### Cycle 32 — SELL (started 2025-10-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-29 12:15:00 | 1387.50 | 1427.50 | 1430.97 | EMA200 below EMA400 |

### Cycle 33 — BUY (started 2025-10-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-30 11:15:00 | 1481.90 | 1434.32 | 1430.88 | EMA200 above EMA400 |

### Cycle 34 — SELL (started 2025-10-31 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-31 15:15:00 | 1410.30 | 1437.81 | 1439.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-03 11:15:00 | 1405.60 | 1422.88 | 1431.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-04 10:15:00 | 1417.90 | 1414.69 | 1422.80 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-04 10:15:00 | 1417.90 | 1414.69 | 1422.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 10:15:00 | 1417.90 | 1414.69 | 1422.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-04 11:00:00 | 1417.90 | 1414.69 | 1422.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 14:15:00 | 1354.10 | 1326.12 | 1349.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-07 15:00:00 | 1354.10 | 1326.12 | 1349.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 15:15:00 | 1360.40 | 1332.97 | 1350.07 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-10 09:45:00 | 1344.70 | 1334.94 | 1349.41 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-10 10:15:00 | 1377.00 | 1343.35 | 1351.92 | SL hit (close>static) qty=1.00 sl=1365.00 alert=retest2 |

### Cycle 35 — BUY (started 2025-11-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-10 12:15:00 | 1391.50 | 1359.16 | 1358.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-10 13:15:00 | 1401.60 | 1367.65 | 1361.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-11 11:15:00 | 1393.00 | 1398.53 | 1381.76 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-11 12:00:00 | 1393.00 | 1398.53 | 1381.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 12:15:00 | 1388.00 | 1396.42 | 1382.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-11 12:30:00 | 1385.00 | 1396.42 | 1382.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 13:15:00 | 1384.30 | 1394.00 | 1382.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-11 13:45:00 | 1377.70 | 1394.00 | 1382.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 14:15:00 | 1369.80 | 1389.16 | 1381.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-11 15:00:00 | 1369.80 | 1389.16 | 1381.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 15:15:00 | 1370.00 | 1385.33 | 1380.32 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-12 09:15:00 | 1391.00 | 1385.33 | 1380.32 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-12 14:15:00 | 1367.60 | 1378.22 | 1379.10 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 36 — SELL (started 2025-11-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-12 14:15:00 | 1367.60 | 1378.22 | 1379.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-13 09:15:00 | 1362.80 | 1373.82 | 1376.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-14 10:15:00 | 1361.10 | 1359.24 | 1366.04 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-14 11:00:00 | 1361.10 | 1359.24 | 1366.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 12:15:00 | 1360.50 | 1359.96 | 1365.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-14 12:45:00 | 1366.70 | 1359.96 | 1365.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 14:15:00 | 1371.80 | 1362.39 | 1365.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-14 15:00:00 | 1371.80 | 1362.39 | 1365.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 15:15:00 | 1370.90 | 1364.09 | 1365.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-17 09:15:00 | 1370.50 | 1364.09 | 1365.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 10:15:00 | 1368.40 | 1366.39 | 1366.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-17 10:45:00 | 1371.90 | 1366.39 | 1366.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 09:15:00 | 1309.20 | 1318.81 | 1329.87 | EMA400 retest candle locked (from downside) |

### Cycle 37 — BUY (started 2025-11-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-21 11:15:00 | 1346.40 | 1331.64 | 1330.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-21 12:15:00 | 1353.50 | 1336.01 | 1332.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-25 15:15:00 | 1380.10 | 1388.94 | 1377.78 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-26 09:15:00 | 1385.00 | 1388.94 | 1377.78 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 09:15:00 | 1369.90 | 1385.13 | 1377.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-26 09:45:00 | 1368.90 | 1385.13 | 1377.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 10:15:00 | 1357.00 | 1379.51 | 1375.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-26 10:45:00 | 1359.90 | 1379.51 | 1375.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 38 — SELL (started 2025-11-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-26 12:15:00 | 1358.70 | 1371.87 | 1372.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-27 11:15:00 | 1350.60 | 1364.22 | 1368.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-27 13:15:00 | 1368.90 | 1363.82 | 1367.09 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-27 13:15:00 | 1368.90 | 1363.82 | 1367.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 13:15:00 | 1368.90 | 1363.82 | 1367.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-27 14:00:00 | 1368.90 | 1363.82 | 1367.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 14:15:00 | 1369.90 | 1365.04 | 1367.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-27 15:00:00 | 1369.90 | 1365.04 | 1367.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 15:15:00 | 1372.90 | 1366.61 | 1367.85 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-28 09:15:00 | 1353.00 | 1366.61 | 1367.85 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-05 15:15:00 | 1285.35 | 1306.75 | 1317.30 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-09 11:15:00 | 1267.20 | 1261.93 | 1279.31 | SL hit (close>ema200) qty=0.50 sl=1261.93 alert=retest2 |

### Cycle 39 — BUY (started 2025-12-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-12 09:15:00 | 1291.20 | 1274.28 | 1274.15 | EMA200 above EMA400 |

### Cycle 40 — SELL (started 2025-12-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-15 10:15:00 | 1257.00 | 1276.91 | 1277.61 | EMA200 below EMA400 |

### Cycle 41 — BUY (started 2025-12-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-16 09:15:00 | 1288.40 | 1279.00 | 1277.97 | EMA200 above EMA400 |

### Cycle 42 — SELL (started 2025-12-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-16 14:15:00 | 1266.70 | 1278.44 | 1278.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-16 15:15:00 | 1260.00 | 1274.75 | 1276.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-17 14:15:00 | 1281.10 | 1267.20 | 1270.85 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-17 14:15:00 | 1281.10 | 1267.20 | 1270.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 14:15:00 | 1281.10 | 1267.20 | 1270.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-17 15:00:00 | 1281.10 | 1267.20 | 1270.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 15:15:00 | 1278.20 | 1269.40 | 1271.52 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-18 09:15:00 | 1262.00 | 1269.40 | 1271.52 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-18 11:30:00 | 1265.00 | 1267.24 | 1269.83 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-18 12:15:00 | 1265.90 | 1267.24 | 1269.83 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-19 11:15:00 | 1279.00 | 1270.95 | 1270.15 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 43 — BUY (started 2025-12-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-19 11:15:00 | 1279.00 | 1270.95 | 1270.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-22 09:15:00 | 1285.10 | 1277.33 | 1273.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-23 11:15:00 | 1292.40 | 1296.41 | 1288.41 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-23 11:45:00 | 1293.40 | 1296.41 | 1288.41 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 14:15:00 | 1302.00 | 1298.99 | 1291.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-23 14:45:00 | 1301.80 | 1298.99 | 1291.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 09:15:00 | 1313.10 | 1302.14 | 1294.41 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-24 10:15:00 | 1314.10 | 1302.14 | 1294.41 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-26 09:45:00 | 1316.90 | 1305.87 | 1300.09 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-26 13:15:00 | 1288.10 | 1300.30 | 1299.09 | SL hit (close<static) qty=1.00 sl=1294.10 alert=retest2 |

### Cycle 44 — SELL (started 2025-12-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-26 14:15:00 | 1261.50 | 1292.54 | 1295.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-30 11:15:00 | 1243.20 | 1267.44 | 1278.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-31 09:15:00 | 1250.00 | 1249.83 | 1264.05 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-31 10:00:00 | 1250.00 | 1249.83 | 1264.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 12:15:00 | 1267.00 | 1253.44 | 1262.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 13:00:00 | 1267.00 | 1253.44 | 1262.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 13:15:00 | 1269.50 | 1256.65 | 1262.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 13:30:00 | 1269.90 | 1256.65 | 1262.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 45 — BUY (started 2026-01-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-01 10:15:00 | 1270.70 | 1266.27 | 1266.05 | EMA200 above EMA400 |

### Cycle 46 — SELL (started 2026-01-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-01 11:15:00 | 1256.30 | 1264.28 | 1265.16 | EMA200 below EMA400 |

### Cycle 47 — BUY (started 2026-01-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-02 09:15:00 | 1293.40 | 1269.72 | 1267.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-02 11:15:00 | 1298.90 | 1279.47 | 1272.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-06 09:15:00 | 1301.30 | 1311.39 | 1298.23 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-06 09:15:00 | 1301.30 | 1311.39 | 1298.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 09:15:00 | 1301.30 | 1311.39 | 1298.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-06 09:30:00 | 1304.80 | 1311.39 | 1298.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 10:15:00 | 1315.80 | 1312.28 | 1299.83 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-06 11:30:00 | 1321.30 | 1314.28 | 1301.87 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-06 12:00:00 | 1322.30 | 1314.28 | 1301.87 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-07 09:45:00 | 1320.80 | 1322.17 | 1310.90 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-07 11:00:00 | 1319.20 | 1321.58 | 1311.65 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 11:15:00 | 1303.50 | 1317.96 | 1310.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-07 11:45:00 | 1304.10 | 1317.96 | 1310.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 12:15:00 | 1322.20 | 1318.81 | 1311.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-07 12:30:00 | 1307.50 | 1318.81 | 1311.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 15:15:00 | 1325.00 | 1333.95 | 1327.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-09 09:15:00 | 1320.00 | 1333.95 | 1327.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 09:15:00 | 1334.00 | 1333.96 | 1327.67 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-09 13:30:00 | 1337.30 | 1332.12 | 1328.61 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-12 09:15:00 | 1301.40 | 1323.72 | 1325.50 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 48 — SELL (started 2026-01-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-12 09:15:00 | 1301.40 | 1323.72 | 1325.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-12 11:15:00 | 1297.60 | 1315.13 | 1321.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-13 11:15:00 | 1308.40 | 1300.43 | 1309.25 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-13 11:15:00 | 1308.40 | 1300.43 | 1309.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 11:15:00 | 1308.40 | 1300.43 | 1309.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-13 11:45:00 | 1304.20 | 1300.43 | 1309.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 12:15:00 | 1306.30 | 1301.60 | 1308.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-13 12:30:00 | 1317.30 | 1301.60 | 1308.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 13:15:00 | 1283.40 | 1297.96 | 1306.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-13 13:30:00 | 1308.60 | 1297.96 | 1306.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 09:15:00 | 1298.50 | 1298.17 | 1304.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-14 09:30:00 | 1312.40 | 1298.17 | 1304.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 11:15:00 | 1296.00 | 1297.92 | 1303.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-14 11:30:00 | 1299.60 | 1297.92 | 1303.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 13:15:00 | 1290.00 | 1295.29 | 1301.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-14 13:30:00 | 1300.00 | 1295.29 | 1301.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 09:15:00 | 1299.20 | 1293.72 | 1298.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-16 10:00:00 | 1299.20 | 1293.72 | 1298.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 10:15:00 | 1300.00 | 1294.98 | 1298.87 | EMA400 retest candle locked (from downside) |

### Cycle 49 — BUY (started 2026-01-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-19 10:15:00 | 1308.20 | 1301.56 | 1300.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-19 12:15:00 | 1312.30 | 1304.59 | 1302.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-19 13:15:00 | 1298.00 | 1303.27 | 1301.87 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-19 13:15:00 | 1298.00 | 1303.27 | 1301.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 13:15:00 | 1298.00 | 1303.27 | 1301.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-19 14:00:00 | 1298.00 | 1303.27 | 1301.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 50 — SELL (started 2026-01-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-19 14:15:00 | 1283.20 | 1299.26 | 1300.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-19 15:15:00 | 1280.00 | 1295.41 | 1298.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-21 09:15:00 | 1344.80 | 1274.59 | 1280.06 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-21 09:15:00 | 1344.80 | 1274.59 | 1280.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-21 09:15:00 | 1344.80 | 1274.59 | 1280.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-21 09:45:00 | 1356.30 | 1274.59 | 1280.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 51 — BUY (started 2026-01-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-21 10:15:00 | 1370.60 | 1293.79 | 1288.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-21 12:15:00 | 1384.00 | 1324.44 | 1303.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-23 10:15:00 | 1409.90 | 1432.74 | 1396.01 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-23 11:00:00 | 1409.90 | 1432.74 | 1396.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 11:15:00 | 1383.00 | 1422.79 | 1394.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-23 12:00:00 | 1383.00 | 1422.79 | 1394.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 12:15:00 | 1403.90 | 1419.01 | 1395.65 | EMA400 retest candle locked (from upside) |

### Cycle 52 — SELL (started 2026-01-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-27 11:15:00 | 1348.20 | 1382.34 | 1385.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-27 13:15:00 | 1328.10 | 1366.04 | 1377.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-30 09:15:00 | 1340.60 | 1284.86 | 1301.82 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-30 09:15:00 | 1340.60 | 1284.86 | 1301.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 09:15:00 | 1340.60 | 1284.86 | 1301.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-30 10:00:00 | 1340.60 | 1284.86 | 1301.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 10:15:00 | 1331.40 | 1294.17 | 1304.51 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-30 11:15:00 | 1321.00 | 1294.17 | 1304.51 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-30 13:15:00 | 1335.00 | 1311.46 | 1310.63 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 53 — BUY (started 2026-01-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-30 13:15:00 | 1335.00 | 1311.46 | 1310.63 | EMA200 above EMA400 |

### Cycle 54 — SELL (started 2026-02-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-01 10:15:00 | 1300.70 | 1310.21 | 1310.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-01 12:15:00 | 1282.10 | 1301.83 | 1306.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-03 09:15:00 | 1295.60 | 1264.88 | 1276.88 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-03 09:15:00 | 1295.60 | 1264.88 | 1276.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 09:15:00 | 1295.60 | 1264.88 | 1276.88 | EMA400 retest candle locked (from downside) |

### Cycle 55 — BUY (started 2026-02-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 12:15:00 | 1318.00 | 1289.65 | 1286.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-04 09:15:00 | 1340.20 | 1311.16 | 1298.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-05 11:15:00 | 1314.50 | 1328.82 | 1318.34 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-05 11:15:00 | 1314.50 | 1328.82 | 1318.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 11:15:00 | 1314.50 | 1328.82 | 1318.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-05 12:00:00 | 1314.50 | 1328.82 | 1318.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 12:15:00 | 1299.20 | 1322.89 | 1316.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-05 12:45:00 | 1298.00 | 1322.89 | 1316.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 56 — SELL (started 2026-02-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-05 14:15:00 | 1288.70 | 1310.60 | 1311.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-06 09:15:00 | 1285.10 | 1302.99 | 1307.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-10 11:15:00 | 1296.80 | 1282.27 | 1288.48 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-10 11:15:00 | 1296.80 | 1282.27 | 1288.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 11:15:00 | 1296.80 | 1282.27 | 1288.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-10 11:45:00 | 1297.00 | 1282.27 | 1288.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 12:15:00 | 1293.90 | 1284.59 | 1288.98 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-10 15:15:00 | 1281.10 | 1284.84 | 1288.32 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-11 09:30:00 | 1277.00 | 1283.01 | 1286.85 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-18 09:30:00 | 1276.20 | 1261.93 | 1262.90 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-18 10:15:00 | 1282.10 | 1265.96 | 1264.65 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 57 — BUY (started 2026-02-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-18 10:15:00 | 1282.10 | 1265.96 | 1264.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-18 11:15:00 | 1285.10 | 1269.79 | 1266.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-19 15:15:00 | 1280.00 | 1280.15 | 1275.93 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-20 09:15:00 | 1270.70 | 1280.15 | 1275.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 09:15:00 | 1277.10 | 1279.54 | 1276.03 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-20 10:30:00 | 1286.60 | 1281.23 | 1277.12 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-20 14:30:00 | 1281.40 | 1282.54 | 1279.14 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-24 09:45:00 | 1284.90 | 1293.18 | 1289.32 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-24 11:00:00 | 1281.10 | 1290.76 | 1288.57 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 11:15:00 | 1289.50 | 1290.51 | 1288.66 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-02-24 14:15:00 | 1278.50 | 1286.05 | 1286.90 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 58 — SELL (started 2026-02-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-24 14:15:00 | 1278.50 | 1286.05 | 1286.90 | EMA200 below EMA400 |

### Cycle 59 — BUY (started 2026-02-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-25 13:15:00 | 1302.50 | 1289.46 | 1287.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-25 14:15:00 | 1307.50 | 1293.07 | 1289.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-27 09:15:00 | 1282.10 | 1318.14 | 1309.28 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-27 09:15:00 | 1282.10 | 1318.14 | 1309.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 09:15:00 | 1282.10 | 1318.14 | 1309.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-27 10:00:00 | 1282.10 | 1318.14 | 1309.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 10:15:00 | 1265.10 | 1307.53 | 1305.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-27 11:00:00 | 1265.10 | 1307.53 | 1305.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 60 — SELL (started 2026-02-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-27 11:15:00 | 1251.50 | 1296.32 | 1300.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-02 09:15:00 | 1231.50 | 1268.08 | 1283.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-04 09:15:00 | 1256.90 | 1229.44 | 1249.97 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-04 09:15:00 | 1256.90 | 1229.44 | 1249.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-04 09:15:00 | 1256.90 | 1229.44 | 1249.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-04 10:00:00 | 1256.90 | 1229.44 | 1249.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-04 10:15:00 | 1236.00 | 1230.75 | 1248.70 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-04 11:45:00 | 1228.00 | 1230.98 | 1247.17 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-04 12:15:00 | 1228.10 | 1230.98 | 1247.17 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-04 13:30:00 | 1229.90 | 1230.91 | 1244.36 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-05 09:30:00 | 1229.50 | 1230.01 | 1240.49 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 10:15:00 | 1241.70 | 1232.35 | 1240.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-05 10:45:00 | 1238.60 | 1232.35 | 1240.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 11:15:00 | 1235.70 | 1233.02 | 1240.15 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-05 13:00:00 | 1220.00 | 1230.41 | 1238.32 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-09 09:15:00 | 1166.60 | 1191.72 | 1209.04 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-09 09:15:00 | 1166.69 | 1191.72 | 1209.04 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-09 09:15:00 | 1168.40 | 1191.72 | 1209.04 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-09 09:15:00 | 1168.02 | 1191.72 | 1209.04 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-09 09:15:00 | 1159.00 | 1191.72 | 1209.04 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-10 09:15:00 | 1193.60 | 1169.76 | 1185.82 | SL hit (close>ema200) qty=0.50 sl=1169.76 alert=retest2 |

### Cycle 61 — BUY (started 2026-03-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-10 13:15:00 | 1214.60 | 1194.97 | 1194.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-11 09:15:00 | 1222.20 | 1206.76 | 1200.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-11 10:15:00 | 1206.10 | 1206.63 | 1200.76 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-11 11:00:00 | 1206.10 | 1206.63 | 1200.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 12:15:00 | 1200.70 | 1206.73 | 1201.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-11 13:00:00 | 1200.70 | 1206.73 | 1201.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 13:15:00 | 1210.60 | 1207.51 | 1202.70 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-11 14:15:00 | 1212.10 | 1207.51 | 1202.70 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-11 14:15:00 | 1195.30 | 1205.07 | 1202.02 | SL hit (close<static) qty=1.00 sl=1199.70 alert=retest2 |

### Cycle 62 — SELL (started 2026-03-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-12 09:15:00 | 1166.00 | 1195.48 | 1198.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-13 12:15:00 | 1147.00 | 1166.43 | 1179.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-16 10:15:00 | 1165.00 | 1163.54 | 1172.54 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-16 11:00:00 | 1165.00 | 1163.54 | 1172.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 12:15:00 | 1168.50 | 1164.70 | 1171.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-16 13:00:00 | 1168.50 | 1164.70 | 1171.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 13:15:00 | 1169.10 | 1165.58 | 1171.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-16 14:15:00 | 1169.60 | 1165.58 | 1171.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 14:15:00 | 1176.30 | 1167.72 | 1171.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-16 14:45:00 | 1177.50 | 1167.72 | 1171.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 15:15:00 | 1172.00 | 1168.58 | 1171.77 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-17 09:15:00 | 1166.70 | 1168.58 | 1171.77 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-18 09:15:00 | 1204.00 | 1174.19 | 1172.00 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 63 — BUY (started 2026-03-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 09:15:00 | 1204.00 | 1174.19 | 1172.00 | EMA200 above EMA400 |

### Cycle 64 — SELL (started 2026-03-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 10:15:00 | 1161.10 | 1173.24 | 1173.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 14:15:00 | 1150.40 | 1162.25 | 1167.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-20 09:15:00 | 1165.10 | 1161.74 | 1166.62 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-20 09:15:00 | 1165.10 | 1161.74 | 1166.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 09:15:00 | 1165.10 | 1161.74 | 1166.62 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-23 09:30:00 | 1149.60 | 1153.29 | 1160.13 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-24 15:15:00 | 1147.00 | 1143.18 | 1145.53 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-25 09:15:00 | 1169.20 | 1148.99 | 1147.80 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 65 — BUY (started 2026-03-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 09:15:00 | 1169.20 | 1148.99 | 1147.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 10:15:00 | 1179.60 | 1155.11 | 1150.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-25 15:15:00 | 1160.10 | 1165.70 | 1158.73 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-25 15:15:00 | 1160.10 | 1165.70 | 1158.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 15:15:00 | 1160.10 | 1165.70 | 1158.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 09:15:00 | 1158.00 | 1165.70 | 1158.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 09:15:00 | 1161.30 | 1164.82 | 1158.96 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-27 10:30:00 | 1164.10 | 1165.64 | 1159.86 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-27 13:00:00 | 1165.00 | 1165.73 | 1160.92 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-27 14:45:00 | 1166.70 | 1166.14 | 1161.92 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-30 09:30:00 | 1164.30 | 1165.05 | 1162.10 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-30 10:15:00 | 1168.60 | 1165.76 | 1162.69 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-30 12:00:00 | 1174.60 | 1167.53 | 1163.77 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-30 14:15:00 | 1160.80 | 1170.23 | 1166.32 | SL hit (close<static) qty=1.00 sl=1162.00 alert=retest2 |

### Cycle 66 — SELL (started 2026-04-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-13 11:15:00 | 1228.60 | 1229.90 | 1229.91 | EMA200 below EMA400 |

### Cycle 67 — BUY (started 2026-04-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-13 12:15:00 | 1232.20 | 1230.36 | 1230.12 | EMA200 above EMA400 |

### Cycle 68 — SELL (started 2026-04-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-13 14:15:00 | 1225.30 | 1229.42 | 1229.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-13 15:15:00 | 1220.00 | 1227.54 | 1228.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-15 09:15:00 | 1237.80 | 1229.59 | 1229.67 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-15 09:15:00 | 1237.80 | 1229.59 | 1229.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-15 09:15:00 | 1237.80 | 1229.59 | 1229.67 | EMA400 retest candle locked (from downside) |

### Cycle 69 — BUY (started 2026-04-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-15 10:15:00 | 1234.90 | 1230.65 | 1230.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-15 11:15:00 | 1238.10 | 1232.14 | 1230.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-15 14:15:00 | 1235.00 | 1235.36 | 1232.89 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-15 14:45:00 | 1235.10 | 1235.36 | 1232.89 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-15 15:15:00 | 1234.00 | 1235.09 | 1232.99 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-16 09:15:00 | 1241.00 | 1235.09 | 1232.99 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-16 13:15:00 | 1224.60 | 1237.04 | 1235.66 | SL hit (close<static) qty=1.00 sl=1230.30 alert=retest2 |

### Cycle 70 — SELL (started 2026-04-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-22 10:15:00 | 1247.90 | 1260.91 | 1261.48 | EMA200 below EMA400 |

### Cycle 71 — BUY (started 2026-04-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-23 09:15:00 | 1269.90 | 1261.06 | 1260.79 | EMA200 above EMA400 |

### Cycle 72 — SELL (started 2026-04-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-23 12:15:00 | 1246.30 | 1257.82 | 1259.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-24 09:15:00 | 1229.70 | 1245.63 | 1252.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-24 14:15:00 | 1236.00 | 1232.84 | 1242.40 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-24 15:00:00 | 1236.00 | 1232.84 | 1242.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 09:15:00 | 1249.00 | 1236.74 | 1242.55 | EMA400 retest candle locked (from downside) |

### Cycle 73 — BUY (started 2026-04-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-28 09:15:00 | 1258.50 | 1245.31 | 1244.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-28 10:15:00 | 1266.00 | 1249.45 | 1246.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-07 09:15:00 | 1463.10 | 1466.36 | 1429.62 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-05-07 09:30:00 | 1454.80 | 1466.36 | 1429.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-05-15 09:15:00 | 1216.00 | 2025-05-19 09:15:00 | 1133.20 | STOP_HIT | 1.00 | -6.81% |
| BUY | retest1 | 2025-05-15 13:00:00 | 1193.90 | 2025-05-19 09:15:00 | 1133.20 | STOP_HIT | 1.00 | -5.08% |
| BUY | retest1 | 2025-05-15 13:30:00 | 1191.90 | 2025-05-19 09:15:00 | 1133.20 | STOP_HIT | 1.00 | -4.92% |
| BUY | retest2 | 2025-05-16 09:30:00 | 1202.80 | 2025-05-19 09:15:00 | 1133.20 | STOP_HIT | 1.00 | -5.79% |
| BUY | retest2 | 2025-05-16 13:00:00 | 1206.00 | 2025-05-19 09:15:00 | 1133.20 | STOP_HIT | 1.00 | -6.04% |
| BUY | retest2 | 2025-05-28 09:15:00 | 1185.50 | 2025-05-30 09:15:00 | 1158.80 | STOP_HIT | 1.00 | -2.25% |
| BUY | retest2 | 2025-06-11 09:15:00 | 1213.70 | 2025-06-12 12:15:00 | 1196.40 | STOP_HIT | 1.00 | -1.43% |
| BUY | retest2 | 2025-06-12 09:45:00 | 1212.00 | 2025-06-12 12:15:00 | 1196.40 | STOP_HIT | 1.00 | -1.29% |
| BUY | retest2 | 2025-06-12 10:15:00 | 1211.30 | 2025-06-12 12:15:00 | 1196.40 | STOP_HIT | 1.00 | -1.23% |
| SELL | retest2 | 2025-06-17 09:15:00 | 1168.00 | 2025-06-20 12:15:00 | 1109.60 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-06-17 11:00:00 | 1168.50 | 2025-06-20 12:15:00 | 1110.08 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-06-17 09:15:00 | 1168.00 | 2025-06-23 09:15:00 | 1124.90 | STOP_HIT | 0.50 | 3.69% |
| SELL | retest2 | 2025-06-17 11:00:00 | 1168.50 | 2025-06-23 09:15:00 | 1124.90 | STOP_HIT | 0.50 | 3.73% |
| BUY | retest2 | 2025-06-27 09:15:00 | 1196.50 | 2025-07-04 09:15:00 | 1316.15 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-07-31 14:15:00 | 1255.20 | 2025-08-01 09:15:00 | 1267.30 | STOP_HIT | 1.00 | -0.96% |
| SELL | retest2 | 2025-07-31 15:00:00 | 1251.40 | 2025-08-01 09:15:00 | 1267.30 | STOP_HIT | 1.00 | -1.27% |
| SELL | retest2 | 2025-08-01 11:30:00 | 1254.60 | 2025-08-11 09:15:00 | 1245.00 | STOP_HIT | 1.00 | 0.77% |
| BUY | retest2 | 2025-08-14 09:45:00 | 1316.70 | 2025-08-26 11:15:00 | 1365.50 | STOP_HIT | 1.00 | 3.71% |
| BUY | retest2 | 2025-08-18 11:30:00 | 1318.80 | 2025-08-26 11:15:00 | 1365.50 | STOP_HIT | 1.00 | 3.54% |
| SELL | retest1 | 2025-09-11 15:15:00 | 1295.20 | 2025-09-15 11:15:00 | 1313.10 | STOP_HIT | 1.00 | -1.38% |
| SELL | retest1 | 2025-09-12 11:30:00 | 1292.90 | 2025-09-15 11:15:00 | 1313.10 | STOP_HIT | 1.00 | -1.56% |
| SELL | retest1 | 2025-09-12 14:45:00 | 1295.30 | 2025-09-15 11:15:00 | 1313.10 | STOP_HIT | 1.00 | -1.37% |
| BUY | retest2 | 2025-10-07 15:15:00 | 1395.00 | 2025-10-08 10:15:00 | 1373.10 | STOP_HIT | 1.00 | -1.57% |
| SELL | retest2 | 2025-10-10 10:45:00 | 1375.00 | 2025-10-10 15:15:00 | 1385.60 | STOP_HIT | 1.00 | -0.77% |
| SELL | retest2 | 2025-10-10 11:30:00 | 1375.40 | 2025-10-10 15:15:00 | 1385.60 | STOP_HIT | 1.00 | -0.74% |
| SELL | retest2 | 2025-11-10 09:45:00 | 1344.70 | 2025-11-10 10:15:00 | 1377.00 | STOP_HIT | 1.00 | -2.40% |
| BUY | retest2 | 2025-11-12 09:15:00 | 1391.00 | 2025-11-12 14:15:00 | 1367.60 | STOP_HIT | 1.00 | -1.68% |
| SELL | retest2 | 2025-11-28 09:15:00 | 1353.00 | 2025-12-05 15:15:00 | 1285.35 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-28 09:15:00 | 1353.00 | 2025-12-09 11:15:00 | 1267.20 | STOP_HIT | 0.50 | 6.34% |
| SELL | retest2 | 2025-12-18 09:15:00 | 1262.00 | 2025-12-19 11:15:00 | 1279.00 | STOP_HIT | 1.00 | -1.35% |
| SELL | retest2 | 2025-12-18 11:30:00 | 1265.00 | 2025-12-19 11:15:00 | 1279.00 | STOP_HIT | 1.00 | -1.11% |
| SELL | retest2 | 2025-12-18 12:15:00 | 1265.90 | 2025-12-19 11:15:00 | 1279.00 | STOP_HIT | 1.00 | -1.03% |
| BUY | retest2 | 2025-12-24 10:15:00 | 1314.10 | 2025-12-26 13:15:00 | 1288.10 | STOP_HIT | 1.00 | -1.98% |
| BUY | retest2 | 2025-12-26 09:45:00 | 1316.90 | 2025-12-26 13:15:00 | 1288.10 | STOP_HIT | 1.00 | -2.19% |
| BUY | retest2 | 2026-01-06 11:30:00 | 1321.30 | 2026-01-12 09:15:00 | 1301.40 | STOP_HIT | 1.00 | -1.51% |
| BUY | retest2 | 2026-01-06 12:00:00 | 1322.30 | 2026-01-12 09:15:00 | 1301.40 | STOP_HIT | 1.00 | -1.58% |
| BUY | retest2 | 2026-01-07 09:45:00 | 1320.80 | 2026-01-12 09:15:00 | 1301.40 | STOP_HIT | 1.00 | -1.47% |
| BUY | retest2 | 2026-01-07 11:00:00 | 1319.20 | 2026-01-12 09:15:00 | 1301.40 | STOP_HIT | 1.00 | -1.35% |
| BUY | retest2 | 2026-01-09 13:30:00 | 1337.30 | 2026-01-12 09:15:00 | 1301.40 | STOP_HIT | 1.00 | -2.68% |
| SELL | retest2 | 2026-01-30 11:15:00 | 1321.00 | 2026-01-30 13:15:00 | 1335.00 | STOP_HIT | 1.00 | -1.06% |
| SELL | retest2 | 2026-02-10 15:15:00 | 1281.10 | 2026-02-18 10:15:00 | 1282.10 | STOP_HIT | 1.00 | -0.08% |
| SELL | retest2 | 2026-02-11 09:30:00 | 1277.00 | 2026-02-18 10:15:00 | 1282.10 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest2 | 2026-02-18 09:30:00 | 1276.20 | 2026-02-18 10:15:00 | 1282.10 | STOP_HIT | 1.00 | -0.46% |
| BUY | retest2 | 2026-02-20 10:30:00 | 1286.60 | 2026-02-24 14:15:00 | 1278.50 | STOP_HIT | 1.00 | -0.63% |
| BUY | retest2 | 2026-02-20 14:30:00 | 1281.40 | 2026-02-24 14:15:00 | 1278.50 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest2 | 2026-02-24 09:45:00 | 1284.90 | 2026-02-24 14:15:00 | 1278.50 | STOP_HIT | 1.00 | -0.50% |
| BUY | retest2 | 2026-02-24 11:00:00 | 1281.10 | 2026-02-24 14:15:00 | 1278.50 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest2 | 2026-03-04 11:45:00 | 1228.00 | 2026-03-09 09:15:00 | 1166.60 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-04 12:15:00 | 1228.10 | 2026-03-09 09:15:00 | 1166.69 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-04 13:30:00 | 1229.90 | 2026-03-09 09:15:00 | 1168.40 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-05 09:30:00 | 1229.50 | 2026-03-09 09:15:00 | 1168.02 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-05 13:00:00 | 1220.00 | 2026-03-09 09:15:00 | 1159.00 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-04 11:45:00 | 1228.00 | 2026-03-10 09:15:00 | 1193.60 | STOP_HIT | 0.50 | 2.80% |
| SELL | retest2 | 2026-03-04 12:15:00 | 1228.10 | 2026-03-10 09:15:00 | 1193.60 | STOP_HIT | 0.50 | 2.81% |
| SELL | retest2 | 2026-03-04 13:30:00 | 1229.90 | 2026-03-10 09:15:00 | 1193.60 | STOP_HIT | 0.50 | 2.95% |
| SELL | retest2 | 2026-03-05 09:30:00 | 1229.50 | 2026-03-10 09:15:00 | 1193.60 | STOP_HIT | 0.50 | 2.92% |
| SELL | retest2 | 2026-03-05 13:00:00 | 1220.00 | 2026-03-10 09:15:00 | 1193.60 | STOP_HIT | 0.50 | 2.16% |
| BUY | retest2 | 2026-03-11 14:15:00 | 1212.10 | 2026-03-11 14:15:00 | 1195.30 | STOP_HIT | 1.00 | -1.39% |
| SELL | retest2 | 2026-03-17 09:15:00 | 1166.70 | 2026-03-18 09:15:00 | 1204.00 | STOP_HIT | 1.00 | -3.20% |
| SELL | retest2 | 2026-03-23 09:30:00 | 1149.60 | 2026-03-25 09:15:00 | 1169.20 | STOP_HIT | 1.00 | -1.70% |
| SELL | retest2 | 2026-03-24 15:15:00 | 1147.00 | 2026-03-25 09:15:00 | 1169.20 | STOP_HIT | 1.00 | -1.94% |
| BUY | retest2 | 2026-03-27 10:30:00 | 1164.10 | 2026-03-30 14:15:00 | 1160.80 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest2 | 2026-03-27 13:00:00 | 1165.00 | 2026-04-07 09:15:00 | 1280.51 | TARGET_HIT | 1.00 | 9.92% |
| BUY | retest2 | 2026-03-27 14:45:00 | 1166.70 | 2026-04-07 09:15:00 | 1281.50 | TARGET_HIT | 1.00 | 9.84% |
| BUY | retest2 | 2026-03-30 09:30:00 | 1164.30 | 2026-04-07 09:15:00 | 1283.37 | TARGET_HIT | 1.00 | 10.23% |
| BUY | retest2 | 2026-03-30 12:00:00 | 1174.60 | 2026-04-07 09:15:00 | 1280.73 | TARGET_HIT | 1.00 | 9.04% |
| BUY | retest2 | 2026-04-01 09:15:00 | 1187.80 | 2026-04-07 09:15:00 | 1292.28 | TARGET_HIT | 1.00 | 8.80% |
| BUY | retest2 | 2026-04-01 09:45:00 | 1174.80 | 2026-04-07 09:15:00 | 1295.91 | TARGET_HIT | 1.00 | 10.31% |
| BUY | retest2 | 2026-04-01 10:30:00 | 1178.10 | 2026-04-13 11:15:00 | 1228.60 | STOP_HIT | 1.00 | 4.29% |
| BUY | retest2 | 2026-04-06 12:15:00 | 1191.00 | 2026-04-13 11:15:00 | 1228.60 | STOP_HIT | 1.00 | 3.16% |
| BUY | retest2 | 2026-04-06 14:15:00 | 1189.70 | 2026-04-13 11:15:00 | 1228.60 | STOP_HIT | 1.00 | 3.27% |
| BUY | retest2 | 2026-04-07 09:15:00 | 1257.00 | 2026-04-13 11:15:00 | 1228.60 | STOP_HIT | 1.00 | -2.26% |
| BUY | retest2 | 2026-04-16 09:15:00 | 1241.00 | 2026-04-16 13:15:00 | 1224.60 | STOP_HIT | 1.00 | -1.32% |
| BUY | retest2 | 2026-04-16 15:00:00 | 1241.80 | 2026-04-22 10:15:00 | 1247.90 | STOP_HIT | 1.00 | 0.49% |
