# Carborundum Universal Ltd. (CARBORUNIV)

## Backtest Summary

- **Window:** 2023-03-14 09:15:00 → 2026-05-08 15:15:00 (5436 bars)
- **Last close:** 1020.20
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 216 |
| ALERT1 | 137 |
| ALERT2 | 136 |
| ALERT2_SKIP | 79 |
| ALERT3 | 423 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 5 |
| ENTRY2 | 214 |
| PARTIAL | 28 |
| TARGET_HIT | 11 |
| STOP_HIT | 208 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 247 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 103 / 144
- **Target hits / Stop hits / Partials:** 11 / 208 / 28
- **Avg / median % per leg:** 0.76% / -0.61%
- **Sum % (uncompounded):** 187.41%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 95 | 39 | 41.1% | 11 | 82 | 2 | 0.94% | 88.9% |
| BUY @ 2nd Alert (retest1) | 5 | 4 | 80.0% | 1 | 2 | 2 | 4.20% | 21.0% |
| BUY @ 3rd Alert (retest2) | 90 | 35 | 38.9% | 10 | 80 | 0 | 0.75% | 67.9% |
| SELL (all) | 152 | 64 | 42.1% | 0 | 126 | 26 | 0.65% | 98.5% |
| SELL @ 2nd Alert (retest1) | 3 | 3 | 100.0% | 0 | 2 | 1 | 3.86% | 11.6% |
| SELL @ 3rd Alert (retest2) | 149 | 61 | 40.9% | 0 | 124 | 25 | 0.58% | 87.0% |
| retest1 (combined) | 8 | 7 | 87.5% | 1 | 4 | 3 | 4.07% | 32.6% |
| retest2 (combined) | 239 | 96 | 40.2% | 10 | 204 | 25 | 0.65% | 154.9% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-05-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-24 12:15:00 | 1181.00 | 1197.90 | 1199.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-05-24 13:15:00 | 1167.50 | 1191.82 | 1196.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-05-26 14:15:00 | 1162.50 | 1160.87 | 1169.86 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-05-26 15:00:00 | 1162.50 | 1160.87 | 1169.86 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-29 13:15:00 | 1168.50 | 1159.14 | 1164.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-05-29 13:45:00 | 1168.05 | 1159.14 | 1164.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-29 14:15:00 | 1158.90 | 1159.10 | 1163.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-05-29 14:30:00 | 1169.95 | 1159.10 | 1163.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-30 10:15:00 | 1151.65 | 1156.27 | 1161.29 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-05-30 12:15:00 | 1150.45 | 1155.22 | 1160.35 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-05-30 14:15:00 | 1150.45 | 1155.24 | 1159.51 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-05-30 15:15:00 | 1150.00 | 1154.74 | 1158.90 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-05-31 11:00:00 | 1150.05 | 1152.25 | 1156.59 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-01 09:15:00 | 1156.45 | 1152.11 | 1154.47 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2023-06-02 09:15:00 | 1164.95 | 1156.48 | 1155.69 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 2 — BUY (started 2023-06-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-02 09:15:00 | 1164.95 | 1156.48 | 1155.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-07 10:15:00 | 1181.50 | 1167.32 | 1164.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-09 11:15:00 | 1194.40 | 1195.37 | 1188.74 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-09 11:15:00 | 1194.40 | 1195.37 | 1188.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-09 11:15:00 | 1194.40 | 1195.37 | 1188.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-09 11:45:00 | 1188.20 | 1195.37 | 1188.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-09 12:15:00 | 1191.55 | 1194.60 | 1189.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-09 12:45:00 | 1189.10 | 1194.60 | 1189.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-09 13:15:00 | 1187.90 | 1193.26 | 1188.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-09 13:45:00 | 1187.15 | 1193.26 | 1188.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-09 14:15:00 | 1186.00 | 1191.81 | 1188.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-09 15:00:00 | 1186.00 | 1191.81 | 1188.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-09 15:15:00 | 1172.05 | 1187.86 | 1187.13 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-12 09:15:00 | 1192.10 | 1187.86 | 1187.13 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-13 09:30:00 | 1193.55 | 1193.25 | 1191.37 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-06-20 09:15:00 | 1216.65 | 1229.11 | 1229.25 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 3 — SELL (started 2023-06-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-20 09:15:00 | 1216.65 | 1229.11 | 1229.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-20 11:15:00 | 1206.35 | 1221.78 | 1225.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-21 09:15:00 | 1219.45 | 1215.63 | 1220.54 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-06-21 09:45:00 | 1219.20 | 1215.63 | 1220.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-21 10:15:00 | 1204.15 | 1213.33 | 1219.05 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-22 12:30:00 | 1201.25 | 1215.41 | 1217.20 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-06-22 14:15:00 | 1220.00 | 1217.03 | 1217.67 | SL hit (close>static) qty=1.00 sl=1219.95 alert=retest2 |

### Cycle 4 — BUY (started 2023-06-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-30 10:15:00 | 1189.15 | 1184.08 | 1183.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-30 15:15:00 | 1202.00 | 1192.05 | 1188.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-03 13:15:00 | 1207.50 | 1207.65 | 1198.53 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-07-03 14:00:00 | 1207.50 | 1207.65 | 1198.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-03 15:15:00 | 1209.80 | 1207.51 | 1200.02 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-04 09:15:00 | 1218.60 | 1207.51 | 1200.02 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-07-04 12:15:00 | 1198.00 | 1208.22 | 1203.20 | SL hit (close<static) qty=1.00 sl=1200.00 alert=retest2 |

### Cycle 5 — SELL (started 2023-07-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-06 15:15:00 | 1197.10 | 1204.80 | 1205.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-07 13:15:00 | 1192.10 | 1200.53 | 1203.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-11 09:15:00 | 1187.80 | 1185.91 | 1191.86 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-11 09:15:00 | 1187.80 | 1185.91 | 1191.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-11 09:15:00 | 1187.80 | 1185.91 | 1191.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-11 09:45:00 | 1204.05 | 1185.91 | 1191.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-11 10:15:00 | 1178.65 | 1184.46 | 1190.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-11 10:30:00 | 1184.05 | 1184.46 | 1190.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-11 12:15:00 | 1182.05 | 1182.81 | 1188.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-11 12:45:00 | 1184.40 | 1182.81 | 1188.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-11 13:15:00 | 1185.15 | 1183.28 | 1188.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-11 13:45:00 | 1183.05 | 1183.28 | 1188.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-11 14:15:00 | 1193.95 | 1185.41 | 1188.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-11 15:00:00 | 1193.95 | 1185.41 | 1188.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-11 15:15:00 | 1199.00 | 1188.13 | 1189.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-12 09:15:00 | 1204.45 | 1188.13 | 1189.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-12 09:15:00 | 1201.20 | 1190.74 | 1190.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-12 09:30:00 | 1205.10 | 1190.74 | 1190.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 6 — BUY (started 2023-07-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-12 10:15:00 | 1196.15 | 1191.82 | 1191.36 | EMA200 above EMA400 |

### Cycle 7 — SELL (started 2023-07-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-12 12:15:00 | 1184.90 | 1190.36 | 1190.77 | EMA200 below EMA400 |

### Cycle 8 — BUY (started 2023-07-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-12 14:15:00 | 1195.50 | 1190.98 | 1190.95 | EMA200 above EMA400 |

### Cycle 9 — SELL (started 2023-07-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-13 11:15:00 | 1186.00 | 1190.30 | 1190.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-14 10:15:00 | 1177.70 | 1184.62 | 1187.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-14 12:15:00 | 1186.20 | 1184.33 | 1186.89 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-14 12:15:00 | 1186.20 | 1184.33 | 1186.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-14 12:15:00 | 1186.20 | 1184.33 | 1186.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-14 12:45:00 | 1185.60 | 1184.33 | 1186.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-14 13:15:00 | 1189.15 | 1185.30 | 1187.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-14 14:00:00 | 1189.15 | 1185.30 | 1187.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-14 14:15:00 | 1179.70 | 1184.18 | 1186.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-14 15:15:00 | 1189.00 | 1184.18 | 1186.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-14 15:15:00 | 1189.00 | 1185.14 | 1186.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-17 09:15:00 | 1198.15 | 1185.14 | 1186.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-17 09:15:00 | 1196.25 | 1187.36 | 1187.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-17 09:30:00 | 1197.30 | 1187.36 | 1187.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 10 — BUY (started 2023-07-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-17 10:15:00 | 1193.40 | 1188.57 | 1188.06 | EMA200 above EMA400 |

### Cycle 11 — SELL (started 2023-07-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-17 12:15:00 | 1174.90 | 1186.58 | 1187.29 | EMA200 below EMA400 |

### Cycle 12 — BUY (started 2023-07-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-19 09:15:00 | 1209.85 | 1190.33 | 1187.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-19 10:15:00 | 1218.05 | 1195.87 | 1190.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-20 11:15:00 | 1212.10 | 1213.20 | 1204.52 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-07-20 12:00:00 | 1212.10 | 1213.20 | 1204.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-20 14:15:00 | 1219.30 | 1215.34 | 1207.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-20 14:45:00 | 1212.85 | 1215.34 | 1207.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-20 15:15:00 | 1207.50 | 1213.77 | 1207.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-21 09:30:00 | 1205.45 | 1211.98 | 1207.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-21 10:15:00 | 1200.80 | 1209.74 | 1206.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-21 10:30:00 | 1199.90 | 1209.74 | 1206.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 13 — SELL (started 2023-07-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-21 14:15:00 | 1198.35 | 1204.43 | 1205.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-24 09:15:00 | 1177.10 | 1198.23 | 1202.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-24 12:15:00 | 1202.50 | 1194.92 | 1199.16 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-24 12:15:00 | 1202.50 | 1194.92 | 1199.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-24 12:15:00 | 1202.50 | 1194.92 | 1199.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-24 13:00:00 | 1202.50 | 1194.92 | 1199.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-24 13:15:00 | 1197.50 | 1195.44 | 1199.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-24 14:15:00 | 1200.70 | 1195.44 | 1199.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-24 14:15:00 | 1188.70 | 1194.09 | 1198.07 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-25 11:00:00 | 1184.10 | 1192.03 | 1196.10 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-07-26 09:15:00 | 1215.35 | 1190.49 | 1192.41 | SL hit (close>static) qty=1.00 sl=1204.50 alert=retest2 |

### Cycle 14 — BUY (started 2023-07-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-26 10:15:00 | 1208.80 | 1194.15 | 1193.90 | EMA200 above EMA400 |

### Cycle 15 — SELL (started 2023-07-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-26 12:15:00 | 1191.75 | 1193.43 | 1193.60 | EMA200 below EMA400 |

### Cycle 16 — BUY (started 2023-07-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-27 09:15:00 | 1204.05 | 1195.17 | 1194.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-27 10:15:00 | 1212.00 | 1198.53 | 1195.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-31 09:15:00 | 1210.20 | 1217.56 | 1211.73 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-31 09:15:00 | 1210.20 | 1217.56 | 1211.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-31 09:15:00 | 1210.20 | 1217.56 | 1211.73 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-01 09:30:00 | 1240.00 | 1218.20 | 1214.53 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-01 10:45:00 | 1243.50 | 1223.65 | 1217.34 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-08-07 09:15:00 | 1225.00 | 1251.83 | 1253.53 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 17 — SELL (started 2023-08-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-07 09:15:00 | 1225.00 | 1251.83 | 1253.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-07 10:15:00 | 1209.15 | 1243.30 | 1249.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-16 12:15:00 | 1079.85 | 1060.90 | 1077.01 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-16 12:15:00 | 1079.85 | 1060.90 | 1077.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-16 12:15:00 | 1079.85 | 1060.90 | 1077.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-16 13:00:00 | 1079.85 | 1060.90 | 1077.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-16 13:15:00 | 1089.85 | 1066.69 | 1078.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-16 13:45:00 | 1087.00 | 1066.69 | 1078.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-16 14:15:00 | 1088.55 | 1071.06 | 1079.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-16 14:30:00 | 1080.00 | 1071.06 | 1079.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-16 15:15:00 | 1097.50 | 1076.35 | 1080.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-17 09:15:00 | 1087.85 | 1076.35 | 1080.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-18 09:15:00 | 1091.00 | 1072.89 | 1075.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-18 10:00:00 | 1091.00 | 1072.89 | 1075.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-18 10:15:00 | 1088.10 | 1075.93 | 1076.52 | EMA400 retest candle locked (from downside) |

### Cycle 18 — BUY (started 2023-08-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-18 11:15:00 | 1100.00 | 1080.74 | 1078.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-21 09:15:00 | 1139.70 | 1096.99 | 1087.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-22 09:15:00 | 1112.75 | 1115.37 | 1104.20 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-08-22 10:00:00 | 1112.75 | 1115.37 | 1104.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-25 09:15:00 | 1131.05 | 1138.23 | 1131.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-25 09:45:00 | 1131.75 | 1138.23 | 1131.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-25 10:15:00 | 1138.95 | 1138.38 | 1132.23 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-28 14:00:00 | 1141.05 | 1136.67 | 1134.45 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-29 14:45:00 | 1141.40 | 1137.47 | 1136.22 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-30 10:15:00 | 1142.90 | 1136.75 | 1136.07 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-30 12:30:00 | 1142.10 | 1138.18 | 1136.94 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-31 09:15:00 | 1135.30 | 1139.27 | 1138.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-31 09:45:00 | 1136.50 | 1139.27 | 1138.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-31 10:15:00 | 1135.00 | 1138.42 | 1137.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-31 10:30:00 | 1134.85 | 1138.42 | 1137.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-31 12:15:00 | 1135.10 | 1137.47 | 1137.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-31 13:00:00 | 1135.10 | 1137.47 | 1137.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2023-08-31 13:15:00 | 1135.50 | 1137.08 | 1137.26 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 19 — SELL (started 2023-08-31 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-31 13:15:00 | 1135.50 | 1137.08 | 1137.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-01 09:15:00 | 1114.70 | 1132.77 | 1135.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-01 13:15:00 | 1127.90 | 1125.08 | 1130.16 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-01 13:15:00 | 1127.90 | 1125.08 | 1130.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-01 13:15:00 | 1127.90 | 1125.08 | 1130.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-01 14:00:00 | 1127.90 | 1125.08 | 1130.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-01 14:15:00 | 1138.00 | 1127.67 | 1130.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-01 15:00:00 | 1138.00 | 1127.67 | 1130.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-01 15:15:00 | 1132.00 | 1128.53 | 1130.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-04 09:15:00 | 1134.30 | 1128.53 | 1130.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-04 09:15:00 | 1142.25 | 1131.28 | 1132.00 | EMA400 retest candle locked (from downside) |

### Cycle 20 — BUY (started 2023-09-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-04 10:15:00 | 1149.00 | 1134.82 | 1133.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-04 12:15:00 | 1160.90 | 1141.24 | 1136.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-06 11:15:00 | 1178.65 | 1179.94 | 1169.41 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-09-06 11:30:00 | 1177.05 | 1179.94 | 1169.41 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-06 12:15:00 | 1183.15 | 1180.58 | 1170.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-06 12:45:00 | 1177.90 | 1180.58 | 1170.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-07 09:15:00 | 1185.00 | 1182.60 | 1174.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-07 09:30:00 | 1173.90 | 1182.60 | 1174.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-07 11:15:00 | 1177.35 | 1180.97 | 1175.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-07 12:00:00 | 1177.35 | 1180.97 | 1175.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-07 12:15:00 | 1184.65 | 1181.71 | 1176.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-07 12:30:00 | 1181.90 | 1181.71 | 1176.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-11 12:15:00 | 1207.10 | 1202.78 | 1194.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-11 12:45:00 | 1201.35 | 1202.78 | 1194.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-12 09:15:00 | 1186.65 | 1200.92 | 1196.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-12 09:45:00 | 1186.40 | 1200.92 | 1196.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-12 10:15:00 | 1186.40 | 1198.02 | 1195.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-12 10:30:00 | 1193.45 | 1198.02 | 1195.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 21 — SELL (started 2023-09-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-12 11:15:00 | 1168.00 | 1192.01 | 1192.97 | EMA200 below EMA400 |

### Cycle 22 — BUY (started 2023-09-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-13 14:15:00 | 1194.90 | 1192.88 | 1192.70 | EMA200 above EMA400 |

### Cycle 23 — SELL (started 2023-09-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-13 15:15:00 | 1186.00 | 1191.51 | 1192.09 | EMA200 below EMA400 |

### Cycle 24 — BUY (started 2023-09-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-14 09:15:00 | 1208.85 | 1194.98 | 1193.61 | EMA200 above EMA400 |

### Cycle 25 — SELL (started 2023-09-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-15 10:15:00 | 1185.30 | 1194.62 | 1194.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-18 10:15:00 | 1179.00 | 1189.07 | 1191.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-21 10:15:00 | 1165.25 | 1163.87 | 1172.23 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-09-21 10:45:00 | 1168.55 | 1163.87 | 1172.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-21 12:15:00 | 1170.00 | 1165.65 | 1171.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-21 12:45:00 | 1169.90 | 1165.65 | 1171.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-21 13:15:00 | 1170.00 | 1166.52 | 1171.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-21 13:30:00 | 1170.15 | 1166.52 | 1171.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-22 09:15:00 | 1191.00 | 1169.18 | 1171.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-22 10:00:00 | 1191.00 | 1169.18 | 1171.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 26 — BUY (started 2023-09-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-22 10:15:00 | 1243.35 | 1184.02 | 1177.84 | EMA200 above EMA400 |

### Cycle 27 — SELL (started 2023-09-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-26 10:15:00 | 1172.55 | 1185.99 | 1186.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-26 12:15:00 | 1166.80 | 1180.10 | 1183.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-27 09:15:00 | 1180.85 | 1177.55 | 1180.89 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-27 09:15:00 | 1180.85 | 1177.55 | 1180.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-27 09:15:00 | 1180.85 | 1177.55 | 1180.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-27 09:45:00 | 1182.95 | 1177.55 | 1180.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-27 10:15:00 | 1182.05 | 1178.45 | 1181.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-27 10:30:00 | 1178.60 | 1178.45 | 1181.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-27 11:15:00 | 1180.65 | 1178.89 | 1180.97 | EMA400 retest candle locked (from downside) |

### Cycle 28 — BUY (started 2023-09-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-28 09:15:00 | 1193.85 | 1183.04 | 1182.13 | EMA200 above EMA400 |

### Cycle 29 — SELL (started 2023-09-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-28 14:15:00 | 1175.85 | 1181.84 | 1182.14 | EMA200 below EMA400 |

### Cycle 30 — BUY (started 2023-09-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-29 09:15:00 | 1189.90 | 1183.64 | 1182.92 | EMA200 above EMA400 |

### Cycle 31 — SELL (started 2023-09-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-29 15:15:00 | 1173.00 | 1182.28 | 1182.97 | EMA200 below EMA400 |

### Cycle 32 — BUY (started 2023-10-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-04 13:15:00 | 1185.05 | 1179.25 | 1178.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-05 14:15:00 | 1192.85 | 1182.79 | 1180.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-06 10:15:00 | 1179.50 | 1183.75 | 1182.01 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-06 10:15:00 | 1179.50 | 1183.75 | 1182.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-06 10:15:00 | 1179.50 | 1183.75 | 1182.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-06 11:00:00 | 1179.50 | 1183.75 | 1182.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-06 11:15:00 | 1172.00 | 1181.40 | 1181.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-06 12:00:00 | 1172.00 | 1181.40 | 1181.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 33 — SELL (started 2023-10-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-06 12:15:00 | 1169.50 | 1179.02 | 1180.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-06 15:15:00 | 1167.00 | 1173.86 | 1177.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-10 09:15:00 | 1154.50 | 1143.00 | 1154.92 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-10 09:15:00 | 1154.50 | 1143.00 | 1154.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-10 09:15:00 | 1154.50 | 1143.00 | 1154.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-10 10:00:00 | 1154.50 | 1143.00 | 1154.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-10 10:15:00 | 1151.80 | 1144.76 | 1154.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-10 11:15:00 | 1155.00 | 1144.76 | 1154.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-10 11:15:00 | 1159.70 | 1147.75 | 1155.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-10 12:00:00 | 1159.70 | 1147.75 | 1155.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-10 12:15:00 | 1150.65 | 1148.33 | 1154.69 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-10 13:30:00 | 1148.50 | 1147.04 | 1153.53 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-10-11 10:15:00 | 1164.00 | 1152.87 | 1154.38 | SL hit (close>static) qty=1.00 sl=1160.00 alert=retest2 |

### Cycle 34 — BUY (started 2023-10-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-11 11:15:00 | 1167.75 | 1155.85 | 1155.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-13 09:15:00 | 1189.50 | 1167.30 | 1163.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-13 14:15:00 | 1168.15 | 1172.94 | 1168.56 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-13 14:15:00 | 1168.15 | 1172.94 | 1168.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-13 14:15:00 | 1168.15 | 1172.94 | 1168.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-13 15:00:00 | 1168.15 | 1172.94 | 1168.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-13 15:15:00 | 1163.00 | 1170.95 | 1168.06 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-16 09:15:00 | 1171.75 | 1170.95 | 1168.06 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-10-16 11:15:00 | 1159.90 | 1167.11 | 1166.89 | SL hit (close<static) qty=1.00 sl=1163.00 alert=retest2 |

### Cycle 35 — SELL (started 2023-10-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-16 12:15:00 | 1161.75 | 1166.04 | 1166.42 | EMA200 below EMA400 |

### Cycle 36 — BUY (started 2023-10-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-17 12:15:00 | 1170.75 | 1166.85 | 1166.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-17 14:15:00 | 1172.00 | 1168.38 | 1167.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-18 10:15:00 | 1169.25 | 1169.80 | 1168.35 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-10-18 10:30:00 | 1170.00 | 1169.80 | 1168.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-18 11:15:00 | 1170.00 | 1169.84 | 1168.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-18 11:45:00 | 1168.75 | 1169.84 | 1168.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-18 12:15:00 | 1168.50 | 1169.57 | 1168.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-18 13:00:00 | 1168.50 | 1169.57 | 1168.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 37 — SELL (started 2023-10-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-18 13:15:00 | 1160.00 | 1167.66 | 1167.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-18 14:15:00 | 1144.00 | 1162.93 | 1165.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-19 09:15:00 | 1164.45 | 1159.68 | 1163.41 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-19 09:15:00 | 1164.45 | 1159.68 | 1163.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-19 09:15:00 | 1164.45 | 1159.68 | 1163.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-19 09:45:00 | 1164.30 | 1159.68 | 1163.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-19 10:15:00 | 1161.10 | 1159.96 | 1163.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-19 10:45:00 | 1163.80 | 1159.96 | 1163.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-19 11:15:00 | 1156.95 | 1159.36 | 1162.63 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-19 14:15:00 | 1143.00 | 1156.60 | 1160.76 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-20 11:00:00 | 1145.30 | 1148.04 | 1154.79 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-25 12:15:00 | 1085.85 | 1098.18 | 1115.20 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-25 12:15:00 | 1088.03 | 1098.18 | 1115.20 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2023-10-25 15:15:00 | 1095.00 | 1092.82 | 1108.02 | SL hit (close>ema200) qty=0.50 sl=1092.82 alert=retest2 |

### Cycle 38 — BUY (started 2023-10-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-27 10:15:00 | 1140.45 | 1111.61 | 1108.71 | EMA200 above EMA400 |

### Cycle 39 — SELL (started 2023-10-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-27 15:15:00 | 1085.95 | 1108.84 | 1109.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-30 09:15:00 | 1078.70 | 1102.81 | 1106.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-31 09:15:00 | 1082.35 | 1082.03 | 1091.81 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-10-31 15:00:00 | 1074.00 | 1078.75 | 1086.44 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-02 09:15:00 | 1071.20 | 1065.59 | 1072.67 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2023-11-02 10:15:00 | 1073.00 | 1067.07 | 1072.70 | SL hit (close>ema400) qty=1.00 sl=1072.70 alert=retest1 |

### Cycle 40 — BUY (started 2023-11-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-03 09:15:00 | 1094.00 | 1075.96 | 1074.92 | EMA200 above EMA400 |

### Cycle 41 — SELL (started 2023-11-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-03 14:15:00 | 1065.00 | 1073.36 | 1074.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-06 09:15:00 | 1053.25 | 1067.84 | 1071.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-06 11:15:00 | 1070.80 | 1067.18 | 1070.56 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-06 11:15:00 | 1070.80 | 1067.18 | 1070.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-06 11:15:00 | 1070.80 | 1067.18 | 1070.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-06 12:00:00 | 1070.80 | 1067.18 | 1070.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-06 12:15:00 | 1086.00 | 1070.94 | 1071.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-06 13:00:00 | 1086.00 | 1070.94 | 1071.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 42 — BUY (started 2023-11-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-06 13:15:00 | 1085.00 | 1073.75 | 1073.15 | EMA200 above EMA400 |

### Cycle 43 — SELL (started 2023-11-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-08 13:15:00 | 1069.90 | 1075.46 | 1075.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-09 09:15:00 | 1050.40 | 1068.87 | 1072.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-09 13:15:00 | 1069.40 | 1067.82 | 1070.86 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-11-09 14:00:00 | 1069.40 | 1067.82 | 1070.86 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-09 14:15:00 | 1071.35 | 1068.53 | 1070.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-09 14:30:00 | 1076.45 | 1068.53 | 1070.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-09 15:15:00 | 1072.00 | 1069.22 | 1071.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-10 09:15:00 | 1071.90 | 1069.22 | 1071.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-10 09:15:00 | 1077.25 | 1070.83 | 1071.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-10 10:00:00 | 1077.25 | 1070.83 | 1071.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 44 — BUY (started 2023-11-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-10 10:15:00 | 1082.50 | 1073.16 | 1072.56 | EMA200 above EMA400 |

### Cycle 45 — SELL (started 2023-11-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-13 14:15:00 | 1071.50 | 1072.70 | 1072.81 | EMA200 below EMA400 |

### Cycle 46 — BUY (started 2023-11-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-15 09:15:00 | 1089.20 | 1075.56 | 1074.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-16 10:15:00 | 1102.00 | 1088.81 | 1082.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-17 12:15:00 | 1102.40 | 1106.38 | 1097.16 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-11-17 12:45:00 | 1103.40 | 1106.38 | 1097.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-20 11:15:00 | 1107.20 | 1108.29 | 1102.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-20 11:45:00 | 1105.30 | 1108.29 | 1102.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-20 12:15:00 | 1097.75 | 1106.18 | 1101.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-20 13:00:00 | 1097.75 | 1106.18 | 1101.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-20 13:15:00 | 1099.05 | 1104.75 | 1101.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-20 13:45:00 | 1098.60 | 1104.75 | 1101.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-20 14:15:00 | 1097.40 | 1103.28 | 1101.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-20 14:30:00 | 1090.00 | 1103.28 | 1101.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-20 15:15:00 | 1094.00 | 1101.43 | 1100.59 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-21 09:15:00 | 1101.05 | 1101.43 | 1100.59 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-11-21 11:15:00 | 1095.00 | 1099.18 | 1099.73 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 47 — SELL (started 2023-11-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-21 11:15:00 | 1095.00 | 1099.18 | 1099.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-21 13:15:00 | 1089.95 | 1096.51 | 1098.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-22 10:15:00 | 1095.75 | 1095.09 | 1097.02 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-22 10:15:00 | 1095.75 | 1095.09 | 1097.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-22 10:15:00 | 1095.75 | 1095.09 | 1097.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-22 11:00:00 | 1095.75 | 1095.09 | 1097.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-22 11:15:00 | 1099.00 | 1095.87 | 1097.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-22 12:00:00 | 1099.00 | 1095.87 | 1097.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 48 — BUY (started 2023-11-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-22 12:15:00 | 1163.30 | 1109.36 | 1103.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-28 09:15:00 | 1195.80 | 1147.45 | 1137.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-28 15:15:00 | 1149.90 | 1155.74 | 1147.07 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-28 15:15:00 | 1149.90 | 1155.74 | 1147.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-28 15:15:00 | 1149.90 | 1155.74 | 1147.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-29 09:30:00 | 1141.45 | 1154.24 | 1147.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-29 10:15:00 | 1150.45 | 1153.48 | 1147.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-29 10:30:00 | 1149.20 | 1153.48 | 1147.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-29 11:15:00 | 1151.05 | 1153.00 | 1147.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-29 11:30:00 | 1149.45 | 1153.00 | 1147.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-29 13:15:00 | 1156.00 | 1153.80 | 1149.08 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-29 15:15:00 | 1159.90 | 1154.09 | 1149.64 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-11-30 09:15:00 | 1131.00 | 1150.40 | 1148.79 | SL hit (close<static) qty=1.00 sl=1145.20 alert=retest2 |

### Cycle 49 — SELL (started 2023-11-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-30 10:15:00 | 1125.00 | 1145.32 | 1146.63 | EMA200 below EMA400 |

### Cycle 50 — BUY (started 2023-11-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-30 14:15:00 | 1209.95 | 1154.61 | 1149.84 | EMA200 above EMA400 |

### Cycle 51 — SELL (started 2023-12-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-11 13:15:00 | 1190.50 | 1194.30 | 1194.58 | EMA200 below EMA400 |

### Cycle 52 — BUY (started 2023-12-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-12 09:15:00 | 1216.00 | 1198.17 | 1196.22 | EMA200 above EMA400 |

### Cycle 53 — SELL (started 2023-12-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-13 13:15:00 | 1183.05 | 1198.74 | 1200.66 | EMA200 below EMA400 |

### Cycle 54 — BUY (started 2023-12-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-18 10:15:00 | 1203.70 | 1196.50 | 1196.02 | EMA200 above EMA400 |

### Cycle 55 — SELL (started 2023-12-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-18 14:15:00 | 1191.45 | 1195.09 | 1195.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-19 13:15:00 | 1179.30 | 1192.12 | 1193.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-22 09:15:00 | 1124.20 | 1120.13 | 1138.07 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-12-22 10:00:00 | 1124.20 | 1120.13 | 1138.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-27 09:15:00 | 1117.00 | 1106.56 | 1112.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-27 09:45:00 | 1116.90 | 1106.56 | 1112.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-27 10:15:00 | 1122.05 | 1109.66 | 1113.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-27 10:45:00 | 1127.95 | 1109.66 | 1113.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-27 12:15:00 | 1112.45 | 1110.96 | 1113.38 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-12-27 15:00:00 | 1109.70 | 1111.20 | 1113.10 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-12-28 09:15:00 | 1121.00 | 1113.33 | 1113.75 | SL hit (close>static) qty=1.00 sl=1117.40 alert=retest2 |

### Cycle 56 — BUY (started 2023-12-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-28 10:15:00 | 1118.00 | 1114.26 | 1114.13 | EMA200 above EMA400 |

### Cycle 57 — SELL (started 2023-12-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-28 13:15:00 | 1110.00 | 1114.08 | 1114.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-28 14:15:00 | 1094.20 | 1110.10 | 1112.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-29 11:15:00 | 1111.80 | 1108.16 | 1110.45 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-29 11:15:00 | 1111.80 | 1108.16 | 1110.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-29 11:15:00 | 1111.80 | 1108.16 | 1110.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-29 11:45:00 | 1111.95 | 1108.16 | 1110.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-29 12:15:00 | 1110.00 | 1108.53 | 1110.41 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-01 09:45:00 | 1107.20 | 1110.38 | 1110.78 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-01-01 10:15:00 | 1116.60 | 1111.62 | 1111.31 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 58 — BUY (started 2024-01-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-01 10:15:00 | 1116.60 | 1111.62 | 1111.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-02 15:15:00 | 1123.20 | 1115.56 | 1113.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-03 09:15:00 | 1111.05 | 1114.66 | 1113.67 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-03 09:15:00 | 1111.05 | 1114.66 | 1113.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-03 09:15:00 | 1111.05 | 1114.66 | 1113.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-03 09:30:00 | 1108.50 | 1114.66 | 1113.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-03 10:15:00 | 1121.70 | 1116.07 | 1114.40 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-03 12:45:00 | 1125.40 | 1119.23 | 1116.20 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-03 13:45:00 | 1126.80 | 1120.91 | 1117.24 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-03 14:45:00 | 1127.00 | 1120.90 | 1117.57 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-04 09:30:00 | 1130.20 | 1122.41 | 1118.87 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-04 13:15:00 | 1118.65 | 1123.13 | 1120.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-04 14:00:00 | 1118.65 | 1123.13 | 1120.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-04 14:15:00 | 1116.20 | 1121.75 | 1120.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-04 14:45:00 | 1112.00 | 1121.75 | 1120.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-05 09:15:00 | 1123.05 | 1121.12 | 1120.08 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-05 10:45:00 | 1126.20 | 1122.88 | 1120.97 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-01-10 12:15:00 | 1134.95 | 1141.37 | 1141.81 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 59 — SELL (started 2024-01-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-10 12:15:00 | 1134.95 | 1141.37 | 1141.81 | EMA200 below EMA400 |

### Cycle 60 — BUY (started 2024-01-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-10 14:15:00 | 1145.50 | 1142.70 | 1142.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-10 15:15:00 | 1147.65 | 1143.69 | 1142.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-11 13:15:00 | 1152.65 | 1153.33 | 1148.83 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-11 13:15:00 | 1152.65 | 1153.33 | 1148.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-11 13:15:00 | 1152.65 | 1153.33 | 1148.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-11 13:30:00 | 1148.50 | 1153.33 | 1148.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-12 09:15:00 | 1157.20 | 1154.65 | 1150.61 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-12 11:15:00 | 1167.50 | 1156.30 | 1151.73 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-12 14:30:00 | 1166.15 | 1162.06 | 1156.42 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-15 09:15:00 | 1170.25 | 1161.66 | 1156.75 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-15 10:00:00 | 1167.90 | 1162.90 | 1157.76 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-16 11:15:00 | 1167.80 | 1178.73 | 1171.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-16 11:45:00 | 1167.90 | 1178.73 | 1171.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-16 12:15:00 | 1159.80 | 1174.95 | 1170.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-16 13:00:00 | 1159.80 | 1174.95 | 1170.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-01-16 15:15:00 | 1160.00 | 1167.62 | 1168.16 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 61 — SELL (started 2024-01-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-16 15:15:00 | 1160.00 | 1167.62 | 1168.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-17 09:15:00 | 1148.70 | 1163.84 | 1166.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-19 09:15:00 | 1146.20 | 1136.71 | 1144.02 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-19 09:15:00 | 1146.20 | 1136.71 | 1144.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-19 09:15:00 | 1146.20 | 1136.71 | 1144.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-19 10:00:00 | 1146.20 | 1136.71 | 1144.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-19 10:15:00 | 1138.00 | 1136.97 | 1143.48 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-20 09:15:00 | 1136.30 | 1138.92 | 1142.17 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-20 11:30:00 | 1135.00 | 1138.34 | 1141.13 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-01-24 09:15:00 | 1195.00 | 1138.71 | 1135.97 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 62 — BUY (started 2024-01-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-24 09:15:00 | 1195.00 | 1138.71 | 1135.97 | EMA200 above EMA400 |

### Cycle 63 — SELL (started 2024-01-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-29 09:15:00 | 1139.25 | 1152.48 | 1152.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-29 15:15:00 | 1138.00 | 1146.08 | 1148.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-31 09:15:00 | 1140.00 | 1130.50 | 1138.30 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-31 09:15:00 | 1140.00 | 1130.50 | 1138.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-31 09:15:00 | 1140.00 | 1130.50 | 1138.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-31 10:00:00 | 1140.00 | 1130.50 | 1138.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-31 10:15:00 | 1138.75 | 1132.15 | 1138.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-31 10:45:00 | 1143.05 | 1132.15 | 1138.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-31 11:15:00 | 1136.45 | 1133.01 | 1138.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-31 11:30:00 | 1140.00 | 1133.01 | 1138.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-31 12:15:00 | 1134.05 | 1133.22 | 1137.79 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-31 13:45:00 | 1128.00 | 1132.17 | 1136.90 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-01 10:30:00 | 1130.50 | 1129.79 | 1134.05 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-01 13:30:00 | 1131.25 | 1130.92 | 1133.56 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-01 15:00:00 | 1131.00 | 1130.94 | 1133.33 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-01 15:15:00 | 1134.00 | 1131.55 | 1133.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-02 09:15:00 | 1133.75 | 1131.55 | 1133.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-02 09:15:00 | 1134.95 | 1132.23 | 1133.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-02 10:00:00 | 1134.95 | 1132.23 | 1133.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-02 10:15:00 | 1126.55 | 1131.09 | 1132.90 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-02 13:15:00 | 1121.30 | 1129.63 | 1131.91 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-02 14:00:00 | 1116.20 | 1126.94 | 1130.48 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-06 14:45:00 | 1118.45 | 1112.44 | 1113.51 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-06 15:15:00 | 1120.50 | 1112.44 | 1113.51 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-02-07 09:15:00 | 1169.55 | 1125.15 | 1119.18 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 64 — BUY (started 2024-02-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-07 09:15:00 | 1169.55 | 1125.15 | 1119.18 | EMA200 above EMA400 |

### Cycle 65 — SELL (started 2024-02-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-09 11:15:00 | 1120.00 | 1132.53 | 1132.84 | EMA200 below EMA400 |

### Cycle 66 — BUY (started 2024-02-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-09 13:15:00 | 1139.00 | 1133.30 | 1133.11 | EMA200 above EMA400 |

### Cycle 67 — SELL (started 2024-02-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-09 14:15:00 | 1127.55 | 1132.15 | 1132.60 | EMA200 below EMA400 |

### Cycle 68 — BUY (started 2024-02-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-12 09:15:00 | 1144.00 | 1134.02 | 1133.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-12 10:15:00 | 1151.20 | 1137.45 | 1134.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-13 09:15:00 | 1138.90 | 1145.98 | 1141.49 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-13 09:15:00 | 1138.90 | 1145.98 | 1141.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-13 09:15:00 | 1138.90 | 1145.98 | 1141.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-13 09:45:00 | 1137.85 | 1145.98 | 1141.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-13 10:15:00 | 1137.45 | 1144.28 | 1141.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-13 10:30:00 | 1138.70 | 1144.28 | 1141.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-13 14:15:00 | 1138.95 | 1141.09 | 1140.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-13 15:15:00 | 1132.00 | 1141.09 | 1140.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 69 — SELL (started 2024-02-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-13 15:15:00 | 1132.00 | 1139.27 | 1139.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-14 09:15:00 | 1120.85 | 1135.58 | 1137.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-15 11:15:00 | 1121.15 | 1119.34 | 1125.39 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-02-15 11:30:00 | 1120.90 | 1119.34 | 1125.39 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-15 12:15:00 | 1118.55 | 1119.18 | 1124.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-15 12:45:00 | 1126.90 | 1119.18 | 1124.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-16 10:15:00 | 1116.00 | 1115.39 | 1120.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-16 10:30:00 | 1117.00 | 1115.39 | 1120.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-16 12:15:00 | 1121.95 | 1116.90 | 1120.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-16 12:30:00 | 1123.15 | 1116.90 | 1120.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-16 13:15:00 | 1119.70 | 1117.46 | 1120.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-16 13:45:00 | 1124.25 | 1117.46 | 1120.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-16 14:15:00 | 1117.90 | 1117.55 | 1119.95 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-16 15:15:00 | 1108.30 | 1117.55 | 1119.95 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-02-19 10:15:00 | 1122.35 | 1115.65 | 1118.23 | SL hit (close>static) qty=1.00 sl=1121.00 alert=retest2 |

### Cycle 70 — BUY (started 2024-03-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-01 09:15:00 | 1079.90 | 1058.74 | 1057.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-02 12:15:00 | 1100.00 | 1081.27 | 1071.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-04 09:15:00 | 1060.80 | 1077.17 | 1070.70 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-04 09:15:00 | 1060.80 | 1077.17 | 1070.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-04 09:15:00 | 1060.80 | 1077.17 | 1070.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-04 09:45:00 | 1062.10 | 1077.17 | 1070.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-04 10:15:00 | 1063.25 | 1074.39 | 1070.02 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-04 11:45:00 | 1071.90 | 1073.95 | 1070.22 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-03-06 11:15:00 | 1056.40 | 1074.27 | 1075.35 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 71 — SELL (started 2024-03-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-06 11:15:00 | 1056.40 | 1074.27 | 1075.35 | EMA200 below EMA400 |

### Cycle 72 — BUY (started 2024-03-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-12 13:15:00 | 1074.85 | 1069.02 | 1068.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-12 14:15:00 | 1080.70 | 1071.36 | 1069.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-12 15:15:00 | 1069.00 | 1070.89 | 1069.48 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-12 15:15:00 | 1069.00 | 1070.89 | 1069.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-12 15:15:00 | 1069.00 | 1070.89 | 1069.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-13 09:15:00 | 1059.70 | 1070.89 | 1069.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 73 — SELL (started 2024-03-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-13 09:15:00 | 1056.25 | 1067.96 | 1068.28 | EMA200 below EMA400 |

### Cycle 74 — BUY (started 2024-03-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-14 10:15:00 | 1087.75 | 1070.01 | 1067.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-15 12:15:00 | 1095.70 | 1078.58 | 1073.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-18 12:15:00 | 1089.90 | 1092.61 | 1084.78 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-03-18 12:45:00 | 1089.95 | 1092.61 | 1084.78 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-19 09:15:00 | 1087.20 | 1090.35 | 1086.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-19 10:00:00 | 1087.20 | 1090.35 | 1086.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-19 10:15:00 | 1088.00 | 1089.88 | 1086.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-19 10:30:00 | 1086.20 | 1089.88 | 1086.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-19 11:15:00 | 1088.75 | 1089.66 | 1086.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-19 11:45:00 | 1087.35 | 1089.66 | 1086.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-19 12:15:00 | 1097.75 | 1091.28 | 1087.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-19 12:30:00 | 1086.00 | 1091.28 | 1087.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-27 15:15:00 | 1239.00 | 1245.05 | 1230.10 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-28 09:15:00 | 1272.65 | 1245.05 | 1230.10 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-04-04 13:15:00 | 1296.95 | 1309.96 | 1310.92 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 75 — SELL (started 2024-04-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-04 13:15:00 | 1296.95 | 1309.96 | 1310.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-04 14:15:00 | 1285.65 | 1305.10 | 1308.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-05 09:15:00 | 1328.75 | 1308.21 | 1309.33 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-05 09:15:00 | 1328.75 | 1308.21 | 1309.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-05 09:15:00 | 1328.75 | 1308.21 | 1309.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-05 09:45:00 | 1342.00 | 1308.21 | 1309.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-05 10:15:00 | 1309.00 | 1308.37 | 1309.30 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-05 11:15:00 | 1295.85 | 1308.37 | 1309.30 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-04-09 15:15:00 | 1306.00 | 1287.25 | 1284.80 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 76 — BUY (started 2024-04-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-09 15:15:00 | 1306.00 | 1287.25 | 1284.80 | EMA200 above EMA400 |

### Cycle 77 — SELL (started 2024-04-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-10 14:15:00 | 1278.00 | 1283.95 | 1284.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-12 11:15:00 | 1265.25 | 1277.84 | 1281.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-15 09:15:00 | 1264.95 | 1264.65 | 1272.49 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-04-15 10:00:00 | 1264.95 | 1264.65 | 1272.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-15 12:15:00 | 1275.05 | 1266.38 | 1271.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-15 13:00:00 | 1275.05 | 1266.38 | 1271.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-15 13:15:00 | 1263.75 | 1265.85 | 1270.66 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-15 14:15:00 | 1259.00 | 1265.85 | 1270.66 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-04-19 15:15:00 | 1255.20 | 1247.11 | 1246.74 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 78 — BUY (started 2024-04-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-19 15:15:00 | 1255.20 | 1247.11 | 1246.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-22 09:15:00 | 1279.50 | 1253.59 | 1249.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-23 15:15:00 | 1290.00 | 1293.42 | 1280.79 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-24 10:00:00 | 1316.40 | 1298.02 | 1284.02 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-25 09:15:00 | 1382.22 | 1338.21 | 1314.21 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Target hit | 2024-04-26 09:15:00 | 1448.04 | 1394.76 | 1359.16 | Target hit (10%) qty=0.50 alert=retest1 |

### Cycle 79 — SELL (started 2024-05-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-06 09:15:00 | 1443.60 | 1450.69 | 1451.47 | EMA200 below EMA400 |

### Cycle 80 — BUY (started 2024-05-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-06 10:15:00 | 1482.75 | 1457.10 | 1454.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-06 11:15:00 | 1535.00 | 1472.68 | 1461.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-07 09:15:00 | 1500.00 | 1500.49 | 1482.25 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-07 09:30:00 | 1500.65 | 1500.49 | 1482.25 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-07 12:15:00 | 1488.00 | 1497.08 | 1485.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-07 12:30:00 | 1480.15 | 1497.08 | 1485.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-08 09:15:00 | 1473.70 | 1490.79 | 1485.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-08 09:45:00 | 1473.25 | 1490.79 | 1485.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-08 10:15:00 | 1490.00 | 1490.63 | 1486.31 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-08 12:30:00 | 1503.05 | 1493.30 | 1488.29 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-09 11:15:00 | 1478.50 | 1487.52 | 1487.95 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 81 — SELL (started 2024-05-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-09 11:15:00 | 1478.50 | 1487.52 | 1487.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-09 15:15:00 | 1460.10 | 1475.64 | 1481.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-10 09:15:00 | 1476.55 | 1475.82 | 1481.17 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-10 10:00:00 | 1476.55 | 1475.82 | 1481.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-10 11:15:00 | 1480.45 | 1476.31 | 1480.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-10 11:45:00 | 1480.75 | 1476.31 | 1480.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-10 12:15:00 | 1473.70 | 1475.79 | 1479.83 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-10 13:30:00 | 1470.00 | 1474.03 | 1478.66 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-13 09:15:00 | 1486.25 | 1473.55 | 1477.01 | SL hit (close>static) qty=1.00 sl=1482.95 alert=retest2 |

### Cycle 82 — BUY (started 2024-05-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-13 12:15:00 | 1495.05 | 1480.07 | 1479.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-13 13:15:00 | 1506.30 | 1485.31 | 1481.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-14 09:15:00 | 1482.00 | 1489.65 | 1485.11 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-14 09:15:00 | 1482.00 | 1489.65 | 1485.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-14 09:15:00 | 1482.00 | 1489.65 | 1485.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-14 09:45:00 | 1484.50 | 1489.65 | 1485.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-14 10:15:00 | 1483.95 | 1488.51 | 1485.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-14 10:45:00 | 1479.30 | 1488.51 | 1485.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-14 12:15:00 | 1477.10 | 1484.77 | 1483.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-14 12:45:00 | 1478.15 | 1484.77 | 1483.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-14 14:15:00 | 1485.00 | 1484.37 | 1483.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-14 14:30:00 | 1483.00 | 1484.37 | 1483.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-21 15:15:00 | 1689.00 | 1714.11 | 1688.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-22 09:15:00 | 1664.00 | 1714.11 | 1688.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-22 09:15:00 | 1650.00 | 1701.29 | 1684.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-22 09:45:00 | 1661.00 | 1701.29 | 1684.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-22 10:15:00 | 1641.45 | 1689.32 | 1680.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-22 11:00:00 | 1641.45 | 1689.32 | 1680.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 83 — SELL (started 2024-05-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-22 12:15:00 | 1633.55 | 1669.06 | 1672.43 | EMA200 below EMA400 |

### Cycle 84 — BUY (started 2024-05-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-22 13:15:00 | 1713.60 | 1677.97 | 1676.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-22 14:15:00 | 1734.85 | 1689.34 | 1681.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-23 10:15:00 | 1691.05 | 1702.74 | 1691.04 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-23 10:15:00 | 1691.05 | 1702.74 | 1691.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-23 10:15:00 | 1691.05 | 1702.74 | 1691.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-23 10:45:00 | 1696.20 | 1702.74 | 1691.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-23 11:15:00 | 1705.00 | 1703.19 | 1692.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-23 11:45:00 | 1699.00 | 1703.19 | 1692.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-23 13:15:00 | 1700.00 | 1701.07 | 1693.14 | EMA400 retest candle locked (from upside) |

### Cycle 85 — SELL (started 2024-05-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-24 13:15:00 | 1668.15 | 1689.46 | 1691.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-24 14:15:00 | 1654.95 | 1682.56 | 1687.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-30 09:15:00 | 1619.15 | 1605.89 | 1617.38 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-30 09:15:00 | 1619.15 | 1605.89 | 1617.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-30 09:15:00 | 1619.15 | 1605.89 | 1617.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-30 09:45:00 | 1642.95 | 1605.89 | 1617.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-30 10:15:00 | 1624.90 | 1609.69 | 1618.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-30 10:30:00 | 1633.75 | 1609.69 | 1618.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-30 11:15:00 | 1608.90 | 1609.53 | 1617.23 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-30 13:00:00 | 1599.15 | 1607.45 | 1615.59 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-31 11:00:00 | 1598.85 | 1602.05 | 1609.29 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-31 14:30:00 | 1600.20 | 1601.37 | 1606.62 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-31 15:00:00 | 1600.20 | 1601.37 | 1606.62 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-03 09:15:00 | 1660.15 | 1612.91 | 1610.94 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 86 — BUY (started 2024-06-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-03 09:15:00 | 1660.15 | 1612.91 | 1610.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-03 15:15:00 | 1670.00 | 1650.34 | 1633.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-04 09:15:00 | 1639.70 | 1648.21 | 1634.18 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-04 09:15:00 | 1639.70 | 1648.21 | 1634.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 09:15:00 | 1639.70 | 1648.21 | 1634.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-04 09:30:00 | 1616.50 | 1648.21 | 1634.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 10:15:00 | 1593.25 | 1637.22 | 1630.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-04 11:00:00 | 1593.25 | 1637.22 | 1630.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 87 — SELL (started 2024-06-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-04 11:15:00 | 1544.40 | 1618.66 | 1622.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-04 14:15:00 | 1516.95 | 1582.24 | 1603.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-05 12:15:00 | 1562.10 | 1549.73 | 1575.98 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-06-05 12:45:00 | 1556.20 | 1549.73 | 1575.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 14:15:00 | 1559.05 | 1551.90 | 1572.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-05 14:45:00 | 1559.50 | 1551.90 | 1572.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 15:15:00 | 1567.55 | 1555.03 | 1571.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-06 09:15:00 | 1575.00 | 1555.03 | 1571.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 09:15:00 | 1583.00 | 1560.63 | 1572.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-06 10:00:00 | 1583.00 | 1560.63 | 1572.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 10:15:00 | 1583.05 | 1565.11 | 1573.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-06 10:45:00 | 1586.55 | 1565.11 | 1573.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 11:15:00 | 1575.45 | 1567.18 | 1574.04 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-06 13:00:00 | 1572.85 | 1568.31 | 1573.93 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-06 13:45:00 | 1573.00 | 1569.33 | 1573.88 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-07 09:15:00 | 1571.00 | 1572.03 | 1574.38 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-07 09:15:00 | 1590.30 | 1575.68 | 1575.83 | SL hit (close>static) qty=1.00 sl=1584.45 alert=retest2 |

### Cycle 88 — BUY (started 2024-06-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-07 13:15:00 | 1588.55 | 1576.95 | 1576.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-07 14:15:00 | 1624.45 | 1586.45 | 1580.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-11 15:15:00 | 1639.00 | 1642.14 | 1625.86 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-12 09:15:00 | 1708.95 | 1642.14 | 1625.86 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-18 09:15:00 | 1794.40 | 1757.34 | 1733.28 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Stop hit — per-position SL triggered | 2024-06-18 10:15:00 | 1750.05 | 1755.88 | 1734.80 | SL hit (close<ema200) qty=0.50 sl=1755.88 alert=retest1 |

### Cycle 89 — SELL (started 2024-06-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-21 09:15:00 | 1724.65 | 1763.96 | 1767.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-21 11:15:00 | 1718.40 | 1748.18 | 1759.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-25 15:15:00 | 1729.75 | 1698.75 | 1708.65 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-25 15:15:00 | 1729.75 | 1698.75 | 1708.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-25 15:15:00 | 1729.75 | 1698.75 | 1708.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-26 09:15:00 | 1685.05 | 1698.75 | 1708.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-26 09:15:00 | 1699.65 | 1698.93 | 1707.84 | EMA400 retest candle locked (from downside) |

### Cycle 90 — BUY (started 2024-07-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-03 09:15:00 | 1737.00 | 1697.91 | 1694.54 | EMA200 above EMA400 |

### Cycle 91 — SELL (started 2024-07-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-05 11:15:00 | 1688.90 | 1701.04 | 1702.17 | EMA200 below EMA400 |

### Cycle 92 — BUY (started 2024-07-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-08 09:15:00 | 1758.65 | 1709.43 | 1705.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-08 11:15:00 | 1776.75 | 1732.22 | 1716.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-08 13:15:00 | 1710.00 | 1728.96 | 1718.03 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-08 13:15:00 | 1710.00 | 1728.96 | 1718.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-08 13:15:00 | 1710.00 | 1728.96 | 1718.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-08 14:00:00 | 1710.00 | 1728.96 | 1718.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-08 14:15:00 | 1701.90 | 1723.55 | 1716.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-08 15:00:00 | 1701.90 | 1723.55 | 1716.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-09 10:15:00 | 1706.35 | 1718.67 | 1715.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-09 11:00:00 | 1706.35 | 1718.67 | 1715.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-09 11:15:00 | 1700.35 | 1715.01 | 1714.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-09 11:30:00 | 1698.50 | 1715.01 | 1714.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 93 — SELL (started 2024-07-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-09 12:15:00 | 1700.40 | 1712.09 | 1713.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-09 13:15:00 | 1694.30 | 1708.53 | 1711.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-11 10:15:00 | 1693.10 | 1682.08 | 1691.12 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-11 10:15:00 | 1693.10 | 1682.08 | 1691.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-11 10:15:00 | 1693.10 | 1682.08 | 1691.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-11 10:45:00 | 1687.90 | 1682.08 | 1691.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-11 11:15:00 | 1699.35 | 1685.54 | 1691.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-11 12:00:00 | 1699.35 | 1685.54 | 1691.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-11 12:15:00 | 1692.40 | 1686.91 | 1691.91 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-11 14:00:00 | 1682.95 | 1686.12 | 1691.10 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-11 15:00:00 | 1685.15 | 1685.92 | 1690.56 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-12 10:15:00 | 1702.20 | 1691.00 | 1691.86 | SL hit (close>static) qty=1.00 sl=1700.95 alert=retest2 |

### Cycle 94 — BUY (started 2024-07-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-12 12:15:00 | 1700.00 | 1693.84 | 1693.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-15 10:15:00 | 1740.95 | 1703.28 | 1697.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-16 15:15:00 | 1726.50 | 1737.89 | 1725.99 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-18 09:15:00 | 1715.00 | 1733.31 | 1724.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-18 09:15:00 | 1715.00 | 1733.31 | 1724.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-18 09:30:00 | 1715.30 | 1733.31 | 1724.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-18 10:15:00 | 1706.60 | 1727.97 | 1723.32 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-18 11:15:00 | 1735.50 | 1727.97 | 1723.32 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-19 12:00:00 | 1720.00 | 1724.68 | 1724.16 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-19 12:15:00 | 1720.00 | 1723.75 | 1723.79 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 95 — SELL (started 2024-07-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-19 12:15:00 | 1720.00 | 1723.75 | 1723.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-22 09:15:00 | 1691.00 | 1715.60 | 1719.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-23 10:15:00 | 1694.15 | 1694.06 | 1703.65 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-07-23 10:30:00 | 1699.35 | 1694.06 | 1703.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-24 09:15:00 | 1701.00 | 1683.37 | 1692.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-24 10:00:00 | 1701.00 | 1683.37 | 1692.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-24 10:15:00 | 1702.95 | 1687.29 | 1693.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-24 11:00:00 | 1702.95 | 1687.29 | 1693.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-24 11:15:00 | 1706.25 | 1691.08 | 1694.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-24 12:00:00 | 1706.25 | 1691.08 | 1694.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-24 12:15:00 | 1706.25 | 1694.11 | 1695.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-24 12:30:00 | 1706.00 | 1694.11 | 1695.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 96 — BUY (started 2024-07-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-24 13:15:00 | 1717.25 | 1698.74 | 1697.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-25 14:15:00 | 1727.10 | 1713.11 | 1707.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-25 15:15:00 | 1709.75 | 1712.44 | 1707.30 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-25 15:15:00 | 1709.75 | 1712.44 | 1707.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-25 15:15:00 | 1709.75 | 1712.44 | 1707.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-26 09:15:00 | 1702.00 | 1712.44 | 1707.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-26 09:15:00 | 1701.45 | 1710.24 | 1706.77 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-26 11:00:00 | 1725.10 | 1713.21 | 1708.43 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-30 15:15:00 | 1719.50 | 1733.25 | 1734.37 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 97 — SELL (started 2024-07-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-30 15:15:00 | 1719.50 | 1733.25 | 1734.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-01 12:15:00 | 1709.80 | 1722.55 | 1727.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-06 09:15:00 | 1627.10 | 1619.68 | 1650.07 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-06 14:15:00 | 1631.05 | 1624.86 | 1641.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-06 14:15:00 | 1631.05 | 1624.86 | 1641.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-06 15:15:00 | 1620.10 | 1624.86 | 1641.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-06 15:15:00 | 1620.10 | 1623.91 | 1639.30 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-07 10:30:00 | 1603.35 | 1619.09 | 1634.46 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-07 13:30:00 | 1608.80 | 1615.10 | 1628.53 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-07 14:45:00 | 1604.25 | 1612.06 | 1625.92 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-09 09:15:00 | 1528.36 | 1564.90 | 1588.11 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-09 14:15:00 | 1523.18 | 1539.62 | 1565.13 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-09 14:15:00 | 1524.04 | 1539.62 | 1565.13 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-08-12 13:15:00 | 1534.45 | 1531.36 | 1548.93 | SL hit (close>ema200) qty=0.50 sl=1531.36 alert=retest2 |

### Cycle 98 — BUY (started 2024-08-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-14 14:15:00 | 1557.60 | 1550.48 | 1550.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-16 09:15:00 | 1607.80 | 1561.70 | 1555.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-19 09:15:00 | 1569.90 | 1577.97 | 1568.94 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-19 09:15:00 | 1569.90 | 1577.97 | 1568.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-19 09:15:00 | 1569.90 | 1577.97 | 1568.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-19 10:00:00 | 1569.90 | 1577.97 | 1568.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-19 10:15:00 | 1550.65 | 1572.51 | 1567.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-19 10:30:00 | 1540.05 | 1572.51 | 1567.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-19 11:15:00 | 1554.15 | 1568.84 | 1566.09 | EMA400 retest candle locked (from upside) |

### Cycle 99 — SELL (started 2024-08-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-19 13:15:00 | 1550.35 | 1562.60 | 1563.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-20 13:15:00 | 1545.00 | 1555.52 | 1559.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-21 09:15:00 | 1573.05 | 1553.60 | 1557.26 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-21 09:15:00 | 1573.05 | 1553.60 | 1557.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-21 09:15:00 | 1573.05 | 1553.60 | 1557.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-21 09:30:00 | 1571.80 | 1553.60 | 1557.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-21 10:15:00 | 1569.95 | 1556.87 | 1558.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-21 11:15:00 | 1578.20 | 1556.87 | 1558.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 100 — BUY (started 2024-08-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-21 11:15:00 | 1574.00 | 1560.29 | 1559.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-21 13:15:00 | 1587.80 | 1568.84 | 1563.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-22 12:15:00 | 1570.05 | 1579.07 | 1572.36 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-22 12:15:00 | 1570.05 | 1579.07 | 1572.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-22 12:15:00 | 1570.05 | 1579.07 | 1572.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-22 13:00:00 | 1570.05 | 1579.07 | 1572.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-22 13:15:00 | 1571.80 | 1577.62 | 1572.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-22 13:45:00 | 1560.00 | 1577.62 | 1572.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-22 14:15:00 | 1576.95 | 1577.48 | 1572.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-22 14:30:00 | 1570.00 | 1577.48 | 1572.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-22 15:15:00 | 1578.00 | 1577.59 | 1573.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-23 09:15:00 | 1577.00 | 1577.59 | 1573.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 09:15:00 | 1573.15 | 1576.70 | 1573.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-23 09:30:00 | 1570.15 | 1576.70 | 1573.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 10:15:00 | 1568.25 | 1575.01 | 1572.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-23 11:15:00 | 1565.30 | 1575.01 | 1572.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 11:15:00 | 1564.00 | 1572.81 | 1571.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-23 11:45:00 | 1565.55 | 1572.81 | 1571.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 14:15:00 | 1575.60 | 1576.81 | 1574.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-23 15:00:00 | 1575.60 | 1576.81 | 1574.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 15:15:00 | 1556.80 | 1572.80 | 1572.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-26 09:15:00 | 1550.70 | 1572.80 | 1572.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 101 — SELL (started 2024-08-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-26 09:15:00 | 1549.00 | 1568.04 | 1570.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-26 12:15:00 | 1533.50 | 1554.35 | 1563.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-27 09:15:00 | 1555.90 | 1548.40 | 1556.83 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-27 09:15:00 | 1555.90 | 1548.40 | 1556.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-27 09:15:00 | 1555.90 | 1548.40 | 1556.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-27 10:00:00 | 1555.90 | 1548.40 | 1556.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-27 10:15:00 | 1550.00 | 1548.72 | 1556.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-27 10:30:00 | 1562.15 | 1548.72 | 1556.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-27 12:15:00 | 1549.95 | 1549.58 | 1555.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-27 12:45:00 | 1550.00 | 1549.58 | 1555.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-28 09:15:00 | 1538.85 | 1544.87 | 1551.13 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-28 11:00:00 | 1529.90 | 1541.87 | 1549.20 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-28 13:30:00 | 1524.00 | 1534.38 | 1543.68 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-05 11:15:00 | 1532.30 | 1519.33 | 1518.08 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 102 — BUY (started 2024-09-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-05 11:15:00 | 1532.30 | 1519.33 | 1518.08 | EMA200 above EMA400 |

### Cycle 103 — SELL (started 2024-09-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-06 14:15:00 | 1511.60 | 1525.41 | 1525.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-09 09:15:00 | 1481.45 | 1514.79 | 1520.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-09 14:15:00 | 1495.75 | 1494.78 | 1506.74 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-09 15:00:00 | 1495.75 | 1494.78 | 1506.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-10 09:15:00 | 1511.90 | 1497.91 | 1506.07 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-10 12:15:00 | 1496.15 | 1500.86 | 1506.12 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-10 14:15:00 | 1525.80 | 1508.64 | 1508.55 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 104 — BUY (started 2024-09-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-10 14:15:00 | 1525.80 | 1508.64 | 1508.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-10 15:15:00 | 1530.35 | 1512.98 | 1510.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-12 09:15:00 | 1517.55 | 1526.28 | 1520.50 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-12 09:15:00 | 1517.55 | 1526.28 | 1520.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-12 09:15:00 | 1517.55 | 1526.28 | 1520.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-12 10:00:00 | 1517.55 | 1526.28 | 1520.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-12 10:15:00 | 1524.00 | 1525.83 | 1520.82 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-12 11:15:00 | 1529.00 | 1525.83 | 1520.82 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-12 13:15:00 | 1526.80 | 1526.34 | 1521.96 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-13 10:15:00 | 1505.85 | 1520.02 | 1520.43 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 105 — SELL (started 2024-09-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-13 10:15:00 | 1505.85 | 1520.02 | 1520.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-13 15:15:00 | 1500.00 | 1510.40 | 1515.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-17 09:15:00 | 1525.15 | 1495.48 | 1502.39 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-17 09:15:00 | 1525.15 | 1495.48 | 1502.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-17 09:15:00 | 1525.15 | 1495.48 | 1502.39 | EMA400 retest candle locked (from downside) |

### Cycle 106 — BUY (started 2024-09-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-17 12:15:00 | 1519.40 | 1508.56 | 1507.41 | EMA200 above EMA400 |

### Cycle 107 — SELL (started 2024-09-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-18 14:15:00 | 1499.10 | 1509.06 | 1510.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-19 09:15:00 | 1488.00 | 1503.36 | 1507.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-20 09:15:00 | 1504.55 | 1494.97 | 1499.78 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-20 09:15:00 | 1504.55 | 1494.97 | 1499.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 09:15:00 | 1504.55 | 1494.97 | 1499.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-20 09:45:00 | 1508.00 | 1494.97 | 1499.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 10:15:00 | 1504.10 | 1496.80 | 1500.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-20 11:15:00 | 1504.10 | 1496.80 | 1500.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 11:15:00 | 1499.40 | 1497.32 | 1500.10 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-20 13:30:00 | 1494.40 | 1497.93 | 1499.93 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-20 15:15:00 | 1495.00 | 1498.33 | 1499.93 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-23 10:00:00 | 1495.20 | 1497.17 | 1499.10 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-23 12:15:00 | 1504.40 | 1500.68 | 1500.40 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 108 — BUY (started 2024-09-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-23 12:15:00 | 1504.40 | 1500.68 | 1500.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-24 09:15:00 | 1515.00 | 1505.02 | 1502.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-24 14:15:00 | 1507.85 | 1508.45 | 1505.50 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-24 14:15:00 | 1507.85 | 1508.45 | 1505.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-24 14:15:00 | 1507.85 | 1508.45 | 1505.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-24 15:00:00 | 1507.85 | 1508.45 | 1505.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 12:15:00 | 1510.25 | 1513.71 | 1509.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-25 12:45:00 | 1510.00 | 1513.71 | 1509.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 13:15:00 | 1509.90 | 1512.94 | 1509.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-25 13:45:00 | 1511.40 | 1512.94 | 1509.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 14:15:00 | 1510.05 | 1512.37 | 1509.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-25 15:15:00 | 1504.90 | 1512.37 | 1509.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 15:15:00 | 1504.90 | 1510.87 | 1509.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-26 09:15:00 | 1508.75 | 1510.87 | 1509.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-26 09:15:00 | 1506.45 | 1509.99 | 1508.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-26 10:00:00 | 1506.45 | 1509.99 | 1508.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-26 10:15:00 | 1515.00 | 1510.99 | 1509.54 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-26 12:15:00 | 1519.90 | 1510.79 | 1509.58 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-26 15:00:00 | 1520.00 | 1517.98 | 1513.65 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-27 10:15:00 | 1520.20 | 1518.86 | 1514.86 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-27 11:30:00 | 1522.90 | 1519.42 | 1515.80 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-27 14:15:00 | 1500.00 | 1516.66 | 1515.56 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-09-27 14:15:00 | 1500.00 | 1516.66 | 1515.56 | SL hit (close<static) qty=1.00 sl=1505.00 alert=retest2 |

### Cycle 109 — SELL (started 2024-09-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-30 09:15:00 | 1494.95 | 1511.42 | 1513.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-30 10:15:00 | 1480.50 | 1505.23 | 1510.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-01 09:15:00 | 1499.00 | 1496.43 | 1502.67 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-01 09:15:00 | 1499.00 | 1496.43 | 1502.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-01 09:15:00 | 1499.00 | 1496.43 | 1502.67 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-01 11:15:00 | 1493.45 | 1496.29 | 1502.04 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-07 09:15:00 | 1418.78 | 1452.10 | 1463.44 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-10-08 11:15:00 | 1431.45 | 1423.19 | 1438.00 | SL hit (close>ema200) qty=0.50 sl=1423.19 alert=retest2 |

### Cycle 110 — BUY (started 2024-10-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-09 13:15:00 | 1454.85 | 1439.81 | 1438.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-10 09:15:00 | 1463.55 | 1447.90 | 1442.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-11 10:15:00 | 1442.40 | 1455.83 | 1451.18 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-11 10:15:00 | 1442.40 | 1455.83 | 1451.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-11 10:15:00 | 1442.40 | 1455.83 | 1451.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-11 11:00:00 | 1442.40 | 1455.83 | 1451.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-11 11:15:00 | 1455.95 | 1455.85 | 1451.61 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-11 14:00:00 | 1465.20 | 1457.03 | 1452.84 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-14 09:45:00 | 1463.85 | 1461.03 | 1455.97 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-14 10:45:00 | 1460.40 | 1462.39 | 1457.05 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-17 11:15:00 | 1464.50 | 1485.61 | 1487.13 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 111 — SELL (started 2024-10-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-17 11:15:00 | 1464.50 | 1485.61 | 1487.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-21 11:15:00 | 1446.85 | 1464.26 | 1471.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-24 10:15:00 | 1412.70 | 1409.10 | 1421.95 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-24 13:15:00 | 1418.70 | 1410.82 | 1419.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-24 13:15:00 | 1418.70 | 1410.82 | 1419.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-24 13:45:00 | 1417.55 | 1410.82 | 1419.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-24 14:15:00 | 1422.10 | 1413.08 | 1419.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-24 14:45:00 | 1429.70 | 1413.08 | 1419.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-24 15:15:00 | 1419.95 | 1414.45 | 1419.77 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-25 09:15:00 | 1404.10 | 1414.45 | 1419.77 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-30 11:15:00 | 1397.10 | 1381.98 | 1380.15 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 112 — BUY (started 2024-10-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-30 11:15:00 | 1397.10 | 1381.98 | 1380.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-30 12:15:00 | 1400.60 | 1385.70 | 1382.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-31 10:15:00 | 1391.00 | 1393.85 | 1388.21 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-31 11:00:00 | 1391.00 | 1393.85 | 1388.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-31 11:15:00 | 1383.00 | 1391.68 | 1387.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-31 12:00:00 | 1383.00 | 1391.68 | 1387.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-31 12:15:00 | 1385.50 | 1390.45 | 1387.53 | EMA400 retest candle locked (from upside) |

### Cycle 113 — SELL (started 2024-10-31 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-31 15:15:00 | 1377.00 | 1384.86 | 1385.47 | EMA200 below EMA400 |

### Cycle 114 — BUY (started 2024-11-01 18:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-01 18:15:00 | 1394.00 | 1386.56 | 1386.13 | EMA200 above EMA400 |

### Cycle 115 — SELL (started 2024-11-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-04 09:15:00 | 1362.95 | 1381.84 | 1384.02 | EMA200 below EMA400 |

### Cycle 116 — BUY (started 2024-11-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-04 13:15:00 | 1399.80 | 1387.03 | 1385.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-04 14:15:00 | 1411.00 | 1391.82 | 1387.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-05 12:15:00 | 1392.05 | 1395.22 | 1391.48 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-05 13:00:00 | 1392.05 | 1395.22 | 1391.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-05 13:15:00 | 1395.60 | 1395.29 | 1391.86 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-06 09:15:00 | 1419.95 | 1395.32 | 1392.47 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-06 11:15:00 | 1423.85 | 1398.11 | 1394.35 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-08 11:30:00 | 1406.90 | 1424.24 | 1422.52 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-11 09:15:00 | 1410.50 | 1420.48 | 1421.20 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 117 — SELL (started 2024-11-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-11 09:15:00 | 1410.50 | 1420.48 | 1421.20 | EMA200 below EMA400 |

### Cycle 118 — BUY (started 2024-11-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-12 10:15:00 | 1426.00 | 1419.76 | 1419.59 | EMA200 above EMA400 |

### Cycle 119 — SELL (started 2024-11-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-12 14:15:00 | 1415.00 | 1419.77 | 1419.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-12 15:15:00 | 1404.00 | 1416.62 | 1418.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-13 10:15:00 | 1415.05 | 1412.94 | 1416.20 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-13 10:15:00 | 1415.05 | 1412.94 | 1416.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-13 10:15:00 | 1415.05 | 1412.94 | 1416.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-13 10:45:00 | 1415.50 | 1412.94 | 1416.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-13 11:15:00 | 1419.40 | 1414.23 | 1416.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-13 12:00:00 | 1419.40 | 1414.23 | 1416.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-13 12:15:00 | 1419.90 | 1415.37 | 1416.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-13 13:15:00 | 1419.20 | 1415.37 | 1416.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-13 13:15:00 | 1420.05 | 1416.30 | 1417.09 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-13 14:30:00 | 1411.00 | 1416.58 | 1417.15 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-13 15:15:00 | 1424.45 | 1418.16 | 1417.81 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 120 — BUY (started 2024-11-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-13 15:15:00 | 1424.45 | 1418.16 | 1417.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-14 09:15:00 | 1450.00 | 1424.52 | 1420.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-18 12:15:00 | 1464.05 | 1465.85 | 1450.53 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-18 13:00:00 | 1464.05 | 1465.85 | 1450.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-18 13:15:00 | 1449.40 | 1462.56 | 1450.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-18 13:30:00 | 1450.75 | 1462.56 | 1450.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-18 14:15:00 | 1449.55 | 1459.96 | 1450.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-18 14:30:00 | 1439.95 | 1459.96 | 1450.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-18 15:15:00 | 1451.00 | 1458.17 | 1450.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-19 09:15:00 | 1442.15 | 1458.17 | 1450.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 09:15:00 | 1429.90 | 1452.51 | 1448.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-19 10:00:00 | 1429.90 | 1452.51 | 1448.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 121 — SELL (started 2024-11-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-19 10:15:00 | 1412.70 | 1444.55 | 1445.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-19 11:15:00 | 1407.20 | 1437.08 | 1441.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-21 09:15:00 | 1427.90 | 1421.04 | 1430.55 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-21 09:15:00 | 1427.90 | 1421.04 | 1430.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-21 09:15:00 | 1427.90 | 1421.04 | 1430.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-21 09:30:00 | 1433.35 | 1421.04 | 1430.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-21 10:15:00 | 1467.90 | 1430.41 | 1433.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-21 11:00:00 | 1467.90 | 1430.41 | 1433.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-21 11:15:00 | 1433.55 | 1431.04 | 1433.91 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-21 12:15:00 | 1424.40 | 1431.04 | 1433.91 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-22 15:00:00 | 1410.30 | 1400.85 | 1412.36 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-26 09:30:00 | 1425.65 | 1409.92 | 1411.73 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-26 13:15:00 | 1417.30 | 1412.88 | 1412.71 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 122 — BUY (started 2024-11-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-26 13:15:00 | 1417.30 | 1412.88 | 1412.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-26 15:15:00 | 1422.00 | 1414.51 | 1413.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-27 09:15:00 | 1412.80 | 1414.17 | 1413.40 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-27 09:15:00 | 1412.80 | 1414.17 | 1413.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-27 09:15:00 | 1412.80 | 1414.17 | 1413.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-27 10:00:00 | 1412.80 | 1414.17 | 1413.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-27 10:15:00 | 1415.45 | 1414.42 | 1413.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-27 10:30:00 | 1411.55 | 1414.42 | 1413.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-27 11:15:00 | 1414.60 | 1414.46 | 1413.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-27 12:00:00 | 1414.60 | 1414.46 | 1413.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 13:15:00 | 1424.45 | 1426.35 | 1422.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-28 14:00:00 | 1424.45 | 1426.35 | 1422.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 14:15:00 | 1422.65 | 1425.61 | 1422.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-28 15:00:00 | 1422.65 | 1425.61 | 1422.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 15:15:00 | 1417.50 | 1423.99 | 1421.74 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-29 09:30:00 | 1427.70 | 1425.11 | 1422.45 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-03 12:15:00 | 1421.40 | 1434.48 | 1434.71 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 123 — SELL (started 2024-12-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-03 12:15:00 | 1421.40 | 1434.48 | 1434.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-03 13:15:00 | 1413.45 | 1430.27 | 1432.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-09 12:15:00 | 1375.55 | 1369.54 | 1380.24 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-09 12:45:00 | 1373.80 | 1369.54 | 1380.24 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-10 09:15:00 | 1368.05 | 1368.23 | 1376.19 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-10 10:15:00 | 1360.00 | 1368.23 | 1376.19 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-10 12:30:00 | 1365.05 | 1364.70 | 1372.50 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-10 13:00:00 | 1354.65 | 1364.70 | 1372.50 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-11 09:15:00 | 1378.45 | 1359.21 | 1366.58 | SL hit (close>static) qty=1.00 sl=1376.30 alert=retest2 |

### Cycle 124 — BUY (started 2024-12-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-11 11:15:00 | 1398.00 | 1372.28 | 1371.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-11 12:15:00 | 1407.60 | 1379.34 | 1374.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-12 09:15:00 | 1371.00 | 1379.32 | 1376.42 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-12 09:15:00 | 1371.00 | 1379.32 | 1376.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-12 09:15:00 | 1371.00 | 1379.32 | 1376.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-12 10:00:00 | 1371.00 | 1379.32 | 1376.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-12 10:15:00 | 1370.05 | 1377.47 | 1375.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-12 10:30:00 | 1369.00 | 1377.47 | 1375.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 125 — SELL (started 2024-12-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-12 11:15:00 | 1363.20 | 1374.61 | 1374.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-12 14:15:00 | 1360.10 | 1367.94 | 1371.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-18 15:15:00 | 1330.00 | 1321.86 | 1327.96 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-18 15:15:00 | 1330.00 | 1321.86 | 1327.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-18 15:15:00 | 1330.00 | 1321.86 | 1327.96 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-19 09:15:00 | 1310.90 | 1321.86 | 1327.96 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-19 09:45:00 | 1313.05 | 1321.00 | 1327.01 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-20 09:15:00 | 1355.00 | 1328.38 | 1327.47 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 126 — BUY (started 2024-12-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-20 09:15:00 | 1355.00 | 1328.38 | 1327.47 | EMA200 above EMA400 |

### Cycle 127 — SELL (started 2024-12-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-20 12:15:00 | 1309.90 | 1324.95 | 1326.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-20 13:15:00 | 1299.75 | 1319.91 | 1323.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-24 09:15:00 | 1295.70 | 1288.21 | 1300.07 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-24 09:15:00 | 1295.70 | 1288.21 | 1300.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-24 09:15:00 | 1295.70 | 1288.21 | 1300.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-24 10:00:00 | 1295.70 | 1288.21 | 1300.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-24 14:15:00 | 1293.30 | 1287.05 | 1294.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-24 15:00:00 | 1293.30 | 1287.05 | 1294.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-24 15:15:00 | 1289.00 | 1287.44 | 1294.18 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-26 09:30:00 | 1276.50 | 1284.61 | 1292.29 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-27 10:15:00 | 1279.75 | 1272.75 | 1280.37 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-30 09:30:00 | 1279.10 | 1281.39 | 1281.97 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-31 10:00:00 | 1278.40 | 1270.50 | 1275.07 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-31 10:15:00 | 1280.60 | 1272.52 | 1275.58 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-31 11:15:00 | 1275.60 | 1272.52 | 1275.58 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-31 12:30:00 | 1275.55 | 1273.62 | 1275.57 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-31 13:15:00 | 1275.00 | 1273.62 | 1275.57 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-31 14:30:00 | 1275.35 | 1275.06 | 1275.96 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-31 15:15:00 | 1285.05 | 1277.06 | 1276.79 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 128 — BUY (started 2024-12-31 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-31 15:15:00 | 1285.05 | 1277.06 | 1276.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-01 09:15:00 | 1298.55 | 1281.36 | 1278.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-02 09:15:00 | 1282.05 | 1291.09 | 1286.35 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-02 09:15:00 | 1282.05 | 1291.09 | 1286.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-02 09:15:00 | 1282.05 | 1291.09 | 1286.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-02 09:45:00 | 1282.40 | 1291.09 | 1286.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-02 10:15:00 | 1280.00 | 1288.87 | 1285.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-02 10:45:00 | 1279.05 | 1288.87 | 1285.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-03 14:15:00 | 1300.45 | 1304.04 | 1298.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-03 15:00:00 | 1300.45 | 1304.04 | 1298.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-03 15:15:00 | 1301.65 | 1303.56 | 1298.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 09:15:00 | 1290.00 | 1303.56 | 1298.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 09:15:00 | 1292.80 | 1301.41 | 1298.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 09:30:00 | 1292.55 | 1301.41 | 1298.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 129 — SELL (started 2025-01-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-06 10:15:00 | 1268.05 | 1294.74 | 1295.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-06 11:15:00 | 1265.55 | 1288.90 | 1292.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-07 09:15:00 | 1274.00 | 1268.61 | 1279.59 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-07 10:00:00 | 1274.00 | 1268.61 | 1279.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 10:15:00 | 1276.65 | 1270.22 | 1279.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-07 11:00:00 | 1276.65 | 1270.22 | 1279.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 11:15:00 | 1277.90 | 1271.76 | 1279.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-07 11:30:00 | 1278.75 | 1271.76 | 1279.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 12:15:00 | 1283.00 | 1274.01 | 1279.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-07 12:30:00 | 1282.50 | 1274.01 | 1279.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 13:15:00 | 1275.40 | 1274.28 | 1279.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-07 13:45:00 | 1275.00 | 1274.28 | 1279.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-09 09:15:00 | 1268.00 | 1255.23 | 1263.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-09 10:00:00 | 1268.00 | 1255.23 | 1263.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-09 10:15:00 | 1269.40 | 1258.07 | 1263.90 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-09 15:15:00 | 1262.00 | 1262.59 | 1264.49 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-15 10:15:00 | 1256.35 | 1237.53 | 1237.39 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 130 — BUY (started 2025-01-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-15 10:15:00 | 1256.35 | 1237.53 | 1237.39 | EMA200 above EMA400 |

### Cycle 131 — SELL (started 2025-01-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-15 14:15:00 | 1229.10 | 1237.07 | 1237.58 | EMA200 below EMA400 |

### Cycle 132 — BUY (started 2025-01-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-16 09:15:00 | 1244.25 | 1238.37 | 1238.07 | EMA200 above EMA400 |

### Cycle 133 — SELL (started 2025-01-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-17 11:15:00 | 1227.60 | 1238.11 | 1239.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-17 12:15:00 | 1221.60 | 1234.81 | 1237.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-20 10:15:00 | 1232.35 | 1226.31 | 1231.81 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-20 10:15:00 | 1232.35 | 1226.31 | 1231.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-20 10:15:00 | 1232.35 | 1226.31 | 1231.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-20 11:00:00 | 1232.35 | 1226.31 | 1231.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-20 11:15:00 | 1231.50 | 1227.35 | 1231.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-20 11:30:00 | 1233.30 | 1227.35 | 1231.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-20 12:15:00 | 1236.40 | 1229.16 | 1232.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-20 12:45:00 | 1234.55 | 1229.16 | 1232.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-20 13:15:00 | 1235.20 | 1230.37 | 1232.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-20 13:30:00 | 1236.05 | 1230.37 | 1232.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-20 14:15:00 | 1234.95 | 1231.28 | 1232.70 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-20 15:15:00 | 1221.20 | 1231.28 | 1232.70 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-21 09:45:00 | 1225.45 | 1227.23 | 1230.51 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-27 09:15:00 | 1160.14 | 1182.92 | 1194.21 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-27 09:15:00 | 1164.18 | 1182.92 | 1194.21 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-01-28 14:15:00 | 1146.10 | 1136.30 | 1153.70 | SL hit (close>ema200) qty=0.50 sl=1136.30 alert=retest2 |

### Cycle 134 — BUY (started 2025-01-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-30 09:15:00 | 1170.00 | 1156.76 | 1155.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-31 11:15:00 | 1176.75 | 1168.08 | 1163.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-01 13:15:00 | 1178.55 | 1178.56 | 1173.32 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-01 15:00:00 | 1186.40 | 1180.13 | 1174.51 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-03 09:15:00 | 1169.70 | 1178.17 | 1174.61 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-02-03 09:15:00 | 1169.70 | 1178.17 | 1174.61 | SL hit (close<ema400) qty=1.00 sl=1174.61 alert=retest1 |

### Cycle 135 — SELL (started 2025-02-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-03 11:15:00 | 1158.10 | 1169.93 | 1171.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-04 09:15:00 | 1143.60 | 1155.77 | 1163.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-06 12:15:00 | 1110.05 | 1109.85 | 1121.05 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-06 12:45:00 | 1110.00 | 1109.85 | 1121.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 14:15:00 | 1110.00 | 1109.91 | 1119.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-06 15:00:00 | 1110.00 | 1109.91 | 1119.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 09:15:00 | 1095.90 | 1107.44 | 1116.43 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-10 09:15:00 | 1086.30 | 1106.11 | 1111.73 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-11 09:15:00 | 1031.98 | 1059.31 | 1080.50 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-02-13 11:15:00 | 1032.00 | 1020.71 | 1033.30 | SL hit (close>ema200) qty=0.50 sl=1020.71 alert=retest2 |

### Cycle 136 — BUY (started 2025-03-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-04 09:15:00 | 891.00 | 846.37 | 843.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-04 10:15:00 | 898.70 | 856.84 | 848.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-07 14:15:00 | 920.80 | 921.17 | 911.19 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-07 15:00:00 | 920.80 | 921.17 | 911.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 09:15:00 | 918.50 | 921.25 | 912.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-10 09:30:00 | 919.20 | 921.25 | 912.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 15:15:00 | 960.20 | 968.72 | 960.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-13 09:15:00 | 954.20 | 968.72 | 960.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-13 09:15:00 | 965.15 | 968.00 | 960.95 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-17 11:30:00 | 976.65 | 965.72 | 962.96 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-17 12:45:00 | 972.25 | 967.18 | 963.87 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-17 13:30:00 | 972.70 | 968.06 | 964.57 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-17 14:15:00 | 972.25 | 968.06 | 964.57 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-19 11:15:00 | 977.70 | 981.46 | 977.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-19 12:00:00 | 977.70 | 981.46 | 977.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-19 12:15:00 | 971.55 | 979.48 | 976.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-19 13:00:00 | 971.55 | 979.48 | 976.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-19 13:15:00 | 969.85 | 977.55 | 975.94 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-19 14:30:00 | 976.80 | 977.24 | 975.95 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-20 10:15:00 | 977.05 | 975.41 | 975.33 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2025-03-25 09:15:00 | 1074.32 | 1023.49 | 1008.87 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 137 — SELL (started 2025-03-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-26 09:15:00 | 996.40 | 1006.77 | 1006.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-26 11:15:00 | 985.25 | 1000.58 | 1003.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-28 09:15:00 | 1024.85 | 983.85 | 987.73 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-28 09:15:00 | 1024.85 | 983.85 | 987.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-28 09:15:00 | 1024.85 | 983.85 | 987.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-28 10:00:00 | 1024.85 | 983.85 | 987.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 138 — BUY (started 2025-03-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-28 10:15:00 | 1024.10 | 991.90 | 991.04 | EMA200 above EMA400 |

### Cycle 139 — SELL (started 2025-04-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-02 09:15:00 | 990.30 | 998.25 | 998.38 | EMA200 below EMA400 |

### Cycle 140 — BUY (started 2025-04-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-02 14:15:00 | 1010.65 | 999.73 | 998.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-03 10:15:00 | 1014.55 | 1004.74 | 1001.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-04 09:15:00 | 1001.50 | 1010.88 | 1006.89 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-04 09:15:00 | 1001.50 | 1010.88 | 1006.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 09:15:00 | 1001.50 | 1010.88 | 1006.89 | EMA400 retest candle locked (from upside) |

### Cycle 141 — SELL (started 2025-04-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-04 12:15:00 | 996.45 | 1004.89 | 1004.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-07 09:15:00 | 941.10 | 987.28 | 996.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-07 15:15:00 | 969.00 | 961.41 | 976.26 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-07 15:15:00 | 969.00 | 961.41 | 976.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-07 15:15:00 | 969.00 | 961.41 | 976.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-08 09:15:00 | 964.65 | 961.41 | 976.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 09:15:00 | 959.15 | 960.96 | 974.70 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-08 10:45:00 | 952.50 | 958.38 | 972.28 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-11 11:15:00 | 949.90 | 952.12 | 956.09 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-11 11:45:00 | 949.85 | 951.46 | 955.43 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-15 09:15:00 | 954.50 | 952.02 | 954.23 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-15 09:15:00 | 962.40 | 954.09 | 954.97 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-04-15 10:15:00 | 979.30 | 959.13 | 957.18 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 142 — BUY (started 2025-04-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-15 10:15:00 | 979.30 | 959.13 | 957.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-15 11:15:00 | 985.60 | 964.43 | 959.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-17 14:15:00 | 1012.10 | 1013.62 | 1001.29 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-17 15:00:00 | 1012.10 | 1013.62 | 1001.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 10:15:00 | 1054.00 | 1054.81 | 1042.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-23 10:45:00 | 1045.90 | 1054.81 | 1042.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-24 09:15:00 | 1057.10 | 1063.63 | 1052.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-24 09:45:00 | 1054.70 | 1063.63 | 1052.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-24 10:15:00 | 1054.40 | 1061.79 | 1052.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-24 10:45:00 | 1054.80 | 1061.79 | 1052.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-24 12:15:00 | 1048.90 | 1058.12 | 1052.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-24 13:00:00 | 1048.90 | 1058.12 | 1052.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-24 13:15:00 | 1049.90 | 1056.48 | 1052.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-24 14:15:00 | 1049.10 | 1056.48 | 1052.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 143 — SELL (started 2025-04-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-25 09:15:00 | 1010.60 | 1045.10 | 1048.12 | EMA200 below EMA400 |

### Cycle 144 — BUY (started 2025-04-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-29 14:15:00 | 1038.80 | 1033.62 | 1032.94 | EMA200 above EMA400 |

### Cycle 145 — SELL (started 2025-04-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-30 15:15:00 | 1019.50 | 1031.84 | 1033.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-02 11:15:00 | 1009.70 | 1023.85 | 1028.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-05 09:15:00 | 1021.00 | 1015.95 | 1022.33 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-05 09:15:00 | 1021.00 | 1015.95 | 1022.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 09:15:00 | 1021.00 | 1015.95 | 1022.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-05 10:00:00 | 1021.00 | 1015.95 | 1022.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 10:15:00 | 1021.50 | 1017.06 | 1022.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-05 11:00:00 | 1021.50 | 1017.06 | 1022.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 11:15:00 | 1022.20 | 1018.09 | 1022.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-05 12:00:00 | 1022.20 | 1018.09 | 1022.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 12:15:00 | 1022.20 | 1018.91 | 1022.25 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-05 13:30:00 | 1008.80 | 1016.27 | 1020.74 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-07 12:15:00 | 958.36 | 977.27 | 993.04 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-05-08 09:15:00 | 1008.00 | 976.92 | 987.07 | SL hit (close>ema200) qty=0.50 sl=976.92 alert=retest2 |

### Cycle 146 — BUY (started 2025-05-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 11:15:00 | 990.80 | 979.12 | 977.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 13:15:00 | 996.00 | 984.72 | 980.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-13 09:15:00 | 984.70 | 988.97 | 984.04 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-13 09:15:00 | 984.70 | 988.97 | 984.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-13 09:15:00 | 984.70 | 988.97 | 984.04 | EMA400 retest candle locked (from upside) |

### Cycle 147 — SELL (started 2025-05-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-14 10:15:00 | 965.10 | 979.09 | 980.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-14 11:15:00 | 960.50 | 975.37 | 979.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-15 09:15:00 | 978.10 | 969.79 | 974.15 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-15 09:15:00 | 978.10 | 969.79 | 974.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-15 09:15:00 | 978.10 | 969.79 | 974.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-15 10:00:00 | 978.10 | 969.79 | 974.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-15 10:15:00 | 988.70 | 973.57 | 975.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-15 11:00:00 | 988.70 | 973.57 | 975.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-15 11:15:00 | 981.10 | 975.08 | 975.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-15 11:30:00 | 991.00 | 975.08 | 975.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 148 — BUY (started 2025-05-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-15 12:15:00 | 987.30 | 977.52 | 977.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-16 09:15:00 | 989.40 | 981.33 | 979.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-16 13:15:00 | 982.00 | 982.41 | 980.45 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-16 13:30:00 | 981.20 | 982.41 | 980.45 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 09:15:00 | 991.50 | 984.31 | 981.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-19 09:30:00 | 987.30 | 984.31 | 981.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 10:15:00 | 995.90 | 995.56 | 990.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 10:30:00 | 997.00 | 995.56 | 990.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 11:15:00 | 995.20 | 995.49 | 990.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 11:45:00 | 992.60 | 995.49 | 990.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 11:15:00 | 994.10 | 1000.09 | 996.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-21 12:00:00 | 994.10 | 1000.09 | 996.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 12:15:00 | 998.50 | 999.77 | 996.49 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-21 13:15:00 | 1005.40 | 999.77 | 996.49 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-22 09:30:00 | 1001.20 | 1003.46 | 999.63 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-23 12:00:00 | 1003.20 | 1000.31 | 999.64 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-23 12:45:00 | 1002.00 | 1000.49 | 999.78 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 15:15:00 | 1000.00 | 1000.71 | 1000.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-26 09:15:00 | 1000.10 | 1000.71 | 1000.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 09:15:00 | 999.50 | 1000.47 | 1000.03 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-05-26 11:15:00 | 997.10 | 999.58 | 999.69 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 149 — SELL (started 2025-05-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-26 11:15:00 | 997.10 | 999.58 | 999.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-26 12:15:00 | 994.50 | 998.56 | 999.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-26 14:15:00 | 1002.30 | 999.00 | 999.28 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-26 14:15:00 | 1002.30 | 999.00 | 999.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 14:15:00 | 1002.30 | 999.00 | 999.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-26 15:00:00 | 1002.30 | 999.00 | 999.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 15:15:00 | 999.50 | 999.10 | 999.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-27 09:15:00 | 1000.00 | 999.10 | 999.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 09:15:00 | 999.00 | 999.08 | 999.27 | EMA400 retest candle locked (from downside) |

### Cycle 150 — BUY (started 2025-05-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-27 10:15:00 | 1001.20 | 999.50 | 999.45 | EMA200 above EMA400 |

### Cycle 151 — SELL (started 2025-05-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-27 13:15:00 | 997.70 | 999.24 | 999.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-27 15:15:00 | 993.10 | 997.99 | 998.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-28 11:15:00 | 1000.20 | 997.85 | 998.46 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-28 11:15:00 | 1000.20 | 997.85 | 998.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 11:15:00 | 1000.20 | 997.85 | 998.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-28 12:00:00 | 1000.20 | 997.85 | 998.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 12:15:00 | 995.50 | 997.38 | 998.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-28 12:30:00 | 999.30 | 997.38 | 998.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 09:15:00 | 994.20 | 995.18 | 996.78 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-29 10:15:00 | 988.70 | 995.18 | 996.78 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-30 09:30:00 | 991.30 | 988.97 | 992.01 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-02 10:00:00 | 989.00 | 988.16 | 989.66 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-03 10:30:00 | 990.60 | 983.40 | 985.75 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 11:15:00 | 986.40 | 984.00 | 985.81 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-03 12:45:00 | 981.60 | 983.39 | 985.36 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-12 11:15:00 | 939.26 | 948.54 | 951.31 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-12 11:15:00 | 941.73 | 948.54 | 951.31 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-12 11:15:00 | 939.55 | 948.54 | 951.31 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-12 11:15:00 | 941.07 | 948.54 | 951.31 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-12 13:15:00 | 932.52 | 943.16 | 948.30 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-13 11:15:00 | 943.40 | 938.63 | 943.51 | SL hit (close>ema200) qty=0.50 sl=938.63 alert=retest2 |

### Cycle 152 — BUY (started 2025-06-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-17 11:15:00 | 950.55 | 943.04 | 942.72 | EMA200 above EMA400 |

### Cycle 153 — SELL (started 2025-06-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-17 14:15:00 | 937.15 | 942.36 | 942.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-18 10:15:00 | 930.00 | 938.85 | 940.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-18 13:15:00 | 936.25 | 935.42 | 938.49 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-18 14:00:00 | 936.25 | 935.42 | 938.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 14:15:00 | 938.05 | 935.94 | 938.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-18 14:30:00 | 936.25 | 935.94 | 938.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 09:15:00 | 931.35 | 934.85 | 937.51 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-19 10:30:00 | 928.60 | 933.30 | 936.56 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-19 12:45:00 | 924.95 | 930.08 | 934.43 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-20 14:15:00 | 939.35 | 929.63 | 931.07 | SL hit (close>static) qty=1.00 sl=938.35 alert=retest2 |

### Cycle 154 — BUY (started 2025-06-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-24 10:15:00 | 942.10 | 932.01 | 930.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-24 11:15:00 | 954.35 | 936.48 | 933.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-26 09:15:00 | 967.05 | 968.35 | 957.24 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-26 10:00:00 | 967.05 | 968.35 | 957.24 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 14:15:00 | 960.70 | 964.92 | 959.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-26 14:30:00 | 960.85 | 964.92 | 959.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 15:15:00 | 964.50 | 964.84 | 960.08 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-27 10:15:00 | 976.70 | 964.97 | 960.58 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-30 09:15:00 | 973.00 | 965.14 | 962.59 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-30 12:30:00 | 968.40 | 963.77 | 962.78 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-30 14:15:00 | 969.10 | 964.57 | 963.24 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 14:15:00 | 964.50 | 964.56 | 963.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-30 15:00:00 | 964.50 | 964.56 | 963.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 15:15:00 | 974.40 | 966.53 | 964.36 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-01 09:15:00 | 982.35 | 966.53 | 964.36 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-07 10:15:00 | 983.40 | 989.91 | 990.47 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 155 — SELL (started 2025-07-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-07 10:15:00 | 983.40 | 989.91 | 990.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-07 11:15:00 | 980.45 | 988.02 | 989.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-07 15:15:00 | 985.00 | 984.91 | 987.29 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-08 09:15:00 | 995.00 | 984.91 | 987.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 09:15:00 | 987.00 | 985.33 | 987.27 | EMA400 retest candle locked (from downside) |

### Cycle 156 — BUY (started 2025-07-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-08 15:15:00 | 993.00 | 987.76 | 987.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-09 09:15:00 | 999.20 | 990.05 | 988.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-09 13:15:00 | 992.15 | 992.41 | 990.48 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-09 14:15:00 | 991.25 | 992.41 | 990.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 14:15:00 | 999.80 | 993.89 | 991.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-09 14:30:00 | 992.00 | 993.89 | 991.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 10:15:00 | 997.40 | 995.06 | 992.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-10 10:30:00 | 996.60 | 995.06 | 992.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 09:15:00 | 991.75 | 995.13 | 993.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-11 10:00:00 | 991.75 | 995.13 | 993.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 157 — SELL (started 2025-07-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-11 10:15:00 | 983.00 | 992.70 | 992.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-11 13:15:00 | 979.00 | 986.80 | 989.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-14 09:15:00 | 983.80 | 981.95 | 986.50 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-14 10:00:00 | 983.80 | 981.95 | 986.50 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 10:15:00 | 982.85 | 982.13 | 986.17 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-14 12:15:00 | 976.90 | 981.51 | 985.52 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-14 13:00:00 | 978.00 | 980.81 | 984.83 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-14 14:15:00 | 995.80 | 984.44 | 985.83 | SL hit (close>static) qty=1.00 sl=986.45 alert=retest2 |

### Cycle 158 — BUY (started 2025-07-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-15 09:15:00 | 993.05 | 987.13 | 986.87 | EMA200 above EMA400 |

### Cycle 159 — SELL (started 2025-07-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-18 13:15:00 | 989.85 | 993.49 | 993.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-18 15:15:00 | 985.60 | 991.33 | 992.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-21 09:15:00 | 994.65 | 991.99 | 992.72 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-21 09:15:00 | 994.65 | 991.99 | 992.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 09:15:00 | 994.65 | 991.99 | 992.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-21 09:45:00 | 995.10 | 991.99 | 992.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 10:15:00 | 988.35 | 991.26 | 992.32 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-22 10:45:00 | 986.40 | 989.86 | 991.19 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-22 12:00:00 | 983.75 | 988.64 | 990.51 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-28 13:15:00 | 937.08 | 950.86 | 961.62 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-28 13:15:00 | 934.56 | 950.86 | 961.62 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-30 09:15:00 | 940.00 | 933.66 | 942.91 | SL hit (close>ema200) qty=0.50 sl=933.66 alert=retest2 |

### Cycle 160 — BUY (started 2025-07-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-31 11:15:00 | 947.55 | 944.81 | 944.53 | EMA200 above EMA400 |

### Cycle 161 — SELL (started 2025-07-31 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-31 14:15:00 | 930.40 | 943.94 | 944.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-01 14:15:00 | 926.25 | 936.47 | 939.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-05 09:15:00 | 920.85 | 920.53 | 927.44 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-05 11:00:00 | 910.55 | 918.54 | 925.90 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-07 10:15:00 | 865.02 | 883.60 | 898.18 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-08-11 11:15:00 | 851.60 | 846.99 | 860.14 | SL hit (close>ema200) qty=0.50 sl=846.99 alert=retest1 |

### Cycle 162 — BUY (started 2025-08-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-18 09:15:00 | 869.90 | 853.22 | 852.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-18 10:15:00 | 880.00 | 858.57 | 854.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-19 11:15:00 | 871.20 | 871.49 | 864.87 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-19 11:30:00 | 871.95 | 871.49 | 864.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 12:15:00 | 874.95 | 872.18 | 865.79 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-19 14:30:00 | 875.50 | 871.87 | 866.75 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-20 09:15:00 | 875.55 | 871.89 | 867.23 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2025-08-20 11:15:00 | 963.05 | 905.72 | 885.31 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 163 — SELL (started 2025-08-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-26 09:15:00 | 913.50 | 927.76 | 928.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-26 14:15:00 | 904.10 | 914.94 | 921.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-28 15:15:00 | 905.00 | 904.07 | 910.91 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-29 09:15:00 | 905.75 | 904.07 | 910.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 09:15:00 | 911.20 | 905.50 | 910.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-29 10:00:00 | 911.20 | 905.50 | 910.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 10:15:00 | 919.15 | 908.23 | 911.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-29 11:00:00 | 919.15 | 908.23 | 911.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 11:15:00 | 918.60 | 910.30 | 912.31 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-29 12:15:00 | 915.00 | 910.30 | 912.31 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-01 10:15:00 | 916.00 | 913.07 | 913.03 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 164 — BUY (started 2025-09-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-01 10:15:00 | 916.00 | 913.07 | 913.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-01 11:15:00 | 922.00 | 914.85 | 913.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-03 14:15:00 | 948.70 | 949.92 | 940.96 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-03 15:00:00 | 948.70 | 949.92 | 940.96 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 15:15:00 | 945.00 | 948.94 | 941.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 09:15:00 | 947.50 | 948.94 | 941.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 09:15:00 | 945.80 | 948.31 | 941.74 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-05 09:45:00 | 957.45 | 948.48 | 944.86 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-05 13:30:00 | 953.05 | 949.48 | 946.49 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-05 14:15:00 | 956.10 | 949.48 | 946.49 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-08 09:30:00 | 962.30 | 954.86 | 949.55 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 10:15:00 | 970.60 | 966.15 | 959.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-09 10:30:00 | 959.50 | 966.15 | 959.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 15:15:00 | 988.35 | 992.71 | 987.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-15 09:15:00 | 990.00 | 992.71 | 987.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 09:15:00 | 985.80 | 991.33 | 987.61 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-09-15 12:15:00 | 979.70 | 984.83 | 985.20 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 165 — SELL (started 2025-09-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-15 12:15:00 | 979.70 | 984.83 | 985.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-16 10:15:00 | 973.95 | 980.04 | 982.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-16 13:15:00 | 980.00 | 978.91 | 981.27 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-16 14:00:00 | 980.00 | 978.91 | 981.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |

### Cycle 166 — BUY (started 2025-09-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-16 14:15:00 | 999.90 | 983.11 | 982.96 | EMA200 above EMA400 |

### Cycle 167 — SELL (started 2025-09-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-18 12:15:00 | 982.10 | 989.21 | 989.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-19 11:15:00 | 977.80 | 984.36 | 986.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-22 15:15:00 | 957.00 | 955.50 | 965.85 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-23 09:15:00 | 962.40 | 955.50 | 965.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 09:15:00 | 955.75 | 955.55 | 964.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-23 09:30:00 | 962.45 | 955.55 | 964.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 11:15:00 | 970.00 | 959.39 | 965.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-23 11:45:00 | 968.50 | 959.39 | 965.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 12:15:00 | 971.00 | 961.71 | 965.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-23 13:00:00 | 971.00 | 961.71 | 965.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 168 — BUY (started 2025-09-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-24 09:15:00 | 974.40 | 968.37 | 967.96 | EMA200 above EMA400 |

### Cycle 169 — SELL (started 2025-09-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-24 12:15:00 | 963.60 | 967.20 | 967.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-24 13:15:00 | 960.40 | 965.84 | 966.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-30 09:15:00 | 924.90 | 919.75 | 928.83 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-30 09:15:00 | 924.90 | 919.75 | 928.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 09:15:00 | 924.90 | 919.75 | 928.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-30 10:00:00 | 924.90 | 919.75 | 928.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 14:15:00 | 925.75 | 920.99 | 926.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-30 15:00:00 | 925.75 | 920.99 | 926.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 15:15:00 | 921.65 | 921.12 | 925.61 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-01 09:15:00 | 920.60 | 921.12 | 925.61 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-01 09:45:00 | 918.90 | 920.37 | 924.86 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-01 14:45:00 | 919.30 | 916.95 | 921.15 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-03 10:00:00 | 918.65 | 917.94 | 920.91 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 10:15:00 | 920.95 | 918.54 | 920.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-03 11:00:00 | 920.95 | 918.54 | 920.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 11:15:00 | 929.45 | 920.72 | 921.69 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-10-03 11:15:00 | 929.45 | 920.72 | 921.69 | SL hit (close>static) qty=1.00 sl=926.35 alert=retest2 |

### Cycle 170 — BUY (started 2025-10-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-03 12:15:00 | 930.25 | 922.63 | 922.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-03 14:15:00 | 938.60 | 927.19 | 924.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-06 10:15:00 | 931.10 | 931.30 | 927.47 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-06 11:00:00 | 931.10 | 931.30 | 927.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 11:15:00 | 928.00 | 930.64 | 927.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-06 11:45:00 | 927.25 | 930.64 | 927.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 12:15:00 | 921.20 | 928.75 | 926.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-06 13:00:00 | 921.20 | 928.75 | 926.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 13:15:00 | 921.20 | 927.24 | 926.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-06 13:45:00 | 922.00 | 927.24 | 926.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 171 — SELL (started 2025-10-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-06 14:15:00 | 919.90 | 925.78 | 925.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-07 11:15:00 | 914.90 | 921.59 | 923.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-08 09:15:00 | 928.35 | 922.17 | 923.04 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-08 09:15:00 | 928.35 | 922.17 | 923.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 09:15:00 | 928.35 | 922.17 | 923.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-08 09:45:00 | 929.85 | 922.17 | 923.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 10:15:00 | 919.25 | 921.59 | 922.69 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-08 11:15:00 | 918.25 | 921.59 | 922.69 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-08 12:45:00 | 918.70 | 920.24 | 921.86 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-10 09:45:00 | 915.45 | 914.01 | 915.85 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-10 10:15:00 | 915.80 | 914.01 | 915.85 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 10:15:00 | 921.50 | 915.51 | 916.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-10 10:30:00 | 920.25 | 915.51 | 916.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 11:15:00 | 916.40 | 915.68 | 916.36 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-10-10 13:15:00 | 926.00 | 918.16 | 917.40 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 172 — BUY (started 2025-10-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-10 13:15:00 | 926.00 | 918.16 | 917.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-10 14:15:00 | 939.30 | 922.39 | 919.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-13 09:15:00 | 923.15 | 924.64 | 921.06 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-13 09:15:00 | 923.15 | 924.64 | 921.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 09:15:00 | 923.15 | 924.64 | 921.06 | EMA400 retest candle locked (from upside) |

### Cycle 173 — SELL (started 2025-10-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-14 14:15:00 | 913.40 | 921.61 | 922.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-15 14:15:00 | 910.00 | 916.24 | 918.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-16 13:15:00 | 915.65 | 914.45 | 916.52 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-16 14:00:00 | 915.65 | 914.45 | 916.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 14:15:00 | 916.55 | 914.87 | 916.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-16 14:45:00 | 916.50 | 914.87 | 916.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 15:15:00 | 917.25 | 915.35 | 916.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-17 09:15:00 | 917.40 | 915.35 | 916.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 09:15:00 | 912.95 | 914.87 | 916.26 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-17 11:30:00 | 909.00 | 912.38 | 914.85 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-21 14:15:00 | 908.00 | 903.15 | 906.44 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-27 12:15:00 | 906.00 | 904.25 | 904.17 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 174 — BUY (started 2025-10-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-27 12:15:00 | 906.00 | 904.25 | 904.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-28 09:15:00 | 914.90 | 907.37 | 905.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-28 12:15:00 | 909.55 | 909.68 | 907.37 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-28 13:00:00 | 909.55 | 909.68 | 907.37 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 14:15:00 | 909.50 | 909.49 | 907.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-28 14:30:00 | 903.35 | 909.49 | 907.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 15:15:00 | 910.25 | 909.64 | 907.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-29 09:15:00 | 907.20 | 909.64 | 907.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 09:15:00 | 910.80 | 909.87 | 908.17 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-29 12:00:00 | 914.65 | 910.83 | 908.91 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-29 13:30:00 | 914.70 | 912.17 | 909.88 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-31 11:15:00 | 905.05 | 916.30 | 916.39 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 175 — SELL (started 2025-10-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-31 11:15:00 | 905.05 | 916.30 | 916.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-31 14:15:00 | 897.65 | 910.79 | 913.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-03 12:15:00 | 907.95 | 905.94 | 909.72 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-03 12:45:00 | 908.50 | 905.94 | 909.72 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 13:15:00 | 907.95 | 906.34 | 909.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-03 13:30:00 | 908.35 | 906.34 | 909.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 14:15:00 | 910.40 | 907.15 | 909.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-03 15:00:00 | 910.40 | 907.15 | 909.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 15:15:00 | 909.90 | 907.70 | 909.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-04 09:15:00 | 915.40 | 907.70 | 909.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 09:15:00 | 913.95 | 908.95 | 910.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-04 09:30:00 | 916.55 | 908.95 | 910.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 176 — BUY (started 2025-11-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-04 10:15:00 | 922.10 | 911.58 | 911.15 | EMA200 above EMA400 |

### Cycle 177 — SELL (started 2025-11-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-04 11:15:00 | 907.85 | 910.84 | 910.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-04 12:15:00 | 901.50 | 908.97 | 910.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-07 10:15:00 | 895.00 | 890.13 | 895.41 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-07 10:15:00 | 895.00 | 890.13 | 895.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 10:15:00 | 895.00 | 890.13 | 895.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-07 11:00:00 | 895.00 | 890.13 | 895.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 11:15:00 | 904.50 | 893.01 | 896.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-07 12:00:00 | 904.50 | 893.01 | 896.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 12:15:00 | 905.50 | 895.51 | 897.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-07 13:00:00 | 905.50 | 895.51 | 897.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 178 — BUY (started 2025-11-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-07 13:15:00 | 916.10 | 899.62 | 898.81 | EMA200 above EMA400 |

### Cycle 179 — SELL (started 2025-11-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-11 12:15:00 | 893.45 | 903.09 | 904.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-11 15:15:00 | 890.10 | 897.81 | 901.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-13 10:15:00 | 893.95 | 891.83 | 895.23 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-13 10:15:00 | 893.95 | 891.83 | 895.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 10:15:00 | 893.95 | 891.83 | 895.23 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-14 09:15:00 | 885.00 | 892.42 | 894.30 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-14 11:45:00 | 888.00 | 889.74 | 892.48 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-17 11:00:00 | 885.05 | 886.75 | 889.49 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-19 10:15:00 | 843.60 | 856.72 | 867.52 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-19 12:15:00 | 840.75 | 851.30 | 863.07 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-19 12:15:00 | 840.80 | 851.30 | 863.07 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-24 10:15:00 | 830.90 | 826.71 | 834.01 | SL hit (close>ema200) qty=0.50 sl=826.71 alert=retest2 |

### Cycle 180 — BUY (started 2025-11-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-25 15:15:00 | 835.15 | 833.19 | 833.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-26 09:15:00 | 874.50 | 841.45 | 836.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-26 15:15:00 | 874.00 | 875.55 | 859.99 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-27 09:15:00 | 873.85 | 875.55 | 859.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 11:15:00 | 868.70 | 871.60 | 861.85 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-27 13:00:00 | 870.60 | 871.40 | 862.64 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-27 14:15:00 | 870.00 | 870.77 | 863.15 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-27 15:15:00 | 869.90 | 870.02 | 863.50 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-28 09:15:00 | 855.00 | 867.00 | 863.26 | SL hit (close<static) qty=1.00 sl=861.00 alert=retest2 |

### Cycle 181 — SELL (started 2025-11-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-28 12:15:00 | 851.50 | 859.93 | 860.63 | EMA200 below EMA400 |

### Cycle 182 — BUY (started 2025-12-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-01 10:15:00 | 871.00 | 861.30 | 860.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-01 13:15:00 | 874.90 | 865.92 | 863.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-02 11:15:00 | 872.85 | 875.09 | 869.63 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-02 12:00:00 | 872.85 | 875.09 | 869.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 12:15:00 | 871.60 | 874.39 | 869.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-02 12:45:00 | 870.95 | 874.39 | 869.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 13:15:00 | 885.10 | 876.53 | 871.20 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-03 10:00:00 | 888.15 | 881.79 | 875.19 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-03 10:30:00 | 891.20 | 883.10 | 876.38 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-03 14:00:00 | 888.45 | 885.70 | 879.41 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-03 15:15:00 | 894.25 | 886.06 | 880.15 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 12:15:00 | 880.10 | 885.55 | 882.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-04 12:45:00 | 880.95 | 885.55 | 882.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 13:15:00 | 878.25 | 884.09 | 881.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-04 14:15:00 | 876.15 | 884.09 | 881.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 14:15:00 | 878.90 | 883.05 | 881.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-04 14:30:00 | 877.20 | 883.05 | 881.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 15:15:00 | 887.00 | 883.84 | 882.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-05 09:15:00 | 878.40 | 883.84 | 882.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 09:15:00 | 875.10 | 882.09 | 881.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-05 09:45:00 | 873.00 | 882.09 | 881.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 10:15:00 | 883.40 | 882.35 | 881.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-05 10:30:00 | 882.90 | 882.35 | 881.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-12-05 11:15:00 | 875.00 | 880.88 | 881.10 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 183 — SELL (started 2025-12-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-05 11:15:00 | 875.00 | 880.88 | 881.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-05 12:15:00 | 873.20 | 879.35 | 880.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-09 11:15:00 | 850.55 | 848.23 | 857.58 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-09 12:00:00 | 850.55 | 848.23 | 857.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 14:15:00 | 874.90 | 853.86 | 857.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-09 15:00:00 | 874.90 | 853.86 | 857.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 15:15:00 | 877.00 | 858.49 | 859.59 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-10 09:15:00 | 863.55 | 858.49 | 859.59 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-11 14:15:00 | 869.75 | 857.53 | 857.42 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 184 — BUY (started 2025-12-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-11 14:15:00 | 869.75 | 857.53 | 857.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-12 13:15:00 | 873.90 | 865.78 | 862.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-15 14:15:00 | 869.80 | 871.42 | 867.96 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-15 15:00:00 | 869.80 | 871.42 | 867.96 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 09:15:00 | 859.10 | 869.04 | 867.48 | EMA400 retest candle locked (from upside) |

### Cycle 185 — SELL (started 2025-12-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-16 11:15:00 | 862.20 | 866.20 | 866.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-17 09:15:00 | 858.10 | 862.63 | 864.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-18 14:15:00 | 849.55 | 846.75 | 852.30 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-18 15:00:00 | 849.55 | 846.75 | 852.30 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 15:15:00 | 852.00 | 847.80 | 852.28 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-19 09:30:00 | 848.00 | 849.18 | 852.50 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-19 13:15:00 | 859.80 | 852.22 | 852.91 | SL hit (close>static) qty=1.00 sl=856.00 alert=retest2 |

### Cycle 186 — BUY (started 2025-12-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-24 11:15:00 | 853.45 | 850.83 | 850.54 | EMA200 above EMA400 |

### Cycle 187 — SELL (started 2025-12-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-24 14:15:00 | 846.00 | 849.84 | 850.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-26 10:15:00 | 842.40 | 847.92 | 849.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-26 11:15:00 | 850.45 | 848.42 | 849.31 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-26 11:15:00 | 850.45 | 848.42 | 849.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 11:15:00 | 850.45 | 848.42 | 849.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-26 12:00:00 | 850.45 | 848.42 | 849.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 12:15:00 | 853.85 | 849.51 | 849.73 | EMA400 retest candle locked (from downside) |

### Cycle 188 — BUY (started 2025-12-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-26 13:15:00 | 851.75 | 849.96 | 849.91 | EMA200 above EMA400 |

### Cycle 189 — SELL (started 2025-12-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-26 14:15:00 | 847.75 | 849.52 | 849.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-29 10:15:00 | 837.40 | 847.06 | 848.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-31 09:15:00 | 845.60 | 834.67 | 837.92 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-31 09:15:00 | 845.60 | 834.67 | 837.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 09:15:00 | 845.60 | 834.67 | 837.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 10:00:00 | 845.60 | 834.67 | 837.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 10:15:00 | 853.00 | 838.33 | 839.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 11:00:00 | 853.00 | 838.33 | 839.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 190 — BUY (started 2025-12-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-31 11:15:00 | 852.10 | 841.09 | 840.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-31 12:15:00 | 853.20 | 843.51 | 841.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-01 10:15:00 | 844.00 | 847.80 | 845.00 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-01 10:15:00 | 844.00 | 847.80 | 845.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 10:15:00 | 844.00 | 847.80 | 845.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-01 11:15:00 | 845.10 | 847.80 | 845.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 11:15:00 | 845.00 | 847.24 | 845.00 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-01 12:30:00 | 848.95 | 849.58 | 846.27 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-06 09:45:00 | 850.30 | 858.25 | 856.80 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-06 10:15:00 | 839.00 | 854.40 | 855.18 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 191 — SELL (started 2026-01-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-06 10:15:00 | 839.00 | 854.40 | 855.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-06 13:15:00 | 837.35 | 846.52 | 851.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-12 12:15:00 | 807.20 | 804.36 | 812.30 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-12 12:30:00 | 805.85 | 804.36 | 812.30 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 09:15:00 | 808.25 | 803.81 | 809.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-13 10:15:00 | 814.05 | 803.81 | 809.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 10:15:00 | 809.50 | 804.95 | 809.36 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-14 14:30:00 | 804.80 | 808.19 | 809.01 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-16 09:15:00 | 816.90 | 809.90 | 809.64 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 192 — BUY (started 2026-01-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-16 09:15:00 | 816.90 | 809.90 | 809.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-16 12:15:00 | 820.75 | 813.54 | 811.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-16 15:15:00 | 814.25 | 814.25 | 812.37 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-16 15:15:00 | 814.25 | 814.25 | 812.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 15:15:00 | 814.25 | 814.25 | 812.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-19 09:15:00 | 803.35 | 814.25 | 812.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 09:15:00 | 805.70 | 812.54 | 811.77 | EMA400 retest candle locked (from upside) |

### Cycle 193 — SELL (started 2026-01-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-19 11:15:00 | 803.10 | 809.70 | 810.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-20 12:15:00 | 795.75 | 806.50 | 808.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-21 09:15:00 | 812.35 | 798.15 | 802.84 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-21 09:15:00 | 812.35 | 798.15 | 802.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-21 09:15:00 | 812.35 | 798.15 | 802.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-21 10:00:00 | 812.35 | 798.15 | 802.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-21 10:15:00 | 805.80 | 799.68 | 803.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-21 10:45:00 | 810.60 | 799.68 | 803.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-21 12:15:00 | 806.10 | 802.87 | 804.08 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-21 13:15:00 | 802.50 | 802.87 | 804.08 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-22 09:30:00 | 801.40 | 803.67 | 803.97 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-22 10:15:00 | 819.10 | 806.75 | 805.34 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 194 — BUY (started 2026-01-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-22 10:15:00 | 819.10 | 806.75 | 805.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-28 11:15:00 | 830.15 | 820.23 | 816.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-29 09:15:00 | 819.35 | 827.88 | 822.74 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-29 09:15:00 | 819.35 | 827.88 | 822.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 09:15:00 | 819.35 | 827.88 | 822.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-29 09:30:00 | 821.00 | 827.88 | 822.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 10:15:00 | 816.65 | 825.63 | 822.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-29 10:30:00 | 816.10 | 825.63 | 822.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 13:15:00 | 815.65 | 821.85 | 821.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-29 14:00:00 | 815.65 | 821.85 | 821.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 195 — SELL (started 2026-01-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-29 14:15:00 | 804.30 | 818.34 | 819.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-30 09:15:00 | 799.00 | 812.31 | 816.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-01 10:15:00 | 798.15 | 794.62 | 802.41 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-01 11:00:00 | 798.15 | 794.62 | 802.41 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 09:15:00 | 777.15 | 765.80 | 777.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-03 09:30:00 | 782.00 | 765.80 | 777.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 10:15:00 | 787.10 | 770.06 | 778.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-03 11:00:00 | 787.10 | 770.06 | 778.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 11:15:00 | 799.00 | 775.85 | 780.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-03 12:00:00 | 799.00 | 775.85 | 780.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 196 — BUY (started 2026-02-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 13:15:00 | 805.75 | 785.78 | 784.05 | EMA200 above EMA400 |

### Cycle 197 — SELL (started 2026-02-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-06 09:15:00 | 765.15 | 787.38 | 790.18 | EMA200 below EMA400 |

### Cycle 198 — BUY (started 2026-02-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-09 10:15:00 | 821.95 | 788.56 | 786.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-09 11:15:00 | 827.25 | 796.30 | 790.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-10 15:15:00 | 823.00 | 823.84 | 814.20 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-11 09:15:00 | 820.25 | 823.84 | 814.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 10:15:00 | 821.10 | 830.26 | 823.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-12 10:30:00 | 823.65 | 830.26 | 823.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 11:15:00 | 831.60 | 830.53 | 824.67 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-12 12:15:00 | 837.85 | 830.53 | 824.67 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-13 12:00:00 | 833.50 | 834.10 | 829.85 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-16 10:15:00 | 817.05 | 826.62 | 827.80 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 199 — SELL (started 2026-02-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-16 10:15:00 | 817.05 | 826.62 | 827.80 | EMA200 below EMA400 |

### Cycle 200 — BUY (started 2026-02-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-17 12:15:00 | 829.00 | 826.42 | 826.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-18 09:15:00 | 831.20 | 828.49 | 827.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-19 14:15:00 | 832.25 | 841.09 | 837.12 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-19 14:15:00 | 832.25 | 841.09 | 837.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 14:15:00 | 832.25 | 841.09 | 837.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 15:00:00 | 832.25 | 841.09 | 837.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 15:15:00 | 841.00 | 841.08 | 837.47 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-20 09:45:00 | 857.00 | 843.27 | 838.79 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-23 09:15:00 | 850.00 | 848.44 | 844.24 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-23 10:45:00 | 849.35 | 848.01 | 844.73 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-24 11:15:00 | 836.60 | 844.73 | 845.07 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 201 — SELL (started 2026-02-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-24 11:15:00 | 836.60 | 844.73 | 845.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-24 12:15:00 | 831.55 | 842.09 | 843.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-24 14:15:00 | 842.95 | 842.00 | 843.48 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-24 15:00:00 | 842.95 | 842.00 | 843.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 09:15:00 | 841.15 | 841.89 | 843.18 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-25 11:15:00 | 837.20 | 841.27 | 842.78 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-25 13:30:00 | 837.30 | 838.72 | 841.11 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-26 14:45:00 | 838.00 | 835.92 | 837.90 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-02 09:15:00 | 795.34 | 819.15 | 827.45 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-02 09:15:00 | 795.43 | 819.15 | 827.45 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-02 09:15:00 | 796.10 | 819.15 | 827.45 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-02 15:15:00 | 813.00 | 806.62 | 815.78 | SL hit (close>ema200) qty=0.50 sl=806.62 alert=retest2 |

### Cycle 202 — BUY (started 2026-03-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-06 09:15:00 | 813.00 | 808.01 | 807.75 | EMA200 above EMA400 |

### Cycle 203 — SELL (started 2026-03-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-06 10:15:00 | 805.40 | 807.49 | 807.54 | EMA200 below EMA400 |

### Cycle 204 — BUY (started 2026-03-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-06 11:15:00 | 808.15 | 807.62 | 807.60 | EMA200 above EMA400 |

### Cycle 205 — SELL (started 2026-03-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-06 13:15:00 | 805.45 | 807.17 | 807.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-06 15:15:00 | 804.25 | 806.60 | 807.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-09 14:15:00 | 797.40 | 794.55 | 799.46 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-09 14:15:00 | 797.40 | 794.55 | 799.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-09 14:15:00 | 797.40 | 794.55 | 799.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-09 15:00:00 | 797.40 | 794.55 | 799.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-09 15:15:00 | 799.95 | 795.63 | 799.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-10 09:15:00 | 798.90 | 795.63 | 799.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 09:15:00 | 800.00 | 796.50 | 799.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-10 10:15:00 | 801.30 | 796.50 | 799.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 10:15:00 | 806.40 | 798.48 | 800.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-10 10:45:00 | 807.75 | 798.48 | 800.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 11:15:00 | 804.00 | 799.58 | 800.52 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-10 12:15:00 | 800.85 | 799.58 | 800.52 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-11 09:15:00 | 806.50 | 801.24 | 800.93 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 206 — BUY (started 2026-03-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-11 09:15:00 | 806.50 | 801.24 | 800.93 | EMA200 above EMA400 |

### Cycle 207 — SELL (started 2026-03-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-11 10:15:00 | 798.55 | 800.71 | 800.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-11 11:15:00 | 796.95 | 799.95 | 800.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-12 13:15:00 | 790.60 | 786.28 | 791.04 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-12 13:15:00 | 790.60 | 786.28 | 791.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 13:15:00 | 790.60 | 786.28 | 791.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-12 13:45:00 | 790.00 | 786.28 | 791.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 14:15:00 | 786.85 | 786.39 | 790.66 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-13 09:15:00 | 773.45 | 786.63 | 790.38 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-18 13:15:00 | 770.20 | 761.48 | 760.97 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 208 — BUY (started 2026-03-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 13:15:00 | 770.20 | 761.48 | 760.97 | EMA200 above EMA400 |

### Cycle 209 — SELL (started 2026-03-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 10:15:00 | 754.05 | 759.85 | 760.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 14:15:00 | 748.45 | 756.30 | 758.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-20 09:15:00 | 758.10 | 755.49 | 757.67 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-20 09:15:00 | 758.10 | 755.49 | 757.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 09:15:00 | 758.10 | 755.49 | 757.67 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-20 10:45:00 | 749.80 | 754.19 | 756.88 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-23 09:15:00 | 747.80 | 757.21 | 757.26 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-23 12:15:00 | 748.65 | 754.33 | 755.73 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-24 11:15:00 | 760.40 | 755.55 | 755.43 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 210 — BUY (started 2026-03-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-24 11:15:00 | 760.40 | 755.55 | 755.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-24 12:15:00 | 766.50 | 757.74 | 756.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-24 15:15:00 | 755.40 | 759.38 | 757.69 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-24 15:15:00 | 755.40 | 759.38 | 757.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 15:15:00 | 755.40 | 759.38 | 757.69 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-25 09:15:00 | 781.50 | 759.38 | 757.69 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-27 13:15:00 | 768.40 | 770.82 | 768.91 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-27 14:15:00 | 766.15 | 769.65 | 768.56 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-30 10:15:00 | 765.80 | 767.84 | 767.96 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 211 — SELL (started 2026-03-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-30 10:15:00 | 765.80 | 767.84 | 767.96 | EMA200 below EMA400 |

### Cycle 212 — BUY (started 2026-03-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-30 11:15:00 | 771.15 | 768.50 | 768.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-30 14:15:00 | 777.15 | 771.68 | 769.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-06 13:15:00 | 832.05 | 833.68 | 820.40 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-06 14:00:00 | 832.05 | 833.68 | 820.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-07 09:15:00 | 833.75 | 831.69 | 822.63 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-07 10:15:00 | 835.35 | 831.69 | 822.63 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-07 11:15:00 | 838.55 | 832.30 | 823.73 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2026-04-10 14:15:00 | 918.89 | 873.26 | 864.37 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 213 — SELL (started 2026-04-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-24 13:15:00 | 929.15 | 939.69 | 940.19 | EMA200 below EMA400 |

### Cycle 214 — BUY (started 2026-04-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 12:15:00 | 946.25 | 939.33 | 939.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-28 10:15:00 | 952.70 | 945.64 | 942.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-30 09:15:00 | 952.35 | 968.48 | 962.29 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-30 09:15:00 | 952.35 | 968.48 | 962.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 09:15:00 | 952.35 | 968.48 | 962.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-30 10:00:00 | 952.35 | 968.48 | 962.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 10:15:00 | 946.25 | 964.03 | 960.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-30 10:45:00 | 945.95 | 964.03 | 960.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 215 — SELL (started 2026-04-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-30 12:15:00 | 948.90 | 958.20 | 958.56 | EMA200 below EMA400 |

### Cycle 216 — BUY (started 2026-05-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-04 09:15:00 | 972.00 | 959.10 | 958.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-06 09:15:00 | 987.95 | 973.94 | 969.99 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2023-05-15 12:00:00 | 1152.45 | 2023-05-24 12:15:00 | 1181.00 | STOP_HIT | 1.00 | 2.48% |
| BUY | retest2 | 2023-05-16 13:30:00 | 1146.70 | 2023-05-24 12:15:00 | 1181.00 | STOP_HIT | 1.00 | 2.99% |
| BUY | retest2 | 2023-05-17 09:15:00 | 1165.90 | 2023-05-24 12:15:00 | 1181.00 | STOP_HIT | 1.00 | 1.30% |
| SELL | retest2 | 2023-05-30 12:15:00 | 1150.45 | 2023-06-02 09:15:00 | 1164.95 | STOP_HIT | 1.00 | -1.26% |
| SELL | retest2 | 2023-05-30 14:15:00 | 1150.45 | 2023-06-02 09:15:00 | 1164.95 | STOP_HIT | 1.00 | -1.26% |
| SELL | retest2 | 2023-05-30 15:15:00 | 1150.00 | 2023-06-02 09:15:00 | 1164.95 | STOP_HIT | 1.00 | -1.30% |
| SELL | retest2 | 2023-05-31 11:00:00 | 1150.05 | 2023-06-02 09:15:00 | 1164.95 | STOP_HIT | 1.00 | -1.30% |
| BUY | retest2 | 2023-06-12 09:15:00 | 1192.10 | 2023-06-20 09:15:00 | 1216.65 | STOP_HIT | 1.00 | 2.06% |
| BUY | retest2 | 2023-06-13 09:30:00 | 1193.55 | 2023-06-20 09:15:00 | 1216.65 | STOP_HIT | 1.00 | 1.94% |
| SELL | retest2 | 2023-06-22 12:30:00 | 1201.25 | 2023-06-22 14:15:00 | 1220.00 | STOP_HIT | 1.00 | -1.56% |
| SELL | retest2 | 2023-06-23 09:30:00 | 1195.75 | 2023-06-30 10:15:00 | 1189.15 | STOP_HIT | 1.00 | 0.55% |
| SELL | retest2 | 2023-06-23 10:15:00 | 1201.85 | 2023-06-30 10:15:00 | 1189.15 | STOP_HIT | 1.00 | 1.06% |
| SELL | retest2 | 2023-06-23 12:00:00 | 1201.60 | 2023-06-30 10:15:00 | 1189.15 | STOP_HIT | 1.00 | 1.04% |
| BUY | retest2 | 2023-07-04 09:15:00 | 1218.60 | 2023-07-04 12:15:00 | 1198.00 | STOP_HIT | 1.00 | -1.69% |
| BUY | retest2 | 2023-07-05 09:15:00 | 1219.40 | 2023-07-05 15:15:00 | 1195.00 | STOP_HIT | 1.00 | -2.00% |
| BUY | retest2 | 2023-07-06 09:15:00 | 1213.95 | 2023-07-06 15:15:00 | 1197.10 | STOP_HIT | 1.00 | -1.39% |
| BUY | retest2 | 2023-07-06 12:00:00 | 1212.00 | 2023-07-06 15:15:00 | 1197.10 | STOP_HIT | 1.00 | -1.23% |
| SELL | retest2 | 2023-07-25 11:00:00 | 1184.10 | 2023-07-26 09:15:00 | 1215.35 | STOP_HIT | 1.00 | -2.64% |
| BUY | retest2 | 2023-08-01 09:30:00 | 1240.00 | 2023-08-07 09:15:00 | 1225.00 | STOP_HIT | 1.00 | -1.21% |
| BUY | retest2 | 2023-08-01 10:45:00 | 1243.50 | 2023-08-07 09:15:00 | 1225.00 | STOP_HIT | 1.00 | -1.49% |
| BUY | retest2 | 2023-08-28 14:00:00 | 1141.05 | 2023-08-31 13:15:00 | 1135.50 | STOP_HIT | 1.00 | -0.49% |
| BUY | retest2 | 2023-08-29 14:45:00 | 1141.40 | 2023-08-31 13:15:00 | 1135.50 | STOP_HIT | 1.00 | -0.52% |
| BUY | retest2 | 2023-08-30 10:15:00 | 1142.90 | 2023-08-31 13:15:00 | 1135.50 | STOP_HIT | 1.00 | -0.65% |
| BUY | retest2 | 2023-08-30 12:30:00 | 1142.10 | 2023-08-31 13:15:00 | 1135.50 | STOP_HIT | 1.00 | -0.58% |
| SELL | retest2 | 2023-10-10 13:30:00 | 1148.50 | 2023-10-11 10:15:00 | 1164.00 | STOP_HIT | 1.00 | -1.35% |
| BUY | retest2 | 2023-10-16 09:15:00 | 1171.75 | 2023-10-16 11:15:00 | 1159.90 | STOP_HIT | 1.00 | -1.01% |
| SELL | retest2 | 2023-10-19 14:15:00 | 1143.00 | 2023-10-25 12:15:00 | 1085.85 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-10-20 11:00:00 | 1145.30 | 2023-10-25 12:15:00 | 1088.03 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-10-19 14:15:00 | 1143.00 | 2023-10-25 15:15:00 | 1095.00 | STOP_HIT | 0.50 | 4.20% |
| SELL | retest2 | 2023-10-20 11:00:00 | 1145.30 | 2023-10-25 15:15:00 | 1095.00 | STOP_HIT | 0.50 | 4.39% |
| SELL | retest1 | 2023-10-31 15:00:00 | 1074.00 | 2023-11-02 10:15:00 | 1073.00 | STOP_HIT | 1.00 | 0.09% |
| BUY | retest2 | 2023-11-21 09:15:00 | 1101.05 | 2023-11-21 11:15:00 | 1095.00 | STOP_HIT | 1.00 | -0.55% |
| BUY | retest2 | 2023-11-29 15:15:00 | 1159.90 | 2023-11-30 09:15:00 | 1131.00 | STOP_HIT | 1.00 | -2.49% |
| SELL | retest2 | 2023-12-27 15:00:00 | 1109.70 | 2023-12-28 09:15:00 | 1121.00 | STOP_HIT | 1.00 | -1.02% |
| SELL | retest2 | 2024-01-01 09:45:00 | 1107.20 | 2024-01-01 10:15:00 | 1116.60 | STOP_HIT | 1.00 | -0.85% |
| BUY | retest2 | 2024-01-03 12:45:00 | 1125.40 | 2024-01-10 12:15:00 | 1134.95 | STOP_HIT | 1.00 | 0.85% |
| BUY | retest2 | 2024-01-03 13:45:00 | 1126.80 | 2024-01-10 12:15:00 | 1134.95 | STOP_HIT | 1.00 | 0.72% |
| BUY | retest2 | 2024-01-03 14:45:00 | 1127.00 | 2024-01-10 12:15:00 | 1134.95 | STOP_HIT | 1.00 | 0.71% |
| BUY | retest2 | 2024-01-04 09:30:00 | 1130.20 | 2024-01-10 12:15:00 | 1134.95 | STOP_HIT | 1.00 | 0.42% |
| BUY | retest2 | 2024-01-05 10:45:00 | 1126.20 | 2024-01-10 12:15:00 | 1134.95 | STOP_HIT | 1.00 | 0.78% |
| BUY | retest2 | 2024-01-12 11:15:00 | 1167.50 | 2024-01-16 15:15:00 | 1160.00 | STOP_HIT | 1.00 | -0.64% |
| BUY | retest2 | 2024-01-12 14:30:00 | 1166.15 | 2024-01-16 15:15:00 | 1160.00 | STOP_HIT | 1.00 | -0.53% |
| BUY | retest2 | 2024-01-15 09:15:00 | 1170.25 | 2024-01-16 15:15:00 | 1160.00 | STOP_HIT | 1.00 | -0.88% |
| BUY | retest2 | 2024-01-15 10:00:00 | 1167.90 | 2024-01-16 15:15:00 | 1160.00 | STOP_HIT | 1.00 | -0.68% |
| SELL | retest2 | 2024-01-20 09:15:00 | 1136.30 | 2024-01-24 09:15:00 | 1195.00 | STOP_HIT | 1.00 | -5.17% |
| SELL | retest2 | 2024-01-20 11:30:00 | 1135.00 | 2024-01-24 09:15:00 | 1195.00 | STOP_HIT | 1.00 | -5.29% |
| SELL | retest2 | 2024-01-31 13:45:00 | 1128.00 | 2024-02-07 09:15:00 | 1169.55 | STOP_HIT | 1.00 | -3.68% |
| SELL | retest2 | 2024-02-01 10:30:00 | 1130.50 | 2024-02-07 09:15:00 | 1169.55 | STOP_HIT | 1.00 | -3.45% |
| SELL | retest2 | 2024-02-01 13:30:00 | 1131.25 | 2024-02-07 09:15:00 | 1169.55 | STOP_HIT | 1.00 | -3.39% |
| SELL | retest2 | 2024-02-01 15:00:00 | 1131.00 | 2024-02-07 09:15:00 | 1169.55 | STOP_HIT | 1.00 | -3.41% |
| SELL | retest2 | 2024-02-02 13:15:00 | 1121.30 | 2024-02-07 09:15:00 | 1169.55 | STOP_HIT | 1.00 | -4.30% |
| SELL | retest2 | 2024-02-02 14:00:00 | 1116.20 | 2024-02-07 09:15:00 | 1169.55 | STOP_HIT | 1.00 | -4.78% |
| SELL | retest2 | 2024-02-06 14:45:00 | 1118.45 | 2024-02-07 09:15:00 | 1169.55 | STOP_HIT | 1.00 | -4.57% |
| SELL | retest2 | 2024-02-06 15:15:00 | 1120.50 | 2024-02-07 09:15:00 | 1169.55 | STOP_HIT | 1.00 | -4.38% |
| SELL | retest2 | 2024-02-16 15:15:00 | 1108.30 | 2024-02-19 10:15:00 | 1122.35 | STOP_HIT | 1.00 | -1.27% |
| SELL | retest2 | 2024-02-19 12:00:00 | 1114.65 | 2024-02-27 09:15:00 | 1058.92 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-02-19 13:00:00 | 1108.65 | 2024-02-27 09:15:00 | 1053.22 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-02-19 12:00:00 | 1114.65 | 2024-02-28 09:15:00 | 1055.00 | STOP_HIT | 0.50 | 5.35% |
| SELL | retest2 | 2024-02-19 13:00:00 | 1108.65 | 2024-02-28 09:15:00 | 1055.00 | STOP_HIT | 0.50 | 4.84% |
| BUY | retest2 | 2024-03-04 11:45:00 | 1071.90 | 2024-03-06 11:15:00 | 1056.40 | STOP_HIT | 1.00 | -1.45% |
| BUY | retest2 | 2024-03-28 09:15:00 | 1272.65 | 2024-04-04 13:15:00 | 1296.95 | STOP_HIT | 1.00 | 1.91% |
| SELL | retest2 | 2024-04-05 11:15:00 | 1295.85 | 2024-04-09 15:15:00 | 1306.00 | STOP_HIT | 1.00 | -0.78% |
| SELL | retest2 | 2024-04-15 14:15:00 | 1259.00 | 2024-04-19 15:15:00 | 1255.20 | STOP_HIT | 1.00 | 0.30% |
| BUY | retest1 | 2024-04-24 10:00:00 | 1316.40 | 2024-04-25 09:15:00 | 1382.22 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2024-04-24 10:00:00 | 1316.40 | 2024-04-26 09:15:00 | 1448.04 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2024-05-02 09:15:00 | 1440.30 | 2024-05-06 09:15:00 | 1443.60 | STOP_HIT | 1.00 | 0.23% |
| BUY | retest2 | 2024-05-03 14:00:00 | 1431.80 | 2024-05-06 09:15:00 | 1443.60 | STOP_HIT | 1.00 | 0.82% |
| BUY | retest2 | 2024-05-08 12:30:00 | 1503.05 | 2024-05-09 11:15:00 | 1478.50 | STOP_HIT | 1.00 | -1.63% |
| SELL | retest2 | 2024-05-10 13:30:00 | 1470.00 | 2024-05-13 09:15:00 | 1486.25 | STOP_HIT | 1.00 | -1.11% |
| SELL | retest2 | 2024-05-30 13:00:00 | 1599.15 | 2024-06-03 09:15:00 | 1660.15 | STOP_HIT | 1.00 | -3.81% |
| SELL | retest2 | 2024-05-31 11:00:00 | 1598.85 | 2024-06-03 09:15:00 | 1660.15 | STOP_HIT | 1.00 | -3.83% |
| SELL | retest2 | 2024-05-31 14:30:00 | 1600.20 | 2024-06-03 09:15:00 | 1660.15 | STOP_HIT | 1.00 | -3.75% |
| SELL | retest2 | 2024-05-31 15:00:00 | 1600.20 | 2024-06-03 09:15:00 | 1660.15 | STOP_HIT | 1.00 | -3.75% |
| SELL | retest2 | 2024-06-06 13:00:00 | 1572.85 | 2024-06-07 09:15:00 | 1590.30 | STOP_HIT | 1.00 | -1.11% |
| SELL | retest2 | 2024-06-06 13:45:00 | 1573.00 | 2024-06-07 09:15:00 | 1590.30 | STOP_HIT | 1.00 | -1.10% |
| SELL | retest2 | 2024-06-07 09:15:00 | 1571.00 | 2024-06-07 09:15:00 | 1590.30 | STOP_HIT | 1.00 | -1.23% |
| SELL | retest2 | 2024-06-07 11:45:00 | 1571.55 | 2024-06-07 13:15:00 | 1588.55 | STOP_HIT | 1.00 | -1.08% |
| BUY | retest1 | 2024-06-12 09:15:00 | 1708.95 | 2024-06-18 09:15:00 | 1794.40 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2024-06-12 09:15:00 | 1708.95 | 2024-06-18 10:15:00 | 1750.05 | STOP_HIT | 0.50 | 2.40% |
| SELL | retest2 | 2024-07-11 14:00:00 | 1682.95 | 2024-07-12 10:15:00 | 1702.20 | STOP_HIT | 1.00 | -1.14% |
| SELL | retest2 | 2024-07-11 15:00:00 | 1685.15 | 2024-07-12 10:15:00 | 1702.20 | STOP_HIT | 1.00 | -1.01% |
| BUY | retest2 | 2024-07-18 11:15:00 | 1735.50 | 2024-07-19 12:15:00 | 1720.00 | STOP_HIT | 1.00 | -0.89% |
| BUY | retest2 | 2024-07-19 12:00:00 | 1720.00 | 2024-07-19 12:15:00 | 1720.00 | STOP_HIT | 1.00 | 0.00% |
| BUY | retest2 | 2024-07-26 11:00:00 | 1725.10 | 2024-07-30 15:15:00 | 1719.50 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest2 | 2024-08-07 10:30:00 | 1603.35 | 2024-08-09 09:15:00 | 1528.36 | PARTIAL | 0.50 | 4.68% |
| SELL | retest2 | 2024-08-07 13:30:00 | 1608.80 | 2024-08-09 14:15:00 | 1523.18 | PARTIAL | 0.50 | 5.32% |
| SELL | retest2 | 2024-08-07 14:45:00 | 1604.25 | 2024-08-09 14:15:00 | 1524.04 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-08-07 10:30:00 | 1603.35 | 2024-08-12 13:15:00 | 1534.45 | STOP_HIT | 0.50 | 4.30% |
| SELL | retest2 | 2024-08-07 13:30:00 | 1608.80 | 2024-08-12 13:15:00 | 1534.45 | STOP_HIT | 0.50 | 4.62% |
| SELL | retest2 | 2024-08-07 14:45:00 | 1604.25 | 2024-08-12 13:15:00 | 1534.45 | STOP_HIT | 0.50 | 4.35% |
| SELL | retest2 | 2024-08-28 11:00:00 | 1529.90 | 2024-09-05 11:15:00 | 1532.30 | STOP_HIT | 1.00 | -0.16% |
| SELL | retest2 | 2024-08-28 13:30:00 | 1524.00 | 2024-09-05 11:15:00 | 1532.30 | STOP_HIT | 1.00 | -0.54% |
| SELL | retest2 | 2024-09-10 12:15:00 | 1496.15 | 2024-09-10 14:15:00 | 1525.80 | STOP_HIT | 1.00 | -1.98% |
| BUY | retest2 | 2024-09-12 11:15:00 | 1529.00 | 2024-09-13 10:15:00 | 1505.85 | STOP_HIT | 1.00 | -1.51% |
| BUY | retest2 | 2024-09-12 13:15:00 | 1526.80 | 2024-09-13 10:15:00 | 1505.85 | STOP_HIT | 1.00 | -1.37% |
| SELL | retest2 | 2024-09-20 13:30:00 | 1494.40 | 2024-09-23 12:15:00 | 1504.40 | STOP_HIT | 1.00 | -0.67% |
| SELL | retest2 | 2024-09-20 15:15:00 | 1495.00 | 2024-09-23 12:15:00 | 1504.40 | STOP_HIT | 1.00 | -0.63% |
| SELL | retest2 | 2024-09-23 10:00:00 | 1495.20 | 2024-09-23 12:15:00 | 1504.40 | STOP_HIT | 1.00 | -0.62% |
| BUY | retest2 | 2024-09-26 12:15:00 | 1519.90 | 2024-09-27 14:15:00 | 1500.00 | STOP_HIT | 1.00 | -1.31% |
| BUY | retest2 | 2024-09-26 15:00:00 | 1520.00 | 2024-09-27 14:15:00 | 1500.00 | STOP_HIT | 1.00 | -1.32% |
| BUY | retest2 | 2024-09-27 10:15:00 | 1520.20 | 2024-09-27 14:15:00 | 1500.00 | STOP_HIT | 1.00 | -1.33% |
| BUY | retest2 | 2024-09-27 11:30:00 | 1522.90 | 2024-09-27 14:15:00 | 1500.00 | STOP_HIT | 1.00 | -1.50% |
| SELL | retest2 | 2024-10-01 11:15:00 | 1493.45 | 2024-10-07 09:15:00 | 1418.78 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-01 11:15:00 | 1493.45 | 2024-10-08 11:15:00 | 1431.45 | STOP_HIT | 0.50 | 4.15% |
| BUY | retest2 | 2024-10-11 14:00:00 | 1465.20 | 2024-10-17 11:15:00 | 1464.50 | STOP_HIT | 1.00 | -0.05% |
| BUY | retest2 | 2024-10-14 09:45:00 | 1463.85 | 2024-10-17 11:15:00 | 1464.50 | STOP_HIT | 1.00 | 0.04% |
| BUY | retest2 | 2024-10-14 10:45:00 | 1460.40 | 2024-10-17 11:15:00 | 1464.50 | STOP_HIT | 1.00 | 0.28% |
| SELL | retest2 | 2024-10-25 09:15:00 | 1404.10 | 2024-10-30 11:15:00 | 1397.10 | STOP_HIT | 1.00 | 0.50% |
| BUY | retest2 | 2024-11-06 09:15:00 | 1419.95 | 2024-11-11 09:15:00 | 1410.50 | STOP_HIT | 1.00 | -0.67% |
| BUY | retest2 | 2024-11-06 11:15:00 | 1423.85 | 2024-11-11 09:15:00 | 1410.50 | STOP_HIT | 1.00 | -0.94% |
| BUY | retest2 | 2024-11-08 11:30:00 | 1406.90 | 2024-11-11 09:15:00 | 1410.50 | STOP_HIT | 1.00 | 0.26% |
| SELL | retest2 | 2024-11-13 14:30:00 | 1411.00 | 2024-11-13 15:15:00 | 1424.45 | STOP_HIT | 1.00 | -0.95% |
| SELL | retest2 | 2024-11-21 12:15:00 | 1424.40 | 2024-11-26 13:15:00 | 1417.30 | STOP_HIT | 1.00 | 0.50% |
| SELL | retest2 | 2024-11-22 15:00:00 | 1410.30 | 2024-11-26 13:15:00 | 1417.30 | STOP_HIT | 1.00 | -0.50% |
| SELL | retest2 | 2024-11-26 09:30:00 | 1425.65 | 2024-11-26 13:15:00 | 1417.30 | STOP_HIT | 1.00 | 0.59% |
| BUY | retest2 | 2024-11-29 09:30:00 | 1427.70 | 2024-12-03 12:15:00 | 1421.40 | STOP_HIT | 1.00 | -0.44% |
| SELL | retest2 | 2024-12-10 10:15:00 | 1360.00 | 2024-12-11 09:15:00 | 1378.45 | STOP_HIT | 1.00 | -1.36% |
| SELL | retest2 | 2024-12-10 12:30:00 | 1365.05 | 2024-12-11 09:15:00 | 1378.45 | STOP_HIT | 1.00 | -0.98% |
| SELL | retest2 | 2024-12-10 13:00:00 | 1354.65 | 2024-12-11 09:15:00 | 1378.45 | STOP_HIT | 1.00 | -1.76% |
| SELL | retest2 | 2024-12-19 09:15:00 | 1310.90 | 2024-12-20 09:15:00 | 1355.00 | STOP_HIT | 1.00 | -3.36% |
| SELL | retest2 | 2024-12-19 09:45:00 | 1313.05 | 2024-12-20 09:15:00 | 1355.00 | STOP_HIT | 1.00 | -3.19% |
| SELL | retest2 | 2024-12-26 09:30:00 | 1276.50 | 2024-12-31 15:15:00 | 1285.05 | STOP_HIT | 1.00 | -0.67% |
| SELL | retest2 | 2024-12-27 10:15:00 | 1279.75 | 2024-12-31 15:15:00 | 1285.05 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest2 | 2024-12-30 09:30:00 | 1279.10 | 2024-12-31 15:15:00 | 1285.05 | STOP_HIT | 1.00 | -0.47% |
| SELL | retest2 | 2024-12-31 10:00:00 | 1278.40 | 2024-12-31 15:15:00 | 1285.05 | STOP_HIT | 1.00 | -0.52% |
| SELL | retest2 | 2024-12-31 11:15:00 | 1275.60 | 2024-12-31 15:15:00 | 1285.05 | STOP_HIT | 1.00 | -0.74% |
| SELL | retest2 | 2024-12-31 12:30:00 | 1275.55 | 2024-12-31 15:15:00 | 1285.05 | STOP_HIT | 1.00 | -0.74% |
| SELL | retest2 | 2024-12-31 13:15:00 | 1275.00 | 2024-12-31 15:15:00 | 1285.05 | STOP_HIT | 1.00 | -0.79% |
| SELL | retest2 | 2024-12-31 14:30:00 | 1275.35 | 2024-12-31 15:15:00 | 1285.05 | STOP_HIT | 1.00 | -0.76% |
| SELL | retest2 | 2025-01-09 15:15:00 | 1262.00 | 2025-01-15 10:15:00 | 1256.35 | STOP_HIT | 1.00 | 0.45% |
| SELL | retest2 | 2025-01-20 15:15:00 | 1221.20 | 2025-01-27 09:15:00 | 1160.14 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-21 09:45:00 | 1225.45 | 2025-01-27 09:15:00 | 1164.18 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-20 15:15:00 | 1221.20 | 2025-01-28 14:15:00 | 1146.10 | STOP_HIT | 0.50 | 6.15% |
| SELL | retest2 | 2025-01-21 09:45:00 | 1225.45 | 2025-01-28 14:15:00 | 1146.10 | STOP_HIT | 0.50 | 6.48% |
| BUY | retest1 | 2025-02-01 15:00:00 | 1186.40 | 2025-02-03 09:15:00 | 1169.70 | STOP_HIT | 1.00 | -1.41% |
| SELL | retest2 | 2025-02-10 09:15:00 | 1086.30 | 2025-02-11 09:15:00 | 1031.98 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-10 09:15:00 | 1086.30 | 2025-02-13 11:15:00 | 1032.00 | STOP_HIT | 0.50 | 5.00% |
| BUY | retest2 | 2025-03-17 11:30:00 | 976.65 | 2025-03-25 09:15:00 | 1074.32 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-03-17 12:45:00 | 972.25 | 2025-03-25 09:15:00 | 1069.48 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-03-17 13:30:00 | 972.70 | 2025-03-25 09:15:00 | 1069.97 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-03-17 14:15:00 | 972.25 | 2025-03-25 09:15:00 | 1069.48 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-03-19 14:30:00 | 976.80 | 2025-03-25 09:15:00 | 1074.48 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-03-20 10:15:00 | 977.05 | 2025-03-25 09:15:00 | 1074.76 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-04-08 10:45:00 | 952.50 | 2025-04-15 10:15:00 | 979.30 | STOP_HIT | 1.00 | -2.81% |
| SELL | retest2 | 2025-04-11 11:15:00 | 949.90 | 2025-04-15 10:15:00 | 979.30 | STOP_HIT | 1.00 | -3.10% |
| SELL | retest2 | 2025-04-11 11:45:00 | 949.85 | 2025-04-15 10:15:00 | 979.30 | STOP_HIT | 1.00 | -3.10% |
| SELL | retest2 | 2025-04-15 09:15:00 | 954.50 | 2025-04-15 10:15:00 | 979.30 | STOP_HIT | 1.00 | -2.60% |
| SELL | retest2 | 2025-05-05 13:30:00 | 1008.80 | 2025-05-07 12:15:00 | 958.36 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-05-05 13:30:00 | 1008.80 | 2025-05-08 09:15:00 | 1008.00 | STOP_HIT | 0.50 | 0.08% |
| BUY | retest2 | 2025-05-21 13:15:00 | 1005.40 | 2025-05-26 11:15:00 | 997.10 | STOP_HIT | 1.00 | -0.83% |
| BUY | retest2 | 2025-05-22 09:30:00 | 1001.20 | 2025-05-26 11:15:00 | 997.10 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest2 | 2025-05-23 12:00:00 | 1003.20 | 2025-05-26 11:15:00 | 997.10 | STOP_HIT | 1.00 | -0.61% |
| BUY | retest2 | 2025-05-23 12:45:00 | 1002.00 | 2025-05-26 11:15:00 | 997.10 | STOP_HIT | 1.00 | -0.49% |
| SELL | retest2 | 2025-05-29 10:15:00 | 988.70 | 2025-06-12 11:15:00 | 939.26 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-05-30 09:30:00 | 991.30 | 2025-06-12 11:15:00 | 941.73 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-06-02 10:00:00 | 989.00 | 2025-06-12 11:15:00 | 939.55 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-06-03 10:30:00 | 990.60 | 2025-06-12 11:15:00 | 941.07 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-06-03 12:45:00 | 981.60 | 2025-06-12 13:15:00 | 932.52 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-05-29 10:15:00 | 988.70 | 2025-06-13 11:15:00 | 943.40 | STOP_HIT | 0.50 | 4.58% |
| SELL | retest2 | 2025-05-30 09:30:00 | 991.30 | 2025-06-13 11:15:00 | 943.40 | STOP_HIT | 0.50 | 4.83% |
| SELL | retest2 | 2025-06-02 10:00:00 | 989.00 | 2025-06-13 11:15:00 | 943.40 | STOP_HIT | 0.50 | 4.61% |
| SELL | retest2 | 2025-06-03 10:30:00 | 990.60 | 2025-06-13 11:15:00 | 943.40 | STOP_HIT | 0.50 | 4.76% |
| SELL | retest2 | 2025-06-03 12:45:00 | 981.60 | 2025-06-13 11:15:00 | 943.40 | STOP_HIT | 0.50 | 3.89% |
| SELL | retest2 | 2025-06-19 10:30:00 | 928.60 | 2025-06-20 14:15:00 | 939.35 | STOP_HIT | 1.00 | -1.16% |
| SELL | retest2 | 2025-06-19 12:45:00 | 924.95 | 2025-06-20 14:15:00 | 939.35 | STOP_HIT | 1.00 | -1.56% |
| SELL | retest2 | 2025-06-20 15:15:00 | 923.90 | 2025-06-24 10:15:00 | 942.10 | STOP_HIT | 1.00 | -1.97% |
| SELL | retest2 | 2025-06-23 09:30:00 | 926.35 | 2025-06-24 10:15:00 | 942.10 | STOP_HIT | 1.00 | -1.70% |
| BUY | retest2 | 2025-06-27 10:15:00 | 976.70 | 2025-07-07 10:15:00 | 983.40 | STOP_HIT | 1.00 | 0.69% |
| BUY | retest2 | 2025-06-30 09:15:00 | 973.00 | 2025-07-07 10:15:00 | 983.40 | STOP_HIT | 1.00 | 1.07% |
| BUY | retest2 | 2025-06-30 12:30:00 | 968.40 | 2025-07-07 10:15:00 | 983.40 | STOP_HIT | 1.00 | 1.55% |
| BUY | retest2 | 2025-06-30 14:15:00 | 969.10 | 2025-07-07 10:15:00 | 983.40 | STOP_HIT | 1.00 | 1.48% |
| BUY | retest2 | 2025-07-01 09:15:00 | 982.35 | 2025-07-07 10:15:00 | 983.40 | STOP_HIT | 1.00 | 0.11% |
| SELL | retest2 | 2025-07-14 12:15:00 | 976.90 | 2025-07-14 14:15:00 | 995.80 | STOP_HIT | 1.00 | -1.93% |
| SELL | retest2 | 2025-07-14 13:00:00 | 978.00 | 2025-07-14 14:15:00 | 995.80 | STOP_HIT | 1.00 | -1.82% |
| SELL | retest2 | 2025-07-22 10:45:00 | 986.40 | 2025-07-28 13:15:00 | 937.08 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-22 12:00:00 | 983.75 | 2025-07-28 13:15:00 | 934.56 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-22 10:45:00 | 986.40 | 2025-07-30 09:15:00 | 940.00 | STOP_HIT | 0.50 | 4.70% |
| SELL | retest2 | 2025-07-22 12:00:00 | 983.75 | 2025-07-30 09:15:00 | 940.00 | STOP_HIT | 0.50 | 4.45% |
| SELL | retest1 | 2025-08-05 11:00:00 | 910.55 | 2025-08-07 10:15:00 | 865.02 | PARTIAL | 0.50 | 5.00% |
| SELL | retest1 | 2025-08-05 11:00:00 | 910.55 | 2025-08-11 11:15:00 | 851.60 | STOP_HIT | 0.50 | 6.47% |
| SELL | retest2 | 2025-08-12 11:30:00 | 852.10 | 2025-08-18 09:15:00 | 869.90 | STOP_HIT | 1.00 | -2.09% |
| SELL | retest2 | 2025-08-12 12:30:00 | 852.10 | 2025-08-18 09:15:00 | 869.90 | STOP_HIT | 1.00 | -2.09% |
| SELL | retest2 | 2025-08-13 13:30:00 | 850.90 | 2025-08-18 09:15:00 | 869.90 | STOP_HIT | 1.00 | -2.23% |
| SELL | retest2 | 2025-08-14 10:30:00 | 852.85 | 2025-08-18 09:15:00 | 869.90 | STOP_HIT | 1.00 | -2.00% |
| BUY | retest2 | 2025-08-19 14:30:00 | 875.50 | 2025-08-20 11:15:00 | 963.05 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-08-20 09:15:00 | 875.55 | 2025-08-20 11:15:00 | 963.11 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-08-29 12:15:00 | 915.00 | 2025-09-01 10:15:00 | 916.00 | STOP_HIT | 1.00 | -0.11% |
| BUY | retest2 | 2025-09-05 09:45:00 | 957.45 | 2025-09-15 12:15:00 | 979.70 | STOP_HIT | 1.00 | 2.32% |
| BUY | retest2 | 2025-09-05 13:30:00 | 953.05 | 2025-09-15 12:15:00 | 979.70 | STOP_HIT | 1.00 | 2.80% |
| BUY | retest2 | 2025-09-05 14:15:00 | 956.10 | 2025-09-15 12:15:00 | 979.70 | STOP_HIT | 1.00 | 2.47% |
| BUY | retest2 | 2025-09-08 09:30:00 | 962.30 | 2025-09-15 12:15:00 | 979.70 | STOP_HIT | 1.00 | 1.81% |
| SELL | retest2 | 2025-10-01 09:15:00 | 920.60 | 2025-10-03 11:15:00 | 929.45 | STOP_HIT | 1.00 | -0.96% |
| SELL | retest2 | 2025-10-01 09:45:00 | 918.90 | 2025-10-03 11:15:00 | 929.45 | STOP_HIT | 1.00 | -1.15% |
| SELL | retest2 | 2025-10-01 14:45:00 | 919.30 | 2025-10-03 11:15:00 | 929.45 | STOP_HIT | 1.00 | -1.10% |
| SELL | retest2 | 2025-10-03 10:00:00 | 918.65 | 2025-10-03 11:15:00 | 929.45 | STOP_HIT | 1.00 | -1.18% |
| SELL | retest2 | 2025-10-08 11:15:00 | 918.25 | 2025-10-10 13:15:00 | 926.00 | STOP_HIT | 1.00 | -0.84% |
| SELL | retest2 | 2025-10-08 12:45:00 | 918.70 | 2025-10-10 13:15:00 | 926.00 | STOP_HIT | 1.00 | -0.79% |
| SELL | retest2 | 2025-10-10 09:45:00 | 915.45 | 2025-10-10 13:15:00 | 926.00 | STOP_HIT | 1.00 | -1.15% |
| SELL | retest2 | 2025-10-10 10:15:00 | 915.80 | 2025-10-10 13:15:00 | 926.00 | STOP_HIT | 1.00 | -1.11% |
| SELL | retest2 | 2025-10-17 11:30:00 | 909.00 | 2025-10-27 12:15:00 | 906.00 | STOP_HIT | 1.00 | 0.33% |
| SELL | retest2 | 2025-10-21 14:15:00 | 908.00 | 2025-10-27 12:15:00 | 906.00 | STOP_HIT | 1.00 | 0.22% |
| BUY | retest2 | 2025-10-29 12:00:00 | 914.65 | 2025-10-31 11:15:00 | 905.05 | STOP_HIT | 1.00 | -1.05% |
| BUY | retest2 | 2025-10-29 13:30:00 | 914.70 | 2025-10-31 11:15:00 | 905.05 | STOP_HIT | 1.00 | -1.05% |
| SELL | retest2 | 2025-11-14 09:15:00 | 885.00 | 2025-11-19 10:15:00 | 843.60 | PARTIAL | 0.50 | 4.68% |
| SELL | retest2 | 2025-11-14 11:45:00 | 888.00 | 2025-11-19 12:15:00 | 840.75 | PARTIAL | 0.50 | 5.32% |
| SELL | retest2 | 2025-11-17 11:00:00 | 885.05 | 2025-11-19 12:15:00 | 840.80 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-14 09:15:00 | 885.00 | 2025-11-24 10:15:00 | 830.90 | STOP_HIT | 0.50 | 6.11% |
| SELL | retest2 | 2025-11-14 11:45:00 | 888.00 | 2025-11-24 10:15:00 | 830.90 | STOP_HIT | 0.50 | 6.43% |
| SELL | retest2 | 2025-11-17 11:00:00 | 885.05 | 2025-11-24 10:15:00 | 830.90 | STOP_HIT | 0.50 | 6.12% |
| BUY | retest2 | 2025-11-27 13:00:00 | 870.60 | 2025-11-28 09:15:00 | 855.00 | STOP_HIT | 1.00 | -1.79% |
| BUY | retest2 | 2025-11-27 14:15:00 | 870.00 | 2025-11-28 09:15:00 | 855.00 | STOP_HIT | 1.00 | -1.72% |
| BUY | retest2 | 2025-11-27 15:15:00 | 869.90 | 2025-11-28 09:15:00 | 855.00 | STOP_HIT | 1.00 | -1.71% |
| BUY | retest2 | 2025-12-03 10:00:00 | 888.15 | 2025-12-05 11:15:00 | 875.00 | STOP_HIT | 1.00 | -1.48% |
| BUY | retest2 | 2025-12-03 10:30:00 | 891.20 | 2025-12-05 11:15:00 | 875.00 | STOP_HIT | 1.00 | -1.82% |
| BUY | retest2 | 2025-12-03 14:00:00 | 888.45 | 2025-12-05 11:15:00 | 875.00 | STOP_HIT | 1.00 | -1.51% |
| BUY | retest2 | 2025-12-03 15:15:00 | 894.25 | 2025-12-05 11:15:00 | 875.00 | STOP_HIT | 1.00 | -2.15% |
| SELL | retest2 | 2025-12-10 09:15:00 | 863.55 | 2025-12-11 14:15:00 | 869.75 | STOP_HIT | 1.00 | -0.72% |
| SELL | retest2 | 2025-12-19 09:30:00 | 848.00 | 2025-12-19 13:15:00 | 859.80 | STOP_HIT | 1.00 | -1.39% |
| SELL | retest2 | 2025-12-22 10:00:00 | 848.00 | 2025-12-23 15:15:00 | 857.00 | STOP_HIT | 1.00 | -1.06% |
| SELL | retest2 | 2025-12-23 09:15:00 | 844.80 | 2025-12-23 15:15:00 | 857.00 | STOP_HIT | 1.00 | -1.44% |
| SELL | retest2 | 2025-12-23 14:45:00 | 847.70 | 2025-12-23 15:15:00 | 857.00 | STOP_HIT | 1.00 | -1.10% |
| BUY | retest2 | 2026-01-01 12:30:00 | 848.95 | 2026-01-06 10:15:00 | 839.00 | STOP_HIT | 1.00 | -1.17% |
| BUY | retest2 | 2026-01-06 09:45:00 | 850.30 | 2026-01-06 10:15:00 | 839.00 | STOP_HIT | 1.00 | -1.33% |
| SELL | retest2 | 2026-01-14 14:30:00 | 804.80 | 2026-01-16 09:15:00 | 816.90 | STOP_HIT | 1.00 | -1.50% |
| SELL | retest2 | 2026-01-21 13:15:00 | 802.50 | 2026-01-22 10:15:00 | 819.10 | STOP_HIT | 1.00 | -2.07% |
| SELL | retest2 | 2026-01-22 09:30:00 | 801.40 | 2026-01-22 10:15:00 | 819.10 | STOP_HIT | 1.00 | -2.21% |
| BUY | retest2 | 2026-02-12 12:15:00 | 837.85 | 2026-02-16 10:15:00 | 817.05 | STOP_HIT | 1.00 | -2.48% |
| BUY | retest2 | 2026-02-13 12:00:00 | 833.50 | 2026-02-16 10:15:00 | 817.05 | STOP_HIT | 1.00 | -1.97% |
| BUY | retest2 | 2026-02-20 09:45:00 | 857.00 | 2026-02-24 11:15:00 | 836.60 | STOP_HIT | 1.00 | -2.38% |
| BUY | retest2 | 2026-02-23 09:15:00 | 850.00 | 2026-02-24 11:15:00 | 836.60 | STOP_HIT | 1.00 | -1.58% |
| BUY | retest2 | 2026-02-23 10:45:00 | 849.35 | 2026-02-24 11:15:00 | 836.60 | STOP_HIT | 1.00 | -1.50% |
| SELL | retest2 | 2026-02-25 11:15:00 | 837.20 | 2026-03-02 09:15:00 | 795.34 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-25 13:30:00 | 837.30 | 2026-03-02 09:15:00 | 795.43 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-26 14:45:00 | 838.00 | 2026-03-02 09:15:00 | 796.10 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-25 11:15:00 | 837.20 | 2026-03-02 15:15:00 | 813.00 | STOP_HIT | 0.50 | 2.89% |
| SELL | retest2 | 2026-02-25 13:30:00 | 837.30 | 2026-03-02 15:15:00 | 813.00 | STOP_HIT | 0.50 | 2.90% |
| SELL | retest2 | 2026-02-26 14:45:00 | 838.00 | 2026-03-02 15:15:00 | 813.00 | STOP_HIT | 0.50 | 2.98% |
| SELL | retest2 | 2026-03-10 12:15:00 | 800.85 | 2026-03-11 09:15:00 | 806.50 | STOP_HIT | 1.00 | -0.71% |
| SELL | retest2 | 2026-03-13 09:15:00 | 773.45 | 2026-03-18 13:15:00 | 770.20 | STOP_HIT | 1.00 | 0.42% |
| SELL | retest2 | 2026-03-20 10:45:00 | 749.80 | 2026-03-24 11:15:00 | 760.40 | STOP_HIT | 1.00 | -1.41% |
| SELL | retest2 | 2026-03-23 09:15:00 | 747.80 | 2026-03-24 11:15:00 | 760.40 | STOP_HIT | 1.00 | -1.68% |
| SELL | retest2 | 2026-03-23 12:15:00 | 748.65 | 2026-03-24 11:15:00 | 760.40 | STOP_HIT | 1.00 | -1.57% |
| BUY | retest2 | 2026-03-25 09:15:00 | 781.50 | 2026-03-30 10:15:00 | 765.80 | STOP_HIT | 1.00 | -2.01% |
| BUY | retest2 | 2026-03-27 13:15:00 | 768.40 | 2026-03-30 10:15:00 | 765.80 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest2 | 2026-03-27 14:15:00 | 766.15 | 2026-03-30 10:15:00 | 765.80 | STOP_HIT | 1.00 | -0.05% |
| BUY | retest2 | 2026-04-07 10:15:00 | 835.35 | 2026-04-10 14:15:00 | 918.89 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-07 11:15:00 | 838.55 | 2026-04-17 10:15:00 | 922.40 | TARGET_HIT | 1.00 | 10.00% |
