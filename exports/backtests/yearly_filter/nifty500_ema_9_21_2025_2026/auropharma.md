# Aurobindo Pharma Ltd. (AUROPHARMA)

## Backtest Summary

- **Window:** 2025-03-13 09:15:00 → 2026-05-08 15:15:00 (1976 bars)
- **Last close:** 1487.70
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
| ALERT2_SKIP | 27 |
| ALERT3 | 149 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 79 |
| PARTIAL | 4 |
| TARGET_HIT | 0 |
| STOP_HIT | 79 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 83 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 34 / 49
- **Target hits / Stop hits / Partials:** 0 / 79 / 4
- **Avg / median % per leg:** 0.15% / -0.46%
- **Sum % (uncompounded):** 12.76%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 44 | 15 | 34.1% | 0 | 44 | 0 | -0.35% | -15.3% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 44 | 15 | 34.1% | 0 | 44 | 0 | -0.35% | -15.3% |
| SELL (all) | 39 | 19 | 48.7% | 0 | 35 | 4 | 0.72% | 28.1% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 39 | 19 | 48.7% | 0 | 35 | 4 | 0.72% | 28.1% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 83 | 34 | 41.0% | 0 | 79 | 4 | 0.15% | 12.8% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 12:15:00 | 1205.90 | 1180.73 | 1179.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-13 09:15:00 | 1243.00 | 1202.40 | 1190.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-14 10:15:00 | 1216.40 | 1220.34 | 1209.47 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-14 11:00:00 | 1216.40 | 1220.34 | 1209.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-14 12:15:00 | 1216.60 | 1219.14 | 1210.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-14 12:30:00 | 1213.20 | 1219.14 | 1210.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-14 13:15:00 | 1212.50 | 1217.81 | 1210.93 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-14 14:45:00 | 1217.10 | 1218.35 | 1211.80 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-15 09:45:00 | 1219.00 | 1219.44 | 1213.47 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-15 11:15:00 | 1217.80 | 1218.15 | 1213.43 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-15 11:45:00 | 1218.30 | 1217.92 | 1213.75 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-15 14:15:00 | 1216.90 | 1218.98 | 1215.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-15 15:00:00 | 1216.90 | 1218.98 | 1215.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-15 15:15:00 | 1220.60 | 1219.30 | 1215.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-16 09:15:00 | 1215.50 | 1219.30 | 1215.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-16 09:15:00 | 1212.10 | 1217.86 | 1215.51 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-05-16 11:15:00 | 1206.70 | 1214.47 | 1214.30 | SL hit (close<static) qty=1.00 sl=1208.70 alert=retest2 |

### Cycle 2 — SELL (started 2025-05-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-16 12:15:00 | 1201.00 | 1211.77 | 1213.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-20 13:15:00 | 1197.60 | 1206.78 | 1209.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-21 09:15:00 | 1219.00 | 1205.44 | 1207.48 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-21 09:15:00 | 1219.00 | 1205.44 | 1207.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 09:15:00 | 1219.00 | 1205.44 | 1207.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-21 09:30:00 | 1217.30 | 1205.44 | 1207.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 10:15:00 | 1215.00 | 1207.35 | 1208.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-21 10:30:00 | 1219.70 | 1207.35 | 1208.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 09:15:00 | 1194.40 | 1203.23 | 1205.79 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-22 11:30:00 | 1188.10 | 1197.33 | 1202.53 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-22 15:15:00 | 1187.10 | 1191.10 | 1197.95 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-23 12:45:00 | 1184.40 | 1190.21 | 1194.93 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-23 13:30:00 | 1185.90 | 1189.39 | 1194.13 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 14:15:00 | 1196.70 | 1190.85 | 1194.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-23 15:00:00 | 1196.70 | 1190.85 | 1194.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 15:15:00 | 1191.30 | 1190.94 | 1194.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-26 09:15:00 | 1201.40 | 1190.94 | 1194.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 09:15:00 | 1194.00 | 1191.55 | 1194.08 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-26 10:15:00 | 1193.50 | 1191.55 | 1194.08 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-26 11:30:00 | 1193.60 | 1190.73 | 1193.29 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-27 10:00:00 | 1193.40 | 1188.02 | 1190.59 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-27 10:30:00 | 1193.50 | 1187.39 | 1190.07 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 12:15:00 | 1186.50 | 1185.65 | 1188.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-27 12:30:00 | 1190.20 | 1185.65 | 1188.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 13:15:00 | 1185.40 | 1185.60 | 1188.41 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-28 09:15:00 | 1167.50 | 1187.11 | 1188.67 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-02 09:15:00 | 1133.83 | 1150.99 | 1159.25 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-02 09:15:00 | 1133.92 | 1150.99 | 1159.25 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-02 09:15:00 | 1133.73 | 1150.99 | 1159.25 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-02 09:15:00 | 1133.83 | 1150.99 | 1159.25 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-03 10:15:00 | 1145.40 | 1141.05 | 1148.26 | SL hit (close>ema200) qty=0.50 sl=1141.05 alert=retest2 |

### Cycle 3 — BUY (started 2025-06-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-05 10:15:00 | 1156.70 | 1146.26 | 1145.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-05 14:15:00 | 1159.40 | 1152.99 | 1149.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-06 10:15:00 | 1153.60 | 1154.61 | 1151.38 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-06 10:15:00 | 1153.60 | 1154.61 | 1151.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 10:15:00 | 1153.60 | 1154.61 | 1151.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-06 10:45:00 | 1152.20 | 1154.61 | 1151.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 12:15:00 | 1160.00 | 1155.70 | 1152.43 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-06 14:00:00 | 1163.50 | 1157.26 | 1153.44 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-06 15:15:00 | 1161.10 | 1158.01 | 1154.13 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-10 09:45:00 | 1162.70 | 1162.62 | 1159.68 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-10 11:00:00 | 1164.90 | 1163.08 | 1160.15 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 10:15:00 | 1165.20 | 1167.58 | 1164.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-11 11:00:00 | 1165.20 | 1167.58 | 1164.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 11:15:00 | 1159.80 | 1166.02 | 1164.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-11 11:30:00 | 1159.10 | 1166.02 | 1164.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 12:15:00 | 1163.30 | 1165.48 | 1164.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-11 12:45:00 | 1163.20 | 1165.48 | 1164.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 13:15:00 | 1157.20 | 1163.82 | 1163.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-11 14:00:00 | 1157.20 | 1163.82 | 1163.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-06-11 14:15:00 | 1160.10 | 1163.08 | 1163.10 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 4 — SELL (started 2025-06-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-11 14:15:00 | 1160.10 | 1163.08 | 1163.10 | EMA200 below EMA400 |

### Cycle 5 — BUY (started 2025-06-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-11 15:15:00 | 1166.80 | 1163.82 | 1163.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-12 09:15:00 | 1174.10 | 1165.88 | 1164.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-12 11:15:00 | 1165.30 | 1166.37 | 1164.92 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-12 11:15:00 | 1165.30 | 1166.37 | 1164.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 11:15:00 | 1165.30 | 1166.37 | 1164.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-12 12:00:00 | 1165.30 | 1166.37 | 1164.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 12:15:00 | 1163.50 | 1165.80 | 1164.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-12 13:15:00 | 1162.00 | 1165.80 | 1164.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 6 — SELL (started 2025-06-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-12 13:15:00 | 1153.60 | 1163.36 | 1163.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-13 09:15:00 | 1145.70 | 1157.28 | 1160.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-13 14:15:00 | 1151.00 | 1149.40 | 1154.64 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-13 14:30:00 | 1149.60 | 1149.40 | 1154.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 09:15:00 | 1142.00 | 1147.68 | 1152.94 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-17 10:00:00 | 1134.50 | 1144.13 | 1148.59 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-24 09:15:00 | 1112.30 | 1100.89 | 1100.40 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 7 — BUY (started 2025-06-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-24 09:15:00 | 1112.30 | 1100.89 | 1100.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-25 12:15:00 | 1117.70 | 1110.40 | 1106.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-27 12:15:00 | 1125.50 | 1127.15 | 1121.00 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-27 13:00:00 | 1125.50 | 1127.15 | 1121.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 14:15:00 | 1121.30 | 1125.41 | 1121.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-27 15:00:00 | 1121.30 | 1125.41 | 1121.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 15:15:00 | 1125.00 | 1125.33 | 1121.59 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-30 09:15:00 | 1153.60 | 1125.33 | 1121.59 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-01 09:15:00 | 1119.00 | 1130.28 | 1127.55 | SL hit (close<static) qty=1.00 sl=1120.60 alert=retest2 |

### Cycle 8 — SELL (started 2025-07-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-01 11:15:00 | 1119.50 | 1125.70 | 1125.79 | EMA200 below EMA400 |

### Cycle 9 — BUY (started 2025-07-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-01 14:15:00 | 1131.90 | 1126.83 | 1126.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-01 15:15:00 | 1136.20 | 1128.71 | 1127.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-03 10:15:00 | 1146.30 | 1151.39 | 1143.35 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-03 10:45:00 | 1145.00 | 1151.39 | 1143.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 09:15:00 | 1147.50 | 1175.99 | 1172.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-08 10:00:00 | 1147.50 | 1175.99 | 1172.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 10 — SELL (started 2025-07-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-08 10:15:00 | 1136.80 | 1168.15 | 1169.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-10 10:15:00 | 1131.60 | 1141.55 | 1149.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-11 13:15:00 | 1133.10 | 1129.62 | 1136.41 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-11 14:00:00 | 1133.10 | 1129.62 | 1136.41 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 09:15:00 | 1137.70 | 1129.94 | 1134.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-14 09:45:00 | 1140.40 | 1129.94 | 1134.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 10:15:00 | 1139.00 | 1131.75 | 1135.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-14 11:00:00 | 1139.00 | 1131.75 | 1135.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 12:15:00 | 1141.50 | 1134.70 | 1135.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-14 12:45:00 | 1143.20 | 1134.70 | 1135.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 11 — BUY (started 2025-07-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-14 15:15:00 | 1138.00 | 1136.80 | 1136.74 | EMA200 above EMA400 |

### Cycle 12 — SELL (started 2025-07-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-15 09:15:00 | 1135.70 | 1136.58 | 1136.65 | EMA200 below EMA400 |

### Cycle 13 — BUY (started 2025-07-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-15 10:15:00 | 1138.90 | 1137.04 | 1136.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-15 11:15:00 | 1152.90 | 1140.21 | 1138.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-17 13:15:00 | 1153.90 | 1155.59 | 1151.28 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-17 13:45:00 | 1155.00 | 1155.59 | 1151.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 09:15:00 | 1148.00 | 1153.61 | 1151.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-18 09:45:00 | 1147.40 | 1153.61 | 1151.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 10:15:00 | 1144.20 | 1151.73 | 1150.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-18 10:30:00 | 1145.40 | 1151.73 | 1150.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 14 — SELL (started 2025-07-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-18 12:15:00 | 1143.70 | 1149.41 | 1149.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-18 13:15:00 | 1141.10 | 1147.75 | 1149.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-21 11:15:00 | 1145.70 | 1144.15 | 1146.38 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-21 11:15:00 | 1145.70 | 1144.15 | 1146.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 11:15:00 | 1145.70 | 1144.15 | 1146.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-21 12:00:00 | 1145.70 | 1144.15 | 1146.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 12:15:00 | 1144.90 | 1144.30 | 1146.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-21 12:30:00 | 1148.20 | 1144.30 | 1146.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 09:15:00 | 1117.50 | 1115.43 | 1120.55 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-24 10:15:00 | 1112.00 | 1115.43 | 1120.55 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-24 14:15:00 | 1115.90 | 1118.66 | 1120.62 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-25 12:00:00 | 1112.50 | 1117.65 | 1119.30 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-25 14:15:00 | 1125.40 | 1120.80 | 1120.42 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 15 — BUY (started 2025-07-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-25 14:15:00 | 1125.40 | 1120.80 | 1120.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-28 09:15:00 | 1132.20 | 1122.98 | 1121.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-28 15:15:00 | 1124.90 | 1125.12 | 1123.38 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-29 09:15:00 | 1133.90 | 1125.12 | 1123.38 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 09:15:00 | 1137.10 | 1127.52 | 1124.63 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-29 11:15:00 | 1146.50 | 1130.30 | 1126.16 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-01 09:15:00 | 1103.20 | 1139.98 | 1144.05 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 16 — SELL (started 2025-08-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-01 09:15:00 | 1103.20 | 1139.98 | 1144.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-01 13:15:00 | 1085.50 | 1110.45 | 1127.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-04 12:15:00 | 1094.90 | 1091.42 | 1108.26 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-04 13:00:00 | 1094.90 | 1091.42 | 1108.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 09:15:00 | 1076.40 | 1088.60 | 1101.73 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-05 11:30:00 | 1066.40 | 1081.89 | 1096.27 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-05 14:15:00 | 1066.40 | 1078.34 | 1092.06 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-07 10:15:00 | 1066.30 | 1072.35 | 1079.93 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-08 09:15:00 | 1061.10 | 1065.30 | 1071.90 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 14:15:00 | 1052.00 | 1049.93 | 1055.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-11 14:45:00 | 1055.50 | 1049.93 | 1055.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 09:15:00 | 1056.10 | 1051.26 | 1055.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-12 10:00:00 | 1056.10 | 1051.26 | 1055.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 10:15:00 | 1054.70 | 1051.94 | 1055.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-12 10:30:00 | 1056.20 | 1051.94 | 1055.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 11:15:00 | 1059.60 | 1053.48 | 1055.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-12 11:45:00 | 1059.80 | 1053.48 | 1055.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 12:15:00 | 1063.60 | 1055.50 | 1056.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-12 13:00:00 | 1063.60 | 1055.50 | 1056.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 15:15:00 | 1055.80 | 1056.65 | 1056.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-13 09:15:00 | 1058.90 | 1056.65 | 1056.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-08-13 09:15:00 | 1060.50 | 1057.42 | 1057.12 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 17 — BUY (started 2025-08-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-13 09:15:00 | 1060.50 | 1057.42 | 1057.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-13 11:15:00 | 1069.50 | 1060.49 | 1058.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-18 09:15:00 | 1079.00 | 1080.72 | 1074.67 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-18 10:00:00 | 1079.00 | 1080.72 | 1074.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 11:15:00 | 1072.20 | 1078.28 | 1074.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-18 11:45:00 | 1072.40 | 1078.28 | 1074.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 12:15:00 | 1084.00 | 1079.42 | 1075.42 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-18 13:15:00 | 1086.90 | 1079.42 | 1075.42 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-20 09:15:00 | 1049.80 | 1082.22 | 1082.35 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 18 — SELL (started 2025-08-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-20 09:15:00 | 1049.80 | 1082.22 | 1082.35 | EMA200 below EMA400 |

### Cycle 19 — BUY (started 2025-08-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-25 11:15:00 | 1064.00 | 1056.28 | 1056.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-25 12:15:00 | 1068.60 | 1058.74 | 1057.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-26 09:15:00 | 1051.30 | 1061.25 | 1059.36 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-26 09:15:00 | 1051.30 | 1061.25 | 1059.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 09:15:00 | 1051.30 | 1061.25 | 1059.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-26 10:00:00 | 1051.30 | 1061.25 | 1059.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 10:15:00 | 1052.20 | 1059.44 | 1058.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-26 10:45:00 | 1051.50 | 1059.44 | 1058.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 20 — SELL (started 2025-08-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-26 11:15:00 | 1051.70 | 1057.89 | 1058.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-26 12:15:00 | 1049.00 | 1056.11 | 1057.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-29 10:15:00 | 1032.80 | 1032.12 | 1040.03 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-29 11:00:00 | 1032.80 | 1032.12 | 1040.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 13:15:00 | 1033.10 | 1033.39 | 1038.75 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-29 14:15:00 | 1031.10 | 1033.39 | 1038.75 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-03 12:15:00 | 1036.70 | 1030.01 | 1029.60 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 21 — BUY (started 2025-09-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-03 12:15:00 | 1036.70 | 1030.01 | 1029.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-03 13:15:00 | 1039.40 | 1031.89 | 1030.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-04 14:15:00 | 1034.70 | 1037.76 | 1035.09 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-04 14:15:00 | 1034.70 | 1037.76 | 1035.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 14:15:00 | 1034.70 | 1037.76 | 1035.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 15:00:00 | 1034.70 | 1037.76 | 1035.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 15:15:00 | 1034.80 | 1037.17 | 1035.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-05 09:15:00 | 1040.00 | 1037.17 | 1035.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 09:15:00 | 1049.10 | 1039.56 | 1036.34 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-10 09:15:00 | 1058.20 | 1044.63 | 1043.20 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-11 09:15:00 | 1080.20 | 1051.19 | 1048.46 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-23 12:15:00 | 1103.90 | 1114.58 | 1115.16 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 22 — SELL (started 2025-09-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-23 12:15:00 | 1103.90 | 1114.58 | 1115.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-24 09:15:00 | 1097.00 | 1108.49 | 1111.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-25 09:15:00 | 1102.90 | 1099.28 | 1104.47 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-25 09:15:00 | 1102.90 | 1099.28 | 1104.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 09:15:00 | 1102.90 | 1099.28 | 1104.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-25 10:00:00 | 1102.90 | 1099.28 | 1104.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 10:15:00 | 1106.00 | 1100.62 | 1104.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-25 10:30:00 | 1101.60 | 1100.62 | 1104.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 11:15:00 | 1103.90 | 1101.28 | 1104.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-25 12:15:00 | 1107.30 | 1101.28 | 1104.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 12:15:00 | 1102.20 | 1101.46 | 1104.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-25 12:30:00 | 1105.20 | 1101.46 | 1104.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 09:15:00 | 1092.20 | 1090.90 | 1095.14 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-29 11:30:00 | 1081.10 | 1089.31 | 1093.70 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-30 12:15:00 | 1082.40 | 1088.64 | 1091.41 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-30 13:15:00 | 1082.00 | 1087.59 | 1090.68 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-03 09:15:00 | 1099.60 | 1090.67 | 1090.24 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 23 — BUY (started 2025-10-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-03 09:15:00 | 1099.60 | 1090.67 | 1090.24 | EMA200 above EMA400 |

### Cycle 24 — SELL (started 2025-10-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-06 09:15:00 | 1079.80 | 1090.40 | 1090.80 | EMA200 below EMA400 |

### Cycle 25 — BUY (started 2025-10-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-06 14:15:00 | 1097.00 | 1090.79 | 1090.59 | EMA200 above EMA400 |

### Cycle 26 — SELL (started 2025-10-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-07 11:15:00 | 1082.70 | 1089.84 | 1090.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-08 14:15:00 | 1070.90 | 1081.57 | 1085.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-09 09:15:00 | 1115.90 | 1086.35 | 1086.84 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-09 09:15:00 | 1115.90 | 1086.35 | 1086.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 09:15:00 | 1115.90 | 1086.35 | 1086.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-09 09:30:00 | 1112.00 | 1086.35 | 1086.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 27 — BUY (started 2025-10-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-09 10:15:00 | 1113.00 | 1091.68 | 1089.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-09 14:15:00 | 1122.00 | 1104.76 | 1096.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-13 12:15:00 | 1126.40 | 1127.52 | 1119.15 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-13 13:00:00 | 1126.40 | 1127.52 | 1119.15 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 09:15:00 | 1122.30 | 1128.85 | 1122.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-14 10:00:00 | 1122.30 | 1128.85 | 1122.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 10:15:00 | 1118.00 | 1126.68 | 1122.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-14 11:00:00 | 1118.00 | 1126.68 | 1122.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 11:15:00 | 1112.10 | 1123.76 | 1121.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-14 11:30:00 | 1112.90 | 1123.76 | 1121.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 15:15:00 | 1122.10 | 1121.76 | 1120.88 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-15 09:15:00 | 1126.70 | 1121.76 | 1120.88 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-15 10:45:00 | 1125.70 | 1123.60 | 1121.92 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-15 11:15:00 | 1117.30 | 1122.34 | 1121.50 | SL hit (close<static) qty=1.00 sl=1120.00 alert=retest2 |

### Cycle 28 — SELL (started 2025-10-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-15 12:15:00 | 1114.40 | 1120.75 | 1120.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-15 13:15:00 | 1111.10 | 1118.82 | 1119.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-16 14:15:00 | 1108.30 | 1107.94 | 1112.30 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-16 15:00:00 | 1108.30 | 1107.94 | 1112.30 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 12:15:00 | 1098.70 | 1103.83 | 1108.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-17 12:30:00 | 1107.10 | 1103.83 | 1108.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 14:15:00 | 1100.50 | 1103.06 | 1107.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-17 14:30:00 | 1109.60 | 1103.06 | 1107.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 09:15:00 | 1107.90 | 1103.57 | 1106.77 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-20 12:45:00 | 1103.50 | 1104.72 | 1106.56 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-20 13:30:00 | 1103.80 | 1105.02 | 1106.53 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-20 14:15:00 | 1102.80 | 1105.02 | 1106.53 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-23 11:15:00 | 1108.60 | 1106.35 | 1106.14 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 29 — BUY (started 2025-10-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-23 11:15:00 | 1108.60 | 1106.35 | 1106.14 | EMA200 above EMA400 |

### Cycle 30 — SELL (started 2025-10-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-23 14:15:00 | 1097.50 | 1104.38 | 1105.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-24 12:15:00 | 1088.30 | 1098.50 | 1101.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-27 13:15:00 | 1091.90 | 1090.97 | 1095.05 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-27 13:15:00 | 1091.90 | 1090.97 | 1095.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 13:15:00 | 1091.90 | 1090.97 | 1095.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-27 13:45:00 | 1095.00 | 1090.97 | 1095.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 14:15:00 | 1094.20 | 1091.62 | 1094.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-27 15:00:00 | 1094.20 | 1091.62 | 1094.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 15:15:00 | 1093.00 | 1091.90 | 1094.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-28 09:15:00 | 1095.00 | 1091.90 | 1094.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 09:15:00 | 1095.00 | 1092.52 | 1094.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-28 09:30:00 | 1096.10 | 1092.52 | 1094.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 10:15:00 | 1089.00 | 1091.81 | 1094.28 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-28 11:30:00 | 1087.10 | 1090.37 | 1093.40 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-28 13:00:00 | 1087.90 | 1089.88 | 1092.90 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-28 13:30:00 | 1087.90 | 1089.04 | 1092.25 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-28 14:15:00 | 1097.80 | 1090.79 | 1092.75 | SL hit (close>static) qty=1.00 sl=1096.00 alert=retest2 |

### Cycle 31 — BUY (started 2025-10-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-29 09:15:00 | 1109.90 | 1096.65 | 1095.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-29 10:15:00 | 1116.80 | 1100.68 | 1097.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-30 09:15:00 | 1097.80 | 1106.24 | 1102.46 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-30 09:15:00 | 1097.80 | 1106.24 | 1102.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 09:15:00 | 1097.80 | 1106.24 | 1102.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-30 10:00:00 | 1097.80 | 1106.24 | 1102.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 10:15:00 | 1103.20 | 1105.63 | 1102.52 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-30 11:15:00 | 1105.70 | 1105.63 | 1102.52 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-30 12:30:00 | 1105.10 | 1105.74 | 1103.10 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-30 14:30:00 | 1105.00 | 1104.96 | 1103.19 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-30 15:15:00 | 1105.00 | 1104.96 | 1103.19 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 15:15:00 | 1105.00 | 1104.97 | 1103.35 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-31 09:15:00 | 1116.00 | 1104.97 | 1103.35 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-06 15:15:00 | 1135.00 | 1139.18 | 1139.34 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 32 — SELL (started 2025-11-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-06 15:15:00 | 1135.00 | 1139.18 | 1139.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-07 09:15:00 | 1131.70 | 1137.68 | 1138.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-10 09:15:00 | 1138.00 | 1129.79 | 1133.10 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-10 09:15:00 | 1138.00 | 1129.79 | 1133.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 09:15:00 | 1138.00 | 1129.79 | 1133.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-10 10:00:00 | 1138.00 | 1129.79 | 1133.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 10:15:00 | 1135.40 | 1130.92 | 1133.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-10 10:30:00 | 1135.60 | 1130.92 | 1133.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 33 — BUY (started 2025-11-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-10 12:15:00 | 1142.80 | 1136.03 | 1135.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-10 14:15:00 | 1154.70 | 1141.68 | 1138.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-18 15:15:00 | 1236.90 | 1238.84 | 1229.41 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-19 09:15:00 | 1231.40 | 1238.84 | 1229.41 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 09:15:00 | 1239.00 | 1238.87 | 1230.28 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-19 12:00:00 | 1242.50 | 1238.61 | 1231.59 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-20 09:15:00 | 1223.00 | 1234.33 | 1232.22 | SL hit (close<static) qty=1.00 sl=1228.70 alert=retest2 |

### Cycle 34 — SELL (started 2025-11-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-20 11:15:00 | 1224.30 | 1230.61 | 1230.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-20 12:15:00 | 1217.50 | 1227.98 | 1229.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-21 12:15:00 | 1216.20 | 1215.00 | 1220.64 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-21 13:00:00 | 1216.20 | 1215.00 | 1220.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 09:15:00 | 1212.70 | 1204.80 | 1210.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-25 10:00:00 | 1212.70 | 1204.80 | 1210.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 10:15:00 | 1209.20 | 1205.68 | 1210.02 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-25 15:15:00 | 1205.00 | 1208.88 | 1210.44 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-26 09:45:00 | 1202.10 | 1207.27 | 1209.40 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-26 10:15:00 | 1216.00 | 1209.01 | 1210.00 | SL hit (close>static) qty=1.00 sl=1212.90 alert=retest2 |

### Cycle 35 — BUY (started 2025-11-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-26 11:15:00 | 1218.60 | 1210.93 | 1210.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-26 13:15:00 | 1220.50 | 1213.77 | 1212.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-28 09:15:00 | 1228.70 | 1230.65 | 1224.67 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-28 09:45:00 | 1229.00 | 1230.65 | 1224.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 12:15:00 | 1227.90 | 1230.20 | 1225.97 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-28 13:45:00 | 1238.20 | 1230.58 | 1226.52 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-28 14:30:00 | 1231.60 | 1230.29 | 1226.76 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-28 15:15:00 | 1219.10 | 1228.05 | 1226.06 | SL hit (close<static) qty=1.00 sl=1224.30 alert=retest2 |

### Cycle 36 — SELL (started 2025-12-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-01 10:15:00 | 1214.40 | 1222.72 | 1223.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-01 11:15:00 | 1210.40 | 1220.26 | 1222.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-01 14:15:00 | 1218.30 | 1216.64 | 1220.10 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-01 15:00:00 | 1218.30 | 1216.64 | 1220.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 09:15:00 | 1208.10 | 1214.38 | 1218.44 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-03 09:30:00 | 1204.60 | 1210.82 | 1214.73 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-04 12:15:00 | 1221.30 | 1214.22 | 1213.99 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 37 — BUY (started 2025-12-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-04 12:15:00 | 1221.30 | 1214.22 | 1213.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-04 13:15:00 | 1222.50 | 1215.87 | 1214.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-05 09:15:00 | 1217.50 | 1217.91 | 1216.15 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-05 09:15:00 | 1217.50 | 1217.91 | 1216.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 09:15:00 | 1217.50 | 1217.91 | 1216.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-05 10:00:00 | 1217.50 | 1217.91 | 1216.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 10:15:00 | 1222.90 | 1218.91 | 1216.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-05 10:30:00 | 1215.20 | 1218.91 | 1216.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 13:15:00 | 1216.40 | 1218.87 | 1217.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-05 14:00:00 | 1216.40 | 1218.87 | 1217.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 14:15:00 | 1217.40 | 1218.58 | 1217.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-05 14:45:00 | 1214.40 | 1218.58 | 1217.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 15:15:00 | 1217.90 | 1218.44 | 1217.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-08 09:15:00 | 1213.80 | 1218.44 | 1217.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 38 — SELL (started 2025-12-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-08 09:15:00 | 1204.00 | 1215.55 | 1216.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-08 11:15:00 | 1194.00 | 1209.71 | 1213.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-10 09:15:00 | 1182.50 | 1173.92 | 1185.22 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-10 10:00:00 | 1182.50 | 1173.92 | 1185.22 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 10:15:00 | 1176.60 | 1173.62 | 1178.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-11 10:30:00 | 1182.30 | 1173.62 | 1178.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 11:15:00 | 1175.70 | 1174.04 | 1178.47 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-11 12:45:00 | 1174.10 | 1173.41 | 1177.78 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-12 09:15:00 | 1181.30 | 1174.79 | 1176.96 | SL hit (close>static) qty=1.00 sl=1180.70 alert=retest2 |

### Cycle 39 — BUY (started 2025-12-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-12 11:15:00 | 1191.80 | 1179.11 | 1178.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-12 12:15:00 | 1194.50 | 1182.18 | 1180.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-15 09:15:00 | 1183.50 | 1187.84 | 1183.99 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-15 09:15:00 | 1183.50 | 1187.84 | 1183.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 09:15:00 | 1183.50 | 1187.84 | 1183.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-15 09:45:00 | 1186.50 | 1187.84 | 1183.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 10:15:00 | 1186.00 | 1187.47 | 1184.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-15 10:30:00 | 1184.20 | 1187.47 | 1184.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 11:15:00 | 1183.80 | 1186.73 | 1184.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-15 12:00:00 | 1183.80 | 1186.73 | 1184.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 12:15:00 | 1185.00 | 1186.39 | 1184.22 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-15 13:30:00 | 1188.50 | 1186.61 | 1184.51 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-15 15:00:00 | 1187.50 | 1186.79 | 1184.79 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-15 15:15:00 | 1183.40 | 1186.11 | 1184.66 | SL hit (close<static) qty=1.00 sl=1184.00 alert=retest2 |

### Cycle 40 — SELL (started 2025-12-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-16 10:15:00 | 1180.80 | 1183.22 | 1183.49 | EMA200 below EMA400 |

### Cycle 41 — BUY (started 2025-12-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-17 12:15:00 | 1188.00 | 1183.42 | 1182.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-17 13:15:00 | 1192.90 | 1185.31 | 1183.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-22 14:15:00 | 1225.00 | 1226.29 | 1217.35 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-22 15:00:00 | 1225.00 | 1226.29 | 1217.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 12:15:00 | 1212.70 | 1223.80 | 1219.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-23 13:00:00 | 1212.70 | 1223.80 | 1219.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 13:15:00 | 1212.40 | 1221.52 | 1219.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-23 14:00:00 | 1212.40 | 1221.52 | 1219.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 15:15:00 | 1216.60 | 1220.20 | 1218.87 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-24 09:15:00 | 1224.90 | 1220.20 | 1218.87 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-24 12:00:00 | 1219.70 | 1220.61 | 1219.45 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-24 13:15:00 | 1213.20 | 1218.46 | 1218.63 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 42 — SELL (started 2025-12-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-24 13:15:00 | 1213.20 | 1218.46 | 1218.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-24 15:15:00 | 1210.30 | 1215.81 | 1217.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-31 12:15:00 | 1189.20 | 1187.10 | 1193.13 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-31 12:45:00 | 1191.60 | 1187.10 | 1193.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 14:15:00 | 1180.80 | 1185.74 | 1191.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 15:00:00 | 1180.80 | 1185.74 | 1191.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 09:15:00 | 1173.40 | 1183.54 | 1189.48 | EMA400 retest candle locked (from downside) |

### Cycle 43 — BUY (started 2026-01-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-02 09:15:00 | 1204.80 | 1190.48 | 1189.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-02 10:15:00 | 1208.60 | 1194.10 | 1191.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-05 13:15:00 | 1202.40 | 1209.31 | 1204.00 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-05 13:15:00 | 1202.40 | 1209.31 | 1204.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 13:15:00 | 1202.40 | 1209.31 | 1204.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-05 14:00:00 | 1202.40 | 1209.31 | 1204.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 14:15:00 | 1207.20 | 1208.89 | 1204.30 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-06 09:15:00 | 1213.20 | 1208.01 | 1204.31 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-08 13:15:00 | 1213.80 | 1225.71 | 1225.79 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 44 — SELL (started 2026-01-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-08 13:15:00 | 1213.80 | 1225.71 | 1225.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-08 14:15:00 | 1206.00 | 1221.76 | 1223.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-13 14:15:00 | 1169.80 | 1169.50 | 1181.57 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-13 15:00:00 | 1169.80 | 1169.50 | 1181.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 10:15:00 | 1178.70 | 1170.48 | 1178.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-14 10:30:00 | 1175.80 | 1170.48 | 1178.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 11:15:00 | 1186.50 | 1173.69 | 1179.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-14 12:00:00 | 1186.50 | 1173.69 | 1179.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 12:15:00 | 1183.00 | 1175.55 | 1179.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-14 12:30:00 | 1187.50 | 1175.55 | 1179.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 14:15:00 | 1180.70 | 1177.10 | 1179.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-14 14:45:00 | 1180.90 | 1177.10 | 1179.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 15:15:00 | 1180.00 | 1177.68 | 1179.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-16 09:15:00 | 1180.30 | 1177.68 | 1179.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 09:15:00 | 1188.70 | 1179.88 | 1180.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-16 10:00:00 | 1188.70 | 1179.88 | 1180.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 10:15:00 | 1186.10 | 1181.13 | 1181.18 | EMA400 retest candle locked (from downside) |

### Cycle 45 — BUY (started 2026-01-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-16 11:15:00 | 1183.00 | 1181.50 | 1181.34 | EMA200 above EMA400 |

### Cycle 46 — SELL (started 2026-01-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-16 12:15:00 | 1176.50 | 1180.50 | 1180.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-16 13:15:00 | 1168.60 | 1178.12 | 1179.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-19 12:15:00 | 1172.40 | 1171.43 | 1175.04 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-19 12:15:00 | 1172.40 | 1171.43 | 1175.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 12:15:00 | 1172.40 | 1171.43 | 1175.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-19 12:45:00 | 1175.10 | 1171.43 | 1175.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 13:15:00 | 1165.00 | 1170.15 | 1174.13 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-19 15:15:00 | 1164.10 | 1169.88 | 1173.64 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-28 15:15:00 | 1142.60 | 1134.20 | 1133.72 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 47 — BUY (started 2026-01-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-28 15:15:00 | 1142.60 | 1134.20 | 1133.72 | EMA200 above EMA400 |

### Cycle 48 — SELL (started 2026-01-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-29 09:15:00 | 1129.10 | 1133.18 | 1133.30 | EMA200 below EMA400 |

### Cycle 49 — BUY (started 2026-01-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-29 10:15:00 | 1138.20 | 1134.18 | 1133.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-29 11:15:00 | 1139.10 | 1135.17 | 1134.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-01 13:15:00 | 1183.00 | 1186.23 | 1173.38 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-01 14:00:00 | 1183.00 | 1186.23 | 1173.38 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 14:15:00 | 1174.50 | 1183.89 | 1173.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 14:45:00 | 1174.00 | 1183.89 | 1173.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 15:15:00 | 1164.80 | 1180.07 | 1172.70 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-02 09:45:00 | 1178.70 | 1179.98 | 1173.32 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-02 10:15:00 | 1157.90 | 1175.56 | 1171.92 | SL hit (close<static) qty=1.00 sl=1163.10 alert=retest2 |

### Cycle 50 — SELL (started 2026-02-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-02 12:15:00 | 1158.60 | 1169.26 | 1169.51 | EMA200 below EMA400 |

### Cycle 51 — BUY (started 2026-02-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-02 15:15:00 | 1178.00 | 1170.07 | 1169.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-03 09:15:00 | 1228.60 | 1181.77 | 1175.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-04 10:15:00 | 1180.40 | 1212.68 | 1200.80 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-04 10:15:00 | 1180.40 | 1212.68 | 1200.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 10:15:00 | 1180.40 | 1212.68 | 1200.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-04 10:45:00 | 1182.80 | 1212.68 | 1200.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 11:15:00 | 1209.10 | 1211.96 | 1201.56 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-04 13:15:00 | 1211.50 | 1211.73 | 1202.40 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-04 14:00:00 | 1212.10 | 1211.80 | 1203.28 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-04 15:00:00 | 1214.70 | 1212.38 | 1204.32 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-05 11:15:00 | 1183.30 | 1199.22 | 1200.24 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 52 — SELL (started 2026-02-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-05 11:15:00 | 1183.30 | 1199.22 | 1200.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-05 14:15:00 | 1180.30 | 1192.51 | 1196.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-06 13:15:00 | 1184.90 | 1179.86 | 1187.13 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-06 13:15:00 | 1184.90 | 1179.86 | 1187.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 13:15:00 | 1184.90 | 1179.86 | 1187.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-06 14:00:00 | 1184.90 | 1179.86 | 1187.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 14:15:00 | 1191.20 | 1182.13 | 1187.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-06 15:00:00 | 1191.20 | 1182.13 | 1187.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 15:15:00 | 1195.00 | 1184.70 | 1188.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-09 09:30:00 | 1185.20 | 1185.62 | 1188.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 10:15:00 | 1195.60 | 1187.62 | 1188.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-09 11:00:00 | 1195.60 | 1187.62 | 1188.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 53 — BUY (started 2026-02-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-09 11:15:00 | 1203.30 | 1190.75 | 1190.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-09 12:15:00 | 1206.00 | 1193.80 | 1191.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-10 09:15:00 | 1197.80 | 1198.14 | 1194.77 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-10 09:15:00 | 1197.80 | 1198.14 | 1194.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 09:15:00 | 1197.80 | 1198.14 | 1194.77 | EMA400 retest candle locked (from upside) |

### Cycle 54 — SELL (started 2026-02-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-10 12:15:00 | 1146.80 | 1184.24 | 1189.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-10 13:15:00 | 1125.00 | 1172.39 | 1183.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-12 12:15:00 | 1145.50 | 1144.77 | 1154.42 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-12 12:30:00 | 1146.80 | 1144.77 | 1154.42 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 14:15:00 | 1156.00 | 1147.66 | 1154.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-12 15:00:00 | 1156.00 | 1147.66 | 1154.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 15:15:00 | 1157.60 | 1149.65 | 1154.42 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-13 09:15:00 | 1151.60 | 1149.65 | 1154.42 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-16 10:15:00 | 1171.70 | 1155.06 | 1153.43 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 55 — BUY (started 2026-02-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-16 10:15:00 | 1171.70 | 1155.06 | 1153.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-16 11:15:00 | 1177.70 | 1159.59 | 1155.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-17 13:15:00 | 1182.70 | 1184.45 | 1174.55 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-17 13:15:00 | 1182.70 | 1184.45 | 1174.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 13:15:00 | 1182.70 | 1184.45 | 1174.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-17 13:30:00 | 1178.10 | 1184.45 | 1174.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 14:15:00 | 1189.80 | 1185.52 | 1175.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-17 14:45:00 | 1182.60 | 1185.52 | 1175.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 15:15:00 | 1181.00 | 1184.62 | 1176.40 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-18 09:15:00 | 1192.70 | 1184.62 | 1176.40 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-18 09:15:00 | 1169.40 | 1181.57 | 1175.76 | SL hit (close<static) qty=1.00 sl=1172.90 alert=retest2 |

### Cycle 56 — SELL (started 2026-02-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-18 11:15:00 | 1155.20 | 1170.27 | 1171.26 | EMA200 below EMA400 |

### Cycle 57 — BUY (started 2026-02-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-19 10:15:00 | 1185.80 | 1171.86 | 1170.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-19 12:15:00 | 1199.70 | 1180.11 | 1174.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-19 14:15:00 | 1173.00 | 1180.86 | 1176.28 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-19 14:15:00 | 1173.00 | 1180.86 | 1176.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 14:15:00 | 1173.00 | 1180.86 | 1176.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 15:00:00 | 1173.00 | 1180.86 | 1176.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 15:15:00 | 1172.00 | 1179.09 | 1175.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-20 09:15:00 | 1162.80 | 1179.09 | 1175.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 09:15:00 | 1162.60 | 1175.79 | 1174.69 | EMA400 retest candle locked (from upside) |

### Cycle 58 — SELL (started 2026-02-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-20 10:15:00 | 1163.10 | 1173.25 | 1173.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-20 11:15:00 | 1161.30 | 1170.86 | 1172.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-23 09:15:00 | 1164.80 | 1162.89 | 1167.21 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-23 09:15:00 | 1164.80 | 1162.89 | 1167.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 09:15:00 | 1164.80 | 1162.89 | 1167.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-23 09:30:00 | 1162.90 | 1162.89 | 1167.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 10:15:00 | 1166.70 | 1163.65 | 1167.16 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-23 11:15:00 | 1160.80 | 1163.65 | 1167.16 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-24 14:15:00 | 1162.20 | 1159.12 | 1160.74 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-25 09:15:00 | 1181.80 | 1165.31 | 1163.30 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 59 — BUY (started 2026-02-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-25 09:15:00 | 1181.80 | 1165.31 | 1163.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-25 10:15:00 | 1190.80 | 1170.41 | 1165.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-27 14:15:00 | 1219.80 | 1225.02 | 1213.94 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-27 15:00:00 | 1219.80 | 1225.02 | 1213.94 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 09:15:00 | 1217.80 | 1222.77 | 1214.79 | EMA400 retest candle locked (from upside) |

### Cycle 60 — SELL (started 2026-03-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-04 09:15:00 | 1183.90 | 1206.17 | 1208.96 | EMA200 below EMA400 |

### Cycle 61 — BUY (started 2026-03-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-05 12:15:00 | 1217.10 | 1206.44 | 1205.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-05 14:15:00 | 1222.20 | 1211.36 | 1207.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-09 09:15:00 | 1216.30 | 1224.01 | 1218.26 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-09 09:15:00 | 1216.30 | 1224.01 | 1218.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-09 09:15:00 | 1216.30 | 1224.01 | 1218.26 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-09 10:45:00 | 1219.90 | 1223.57 | 1218.58 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-16 10:15:00 | 1281.00 | 1293.92 | 1294.49 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 62 — SELL (started 2026-03-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-16 10:15:00 | 1281.00 | 1293.92 | 1294.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 09:15:00 | 1257.60 | 1270.66 | 1277.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-20 09:15:00 | 1270.50 | 1259.16 | 1266.66 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-20 09:15:00 | 1270.50 | 1259.16 | 1266.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 09:15:00 | 1270.50 | 1259.16 | 1266.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 09:45:00 | 1279.00 | 1259.16 | 1266.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 10:15:00 | 1277.70 | 1262.87 | 1267.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 10:45:00 | 1274.90 | 1262.87 | 1267.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 11:15:00 | 1283.00 | 1266.90 | 1269.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 11:45:00 | 1284.20 | 1266.90 | 1269.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 63 — BUY (started 2026-03-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-20 12:15:00 | 1287.50 | 1271.02 | 1270.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-20 13:15:00 | 1292.30 | 1275.27 | 1272.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-23 09:15:00 | 1274.00 | 1279.75 | 1275.81 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-23 09:15:00 | 1274.00 | 1279.75 | 1275.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-23 09:15:00 | 1274.00 | 1279.75 | 1275.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-23 10:00:00 | 1274.00 | 1279.75 | 1275.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-23 10:15:00 | 1279.00 | 1279.60 | 1276.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-23 10:30:00 | 1270.80 | 1279.60 | 1276.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-23 11:15:00 | 1275.30 | 1278.74 | 1276.03 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-23 14:30:00 | 1283.00 | 1278.07 | 1276.30 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-23 15:15:00 | 1284.00 | 1278.07 | 1276.30 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-24 09:15:00 | 1265.20 | 1276.44 | 1275.93 | SL hit (close<static) qty=1.00 sl=1268.30 alert=retest2 |

### Cycle 64 — SELL (started 2026-03-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-24 10:15:00 | 1258.90 | 1272.94 | 1274.38 | EMA200 below EMA400 |

### Cycle 65 — BUY (started 2026-03-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-24 14:15:00 | 1282.20 | 1276.28 | 1275.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 09:15:00 | 1310.80 | 1283.75 | 1279.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-30 09:15:00 | 1305.10 | 1312.26 | 1305.32 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-30 09:15:00 | 1305.10 | 1312.26 | 1305.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-30 09:15:00 | 1305.10 | 1312.26 | 1305.32 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-30 12:00:00 | 1318.20 | 1313.42 | 1307.05 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-01 09:15:00 | 1356.30 | 1309.52 | 1307.20 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-02 09:15:00 | 1273.70 | 1323.55 | 1320.08 | SL hit (close<static) qty=1.00 sl=1300.00 alert=retest2 |

### Cycle 66 — SELL (started 2026-04-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-08 10:15:00 | 1327.30 | 1329.61 | 1329.79 | EMA200 below EMA400 |

### Cycle 67 — BUY (started 2026-04-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-08 11:15:00 | 1337.60 | 1331.21 | 1330.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-08 13:15:00 | 1341.20 | 1334.47 | 1332.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-09 09:15:00 | 1328.40 | 1333.56 | 1332.37 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-09 09:15:00 | 1328.40 | 1333.56 | 1332.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 09:15:00 | 1328.40 | 1333.56 | 1332.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-09 09:45:00 | 1324.10 | 1333.56 | 1332.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 10:15:00 | 1327.10 | 1332.26 | 1331.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-09 10:30:00 | 1328.10 | 1332.26 | 1331.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 11:15:00 | 1343.60 | 1334.53 | 1332.95 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-10 09:15:00 | 1346.80 | 1337.57 | 1335.11 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-10 11:30:00 | 1349.70 | 1342.25 | 1338.08 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 09:45:00 | 1346.40 | 1347.22 | 1342.47 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 10:45:00 | 1348.30 | 1346.59 | 1342.62 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 11:15:00 | 1344.50 | 1346.17 | 1342.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-13 11:30:00 | 1343.60 | 1346.17 | 1342.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 12:15:00 | 1347.00 | 1346.34 | 1343.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-13 12:30:00 | 1345.40 | 1346.34 | 1343.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 14:15:00 | 1340.30 | 1345.56 | 1343.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-13 15:00:00 | 1340.30 | 1345.56 | 1343.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 15:15:00 | 1340.00 | 1344.45 | 1343.09 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-15 09:15:00 | 1353.90 | 1344.45 | 1343.09 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-20 15:15:00 | 1365.40 | 1373.20 | 1373.85 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 68 — SELL (started 2026-04-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-20 15:15:00 | 1365.40 | 1373.20 | 1373.85 | EMA200 below EMA400 |

### Cycle 69 — BUY (started 2026-04-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-21 09:15:00 | 1379.60 | 1374.48 | 1374.38 | EMA200 above EMA400 |

### Cycle 70 — SELL (started 2026-04-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-21 10:15:00 | 1371.50 | 1373.88 | 1374.11 | EMA200 below EMA400 |

### Cycle 71 — BUY (started 2026-04-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-21 12:15:00 | 1399.10 | 1378.64 | 1376.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-22 10:15:00 | 1404.20 | 1390.48 | 1383.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-24 10:15:00 | 1428.60 | 1431.13 | 1418.34 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-24 10:30:00 | 1428.00 | 1431.13 | 1418.34 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 11:15:00 | 1422.90 | 1429.49 | 1418.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-24 12:00:00 | 1422.90 | 1429.49 | 1418.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 12:15:00 | 1419.80 | 1427.55 | 1418.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-24 13:00:00 | 1419.80 | 1427.55 | 1418.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 13:15:00 | 1420.60 | 1426.16 | 1419.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-24 14:00:00 | 1420.60 | 1426.16 | 1419.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 14:15:00 | 1413.40 | 1423.61 | 1418.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-24 15:00:00 | 1413.40 | 1423.61 | 1418.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 15:15:00 | 1415.90 | 1422.07 | 1418.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-27 09:15:00 | 1409.30 | 1422.07 | 1418.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 09:15:00 | 1419.10 | 1421.47 | 1418.34 | EMA400 retest candle locked (from upside) |

### Cycle 72 — SELL (started 2026-04-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-29 10:15:00 | 1412.20 | 1418.63 | 1419.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-29 11:15:00 | 1406.50 | 1416.21 | 1418.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-05-05 10:15:00 | 1394.20 | 1384.44 | 1390.47 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-05 10:15:00 | 1394.20 | 1384.44 | 1390.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 10:15:00 | 1394.20 | 1384.44 | 1390.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-05 11:00:00 | 1394.20 | 1384.44 | 1390.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 11:15:00 | 1397.00 | 1386.96 | 1391.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-05 11:30:00 | 1404.70 | 1386.96 | 1391.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 73 — BUY (started 2026-05-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-05 13:15:00 | 1422.70 | 1397.36 | 1395.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-05 14:15:00 | 1429.60 | 1403.81 | 1398.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-08 12:15:00 | 1486.40 | 1489.05 | 1471.87 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-05-08 13:00:00 | 1486.40 | 1489.05 | 1471.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-05-14 14:45:00 | 1217.10 | 2025-05-16 11:15:00 | 1206.70 | STOP_HIT | 1.00 | -0.85% |
| BUY | retest2 | 2025-05-15 09:45:00 | 1219.00 | 2025-05-16 11:15:00 | 1206.70 | STOP_HIT | 1.00 | -1.01% |
| BUY | retest2 | 2025-05-15 11:15:00 | 1217.80 | 2025-05-16 11:15:00 | 1206.70 | STOP_HIT | 1.00 | -0.91% |
| BUY | retest2 | 2025-05-15 11:45:00 | 1218.30 | 2025-05-16 11:15:00 | 1206.70 | STOP_HIT | 1.00 | -0.95% |
| SELL | retest2 | 2025-05-22 11:30:00 | 1188.10 | 2025-06-02 09:15:00 | 1133.83 | PARTIAL | 0.50 | 4.57% |
| SELL | retest2 | 2025-05-22 15:15:00 | 1187.10 | 2025-06-02 09:15:00 | 1133.92 | PARTIAL | 0.50 | 4.48% |
| SELL | retest2 | 2025-05-23 12:45:00 | 1184.40 | 2025-06-02 09:15:00 | 1133.73 | PARTIAL | 0.50 | 4.28% |
| SELL | retest2 | 2025-05-23 13:30:00 | 1185.90 | 2025-06-02 09:15:00 | 1133.83 | PARTIAL | 0.50 | 4.39% |
| SELL | retest2 | 2025-05-22 11:30:00 | 1188.10 | 2025-06-03 10:15:00 | 1145.40 | STOP_HIT | 0.50 | 3.59% |
| SELL | retest2 | 2025-05-22 15:15:00 | 1187.10 | 2025-06-03 10:15:00 | 1145.40 | STOP_HIT | 0.50 | 3.51% |
| SELL | retest2 | 2025-05-23 12:45:00 | 1184.40 | 2025-06-03 10:15:00 | 1145.40 | STOP_HIT | 0.50 | 3.29% |
| SELL | retest2 | 2025-05-23 13:30:00 | 1185.90 | 2025-06-03 10:15:00 | 1145.40 | STOP_HIT | 0.50 | 3.42% |
| SELL | retest2 | 2025-05-26 10:15:00 | 1193.50 | 2025-06-05 10:15:00 | 1156.70 | STOP_HIT | 1.00 | 3.08% |
| SELL | retest2 | 2025-05-26 11:30:00 | 1193.60 | 2025-06-05 10:15:00 | 1156.70 | STOP_HIT | 1.00 | 3.09% |
| SELL | retest2 | 2025-05-27 10:00:00 | 1193.40 | 2025-06-05 10:15:00 | 1156.70 | STOP_HIT | 1.00 | 3.08% |
| SELL | retest2 | 2025-05-27 10:30:00 | 1193.50 | 2025-06-05 10:15:00 | 1156.70 | STOP_HIT | 1.00 | 3.08% |
| SELL | retest2 | 2025-05-28 09:15:00 | 1167.50 | 2025-06-05 10:15:00 | 1156.70 | STOP_HIT | 1.00 | 0.93% |
| BUY | retest2 | 2025-06-06 14:00:00 | 1163.50 | 2025-06-11 14:15:00 | 1160.10 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest2 | 2025-06-06 15:15:00 | 1161.10 | 2025-06-11 14:15:00 | 1160.10 | STOP_HIT | 1.00 | -0.09% |
| BUY | retest2 | 2025-06-10 09:45:00 | 1162.70 | 2025-06-11 14:15:00 | 1160.10 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest2 | 2025-06-10 11:00:00 | 1164.90 | 2025-06-11 14:15:00 | 1160.10 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest2 | 2025-06-17 10:00:00 | 1134.50 | 2025-06-24 09:15:00 | 1112.30 | STOP_HIT | 1.00 | 1.96% |
| BUY | retest2 | 2025-06-30 09:15:00 | 1153.60 | 2025-07-01 09:15:00 | 1119.00 | STOP_HIT | 1.00 | -3.00% |
| SELL | retest2 | 2025-07-24 10:15:00 | 1112.00 | 2025-07-25 14:15:00 | 1125.40 | STOP_HIT | 1.00 | -1.21% |
| SELL | retest2 | 2025-07-24 14:15:00 | 1115.90 | 2025-07-25 14:15:00 | 1125.40 | STOP_HIT | 1.00 | -0.85% |
| SELL | retest2 | 2025-07-25 12:00:00 | 1112.50 | 2025-07-25 14:15:00 | 1125.40 | STOP_HIT | 1.00 | -1.16% |
| BUY | retest2 | 2025-07-29 11:15:00 | 1146.50 | 2025-08-01 09:15:00 | 1103.20 | STOP_HIT | 1.00 | -3.78% |
| SELL | retest2 | 2025-08-05 11:30:00 | 1066.40 | 2025-08-13 09:15:00 | 1060.50 | STOP_HIT | 1.00 | 0.55% |
| SELL | retest2 | 2025-08-05 14:15:00 | 1066.40 | 2025-08-13 09:15:00 | 1060.50 | STOP_HIT | 1.00 | 0.55% |
| SELL | retest2 | 2025-08-07 10:15:00 | 1066.30 | 2025-08-13 09:15:00 | 1060.50 | STOP_HIT | 1.00 | 0.54% |
| SELL | retest2 | 2025-08-08 09:15:00 | 1061.10 | 2025-08-13 09:15:00 | 1060.50 | STOP_HIT | 1.00 | 0.06% |
| BUY | retest2 | 2025-08-18 13:15:00 | 1086.90 | 2025-08-20 09:15:00 | 1049.80 | STOP_HIT | 1.00 | -3.41% |
| SELL | retest2 | 2025-08-29 14:15:00 | 1031.10 | 2025-09-03 12:15:00 | 1036.70 | STOP_HIT | 1.00 | -0.54% |
| BUY | retest2 | 2025-09-10 09:15:00 | 1058.20 | 2025-09-23 12:15:00 | 1103.90 | STOP_HIT | 1.00 | 4.32% |
| BUY | retest2 | 2025-09-11 09:15:00 | 1080.20 | 2025-09-23 12:15:00 | 1103.90 | STOP_HIT | 1.00 | 2.19% |
| SELL | retest2 | 2025-09-29 11:30:00 | 1081.10 | 2025-10-03 09:15:00 | 1099.60 | STOP_HIT | 1.00 | -1.71% |
| SELL | retest2 | 2025-09-30 12:15:00 | 1082.40 | 2025-10-03 09:15:00 | 1099.60 | STOP_HIT | 1.00 | -1.59% |
| SELL | retest2 | 2025-09-30 13:15:00 | 1082.00 | 2025-10-03 09:15:00 | 1099.60 | STOP_HIT | 1.00 | -1.63% |
| BUY | retest2 | 2025-10-15 09:15:00 | 1126.70 | 2025-10-15 11:15:00 | 1117.30 | STOP_HIT | 1.00 | -0.83% |
| BUY | retest2 | 2025-10-15 10:45:00 | 1125.70 | 2025-10-15 11:15:00 | 1117.30 | STOP_HIT | 1.00 | -0.75% |
| SELL | retest2 | 2025-10-20 12:45:00 | 1103.50 | 2025-10-23 11:15:00 | 1108.60 | STOP_HIT | 1.00 | -0.46% |
| SELL | retest2 | 2025-10-20 13:30:00 | 1103.80 | 2025-10-23 11:15:00 | 1108.60 | STOP_HIT | 1.00 | -0.43% |
| SELL | retest2 | 2025-10-20 14:15:00 | 1102.80 | 2025-10-23 11:15:00 | 1108.60 | STOP_HIT | 1.00 | -0.53% |
| SELL | retest2 | 2025-10-28 11:30:00 | 1087.10 | 2025-10-28 14:15:00 | 1097.80 | STOP_HIT | 1.00 | -0.98% |
| SELL | retest2 | 2025-10-28 13:00:00 | 1087.90 | 2025-10-28 14:15:00 | 1097.80 | STOP_HIT | 1.00 | -0.91% |
| SELL | retest2 | 2025-10-28 13:30:00 | 1087.90 | 2025-10-28 14:15:00 | 1097.80 | STOP_HIT | 1.00 | -0.91% |
| BUY | retest2 | 2025-10-30 11:15:00 | 1105.70 | 2025-11-06 15:15:00 | 1135.00 | STOP_HIT | 1.00 | 2.65% |
| BUY | retest2 | 2025-10-30 12:30:00 | 1105.10 | 2025-11-06 15:15:00 | 1135.00 | STOP_HIT | 1.00 | 2.71% |
| BUY | retest2 | 2025-10-30 14:30:00 | 1105.00 | 2025-11-06 15:15:00 | 1135.00 | STOP_HIT | 1.00 | 2.71% |
| BUY | retest2 | 2025-10-30 15:15:00 | 1105.00 | 2025-11-06 15:15:00 | 1135.00 | STOP_HIT | 1.00 | 2.71% |
| BUY | retest2 | 2025-10-31 09:15:00 | 1116.00 | 2025-11-06 15:15:00 | 1135.00 | STOP_HIT | 1.00 | 1.70% |
| BUY | retest2 | 2025-11-19 12:00:00 | 1242.50 | 2025-11-20 09:15:00 | 1223.00 | STOP_HIT | 1.00 | -1.57% |
| SELL | retest2 | 2025-11-25 15:15:00 | 1205.00 | 2025-11-26 10:15:00 | 1216.00 | STOP_HIT | 1.00 | -0.91% |
| SELL | retest2 | 2025-11-26 09:45:00 | 1202.10 | 2025-11-26 10:15:00 | 1216.00 | STOP_HIT | 1.00 | -1.16% |
| BUY | retest2 | 2025-11-28 13:45:00 | 1238.20 | 2025-11-28 15:15:00 | 1219.10 | STOP_HIT | 1.00 | -1.54% |
| BUY | retest2 | 2025-11-28 14:30:00 | 1231.60 | 2025-11-28 15:15:00 | 1219.10 | STOP_HIT | 1.00 | -1.01% |
| SELL | retest2 | 2025-12-03 09:30:00 | 1204.60 | 2025-12-04 12:15:00 | 1221.30 | STOP_HIT | 1.00 | -1.39% |
| SELL | retest2 | 2025-12-11 12:45:00 | 1174.10 | 2025-12-12 09:15:00 | 1181.30 | STOP_HIT | 1.00 | -0.61% |
| BUY | retest2 | 2025-12-15 13:30:00 | 1188.50 | 2025-12-15 15:15:00 | 1183.40 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest2 | 2025-12-15 15:00:00 | 1187.50 | 2025-12-15 15:15:00 | 1183.40 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest2 | 2025-12-24 09:15:00 | 1224.90 | 2025-12-24 13:15:00 | 1213.20 | STOP_HIT | 1.00 | -0.96% |
| BUY | retest2 | 2025-12-24 12:00:00 | 1219.70 | 2025-12-24 13:15:00 | 1213.20 | STOP_HIT | 1.00 | -0.53% |
| BUY | retest2 | 2026-01-06 09:15:00 | 1213.20 | 2026-01-08 13:15:00 | 1213.80 | STOP_HIT | 1.00 | 0.05% |
| SELL | retest2 | 2026-01-19 15:15:00 | 1164.10 | 2026-01-28 15:15:00 | 1142.60 | STOP_HIT | 1.00 | 1.85% |
| BUY | retest2 | 2026-02-02 09:45:00 | 1178.70 | 2026-02-02 10:15:00 | 1157.90 | STOP_HIT | 1.00 | -1.76% |
| BUY | retest2 | 2026-02-04 13:15:00 | 1211.50 | 2026-02-05 11:15:00 | 1183.30 | STOP_HIT | 1.00 | -2.33% |
| BUY | retest2 | 2026-02-04 14:00:00 | 1212.10 | 2026-02-05 11:15:00 | 1183.30 | STOP_HIT | 1.00 | -2.38% |
| BUY | retest2 | 2026-02-04 15:00:00 | 1214.70 | 2026-02-05 11:15:00 | 1183.30 | STOP_HIT | 1.00 | -2.59% |
| SELL | retest2 | 2026-02-13 09:15:00 | 1151.60 | 2026-02-16 10:15:00 | 1171.70 | STOP_HIT | 1.00 | -1.75% |
| BUY | retest2 | 2026-02-18 09:15:00 | 1192.70 | 2026-02-18 09:15:00 | 1169.40 | STOP_HIT | 1.00 | -1.95% |
| SELL | retest2 | 2026-02-23 11:15:00 | 1160.80 | 2026-02-25 09:15:00 | 1181.80 | STOP_HIT | 1.00 | -1.81% |
| SELL | retest2 | 2026-02-24 14:15:00 | 1162.20 | 2026-02-25 09:15:00 | 1181.80 | STOP_HIT | 1.00 | -1.69% |
| BUY | retest2 | 2026-03-09 10:45:00 | 1219.90 | 2026-03-16 10:15:00 | 1281.00 | STOP_HIT | 1.00 | 5.01% |
| BUY | retest2 | 2026-03-23 14:30:00 | 1283.00 | 2026-03-24 09:15:00 | 1265.20 | STOP_HIT | 1.00 | -1.39% |
| BUY | retest2 | 2026-03-23 15:15:00 | 1284.00 | 2026-03-24 09:15:00 | 1265.20 | STOP_HIT | 1.00 | -1.46% |
| BUY | retest2 | 2026-03-30 12:00:00 | 1318.20 | 2026-04-02 09:15:00 | 1273.70 | STOP_HIT | 1.00 | -3.38% |
| BUY | retest2 | 2026-04-01 09:15:00 | 1356.30 | 2026-04-02 09:15:00 | 1273.70 | STOP_HIT | 1.00 | -6.09% |
| BUY | retest2 | 2026-04-02 11:30:00 | 1317.10 | 2026-04-08 10:15:00 | 1327.30 | STOP_HIT | 1.00 | 0.77% |
| BUY | retest2 | 2026-04-10 09:15:00 | 1346.80 | 2026-04-20 15:15:00 | 1365.40 | STOP_HIT | 1.00 | 1.38% |
| BUY | retest2 | 2026-04-10 11:30:00 | 1349.70 | 2026-04-20 15:15:00 | 1365.40 | STOP_HIT | 1.00 | 1.16% |
| BUY | retest2 | 2026-04-13 09:45:00 | 1346.40 | 2026-04-20 15:15:00 | 1365.40 | STOP_HIT | 1.00 | 1.41% |
| BUY | retest2 | 2026-04-13 10:45:00 | 1348.30 | 2026-04-20 15:15:00 | 1365.40 | STOP_HIT | 1.00 | 1.27% |
| BUY | retest2 | 2026-04-15 09:15:00 | 1353.90 | 2026-04-20 15:15:00 | 1365.40 | STOP_HIT | 1.00 | 0.85% |
