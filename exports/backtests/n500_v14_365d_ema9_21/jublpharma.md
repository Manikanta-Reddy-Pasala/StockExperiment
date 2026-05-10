# Jubilant Pharmova Ltd. (JUBLPHARMA)

## Backtest Summary

- **Window:** 2025-04-11 09:15:00 → 2026-05-08 15:15:00 (1850 bars)
- **Last close:** 1009.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 79 |
| ALERT1 | 50 |
| ALERT2 | 48 |
| ALERT2_SKIP | 25 |
| ALERT3 | 147 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 1 |
| ENTRY2 | 90 |
| PARTIAL | 14 |
| TARGET_HIT | 5 |
| STOP_HIT | 89 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 105 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 43 / 62
- **Target hits / Stop hits / Partials:** 5 / 86 / 14
- **Avg / median % per leg:** 1.04% / -0.72%
- **Sum % (uncompounded):** 108.99%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 43 | 16 | 37.2% | 5 | 38 | 0 | 1.17% | 50.2% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 43 | 16 | 37.2% | 5 | 38 | 0 | 1.17% | 50.2% |
| SELL (all) | 62 | 27 | 43.5% | 0 | 48 | 14 | 0.95% | 58.8% |
| SELL @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -1.94% | -1.9% |
| SELL @ 3rd Alert (retest2) | 61 | 27 | 44.3% | 0 | 47 | 14 | 1.00% | 60.8% |
| retest1 (combined) | 1 | 0 | 0.0% | 0 | 1 | 0 | -1.94% | -1.9% |
| retest2 (combined) | 104 | 43 | 41.3% | 5 | 85 | 14 | 1.07% | 110.9% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 11:15:00 | 890.55 | 879.13 | 877.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-13 11:15:00 | 896.05 | 886.45 | 882.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-14 13:15:00 | 899.80 | 900.27 | 894.01 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-14 13:45:00 | 900.45 | 900.27 | 894.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 09:15:00 | 1026.45 | 999.10 | 982.91 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-22 10:45:00 | 1034.90 | 1006.48 | 987.74 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-05-27 10:15:00 | 1138.39 | 1116.02 | 1087.71 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 2 — SELL (started 2025-06-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-03 11:15:00 | 1132.70 | 1153.70 | 1153.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-03 12:15:00 | 1128.70 | 1148.70 | 1151.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-03 14:15:00 | 1154.30 | 1146.96 | 1150.06 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-03 14:15:00 | 1154.30 | 1146.96 | 1150.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 14:15:00 | 1154.30 | 1146.96 | 1150.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-03 15:00:00 | 1154.30 | 1146.96 | 1150.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 15:15:00 | 1158.00 | 1149.16 | 1150.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-04 09:15:00 | 1161.30 | 1149.16 | 1150.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 09:15:00 | 1157.00 | 1150.73 | 1151.35 | EMA400 retest candle locked (from downside) |

### Cycle 3 — BUY (started 2025-06-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-04 10:15:00 | 1160.40 | 1152.67 | 1152.17 | EMA200 above EMA400 |

### Cycle 4 — SELL (started 2025-06-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-05 13:15:00 | 1146.30 | 1154.04 | 1154.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-05 14:15:00 | 1141.50 | 1151.53 | 1153.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-09 09:15:00 | 1143.30 | 1140.77 | 1145.07 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-09 09:15:00 | 1143.30 | 1140.77 | 1145.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-09 09:15:00 | 1143.30 | 1140.77 | 1145.07 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-09 10:15:00 | 1135.60 | 1140.77 | 1145.07 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-09 15:15:00 | 1141.30 | 1144.11 | 1145.34 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-10 10:15:00 | 1151.00 | 1146.73 | 1146.27 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-10 10:15:00 | 1151.00 | 1146.73 | 1146.27 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 5 — BUY (started 2025-06-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-10 10:15:00 | 1151.00 | 1146.73 | 1146.27 | EMA200 above EMA400 |

### Cycle 6 — SELL (started 2025-06-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-11 09:15:00 | 1143.40 | 1146.06 | 1146.15 | EMA200 below EMA400 |

### Cycle 7 — BUY (started 2025-06-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-11 10:15:00 | 1155.50 | 1147.95 | 1147.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-11 11:15:00 | 1156.30 | 1149.62 | 1147.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-11 13:15:00 | 1143.60 | 1149.08 | 1147.95 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-11 13:15:00 | 1143.60 | 1149.08 | 1147.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 13:15:00 | 1143.60 | 1149.08 | 1147.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-11 13:45:00 | 1145.10 | 1149.08 | 1147.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 14:15:00 | 1146.00 | 1148.47 | 1147.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-11 14:30:00 | 1146.50 | 1148.47 | 1147.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 15:15:00 | 1145.00 | 1147.77 | 1147.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-12 09:15:00 | 1143.20 | 1147.77 | 1147.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 8 — SELL (started 2025-06-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-12 09:15:00 | 1145.40 | 1147.30 | 1147.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-12 11:15:00 | 1138.50 | 1144.55 | 1146.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-12 12:15:00 | 1146.20 | 1144.88 | 1146.03 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-12 12:15:00 | 1146.20 | 1144.88 | 1146.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 12:15:00 | 1146.20 | 1144.88 | 1146.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-12 12:45:00 | 1148.00 | 1144.88 | 1146.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 13:15:00 | 1131.00 | 1142.10 | 1144.66 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-12 15:00:00 | 1128.20 | 1139.32 | 1143.16 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-13 09:15:00 | 1071.79 | 1135.09 | 1140.48 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-13 09:45:00 | 1118.10 | 1135.09 | 1140.48 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-16 09:15:00 | 1130.10 | 1111.22 | 1122.49 | SL hit (close>ema200) qty=0.50 sl=1111.22 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-16 10:45:00 | 1125.90 | 1114.91 | 1123.14 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-16 11:30:00 | 1111.20 | 1114.39 | 1122.16 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 14:15:00 | 1127.90 | 1116.46 | 1121.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 15:00:00 | 1127.90 | 1116.46 | 1121.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 15:15:00 | 1128.00 | 1118.76 | 1121.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-17 09:15:00 | 1125.50 | 1118.76 | 1121.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 10:15:00 | 1131.20 | 1122.63 | 1123.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-17 10:45:00 | 1138.00 | 1122.63 | 1123.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-06-17 11:15:00 | 1130.40 | 1124.19 | 1123.72 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-17 11:15:00 | 1130.40 | 1124.19 | 1123.72 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-17 11:15:00 | 1130.40 | 1124.19 | 1123.72 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 9 — BUY (started 2025-06-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-17 11:15:00 | 1130.40 | 1124.19 | 1123.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-18 11:15:00 | 1135.30 | 1128.54 | 1126.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-18 14:15:00 | 1124.40 | 1128.69 | 1127.05 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-18 14:15:00 | 1124.40 | 1128.69 | 1127.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 14:15:00 | 1124.40 | 1128.69 | 1127.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-18 15:00:00 | 1124.40 | 1128.69 | 1127.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 15:15:00 | 1127.00 | 1128.35 | 1127.05 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-19 09:15:00 | 1130.70 | 1128.35 | 1127.05 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-19 10:45:00 | 1141.10 | 1132.68 | 1129.27 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-19 13:45:00 | 1130.00 | 1130.79 | 1129.18 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-20 09:30:00 | 1131.00 | 1135.64 | 1131.68 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 12:15:00 | 1131.70 | 1138.06 | 1134.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-20 13:00:00 | 1131.70 | 1138.06 | 1134.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 13:15:00 | 1129.80 | 1136.41 | 1133.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-20 14:00:00 | 1129.80 | 1136.41 | 1133.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-06-20 14:15:00 | 1107.80 | 1130.69 | 1131.38 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-20 14:15:00 | 1107.80 | 1130.69 | 1131.38 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-20 14:15:00 | 1107.80 | 1130.69 | 1131.38 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-20 14:15:00 | 1107.80 | 1130.69 | 1131.38 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 10 — SELL (started 2025-06-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-20 14:15:00 | 1107.80 | 1130.69 | 1131.38 | EMA200 below EMA400 |

### Cycle 11 — BUY (started 2025-06-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-25 09:15:00 | 1154.00 | 1127.45 | 1125.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-25 10:15:00 | 1162.00 | 1134.36 | 1129.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-27 14:15:00 | 1172.60 | 1181.41 | 1172.07 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-27 14:15:00 | 1172.60 | 1181.41 | 1172.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 14:15:00 | 1172.60 | 1181.41 | 1172.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-27 15:00:00 | 1172.60 | 1181.41 | 1172.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 09:15:00 | 1193.10 | 1183.04 | 1174.39 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-30 12:15:00 | 1215.50 | 1189.07 | 1178.68 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-04 13:15:00 | 1190.90 | 1196.37 | 1196.98 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 12 — SELL (started 2025-07-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-04 13:15:00 | 1190.90 | 1196.37 | 1196.98 | EMA200 below EMA400 |

### Cycle 13 — BUY (started 2025-07-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-07 09:15:00 | 1204.00 | 1197.88 | 1197.50 | EMA200 above EMA400 |

### Cycle 14 — SELL (started 2025-07-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-08 09:15:00 | 1170.50 | 1192.90 | 1195.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-08 10:15:00 | 1155.70 | 1185.46 | 1191.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-10 09:15:00 | 1160.70 | 1157.10 | 1166.40 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-10 09:30:00 | 1161.40 | 1157.10 | 1166.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 11:15:00 | 1169.70 | 1159.70 | 1165.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-10 12:00:00 | 1169.70 | 1159.70 | 1165.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 12:15:00 | 1167.40 | 1161.24 | 1166.10 | EMA400 retest candle locked (from downside) |

### Cycle 15 — BUY (started 2025-07-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-11 09:15:00 | 1180.00 | 1168.86 | 1168.47 | EMA200 above EMA400 |

### Cycle 16 — SELL (started 2025-07-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-11 15:15:00 | 1161.90 | 1167.87 | 1168.22 | EMA200 below EMA400 |

### Cycle 17 — BUY (started 2025-07-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-14 09:15:00 | 1177.50 | 1169.79 | 1169.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-14 11:15:00 | 1182.00 | 1173.15 | 1170.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-16 09:15:00 | 1194.80 | 1198.21 | 1190.70 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-16 09:15:00 | 1194.80 | 1198.21 | 1190.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 09:15:00 | 1194.80 | 1198.21 | 1190.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-16 09:45:00 | 1187.70 | 1198.21 | 1190.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 10:15:00 | 1192.00 | 1196.97 | 1190.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-16 10:45:00 | 1192.50 | 1196.97 | 1190.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 15:15:00 | 1195.00 | 1195.81 | 1192.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-17 09:15:00 | 1200.00 | 1195.81 | 1192.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 09:15:00 | 1185.50 | 1193.75 | 1191.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-17 10:00:00 | 1185.50 | 1193.75 | 1191.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 10:15:00 | 1192.20 | 1193.44 | 1191.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-17 10:30:00 | 1183.60 | 1193.44 | 1191.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 11:15:00 | 1200.40 | 1194.83 | 1192.62 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-17 14:30:00 | 1201.80 | 1196.93 | 1194.19 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-17 15:00:00 | 1201.90 | 1196.93 | 1194.19 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-18 11:30:00 | 1207.00 | 1200.33 | 1196.76 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-22 15:15:00 | 1205.90 | 1211.80 | 1212.57 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-22 15:15:00 | 1205.90 | 1211.80 | 1212.57 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-22 15:15:00 | 1205.90 | 1211.80 | 1212.57 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 18 — SELL (started 2025-07-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-22 15:15:00 | 1205.90 | 1211.80 | 1212.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-23 10:15:00 | 1198.50 | 1207.94 | 1210.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-25 12:15:00 | 1177.30 | 1175.67 | 1186.21 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-25 13:00:00 | 1177.30 | 1175.67 | 1186.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 14:15:00 | 1181.50 | 1176.91 | 1184.95 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-28 09:15:00 | 1168.60 | 1176.71 | 1184.12 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-28 10:30:00 | 1167.20 | 1172.06 | 1180.63 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-28 11:15:00 | 1189.80 | 1175.60 | 1181.47 | SL hit (close>static) qty=1.00 sl=1185.60 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-28 11:15:00 | 1189.80 | 1175.60 | 1181.47 | SL hit (close>static) qty=1.00 sl=1185.60 alert=retest2 |

### Cycle 19 — BUY (started 2025-07-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-29 11:15:00 | 1193.50 | 1184.66 | 1183.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-29 12:15:00 | 1209.70 | 1189.67 | 1185.89 | Break + close above crossover candle high |

### Cycle 20 — SELL (started 2025-07-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-29 13:15:00 | 1150.90 | 1181.92 | 1182.71 | EMA200 below EMA400 |

### Cycle 21 — BUY (started 2025-07-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-30 10:15:00 | 1214.20 | 1185.00 | 1182.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-30 11:15:00 | 1222.00 | 1192.40 | 1186.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-31 09:15:00 | 1196.80 | 1212.97 | 1201.08 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-31 09:15:00 | 1196.80 | 1212.97 | 1201.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 09:15:00 | 1196.80 | 1212.97 | 1201.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-31 10:00:00 | 1196.80 | 1212.97 | 1201.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 10:15:00 | 1202.90 | 1210.96 | 1201.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-31 10:45:00 | 1204.80 | 1210.96 | 1201.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 11:15:00 | 1210.00 | 1210.77 | 1202.04 | EMA400 retest candle locked (from upside) |

### Cycle 22 — SELL (started 2025-08-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-01 09:15:00 | 1160.10 | 1195.15 | 1197.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-01 10:15:00 | 1160.00 | 1188.12 | 1194.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-04 13:15:00 | 1151.50 | 1150.50 | 1165.76 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-04 14:00:00 | 1151.50 | 1150.50 | 1165.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 09:15:00 | 1142.70 | 1148.59 | 1161.03 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-05 11:30:00 | 1129.10 | 1140.99 | 1155.29 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-11 09:15:00 | 1072.64 | 1096.30 | 1104.45 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-11 13:15:00 | 1095.10 | 1092.65 | 1099.75 | SL hit (close>ema200) qty=0.50 sl=1092.65 alert=retest2 |

### Cycle 23 — BUY (started 2025-08-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-19 11:15:00 | 1094.00 | 1068.58 | 1066.34 | EMA200 above EMA400 |

### Cycle 24 — SELL (started 2025-08-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-21 12:15:00 | 1061.00 | 1073.22 | 1073.45 | EMA200 below EMA400 |

### Cycle 25 — BUY (started 2025-08-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-21 13:15:00 | 1078.10 | 1074.19 | 1073.87 | EMA200 above EMA400 |

### Cycle 26 — SELL (started 2025-08-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-21 14:15:00 | 1066.70 | 1072.70 | 1073.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-22 11:15:00 | 1062.50 | 1068.42 | 1070.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-22 12:15:00 | 1072.50 | 1069.23 | 1070.98 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-22 12:15:00 | 1072.50 | 1069.23 | 1070.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 12:15:00 | 1072.50 | 1069.23 | 1070.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-22 13:00:00 | 1072.50 | 1069.23 | 1070.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 13:15:00 | 1082.70 | 1071.93 | 1072.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-22 14:00:00 | 1082.70 | 1071.93 | 1072.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 27 — BUY (started 2025-08-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-22 14:15:00 | 1084.40 | 1074.42 | 1073.16 | EMA200 above EMA400 |

### Cycle 28 — SELL (started 2025-08-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-26 09:15:00 | 1058.30 | 1075.74 | 1075.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-26 13:15:00 | 1054.80 | 1065.72 | 1070.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-01 09:15:00 | 1025.50 | 1023.73 | 1034.94 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-01 10:00:00 | 1025.50 | 1023.73 | 1034.94 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 11:15:00 | 1030.50 | 1024.65 | 1033.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 11:45:00 | 1034.10 | 1024.65 | 1033.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 12:15:00 | 1041.30 | 1027.98 | 1034.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 13:00:00 | 1041.30 | 1027.98 | 1034.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 13:15:00 | 1046.40 | 1031.66 | 1035.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 14:00:00 | 1046.40 | 1031.66 | 1035.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 29 — BUY (started 2025-09-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-02 09:15:00 | 1047.50 | 1038.01 | 1037.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-03 12:15:00 | 1055.40 | 1045.77 | 1042.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-04 10:15:00 | 1046.60 | 1049.45 | 1045.95 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-04 10:15:00 | 1046.60 | 1049.45 | 1045.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 10:15:00 | 1046.60 | 1049.45 | 1045.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 10:45:00 | 1048.60 | 1049.45 | 1045.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 11:15:00 | 1043.40 | 1048.24 | 1045.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 11:45:00 | 1045.00 | 1048.24 | 1045.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 12:15:00 | 1059.50 | 1050.49 | 1046.97 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-04 14:00:00 | 1061.50 | 1052.69 | 1048.29 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-05 09:30:00 | 1065.70 | 1058.03 | 1052.09 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-05 11:00:00 | 1061.00 | 1058.63 | 1052.90 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-16 10:15:00 | 1115.80 | 1120.00 | 1120.30 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-16 10:15:00 | 1115.80 | 1120.00 | 1120.30 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-16 10:15:00 | 1115.80 | 1120.00 | 1120.30 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 30 — SELL (started 2025-09-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-16 10:15:00 | 1115.80 | 1120.00 | 1120.30 | EMA200 below EMA400 |

### Cycle 31 — BUY (started 2025-09-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-16 12:15:00 | 1125.70 | 1120.44 | 1120.40 | EMA200 above EMA400 |

### Cycle 32 — SELL (started 2025-09-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-16 15:15:00 | 1117.30 | 1120.47 | 1120.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-17 09:15:00 | 1113.00 | 1118.97 | 1119.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-17 13:15:00 | 1113.00 | 1110.60 | 1114.99 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-17 14:00:00 | 1113.00 | 1110.60 | 1114.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 14:15:00 | 1113.70 | 1111.22 | 1114.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-17 15:00:00 | 1113.70 | 1111.22 | 1114.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 15:15:00 | 1111.00 | 1111.18 | 1114.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-18 09:15:00 | 1114.20 | 1111.18 | 1114.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 09:15:00 | 1112.70 | 1111.48 | 1114.36 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-18 13:00:00 | 1099.30 | 1109.47 | 1112.78 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-18 14:15:00 | 1116.80 | 1109.09 | 1111.91 | SL hit (close>static) qty=1.00 sl=1116.70 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-19 15:00:00 | 1098.20 | 1107.12 | 1109.52 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-22 10:15:00 | 1100.00 | 1104.04 | 1107.53 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-26 09:15:00 | 1043.29 | 1068.38 | 1076.34 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-26 09:15:00 | 1045.00 | 1068.38 | 1076.34 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-29 09:15:00 | 1061.40 | 1051.44 | 1061.46 | SL hit (close>ema200) qty=0.50 sl=1051.44 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-29 09:15:00 | 1061.40 | 1051.44 | 1061.46 | SL hit (close>ema200) qty=0.50 sl=1051.44 alert=retest2 |

### Cycle 33 — BUY (started 2025-09-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-30 09:15:00 | 1082.40 | 1063.57 | 1063.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-01 13:15:00 | 1102.70 | 1083.44 | 1076.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-06 10:15:00 | 1116.20 | 1116.94 | 1105.00 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-06 11:00:00 | 1116.20 | 1116.94 | 1105.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 12:15:00 | 1105.00 | 1113.51 | 1105.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-06 12:45:00 | 1104.60 | 1113.51 | 1105.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 13:15:00 | 1105.20 | 1111.84 | 1105.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-06 13:45:00 | 1105.80 | 1111.84 | 1105.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 14:15:00 | 1098.10 | 1109.10 | 1104.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-06 15:00:00 | 1098.10 | 1109.10 | 1104.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 15:15:00 | 1096.60 | 1106.60 | 1104.02 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-07 09:15:00 | 1102.60 | 1106.60 | 1104.02 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-07 10:15:00 | 1087.80 | 1100.34 | 1101.47 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 34 — SELL (started 2025-10-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-07 10:15:00 | 1087.80 | 1100.34 | 1101.47 | EMA200 below EMA400 |

### Cycle 35 — BUY (started 2025-10-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-09 13:15:00 | 1100.90 | 1096.09 | 1095.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-10 11:15:00 | 1108.50 | 1099.40 | 1097.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-13 10:15:00 | 1103.30 | 1111.41 | 1106.09 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-13 10:15:00 | 1103.30 | 1111.41 | 1106.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 10:15:00 | 1103.30 | 1111.41 | 1106.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-13 10:30:00 | 1106.90 | 1111.41 | 1106.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 11:15:00 | 1104.70 | 1110.07 | 1105.97 | EMA400 retest candle locked (from upside) |

### Cycle 36 — SELL (started 2025-10-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-13 15:15:00 | 1099.00 | 1103.22 | 1103.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-14 09:15:00 | 1077.50 | 1098.08 | 1101.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-15 09:15:00 | 1081.20 | 1080.70 | 1088.68 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-15 10:15:00 | 1086.60 | 1080.70 | 1088.68 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 14:15:00 | 1064.20 | 1066.92 | 1077.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 15:00:00 | 1064.20 | 1066.92 | 1077.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 09:15:00 | 1082.60 | 1069.12 | 1076.97 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-16 13:45:00 | 1070.80 | 1074.14 | 1077.41 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-16 14:45:00 | 1068.40 | 1073.59 | 1076.86 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-17 10:00:00 | 1068.70 | 1072.52 | 1075.80 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-20 09:15:00 | 1080.90 | 1076.67 | 1076.23 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-20 09:15:00 | 1080.90 | 1076.67 | 1076.23 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-20 09:15:00 | 1080.90 | 1076.67 | 1076.23 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 37 — BUY (started 2025-10-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-20 09:15:00 | 1080.90 | 1076.67 | 1076.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-20 11:15:00 | 1089.60 | 1080.25 | 1077.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-24 11:15:00 | 1121.80 | 1123.73 | 1113.31 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-24 11:45:00 | 1119.10 | 1123.73 | 1113.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 09:15:00 | 1122.00 | 1121.47 | 1115.89 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-27 12:30:00 | 1125.70 | 1120.96 | 1116.93 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-28 09:15:00 | 1141.00 | 1121.93 | 1118.41 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-28 10:30:00 | 1127.10 | 1123.86 | 1120.01 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-29 09:15:00 | 1127.40 | 1121.72 | 1120.23 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 09:15:00 | 1121.10 | 1121.59 | 1120.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-29 10:00:00 | 1121.10 | 1121.59 | 1120.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 10:15:00 | 1127.00 | 1122.68 | 1120.92 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-29 13:30:00 | 1131.50 | 1124.55 | 1122.22 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-30 15:15:00 | 1118.20 | 1122.88 | 1123.39 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-30 15:15:00 | 1118.20 | 1122.88 | 1123.39 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-30 15:15:00 | 1118.20 | 1122.88 | 1123.39 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-30 15:15:00 | 1118.20 | 1122.88 | 1123.39 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-30 15:15:00 | 1118.20 | 1122.88 | 1123.39 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 38 — SELL (started 2025-10-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-30 15:15:00 | 1118.20 | 1122.88 | 1123.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-31 12:15:00 | 1095.50 | 1117.34 | 1120.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-03 09:15:00 | 1113.30 | 1108.07 | 1114.49 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-03 09:15:00 | 1113.30 | 1108.07 | 1114.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 09:15:00 | 1113.30 | 1108.07 | 1114.49 | EMA400 retest candle locked (from downside) |

### Cycle 39 — BUY (started 2025-11-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-03 11:15:00 | 1139.50 | 1119.86 | 1119.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-03 13:15:00 | 1165.50 | 1131.67 | 1124.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-06 09:15:00 | 1150.30 | 1162.69 | 1150.66 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-06 09:15:00 | 1150.30 | 1162.69 | 1150.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 09:15:00 | 1150.30 | 1162.69 | 1150.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-06 09:45:00 | 1154.80 | 1162.69 | 1150.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 10:15:00 | 1154.40 | 1161.03 | 1151.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-06 10:45:00 | 1151.60 | 1161.03 | 1151.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 11:15:00 | 1151.50 | 1159.13 | 1151.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-06 11:45:00 | 1155.80 | 1159.13 | 1151.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 12:15:00 | 1157.00 | 1158.70 | 1151.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-06 12:45:00 | 1151.10 | 1158.70 | 1151.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 15:15:00 | 1170.00 | 1161.87 | 1154.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-07 09:15:00 | 1141.10 | 1161.87 | 1154.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 09:15:00 | 1132.90 | 1156.08 | 1152.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-07 10:00:00 | 1132.90 | 1156.08 | 1152.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 10:15:00 | 1135.50 | 1151.96 | 1151.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-07 10:30:00 | 1131.90 | 1151.96 | 1151.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 40 — SELL (started 2025-11-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-07 11:15:00 | 1131.10 | 1147.79 | 1149.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-07 15:15:00 | 1122.20 | 1138.53 | 1144.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-10 11:15:00 | 1137.80 | 1135.26 | 1141.05 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-10 12:00:00 | 1137.80 | 1135.26 | 1141.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 14:15:00 | 1140.30 | 1136.16 | 1140.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-10 15:00:00 | 1140.30 | 1136.16 | 1140.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 15:15:00 | 1138.00 | 1136.53 | 1139.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-11 09:15:00 | 1145.30 | 1136.53 | 1139.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 09:15:00 | 1138.90 | 1137.00 | 1139.74 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-11 10:15:00 | 1132.40 | 1137.00 | 1139.74 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-12 12:45:00 | 1133.90 | 1127.28 | 1130.64 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-12 15:15:00 | 1140.50 | 1134.13 | 1133.38 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-12 15:15:00 | 1140.50 | 1134.13 | 1133.38 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 41 — BUY (started 2025-11-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-12 15:15:00 | 1140.50 | 1134.13 | 1133.38 | EMA200 above EMA400 |

### Cycle 42 — SELL (started 2025-11-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-13 14:15:00 | 1127.30 | 1133.50 | 1133.66 | EMA200 below EMA400 |

### Cycle 43 — BUY (started 2025-11-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-17 09:15:00 | 1142.10 | 1132.76 | 1132.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-17 14:15:00 | 1159.20 | 1143.58 | 1138.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-18 09:15:00 | 1134.10 | 1143.91 | 1139.31 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-18 09:15:00 | 1134.10 | 1143.91 | 1139.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 09:15:00 | 1134.10 | 1143.91 | 1139.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-18 10:00:00 | 1134.10 | 1143.91 | 1139.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 10:15:00 | 1131.40 | 1141.41 | 1138.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-18 10:30:00 | 1137.30 | 1141.41 | 1138.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 11:15:00 | 1131.30 | 1139.39 | 1137.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-18 12:15:00 | 1139.90 | 1139.39 | 1137.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 13:15:00 | 1134.20 | 1137.79 | 1137.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-18 13:45:00 | 1134.00 | 1137.79 | 1137.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 44 — SELL (started 2025-11-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-18 14:15:00 | 1133.00 | 1136.83 | 1137.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-19 09:15:00 | 1124.20 | 1134.01 | 1135.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-24 12:15:00 | 1071.70 | 1064.37 | 1078.16 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-24 13:00:00 | 1071.70 | 1064.37 | 1078.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 14:15:00 | 1074.50 | 1067.62 | 1077.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-24 15:00:00 | 1074.50 | 1067.62 | 1077.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 15:15:00 | 1076.00 | 1069.29 | 1077.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-25 09:45:00 | 1081.90 | 1072.53 | 1077.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 10:15:00 | 1088.20 | 1075.67 | 1078.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-25 10:30:00 | 1089.00 | 1075.67 | 1078.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 45 — BUY (started 2025-11-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-25 13:15:00 | 1091.30 | 1082.22 | 1081.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-26 09:15:00 | 1095.80 | 1086.27 | 1083.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-27 12:15:00 | 1096.70 | 1097.88 | 1093.02 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-27 12:30:00 | 1097.10 | 1097.88 | 1093.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 13:15:00 | 1100.00 | 1098.30 | 1093.65 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-27 14:45:00 | 1101.10 | 1098.64 | 1094.23 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-27 15:15:00 | 1102.20 | 1098.64 | 1094.23 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-01 09:15:00 | 1101.40 | 1097.75 | 1096.27 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-01 10:00:00 | 1103.00 | 1098.80 | 1096.89 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 10:15:00 | 1098.80 | 1098.80 | 1097.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-01 11:00:00 | 1098.80 | 1098.80 | 1097.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 11:15:00 | 1103.20 | 1099.68 | 1097.62 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-01 12:45:00 | 1106.30 | 1100.64 | 1098.24 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-01 14:15:00 | 1106.10 | 1101.59 | 1098.89 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-01 15:00:00 | 1107.20 | 1102.72 | 1099.65 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-02 09:30:00 | 1107.80 | 1104.00 | 1100.79 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 13:15:00 | 1099.30 | 1104.97 | 1102.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-02 14:00:00 | 1099.30 | 1104.97 | 1102.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 14:15:00 | 1099.90 | 1103.95 | 1102.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-02 15:15:00 | 1087.00 | 1103.95 | 1102.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-12-02 15:15:00 | 1087.00 | 1100.56 | 1100.85 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-02 15:15:00 | 1087.00 | 1100.56 | 1100.85 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-02 15:15:00 | 1087.00 | 1100.56 | 1100.85 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-02 15:15:00 | 1087.00 | 1100.56 | 1100.85 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-02 15:15:00 | 1087.00 | 1100.56 | 1100.85 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-02 15:15:00 | 1087.00 | 1100.56 | 1100.85 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-02 15:15:00 | 1087.00 | 1100.56 | 1100.85 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-02 15:15:00 | 1087.00 | 1100.56 | 1100.85 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 46 — SELL (started 2025-12-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-02 15:15:00 | 1087.00 | 1100.56 | 1100.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-03 09:15:00 | 1079.50 | 1096.35 | 1098.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-04 09:15:00 | 1084.60 | 1083.52 | 1089.46 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-04 10:00:00 | 1084.60 | 1083.52 | 1089.46 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 10:15:00 | 1088.80 | 1084.57 | 1089.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-04 11:00:00 | 1088.80 | 1084.57 | 1089.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 11:15:00 | 1079.20 | 1083.50 | 1088.48 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-04 12:15:00 | 1076.40 | 1083.50 | 1088.48 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-04 13:45:00 | 1077.60 | 1081.24 | 1086.52 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-04 14:30:00 | 1075.70 | 1078.23 | 1084.67 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-05 14:30:00 | 1077.40 | 1073.48 | 1078.07 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 15:15:00 | 1073.70 | 1073.53 | 1077.67 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-08 09:15:00 | 1069.40 | 1073.53 | 1077.67 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-09 09:15:00 | 1022.58 | 1045.25 | 1058.88 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-09 09:15:00 | 1023.72 | 1045.25 | 1058.88 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-09 09:15:00 | 1021.91 | 1045.25 | 1058.88 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-09 09:15:00 | 1023.53 | 1045.25 | 1058.88 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-09 09:15:00 | 1015.93 | 1045.25 | 1058.88 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-09 12:15:00 | 1048.90 | 1043.97 | 1054.68 | SL hit (close>ema200) qty=0.50 sl=1043.97 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-09 12:15:00 | 1048.90 | 1043.97 | 1054.68 | SL hit (close>ema200) qty=0.50 sl=1043.97 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-09 12:15:00 | 1048.90 | 1043.97 | 1054.68 | SL hit (close>ema200) qty=0.50 sl=1043.97 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-09 12:15:00 | 1048.90 | 1043.97 | 1054.68 | SL hit (close>ema200) qty=0.50 sl=1043.97 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-09 12:15:00 | 1048.90 | 1043.97 | 1054.68 | SL hit (close>ema200) qty=0.50 sl=1043.97 alert=retest2 |

### Cycle 47 — BUY (started 2025-12-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-10 12:15:00 | 1076.20 | 1060.61 | 1058.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-10 13:15:00 | 1078.20 | 1064.13 | 1060.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-10 14:15:00 | 1062.00 | 1063.71 | 1060.65 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-10 14:15:00 | 1062.00 | 1063.71 | 1060.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 14:15:00 | 1062.00 | 1063.71 | 1060.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-10 14:45:00 | 1062.20 | 1063.71 | 1060.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 15:15:00 | 1065.70 | 1064.10 | 1061.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-11 09:15:00 | 1060.20 | 1064.10 | 1061.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 09:15:00 | 1073.90 | 1066.06 | 1062.27 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-11 10:30:00 | 1084.80 | 1069.25 | 1064.07 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-16 10:15:00 | 1065.50 | 1077.64 | 1078.83 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 48 — SELL (started 2025-12-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-16 10:15:00 | 1065.50 | 1077.64 | 1078.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-16 14:15:00 | 1062.70 | 1073.65 | 1076.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-18 11:15:00 | 1055.10 | 1052.18 | 1059.98 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-18 11:45:00 | 1056.10 | 1052.18 | 1059.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 14:15:00 | 1054.60 | 1053.22 | 1058.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-18 14:30:00 | 1054.20 | 1053.22 | 1058.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 09:15:00 | 1070.00 | 1056.33 | 1059.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 09:45:00 | 1071.70 | 1056.33 | 1059.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 10:15:00 | 1070.90 | 1059.25 | 1060.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 10:30:00 | 1073.30 | 1059.25 | 1060.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 49 — BUY (started 2025-12-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-19 11:15:00 | 1071.30 | 1061.66 | 1061.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-19 13:15:00 | 1076.30 | 1066.34 | 1063.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-22 14:15:00 | 1080.80 | 1082.05 | 1074.95 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-22 15:00:00 | 1080.80 | 1082.05 | 1074.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 09:15:00 | 1081.30 | 1081.89 | 1076.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-23 09:30:00 | 1073.70 | 1081.89 | 1076.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 10:15:00 | 1075.90 | 1080.69 | 1076.09 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-23 12:00:00 | 1083.30 | 1081.22 | 1076.75 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-23 13:45:00 | 1084.00 | 1081.40 | 1077.59 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-23 15:15:00 | 1086.00 | 1081.12 | 1077.81 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-24 12:15:00 | 1073.20 | 1076.84 | 1076.91 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-24 12:15:00 | 1073.20 | 1076.84 | 1076.91 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-24 12:15:00 | 1073.20 | 1076.84 | 1076.91 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 50 — SELL (started 2025-12-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-24 12:15:00 | 1073.20 | 1076.84 | 1076.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-24 14:15:00 | 1069.10 | 1075.13 | 1076.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-26 10:15:00 | 1072.50 | 1071.19 | 1073.72 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-26 10:15:00 | 1072.50 | 1071.19 | 1073.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 10:15:00 | 1072.50 | 1071.19 | 1073.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-26 10:30:00 | 1074.70 | 1071.19 | 1073.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 11:15:00 | 1074.70 | 1071.89 | 1073.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-26 12:00:00 | 1074.70 | 1071.89 | 1073.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 12:15:00 | 1072.90 | 1072.09 | 1073.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-26 12:45:00 | 1073.90 | 1072.09 | 1073.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 13:15:00 | 1073.10 | 1072.29 | 1073.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-26 14:00:00 | 1073.10 | 1072.29 | 1073.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 14:15:00 | 1074.70 | 1072.78 | 1073.77 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-29 09:15:00 | 1065.10 | 1072.78 | 1073.68 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-31 13:15:00 | 1072.70 | 1056.48 | 1055.95 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 51 — BUY (started 2025-12-31 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-31 13:15:00 | 1072.70 | 1056.48 | 1055.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-31 14:15:00 | 1073.60 | 1059.90 | 1057.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-01 10:15:00 | 1053.00 | 1061.01 | 1058.89 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-01 10:15:00 | 1053.00 | 1061.01 | 1058.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 10:15:00 | 1053.00 | 1061.01 | 1058.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-01 11:00:00 | 1053.00 | 1061.01 | 1058.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 11:15:00 | 1080.00 | 1064.81 | 1060.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-01 11:30:00 | 1052.20 | 1064.81 | 1060.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 12:15:00 | 1061.00 | 1064.05 | 1060.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-01 13:00:00 | 1061.00 | 1064.05 | 1060.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 13:15:00 | 1060.00 | 1063.24 | 1060.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-01 13:45:00 | 1055.90 | 1063.24 | 1060.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 14:15:00 | 1054.70 | 1061.53 | 1060.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-01 14:45:00 | 1053.10 | 1061.53 | 1060.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 15:15:00 | 1066.00 | 1062.42 | 1060.73 | EMA400 retest candle locked (from upside) |

### Cycle 52 — SELL (started 2026-01-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-02 11:15:00 | 1053.10 | 1058.96 | 1059.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-02 13:15:00 | 1048.30 | 1055.86 | 1057.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-02 15:15:00 | 1057.80 | 1055.63 | 1057.40 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-02 15:15:00 | 1057.80 | 1055.63 | 1057.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 15:15:00 | 1057.80 | 1055.63 | 1057.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-05 09:15:00 | 1055.80 | 1055.63 | 1057.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 09:15:00 | 1056.60 | 1055.82 | 1057.33 | EMA400 retest candle locked (from downside) |

### Cycle 53 — BUY (started 2026-01-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-05 10:15:00 | 1070.40 | 1058.74 | 1058.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-06 10:15:00 | 1076.40 | 1069.23 | 1064.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-06 15:15:00 | 1076.10 | 1076.33 | 1070.65 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-07 09:15:00 | 1083.80 | 1076.33 | 1070.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 09:15:00 | 1097.40 | 1080.54 | 1073.08 | EMA400 retest candle locked (from upside) |

### Cycle 54 — SELL (started 2026-01-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-09 09:15:00 | 1072.00 | 1076.75 | 1076.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-09 12:15:00 | 1054.60 | 1068.29 | 1072.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-09 14:15:00 | 1065.00 | 1064.56 | 1070.06 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-09 14:30:00 | 1068.70 | 1064.56 | 1070.06 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 14:15:00 | 1050.30 | 1046.19 | 1055.74 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-12 15:15:00 | 1044.00 | 1046.19 | 1055.74 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 10:30:00 | 1043.00 | 1045.17 | 1052.86 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 11:30:00 | 1044.00 | 1044.46 | 1051.83 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 13:30:00 | 1044.80 | 1043.54 | 1050.13 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 14:15:00 | 1041.30 | 1043.09 | 1049.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-13 14:45:00 | 1044.40 | 1043.09 | 1049.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 11:15:00 | 1066.80 | 1046.00 | 1048.45 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-01-14 11:15:00 | 1066.80 | 1046.00 | 1048.45 | SL hit (close>static) qty=1.00 sl=1058.60 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-14 11:15:00 | 1066.80 | 1046.00 | 1048.45 | SL hit (close>static) qty=1.00 sl=1058.60 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-14 11:15:00 | 1066.80 | 1046.00 | 1048.45 | SL hit (close>static) qty=1.00 sl=1058.60 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-14 11:15:00 | 1066.80 | 1046.00 | 1048.45 | SL hit (close>static) qty=1.00 sl=1058.60 alert=retest2 |
| ALERT3_SIDEWAYS | 2026-01-14 12:00:00 | 1066.80 | 1046.00 | 1048.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 55 — BUY (started 2026-01-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-14 12:15:00 | 1068.60 | 1050.52 | 1050.28 | EMA200 above EMA400 |

### Cycle 56 — SELL (started 2026-01-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-16 15:15:00 | 1042.20 | 1052.89 | 1053.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-19 09:15:00 | 1018.80 | 1046.08 | 1050.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-21 15:15:00 | 951.80 | 951.32 | 972.16 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-22 09:15:00 | 964.30 | 951.32 | 972.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 09:15:00 | 966.60 | 954.38 | 971.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 09:45:00 | 969.90 | 954.38 | 971.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 14:15:00 | 969.10 | 959.78 | 967.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 14:45:00 | 969.40 | 959.78 | 967.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 15:15:00 | 961.20 | 960.06 | 967.32 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-23 09:30:00 | 957.30 | 959.07 | 966.21 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-23 14:30:00 | 960.60 | 952.61 | 959.74 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-23 15:00:00 | 948.30 | 952.61 | 959.74 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-28 13:15:00 | 960.70 | 952.53 | 952.50 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-28 13:15:00 | 960.70 | 952.53 | 952.50 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-28 13:15:00 | 960.70 | 952.53 | 952.50 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 57 — BUY (started 2026-01-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-28 13:15:00 | 960.70 | 952.53 | 952.50 | EMA200 above EMA400 |

### Cycle 58 — SELL (started 2026-01-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-29 10:15:00 | 943.60 | 952.11 | 952.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-29 11:15:00 | 939.00 | 949.49 | 951.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-30 09:15:00 | 954.00 | 946.35 | 948.65 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-30 09:15:00 | 954.00 | 946.35 | 948.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 09:15:00 | 954.00 | 946.35 | 948.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-30 09:45:00 | 955.90 | 946.35 | 948.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 10:15:00 | 959.00 | 948.88 | 949.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-30 10:45:00 | 958.90 | 948.88 | 949.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 59 — BUY (started 2026-01-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-30 11:15:00 | 958.10 | 950.73 | 950.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-30 13:15:00 | 975.00 | 956.75 | 953.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-01 11:15:00 | 962.70 | 965.72 | 960.00 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-01 11:15:00 | 962.70 | 965.72 | 960.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 11:15:00 | 962.70 | 965.72 | 960.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 12:00:00 | 962.70 | 965.72 | 960.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 12:15:00 | 957.30 | 964.03 | 959.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 12:30:00 | 960.25 | 964.03 | 959.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 13:15:00 | 953.55 | 961.94 | 959.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 13:45:00 | 953.95 | 961.94 | 959.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 14:15:00 | 960.60 | 961.67 | 959.32 | EMA400 retest candle locked (from upside) |

### Cycle 60 — SELL (started 2026-02-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-02 10:15:00 | 933.35 | 954.65 | 956.72 | EMA200 below EMA400 |

### Cycle 61 — BUY (started 2026-02-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 09:15:00 | 989.10 | 959.08 | 956.80 | EMA200 above EMA400 |

### Cycle 62 — SELL (started 2026-02-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-05 10:15:00 | 960.00 | 971.25 | 971.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-06 10:15:00 | 954.25 | 964.50 | 967.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-09 13:15:00 | 927.45 | 924.25 | 940.69 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-09 14:00:00 | 927.45 | 924.25 | 940.69 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 14:15:00 | 941.70 | 927.74 | 940.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-09 14:45:00 | 939.00 | 927.74 | 940.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 15:15:00 | 940.50 | 930.29 | 940.76 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-10 09:30:00 | 938.00 | 931.57 | 940.38 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-10 11:15:00 | 937.10 | 933.39 | 940.41 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-10 14:30:00 | 938.65 | 936.91 | 940.15 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-11 14:30:00 | 934.95 | 934.64 | 936.62 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 15:15:00 | 937.55 | 935.22 | 936.71 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-12 09:15:00 | 926.35 | 935.22 | 936.71 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-12 14:15:00 | 940.05 | 930.37 | 932.79 | SL hit (close>static) qty=1.00 sl=939.95 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-13 09:15:00 | 924.25 | 934.29 | 934.35 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-16 15:15:00 | 891.10 | 904.46 | 914.24 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-16 15:15:00 | 890.25 | 904.46 | 914.24 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-16 15:15:00 | 891.72 | 904.46 | 914.24 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-16 15:15:00 | 888.20 | 904.46 | 914.24 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-17 09:15:00 | 906.05 | 904.78 | 913.50 | SL hit (close>ema200) qty=0.50 sl=904.78 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-17 09:15:00 | 906.05 | 904.78 | 913.50 | SL hit (close>ema200) qty=0.50 sl=904.78 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-17 09:15:00 | 906.05 | 904.78 | 913.50 | SL hit (close>ema200) qty=0.50 sl=904.78 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-17 09:15:00 | 906.05 | 904.78 | 913.50 | SL hit (close>ema200) qty=0.50 sl=904.78 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-19 12:15:00 | 878.04 | 889.37 | 896.70 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-20 14:15:00 | 870.90 | 865.18 | 876.57 | SL hit (close>ema200) qty=0.50 sl=865.18 alert=retest2 |

### Cycle 63 — BUY (started 2026-02-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-23 14:15:00 | 901.05 | 881.70 | 880.24 | EMA200 above EMA400 |

### Cycle 64 — SELL (started 2026-02-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-24 14:15:00 | 880.30 | 881.21 | 881.26 | EMA200 below EMA400 |

### Cycle 65 — BUY (started 2026-02-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-25 12:15:00 | 888.00 | 882.01 | 881.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-25 14:15:00 | 904.75 | 888.49 | 884.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-26 10:15:00 | 891.30 | 892.66 | 887.87 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-26 10:15:00 | 891.30 | 892.66 | 887.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 10:15:00 | 891.30 | 892.66 | 887.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-26 11:00:00 | 891.30 | 892.66 | 887.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 11:15:00 | 886.70 | 891.46 | 887.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-26 11:45:00 | 882.55 | 891.46 | 887.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 12:15:00 | 881.15 | 889.40 | 887.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-26 12:45:00 | 880.80 | 889.40 | 887.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 14:15:00 | 883.55 | 888.39 | 887.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-26 15:00:00 | 883.55 | 888.39 | 887.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 15:15:00 | 882.45 | 887.20 | 886.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-27 09:15:00 | 879.50 | 887.20 | 886.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 66 — SELL (started 2026-02-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-27 09:15:00 | 877.85 | 885.33 | 885.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-27 10:15:00 | 872.90 | 882.85 | 884.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-02 15:15:00 | 860.00 | 854.77 | 863.95 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-04 09:15:00 | 828.95 | 854.77 | 863.95 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 15:15:00 | 845.00 | 834.78 | 841.17 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-03-05 15:15:00 | 845.00 | 834.78 | 841.17 | SL hit (close>ema400) qty=1.00 sl=841.17 alert=retest1 |
| ALERT3_SIDEWAYS | 2026-03-06 09:15:00 | 849.55 | 834.78 | 841.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 09:15:00 | 845.35 | 836.89 | 841.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-06 09:30:00 | 846.20 | 836.89 | 841.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 10:15:00 | 823.00 | 834.11 | 839.86 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-06 13:45:00 | 821.20 | 829.33 | 836.02 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-06 14:15:00 | 819.40 | 829.33 | 836.02 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-09 09:15:00 | 798.30 | 826.53 | 833.48 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-09 13:15:00 | 819.95 | 816.29 | 825.30 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-09 14:15:00 | 820.95 | 817.85 | 824.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-09 14:30:00 | 825.15 | 817.85 | 824.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-09 15:15:00 | 824.90 | 819.26 | 824.52 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-10 09:30:00 | 819.80 | 817.65 | 823.31 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-10 14:30:00 | 819.00 | 815.90 | 820.04 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-12 12:30:00 | 819.15 | 812.00 | 813.67 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-12 13:15:00 | 828.70 | 815.34 | 815.03 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-12 13:15:00 | 828.70 | 815.34 | 815.03 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-12 13:15:00 | 828.70 | 815.34 | 815.03 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-12 13:15:00 | 828.70 | 815.34 | 815.03 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-12 13:15:00 | 828.70 | 815.34 | 815.03 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-12 13:15:00 | 828.70 | 815.34 | 815.03 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-12 13:15:00 | 828.70 | 815.34 | 815.03 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 67 — BUY (started 2026-03-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-12 13:15:00 | 828.70 | 815.34 | 815.03 | EMA200 above EMA400 |

### Cycle 68 — SELL (started 2026-03-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-13 11:15:00 | 805.55 | 814.35 | 815.25 | EMA200 below EMA400 |

### Cycle 69 — BUY (started 2026-03-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-13 14:15:00 | 843.10 | 820.38 | 817.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-18 10:15:00 | 857.25 | 842.62 | 835.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-18 15:15:00 | 850.00 | 850.98 | 843.18 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-19 09:15:00 | 844.55 | 850.98 | 843.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 09:15:00 | 846.00 | 849.99 | 843.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-19 09:30:00 | 840.15 | 849.99 | 843.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 15:15:00 | 848.00 | 849.86 | 846.39 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-20 09:15:00 | 851.70 | 849.86 | 846.39 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-20 09:45:00 | 849.75 | 849.93 | 846.74 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-20 11:15:00 | 849.65 | 849.32 | 846.75 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-20 12:15:00 | 836.05 | 845.39 | 845.33 | SL hit (close<static) qty=1.00 sl=837.60 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-20 12:15:00 | 836.05 | 845.39 | 845.33 | SL hit (close<static) qty=1.00 sl=837.60 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-20 12:15:00 | 836.05 | 845.39 | 845.33 | SL hit (close<static) qty=1.00 sl=837.60 alert=retest2 |

### Cycle 70 — SELL (started 2026-03-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-20 13:15:00 | 840.15 | 844.34 | 844.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-20 14:15:00 | 833.15 | 842.10 | 843.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 09:15:00 | 836.10 | 822.60 | 829.21 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-24 09:15:00 | 836.10 | 822.60 | 829.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 09:15:00 | 836.10 | 822.60 | 829.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 10:00:00 | 836.10 | 822.60 | 829.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 10:15:00 | 848.55 | 827.79 | 830.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 10:45:00 | 841.75 | 827.79 | 830.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 71 — BUY (started 2026-03-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-24 11:15:00 | 861.70 | 834.58 | 833.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-24 12:15:00 | 874.00 | 842.46 | 837.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-25 14:15:00 | 857.50 | 865.01 | 855.64 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-25 14:15:00 | 857.50 | 865.01 | 855.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 14:15:00 | 857.50 | 865.01 | 855.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-25 15:00:00 | 857.50 | 865.01 | 855.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 15:15:00 | 855.00 | 863.01 | 855.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 09:15:00 | 864.85 | 863.01 | 855.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 09:15:00 | 859.35 | 862.28 | 855.92 | EMA400 retest candle locked (from upside) |

### Cycle 72 — SELL (started 2026-03-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 12:15:00 | 839.80 | 852.58 | 852.62 | EMA200 below EMA400 |

### Cycle 73 — BUY (started 2026-03-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-27 13:15:00 | 858.55 | 853.78 | 853.16 | EMA200 above EMA400 |

### Cycle 74 — SELL (started 2026-03-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 14:15:00 | 845.60 | 852.14 | 852.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-27 15:15:00 | 839.00 | 849.51 | 851.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 836.35 | 823.36 | 833.01 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 836.35 | 823.36 | 833.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 836.35 | 823.36 | 833.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 09:45:00 | 843.50 | 823.36 | 833.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 10:15:00 | 820.65 | 822.82 | 831.88 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-01 13:15:00 | 816.10 | 822.00 | 829.92 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-01 14:00:00 | 818.00 | 821.20 | 828.84 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-02 12:15:00 | 817.35 | 816.78 | 823.37 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-02 13:00:00 | 817.90 | 817.01 | 822.87 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 13:15:00 | 833.85 | 820.38 | 823.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-02 14:00:00 | 833.85 | 820.38 | 823.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 14:15:00 | 839.45 | 824.19 | 825.29 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-04-02 14:15:00 | 839.45 | 824.19 | 825.29 | SL hit (close>static) qty=1.00 sl=836.35 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-02 14:15:00 | 839.45 | 824.19 | 825.29 | SL hit (close>static) qty=1.00 sl=836.35 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-02 14:15:00 | 839.45 | 824.19 | 825.29 | SL hit (close>static) qty=1.00 sl=836.35 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-02 14:15:00 | 839.45 | 824.19 | 825.29 | SL hit (close>static) qty=1.00 sl=836.35 alert=retest2 |
| ALERT3_SIDEWAYS | 2026-04-02 15:00:00 | 839.45 | 824.19 | 825.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 75 — BUY (started 2026-04-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-02 15:15:00 | 842.00 | 827.75 | 826.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-06 12:15:00 | 849.90 | 834.36 | 830.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-07 14:15:00 | 867.15 | 867.23 | 853.44 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-07 15:00:00 | 867.15 | 867.23 | 853.44 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 09:15:00 | 853.95 | 870.29 | 864.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-09 10:00:00 | 853.95 | 870.29 | 864.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 10:15:00 | 868.50 | 869.93 | 864.52 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-09 11:15:00 | 868.95 | 869.93 | 864.52 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-09 11:45:00 | 868.70 | 869.64 | 864.88 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-09 12:15:00 | 869.25 | 869.64 | 864.88 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-09 13:00:00 | 871.15 | 869.94 | 865.45 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 13:15:00 | 879.35 | 871.82 | 866.71 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-10 11:00:00 | 892.10 | 878.88 | 871.87 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-10 12:00:00 | 888.55 | 880.82 | 873.38 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-10 12:30:00 | 888.50 | 882.40 | 874.78 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 10:15:00 | 889.55 | 887.53 | 880.17 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 15:15:00 | 890.00 | 891.40 | 885.74 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-15 09:15:00 | 906.15 | 891.40 | 885.74 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-16 13:45:00 | 902.90 | 902.44 | 897.89 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2026-04-24 09:15:00 | 955.85 | 941.82 | 931.30 | Target hit (10%) qty=1.00 alert=retest2 |
| Target hit | 2026-04-24 09:15:00 | 955.57 | 941.82 | 931.30 | Target hit (10%) qty=1.00 alert=retest2 |
| Target hit | 2026-04-24 09:15:00 | 956.18 | 941.82 | 931.30 | Target hit (10%) qty=1.00 alert=retest2 |
| Target hit | 2026-04-24 09:15:00 | 958.27 | 941.82 | 931.30 | Target hit (10%) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-29 09:15:00 | 931.70 | 941.34 | 942.06 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-29 09:15:00 | 931.70 | 941.34 | 942.06 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-29 09:15:00 | 931.70 | 941.34 | 942.06 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-29 09:15:00 | 931.70 | 941.34 | 942.06 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-29 09:15:00 | 931.70 | 941.34 | 942.06 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-29 09:15:00 | 931.70 | 941.34 | 942.06 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 76 — SELL (started 2026-04-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-29 09:15:00 | 931.70 | 941.34 | 942.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-29 12:15:00 | 926.60 | 935.21 | 938.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-30 11:15:00 | 929.50 | 928.95 | 933.33 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-30 11:30:00 | 925.90 | 928.95 | 933.33 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 09:15:00 | 939.85 | 930.78 | 932.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-04 10:00:00 | 939.85 | 930.78 | 932.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 10:15:00 | 941.40 | 932.91 | 933.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-04 11:00:00 | 941.40 | 932.91 | 933.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 77 — BUY (started 2026-05-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-04 12:15:00 | 936.20 | 933.87 | 933.64 | EMA200 above EMA400 |

### Cycle 78 — SELL (started 2026-05-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-04 13:15:00 | 931.60 | 933.41 | 933.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-05-05 09:15:00 | 928.00 | 931.71 | 932.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-05-05 10:15:00 | 932.45 | 931.86 | 932.59 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-05 10:15:00 | 932.45 | 931.86 | 932.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 10:15:00 | 932.45 | 931.86 | 932.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-05 10:45:00 | 930.60 | 931.86 | 932.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 11:15:00 | 932.85 | 932.06 | 932.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-05 12:00:00 | 932.85 | 932.06 | 932.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 12:15:00 | 932.50 | 932.15 | 932.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-05 12:30:00 | 932.75 | 932.15 | 932.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 79 — BUY (started 2026-05-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-05 13:15:00 | 938.20 | 933.36 | 933.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-05 15:15:00 | 945.00 | 936.04 | 934.39 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-05-22 10:45:00 | 1034.90 | 2025-05-27 10:15:00 | 1138.39 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-06-09 10:15:00 | 1135.60 | 2025-06-10 10:15:00 | 1151.00 | STOP_HIT | 1.00 | -1.36% |
| SELL | retest2 | 2025-06-09 15:15:00 | 1141.30 | 2025-06-10 10:15:00 | 1151.00 | STOP_HIT | 1.00 | -0.85% |
| SELL | retest2 | 2025-06-12 15:00:00 | 1128.20 | 2025-06-13 09:15:00 | 1071.79 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-06-12 15:00:00 | 1128.20 | 2025-06-16 09:15:00 | 1130.10 | STOP_HIT | 0.50 | -0.17% |
| SELL | retest2 | 2025-06-13 09:45:00 | 1118.10 | 2025-06-17 11:15:00 | 1130.40 | STOP_HIT | 1.00 | -1.10% |
| SELL | retest2 | 2025-06-16 10:45:00 | 1125.90 | 2025-06-17 11:15:00 | 1130.40 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest2 | 2025-06-16 11:30:00 | 1111.20 | 2025-06-17 11:15:00 | 1130.40 | STOP_HIT | 1.00 | -1.73% |
| BUY | retest2 | 2025-06-19 09:15:00 | 1130.70 | 2025-06-20 14:15:00 | 1107.80 | STOP_HIT | 1.00 | -2.03% |
| BUY | retest2 | 2025-06-19 10:45:00 | 1141.10 | 2025-06-20 14:15:00 | 1107.80 | STOP_HIT | 1.00 | -2.92% |
| BUY | retest2 | 2025-06-19 13:45:00 | 1130.00 | 2025-06-20 14:15:00 | 1107.80 | STOP_HIT | 1.00 | -1.96% |
| BUY | retest2 | 2025-06-20 09:30:00 | 1131.00 | 2025-06-20 14:15:00 | 1107.80 | STOP_HIT | 1.00 | -2.05% |
| BUY | retest2 | 2025-06-30 12:15:00 | 1215.50 | 2025-07-04 13:15:00 | 1190.90 | STOP_HIT | 1.00 | -2.02% |
| BUY | retest2 | 2025-07-17 14:30:00 | 1201.80 | 2025-07-22 15:15:00 | 1205.90 | STOP_HIT | 1.00 | 0.34% |
| BUY | retest2 | 2025-07-17 15:00:00 | 1201.90 | 2025-07-22 15:15:00 | 1205.90 | STOP_HIT | 1.00 | 0.33% |
| BUY | retest2 | 2025-07-18 11:30:00 | 1207.00 | 2025-07-22 15:15:00 | 1205.90 | STOP_HIT | 1.00 | -0.09% |
| SELL | retest2 | 2025-07-28 09:15:00 | 1168.60 | 2025-07-28 11:15:00 | 1189.80 | STOP_HIT | 1.00 | -1.81% |
| SELL | retest2 | 2025-07-28 10:30:00 | 1167.20 | 2025-07-28 11:15:00 | 1189.80 | STOP_HIT | 1.00 | -1.94% |
| SELL | retest2 | 2025-08-05 11:30:00 | 1129.10 | 2025-08-11 09:15:00 | 1072.64 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-05 11:30:00 | 1129.10 | 2025-08-11 13:15:00 | 1095.10 | STOP_HIT | 0.50 | 3.01% |
| BUY | retest2 | 2025-09-04 14:00:00 | 1061.50 | 2025-09-16 10:15:00 | 1115.80 | STOP_HIT | 1.00 | 5.12% |
| BUY | retest2 | 2025-09-05 09:30:00 | 1065.70 | 2025-09-16 10:15:00 | 1115.80 | STOP_HIT | 1.00 | 4.70% |
| BUY | retest2 | 2025-09-05 11:00:00 | 1061.00 | 2025-09-16 10:15:00 | 1115.80 | STOP_HIT | 1.00 | 5.16% |
| SELL | retest2 | 2025-09-18 13:00:00 | 1099.30 | 2025-09-18 14:15:00 | 1116.80 | STOP_HIT | 1.00 | -1.59% |
| SELL | retest2 | 2025-09-19 15:00:00 | 1098.20 | 2025-09-26 09:15:00 | 1043.29 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-22 10:15:00 | 1100.00 | 2025-09-26 09:15:00 | 1045.00 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-19 15:00:00 | 1098.20 | 2025-09-29 09:15:00 | 1061.40 | STOP_HIT | 0.50 | 3.35% |
| SELL | retest2 | 2025-09-22 10:15:00 | 1100.00 | 2025-09-29 09:15:00 | 1061.40 | STOP_HIT | 0.50 | 3.51% |
| BUY | retest2 | 2025-10-07 09:15:00 | 1102.60 | 2025-10-07 10:15:00 | 1087.80 | STOP_HIT | 1.00 | -1.34% |
| SELL | retest2 | 2025-10-16 13:45:00 | 1070.80 | 2025-10-20 09:15:00 | 1080.90 | STOP_HIT | 1.00 | -0.94% |
| SELL | retest2 | 2025-10-16 14:45:00 | 1068.40 | 2025-10-20 09:15:00 | 1080.90 | STOP_HIT | 1.00 | -1.17% |
| SELL | retest2 | 2025-10-17 10:00:00 | 1068.70 | 2025-10-20 09:15:00 | 1080.90 | STOP_HIT | 1.00 | -1.14% |
| BUY | retest2 | 2025-10-27 12:30:00 | 1125.70 | 2025-10-30 15:15:00 | 1118.20 | STOP_HIT | 1.00 | -0.67% |
| BUY | retest2 | 2025-10-28 09:15:00 | 1141.00 | 2025-10-30 15:15:00 | 1118.20 | STOP_HIT | 1.00 | -2.00% |
| BUY | retest2 | 2025-10-28 10:30:00 | 1127.10 | 2025-10-30 15:15:00 | 1118.20 | STOP_HIT | 1.00 | -0.79% |
| BUY | retest2 | 2025-10-29 09:15:00 | 1127.40 | 2025-10-30 15:15:00 | 1118.20 | STOP_HIT | 1.00 | -0.82% |
| BUY | retest2 | 2025-10-29 13:30:00 | 1131.50 | 2025-10-30 15:15:00 | 1118.20 | STOP_HIT | 1.00 | -1.18% |
| SELL | retest2 | 2025-11-11 10:15:00 | 1132.40 | 2025-11-12 15:15:00 | 1140.50 | STOP_HIT | 1.00 | -0.72% |
| SELL | retest2 | 2025-11-12 12:45:00 | 1133.90 | 2025-11-12 15:15:00 | 1140.50 | STOP_HIT | 1.00 | -0.58% |
| BUY | retest2 | 2025-11-27 14:45:00 | 1101.10 | 2025-12-02 15:15:00 | 1087.00 | STOP_HIT | 1.00 | -1.28% |
| BUY | retest2 | 2025-11-27 15:15:00 | 1102.20 | 2025-12-02 15:15:00 | 1087.00 | STOP_HIT | 1.00 | -1.38% |
| BUY | retest2 | 2025-12-01 09:15:00 | 1101.40 | 2025-12-02 15:15:00 | 1087.00 | STOP_HIT | 1.00 | -1.31% |
| BUY | retest2 | 2025-12-01 10:00:00 | 1103.00 | 2025-12-02 15:15:00 | 1087.00 | STOP_HIT | 1.00 | -1.45% |
| BUY | retest2 | 2025-12-01 12:45:00 | 1106.30 | 2025-12-02 15:15:00 | 1087.00 | STOP_HIT | 1.00 | -1.74% |
| BUY | retest2 | 2025-12-01 14:15:00 | 1106.10 | 2025-12-02 15:15:00 | 1087.00 | STOP_HIT | 1.00 | -1.73% |
| BUY | retest2 | 2025-12-01 15:00:00 | 1107.20 | 2025-12-02 15:15:00 | 1087.00 | STOP_HIT | 1.00 | -1.82% |
| BUY | retest2 | 2025-12-02 09:30:00 | 1107.80 | 2025-12-02 15:15:00 | 1087.00 | STOP_HIT | 1.00 | -1.88% |
| SELL | retest2 | 2025-12-04 12:15:00 | 1076.40 | 2025-12-09 09:15:00 | 1022.58 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-04 13:45:00 | 1077.60 | 2025-12-09 09:15:00 | 1023.72 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-04 14:30:00 | 1075.70 | 2025-12-09 09:15:00 | 1021.91 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-05 14:30:00 | 1077.40 | 2025-12-09 09:15:00 | 1023.53 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-08 09:15:00 | 1069.40 | 2025-12-09 09:15:00 | 1015.93 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-04 12:15:00 | 1076.40 | 2025-12-09 12:15:00 | 1048.90 | STOP_HIT | 0.50 | 2.55% |
| SELL | retest2 | 2025-12-04 13:45:00 | 1077.60 | 2025-12-09 12:15:00 | 1048.90 | STOP_HIT | 0.50 | 2.66% |
| SELL | retest2 | 2025-12-04 14:30:00 | 1075.70 | 2025-12-09 12:15:00 | 1048.90 | STOP_HIT | 0.50 | 2.49% |
| SELL | retest2 | 2025-12-05 14:30:00 | 1077.40 | 2025-12-09 12:15:00 | 1048.90 | STOP_HIT | 0.50 | 2.65% |
| SELL | retest2 | 2025-12-08 09:15:00 | 1069.40 | 2025-12-09 12:15:00 | 1048.90 | STOP_HIT | 0.50 | 1.92% |
| BUY | retest2 | 2025-12-11 10:30:00 | 1084.80 | 2025-12-16 10:15:00 | 1065.50 | STOP_HIT | 1.00 | -1.78% |
| BUY | retest2 | 2025-12-23 12:00:00 | 1083.30 | 2025-12-24 12:15:00 | 1073.20 | STOP_HIT | 1.00 | -0.93% |
| BUY | retest2 | 2025-12-23 13:45:00 | 1084.00 | 2025-12-24 12:15:00 | 1073.20 | STOP_HIT | 1.00 | -1.00% |
| BUY | retest2 | 2025-12-23 15:15:00 | 1086.00 | 2025-12-24 12:15:00 | 1073.20 | STOP_HIT | 1.00 | -1.18% |
| SELL | retest2 | 2025-12-29 09:15:00 | 1065.10 | 2025-12-31 13:15:00 | 1072.70 | STOP_HIT | 1.00 | -0.71% |
| SELL | retest2 | 2026-01-12 15:15:00 | 1044.00 | 2026-01-14 11:15:00 | 1066.80 | STOP_HIT | 1.00 | -2.18% |
| SELL | retest2 | 2026-01-13 10:30:00 | 1043.00 | 2026-01-14 11:15:00 | 1066.80 | STOP_HIT | 1.00 | -2.28% |
| SELL | retest2 | 2026-01-13 11:30:00 | 1044.00 | 2026-01-14 11:15:00 | 1066.80 | STOP_HIT | 1.00 | -2.18% |
| SELL | retest2 | 2026-01-13 13:30:00 | 1044.80 | 2026-01-14 11:15:00 | 1066.80 | STOP_HIT | 1.00 | -2.11% |
| SELL | retest2 | 2026-01-23 09:30:00 | 957.30 | 2026-01-28 13:15:00 | 960.70 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest2 | 2026-01-23 14:30:00 | 960.60 | 2026-01-28 13:15:00 | 960.70 | STOP_HIT | 1.00 | -0.01% |
| SELL | retest2 | 2026-01-23 15:00:00 | 948.30 | 2026-01-28 13:15:00 | 960.70 | STOP_HIT | 1.00 | -1.31% |
| SELL | retest2 | 2026-02-10 09:30:00 | 938.00 | 2026-02-12 14:15:00 | 940.05 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest2 | 2026-02-10 11:15:00 | 937.10 | 2026-02-16 15:15:00 | 891.10 | PARTIAL | 0.50 | 4.91% |
| SELL | retest2 | 2026-02-10 14:30:00 | 938.65 | 2026-02-16 15:15:00 | 890.25 | PARTIAL | 0.50 | 5.16% |
| SELL | retest2 | 2026-02-11 14:30:00 | 934.95 | 2026-02-16 15:15:00 | 891.72 | PARTIAL | 0.50 | 4.62% |
| SELL | retest2 | 2026-02-12 09:15:00 | 926.35 | 2026-02-16 15:15:00 | 888.20 | PARTIAL | 0.50 | 4.12% |
| SELL | retest2 | 2026-02-10 11:15:00 | 937.10 | 2026-02-17 09:15:00 | 906.05 | STOP_HIT | 0.50 | 3.31% |
| SELL | retest2 | 2026-02-10 14:30:00 | 938.65 | 2026-02-17 09:15:00 | 906.05 | STOP_HIT | 0.50 | 3.47% |
| SELL | retest2 | 2026-02-11 14:30:00 | 934.95 | 2026-02-17 09:15:00 | 906.05 | STOP_HIT | 0.50 | 3.09% |
| SELL | retest2 | 2026-02-12 09:15:00 | 926.35 | 2026-02-17 09:15:00 | 906.05 | STOP_HIT | 0.50 | 2.19% |
| SELL | retest2 | 2026-02-13 09:15:00 | 924.25 | 2026-02-19 12:15:00 | 878.04 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-13 09:15:00 | 924.25 | 2026-02-20 14:15:00 | 870.90 | STOP_HIT | 0.50 | 5.77% |
| SELL | retest1 | 2026-03-04 09:15:00 | 828.95 | 2026-03-05 15:15:00 | 845.00 | STOP_HIT | 1.00 | -1.94% |
| SELL | retest2 | 2026-03-06 13:45:00 | 821.20 | 2026-03-12 13:15:00 | 828.70 | STOP_HIT | 1.00 | -0.91% |
| SELL | retest2 | 2026-03-06 14:15:00 | 819.40 | 2026-03-12 13:15:00 | 828.70 | STOP_HIT | 1.00 | -1.13% |
| SELL | retest2 | 2026-03-09 09:15:00 | 798.30 | 2026-03-12 13:15:00 | 828.70 | STOP_HIT | 1.00 | -3.81% |
| SELL | retest2 | 2026-03-09 13:15:00 | 819.95 | 2026-03-12 13:15:00 | 828.70 | STOP_HIT | 1.00 | -1.07% |
| SELL | retest2 | 2026-03-10 09:30:00 | 819.80 | 2026-03-12 13:15:00 | 828.70 | STOP_HIT | 1.00 | -1.09% |
| SELL | retest2 | 2026-03-10 14:30:00 | 819.00 | 2026-03-12 13:15:00 | 828.70 | STOP_HIT | 1.00 | -1.18% |
| SELL | retest2 | 2026-03-12 12:30:00 | 819.15 | 2026-03-12 13:15:00 | 828.70 | STOP_HIT | 1.00 | -1.17% |
| BUY | retest2 | 2026-03-20 09:15:00 | 851.70 | 2026-03-20 12:15:00 | 836.05 | STOP_HIT | 1.00 | -1.84% |
| BUY | retest2 | 2026-03-20 09:45:00 | 849.75 | 2026-03-20 12:15:00 | 836.05 | STOP_HIT | 1.00 | -1.61% |
| BUY | retest2 | 2026-03-20 11:15:00 | 849.65 | 2026-03-20 12:15:00 | 836.05 | STOP_HIT | 1.00 | -1.60% |
| SELL | retest2 | 2026-04-01 13:15:00 | 816.10 | 2026-04-02 14:15:00 | 839.45 | STOP_HIT | 1.00 | -2.86% |
| SELL | retest2 | 2026-04-01 14:00:00 | 818.00 | 2026-04-02 14:15:00 | 839.45 | STOP_HIT | 1.00 | -2.62% |
| SELL | retest2 | 2026-04-02 12:15:00 | 817.35 | 2026-04-02 14:15:00 | 839.45 | STOP_HIT | 1.00 | -2.70% |
| SELL | retest2 | 2026-04-02 13:00:00 | 817.90 | 2026-04-02 14:15:00 | 839.45 | STOP_HIT | 1.00 | -2.63% |
| BUY | retest2 | 2026-04-09 11:15:00 | 868.95 | 2026-04-24 09:15:00 | 955.85 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-09 11:45:00 | 868.70 | 2026-04-24 09:15:00 | 955.57 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-09 12:15:00 | 869.25 | 2026-04-24 09:15:00 | 956.18 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-09 13:00:00 | 871.15 | 2026-04-24 09:15:00 | 958.27 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-10 11:00:00 | 892.10 | 2026-04-29 09:15:00 | 931.70 | STOP_HIT | 1.00 | 4.44% |
| BUY | retest2 | 2026-04-10 12:00:00 | 888.55 | 2026-04-29 09:15:00 | 931.70 | STOP_HIT | 1.00 | 4.86% |
| BUY | retest2 | 2026-04-10 12:30:00 | 888.50 | 2026-04-29 09:15:00 | 931.70 | STOP_HIT | 1.00 | 4.86% |
| BUY | retest2 | 2026-04-13 10:15:00 | 889.55 | 2026-04-29 09:15:00 | 931.70 | STOP_HIT | 1.00 | 4.74% |
| BUY | retest2 | 2026-04-15 09:15:00 | 906.15 | 2026-04-29 09:15:00 | 931.70 | STOP_HIT | 1.00 | 2.82% |
| BUY | retest2 | 2026-04-16 13:45:00 | 902.90 | 2026-04-29 09:15:00 | 931.70 | STOP_HIT | 1.00 | 3.19% |
