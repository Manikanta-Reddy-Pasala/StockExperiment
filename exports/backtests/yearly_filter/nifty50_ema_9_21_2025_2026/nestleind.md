# NESTLEIND (NESTLEIND)

## Backtest Summary

- **Window:** 2025-03-13 09:15:00 → 2026-05-08 15:15:00 (1976 bars)
- **Last close:** 1475.30
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 75 |
| ALERT1 | 52 |
| ALERT2 | 51 |
| ALERT2_SKIP | 19 |
| ALERT3 | 146 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 54 |
| PARTIAL | 1 |
| TARGET_HIT | 5 |
| STOP_HIT | 51 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 55 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 26 / 29
- **Target hits / Stop hits / Partials:** 5 / 49 / 1
- **Avg / median % per leg:** 0.67% / -0.31%
- **Sum % (uncompounded):** 37.07%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 31 | 15 | 48.4% | 5 | 26 | 0 | 1.68% | 52.2% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 31 | 15 | 48.4% | 5 | 26 | 0 | 1.68% | 52.2% |
| SELL (all) | 24 | 11 | 45.8% | 0 | 23 | 1 | -0.63% | -15.1% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 24 | 11 | 45.8% | 0 | 23 | 1 | -0.63% | -15.1% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 55 | 26 | 47.3% | 5 | 49 | 1 | 0.67% | 37.1% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 10:15:00 | 1179.50 | 1167.46 | 1166.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 14:15:00 | 1192.85 | 1178.42 | 1172.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-13 10:15:00 | 1181.05 | 1181.41 | 1175.67 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-13 10:15:00 | 1181.05 | 1181.41 | 1175.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-13 10:15:00 | 1181.05 | 1181.41 | 1175.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-13 10:30:00 | 1177.50 | 1181.41 | 1175.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-13 12:15:00 | 1182.90 | 1182.09 | 1177.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-13 12:30:00 | 1180.20 | 1182.09 | 1177.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-14 09:15:00 | 1184.15 | 1183.27 | 1179.24 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-15 13:00:00 | 1196.35 | 1184.94 | 1182.23 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-20 13:15:00 | 1188.00 | 1196.60 | 1196.84 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 2 — SELL (started 2025-05-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-20 13:15:00 | 1188.00 | 1196.60 | 1196.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-20 14:15:00 | 1178.25 | 1192.93 | 1195.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-21 09:15:00 | 1192.95 | 1190.87 | 1193.70 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-21 09:15:00 | 1192.95 | 1190.87 | 1193.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 09:15:00 | 1192.95 | 1190.87 | 1193.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-21 10:00:00 | 1192.95 | 1190.87 | 1193.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 10:15:00 | 1188.60 | 1190.41 | 1193.23 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-22 09:15:00 | 1176.25 | 1192.26 | 1193.11 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-23 09:15:00 | 1197.50 | 1185.12 | 1187.43 | SL hit (close>static) qty=1.00 sl=1195.05 alert=retest2 |

### Cycle 3 — BUY (started 2025-05-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-23 10:15:00 | 1205.90 | 1189.28 | 1189.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-26 09:15:00 | 1221.60 | 1202.96 | 1196.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-27 12:15:00 | 1221.10 | 1224.24 | 1215.41 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-27 13:00:00 | 1221.10 | 1224.24 | 1215.41 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 09:15:00 | 1214.45 | 1223.85 | 1218.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-28 09:45:00 | 1214.20 | 1223.85 | 1218.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 10:15:00 | 1209.35 | 1220.95 | 1217.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-28 11:00:00 | 1209.35 | 1220.95 | 1217.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 4 — SELL (started 2025-05-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-28 14:15:00 | 1209.25 | 1214.56 | 1215.07 | EMA200 below EMA400 |

### Cycle 5 — BUY (started 2025-05-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-29 15:15:00 | 1218.90 | 1214.81 | 1214.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-30 09:15:00 | 1221.70 | 1216.19 | 1215.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-30 10:15:00 | 1216.10 | 1216.17 | 1215.27 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-30 10:15:00 | 1216.10 | 1216.17 | 1215.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 10:15:00 | 1216.10 | 1216.17 | 1215.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-30 10:30:00 | 1216.65 | 1216.17 | 1215.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 11:15:00 | 1210.05 | 1214.95 | 1214.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-30 12:00:00 | 1210.05 | 1214.95 | 1214.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 6 — SELL (started 2025-05-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-30 12:15:00 | 1208.20 | 1213.60 | 1214.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-30 13:15:00 | 1207.40 | 1212.36 | 1213.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-03 13:15:00 | 1201.10 | 1200.61 | 1203.77 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-03 13:45:00 | 1202.05 | 1200.61 | 1203.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 14:15:00 | 1193.60 | 1199.21 | 1202.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-03 14:30:00 | 1202.60 | 1199.21 | 1202.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 09:15:00 | 1194.00 | 1197.59 | 1201.44 | EMA400 retest candle locked (from downside) |

### Cycle 7 — BUY (started 2025-06-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-06 14:15:00 | 1207.80 | 1200.83 | 1200.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-10 11:15:00 | 1212.60 | 1207.59 | 1204.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-11 12:15:00 | 1213.00 | 1213.41 | 1210.13 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-11 13:00:00 | 1213.00 | 1213.41 | 1210.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 14:15:00 | 1210.60 | 1212.99 | 1210.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-11 14:45:00 | 1209.25 | 1212.99 | 1210.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 15:15:00 | 1205.00 | 1211.39 | 1210.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-12 09:30:00 | 1201.85 | 1209.63 | 1209.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 8 — SELL (started 2025-06-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-12 10:15:00 | 1202.55 | 1208.22 | 1208.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-12 11:15:00 | 1194.25 | 1205.42 | 1207.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-13 14:15:00 | 1188.50 | 1188.05 | 1194.43 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-13 15:00:00 | 1188.50 | 1188.05 | 1194.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 09:15:00 | 1193.00 | 1189.27 | 1193.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 10:00:00 | 1193.00 | 1189.27 | 1193.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 10:15:00 | 1194.50 | 1190.32 | 1193.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 11:00:00 | 1194.50 | 1190.32 | 1193.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 11:15:00 | 1193.70 | 1190.99 | 1193.93 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-16 14:15:00 | 1190.85 | 1191.63 | 1193.72 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-17 09:15:00 | 1190.95 | 1192.49 | 1193.78 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-20 14:15:00 | 1182.20 | 1174.59 | 1173.68 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 9 — BUY (started 2025-06-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-20 14:15:00 | 1182.20 | 1174.59 | 1173.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-20 15:15:00 | 1191.55 | 1177.98 | 1175.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-23 13:15:00 | 1179.50 | 1181.66 | 1178.59 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-23 13:45:00 | 1179.65 | 1181.66 | 1178.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 14:15:00 | 1179.85 | 1181.30 | 1178.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-23 14:30:00 | 1177.50 | 1181.30 | 1178.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 15:15:00 | 1179.00 | 1180.84 | 1178.73 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-24 09:15:00 | 1184.70 | 1180.84 | 1178.73 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-24 09:45:00 | 1182.85 | 1181.59 | 1179.27 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-24 14:30:00 | 1181.50 | 1183.40 | 1181.27 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-24 15:00:00 | 1181.75 | 1183.40 | 1181.27 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 15:15:00 | 1187.00 | 1184.12 | 1181.79 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-25 09:15:00 | 1195.55 | 1184.12 | 1181.79 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-01 13:15:00 | 1205.20 | 1217.89 | 1218.68 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 10 — SELL (started 2025-07-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-01 13:15:00 | 1205.20 | 1217.89 | 1218.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-02 09:15:00 | 1195.60 | 1209.70 | 1214.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-03 09:15:00 | 1199.50 | 1198.24 | 1204.91 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-03 10:15:00 | 1200.60 | 1198.24 | 1204.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 14:15:00 | 1196.30 | 1192.37 | 1196.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-04 15:00:00 | 1196.30 | 1192.37 | 1196.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 15:15:00 | 1196.00 | 1193.10 | 1196.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-07 09:15:00 | 1200.00 | 1193.10 | 1196.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 09:15:00 | 1200.80 | 1194.64 | 1196.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-07 09:30:00 | 1200.70 | 1194.64 | 1196.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 10:15:00 | 1201.55 | 1196.02 | 1197.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-07 10:30:00 | 1200.50 | 1196.02 | 1197.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 11 — BUY (started 2025-07-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-07 12:15:00 | 1208.15 | 1199.98 | 1198.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-07 13:15:00 | 1211.30 | 1202.24 | 1200.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-08 12:15:00 | 1207.00 | 1207.17 | 1204.11 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-08 12:30:00 | 1208.35 | 1207.17 | 1204.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 13:15:00 | 1205.75 | 1206.89 | 1204.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-08 13:30:00 | 1204.50 | 1206.89 | 1204.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 14:15:00 | 1209.70 | 1207.45 | 1204.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-08 14:30:00 | 1205.50 | 1207.45 | 1204.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 09:15:00 | 1209.55 | 1212.41 | 1209.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-10 09:45:00 | 1209.30 | 1212.41 | 1209.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 10:15:00 | 1210.25 | 1211.98 | 1209.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-10 10:45:00 | 1207.95 | 1211.98 | 1209.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 11:15:00 | 1209.00 | 1211.38 | 1209.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-10 11:30:00 | 1208.40 | 1211.38 | 1209.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 12:15:00 | 1207.45 | 1210.59 | 1209.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-10 13:00:00 | 1207.45 | 1210.59 | 1209.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 13:15:00 | 1204.75 | 1209.43 | 1209.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-10 13:30:00 | 1204.25 | 1209.43 | 1209.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 12 — SELL (started 2025-07-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-10 14:15:00 | 1201.50 | 1207.84 | 1208.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-14 09:15:00 | 1193.55 | 1203.28 | 1205.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-15 09:15:00 | 1198.60 | 1198.31 | 1201.38 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-15 11:15:00 | 1198.70 | 1198.41 | 1200.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 11:15:00 | 1198.70 | 1198.41 | 1200.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-15 12:00:00 | 1198.70 | 1198.41 | 1200.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 12:15:00 | 1206.10 | 1199.95 | 1201.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-15 13:00:00 | 1206.10 | 1199.95 | 1201.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 13:15:00 | 1204.65 | 1200.89 | 1201.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-15 13:30:00 | 1206.55 | 1200.89 | 1201.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 13 — BUY (started 2025-07-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-15 14:15:00 | 1208.15 | 1202.34 | 1202.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-15 15:15:00 | 1210.00 | 1203.87 | 1202.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-17 14:15:00 | 1224.40 | 1227.94 | 1221.89 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-17 14:45:00 | 1224.40 | 1227.94 | 1221.89 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 09:15:00 | 1220.35 | 1225.94 | 1222.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-18 09:45:00 | 1219.75 | 1225.94 | 1222.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 10:15:00 | 1219.00 | 1224.55 | 1221.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-18 11:00:00 | 1219.00 | 1224.55 | 1221.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 09:15:00 | 1229.45 | 1229.89 | 1225.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-21 09:30:00 | 1225.15 | 1229.89 | 1225.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 11:15:00 | 1235.10 | 1231.35 | 1227.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-21 11:30:00 | 1228.35 | 1231.35 | 1227.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 09:15:00 | 1233.00 | 1234.03 | 1230.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-22 09:30:00 | 1231.60 | 1234.03 | 1230.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 10:15:00 | 1232.70 | 1233.76 | 1230.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-22 10:30:00 | 1233.50 | 1233.76 | 1230.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 11:15:00 | 1225.55 | 1232.12 | 1230.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-22 12:00:00 | 1225.55 | 1232.12 | 1230.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 12:15:00 | 1224.95 | 1230.69 | 1229.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-22 13:00:00 | 1224.95 | 1230.69 | 1229.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 14 — SELL (started 2025-07-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-22 14:15:00 | 1221.15 | 1227.55 | 1228.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-24 10:15:00 | 1212.50 | 1223.95 | 1225.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-30 09:15:00 | 1116.70 | 1116.44 | 1130.00 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-30 10:00:00 | 1116.70 | 1116.44 | 1130.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 11:15:00 | 1125.25 | 1118.11 | 1122.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-31 12:00:00 | 1125.25 | 1118.11 | 1122.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 12:15:00 | 1129.50 | 1120.39 | 1123.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-31 13:00:00 | 1129.50 | 1120.39 | 1123.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 15 — BUY (started 2025-08-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-01 09:15:00 | 1140.25 | 1126.39 | 1125.48 | EMA200 above EMA400 |

### Cycle 16 — SELL (started 2025-08-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-05 12:15:00 | 1127.65 | 1132.30 | 1132.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-06 10:15:00 | 1124.20 | 1130.01 | 1131.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-07 14:15:00 | 1116.85 | 1115.54 | 1120.64 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-07 14:45:00 | 1118.00 | 1115.54 | 1120.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 15:15:00 | 1120.50 | 1116.53 | 1120.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-08 09:15:00 | 1111.50 | 1116.53 | 1120.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 09:15:00 | 1109.00 | 1115.03 | 1119.57 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-08 13:45:00 | 1102.30 | 1110.78 | 1116.06 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-11 13:15:00 | 1105.00 | 1103.44 | 1109.05 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-11 13:45:00 | 1103.40 | 1103.49 | 1108.56 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-12 10:15:00 | 1105.60 | 1104.75 | 1107.93 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 10:15:00 | 1102.00 | 1104.20 | 1107.39 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-12 12:15:00 | 1099.00 | 1103.46 | 1106.77 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-13 15:00:00 | 1095.00 | 1094.28 | 1098.26 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-18 09:15:00 | 1161.20 | 1105.49 | 1100.89 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 17 — BUY (started 2025-08-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-18 09:15:00 | 1161.20 | 1105.49 | 1100.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-18 10:15:00 | 1166.00 | 1117.59 | 1106.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-19 09:15:00 | 1134.50 | 1139.70 | 1125.49 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-19 09:30:00 | 1131.50 | 1139.70 | 1125.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 10:15:00 | 1163.20 | 1174.73 | 1168.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-22 11:00:00 | 1163.20 | 1174.73 | 1168.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 11:15:00 | 1170.20 | 1173.83 | 1168.93 | EMA400 retest candle locked (from upside) |

### Cycle 18 — SELL (started 2025-08-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-25 09:15:00 | 1159.00 | 1165.60 | 1166.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-25 11:15:00 | 1152.30 | 1161.26 | 1164.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-26 09:15:00 | 1169.90 | 1159.01 | 1161.41 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-26 09:15:00 | 1169.90 | 1159.01 | 1161.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 09:15:00 | 1169.90 | 1159.01 | 1161.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-26 09:45:00 | 1175.00 | 1159.01 | 1161.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 10:15:00 | 1170.20 | 1161.25 | 1162.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-26 10:30:00 | 1169.70 | 1161.25 | 1162.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 19 — BUY (started 2025-08-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-26 11:15:00 | 1176.50 | 1164.30 | 1163.51 | EMA200 above EMA400 |

### Cycle 20 — SELL (started 2025-08-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-28 09:15:00 | 1158.60 | 1163.82 | 1163.93 | EMA200 below EMA400 |

### Cycle 21 — BUY (started 2025-08-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-28 13:15:00 | 1167.60 | 1164.50 | 1164.13 | EMA200 above EMA400 |

### Cycle 22 — SELL (started 2025-08-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-28 15:15:00 | 1160.70 | 1163.32 | 1163.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-29 09:15:00 | 1150.50 | 1160.76 | 1162.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-01 09:15:00 | 1159.50 | 1158.09 | 1159.90 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-01 09:15:00 | 1159.50 | 1158.09 | 1159.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 09:15:00 | 1159.50 | 1158.09 | 1159.90 | EMA400 retest candle locked (from downside) |

### Cycle 23 — BUY (started 2025-09-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-01 11:15:00 | 1168.80 | 1161.52 | 1161.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-01 12:15:00 | 1171.90 | 1163.59 | 1162.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-03 12:15:00 | 1194.50 | 1195.22 | 1186.08 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-03 13:00:00 | 1194.50 | 1195.22 | 1186.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 09:15:00 | 1200.00 | 1207.88 | 1200.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-05 09:30:00 | 1197.90 | 1207.88 | 1200.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 10:15:00 | 1197.10 | 1205.73 | 1200.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-05 11:00:00 | 1197.10 | 1205.73 | 1200.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 11:15:00 | 1193.60 | 1203.30 | 1199.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-05 12:00:00 | 1193.60 | 1203.30 | 1199.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 12:15:00 | 1202.00 | 1203.04 | 1200.01 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-05 13:15:00 | 1204.70 | 1203.04 | 1200.01 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-08 09:30:00 | 1205.80 | 1204.71 | 1202.00 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-08 11:00:00 | 1202.40 | 1204.25 | 1202.03 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-08 11:30:00 | 1202.80 | 1203.64 | 1201.96 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 12:15:00 | 1192.20 | 1201.35 | 1201.07 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-09-08 12:15:00 | 1192.20 | 1201.35 | 1201.07 | SL hit (close<static) qty=1.00 sl=1193.30 alert=retest2 |

### Cycle 24 — SELL (started 2025-09-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-08 13:15:00 | 1193.80 | 1199.84 | 1200.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-08 14:15:00 | 1186.60 | 1197.19 | 1199.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-09 12:15:00 | 1195.80 | 1194.27 | 1196.70 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-09 12:15:00 | 1195.80 | 1194.27 | 1196.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 12:15:00 | 1195.80 | 1194.27 | 1196.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-09 13:00:00 | 1195.80 | 1194.27 | 1196.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 13:15:00 | 1198.90 | 1195.20 | 1196.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-09 14:00:00 | 1198.90 | 1195.20 | 1196.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 14:15:00 | 1201.60 | 1196.48 | 1197.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-09 15:00:00 | 1201.60 | 1196.48 | 1197.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 15:15:00 | 1200.60 | 1197.30 | 1197.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-10 09:15:00 | 1208.20 | 1197.30 | 1197.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 25 — BUY (started 2025-09-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-10 09:15:00 | 1206.00 | 1199.04 | 1198.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-10 11:15:00 | 1216.20 | 1204.16 | 1200.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-12 09:15:00 | 1213.00 | 1217.81 | 1213.12 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-12 09:15:00 | 1213.00 | 1217.81 | 1213.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 09:15:00 | 1213.00 | 1217.81 | 1213.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-12 10:00:00 | 1213.00 | 1217.81 | 1213.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 10:15:00 | 1215.40 | 1217.33 | 1213.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-12 10:30:00 | 1212.00 | 1217.33 | 1213.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 13:15:00 | 1213.30 | 1216.73 | 1214.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-12 14:00:00 | 1213.30 | 1216.73 | 1214.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 14:15:00 | 1217.70 | 1216.92 | 1214.39 | EMA400 retest candle locked (from upside) |

### Cycle 26 — SELL (started 2025-09-15 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-15 15:15:00 | 1211.50 | 1213.62 | 1213.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-16 09:15:00 | 1207.10 | 1212.32 | 1213.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-17 15:15:00 | 1203.40 | 1202.46 | 1205.81 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-17 15:15:00 | 1203.40 | 1202.46 | 1205.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 15:15:00 | 1203.40 | 1202.46 | 1205.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-18 09:15:00 | 1200.50 | 1202.46 | 1205.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 09:15:00 | 1201.90 | 1202.35 | 1205.45 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-19 09:15:00 | 1198.20 | 1206.43 | 1206.55 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-22 09:15:00 | 1198.90 | 1198.46 | 1201.46 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-01 14:15:00 | 1165.00 | 1159.18 | 1159.09 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 27 — BUY (started 2025-10-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-01 14:15:00 | 1165.00 | 1159.18 | 1159.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-06 09:15:00 | 1170.60 | 1166.19 | 1163.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-07 10:15:00 | 1177.70 | 1177.72 | 1172.05 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-07 11:00:00 | 1177.70 | 1177.72 | 1172.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 15:15:00 | 1180.00 | 1180.33 | 1175.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-08 09:15:00 | 1173.20 | 1180.33 | 1175.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 09:15:00 | 1166.40 | 1177.54 | 1174.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-08 10:00:00 | 1166.40 | 1177.54 | 1174.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 10:15:00 | 1169.40 | 1175.91 | 1174.45 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-08 12:00:00 | 1172.70 | 1175.27 | 1174.29 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-08 12:45:00 | 1174.00 | 1175.52 | 1174.49 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-14 10:15:00 | 1181.20 | 1186.64 | 1186.78 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 28 — SELL (started 2025-10-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-14 10:15:00 | 1181.20 | 1186.64 | 1186.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-14 11:15:00 | 1175.40 | 1184.39 | 1185.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-15 09:15:00 | 1197.90 | 1182.53 | 1183.64 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-15 09:15:00 | 1197.90 | 1182.53 | 1183.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 09:15:00 | 1197.90 | 1182.53 | 1183.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 10:00:00 | 1197.90 | 1182.53 | 1183.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 29 — BUY (started 2025-10-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-15 10:15:00 | 1209.90 | 1188.01 | 1186.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-15 13:15:00 | 1222.90 | 1202.45 | 1193.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-20 11:15:00 | 1284.10 | 1284.19 | 1265.61 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-20 11:30:00 | 1282.10 | 1284.19 | 1265.61 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 13:15:00 | 1276.60 | 1284.87 | 1278.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-23 13:45:00 | 1273.70 | 1284.87 | 1278.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 14:15:00 | 1271.20 | 1282.14 | 1277.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-23 15:00:00 | 1271.20 | 1282.14 | 1277.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 15:15:00 | 1275.80 | 1280.87 | 1277.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-24 09:15:00 | 1267.70 | 1280.87 | 1277.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 30 — SELL (started 2025-10-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-24 10:15:00 | 1267.60 | 1275.33 | 1275.61 | EMA200 below EMA400 |

### Cycle 31 — BUY (started 2025-10-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-24 11:15:00 | 1280.80 | 1276.42 | 1276.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-24 14:15:00 | 1282.50 | 1278.36 | 1277.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-27 13:15:00 | 1280.30 | 1282.05 | 1279.95 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-27 13:15:00 | 1280.30 | 1282.05 | 1279.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 13:15:00 | 1280.30 | 1282.05 | 1279.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-27 13:30:00 | 1278.60 | 1282.05 | 1279.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 14:15:00 | 1282.90 | 1282.22 | 1280.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-27 14:45:00 | 1281.50 | 1282.22 | 1280.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 15:15:00 | 1280.00 | 1281.78 | 1280.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-28 09:15:00 | 1284.20 | 1281.78 | 1280.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 09:15:00 | 1274.00 | 1280.22 | 1279.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-28 10:00:00 | 1274.00 | 1280.22 | 1279.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 32 — SELL (started 2025-10-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-28 10:15:00 | 1269.50 | 1278.08 | 1278.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-28 11:15:00 | 1266.30 | 1275.72 | 1277.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-29 09:15:00 | 1273.30 | 1272.67 | 1275.07 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-29 09:15:00 | 1273.30 | 1272.67 | 1275.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 09:15:00 | 1273.30 | 1272.67 | 1275.07 | EMA400 retest candle locked (from downside) |

### Cycle 33 — BUY (started 2025-10-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-30 14:15:00 | 1279.90 | 1276.29 | 1275.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-31 09:15:00 | 1281.60 | 1277.51 | 1276.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-31 10:15:00 | 1275.90 | 1277.19 | 1276.42 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-31 10:15:00 | 1275.90 | 1277.19 | 1276.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 10:15:00 | 1275.90 | 1277.19 | 1276.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-31 11:00:00 | 1275.90 | 1277.19 | 1276.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 11:15:00 | 1277.50 | 1277.25 | 1276.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-31 11:45:00 | 1278.50 | 1277.25 | 1276.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 12:15:00 | 1275.20 | 1276.84 | 1276.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-31 12:45:00 | 1274.80 | 1276.84 | 1276.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 13:15:00 | 1275.10 | 1276.49 | 1276.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-31 13:30:00 | 1272.50 | 1276.49 | 1276.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 34 — SELL (started 2025-10-31 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-31 14:15:00 | 1271.10 | 1275.41 | 1275.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-03 09:15:00 | 1259.20 | 1272.11 | 1274.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-03 14:15:00 | 1266.80 | 1265.58 | 1269.62 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-03 15:00:00 | 1266.80 | 1265.58 | 1269.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 09:15:00 | 1253.00 | 1263.58 | 1268.04 | EMA400 retest candle locked (from downside) |

### Cycle 35 — BUY (started 2025-11-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-06 12:15:00 | 1275.00 | 1268.60 | 1267.85 | EMA200 above EMA400 |

### Cycle 36 — SELL (started 2025-11-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-07 10:15:00 | 1260.50 | 1266.52 | 1267.18 | EMA200 below EMA400 |

### Cycle 37 — BUY (started 2025-11-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-10 12:15:00 | 1269.40 | 1266.47 | 1266.13 | EMA200 above EMA400 |

### Cycle 38 — SELL (started 2025-11-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-10 14:15:00 | 1261.70 | 1265.86 | 1265.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-11 12:15:00 | 1260.70 | 1264.46 | 1265.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-11 13:15:00 | 1265.50 | 1264.67 | 1265.19 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-11 13:15:00 | 1265.50 | 1264.67 | 1265.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 13:15:00 | 1265.50 | 1264.67 | 1265.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-11 13:45:00 | 1265.60 | 1264.67 | 1265.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 14:15:00 | 1268.70 | 1265.47 | 1265.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-11 14:45:00 | 1267.50 | 1265.47 | 1265.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 39 — BUY (started 2025-11-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-11 15:15:00 | 1266.20 | 1265.62 | 1265.57 | EMA200 above EMA400 |

### Cycle 40 — SELL (started 2025-11-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-12 09:15:00 | 1263.30 | 1265.16 | 1265.36 | EMA200 below EMA400 |

### Cycle 41 — BUY (started 2025-11-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-12 10:15:00 | 1280.60 | 1268.24 | 1266.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-13 10:15:00 | 1284.30 | 1277.62 | 1273.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-13 13:15:00 | 1277.00 | 1279.75 | 1275.36 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-13 14:00:00 | 1277.00 | 1279.75 | 1275.36 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 14:15:00 | 1275.50 | 1278.90 | 1275.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-13 15:00:00 | 1275.50 | 1278.90 | 1275.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 15:15:00 | 1274.50 | 1278.02 | 1275.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-14 09:15:00 | 1273.40 | 1278.02 | 1275.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 09:15:00 | 1267.70 | 1275.95 | 1274.60 | EMA400 retest candle locked (from upside) |

### Cycle 42 — SELL (started 2025-11-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-14 12:15:00 | 1268.20 | 1272.95 | 1273.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-18 09:15:00 | 1255.90 | 1266.92 | 1269.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-18 12:15:00 | 1266.90 | 1265.48 | 1268.06 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-18 13:00:00 | 1266.90 | 1265.48 | 1268.06 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 13:15:00 | 1269.50 | 1266.28 | 1268.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-18 14:00:00 | 1269.50 | 1266.28 | 1268.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 14:15:00 | 1265.00 | 1266.03 | 1267.90 | EMA400 retest candle locked (from downside) |

### Cycle 43 — BUY (started 2025-11-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-19 12:15:00 | 1276.90 | 1268.79 | 1268.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-19 14:15:00 | 1278.50 | 1271.96 | 1270.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-20 15:15:00 | 1278.00 | 1278.27 | 1275.21 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-21 09:15:00 | 1280.10 | 1278.27 | 1275.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 09:15:00 | 1279.10 | 1278.44 | 1275.56 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-21 12:30:00 | 1284.90 | 1280.47 | 1277.25 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-21 13:30:00 | 1284.60 | 1281.05 | 1277.81 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-24 11:15:00 | 1271.40 | 1277.60 | 1277.33 | SL hit (close<static) qty=1.00 sl=1273.00 alert=retest2 |

### Cycle 44 — SELL (started 2025-11-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-24 13:15:00 | 1275.70 | 1276.92 | 1277.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-24 14:15:00 | 1268.50 | 1275.23 | 1276.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-25 13:15:00 | 1268.30 | 1266.96 | 1270.73 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-25 13:15:00 | 1268.30 | 1266.96 | 1270.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 13:15:00 | 1268.30 | 1266.96 | 1270.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-25 14:00:00 | 1268.30 | 1266.96 | 1270.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 14:15:00 | 1262.60 | 1266.09 | 1269.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-25 14:30:00 | 1268.50 | 1266.09 | 1269.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 09:15:00 | 1277.10 | 1267.80 | 1270.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-26 10:00:00 | 1277.10 | 1267.80 | 1270.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 10:15:00 | 1278.30 | 1269.90 | 1270.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-26 10:45:00 | 1279.00 | 1269.90 | 1270.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 45 — BUY (started 2025-11-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-26 11:15:00 | 1280.60 | 1272.04 | 1271.70 | EMA200 above EMA400 |

### Cycle 46 — SELL (started 2025-11-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-27 12:15:00 | 1266.30 | 1272.55 | 1272.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-27 13:15:00 | 1263.20 | 1270.68 | 1272.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-01 14:15:00 | 1262.30 | 1259.22 | 1262.76 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-01 15:00:00 | 1262.30 | 1259.22 | 1262.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 15:15:00 | 1259.00 | 1259.18 | 1262.41 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-02 09:15:00 | 1257.90 | 1259.18 | 1262.41 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-02 10:00:00 | 1256.80 | 1258.70 | 1261.90 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-02 11:00:00 | 1258.40 | 1258.64 | 1261.59 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-02 14:15:00 | 1258.20 | 1258.15 | 1260.58 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 15:15:00 | 1258.10 | 1258.45 | 1260.31 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-03 09:15:00 | 1238.90 | 1258.45 | 1260.31 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-12 11:15:00 | 1229.90 | 1220.46 | 1219.45 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 47 — BUY (started 2025-12-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-12 11:15:00 | 1229.90 | 1220.46 | 1219.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-12 12:15:00 | 1233.00 | 1222.97 | 1220.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-16 12:15:00 | 1241.70 | 1242.41 | 1236.48 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-16 13:00:00 | 1241.70 | 1242.41 | 1236.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 15:15:00 | 1241.50 | 1242.18 | 1237.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-17 09:15:00 | 1234.80 | 1242.18 | 1237.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 09:15:00 | 1237.00 | 1241.14 | 1237.78 | EMA400 retest candle locked (from upside) |

### Cycle 48 — SELL (started 2025-12-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-17 13:15:00 | 1228.20 | 1234.74 | 1235.55 | EMA200 below EMA400 |

### Cycle 49 — BUY (started 2025-12-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-18 12:15:00 | 1242.30 | 1236.22 | 1235.76 | EMA200 above EMA400 |

### Cycle 50 — SELL (started 2025-12-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-18 15:15:00 | 1230.80 | 1235.07 | 1235.37 | EMA200 below EMA400 |

### Cycle 51 — BUY (started 2025-12-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-19 10:15:00 | 1239.70 | 1236.15 | 1235.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-19 11:15:00 | 1243.60 | 1237.64 | 1236.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-23 15:15:00 | 1253.60 | 1254.37 | 1250.27 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-24 09:15:00 | 1254.70 | 1254.37 | 1250.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 09:15:00 | 1259.30 | 1255.36 | 1251.09 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-24 11:15:00 | 1262.40 | 1256.12 | 1251.83 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-24 12:15:00 | 1261.50 | 1257.04 | 1252.64 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-26 10:15:00 | 1261.50 | 1258.47 | 1255.31 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-02 15:15:00 | 1280.40 | 1284.49 | 1284.65 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 52 — SELL (started 2026-01-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-02 15:15:00 | 1280.40 | 1284.49 | 1284.65 | EMA200 below EMA400 |

### Cycle 53 — BUY (started 2026-01-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-05 09:15:00 | 1295.50 | 1286.69 | 1285.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-05 10:15:00 | 1308.40 | 1291.03 | 1287.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-06 10:15:00 | 1302.70 | 1306.42 | 1299.23 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-06 10:15:00 | 1302.70 | 1306.42 | 1299.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 10:15:00 | 1302.70 | 1306.42 | 1299.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-06 10:45:00 | 1301.40 | 1306.42 | 1299.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 09:15:00 | 1330.80 | 1316.38 | 1307.52 | EMA400 retest candle locked (from upside) |

### Cycle 54 — SELL (started 2026-01-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-08 13:15:00 | 1306.60 | 1308.94 | 1309.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-09 09:15:00 | 1305.70 | 1307.56 | 1308.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-12 12:15:00 | 1305.00 | 1300.91 | 1302.80 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-12 12:15:00 | 1305.00 | 1300.91 | 1302.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 12:15:00 | 1305.00 | 1300.91 | 1302.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-12 12:30:00 | 1307.30 | 1300.91 | 1302.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 13:15:00 | 1314.40 | 1303.61 | 1303.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-12 14:00:00 | 1314.40 | 1303.61 | 1303.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 55 — BUY (started 2026-01-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-12 14:15:00 | 1314.10 | 1305.71 | 1304.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-14 09:15:00 | 1324.40 | 1315.35 | 1311.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-14 11:15:00 | 1314.60 | 1315.21 | 1311.84 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-14 12:00:00 | 1314.60 | 1315.21 | 1311.84 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 12:15:00 | 1307.90 | 1313.75 | 1311.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-14 12:30:00 | 1307.00 | 1313.75 | 1311.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 13:15:00 | 1305.90 | 1312.18 | 1310.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-14 14:00:00 | 1305.90 | 1312.18 | 1310.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 14:15:00 | 1307.90 | 1311.32 | 1310.69 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-16 09:15:00 | 1316.90 | 1310.46 | 1310.36 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-16 09:45:00 | 1313.00 | 1310.85 | 1310.54 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-19 13:00:00 | 1311.30 | 1313.45 | 1312.69 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-20 10:00:00 | 1311.70 | 1314.09 | 1313.33 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 10:15:00 | 1314.00 | 1314.07 | 1313.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-20 10:30:00 | 1311.80 | 1314.07 | 1313.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 11:15:00 | 1309.20 | 1313.10 | 1313.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-20 12:00:00 | 1309.20 | 1313.10 | 1313.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2026-01-20 12:15:00 | 1304.10 | 1311.30 | 1312.20 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 56 — SELL (started 2026-01-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-20 12:15:00 | 1304.10 | 1311.30 | 1312.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-20 14:15:00 | 1295.20 | 1307.73 | 1310.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-22 09:15:00 | 1290.50 | 1288.83 | 1296.59 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-22 09:30:00 | 1289.00 | 1288.83 | 1296.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 10:15:00 | 1297.40 | 1290.54 | 1296.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 11:00:00 | 1297.40 | 1290.54 | 1296.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 11:15:00 | 1296.40 | 1291.71 | 1296.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 11:30:00 | 1297.90 | 1291.71 | 1296.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 12:15:00 | 1290.30 | 1291.43 | 1296.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 13:15:00 | 1293.50 | 1291.43 | 1296.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 13:15:00 | 1301.50 | 1293.44 | 1296.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 14:00:00 | 1301.50 | 1293.44 | 1296.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 14:15:00 | 1305.80 | 1295.92 | 1297.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 14:45:00 | 1308.20 | 1295.92 | 1297.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 57 — BUY (started 2026-01-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-22 15:15:00 | 1309.00 | 1298.53 | 1298.45 | EMA200 above EMA400 |

### Cycle 58 — SELL (started 2026-01-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-23 12:15:00 | 1295.30 | 1298.00 | 1298.32 | EMA200 below EMA400 |

### Cycle 59 — BUY (started 2026-01-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-27 09:15:00 | 1305.60 | 1298.50 | 1298.31 | EMA200 above EMA400 |

### Cycle 60 — SELL (started 2026-01-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-27 10:15:00 | 1296.10 | 1298.02 | 1298.11 | EMA200 below EMA400 |

### Cycle 61 — BUY (started 2026-01-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-27 11:15:00 | 1302.50 | 1298.91 | 1298.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-27 14:15:00 | 1305.60 | 1300.27 | 1299.24 | Break + close above crossover candle high |

### Cycle 62 — SELL (started 2026-01-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-28 09:15:00 | 1286.80 | 1298.49 | 1298.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-28 10:15:00 | 1277.20 | 1294.24 | 1296.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-28 14:15:00 | 1292.00 | 1289.80 | 1293.47 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-28 14:15:00 | 1292.00 | 1289.80 | 1293.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 14:15:00 | 1292.00 | 1289.80 | 1293.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-28 15:00:00 | 1292.00 | 1289.80 | 1293.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 15:15:00 | 1295.40 | 1290.92 | 1293.64 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-29 09:15:00 | 1283.50 | 1290.92 | 1293.64 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-29 14:45:00 | 1288.50 | 1281.22 | 1286.00 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-30 09:15:00 | 1302.60 | 1286.28 | 1287.52 | SL hit (close>static) qty=1.00 sl=1295.40 alert=retest2 |

### Cycle 63 — BUY (started 2026-01-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-30 10:15:00 | 1303.70 | 1289.76 | 1288.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-30 13:15:00 | 1333.50 | 1303.01 | 1295.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-01 09:15:00 | 1311.00 | 1313.15 | 1302.91 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-01 09:30:00 | 1313.20 | 1313.15 | 1302.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 11:15:00 | 1305.00 | 1311.90 | 1304.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 12:00:00 | 1305.00 | 1311.90 | 1304.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 12:15:00 | 1301.00 | 1309.72 | 1303.85 | EMA400 retest candle locked (from upside) |

### Cycle 64 — SELL (started 2026-02-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-01 15:15:00 | 1271.00 | 1295.62 | 1298.38 | EMA200 below EMA400 |

### Cycle 65 — BUY (started 2026-02-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-02 13:15:00 | 1306.30 | 1298.49 | 1298.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-04 09:15:00 | 1326.60 | 1309.11 | 1304.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-04 12:15:00 | 1311.30 | 1311.53 | 1306.91 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-04 12:45:00 | 1310.70 | 1311.53 | 1306.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 13:15:00 | 1311.10 | 1311.44 | 1307.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-04 13:45:00 | 1308.60 | 1311.44 | 1307.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 14:15:00 | 1303.70 | 1309.89 | 1306.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-04 15:00:00 | 1303.70 | 1309.89 | 1306.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 15:15:00 | 1295.60 | 1307.04 | 1305.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-05 09:15:00 | 1309.60 | 1307.04 | 1305.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 09:15:00 | 1306.20 | 1306.87 | 1305.95 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-05 11:00:00 | 1310.40 | 1307.57 | 1306.36 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-05 13:15:00 | 1299.70 | 1304.51 | 1305.14 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 66 — SELL (started 2026-02-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-05 13:15:00 | 1299.70 | 1304.51 | 1305.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-06 09:15:00 | 1286.30 | 1300.08 | 1302.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-06 13:15:00 | 1296.50 | 1295.32 | 1299.31 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-06 14:00:00 | 1296.50 | 1295.32 | 1299.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 14:15:00 | 1305.00 | 1297.25 | 1299.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-06 15:00:00 | 1305.00 | 1297.25 | 1299.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 15:15:00 | 1300.00 | 1297.80 | 1299.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-09 09:15:00 | 1295.00 | 1297.80 | 1299.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 11:15:00 | 1297.60 | 1296.05 | 1298.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-09 11:45:00 | 1298.00 | 1296.05 | 1298.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 12:15:00 | 1299.90 | 1296.82 | 1298.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-09 13:00:00 | 1299.90 | 1296.82 | 1298.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 13:15:00 | 1295.80 | 1296.62 | 1298.27 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-09 14:30:00 | 1292.60 | 1296.21 | 1297.94 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-10 09:15:00 | 1304.50 | 1297.68 | 1298.29 | SL hit (close>static) qty=1.00 sl=1301.50 alert=retest2 |

### Cycle 67 — BUY (started 2026-02-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-10 12:15:00 | 1301.90 | 1299.17 | 1298.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-10 13:15:00 | 1303.60 | 1300.05 | 1299.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-11 11:15:00 | 1302.50 | 1305.78 | 1302.95 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-11 11:15:00 | 1302.50 | 1305.78 | 1302.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 11:15:00 | 1302.50 | 1305.78 | 1302.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-11 12:00:00 | 1302.50 | 1305.78 | 1302.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 12:15:00 | 1297.30 | 1304.08 | 1302.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-11 13:00:00 | 1297.30 | 1304.08 | 1302.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 13:15:00 | 1301.00 | 1303.46 | 1302.30 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-11 14:15:00 | 1302.80 | 1303.46 | 1302.30 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-12 09:30:00 | 1305.20 | 1304.20 | 1302.97 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-12 10:15:00 | 1296.10 | 1302.58 | 1302.34 | SL hit (close<static) qty=1.00 sl=1297.00 alert=retest2 |

### Cycle 68 — SELL (started 2026-02-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-13 10:15:00 | 1299.30 | 1302.61 | 1302.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-13 11:15:00 | 1291.20 | 1300.33 | 1301.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-16 10:15:00 | 1292.20 | 1289.76 | 1294.69 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-16 11:00:00 | 1292.20 | 1289.76 | 1294.69 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 11:15:00 | 1291.40 | 1290.09 | 1294.39 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-16 12:30:00 | 1290.20 | 1289.87 | 1293.90 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-16 13:45:00 | 1287.60 | 1289.65 | 1293.43 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-17 09:15:00 | 1284.60 | 1289.99 | 1292.94 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-18 14:15:00 | 1301.00 | 1289.10 | 1288.76 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 69 — BUY (started 2026-02-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-18 14:15:00 | 1301.00 | 1289.10 | 1288.76 | EMA200 above EMA400 |

### Cycle 70 — SELL (started 2026-02-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-19 13:15:00 | 1275.80 | 1286.99 | 1288.36 | EMA200 below EMA400 |

### Cycle 71 — BUY (started 2026-02-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-20 11:15:00 | 1297.80 | 1288.91 | 1288.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-23 09:15:00 | 1312.20 | 1296.75 | 1292.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-25 12:15:00 | 1318.70 | 1321.20 | 1314.11 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-25 12:45:00 | 1315.50 | 1321.20 | 1314.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 09:15:00 | 1320.00 | 1321.49 | 1316.55 | EMA400 retest candle locked (from upside) |

### Cycle 72 — SELL (started 2026-02-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-27 09:15:00 | 1292.70 | 1313.50 | 1314.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-27 11:15:00 | 1287.70 | 1305.22 | 1310.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-02 14:15:00 | 1282.80 | 1280.81 | 1291.59 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-02 15:00:00 | 1282.80 | 1280.81 | 1291.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 11:15:00 | 1257.00 | 1249.28 | 1255.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-06 11:45:00 | 1257.40 | 1249.28 | 1255.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 12:15:00 | 1258.90 | 1251.20 | 1255.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-06 13:00:00 | 1258.90 | 1251.20 | 1255.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 13:15:00 | 1261.10 | 1253.18 | 1256.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-06 13:30:00 | 1261.30 | 1253.18 | 1256.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 15:15:00 | 1257.90 | 1253.71 | 1255.80 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-09 09:15:00 | 1232.20 | 1253.71 | 1255.80 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-23 10:15:00 | 1170.59 | 1186.44 | 1191.50 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-24 11:15:00 | 1182.20 | 1175.66 | 1181.50 | SL hit (close>ema200) qty=0.50 sl=1175.66 alert=retest2 |

### Cycle 73 — BUY (started 2026-03-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 09:15:00 | 1198.80 | 1185.82 | 1184.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 10:15:00 | 1204.90 | 1189.64 | 1186.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-27 13:15:00 | 1205.00 | 1205.73 | 1199.54 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-27 13:45:00 | 1204.70 | 1205.73 | 1199.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 14:15:00 | 1190.50 | 1202.68 | 1198.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 15:00:00 | 1190.50 | 1202.68 | 1198.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 15:15:00 | 1194.10 | 1200.97 | 1198.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-30 09:15:00 | 1183.50 | 1200.97 | 1198.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-30 09:15:00 | 1185.50 | 1197.87 | 1197.13 | EMA400 retest candle locked (from upside) |

### Cycle 74 — SELL (started 2026-03-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-30 10:15:00 | 1178.80 | 1194.06 | 1195.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 14:15:00 | 1175.80 | 1185.75 | 1190.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 14:15:00 | 1181.20 | 1179.60 | 1184.49 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-01 15:00:00 | 1181.20 | 1179.60 | 1184.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 12:15:00 | 1189.70 | 1176.89 | 1180.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-02 13:00:00 | 1189.70 | 1176.89 | 1180.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 13:15:00 | 1194.00 | 1180.31 | 1182.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-02 13:30:00 | 1192.70 | 1180.31 | 1182.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 75 — BUY (started 2026-04-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-02 15:15:00 | 1190.00 | 1183.98 | 1183.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-06 12:15:00 | 1205.00 | 1188.63 | 1185.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-08 11:15:00 | 1210.30 | 1216.28 | 1207.85 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-08 12:00:00 | 1210.30 | 1216.28 | 1207.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 09:15:00 | 1215.90 | 1215.10 | 1210.35 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-09 15:00:00 | 1229.50 | 1219.94 | 1214.65 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 10:15:00 | 1226.40 | 1235.13 | 1227.63 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 12:30:00 | 1228.20 | 1231.57 | 1227.67 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 13:30:00 | 1225.70 | 1230.45 | 1227.52 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 14:15:00 | 1225.40 | 1229.44 | 1227.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-13 15:15:00 | 1224.80 | 1229.44 | 1227.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 15:15:00 | 1224.80 | 1228.51 | 1227.09 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-15 09:15:00 | 1235.40 | 1228.51 | 1227.09 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2026-04-21 14:15:00 | 1352.45 | 1314.21 | 1295.08 | Target hit (10%) qty=1.00 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-05-15 13:00:00 | 1196.35 | 2025-05-20 13:15:00 | 1188.00 | STOP_HIT | 1.00 | -0.70% |
| SELL | retest2 | 2025-05-22 09:15:00 | 1176.25 | 2025-05-23 09:15:00 | 1197.50 | STOP_HIT | 1.00 | -1.81% |
| SELL | retest2 | 2025-06-16 14:15:00 | 1190.85 | 2025-06-20 14:15:00 | 1182.20 | STOP_HIT | 1.00 | 0.73% |
| SELL | retest2 | 2025-06-17 09:15:00 | 1190.95 | 2025-06-20 14:15:00 | 1182.20 | STOP_HIT | 1.00 | 0.73% |
| BUY | retest2 | 2025-06-24 09:15:00 | 1184.70 | 2025-07-01 13:15:00 | 1205.20 | STOP_HIT | 1.00 | 1.73% |
| BUY | retest2 | 2025-06-24 09:45:00 | 1182.85 | 2025-07-01 13:15:00 | 1205.20 | STOP_HIT | 1.00 | 1.89% |
| BUY | retest2 | 2025-06-24 14:30:00 | 1181.50 | 2025-07-01 13:15:00 | 1205.20 | STOP_HIT | 1.00 | 2.01% |
| BUY | retest2 | 2025-06-24 15:00:00 | 1181.75 | 2025-07-01 13:15:00 | 1205.20 | STOP_HIT | 1.00 | 1.98% |
| BUY | retest2 | 2025-06-25 09:15:00 | 1195.55 | 2025-07-01 13:15:00 | 1205.20 | STOP_HIT | 1.00 | 0.81% |
| SELL | retest2 | 2025-08-08 13:45:00 | 1102.30 | 2025-08-18 09:15:00 | 1161.20 | STOP_HIT | 1.00 | -5.34% |
| SELL | retest2 | 2025-08-11 13:15:00 | 1105.00 | 2025-08-18 09:15:00 | 1161.20 | STOP_HIT | 1.00 | -5.09% |
| SELL | retest2 | 2025-08-11 13:45:00 | 1103.40 | 2025-08-18 09:15:00 | 1161.20 | STOP_HIT | 1.00 | -5.24% |
| SELL | retest2 | 2025-08-12 10:15:00 | 1105.60 | 2025-08-18 09:15:00 | 1161.20 | STOP_HIT | 1.00 | -5.03% |
| SELL | retest2 | 2025-08-12 12:15:00 | 1099.00 | 2025-08-18 09:15:00 | 1161.20 | STOP_HIT | 1.00 | -5.66% |
| SELL | retest2 | 2025-08-13 15:00:00 | 1095.00 | 2025-08-18 09:15:00 | 1161.20 | STOP_HIT | 1.00 | -6.05% |
| BUY | retest2 | 2025-09-05 13:15:00 | 1204.70 | 2025-09-08 12:15:00 | 1192.20 | STOP_HIT | 1.00 | -1.04% |
| BUY | retest2 | 2025-09-08 09:30:00 | 1205.80 | 2025-09-08 12:15:00 | 1192.20 | STOP_HIT | 1.00 | -1.13% |
| BUY | retest2 | 2025-09-08 11:00:00 | 1202.40 | 2025-09-08 12:15:00 | 1192.20 | STOP_HIT | 1.00 | -0.85% |
| BUY | retest2 | 2025-09-08 11:30:00 | 1202.80 | 2025-09-08 12:15:00 | 1192.20 | STOP_HIT | 1.00 | -0.88% |
| SELL | retest2 | 2025-09-19 09:15:00 | 1198.20 | 2025-10-01 14:15:00 | 1165.00 | STOP_HIT | 1.00 | 2.77% |
| SELL | retest2 | 2025-09-22 09:15:00 | 1198.90 | 2025-10-01 14:15:00 | 1165.00 | STOP_HIT | 1.00 | 2.83% |
| BUY | retest2 | 2025-10-08 12:00:00 | 1172.70 | 2025-10-14 10:15:00 | 1181.20 | STOP_HIT | 1.00 | 0.72% |
| BUY | retest2 | 2025-10-08 12:45:00 | 1174.00 | 2025-10-14 10:15:00 | 1181.20 | STOP_HIT | 1.00 | 0.61% |
| BUY | retest2 | 2025-11-21 12:30:00 | 1284.90 | 2025-11-24 11:15:00 | 1271.40 | STOP_HIT | 1.00 | -1.05% |
| BUY | retest2 | 2025-11-21 13:30:00 | 1284.60 | 2025-11-24 11:15:00 | 1271.40 | STOP_HIT | 1.00 | -1.03% |
| SELL | retest2 | 2025-12-02 09:15:00 | 1257.90 | 2025-12-12 11:15:00 | 1229.90 | STOP_HIT | 1.00 | 2.23% |
| SELL | retest2 | 2025-12-02 10:00:00 | 1256.80 | 2025-12-12 11:15:00 | 1229.90 | STOP_HIT | 1.00 | 2.14% |
| SELL | retest2 | 2025-12-02 11:00:00 | 1258.40 | 2025-12-12 11:15:00 | 1229.90 | STOP_HIT | 1.00 | 2.26% |
| SELL | retest2 | 2025-12-02 14:15:00 | 1258.20 | 2025-12-12 11:15:00 | 1229.90 | STOP_HIT | 1.00 | 2.25% |
| SELL | retest2 | 2025-12-03 09:15:00 | 1238.90 | 2025-12-12 11:15:00 | 1229.90 | STOP_HIT | 1.00 | 0.73% |
| BUY | retest2 | 2025-12-24 11:15:00 | 1262.40 | 2026-01-02 15:15:00 | 1280.40 | STOP_HIT | 1.00 | 1.43% |
| BUY | retest2 | 2025-12-24 12:15:00 | 1261.50 | 2026-01-02 15:15:00 | 1280.40 | STOP_HIT | 1.00 | 1.50% |
| BUY | retest2 | 2025-12-26 10:15:00 | 1261.50 | 2026-01-02 15:15:00 | 1280.40 | STOP_HIT | 1.00 | 1.50% |
| BUY | retest2 | 2026-01-16 09:15:00 | 1316.90 | 2026-01-20 12:15:00 | 1304.10 | STOP_HIT | 1.00 | -0.97% |
| BUY | retest2 | 2026-01-16 09:45:00 | 1313.00 | 2026-01-20 12:15:00 | 1304.10 | STOP_HIT | 1.00 | -0.68% |
| BUY | retest2 | 2026-01-19 13:00:00 | 1311.30 | 2026-01-20 12:15:00 | 1304.10 | STOP_HIT | 1.00 | -0.55% |
| BUY | retest2 | 2026-01-20 10:00:00 | 1311.70 | 2026-01-20 12:15:00 | 1304.10 | STOP_HIT | 1.00 | -0.58% |
| SELL | retest2 | 2026-01-29 09:15:00 | 1283.50 | 2026-01-30 09:15:00 | 1302.60 | STOP_HIT | 1.00 | -1.49% |
| SELL | retest2 | 2026-01-29 14:45:00 | 1288.50 | 2026-01-30 09:15:00 | 1302.60 | STOP_HIT | 1.00 | -1.09% |
| BUY | retest2 | 2026-02-05 11:00:00 | 1310.40 | 2026-02-05 13:15:00 | 1299.70 | STOP_HIT | 1.00 | -0.82% |
| SELL | retest2 | 2026-02-09 14:30:00 | 1292.60 | 2026-02-10 09:15:00 | 1304.50 | STOP_HIT | 1.00 | -0.92% |
| BUY | retest2 | 2026-02-11 14:15:00 | 1302.80 | 2026-02-12 10:15:00 | 1296.10 | STOP_HIT | 1.00 | -0.51% |
| BUY | retest2 | 2026-02-12 09:30:00 | 1305.20 | 2026-02-12 10:15:00 | 1296.10 | STOP_HIT | 1.00 | -0.70% |
| BUY | retest2 | 2026-02-12 11:45:00 | 1303.40 | 2026-02-13 10:15:00 | 1299.30 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest2 | 2026-02-12 14:45:00 | 1301.60 | 2026-02-13 10:15:00 | 1299.30 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest2 | 2026-02-16 12:30:00 | 1290.20 | 2026-02-18 14:15:00 | 1301.00 | STOP_HIT | 1.00 | -0.84% |
| SELL | retest2 | 2026-02-16 13:45:00 | 1287.60 | 2026-02-18 14:15:00 | 1301.00 | STOP_HIT | 1.00 | -1.04% |
| SELL | retest2 | 2026-02-17 09:15:00 | 1284.60 | 2026-02-18 14:15:00 | 1301.00 | STOP_HIT | 1.00 | -1.28% |
| SELL | retest2 | 2026-03-09 09:15:00 | 1232.20 | 2026-03-23 10:15:00 | 1170.59 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-09 09:15:00 | 1232.20 | 2026-03-24 11:15:00 | 1182.20 | STOP_HIT | 0.50 | 4.06% |
| BUY | retest2 | 2026-04-09 15:00:00 | 1229.50 | 2026-04-21 14:15:00 | 1352.45 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-13 10:15:00 | 1226.40 | 2026-04-21 14:15:00 | 1349.04 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-13 12:30:00 | 1228.20 | 2026-04-21 14:15:00 | 1351.02 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-13 13:30:00 | 1225.70 | 2026-04-21 14:15:00 | 1348.27 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-15 09:15:00 | 1235.40 | 2026-04-21 14:15:00 | 1358.94 | TARGET_HIT | 1.00 | 10.00% |
