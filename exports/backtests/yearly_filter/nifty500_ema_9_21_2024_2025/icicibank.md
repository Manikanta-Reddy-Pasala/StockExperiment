# ICICI Bank Ltd. (ICICIBANK)

## Backtest Summary

- **Window:** 2024-03-13 09:15:00 → 2026-05-08 15:15:00 (3710 bars)
- **Last close:** 1267.80
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 166 |
| ALERT1 | 105 |
| ALERT2 | 104 |
| ALERT2_SKIP | 59 |
| ALERT3 | 262 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 2 |
| ENTRY2 | 140 |
| PARTIAL | 2 |
| TARGET_HIT | 2 |
| STOP_HIT | 141 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 144 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 30 / 114
- **Target hits / Stop hits / Partials:** 2 / 140 / 2
- **Avg / median % per leg:** -0.30% / -0.57%
- **Sum % (uncompounded):** -43.36%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 49 | 11 | 22.4% | 2 | 47 | 0 | 0.04% | 1.8% |
| BUY @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -1.10% | -1.1% |
| BUY @ 3rd Alert (retest2) | 48 | 11 | 22.9% | 2 | 46 | 0 | 0.06% | 2.9% |
| SELL (all) | 95 | 19 | 20.0% | 0 | 93 | 2 | -0.47% | -45.1% |
| SELL @ 2nd Alert (retest1) | 1 | 1 | 100.0% | 0 | 1 | 0 | 0.21% | 0.2% |
| SELL @ 3rd Alert (retest2) | 94 | 18 | 19.1% | 0 | 92 | 2 | -0.48% | -45.3% |
| retest1 (combined) | 2 | 1 | 50.0% | 0 | 2 | 0 | -0.44% | -0.9% |
| retest2 (combined) | 142 | 29 | 20.4% | 2 | 138 | 2 | -0.30% | -42.5% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-05-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-13 15:15:00 | 1126.65 | 1121.77 | 1121.16 | EMA200 above EMA400 |

### Cycle 2 — SELL (started 2024-05-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-15 09:15:00 | 1118.25 | 1120.60 | 1120.92 | EMA200 below EMA400 |

### Cycle 3 — BUY (started 2024-05-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-15 14:15:00 | 1124.75 | 1120.73 | 1120.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-16 09:15:00 | 1129.30 | 1123.14 | 1121.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-16 10:15:00 | 1117.90 | 1122.09 | 1121.47 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-16 10:15:00 | 1117.90 | 1122.09 | 1121.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-16 10:15:00 | 1117.90 | 1122.09 | 1121.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-16 11:00:00 | 1117.90 | 1122.09 | 1121.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-16 11:15:00 | 1127.60 | 1123.20 | 1122.03 | EMA400 retest candle locked (from upside) |

### Cycle 4 — SELL (started 2024-05-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-16 13:15:00 | 1114.65 | 1120.19 | 1120.79 | EMA200 below EMA400 |

### Cycle 5 — BUY (started 2024-05-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-16 14:15:00 | 1131.10 | 1122.37 | 1121.72 | EMA200 above EMA400 |

### Cycle 6 — SELL (started 2024-05-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-21 14:15:00 | 1120.65 | 1125.18 | 1125.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-21 15:15:00 | 1119.85 | 1124.11 | 1125.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-23 09:15:00 | 1125.00 | 1116.42 | 1118.97 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-23 09:15:00 | 1125.00 | 1116.42 | 1118.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-23 09:15:00 | 1125.00 | 1116.42 | 1118.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-23 10:00:00 | 1125.00 | 1116.42 | 1118.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-23 10:15:00 | 1124.65 | 1118.07 | 1119.49 | EMA400 retest candle locked (from downside) |

### Cycle 7 — BUY (started 2024-05-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-23 12:15:00 | 1130.25 | 1121.58 | 1120.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-23 14:15:00 | 1135.95 | 1126.27 | 1123.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-24 09:15:00 | 1125.25 | 1127.71 | 1124.54 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-24 09:15:00 | 1125.25 | 1127.71 | 1124.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-24 09:15:00 | 1125.25 | 1127.71 | 1124.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-24 10:00:00 | 1125.25 | 1127.71 | 1124.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-24 10:15:00 | 1126.00 | 1127.36 | 1124.68 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-24 14:30:00 | 1131.20 | 1128.30 | 1125.97 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-28 09:15:00 | 1132.65 | 1134.35 | 1131.41 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-28 12:45:00 | 1130.70 | 1130.97 | 1130.49 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-28 14:15:00 | 1126.40 | 1130.20 | 1130.23 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 8 — SELL (started 2024-05-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-28 14:15:00 | 1126.40 | 1130.20 | 1130.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-29 09:15:00 | 1112.40 | 1125.97 | 1128.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-30 10:15:00 | 1109.20 | 1108.84 | 1115.85 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-30 10:30:00 | 1109.00 | 1108.84 | 1115.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-30 12:15:00 | 1116.75 | 1110.63 | 1115.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-30 12:45:00 | 1117.40 | 1110.63 | 1115.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-30 13:15:00 | 1117.00 | 1111.91 | 1115.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-30 13:45:00 | 1117.95 | 1111.91 | 1115.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-30 14:15:00 | 1111.90 | 1111.91 | 1115.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-30 15:15:00 | 1117.95 | 1111.91 | 1115.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-30 15:15:00 | 1117.95 | 1113.11 | 1115.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-31 09:15:00 | 1121.30 | 1113.11 | 1115.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 09:15:00 | 1116.10 | 1113.71 | 1115.56 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-31 11:15:00 | 1114.30 | 1113.97 | 1115.51 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-31 12:15:00 | 1122.90 | 1116.84 | 1116.62 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 9 — BUY (started 2024-05-31 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-31 12:15:00 | 1122.90 | 1116.84 | 1116.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-31 13:15:00 | 1124.50 | 1118.37 | 1117.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-04 09:15:00 | 1129.25 | 1144.70 | 1135.39 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-04 09:15:00 | 1129.25 | 1144.70 | 1135.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 09:15:00 | 1129.25 | 1144.70 | 1135.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-04 09:30:00 | 1135.00 | 1144.70 | 1135.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 10:15:00 | 1096.10 | 1134.98 | 1131.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-04 11:00:00 | 1096.10 | 1134.98 | 1131.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 10 — SELL (started 2024-06-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-04 11:15:00 | 1071.50 | 1122.28 | 1126.34 | EMA200 below EMA400 |

### Cycle 11 — BUY (started 2024-06-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-06 15:15:00 | 1111.75 | 1106.08 | 1105.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-07 09:15:00 | 1115.35 | 1107.93 | 1106.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-10 14:15:00 | 1124.00 | 1124.03 | 1118.84 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-10 14:30:00 | 1127.65 | 1124.03 | 1118.84 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-10 15:15:00 | 1119.80 | 1123.18 | 1118.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-11 09:15:00 | 1118.00 | 1123.18 | 1118.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-11 09:15:00 | 1126.60 | 1123.86 | 1119.62 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-12 09:45:00 | 1129.65 | 1123.24 | 1121.21 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-13 09:15:00 | 1118.35 | 1120.86 | 1121.05 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 12 — SELL (started 2024-06-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-13 09:15:00 | 1118.35 | 1120.86 | 1121.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-13 13:15:00 | 1111.75 | 1117.11 | 1119.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-18 09:15:00 | 1108.70 | 1108.13 | 1111.82 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-18 09:15:00 | 1108.70 | 1108.13 | 1111.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-18 09:15:00 | 1108.70 | 1108.13 | 1111.82 | EMA400 retest candle locked (from downside) |

### Cycle 13 — BUY (started 2024-06-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-18 14:15:00 | 1121.05 | 1114.05 | 1113.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-18 15:15:00 | 1125.75 | 1116.39 | 1114.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-20 09:15:00 | 1136.75 | 1138.85 | 1130.27 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-20 10:00:00 | 1136.75 | 1138.85 | 1130.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-28 11:15:00 | 1206.55 | 1215.64 | 1208.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-28 11:30:00 | 1211.00 | 1215.64 | 1208.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-28 12:15:00 | 1200.70 | 1212.65 | 1207.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-28 13:00:00 | 1200.70 | 1212.65 | 1207.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 14 — SELL (started 2024-07-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-01 09:15:00 | 1201.00 | 1205.41 | 1205.52 | EMA200 below EMA400 |

### Cycle 15 — BUY (started 2024-07-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-01 12:15:00 | 1210.45 | 1206.42 | 1205.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-01 14:15:00 | 1212.65 | 1208.42 | 1206.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-01 15:15:00 | 1208.05 | 1208.35 | 1207.05 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-01 15:15:00 | 1208.05 | 1208.35 | 1207.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-01 15:15:00 | 1208.05 | 1208.35 | 1207.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-02 09:15:00 | 1202.30 | 1208.35 | 1207.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 16 — SELL (started 2024-07-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-02 09:15:00 | 1194.25 | 1205.53 | 1205.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-02 11:15:00 | 1190.00 | 1200.81 | 1203.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-03 09:15:00 | 1205.15 | 1197.70 | 1200.56 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-03 09:15:00 | 1205.15 | 1197.70 | 1200.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-03 09:15:00 | 1205.15 | 1197.70 | 1200.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-03 09:45:00 | 1209.95 | 1197.70 | 1200.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-03 10:15:00 | 1204.85 | 1199.13 | 1200.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-03 10:30:00 | 1205.10 | 1199.13 | 1200.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-03 12:15:00 | 1203.15 | 1200.31 | 1201.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-03 12:30:00 | 1204.90 | 1200.31 | 1201.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-03 14:15:00 | 1202.40 | 1200.96 | 1201.35 | EMA400 retest candle locked (from downside) |

### Cycle 17 — BUY (started 2024-07-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-04 09:15:00 | 1225.95 | 1205.81 | 1203.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-04 14:15:00 | 1233.10 | 1220.65 | 1212.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-08 10:15:00 | 1229.55 | 1230.02 | 1223.68 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-08 11:00:00 | 1229.55 | 1230.02 | 1223.68 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-11 10:15:00 | 1227.55 | 1238.93 | 1237.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-11 11:00:00 | 1227.55 | 1238.93 | 1237.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-11 11:15:00 | 1229.30 | 1237.01 | 1236.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-11 12:00:00 | 1229.30 | 1237.01 | 1236.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 18 — SELL (started 2024-07-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-11 12:15:00 | 1229.00 | 1235.41 | 1236.04 | EMA200 below EMA400 |

### Cycle 19 — BUY (started 2024-07-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-11 15:15:00 | 1240.00 | 1236.92 | 1236.60 | EMA200 above EMA400 |

### Cycle 20 — SELL (started 2024-07-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-12 09:15:00 | 1233.65 | 1236.26 | 1236.33 | EMA200 below EMA400 |

### Cycle 21 — BUY (started 2024-07-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-12 10:15:00 | 1248.60 | 1238.73 | 1237.45 | EMA200 above EMA400 |

### Cycle 22 — SELL (started 2024-07-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-12 15:15:00 | 1234.50 | 1237.22 | 1237.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-15 11:15:00 | 1229.75 | 1234.72 | 1236.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-16 09:15:00 | 1236.40 | 1233.26 | 1234.62 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-16 09:15:00 | 1236.40 | 1233.26 | 1234.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-16 09:15:00 | 1236.40 | 1233.26 | 1234.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-16 09:30:00 | 1242.25 | 1233.26 | 1234.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-16 10:15:00 | 1241.60 | 1234.93 | 1235.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-16 10:45:00 | 1242.40 | 1234.93 | 1235.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 23 — BUY (started 2024-07-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-16 11:15:00 | 1239.40 | 1235.82 | 1235.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-18 09:15:00 | 1247.25 | 1239.96 | 1237.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-18 10:15:00 | 1239.25 | 1239.82 | 1238.03 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-18 11:00:00 | 1239.25 | 1239.82 | 1238.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-18 11:15:00 | 1242.70 | 1240.39 | 1238.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-18 11:45:00 | 1239.10 | 1240.39 | 1238.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-19 09:15:00 | 1246.00 | 1246.12 | 1242.51 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-19 15:00:00 | 1249.20 | 1245.96 | 1243.74 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-22 09:15:00 | 1235.65 | 1244.46 | 1243.47 | SL hit (close<static) qty=1.00 sl=1242.15 alert=retest2 |

### Cycle 24 — SELL (started 2024-07-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-22 11:15:00 | 1240.55 | 1242.78 | 1242.82 | EMA200 below EMA400 |

### Cycle 25 — BUY (started 2024-07-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-23 10:15:00 | 1244.85 | 1243.07 | 1242.89 | EMA200 above EMA400 |

### Cycle 26 — SELL (started 2024-07-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-23 11:15:00 | 1237.40 | 1241.93 | 1242.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-23 12:15:00 | 1213.95 | 1236.34 | 1239.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-24 14:15:00 | 1224.10 | 1220.44 | 1227.71 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-07-24 15:00:00 | 1224.10 | 1220.44 | 1227.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-24 15:15:00 | 1226.60 | 1221.68 | 1227.61 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-25 09:15:00 | 1203.70 | 1221.68 | 1227.61 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-29 09:15:00 | 1228.65 | 1210.50 | 1211.08 | SL hit (close>static) qty=1.00 sl=1228.45 alert=retest2 |

### Cycle 27 — BUY (started 2024-07-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-29 10:15:00 | 1236.35 | 1215.67 | 1213.37 | EMA200 above EMA400 |

### Cycle 28 — SELL (started 2024-07-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-30 15:15:00 | 1206.75 | 1214.82 | 1214.94 | EMA200 below EMA400 |

### Cycle 29 — BUY (started 2024-07-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-31 09:15:00 | 1221.10 | 1216.07 | 1215.50 | EMA200 above EMA400 |

### Cycle 30 — SELL (started 2024-08-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-01 09:15:00 | 1213.15 | 1215.32 | 1215.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-01 13:15:00 | 1210.15 | 1213.96 | 1214.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-07 11:15:00 | 1168.35 | 1168.25 | 1175.88 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-07 12:00:00 | 1168.35 | 1168.25 | 1175.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 14:15:00 | 1172.30 | 1169.20 | 1174.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-07 14:30:00 | 1171.50 | 1169.20 | 1174.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 15:15:00 | 1173.50 | 1170.06 | 1174.33 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-08 09:15:00 | 1170.25 | 1170.06 | 1174.33 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-08 12:30:00 | 1169.30 | 1168.14 | 1171.91 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-09 09:15:00 | 1176.90 | 1168.81 | 1170.91 | SL hit (close>static) qty=1.00 sl=1175.10 alert=retest2 |

### Cycle 31 — BUY (started 2024-08-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-12 14:15:00 | 1173.45 | 1171.25 | 1171.05 | EMA200 above EMA400 |

### Cycle 32 — SELL (started 2024-08-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-13 13:15:00 | 1166.60 | 1170.78 | 1171.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-14 09:15:00 | 1156.85 | 1167.50 | 1169.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-16 09:15:00 | 1169.90 | 1163.67 | 1165.75 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-16 09:15:00 | 1169.90 | 1163.67 | 1165.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-16 09:15:00 | 1169.90 | 1163.67 | 1165.75 | EMA400 retest candle locked (from downside) |

### Cycle 33 — BUY (started 2024-08-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-16 11:15:00 | 1184.70 | 1169.23 | 1168.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-16 13:15:00 | 1191.95 | 1176.35 | 1171.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-19 09:15:00 | 1180.50 | 1180.56 | 1175.01 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-19 10:00:00 | 1180.50 | 1180.56 | 1175.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-19 13:15:00 | 1175.60 | 1179.11 | 1176.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-19 13:45:00 | 1175.65 | 1179.11 | 1176.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-19 14:15:00 | 1175.15 | 1178.32 | 1175.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-19 15:00:00 | 1175.15 | 1178.32 | 1175.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-19 15:15:00 | 1177.70 | 1178.20 | 1176.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-20 09:15:00 | 1172.30 | 1178.20 | 1176.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-20 09:15:00 | 1183.75 | 1179.31 | 1176.83 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-20 11:15:00 | 1184.55 | 1180.18 | 1177.45 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-21 09:15:00 | 1169.55 | 1179.07 | 1178.59 | SL hit (close<static) qty=1.00 sl=1170.00 alert=retest2 |

### Cycle 34 — SELL (started 2024-08-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-21 10:15:00 | 1166.85 | 1176.63 | 1177.52 | EMA200 below EMA400 |

### Cycle 35 — BUY (started 2024-08-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-22 10:15:00 | 1183.30 | 1177.18 | 1176.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-22 11:15:00 | 1185.85 | 1178.92 | 1177.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-28 09:15:00 | 1218.70 | 1220.81 | 1213.02 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-28 10:00:00 | 1218.70 | 1220.81 | 1213.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-29 09:15:00 | 1230.55 | 1224.19 | 1218.72 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-29 10:15:00 | 1232.10 | 1224.19 | 1218.72 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-29 12:15:00 | 1232.30 | 1226.48 | 1220.78 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-30 15:15:00 | 1232.00 | 1226.16 | 1223.64 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-02 12:30:00 | 1231.90 | 1227.52 | 1225.37 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-03 09:15:00 | 1227.30 | 1228.73 | 1226.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-03 09:45:00 | 1225.85 | 1228.73 | 1226.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-03 10:15:00 | 1223.10 | 1227.60 | 1226.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-03 11:00:00 | 1223.10 | 1227.60 | 1226.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-03 11:15:00 | 1226.25 | 1227.33 | 1226.43 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-03 14:00:00 | 1229.75 | 1227.20 | 1226.48 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-06 10:15:00 | 1221.70 | 1231.33 | 1232.62 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 36 — SELL (started 2024-09-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-06 10:15:00 | 1221.70 | 1231.33 | 1232.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-06 11:15:00 | 1217.70 | 1228.60 | 1231.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-09 11:15:00 | 1219.40 | 1218.30 | 1223.38 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-09 11:15:00 | 1219.40 | 1218.30 | 1223.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-09 11:15:00 | 1219.40 | 1218.30 | 1223.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-09 11:45:00 | 1223.10 | 1218.30 | 1223.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-09 12:15:00 | 1224.25 | 1219.49 | 1223.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-09 13:00:00 | 1224.25 | 1219.49 | 1223.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-09 13:15:00 | 1232.15 | 1222.02 | 1224.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-09 14:00:00 | 1232.15 | 1222.02 | 1224.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-09 14:15:00 | 1235.00 | 1224.62 | 1225.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-09 14:45:00 | 1236.50 | 1224.62 | 1225.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 37 — BUY (started 2024-09-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-09 15:15:00 | 1236.95 | 1227.09 | 1226.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-10 13:15:00 | 1239.75 | 1232.32 | 1229.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-11 09:15:00 | 1234.05 | 1234.11 | 1231.01 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-11 09:15:00 | 1234.05 | 1234.11 | 1231.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-11 09:15:00 | 1234.05 | 1234.11 | 1231.01 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-11 10:30:00 | 1238.75 | 1234.39 | 1231.42 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-11 14:15:00 | 1236.60 | 1233.40 | 1231.63 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-11 15:15:00 | 1238.00 | 1233.77 | 1231.96 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-12 14:00:00 | 1243.65 | 1235.03 | 1233.29 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Target hit | 2024-09-20 15:15:00 | 1360.26 | 1320.09 | 1302.91 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 38 — SELL (started 2024-09-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-27 13:15:00 | 1315.80 | 1321.46 | 1321.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-27 14:15:00 | 1307.20 | 1318.61 | 1320.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-01 15:15:00 | 1278.20 | 1277.12 | 1286.92 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-03 09:15:00 | 1260.25 | 1277.12 | 1286.92 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-07 10:15:00 | 1249.95 | 1249.97 | 1258.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-07 10:30:00 | 1256.85 | 1249.97 | 1258.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-07 11:15:00 | 1245.40 | 1249.06 | 1256.94 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-07 13:00:00 | 1240.35 | 1247.32 | 1255.43 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-07 14:45:00 | 1239.90 | 1242.15 | 1251.56 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-09 09:15:00 | 1257.60 | 1242.41 | 1245.48 | SL hit (close>ema400) qty=1.00 sl=1245.48 alert=retest1 |

### Cycle 39 — BUY (started 2024-10-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-15 10:15:00 | 1244.75 | 1236.23 | 1236.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-15 11:15:00 | 1252.65 | 1239.51 | 1237.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-16 10:15:00 | 1248.30 | 1249.28 | 1244.44 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-16 11:00:00 | 1248.30 | 1249.28 | 1244.44 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-16 11:15:00 | 1248.60 | 1249.14 | 1244.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-16 11:30:00 | 1245.45 | 1249.14 | 1244.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-16 14:15:00 | 1243.00 | 1248.63 | 1245.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-16 15:00:00 | 1243.00 | 1248.63 | 1245.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-16 15:15:00 | 1245.00 | 1247.90 | 1245.67 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-17 09:15:00 | 1247.00 | 1247.90 | 1245.67 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-17 09:15:00 | 1234.85 | 1245.29 | 1244.68 | SL hit (close<static) qty=1.00 sl=1241.60 alert=retest2 |

### Cycle 40 — SELL (started 2024-10-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-17 10:15:00 | 1229.40 | 1242.11 | 1243.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-17 11:15:00 | 1228.20 | 1239.33 | 1241.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-18 09:15:00 | 1241.30 | 1235.86 | 1238.72 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-18 09:15:00 | 1241.30 | 1235.86 | 1238.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 09:15:00 | 1241.30 | 1235.86 | 1238.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-18 09:45:00 | 1238.30 | 1235.86 | 1238.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 10:15:00 | 1246.45 | 1237.98 | 1239.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-18 11:00:00 | 1246.45 | 1237.98 | 1239.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 41 — BUY (started 2024-10-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-18 11:15:00 | 1253.75 | 1241.13 | 1240.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-18 12:15:00 | 1257.75 | 1244.46 | 1242.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-21 11:15:00 | 1252.40 | 1254.74 | 1249.68 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-21 12:00:00 | 1252.40 | 1254.74 | 1249.68 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-23 09:15:00 | 1255.90 | 1267.09 | 1262.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-23 09:30:00 | 1257.30 | 1267.09 | 1262.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-23 10:15:00 | 1246.60 | 1262.99 | 1261.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-23 11:00:00 | 1246.60 | 1262.99 | 1261.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 42 — SELL (started 2024-10-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-23 11:15:00 | 1246.70 | 1259.73 | 1259.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-25 10:15:00 | 1236.20 | 1249.69 | 1253.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-25 14:15:00 | 1256.30 | 1249.48 | 1251.82 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-25 14:15:00 | 1256.30 | 1249.48 | 1251.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-25 14:15:00 | 1256.30 | 1249.48 | 1251.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-25 15:00:00 | 1256.30 | 1249.48 | 1251.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-25 15:15:00 | 1259.60 | 1251.50 | 1252.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-28 09:15:00 | 1289.85 | 1251.50 | 1252.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 43 — BUY (started 2024-10-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-28 09:15:00 | 1298.50 | 1260.90 | 1256.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-28 11:15:00 | 1304.85 | 1275.66 | 1264.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-30 09:15:00 | 1310.30 | 1316.61 | 1299.95 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-30 10:00:00 | 1310.30 | 1316.61 | 1299.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-31 10:15:00 | 1305.65 | 1311.00 | 1305.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-31 11:00:00 | 1305.65 | 1311.00 | 1305.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-31 11:15:00 | 1302.50 | 1309.30 | 1305.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-31 11:45:00 | 1301.60 | 1309.30 | 1305.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-31 12:15:00 | 1301.45 | 1307.73 | 1304.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-31 12:45:00 | 1299.75 | 1307.73 | 1304.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 44 — SELL (started 2024-10-31 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-31 14:15:00 | 1291.50 | 1302.62 | 1302.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-04 09:15:00 | 1275.00 | 1293.44 | 1298.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-05 13:15:00 | 1293.25 | 1277.76 | 1283.29 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-05 13:15:00 | 1293.25 | 1277.76 | 1283.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-05 13:15:00 | 1293.25 | 1277.76 | 1283.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-05 14:00:00 | 1293.25 | 1277.76 | 1283.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-05 14:15:00 | 1296.00 | 1281.41 | 1284.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-05 15:00:00 | 1296.00 | 1281.41 | 1284.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 45 — BUY (started 2024-11-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-06 09:15:00 | 1312.55 | 1289.27 | 1287.59 | EMA200 above EMA400 |

### Cycle 46 — SELL (started 2024-11-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-07 11:15:00 | 1279.00 | 1291.74 | 1292.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-08 09:15:00 | 1266.00 | 1280.79 | 1286.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-11 10:15:00 | 1272.50 | 1265.14 | 1272.75 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-11 11:00:00 | 1272.50 | 1265.14 | 1272.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-11 11:15:00 | 1273.05 | 1266.73 | 1272.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-11 11:30:00 | 1275.80 | 1266.73 | 1272.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-11 12:15:00 | 1271.00 | 1267.58 | 1272.62 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-11 13:15:00 | 1268.15 | 1267.58 | 1272.62 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-11 15:15:00 | 1268.65 | 1267.95 | 1271.89 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-12 09:15:00 | 1290.10 | 1272.49 | 1273.27 | SL hit (close>static) qty=1.00 sl=1273.60 alert=retest2 |

### Cycle 47 — BUY (started 2024-11-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-12 10:15:00 | 1287.75 | 1275.54 | 1274.59 | EMA200 above EMA400 |

### Cycle 48 — SELL (started 2024-11-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-12 14:15:00 | 1269.65 | 1274.57 | 1274.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-13 10:15:00 | 1264.15 | 1270.59 | 1272.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-14 15:15:00 | 1258.95 | 1257.30 | 1261.86 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-18 09:15:00 | 1255.90 | 1257.30 | 1261.86 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-18 09:15:00 | 1256.55 | 1257.15 | 1261.37 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-18 13:30:00 | 1251.00 | 1254.79 | 1258.83 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-19 09:15:00 | 1252.00 | 1254.17 | 1257.84 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-19 10:00:00 | 1251.15 | 1253.56 | 1257.23 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-19 10:30:00 | 1250.50 | 1252.75 | 1256.53 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 14:15:00 | 1248.15 | 1251.64 | 1254.81 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-19 15:15:00 | 1244.00 | 1251.64 | 1254.81 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-21 12:45:00 | 1243.00 | 1247.78 | 1251.24 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-21 15:15:00 | 1255.85 | 1250.32 | 1251.63 | SL hit (close>static) qty=1.00 sl=1255.25 alert=retest2 |

### Cycle 49 — BUY (started 2024-11-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-22 09:15:00 | 1277.85 | 1255.82 | 1254.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-25 09:15:00 | 1304.65 | 1277.88 | 1267.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-27 09:15:00 | 1302.40 | 1303.12 | 1294.70 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-27 12:30:00 | 1308.25 | 1304.35 | 1297.35 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 09:15:00 | 1308.30 | 1304.34 | 1299.49 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-11-28 10:15:00 | 1293.90 | 1302.25 | 1298.98 | SL hit (close<ema400) qty=1.00 sl=1298.98 alert=retest1 |

### Cycle 50 — SELL (started 2024-11-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-28 13:15:00 | 1289.95 | 1295.92 | 1296.61 | EMA200 below EMA400 |

### Cycle 51 — BUY (started 2024-11-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-29 14:15:00 | 1299.35 | 1295.85 | 1295.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-02 10:15:00 | 1304.20 | 1298.75 | 1297.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-03 09:15:00 | 1299.95 | 1301.82 | 1299.67 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-03 09:15:00 | 1299.95 | 1301.82 | 1299.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-03 09:15:00 | 1299.95 | 1301.82 | 1299.67 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-03 10:45:00 | 1304.30 | 1302.33 | 1300.09 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-13 09:15:00 | 1320.70 | 1327.73 | 1327.75 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 52 — SELL (started 2024-12-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-13 09:15:00 | 1320.70 | 1327.73 | 1327.75 | EMA200 below EMA400 |

### Cycle 53 — BUY (started 2024-12-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-13 11:15:00 | 1336.65 | 1328.15 | 1327.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-13 12:15:00 | 1340.50 | 1330.62 | 1329.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-17 09:15:00 | 1336.70 | 1342.07 | 1338.51 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-17 09:15:00 | 1336.70 | 1342.07 | 1338.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-17 09:15:00 | 1336.70 | 1342.07 | 1338.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-17 10:00:00 | 1336.70 | 1342.07 | 1338.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-17 10:15:00 | 1334.85 | 1340.63 | 1338.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-17 10:30:00 | 1332.25 | 1340.63 | 1338.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-17 11:15:00 | 1329.35 | 1338.37 | 1337.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-17 12:00:00 | 1329.35 | 1338.37 | 1337.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 54 — SELL (started 2024-12-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-17 13:15:00 | 1328.00 | 1334.99 | 1335.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-18 09:15:00 | 1320.00 | 1331.56 | 1334.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-20 15:15:00 | 1292.00 | 1289.31 | 1298.00 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-23 09:15:00 | 1300.90 | 1289.31 | 1298.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-23 09:15:00 | 1304.90 | 1292.42 | 1298.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-23 10:00:00 | 1304.90 | 1292.42 | 1298.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-23 10:15:00 | 1296.40 | 1293.22 | 1298.43 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-23 11:15:00 | 1293.85 | 1293.22 | 1298.43 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-24 09:15:00 | 1294.25 | 1293.83 | 1296.59 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-24 14:00:00 | 1295.10 | 1294.94 | 1296.09 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-26 10:30:00 | 1293.30 | 1295.36 | 1296.04 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-26 11:15:00 | 1292.60 | 1294.81 | 1295.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-26 11:30:00 | 1292.70 | 1294.81 | 1295.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-26 13:15:00 | 1296.25 | 1295.02 | 1295.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-26 13:30:00 | 1296.55 | 1295.02 | 1295.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-26 14:15:00 | 1298.00 | 1295.61 | 1295.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-26 15:00:00 | 1298.00 | 1295.61 | 1295.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-26 15:15:00 | 1297.00 | 1295.89 | 1295.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-27 09:15:00 | 1309.00 | 1295.89 | 1295.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-12-27 09:15:00 | 1313.95 | 1299.50 | 1297.61 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 55 — BUY (started 2024-12-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-27 09:15:00 | 1313.95 | 1299.50 | 1297.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-30 11:15:00 | 1324.15 | 1311.16 | 1305.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-30 12:15:00 | 1306.75 | 1310.27 | 1305.52 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-30 12:15:00 | 1306.75 | 1310.27 | 1305.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-30 12:15:00 | 1306.75 | 1310.27 | 1305.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-30 13:00:00 | 1306.75 | 1310.27 | 1305.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-30 13:15:00 | 1294.95 | 1307.21 | 1304.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-30 14:00:00 | 1294.95 | 1307.21 | 1304.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-30 14:15:00 | 1292.25 | 1304.22 | 1303.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-30 14:45:00 | 1296.90 | 1304.22 | 1303.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 56 — SELL (started 2024-12-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-30 15:15:00 | 1293.45 | 1302.06 | 1302.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-31 09:15:00 | 1283.95 | 1298.44 | 1300.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-01 10:15:00 | 1289.00 | 1286.11 | 1291.58 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-01 10:45:00 | 1289.75 | 1286.11 | 1291.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-01 12:15:00 | 1286.05 | 1286.53 | 1290.85 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-01 14:30:00 | 1285.25 | 1286.33 | 1290.04 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-02 09:45:00 | 1282.75 | 1284.62 | 1288.58 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-02 12:00:00 | 1283.70 | 1285.08 | 1288.15 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-03 09:15:00 | 1281.30 | 1287.97 | 1288.66 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 10:15:00 | 1267.05 | 1271.34 | 1277.28 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-06 11:45:00 | 1263.00 | 1270.53 | 1276.38 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-06 13:15:00 | 1263.05 | 1269.87 | 1275.55 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-07 09:15:00 | 1279.50 | 1269.11 | 1273.00 | SL hit (close>static) qty=1.00 sl=1277.75 alert=retest2 |

### Cycle 57 — BUY (started 2025-01-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-07 12:15:00 | 1285.80 | 1275.96 | 1275.45 | EMA200 above EMA400 |

### Cycle 58 — SELL (started 2025-01-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-08 10:15:00 | 1262.05 | 1274.66 | 1275.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-08 11:15:00 | 1258.75 | 1271.48 | 1273.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-09 13:15:00 | 1261.40 | 1258.97 | 1264.41 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-09 14:00:00 | 1261.40 | 1258.97 | 1264.41 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-09 15:15:00 | 1263.50 | 1260.27 | 1264.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-10 09:15:00 | 1258.00 | 1260.27 | 1264.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-10 09:15:00 | 1251.60 | 1258.53 | 1262.94 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-10 10:15:00 | 1249.10 | 1258.53 | 1262.94 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-10 12:45:00 | 1248.20 | 1255.22 | 1260.26 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-13 09:15:00 | 1242.00 | 1251.51 | 1257.06 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-16 09:30:00 | 1247.25 | 1239.30 | 1240.10 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-16 11:15:00 | 1245.30 | 1240.83 | 1240.68 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 59 — BUY (started 2025-01-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-16 11:15:00 | 1245.30 | 1240.83 | 1240.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-16 12:15:00 | 1253.10 | 1243.28 | 1241.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-17 09:15:00 | 1226.10 | 1242.98 | 1242.51 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-17 09:15:00 | 1226.10 | 1242.98 | 1242.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-17 09:15:00 | 1226.10 | 1242.98 | 1242.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-17 10:00:00 | 1226.10 | 1242.98 | 1242.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 60 — SELL (started 2025-01-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-17 10:15:00 | 1224.10 | 1239.21 | 1240.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-21 10:15:00 | 1214.65 | 1226.70 | 1230.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-23 10:15:00 | 1203.90 | 1202.38 | 1209.00 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-24 09:15:00 | 1206.25 | 1203.70 | 1206.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-24 09:15:00 | 1206.25 | 1203.70 | 1206.87 | EMA400 retest candle locked (from downside) |

### Cycle 61 — BUY (started 2025-01-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-24 12:15:00 | 1212.55 | 1209.05 | 1208.86 | EMA200 above EMA400 |

### Cycle 62 — SELL (started 2025-01-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-24 13:15:00 | 1206.45 | 1208.53 | 1208.64 | EMA200 below EMA400 |

### Cycle 63 — BUY (started 2025-01-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-24 15:15:00 | 1214.20 | 1209.60 | 1209.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-27 09:15:00 | 1228.90 | 1213.46 | 1210.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-29 12:15:00 | 1245.05 | 1247.15 | 1239.19 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-29 13:00:00 | 1245.05 | 1247.15 | 1239.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-30 09:15:00 | 1249.65 | 1249.26 | 1242.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-30 09:30:00 | 1244.35 | 1249.26 | 1242.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-30 12:15:00 | 1251.55 | 1249.29 | 1244.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-30 12:45:00 | 1248.85 | 1249.29 | 1244.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-31 09:15:00 | 1243.20 | 1250.97 | 1247.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-31 09:30:00 | 1244.80 | 1250.97 | 1247.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-31 10:15:00 | 1241.50 | 1249.08 | 1246.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-31 11:00:00 | 1241.50 | 1249.08 | 1246.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-31 11:15:00 | 1241.55 | 1247.57 | 1246.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-31 11:30:00 | 1239.85 | 1247.57 | 1246.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-31 13:15:00 | 1243.50 | 1247.08 | 1246.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-31 14:00:00 | 1243.50 | 1247.08 | 1246.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-31 14:15:00 | 1253.90 | 1248.45 | 1246.86 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-01 10:15:00 | 1255.25 | 1249.97 | 1247.87 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-01 12:15:00 | 1236.40 | 1247.10 | 1247.14 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 64 — SELL (started 2025-02-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-01 12:15:00 | 1236.40 | 1247.10 | 1247.14 | EMA200 below EMA400 |

### Cycle 65 — BUY (started 2025-02-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-01 13:15:00 | 1250.45 | 1247.77 | 1247.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-01 14:15:00 | 1255.90 | 1249.40 | 1248.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-03 09:15:00 | 1239.60 | 1248.25 | 1247.95 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-03 09:15:00 | 1239.60 | 1248.25 | 1247.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-03 09:15:00 | 1239.60 | 1248.25 | 1247.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-03 10:00:00 | 1239.60 | 1248.25 | 1247.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-03 10:15:00 | 1247.20 | 1248.04 | 1247.88 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-03 12:00:00 | 1250.00 | 1248.43 | 1248.07 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-07 11:15:00 | 1259.75 | 1266.22 | 1266.26 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 66 — SELL (started 2025-02-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-07 11:15:00 | 1259.75 | 1266.22 | 1266.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-07 12:15:00 | 1254.90 | 1263.96 | 1265.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-10 14:15:00 | 1263.15 | 1257.50 | 1259.84 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-10 14:15:00 | 1263.15 | 1257.50 | 1259.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-10 14:15:00 | 1263.15 | 1257.50 | 1259.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-10 15:00:00 | 1263.15 | 1257.50 | 1259.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-10 15:15:00 | 1259.60 | 1257.92 | 1259.82 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-11 09:15:00 | 1256.50 | 1257.92 | 1259.82 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-11 10:30:00 | 1256.15 | 1258.60 | 1259.81 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-11 11:15:00 | 1258.70 | 1258.60 | 1259.81 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-12 13:45:00 | 1254.05 | 1251.70 | 1254.26 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-13 09:15:00 | 1262.90 | 1254.23 | 1254.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-13 10:00:00 | 1262.90 | 1254.23 | 1254.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-02-13 10:15:00 | 1261.50 | 1255.69 | 1255.41 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 67 — BUY (started 2025-02-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-13 10:15:00 | 1261.50 | 1255.69 | 1255.41 | EMA200 above EMA400 |

### Cycle 68 — SELL (started 2025-02-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-13 13:15:00 | 1249.15 | 1254.20 | 1254.82 | EMA200 below EMA400 |

### Cycle 69 — BUY (started 2025-02-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-14 15:15:00 | 1259.00 | 1254.54 | 1254.15 | EMA200 above EMA400 |

### Cycle 70 — SELL (started 2025-02-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-17 09:15:00 | 1236.00 | 1250.83 | 1252.50 | EMA200 below EMA400 |

### Cycle 71 — BUY (started 2025-02-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-19 11:15:00 | 1254.15 | 1247.93 | 1247.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-19 14:15:00 | 1262.10 | 1253.16 | 1250.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-20 10:15:00 | 1254.40 | 1254.85 | 1251.87 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-20 10:15:00 | 1254.40 | 1254.85 | 1251.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-20 10:15:00 | 1254.40 | 1254.85 | 1251.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-20 10:45:00 | 1251.80 | 1254.85 | 1251.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-20 11:15:00 | 1247.60 | 1253.40 | 1251.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-20 12:00:00 | 1247.60 | 1253.40 | 1251.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-20 12:15:00 | 1250.55 | 1252.83 | 1251.40 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-20 13:45:00 | 1251.65 | 1252.13 | 1251.21 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-20 14:30:00 | 1251.60 | 1251.72 | 1251.11 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-21 09:15:00 | 1227.00 | 1246.20 | 1248.67 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 72 — SELL (started 2025-02-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-21 09:15:00 | 1227.00 | 1246.20 | 1248.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-24 09:15:00 | 1213.40 | 1231.57 | 1239.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-25 09:15:00 | 1226.65 | 1222.35 | 1229.32 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-25 09:15:00 | 1226.65 | 1222.35 | 1229.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-25 09:15:00 | 1226.65 | 1222.35 | 1229.32 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-27 15:15:00 | 1220.00 | 1226.47 | 1227.61 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-28 10:15:00 | 1217.60 | 1224.60 | 1226.50 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-05 09:30:00 | 1220.05 | 1212.12 | 1213.13 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-05 13:15:00 | 1216.55 | 1214.09 | 1213.84 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 73 — BUY (started 2025-03-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-05 13:15:00 | 1216.55 | 1214.09 | 1213.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-06 15:15:00 | 1219.00 | 1216.63 | 1215.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-07 09:15:00 | 1213.00 | 1215.90 | 1215.22 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-07 09:15:00 | 1213.00 | 1215.90 | 1215.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-07 09:15:00 | 1213.00 | 1215.90 | 1215.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-07 09:30:00 | 1208.35 | 1215.90 | 1215.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-07 10:15:00 | 1215.50 | 1215.82 | 1215.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-07 11:00:00 | 1215.50 | 1215.82 | 1215.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-07 11:15:00 | 1212.80 | 1215.22 | 1215.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-07 12:00:00 | 1212.80 | 1215.22 | 1215.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-07 12:15:00 | 1214.55 | 1215.09 | 1214.98 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-07 13:15:00 | 1216.95 | 1215.09 | 1214.98 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-07 13:15:00 | 1212.95 | 1214.66 | 1214.79 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 74 — SELL (started 2025-03-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-07 13:15:00 | 1212.95 | 1214.66 | 1214.79 | EMA200 below EMA400 |

### Cycle 75 — BUY (started 2025-03-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-10 09:15:00 | 1222.85 | 1216.06 | 1215.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-11 09:15:00 | 1234.70 | 1220.77 | 1218.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-12 11:15:00 | 1240.30 | 1240.67 | 1232.86 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-12 12:00:00 | 1240.30 | 1240.67 | 1232.86 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 11:15:00 | 1340.95 | 1352.38 | 1343.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-25 12:00:00 | 1340.95 | 1352.38 | 1343.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 12:15:00 | 1334.30 | 1348.76 | 1342.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-25 12:45:00 | 1332.55 | 1348.76 | 1342.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 14:15:00 | 1343.95 | 1347.35 | 1342.94 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-26 09:15:00 | 1347.00 | 1346.27 | 1342.85 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-26 12:15:00 | 1339.30 | 1344.54 | 1343.23 | SL hit (close<static) qty=1.00 sl=1340.10 alert=retest2 |

### Cycle 76 — SELL (started 2025-03-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-26 14:15:00 | 1336.35 | 1341.56 | 1342.02 | EMA200 below EMA400 |

### Cycle 77 — BUY (started 2025-03-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-27 10:15:00 | 1344.85 | 1342.10 | 1342.09 | EMA200 above EMA400 |

### Cycle 78 — SELL (started 2025-03-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-27 14:15:00 | 1336.15 | 1342.17 | 1342.31 | EMA200 below EMA400 |

### Cycle 79 — BUY (started 2025-03-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-28 11:15:00 | 1344.70 | 1342.18 | 1342.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-28 13:15:00 | 1348.75 | 1343.80 | 1342.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-01 09:15:00 | 1341.95 | 1344.78 | 1343.73 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-01 09:15:00 | 1341.95 | 1344.78 | 1343.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-01 09:15:00 | 1341.95 | 1344.78 | 1343.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-01 09:30:00 | 1349.65 | 1344.78 | 1343.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 80 — SELL (started 2025-04-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-01 10:15:00 | 1330.80 | 1341.99 | 1342.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-01 12:15:00 | 1321.65 | 1335.43 | 1339.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-02 09:15:00 | 1332.50 | 1328.62 | 1334.24 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-02 09:15:00 | 1332.50 | 1328.62 | 1334.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 09:15:00 | 1332.50 | 1328.62 | 1334.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-02 09:30:00 | 1336.30 | 1328.62 | 1334.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 10:15:00 | 1331.70 | 1329.23 | 1334.01 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-02 11:15:00 | 1328.65 | 1329.23 | 1334.01 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-02 12:15:00 | 1328.90 | 1329.43 | 1333.66 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-02 12:45:00 | 1327.35 | 1329.41 | 1333.27 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-02 14:45:00 | 1329.00 | 1329.93 | 1332.85 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 15:15:00 | 1330.85 | 1330.11 | 1332.67 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-03 09:15:00 | 1324.60 | 1330.11 | 1332.67 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-03 10:30:00 | 1329.40 | 1329.82 | 1332.09 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-03 13:30:00 | 1327.95 | 1329.08 | 1331.17 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-03 15:15:00 | 1328.95 | 1329.45 | 1331.15 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 09:15:00 | 1329.65 | 1329.41 | 1330.83 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-04-04 12:15:00 | 1336.50 | 1330.96 | 1331.20 | SL hit (close>static) qty=1.00 sl=1333.55 alert=retest2 |

### Cycle 81 — BUY (started 2025-04-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-04 13:15:00 | 1333.95 | 1331.56 | 1331.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-04 14:15:00 | 1338.55 | 1332.96 | 1332.10 | Break + close above crossover candle high |

### Cycle 82 — SELL (started 2025-04-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-07 09:15:00 | 1288.00 | 1324.29 | 1328.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-07 10:15:00 | 1276.05 | 1314.64 | 1323.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-07 15:15:00 | 1294.10 | 1293.86 | 1307.83 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-08 09:15:00 | 1307.95 | 1293.86 | 1307.83 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 09:15:00 | 1295.85 | 1294.26 | 1306.74 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-08 10:30:00 | 1289.45 | 1294.66 | 1305.79 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-08 11:15:00 | 1291.60 | 1294.66 | 1305.79 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-09 09:15:00 | 1291.55 | 1298.85 | 1303.98 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-09 09:45:00 | 1289.00 | 1296.82 | 1302.59 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-09 13:15:00 | 1303.65 | 1296.35 | 1300.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-09 14:00:00 | 1303.65 | 1296.35 | 1300.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-09 14:15:00 | 1300.35 | 1297.15 | 1300.23 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-04-11 09:15:00 | 1316.15 | 1302.04 | 1301.99 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 83 — BUY (started 2025-04-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-11 09:15:00 | 1316.15 | 1302.04 | 1301.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-15 09:15:00 | 1347.30 | 1318.62 | 1311.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-24 09:15:00 | 1413.80 | 1418.23 | 1410.62 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-24 09:15:00 | 1413.80 | 1418.23 | 1410.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-24 09:15:00 | 1413.80 | 1418.23 | 1410.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-24 09:45:00 | 1411.40 | 1418.23 | 1410.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-24 10:15:00 | 1410.00 | 1416.58 | 1410.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-24 11:00:00 | 1410.00 | 1416.58 | 1410.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-24 11:15:00 | 1407.00 | 1414.67 | 1410.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-24 11:45:00 | 1407.20 | 1414.67 | 1410.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-24 12:15:00 | 1405.30 | 1412.79 | 1409.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-24 12:30:00 | 1406.00 | 1412.79 | 1409.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 84 — SELL (started 2025-04-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-24 15:15:00 | 1401.00 | 1408.09 | 1408.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-25 10:15:00 | 1397.60 | 1404.83 | 1406.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-25 13:15:00 | 1402.00 | 1401.92 | 1404.62 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-25 14:00:00 | 1402.00 | 1401.92 | 1404.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-25 14:15:00 | 1406.00 | 1402.73 | 1404.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-25 15:00:00 | 1406.00 | 1402.73 | 1404.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-25 15:15:00 | 1402.00 | 1402.59 | 1404.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-28 09:15:00 | 1420.00 | 1402.59 | 1404.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 85 — BUY (started 2025-04-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-28 09:15:00 | 1431.30 | 1408.33 | 1406.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-02 09:15:00 | 1445.40 | 1431.11 | 1426.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-02 12:15:00 | 1431.20 | 1433.68 | 1429.25 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-02 13:00:00 | 1431.20 | 1433.68 | 1429.25 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-02 14:15:00 | 1434.00 | 1433.94 | 1430.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-02 14:30:00 | 1431.40 | 1433.94 | 1430.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-02 15:15:00 | 1429.00 | 1432.95 | 1430.04 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-05 09:15:00 | 1438.30 | 1432.95 | 1430.04 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-05 11:15:00 | 1428.40 | 1432.15 | 1430.43 | SL hit (close<static) qty=1.00 sl=1428.60 alert=retest2 |

### Cycle 86 — SELL (started 2025-05-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-06 09:15:00 | 1424.90 | 1429.35 | 1429.65 | EMA200 below EMA400 |

### Cycle 87 — BUY (started 2025-05-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-06 14:15:00 | 1431.50 | 1429.59 | 1429.55 | EMA200 above EMA400 |

### Cycle 88 — SELL (started 2025-05-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-07 10:15:00 | 1425.10 | 1428.81 | 1429.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-07 11:15:00 | 1423.10 | 1427.67 | 1428.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-07 13:15:00 | 1426.70 | 1426.52 | 1427.91 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-07 14:00:00 | 1426.70 | 1426.52 | 1427.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 14:15:00 | 1434.50 | 1428.12 | 1428.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-07 15:00:00 | 1434.50 | 1428.12 | 1428.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 89 — BUY (started 2025-05-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-07 15:15:00 | 1437.10 | 1429.91 | 1429.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-08 09:15:00 | 1441.90 | 1432.31 | 1430.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-08 14:15:00 | 1438.10 | 1438.75 | 1434.93 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-08 15:00:00 | 1438.10 | 1438.75 | 1434.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 15:15:00 | 1428.00 | 1436.60 | 1434.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-09 09:15:00 | 1411.30 | 1436.60 | 1434.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 90 — SELL (started 2025-05-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-09 09:15:00 | 1404.60 | 1430.20 | 1431.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-09 10:15:00 | 1400.70 | 1424.30 | 1428.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-12 09:15:00 | 1433.40 | 1408.03 | 1416.24 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-12 09:15:00 | 1433.40 | 1408.03 | 1416.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-12 09:15:00 | 1433.40 | 1408.03 | 1416.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-12 09:45:00 | 1435.10 | 1408.03 | 1416.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-12 10:15:00 | 1436.50 | 1413.72 | 1418.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-12 11:00:00 | 1436.50 | 1413.72 | 1418.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 91 — BUY (started 2025-05-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 12:15:00 | 1440.10 | 1422.69 | 1421.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 13:15:00 | 1442.40 | 1426.63 | 1423.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-13 09:15:00 | 1434.40 | 1434.73 | 1428.56 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-13 09:15:00 | 1434.40 | 1434.73 | 1428.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-13 09:15:00 | 1434.40 | 1434.73 | 1428.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-13 10:00:00 | 1434.40 | 1434.73 | 1428.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-13 10:15:00 | 1432.30 | 1434.24 | 1428.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-13 10:30:00 | 1431.70 | 1434.24 | 1428.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-13 12:15:00 | 1431.10 | 1433.35 | 1429.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-13 12:45:00 | 1429.00 | 1433.35 | 1429.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-13 13:15:00 | 1433.50 | 1433.38 | 1429.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-13 13:30:00 | 1432.00 | 1433.38 | 1429.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-13 14:15:00 | 1430.90 | 1432.88 | 1429.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-13 15:00:00 | 1430.90 | 1432.88 | 1429.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-13 15:15:00 | 1430.80 | 1432.47 | 1429.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-14 09:15:00 | 1436.90 | 1432.47 | 1429.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-14 09:15:00 | 1438.70 | 1433.71 | 1430.76 | EMA400 retest candle locked (from upside) |

### Cycle 92 — SELL (started 2025-05-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-14 13:15:00 | 1420.40 | 1428.46 | 1429.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-15 09:15:00 | 1418.90 | 1426.09 | 1427.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-15 11:15:00 | 1425.80 | 1425.28 | 1427.10 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-15 11:15:00 | 1425.80 | 1425.28 | 1427.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-15 11:15:00 | 1425.80 | 1425.28 | 1427.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-15 12:00:00 | 1425.80 | 1425.28 | 1427.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-15 12:15:00 | 1439.10 | 1428.04 | 1428.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-15 13:00:00 | 1439.10 | 1428.04 | 1428.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 93 — BUY (started 2025-05-15 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-15 13:15:00 | 1453.00 | 1433.04 | 1430.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-16 09:15:00 | 1454.70 | 1442.52 | 1435.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-19 12:15:00 | 1450.10 | 1452.55 | 1447.04 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-19 13:00:00 | 1450.10 | 1452.55 | 1447.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 13:15:00 | 1446.90 | 1451.42 | 1447.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-19 14:00:00 | 1446.90 | 1451.42 | 1447.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 14:15:00 | 1449.20 | 1450.98 | 1447.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-19 14:45:00 | 1446.10 | 1450.98 | 1447.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 15:15:00 | 1447.00 | 1450.18 | 1447.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 09:30:00 | 1444.10 | 1448.76 | 1446.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 10:15:00 | 1443.80 | 1447.77 | 1446.55 | EMA400 retest candle locked (from upside) |

### Cycle 94 — SELL (started 2025-05-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-20 12:15:00 | 1438.40 | 1444.57 | 1445.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-22 09:15:00 | 1434.30 | 1442.03 | 1443.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-22 14:15:00 | 1442.60 | 1437.51 | 1440.01 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-22 14:15:00 | 1442.60 | 1437.51 | 1440.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 14:15:00 | 1442.60 | 1437.51 | 1440.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-22 14:45:00 | 1442.70 | 1437.51 | 1440.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 15:15:00 | 1440.10 | 1438.03 | 1440.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-23 09:15:00 | 1437.60 | 1438.03 | 1440.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 09:15:00 | 1445.60 | 1439.54 | 1440.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-23 10:00:00 | 1445.60 | 1439.54 | 1440.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 95 — BUY (started 2025-05-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-23 10:15:00 | 1449.20 | 1441.47 | 1441.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-26 09:15:00 | 1466.10 | 1450.39 | 1446.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-27 09:15:00 | 1442.50 | 1454.25 | 1451.04 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-27 09:15:00 | 1442.50 | 1454.25 | 1451.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 09:15:00 | 1442.50 | 1454.25 | 1451.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-27 10:00:00 | 1442.50 | 1454.25 | 1451.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 10:15:00 | 1449.50 | 1453.30 | 1450.90 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-27 11:15:00 | 1459.80 | 1453.30 | 1450.90 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-27 13:15:00 | 1441.30 | 1448.90 | 1449.35 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 96 — SELL (started 2025-05-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-27 13:15:00 | 1441.30 | 1448.90 | 1449.35 | EMA200 below EMA400 |

### Cycle 97 — BUY (started 2025-05-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-28 12:15:00 | 1454.80 | 1450.15 | 1449.65 | EMA200 above EMA400 |

### Cycle 98 — SELL (started 2025-05-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-29 13:15:00 | 1448.70 | 1450.14 | 1450.21 | EMA200 below EMA400 |

### Cycle 99 — BUY (started 2025-05-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-29 14:15:00 | 1457.80 | 1451.67 | 1450.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-29 15:15:00 | 1461.50 | 1453.64 | 1451.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-30 11:15:00 | 1453.90 | 1454.95 | 1453.03 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-30 12:00:00 | 1453.90 | 1454.95 | 1453.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 12:15:00 | 1453.20 | 1454.60 | 1453.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-30 12:30:00 | 1453.80 | 1454.60 | 1453.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 13:15:00 | 1446.10 | 1452.90 | 1452.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-30 14:00:00 | 1446.10 | 1452.90 | 1452.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 100 — SELL (started 2025-05-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-30 14:15:00 | 1446.50 | 1451.62 | 1451.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-02 09:15:00 | 1433.80 | 1447.17 | 1449.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-02 14:15:00 | 1451.50 | 1444.97 | 1447.23 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-02 14:15:00 | 1451.50 | 1444.97 | 1447.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 14:15:00 | 1451.50 | 1444.97 | 1447.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-02 15:00:00 | 1451.50 | 1444.97 | 1447.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 15:15:00 | 1447.70 | 1445.51 | 1447.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-03 09:15:00 | 1437.20 | 1445.51 | 1447.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 09:15:00 | 1439.50 | 1444.31 | 1446.56 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-04 09:15:00 | 1431.20 | 1439.15 | 1442.50 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-05 11:15:00 | 1456.10 | 1439.47 | 1439.03 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 101 — BUY (started 2025-06-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-05 11:15:00 | 1456.10 | 1439.47 | 1439.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-06 10:15:00 | 1456.70 | 1449.53 | 1445.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-09 09:15:00 | 1448.90 | 1454.29 | 1450.01 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-09 09:15:00 | 1448.90 | 1454.29 | 1450.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-09 09:15:00 | 1448.90 | 1454.29 | 1450.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-09 09:45:00 | 1446.50 | 1454.29 | 1450.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-09 10:15:00 | 1442.90 | 1452.01 | 1449.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-09 11:00:00 | 1442.90 | 1452.01 | 1449.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-09 11:15:00 | 1441.40 | 1449.89 | 1448.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-09 12:00:00 | 1441.40 | 1449.89 | 1448.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 102 — SELL (started 2025-06-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-09 13:15:00 | 1437.00 | 1445.89 | 1446.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-09 14:15:00 | 1434.90 | 1443.69 | 1445.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-11 09:15:00 | 1427.90 | 1426.69 | 1433.24 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-11 09:45:00 | 1429.60 | 1426.69 | 1433.24 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 11:15:00 | 1431.90 | 1428.63 | 1433.04 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-11 14:00:00 | 1427.00 | 1428.67 | 1432.32 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-12 10:00:00 | 1427.90 | 1429.18 | 1431.69 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-16 15:15:00 | 1427.90 | 1423.60 | 1423.12 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 103 — BUY (started 2025-06-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-16 15:15:00 | 1427.90 | 1423.60 | 1423.12 | EMA200 above EMA400 |

### Cycle 104 — SELL (started 2025-06-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-17 12:15:00 | 1419.90 | 1423.05 | 1423.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-18 10:15:00 | 1416.10 | 1421.38 | 1422.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-20 09:15:00 | 1415.40 | 1412.21 | 1414.71 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-20 09:15:00 | 1415.40 | 1412.21 | 1414.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 09:15:00 | 1415.40 | 1412.21 | 1414.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 09:45:00 | 1415.90 | 1412.21 | 1414.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 10:15:00 | 1425.90 | 1414.95 | 1415.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 11:00:00 | 1425.90 | 1414.95 | 1415.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 105 — BUY (started 2025-06-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-20 11:15:00 | 1430.30 | 1418.02 | 1417.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-20 12:15:00 | 1431.60 | 1420.73 | 1418.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-23 09:15:00 | 1414.00 | 1421.68 | 1419.86 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-23 09:15:00 | 1414.00 | 1421.68 | 1419.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 09:15:00 | 1414.00 | 1421.68 | 1419.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-23 10:00:00 | 1414.00 | 1421.68 | 1419.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 10:15:00 | 1416.10 | 1420.56 | 1419.51 | EMA400 retest candle locked (from upside) |

### Cycle 106 — SELL (started 2025-06-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-23 12:15:00 | 1415.20 | 1418.31 | 1418.60 | EMA200 below EMA400 |

### Cycle 107 — BUY (started 2025-06-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-23 14:15:00 | 1419.20 | 1418.79 | 1418.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-24 09:15:00 | 1431.50 | 1421.38 | 1419.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-24 12:15:00 | 1424.00 | 1425.21 | 1422.36 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-24 12:15:00 | 1424.00 | 1425.21 | 1422.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 12:15:00 | 1424.00 | 1425.21 | 1422.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-24 12:30:00 | 1424.20 | 1425.21 | 1422.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 13:15:00 | 1419.70 | 1424.11 | 1422.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-24 13:30:00 | 1417.90 | 1424.11 | 1422.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 14:15:00 | 1423.60 | 1424.01 | 1422.25 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-25 09:30:00 | 1426.50 | 1423.63 | 1422.37 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-25 14:15:00 | 1425.10 | 1423.07 | 1422.41 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-26 09:15:00 | 1428.90 | 1423.67 | 1422.82 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-01 11:15:00 | 1435.70 | 1442.98 | 1443.12 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 108 — SELL (started 2025-07-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-01 11:15:00 | 1435.70 | 1442.98 | 1443.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-01 14:15:00 | 1430.90 | 1438.66 | 1440.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-03 09:15:00 | 1442.80 | 1432.81 | 1435.45 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-03 09:15:00 | 1442.80 | 1432.81 | 1435.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 09:15:00 | 1442.80 | 1432.81 | 1435.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-03 10:00:00 | 1442.80 | 1432.81 | 1435.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 10:15:00 | 1432.80 | 1432.81 | 1435.21 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-03 11:15:00 | 1431.10 | 1432.81 | 1435.21 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-03 13:45:00 | 1429.00 | 1430.97 | 1433.69 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-04 14:15:00 | 1443.20 | 1432.37 | 1432.29 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 109 — BUY (started 2025-07-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-04 14:15:00 | 1443.20 | 1432.37 | 1432.29 | EMA200 above EMA400 |

### Cycle 110 — SELL (started 2025-07-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-09 13:15:00 | 1431.40 | 1435.27 | 1435.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-10 09:15:00 | 1430.00 | 1433.23 | 1434.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-11 14:15:00 | 1422.00 | 1421.63 | 1425.67 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-11 15:00:00 | 1422.00 | 1421.63 | 1425.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 13:15:00 | 1423.00 | 1420.79 | 1423.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-14 13:30:00 | 1421.80 | 1420.79 | 1423.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 14:15:00 | 1423.30 | 1421.29 | 1423.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-14 14:45:00 | 1423.80 | 1421.29 | 1423.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 15:15:00 | 1421.90 | 1421.42 | 1423.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-15 09:15:00 | 1421.80 | 1421.42 | 1423.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 09:15:00 | 1422.80 | 1421.69 | 1423.20 | EMA400 retest candle locked (from downside) |

### Cycle 111 — BUY (started 2025-07-15 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-15 13:15:00 | 1427.40 | 1424.26 | 1424.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-15 14:15:00 | 1431.40 | 1425.69 | 1424.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-16 09:15:00 | 1421.50 | 1425.86 | 1425.02 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-16 09:15:00 | 1421.50 | 1425.86 | 1425.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 09:15:00 | 1421.50 | 1425.86 | 1425.02 | EMA400 retest candle locked (from upside) |

### Cycle 112 — SELL (started 2025-07-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-17 09:15:00 | 1412.60 | 1422.96 | 1424.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-18 11:15:00 | 1409.50 | 1416.38 | 1419.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-18 13:15:00 | 1420.60 | 1416.94 | 1419.18 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-18 13:15:00 | 1420.60 | 1416.94 | 1419.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 13:15:00 | 1420.60 | 1416.94 | 1419.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-18 14:00:00 | 1420.60 | 1416.94 | 1419.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 14:15:00 | 1425.80 | 1418.71 | 1419.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-18 14:30:00 | 1424.20 | 1418.71 | 1419.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 15:15:00 | 1426.70 | 1420.31 | 1420.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-21 09:15:00 | 1440.50 | 1420.31 | 1420.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 113 — BUY (started 2025-07-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-21 09:15:00 | 1459.80 | 1428.21 | 1423.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-21 14:15:00 | 1465.50 | 1450.91 | 1438.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-24 14:15:00 | 1483.40 | 1483.49 | 1476.02 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-24 15:00:00 | 1483.40 | 1483.49 | 1476.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 11:15:00 | 1475.70 | 1481.43 | 1477.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-25 12:00:00 | 1475.70 | 1481.43 | 1477.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 12:15:00 | 1476.90 | 1480.52 | 1477.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-25 12:30:00 | 1476.20 | 1480.52 | 1477.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 13:15:00 | 1474.60 | 1479.34 | 1477.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-25 14:00:00 | 1474.60 | 1479.34 | 1477.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 14:15:00 | 1478.10 | 1479.09 | 1477.20 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-28 09:15:00 | 1482.90 | 1478.85 | 1477.26 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-31 09:15:00 | 1466.10 | 1479.74 | 1481.57 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 114 — SELL (started 2025-07-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-31 09:15:00 | 1466.10 | 1479.74 | 1481.57 | EMA200 below EMA400 |

### Cycle 115 — BUY (started 2025-07-31 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-31 13:15:00 | 1492.10 | 1483.46 | 1482.65 | EMA200 above EMA400 |

### Cycle 116 — SELL (started 2025-08-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-01 10:15:00 | 1477.10 | 1482.12 | 1482.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-01 11:15:00 | 1473.90 | 1480.48 | 1481.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-07 14:15:00 | 1440.20 | 1437.92 | 1444.93 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-07 15:00:00 | 1440.20 | 1437.92 | 1444.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 09:15:00 | 1437.60 | 1438.35 | 1443.94 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-08 15:15:00 | 1432.00 | 1437.13 | 1441.17 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-12 09:15:00 | 1427.70 | 1432.50 | 1435.46 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-18 09:15:00 | 1447.20 | 1430.50 | 1428.45 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 117 — BUY (started 2025-08-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-18 09:15:00 | 1447.20 | 1430.50 | 1428.45 | EMA200 above EMA400 |

### Cycle 118 — SELL (started 2025-08-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-20 11:15:00 | 1429.80 | 1432.09 | 1432.14 | EMA200 below EMA400 |

### Cycle 119 — BUY (started 2025-08-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-20 13:15:00 | 1433.20 | 1432.22 | 1432.18 | EMA200 above EMA400 |

### Cycle 120 — SELL (started 2025-08-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-20 14:15:00 | 1430.80 | 1431.94 | 1432.06 | EMA200 below EMA400 |

### Cycle 121 — BUY (started 2025-08-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-21 09:15:00 | 1443.60 | 1433.96 | 1432.94 | EMA200 above EMA400 |

### Cycle 122 — SELL (started 2025-08-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-25 09:15:00 | 1425.00 | 1435.33 | 1436.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-26 09:15:00 | 1417.20 | 1429.13 | 1432.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-01 09:15:00 | 1404.50 | 1401.29 | 1406.93 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-01 09:15:00 | 1404.50 | 1401.29 | 1406.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 09:15:00 | 1404.50 | 1401.29 | 1406.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 09:30:00 | 1406.80 | 1401.29 | 1406.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 10:15:00 | 1404.80 | 1401.99 | 1406.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 11:00:00 | 1404.80 | 1401.99 | 1406.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 11:15:00 | 1405.20 | 1402.63 | 1406.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 11:30:00 | 1404.70 | 1402.63 | 1406.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 12:15:00 | 1409.90 | 1404.09 | 1406.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 13:00:00 | 1409.90 | 1404.09 | 1406.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 13:15:00 | 1408.00 | 1404.87 | 1407.00 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-01 14:15:00 | 1406.30 | 1404.87 | 1407.00 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-01 14:15:00 | 1411.80 | 1406.26 | 1407.43 | SL hit (close>static) qty=1.00 sl=1410.70 alert=retest2 |

### Cycle 123 — BUY (started 2025-09-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-02 10:15:00 | 1411.40 | 1408.33 | 1408.19 | EMA200 above EMA400 |

### Cycle 124 — SELL (started 2025-09-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-02 11:15:00 | 1404.50 | 1407.56 | 1407.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-02 12:15:00 | 1403.70 | 1406.79 | 1407.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-03 14:15:00 | 1398.00 | 1395.44 | 1399.34 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-03 14:15:00 | 1398.00 | 1395.44 | 1399.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 14:15:00 | 1398.00 | 1395.44 | 1399.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-03 15:00:00 | 1398.00 | 1395.44 | 1399.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 09:15:00 | 1407.00 | 1397.93 | 1399.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-04 09:30:00 | 1410.80 | 1397.93 | 1399.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 125 — BUY (started 2025-09-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-04 11:15:00 | 1408.90 | 1401.91 | 1401.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-04 12:15:00 | 1412.40 | 1404.01 | 1402.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-05 09:15:00 | 1397.30 | 1403.59 | 1402.82 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-05 09:15:00 | 1397.30 | 1403.59 | 1402.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 09:15:00 | 1397.30 | 1403.59 | 1402.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-05 10:00:00 | 1397.30 | 1403.59 | 1402.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 126 — SELL (started 2025-09-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-05 10:15:00 | 1393.80 | 1401.64 | 1402.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-05 11:15:00 | 1391.00 | 1399.51 | 1401.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-05 14:15:00 | 1401.60 | 1398.61 | 1400.07 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-05 14:15:00 | 1401.60 | 1398.61 | 1400.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 14:15:00 | 1401.60 | 1398.61 | 1400.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-05 15:00:00 | 1401.60 | 1398.61 | 1400.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 15:15:00 | 1402.30 | 1399.35 | 1400.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-08 09:15:00 | 1403.30 | 1399.35 | 1400.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 09:15:00 | 1402.10 | 1399.90 | 1400.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-08 09:30:00 | 1406.60 | 1399.90 | 1400.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 10:15:00 | 1401.80 | 1400.28 | 1400.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-08 11:15:00 | 1402.10 | 1400.28 | 1400.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 127 — BUY (started 2025-09-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-08 11:15:00 | 1403.00 | 1400.82 | 1400.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-08 12:15:00 | 1407.80 | 1402.22 | 1401.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-08 14:15:00 | 1402.20 | 1402.98 | 1401.95 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-08 14:15:00 | 1402.20 | 1402.98 | 1401.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 14:15:00 | 1402.20 | 1402.98 | 1401.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-08 15:00:00 | 1402.20 | 1402.98 | 1401.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 15:15:00 | 1403.00 | 1402.98 | 1402.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-09 09:15:00 | 1400.60 | 1402.98 | 1402.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 09:15:00 | 1399.20 | 1402.23 | 1401.79 | EMA400 retest candle locked (from upside) |

### Cycle 128 — SELL (started 2025-09-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-09 11:15:00 | 1397.10 | 1400.86 | 1401.22 | EMA200 below EMA400 |

### Cycle 129 — BUY (started 2025-09-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-09 13:15:00 | 1405.10 | 1401.48 | 1401.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-10 09:15:00 | 1409.70 | 1403.58 | 1402.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-10 12:15:00 | 1404.80 | 1405.13 | 1403.57 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-10 13:00:00 | 1404.80 | 1405.13 | 1403.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 13:15:00 | 1408.40 | 1405.79 | 1404.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-10 14:15:00 | 1405.50 | 1405.79 | 1404.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 14:15:00 | 1403.80 | 1405.39 | 1403.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-10 14:45:00 | 1403.70 | 1405.39 | 1403.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 15:15:00 | 1403.30 | 1404.97 | 1403.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-11 09:15:00 | 1404.20 | 1404.97 | 1403.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 09:15:00 | 1403.40 | 1404.66 | 1403.88 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-11 13:15:00 | 1407.60 | 1404.91 | 1404.21 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-11 15:15:00 | 1401.60 | 1403.72 | 1403.82 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 130 — SELL (started 2025-09-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-11 15:15:00 | 1401.60 | 1403.72 | 1403.82 | EMA200 below EMA400 |

### Cycle 131 — BUY (started 2025-09-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-12 10:15:00 | 1408.40 | 1404.54 | 1404.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-12 11:15:00 | 1412.10 | 1406.05 | 1404.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-15 14:15:00 | 1419.70 | 1419.70 | 1414.76 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-15 15:00:00 | 1419.70 | 1419.70 | 1414.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 09:15:00 | 1421.20 | 1419.73 | 1415.61 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-18 10:15:00 | 1425.30 | 1420.94 | 1419.55 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-18 12:45:00 | 1425.20 | 1422.00 | 1420.45 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-19 09:15:00 | 1405.10 | 1418.61 | 1419.38 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 132 — SELL (started 2025-09-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-19 09:15:00 | 1405.10 | 1418.61 | 1419.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-19 12:15:00 | 1403.20 | 1411.36 | 1415.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-22 12:15:00 | 1409.60 | 1406.40 | 1410.16 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-22 13:00:00 | 1409.60 | 1406.40 | 1410.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 09:15:00 | 1365.00 | 1353.14 | 1357.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-01 10:00:00 | 1365.00 | 1353.14 | 1357.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 10:15:00 | 1371.30 | 1356.78 | 1358.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-01 11:00:00 | 1371.30 | 1356.78 | 1358.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 133 — BUY (started 2025-10-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-01 11:15:00 | 1375.20 | 1360.46 | 1360.06 | EMA200 above EMA400 |

### Cycle 134 — SELL (started 2025-10-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-06 15:15:00 | 1363.90 | 1365.23 | 1365.25 | EMA200 below EMA400 |

### Cycle 135 — BUY (started 2025-10-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-07 09:15:00 | 1380.80 | 1368.34 | 1366.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-10 10:15:00 | 1384.00 | 1376.62 | 1374.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-10 15:15:00 | 1379.00 | 1379.23 | 1376.62 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-13 09:15:00 | 1381.00 | 1379.23 | 1376.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 09:15:00 | 1379.00 | 1379.18 | 1376.84 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-15 09:15:00 | 1389.50 | 1381.46 | 1379.59 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-20 14:15:00 | 1390.20 | 1406.17 | 1408.12 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 136 — SELL (started 2025-10-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-20 14:15:00 | 1390.20 | 1406.17 | 1408.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-21 13:15:00 | 1386.00 | 1400.03 | 1404.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-24 09:15:00 | 1381.10 | 1376.89 | 1387.37 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-24 10:00:00 | 1381.10 | 1376.89 | 1387.37 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 09:15:00 | 1380.80 | 1377.71 | 1382.62 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-27 13:15:00 | 1375.70 | 1378.33 | 1381.75 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-28 09:15:00 | 1363.60 | 1378.46 | 1380.95 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-07 15:15:00 | 1342.60 | 1337.48 | 1336.92 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 137 — BUY (started 2025-11-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-07 15:15:00 | 1342.60 | 1337.48 | 1336.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-10 12:15:00 | 1345.20 | 1341.17 | 1339.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-12 14:15:00 | 1358.70 | 1358.97 | 1353.62 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-12 15:00:00 | 1358.70 | 1358.97 | 1353.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 10:15:00 | 1373.50 | 1377.81 | 1370.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-14 10:30:00 | 1370.00 | 1377.81 | 1370.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 11:15:00 | 1368.50 | 1375.95 | 1370.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-14 11:45:00 | 1368.00 | 1375.95 | 1370.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 12:15:00 | 1367.00 | 1374.16 | 1369.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-14 13:00:00 | 1367.00 | 1374.16 | 1369.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 09:15:00 | 1376.00 | 1377.23 | 1374.46 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-19 15:00:00 | 1383.30 | 1378.57 | 1376.47 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-20 13:45:00 | 1384.40 | 1381.24 | 1378.89 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-21 09:15:00 | 1368.70 | 1379.09 | 1378.54 | SL hit (close<static) qty=1.00 sl=1373.00 alert=retest2 |

### Cycle 138 — SELL (started 2025-11-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-21 10:15:00 | 1370.40 | 1377.35 | 1377.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-24 15:15:00 | 1366.00 | 1370.32 | 1372.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-25 09:15:00 | 1374.20 | 1371.09 | 1372.63 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-25 09:15:00 | 1374.20 | 1371.09 | 1372.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 09:15:00 | 1374.20 | 1371.09 | 1372.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-25 10:00:00 | 1374.20 | 1371.09 | 1372.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 10:15:00 | 1362.30 | 1369.33 | 1371.69 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-25 13:30:00 | 1361.50 | 1367.00 | 1369.94 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-25 14:15:00 | 1360.20 | 1367.00 | 1369.94 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-26 14:15:00 | 1375.50 | 1369.14 | 1368.97 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 139 — BUY (started 2025-11-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-26 14:15:00 | 1375.50 | 1369.14 | 1368.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-27 09:15:00 | 1389.00 | 1374.05 | 1371.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-28 11:15:00 | 1387.90 | 1388.54 | 1382.47 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-28 12:00:00 | 1387.90 | 1388.54 | 1382.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 14:15:00 | 1390.70 | 1391.52 | 1388.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-01 14:45:00 | 1387.40 | 1391.52 | 1388.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 09:15:00 | 1373.10 | 1387.62 | 1386.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-02 09:45:00 | 1374.20 | 1387.62 | 1386.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 140 — SELL (started 2025-12-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-02 10:15:00 | 1375.20 | 1385.14 | 1385.87 | EMA200 below EMA400 |

### Cycle 141 — BUY (started 2025-12-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-03 14:15:00 | 1392.20 | 1383.14 | 1382.47 | EMA200 above EMA400 |

### Cycle 142 — SELL (started 2025-12-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-09 09:15:00 | 1381.10 | 1387.63 | 1387.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-09 11:15:00 | 1375.60 | 1384.10 | 1386.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-12 09:15:00 | 1367.60 | 1363.02 | 1367.70 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-12 09:15:00 | 1367.60 | 1363.02 | 1367.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 09:15:00 | 1367.60 | 1363.02 | 1367.70 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-12 10:15:00 | 1364.40 | 1363.02 | 1367.70 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-12 13:15:00 | 1364.30 | 1364.60 | 1367.33 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-15 09:15:00 | 1361.00 | 1365.37 | 1367.05 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-15 10:15:00 | 1364.40 | 1365.46 | 1366.93 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 11:15:00 | 1366.00 | 1365.24 | 1366.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-15 12:00:00 | 1366.00 | 1365.24 | 1366.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 12:15:00 | 1367.50 | 1365.69 | 1366.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-15 12:30:00 | 1366.10 | 1365.69 | 1366.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 13:15:00 | 1365.40 | 1365.63 | 1366.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-15 13:30:00 | 1365.70 | 1365.63 | 1366.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 14:15:00 | 1365.10 | 1365.53 | 1366.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-15 14:45:00 | 1366.70 | 1365.53 | 1366.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 09:15:00 | 1364.40 | 1365.22 | 1366.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-16 10:15:00 | 1365.80 | 1365.22 | 1366.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 10:15:00 | 1365.00 | 1365.17 | 1366.00 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-16 11:30:00 | 1362.30 | 1364.84 | 1365.77 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-16 12:15:00 | 1363.10 | 1364.84 | 1365.77 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-16 13:15:00 | 1362.80 | 1364.53 | 1365.55 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-16 14:00:00 | 1362.20 | 1364.06 | 1365.24 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 14:15:00 | 1364.50 | 1364.15 | 1365.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-16 15:00:00 | 1364.50 | 1364.15 | 1365.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 15:15:00 | 1366.60 | 1364.64 | 1365.31 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-17 09:15:00 | 1352.30 | 1364.64 | 1365.31 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-18 11:45:00 | 1363.50 | 1358.53 | 1359.38 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-18 13:00:00 | 1362.30 | 1359.29 | 1359.65 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-22 10:15:00 | 1362.20 | 1357.18 | 1357.71 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-22 10:15:00 | 1363.70 | 1358.48 | 1358.25 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 143 — BUY (started 2025-12-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-22 10:15:00 | 1363.70 | 1358.48 | 1358.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-22 11:15:00 | 1368.30 | 1360.45 | 1359.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-23 09:15:00 | 1360.40 | 1364.25 | 1362.00 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-23 09:15:00 | 1360.40 | 1364.25 | 1362.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 09:15:00 | 1360.40 | 1364.25 | 1362.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-23 09:30:00 | 1360.40 | 1364.25 | 1362.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 10:15:00 | 1360.30 | 1363.46 | 1361.84 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-23 13:00:00 | 1361.40 | 1362.64 | 1361.72 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-23 13:45:00 | 1361.40 | 1362.37 | 1361.69 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-24 12:45:00 | 1361.80 | 1362.80 | 1362.35 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-24 13:45:00 | 1361.70 | 1362.66 | 1362.33 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 14:15:00 | 1360.90 | 1362.31 | 1362.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-24 15:00:00 | 1360.90 | 1362.31 | 1362.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-12-24 15:15:00 | 1359.00 | 1361.65 | 1361.91 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 144 — SELL (started 2025-12-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-24 15:15:00 | 1359.00 | 1361.65 | 1361.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-26 09:15:00 | 1356.30 | 1360.58 | 1361.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-30 14:15:00 | 1342.90 | 1342.50 | 1346.84 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-30 15:00:00 | 1342.90 | 1342.50 | 1346.84 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 15:15:00 | 1347.00 | 1343.40 | 1346.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 09:15:00 | 1345.50 | 1343.40 | 1346.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 09:15:00 | 1342.40 | 1343.20 | 1346.45 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-31 10:15:00 | 1341.00 | 1343.20 | 1346.45 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-31 11:15:00 | 1340.10 | 1342.92 | 1346.03 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-31 12:15:00 | 1340.90 | 1342.74 | 1345.66 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-01 11:45:00 | 1340.70 | 1343.25 | 1344.80 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 09:15:00 | 1344.10 | 1341.25 | 1343.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-02 09:30:00 | 1347.70 | 1341.25 | 1343.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 10:15:00 | 1342.60 | 1341.52 | 1342.98 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-01-02 11:15:00 | 1350.00 | 1343.22 | 1343.62 | SL hit (close>static) qty=1.00 sl=1347.90 alert=retest2 |

### Cycle 145 — BUY (started 2026-01-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-02 12:15:00 | 1350.80 | 1344.73 | 1344.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-02 14:15:00 | 1355.10 | 1348.00 | 1345.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-09 09:15:00 | 1412.90 | 1428.42 | 1418.50 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-09 09:15:00 | 1412.90 | 1428.42 | 1418.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 09:15:00 | 1412.90 | 1428.42 | 1418.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-09 10:00:00 | 1412.90 | 1428.42 | 1418.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 10:15:00 | 1403.00 | 1423.34 | 1417.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-09 10:45:00 | 1405.00 | 1423.34 | 1417.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 146 — SELL (started 2026-01-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-09 13:15:00 | 1403.80 | 1413.72 | 1413.73 | EMA200 below EMA400 |

### Cycle 147 — BUY (started 2026-01-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-12 15:15:00 | 1420.00 | 1413.42 | 1412.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-13 09:15:00 | 1422.70 | 1415.27 | 1413.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-14 09:15:00 | 1421.40 | 1428.68 | 1423.08 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-14 09:15:00 | 1421.40 | 1428.68 | 1423.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 09:15:00 | 1421.40 | 1428.68 | 1423.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-14 10:00:00 | 1421.40 | 1428.68 | 1423.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 10:15:00 | 1417.20 | 1426.39 | 1422.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-14 11:00:00 | 1417.20 | 1426.39 | 1422.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 11:15:00 | 1429.20 | 1426.95 | 1423.15 | EMA400 retest candle locked (from upside) |

### Cycle 148 — SELL (started 2026-01-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-16 09:15:00 | 1413.40 | 1420.45 | 1421.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-16 12:15:00 | 1401.60 | 1415.56 | 1418.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-16 13:15:00 | 1415.60 | 1415.57 | 1418.37 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-16 14:00:00 | 1415.60 | 1415.57 | 1418.37 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 14:15:00 | 1408.30 | 1414.11 | 1417.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-16 14:30:00 | 1418.80 | 1414.11 | 1417.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 15:15:00 | 1413.00 | 1413.89 | 1417.05 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-19 09:15:00 | 1369.90 | 1413.89 | 1417.05 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-27 15:15:00 | 1367.10 | 1354.53 | 1353.19 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 149 — BUY (started 2026-01-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-27 15:15:00 | 1367.10 | 1354.53 | 1353.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-28 09:15:00 | 1370.20 | 1357.66 | 1354.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-29 09:15:00 | 1363.90 | 1365.01 | 1360.89 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-29 09:15:00 | 1363.90 | 1365.01 | 1360.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 09:15:00 | 1363.90 | 1365.01 | 1360.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-29 09:30:00 | 1356.20 | 1365.01 | 1360.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 09:15:00 | 1377.10 | 1375.16 | 1368.58 | EMA400 retest candle locked (from upside) |

### Cycle 150 — SELL (started 2026-01-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-30 14:15:00 | 1355.60 | 1364.45 | 1365.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-01 12:15:00 | 1345.40 | 1357.70 | 1361.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-02 12:15:00 | 1347.30 | 1345.66 | 1351.77 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-02 12:45:00 | 1347.70 | 1345.66 | 1351.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 14:15:00 | 1352.80 | 1347.61 | 1351.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-02 15:00:00 | 1352.80 | 1347.61 | 1351.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 15:15:00 | 1351.00 | 1348.29 | 1351.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-03 09:15:00 | 1392.30 | 1348.29 | 1351.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 151 — BUY (started 2026-02-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 09:15:00 | 1390.10 | 1356.65 | 1355.07 | EMA200 above EMA400 |

### Cycle 152 — SELL (started 2026-02-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-09 13:15:00 | 1391.90 | 1397.48 | 1397.53 | EMA200 below EMA400 |

### Cycle 153 — BUY (started 2026-02-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-10 10:15:00 | 1399.90 | 1397.66 | 1397.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-10 12:15:00 | 1404.90 | 1399.22 | 1398.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-11 10:15:00 | 1402.40 | 1402.54 | 1400.53 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-11 10:15:00 | 1402.40 | 1402.54 | 1400.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 10:15:00 | 1402.40 | 1402.54 | 1400.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-11 10:30:00 | 1402.40 | 1402.54 | 1400.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 11:15:00 | 1411.80 | 1420.92 | 1415.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-13 12:00:00 | 1411.80 | 1420.92 | 1415.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 12:15:00 | 1418.00 | 1420.34 | 1416.04 | EMA400 retest candle locked (from upside) |

### Cycle 154 — SELL (started 2026-02-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-16 10:15:00 | 1406.00 | 1414.20 | 1414.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-16 11:15:00 | 1405.20 | 1412.40 | 1413.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-16 15:15:00 | 1413.00 | 1411.42 | 1412.60 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-16 15:15:00 | 1413.00 | 1411.42 | 1412.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 15:15:00 | 1413.00 | 1411.42 | 1412.60 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-17 09:15:00 | 1394.80 | 1411.42 | 1412.60 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-23 11:15:00 | 1402.50 | 1399.28 | 1399.20 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 155 — BUY (started 2026-02-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-23 11:15:00 | 1402.50 | 1399.28 | 1399.20 | EMA200 above EMA400 |

### Cycle 156 — SELL (started 2026-02-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-23 12:15:00 | 1398.00 | 1399.03 | 1399.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-24 09:15:00 | 1394.60 | 1398.09 | 1398.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-25 09:15:00 | 1399.30 | 1392.64 | 1394.90 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-25 09:15:00 | 1399.30 | 1392.64 | 1394.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 09:15:00 | 1399.30 | 1392.64 | 1394.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-25 09:45:00 | 1400.50 | 1392.64 | 1394.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 10:15:00 | 1394.90 | 1393.09 | 1394.90 | EMA400 retest candle locked (from downside) |

### Cycle 157 — BUY (started 2026-02-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-25 13:15:00 | 1401.90 | 1396.83 | 1396.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-26 10:15:00 | 1405.50 | 1399.99 | 1398.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-26 12:15:00 | 1399.10 | 1400.79 | 1398.84 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-26 12:15:00 | 1399.10 | 1400.79 | 1398.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 12:15:00 | 1399.10 | 1400.79 | 1398.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-26 13:00:00 | 1399.10 | 1400.79 | 1398.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 13:15:00 | 1403.40 | 1401.31 | 1399.26 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-26 14:15:00 | 1406.00 | 1401.31 | 1399.26 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-27 09:15:00 | 1394.40 | 1400.27 | 1399.34 | SL hit (close<static) qty=1.00 sl=1398.00 alert=retest2 |

### Cycle 158 — SELL (started 2026-02-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-27 10:15:00 | 1389.40 | 1398.10 | 1398.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-27 12:15:00 | 1383.70 | 1393.71 | 1396.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-02 14:15:00 | 1376.90 | 1374.41 | 1382.28 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-02 15:00:00 | 1376.90 | 1374.41 | 1382.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-04 14:15:00 | 1363.20 | 1363.25 | 1371.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-04 14:45:00 | 1373.60 | 1363.25 | 1371.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 10:15:00 | 1368.90 | 1364.17 | 1369.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-05 11:00:00 | 1368.90 | 1364.17 | 1369.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 11:15:00 | 1356.90 | 1362.71 | 1368.59 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-05 12:45:00 | 1355.70 | 1361.25 | 1367.39 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-05 15:00:00 | 1354.20 | 1357.88 | 1364.67 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-09 09:15:00 | 1287.91 | 1315.89 | 1336.09 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-09 09:15:00 | 1286.49 | 1315.89 | 1336.09 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-10 09:15:00 | 1297.40 | 1288.45 | 1308.51 | SL hit (close>ema200) qty=0.50 sl=1288.45 alert=retest2 |

### Cycle 159 — BUY (started 2026-03-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-17 10:15:00 | 1280.10 | 1268.54 | 1268.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-17 13:15:00 | 1292.00 | 1277.07 | 1272.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-18 09:15:00 | 1272.20 | 1279.63 | 1275.20 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-18 09:15:00 | 1272.20 | 1279.63 | 1275.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 09:15:00 | 1272.20 | 1279.63 | 1275.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-18 10:00:00 | 1272.20 | 1279.63 | 1275.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 10:15:00 | 1289.10 | 1281.53 | 1276.46 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-18 11:15:00 | 1291.70 | 1281.53 | 1276.46 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-18 12:45:00 | 1291.10 | 1285.08 | 1279.05 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-18 15:15:00 | 1292.00 | 1287.33 | 1281.21 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-19 09:15:00 | 1261.00 | 1282.81 | 1280.26 | SL hit (close<static) qty=1.00 sl=1272.00 alert=retest2 |

### Cycle 160 — SELL (started 2026-03-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 11:15:00 | 1268.20 | 1276.78 | 1277.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 14:15:00 | 1250.40 | 1267.80 | 1273.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-20 09:15:00 | 1267.60 | 1266.46 | 1271.51 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-20 09:15:00 | 1267.60 | 1266.46 | 1271.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 09:15:00 | 1267.60 | 1266.46 | 1271.51 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-20 12:15:00 | 1256.20 | 1266.64 | 1270.78 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-20 14:15:00 | 1257.90 | 1265.27 | 1269.41 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-25 09:15:00 | 1265.40 | 1246.43 | 1245.95 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 161 — BUY (started 2026-03-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 09:15:00 | 1265.40 | 1246.43 | 1245.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 10:15:00 | 1268.20 | 1250.78 | 1247.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-27 09:15:00 | 1253.10 | 1256.97 | 1253.30 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-27 09:15:00 | 1253.10 | 1256.97 | 1253.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 09:15:00 | 1253.10 | 1256.97 | 1253.30 | EMA400 retest candle locked (from upside) |

### Cycle 162 — SELL (started 2026-03-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 13:15:00 | 1242.40 | 1251.10 | 1251.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-27 14:15:00 | 1234.30 | 1247.74 | 1249.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 1222.30 | 1220.78 | 1231.15 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 1222.30 | 1220.78 | 1231.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 1222.30 | 1220.78 | 1231.15 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-01 10:15:00 | 1216.80 | 1220.78 | 1231.15 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-01 13:30:00 | 1210.80 | 1220.05 | 1227.45 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-02 15:15:00 | 1216.50 | 1207.58 | 1215.10 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-06 14:15:00 | 1233.60 | 1219.46 | 1217.68 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 163 — BUY (started 2026-04-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-06 14:15:00 | 1233.60 | 1219.46 | 1217.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-07 13:15:00 | 1236.10 | 1225.61 | 1221.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-09 11:15:00 | 1290.30 | 1292.65 | 1272.73 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-09 12:00:00 | 1290.30 | 1292.65 | 1272.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 1310.00 | 1312.90 | 1298.84 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 10:15:00 | 1314.50 | 1312.90 | 1298.84 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-23 10:15:00 | 1346.60 | 1363.39 | 1364.78 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 164 — SELL (started 2026-04-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-23 10:15:00 | 1346.60 | 1363.39 | 1364.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-23 14:15:00 | 1344.40 | 1354.09 | 1359.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-05-04 09:15:00 | 1274.50 | 1270.73 | 1282.64 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-05-04 09:30:00 | 1278.40 | 1270.73 | 1282.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 09:15:00 | 1258.40 | 1256.42 | 1264.62 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-06 12:30:00 | 1255.00 | 1256.85 | 1262.86 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-05-06 14:15:00 | 1278.90 | 1261.86 | 1264.13 | SL hit (close>static) qty=1.00 sl=1269.40 alert=retest2 |

### Cycle 165 — BUY (started 2026-05-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-07 09:15:00 | 1288.10 | 1269.95 | 1267.59 | EMA200 above EMA400 |

### Cycle 166 — SELL (started 2026-05-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-08 12:15:00 | 1263.90 | 1270.81 | 1271.34 | EMA200 below EMA400 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2024-05-13 12:45:00 | 1119.95 | 2024-05-13 15:15:00 | 1126.65 | STOP_HIT | 1.00 | -0.60% |
| BUY | retest2 | 2024-05-24 14:30:00 | 1131.20 | 2024-05-28 14:15:00 | 1126.40 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest2 | 2024-05-28 09:15:00 | 1132.65 | 2024-05-28 14:15:00 | 1126.40 | STOP_HIT | 1.00 | -0.55% |
| BUY | retest2 | 2024-05-28 12:45:00 | 1130.70 | 2024-05-28 14:15:00 | 1126.40 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest2 | 2024-05-31 11:15:00 | 1114.30 | 2024-05-31 12:15:00 | 1122.90 | STOP_HIT | 1.00 | -0.77% |
| BUY | retest2 | 2024-06-12 09:45:00 | 1129.65 | 2024-06-13 09:15:00 | 1118.35 | STOP_HIT | 1.00 | -1.00% |
| BUY | retest2 | 2024-07-19 15:00:00 | 1249.20 | 2024-07-22 09:15:00 | 1235.65 | STOP_HIT | 1.00 | -1.08% |
| SELL | retest2 | 2024-07-25 09:15:00 | 1203.70 | 2024-07-29 09:15:00 | 1228.65 | STOP_HIT | 1.00 | -2.07% |
| SELL | retest2 | 2024-08-08 09:15:00 | 1170.25 | 2024-08-09 09:15:00 | 1176.90 | STOP_HIT | 1.00 | -0.57% |
| SELL | retest2 | 2024-08-08 12:30:00 | 1169.30 | 2024-08-09 09:15:00 | 1176.90 | STOP_HIT | 1.00 | -0.65% |
| SELL | retest2 | 2024-08-09 11:00:00 | 1170.00 | 2024-08-12 13:15:00 | 1173.90 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest2 | 2024-08-09 14:30:00 | 1171.20 | 2024-08-12 14:15:00 | 1173.45 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest2 | 2024-08-12 09:15:00 | 1161.75 | 2024-08-12 14:15:00 | 1173.45 | STOP_HIT | 1.00 | -1.01% |
| BUY | retest2 | 2024-08-20 11:15:00 | 1184.55 | 2024-08-21 09:15:00 | 1169.55 | STOP_HIT | 1.00 | -1.27% |
| BUY | retest2 | 2024-08-29 10:15:00 | 1232.10 | 2024-09-06 10:15:00 | 1221.70 | STOP_HIT | 1.00 | -0.84% |
| BUY | retest2 | 2024-08-29 12:15:00 | 1232.30 | 2024-09-06 10:15:00 | 1221.70 | STOP_HIT | 1.00 | -0.86% |
| BUY | retest2 | 2024-08-30 15:15:00 | 1232.00 | 2024-09-06 10:15:00 | 1221.70 | STOP_HIT | 1.00 | -0.84% |
| BUY | retest2 | 2024-09-02 12:30:00 | 1231.90 | 2024-09-06 10:15:00 | 1221.70 | STOP_HIT | 1.00 | -0.83% |
| BUY | retest2 | 2024-09-03 14:00:00 | 1229.75 | 2024-09-06 10:15:00 | 1221.70 | STOP_HIT | 1.00 | -0.65% |
| BUY | retest2 | 2024-09-11 10:30:00 | 1238.75 | 2024-09-20 15:15:00 | 1360.26 | TARGET_HIT | 1.00 | 9.81% |
| BUY | retest2 | 2024-09-11 14:15:00 | 1236.60 | 2024-09-20 15:15:00 | 1361.80 | TARGET_HIT | 1.00 | 10.12% |
| BUY | retest2 | 2024-09-11 15:15:00 | 1238.00 | 2024-09-27 13:15:00 | 1315.80 | STOP_HIT | 1.00 | 6.28% |
| BUY | retest2 | 2024-09-12 14:00:00 | 1243.65 | 2024-09-27 13:15:00 | 1315.80 | STOP_HIT | 1.00 | 5.80% |
| BUY | retest2 | 2024-09-25 10:30:00 | 1322.05 | 2024-09-27 13:15:00 | 1315.80 | STOP_HIT | 1.00 | -0.47% |
| BUY | retest2 | 2024-09-25 12:00:00 | 1323.00 | 2024-09-27 13:15:00 | 1315.80 | STOP_HIT | 1.00 | -0.54% |
| BUY | retest2 | 2024-09-27 09:45:00 | 1323.25 | 2024-09-27 13:15:00 | 1315.80 | STOP_HIT | 1.00 | -0.56% |
| BUY | retest2 | 2024-09-27 11:30:00 | 1325.25 | 2024-09-27 13:15:00 | 1315.80 | STOP_HIT | 1.00 | -0.71% |
| SELL | retest1 | 2024-10-03 09:15:00 | 1260.25 | 2024-10-09 09:15:00 | 1257.60 | STOP_HIT | 1.00 | 0.21% |
| SELL | retest2 | 2024-10-07 13:00:00 | 1240.35 | 2024-10-09 10:15:00 | 1259.30 | STOP_HIT | 1.00 | -1.53% |
| SELL | retest2 | 2024-10-07 14:45:00 | 1239.90 | 2024-10-09 10:15:00 | 1259.30 | STOP_HIT | 1.00 | -1.56% |
| SELL | retest2 | 2024-10-09 12:00:00 | 1243.10 | 2024-10-15 10:15:00 | 1244.75 | STOP_HIT | 1.00 | -0.13% |
| SELL | retest2 | 2024-10-09 13:15:00 | 1243.10 | 2024-10-15 10:15:00 | 1244.75 | STOP_HIT | 1.00 | -0.13% |
| SELL | retest2 | 2024-10-11 09:15:00 | 1235.00 | 2024-10-15 10:15:00 | 1244.75 | STOP_HIT | 1.00 | -0.79% |
| BUY | retest2 | 2024-10-17 09:15:00 | 1247.00 | 2024-10-17 09:15:00 | 1234.85 | STOP_HIT | 1.00 | -0.97% |
| SELL | retest2 | 2024-11-11 13:15:00 | 1268.15 | 2024-11-12 09:15:00 | 1290.10 | STOP_HIT | 1.00 | -1.73% |
| SELL | retest2 | 2024-11-11 15:15:00 | 1268.65 | 2024-11-12 09:15:00 | 1290.10 | STOP_HIT | 1.00 | -1.69% |
| SELL | retest2 | 2024-11-18 13:30:00 | 1251.00 | 2024-11-21 15:15:00 | 1255.85 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest2 | 2024-11-19 09:15:00 | 1252.00 | 2024-11-21 15:15:00 | 1255.85 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest2 | 2024-11-19 10:00:00 | 1251.15 | 2024-11-22 09:15:00 | 1277.85 | STOP_HIT | 1.00 | -2.13% |
| SELL | retest2 | 2024-11-19 10:30:00 | 1250.50 | 2024-11-22 09:15:00 | 1277.85 | STOP_HIT | 1.00 | -2.19% |
| SELL | retest2 | 2024-11-19 15:15:00 | 1244.00 | 2024-11-22 09:15:00 | 1277.85 | STOP_HIT | 1.00 | -2.72% |
| SELL | retest2 | 2024-11-21 12:45:00 | 1243.00 | 2024-11-22 09:15:00 | 1277.85 | STOP_HIT | 1.00 | -2.80% |
| BUY | retest1 | 2024-11-27 12:30:00 | 1308.25 | 2024-11-28 10:15:00 | 1293.90 | STOP_HIT | 1.00 | -1.10% |
| BUY | retest2 | 2024-12-03 10:45:00 | 1304.30 | 2024-12-13 09:15:00 | 1320.70 | STOP_HIT | 1.00 | 1.26% |
| SELL | retest2 | 2024-12-23 11:15:00 | 1293.85 | 2024-12-27 09:15:00 | 1313.95 | STOP_HIT | 1.00 | -1.55% |
| SELL | retest2 | 2024-12-24 09:15:00 | 1294.25 | 2024-12-27 09:15:00 | 1313.95 | STOP_HIT | 1.00 | -1.52% |
| SELL | retest2 | 2024-12-24 14:00:00 | 1295.10 | 2024-12-27 09:15:00 | 1313.95 | STOP_HIT | 1.00 | -1.46% |
| SELL | retest2 | 2024-12-26 10:30:00 | 1293.30 | 2024-12-27 09:15:00 | 1313.95 | STOP_HIT | 1.00 | -1.60% |
| SELL | retest2 | 2025-01-01 14:30:00 | 1285.25 | 2025-01-07 09:15:00 | 1279.50 | STOP_HIT | 1.00 | 0.45% |
| SELL | retest2 | 2025-01-02 09:45:00 | 1282.75 | 2025-01-07 09:15:00 | 1279.50 | STOP_HIT | 1.00 | 0.25% |
| SELL | retest2 | 2025-01-02 12:00:00 | 1283.70 | 2025-01-07 12:15:00 | 1285.80 | STOP_HIT | 1.00 | -0.16% |
| SELL | retest2 | 2025-01-03 09:15:00 | 1281.30 | 2025-01-07 12:15:00 | 1285.80 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest2 | 2025-01-06 11:45:00 | 1263.00 | 2025-01-07 12:15:00 | 1285.80 | STOP_HIT | 1.00 | -1.81% |
| SELL | retest2 | 2025-01-06 13:15:00 | 1263.05 | 2025-01-07 12:15:00 | 1285.80 | STOP_HIT | 1.00 | -1.80% |
| SELL | retest2 | 2025-01-10 10:15:00 | 1249.10 | 2025-01-16 11:15:00 | 1245.30 | STOP_HIT | 1.00 | 0.30% |
| SELL | retest2 | 2025-01-10 12:45:00 | 1248.20 | 2025-01-16 11:15:00 | 1245.30 | STOP_HIT | 1.00 | 0.23% |
| SELL | retest2 | 2025-01-13 09:15:00 | 1242.00 | 2025-01-16 11:15:00 | 1245.30 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest2 | 2025-01-16 09:30:00 | 1247.25 | 2025-01-16 11:15:00 | 1245.30 | STOP_HIT | 1.00 | 0.16% |
| BUY | retest2 | 2025-02-01 10:15:00 | 1255.25 | 2025-02-01 12:15:00 | 1236.40 | STOP_HIT | 1.00 | -1.50% |
| BUY | retest2 | 2025-02-03 12:00:00 | 1250.00 | 2025-02-07 11:15:00 | 1259.75 | STOP_HIT | 1.00 | 0.78% |
| SELL | retest2 | 2025-02-11 09:15:00 | 1256.50 | 2025-02-13 10:15:00 | 1261.50 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest2 | 2025-02-11 10:30:00 | 1256.15 | 2025-02-13 10:15:00 | 1261.50 | STOP_HIT | 1.00 | -0.43% |
| SELL | retest2 | 2025-02-11 11:15:00 | 1258.70 | 2025-02-13 10:15:00 | 1261.50 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest2 | 2025-02-12 13:45:00 | 1254.05 | 2025-02-13 10:15:00 | 1261.50 | STOP_HIT | 1.00 | -0.59% |
| BUY | retest2 | 2025-02-20 13:45:00 | 1251.65 | 2025-02-21 09:15:00 | 1227.00 | STOP_HIT | 1.00 | -1.97% |
| BUY | retest2 | 2025-02-20 14:30:00 | 1251.60 | 2025-02-21 09:15:00 | 1227.00 | STOP_HIT | 1.00 | -1.97% |
| SELL | retest2 | 2025-02-27 15:15:00 | 1220.00 | 2025-03-05 13:15:00 | 1216.55 | STOP_HIT | 1.00 | 0.28% |
| SELL | retest2 | 2025-02-28 10:15:00 | 1217.60 | 2025-03-05 13:15:00 | 1216.55 | STOP_HIT | 1.00 | 0.09% |
| SELL | retest2 | 2025-03-05 09:30:00 | 1220.05 | 2025-03-05 13:15:00 | 1216.55 | STOP_HIT | 1.00 | 0.29% |
| BUY | retest2 | 2025-03-07 13:15:00 | 1216.95 | 2025-03-07 13:15:00 | 1212.95 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest2 | 2025-03-26 09:15:00 | 1347.00 | 2025-03-26 12:15:00 | 1339.30 | STOP_HIT | 1.00 | -0.57% |
| SELL | retest2 | 2025-04-02 11:15:00 | 1328.65 | 2025-04-04 12:15:00 | 1336.50 | STOP_HIT | 1.00 | -0.59% |
| SELL | retest2 | 2025-04-02 12:15:00 | 1328.90 | 2025-04-04 12:15:00 | 1336.50 | STOP_HIT | 1.00 | -0.57% |
| SELL | retest2 | 2025-04-02 12:45:00 | 1327.35 | 2025-04-04 12:15:00 | 1336.50 | STOP_HIT | 1.00 | -0.69% |
| SELL | retest2 | 2025-04-02 14:45:00 | 1329.00 | 2025-04-04 12:15:00 | 1336.50 | STOP_HIT | 1.00 | -0.56% |
| SELL | retest2 | 2025-04-03 09:15:00 | 1324.60 | 2025-04-04 13:15:00 | 1333.95 | STOP_HIT | 1.00 | -0.71% |
| SELL | retest2 | 2025-04-03 10:30:00 | 1329.40 | 2025-04-04 13:15:00 | 1333.95 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest2 | 2025-04-03 13:30:00 | 1327.95 | 2025-04-04 13:15:00 | 1333.95 | STOP_HIT | 1.00 | -0.45% |
| SELL | retest2 | 2025-04-03 15:15:00 | 1328.95 | 2025-04-04 13:15:00 | 1333.95 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest2 | 2025-04-08 10:30:00 | 1289.45 | 2025-04-11 09:15:00 | 1316.15 | STOP_HIT | 1.00 | -2.07% |
| SELL | retest2 | 2025-04-08 11:15:00 | 1291.60 | 2025-04-11 09:15:00 | 1316.15 | STOP_HIT | 1.00 | -1.90% |
| SELL | retest2 | 2025-04-09 09:15:00 | 1291.55 | 2025-04-11 09:15:00 | 1316.15 | STOP_HIT | 1.00 | -1.90% |
| SELL | retest2 | 2025-04-09 09:45:00 | 1289.00 | 2025-04-11 09:15:00 | 1316.15 | STOP_HIT | 1.00 | -2.11% |
| BUY | retest2 | 2025-05-05 09:15:00 | 1438.30 | 2025-05-05 11:15:00 | 1428.40 | STOP_HIT | 1.00 | -0.69% |
| BUY | retest2 | 2025-05-27 11:15:00 | 1459.80 | 2025-05-27 13:15:00 | 1441.30 | STOP_HIT | 1.00 | -1.27% |
| SELL | retest2 | 2025-06-04 09:15:00 | 1431.20 | 2025-06-05 11:15:00 | 1456.10 | STOP_HIT | 1.00 | -1.74% |
| SELL | retest2 | 2025-06-11 14:00:00 | 1427.00 | 2025-06-16 15:15:00 | 1427.90 | STOP_HIT | 1.00 | -0.06% |
| SELL | retest2 | 2025-06-12 10:00:00 | 1427.90 | 2025-06-16 15:15:00 | 1427.90 | STOP_HIT | 1.00 | 0.00% |
| BUY | retest2 | 2025-06-25 09:30:00 | 1426.50 | 2025-07-01 11:15:00 | 1435.70 | STOP_HIT | 1.00 | 0.64% |
| BUY | retest2 | 2025-06-25 14:15:00 | 1425.10 | 2025-07-01 11:15:00 | 1435.70 | STOP_HIT | 1.00 | 0.74% |
| BUY | retest2 | 2025-06-26 09:15:00 | 1428.90 | 2025-07-01 11:15:00 | 1435.70 | STOP_HIT | 1.00 | 0.48% |
| SELL | retest2 | 2025-07-03 11:15:00 | 1431.10 | 2025-07-04 14:15:00 | 1443.20 | STOP_HIT | 1.00 | -0.85% |
| SELL | retest2 | 2025-07-03 13:45:00 | 1429.00 | 2025-07-04 14:15:00 | 1443.20 | STOP_HIT | 1.00 | -0.99% |
| BUY | retest2 | 2025-07-28 09:15:00 | 1482.90 | 2025-07-31 09:15:00 | 1466.10 | STOP_HIT | 1.00 | -1.13% |
| SELL | retest2 | 2025-08-08 15:15:00 | 1432.00 | 2025-08-18 09:15:00 | 1447.20 | STOP_HIT | 1.00 | -1.06% |
| SELL | retest2 | 2025-08-12 09:15:00 | 1427.70 | 2025-08-18 09:15:00 | 1447.20 | STOP_HIT | 1.00 | -1.37% |
| SELL | retest2 | 2025-09-01 14:15:00 | 1406.30 | 2025-09-01 14:15:00 | 1411.80 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest2 | 2025-09-11 13:15:00 | 1407.60 | 2025-09-11 15:15:00 | 1401.60 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest2 | 2025-09-18 10:15:00 | 1425.30 | 2025-09-19 09:15:00 | 1405.10 | STOP_HIT | 1.00 | -1.42% |
| BUY | retest2 | 2025-09-18 12:45:00 | 1425.20 | 2025-09-19 09:15:00 | 1405.10 | STOP_HIT | 1.00 | -1.41% |
| BUY | retest2 | 2025-10-15 09:15:00 | 1389.50 | 2025-10-20 14:15:00 | 1390.20 | STOP_HIT | 1.00 | 0.05% |
| SELL | retest2 | 2025-10-27 13:15:00 | 1375.70 | 2025-11-07 15:15:00 | 1342.60 | STOP_HIT | 1.00 | 2.41% |
| SELL | retest2 | 2025-10-28 09:15:00 | 1363.60 | 2025-11-07 15:15:00 | 1342.60 | STOP_HIT | 1.00 | 1.54% |
| BUY | retest2 | 2025-11-19 15:00:00 | 1383.30 | 2025-11-21 09:15:00 | 1368.70 | STOP_HIT | 1.00 | -1.06% |
| BUY | retest2 | 2025-11-20 13:45:00 | 1384.40 | 2025-11-21 09:15:00 | 1368.70 | STOP_HIT | 1.00 | -1.13% |
| SELL | retest2 | 2025-11-25 13:30:00 | 1361.50 | 2025-11-26 14:15:00 | 1375.50 | STOP_HIT | 1.00 | -1.03% |
| SELL | retest2 | 2025-11-25 14:15:00 | 1360.20 | 2025-11-26 14:15:00 | 1375.50 | STOP_HIT | 1.00 | -1.12% |
| SELL | retest2 | 2025-12-12 10:15:00 | 1364.40 | 2025-12-22 10:15:00 | 1363.70 | STOP_HIT | 1.00 | 0.05% |
| SELL | retest2 | 2025-12-12 13:15:00 | 1364.30 | 2025-12-22 10:15:00 | 1363.70 | STOP_HIT | 1.00 | 0.04% |
| SELL | retest2 | 2025-12-15 09:15:00 | 1361.00 | 2025-12-22 10:15:00 | 1363.70 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest2 | 2025-12-15 10:15:00 | 1364.40 | 2025-12-22 10:15:00 | 1363.70 | STOP_HIT | 1.00 | 0.05% |
| SELL | retest2 | 2025-12-16 11:30:00 | 1362.30 | 2025-12-22 10:15:00 | 1363.70 | STOP_HIT | 1.00 | -0.10% |
| SELL | retest2 | 2025-12-16 12:15:00 | 1363.10 | 2025-12-22 10:15:00 | 1363.70 | STOP_HIT | 1.00 | -0.04% |
| SELL | retest2 | 2025-12-16 13:15:00 | 1362.80 | 2025-12-22 10:15:00 | 1363.70 | STOP_HIT | 1.00 | -0.07% |
| SELL | retest2 | 2025-12-16 14:00:00 | 1362.20 | 2025-12-22 10:15:00 | 1363.70 | STOP_HIT | 1.00 | -0.11% |
| SELL | retest2 | 2025-12-17 09:15:00 | 1352.30 | 2025-12-22 10:15:00 | 1363.70 | STOP_HIT | 1.00 | -0.84% |
| SELL | retest2 | 2025-12-18 11:45:00 | 1363.50 | 2025-12-22 10:15:00 | 1363.70 | STOP_HIT | 1.00 | -0.01% |
| SELL | retest2 | 2025-12-18 13:00:00 | 1362.30 | 2025-12-22 10:15:00 | 1363.70 | STOP_HIT | 1.00 | -0.10% |
| SELL | retest2 | 2025-12-22 10:15:00 | 1362.20 | 2025-12-22 10:15:00 | 1363.70 | STOP_HIT | 1.00 | -0.11% |
| BUY | retest2 | 2025-12-23 13:00:00 | 1361.40 | 2025-12-24 15:15:00 | 1359.00 | STOP_HIT | 1.00 | -0.18% |
| BUY | retest2 | 2025-12-23 13:45:00 | 1361.40 | 2025-12-24 15:15:00 | 1359.00 | STOP_HIT | 1.00 | -0.18% |
| BUY | retest2 | 2025-12-24 12:45:00 | 1361.80 | 2025-12-24 15:15:00 | 1359.00 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest2 | 2025-12-24 13:45:00 | 1361.70 | 2025-12-24 15:15:00 | 1359.00 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest2 | 2025-12-31 10:15:00 | 1341.00 | 2026-01-02 11:15:00 | 1350.00 | STOP_HIT | 1.00 | -0.67% |
| SELL | retest2 | 2025-12-31 11:15:00 | 1340.10 | 2026-01-02 11:15:00 | 1350.00 | STOP_HIT | 1.00 | -0.74% |
| SELL | retest2 | 2025-12-31 12:15:00 | 1340.90 | 2026-01-02 11:15:00 | 1350.00 | STOP_HIT | 1.00 | -0.68% |
| SELL | retest2 | 2026-01-01 11:45:00 | 1340.70 | 2026-01-02 11:15:00 | 1350.00 | STOP_HIT | 1.00 | -0.69% |
| SELL | retest2 | 2026-01-19 09:15:00 | 1369.90 | 2026-01-27 15:15:00 | 1367.10 | STOP_HIT | 1.00 | 0.20% |
| SELL | retest2 | 2026-02-17 09:15:00 | 1394.80 | 2026-02-23 11:15:00 | 1402.50 | STOP_HIT | 1.00 | -0.55% |
| BUY | retest2 | 2026-02-26 14:15:00 | 1406.00 | 2026-02-27 09:15:00 | 1394.40 | STOP_HIT | 1.00 | -0.83% |
| SELL | retest2 | 2026-03-05 12:45:00 | 1355.70 | 2026-03-09 09:15:00 | 1287.91 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-05 15:00:00 | 1354.20 | 2026-03-09 09:15:00 | 1286.49 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-05 12:45:00 | 1355.70 | 2026-03-10 09:15:00 | 1297.40 | STOP_HIT | 0.50 | 4.30% |
| SELL | retest2 | 2026-03-05 15:00:00 | 1354.20 | 2026-03-10 09:15:00 | 1297.40 | STOP_HIT | 0.50 | 4.19% |
| BUY | retest2 | 2026-03-18 11:15:00 | 1291.70 | 2026-03-19 09:15:00 | 1261.00 | STOP_HIT | 1.00 | -2.38% |
| BUY | retest2 | 2026-03-18 12:45:00 | 1291.10 | 2026-03-19 09:15:00 | 1261.00 | STOP_HIT | 1.00 | -2.33% |
| BUY | retest2 | 2026-03-18 15:15:00 | 1292.00 | 2026-03-19 09:15:00 | 1261.00 | STOP_HIT | 1.00 | -2.40% |
| SELL | retest2 | 2026-03-20 12:15:00 | 1256.20 | 2026-03-25 09:15:00 | 1265.40 | STOP_HIT | 1.00 | -0.73% |
| SELL | retest2 | 2026-03-20 14:15:00 | 1257.90 | 2026-03-25 09:15:00 | 1265.40 | STOP_HIT | 1.00 | -0.60% |
| SELL | retest2 | 2026-04-01 10:15:00 | 1216.80 | 2026-04-06 14:15:00 | 1233.60 | STOP_HIT | 1.00 | -1.38% |
| SELL | retest2 | 2026-04-01 13:30:00 | 1210.80 | 2026-04-06 14:15:00 | 1233.60 | STOP_HIT | 1.00 | -1.88% |
| SELL | retest2 | 2026-04-02 15:15:00 | 1216.50 | 2026-04-06 14:15:00 | 1233.60 | STOP_HIT | 1.00 | -1.41% |
| BUY | retest2 | 2026-04-13 10:15:00 | 1314.50 | 2026-04-23 10:15:00 | 1346.60 | STOP_HIT | 1.00 | 2.44% |
| SELL | retest2 | 2026-05-06 12:30:00 | 1255.00 | 2026-05-06 14:15:00 | 1278.90 | STOP_HIT | 1.00 | -1.90% |
