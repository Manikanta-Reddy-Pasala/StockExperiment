# Anupam Rasayan India Ltd. (ANURAS)

## Backtest Summary

- **Window:** 2025-03-12 09:15:00 → 2026-05-08 15:15:00 (1983 bars)
- **Last close:** 1369.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 4 |
| ALERT1 | 4 |
| ALERT2 | 4 |
| ALERT2_SKIP | 2 |
| ALERT3 | 29 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 28 |
| PARTIAL | 0 |
| TARGET_HIT | 0 |
| STOP_HIT | 28 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 28 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 0 / 28
- **Target hits / Stop hits / Partials:** 0 / 28 / 0
- **Avg / median % per leg:** -2.20% / -2.15%
- **Sum % (uncompounded):** -61.64%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 16 | 0 | 0.0% | 0 | 16 | 0 | -2.45% | -39.1% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 16 | 0 | 0.0% | 0 | 16 | 0 | -2.45% | -39.1% |
| SELL (all) | 12 | 0 | 0.0% | 0 | 12 | 0 | -1.88% | -22.5% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 12 | 0 | 0.0% | 0 | 12 | 0 | -1.88% | -22.5% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 28 | 0 | 0.0% | 0 | 28 | 0 | -2.20% | -61.6% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-10-31 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-31 14:15:00 | 1072.50 | 1093.45 | 1093.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-31 15:15:00 | 1069.50 | 1093.21 | 1093.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-07 15:15:00 | 1094.40 | 1087.75 | 1090.44 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-07 15:15:00 | 1094.40 | 1087.75 | 1090.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 15:15:00 | 1094.40 | 1087.75 | 1090.44 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-10 09:15:00 | 1074.50 | 1087.75 | 1090.44 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-10 12:15:00 | 1078.30 | 1087.47 | 1090.26 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-10 13:45:00 | 1078.10 | 1087.31 | 1090.15 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-10 14:30:00 | 1076.40 | 1087.22 | 1090.09 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 15:15:00 | 1085.00 | 1087.20 | 1090.06 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-11 09:30:00 | 1069.30 | 1087.04 | 1089.97 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-11 11:15:00 | 1069.20 | 1086.88 | 1089.87 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-11 12:00:00 | 1067.20 | 1086.68 | 1089.76 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-11 15:15:00 | 1067.00 | 1086.24 | 1089.49 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 09:15:00 | 1082.10 | 1082.86 | 1087.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-17 09:30:00 | 1087.50 | 1082.86 | 1087.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 10:15:00 | 1095.10 | 1082.98 | 1087.38 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-11-17 10:15:00 | 1095.10 | 1082.98 | 1087.38 | SL hit (close>static) qty=1.00 sl=1094.40 alert=retest2 |

### Cycle 2 — BUY (started 2025-11-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-21 10:15:00 | 1238.40 | 1092.31 | 1091.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-21 12:15:00 | 1246.40 | 1095.23 | 1093.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-19 15:15:00 | 1291.10 | 1299.43 | 1251.69 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-20 09:15:00 | 1270.40 | 1299.43 | 1251.69 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 12:15:00 | 1250.40 | 1298.31 | 1252.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-20 12:30:00 | 1261.60 | 1298.31 | 1252.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 13:15:00 | 1252.60 | 1297.86 | 1252.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-20 13:45:00 | 1234.20 | 1297.86 | 1252.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 14:15:00 | 1273.80 | 1297.62 | 1252.18 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-05 12:15:00 | 1282.60 | 1261.72 | 1245.40 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-05 13:45:00 | 1283.00 | 1262.22 | 1245.81 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-16 10:45:00 | 1282.70 | 1289.26 | 1264.12 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-16 15:15:00 | 1239.60 | 1287.65 | 1263.93 | SL hit (close<static) qty=1.00 sl=1249.70 alert=retest2 |

### Cycle 3 — SELL (started 2026-03-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-23 12:15:00 | 1210.00 | 1257.40 | 1257.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 10:15:00 | 1204.40 | 1254.96 | 1256.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 12:15:00 | 1269.20 | 1252.95 | 1255.06 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 12:15:00 | 1269.20 | 1252.95 | 1255.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 12:15:00 | 1269.20 | 1252.95 | 1255.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 13:00:00 | 1269.20 | 1252.95 | 1255.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 13:15:00 | 1274.50 | 1253.17 | 1255.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 13:30:00 | 1271.90 | 1253.17 | 1255.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 13:15:00 | 1256.00 | 1253.28 | 1255.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-02 14:00:00 | 1256.00 | 1253.28 | 1255.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 14:15:00 | 1268.60 | 1253.43 | 1255.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-02 15:00:00 | 1268.60 | 1253.43 | 1255.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 15:15:00 | 1265.00 | 1253.54 | 1255.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-06 09:15:00 | 1274.00 | 1253.54 | 1255.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-06 10:15:00 | 1253.70 | 1253.71 | 1255.33 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-06 14:15:00 | 1250.00 | 1253.88 | 1255.39 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-09 12:15:00 | 1275.40 | 1253.79 | 1255.17 | SL hit (close>static) qty=1.00 sl=1271.50 alert=retest2 |

### Cycle 4 — BUY (started 2026-04-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-10 15:15:00 | 1283.50 | 1256.60 | 1256.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-15 09:15:00 | 1291.50 | 1258.45 | 1257.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-29 15:15:00 | 1291.00 | 1291.06 | 1276.77 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-30 09:30:00 | 1291.60 | 1291.18 | 1276.90 | Sideways (15m bar) within 4 candles — skip ENTRY1 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-11-10 09:15:00 | 1074.50 | 2025-11-17 10:15:00 | 1095.10 | STOP_HIT | 1.00 | -1.92% |
| SELL | retest2 | 2025-11-10 12:15:00 | 1078.30 | 2025-11-17 10:15:00 | 1095.10 | STOP_HIT | 1.00 | -1.56% |
| SELL | retest2 | 2025-11-10 13:45:00 | 1078.10 | 2025-11-17 10:15:00 | 1095.10 | STOP_HIT | 1.00 | -1.58% |
| SELL | retest2 | 2025-11-10 14:30:00 | 1076.40 | 2025-11-17 10:15:00 | 1095.10 | STOP_HIT | 1.00 | -1.74% |
| SELL | retest2 | 2025-11-11 09:30:00 | 1069.30 | 2025-11-17 10:15:00 | 1095.10 | STOP_HIT | 1.00 | -2.41% |
| SELL | retest2 | 2025-11-11 11:15:00 | 1069.20 | 2025-11-17 10:15:00 | 1095.10 | STOP_HIT | 1.00 | -2.42% |
| SELL | retest2 | 2025-11-11 12:00:00 | 1067.20 | 2025-11-17 10:15:00 | 1095.10 | STOP_HIT | 1.00 | -2.61% |
| SELL | retest2 | 2025-11-11 15:15:00 | 1067.00 | 2025-11-17 10:15:00 | 1095.10 | STOP_HIT | 1.00 | -2.63% |
| SELL | retest2 | 2025-11-17 13:30:00 | 1091.20 | 2025-11-18 13:15:00 | 1104.80 | STOP_HIT | 1.00 | -1.25% |
| SELL | retest2 | 2025-11-18 12:15:00 | 1091.70 | 2025-11-18 13:15:00 | 1104.80 | STOP_HIT | 1.00 | -1.20% |
| SELL | retest2 | 2025-11-18 13:15:00 | 1092.10 | 2025-11-18 13:15:00 | 1104.80 | STOP_HIT | 1.00 | -1.16% |
| BUY | retest2 | 2026-02-05 12:15:00 | 1282.60 | 2026-02-16 15:15:00 | 1239.60 | STOP_HIT | 1.00 | -3.35% |
| BUY | retest2 | 2026-02-05 13:45:00 | 1283.00 | 2026-02-16 15:15:00 | 1239.60 | STOP_HIT | 1.00 | -3.38% |
| BUY | retest2 | 2026-02-16 10:45:00 | 1282.70 | 2026-02-16 15:15:00 | 1239.60 | STOP_HIT | 1.00 | -3.36% |
| BUY | retest2 | 2026-02-18 10:45:00 | 1282.90 | 2026-02-26 09:15:00 | 1257.00 | STOP_HIT | 1.00 | -2.02% |
| BUY | retest2 | 2026-02-24 15:00:00 | 1270.50 | 2026-02-26 09:15:00 | 1257.00 | STOP_HIT | 1.00 | -1.06% |
| BUY | retest2 | 2026-02-25 09:15:00 | 1276.40 | 2026-02-26 09:15:00 | 1257.00 | STOP_HIT | 1.00 | -1.52% |
| BUY | retest2 | 2026-02-25 14:30:00 | 1270.40 | 2026-02-26 09:15:00 | 1257.00 | STOP_HIT | 1.00 | -1.05% |
| BUY | retest2 | 2026-02-25 15:00:00 | 1270.10 | 2026-02-26 15:15:00 | 1247.00 | STOP_HIT | 1.00 | -1.82% |
| BUY | retest2 | 2026-03-10 12:15:00 | 1263.20 | 2026-03-13 12:15:00 | 1236.00 | STOP_HIT | 1.00 | -2.15% |
| BUY | retest2 | 2026-03-10 12:45:00 | 1265.40 | 2026-03-13 12:15:00 | 1236.00 | STOP_HIT | 1.00 | -2.32% |
| BUY | retest2 | 2026-03-10 13:30:00 | 1265.40 | 2026-03-13 12:15:00 | 1236.00 | STOP_HIT | 1.00 | -2.32% |
| BUY | retest2 | 2026-03-10 14:15:00 | 1267.30 | 2026-03-13 12:15:00 | 1236.00 | STOP_HIT | 1.00 | -2.47% |
| BUY | retest2 | 2026-03-20 09:15:00 | 1267.40 | 2026-03-23 09:15:00 | 1221.50 | STOP_HIT | 1.00 | -3.62% |
| BUY | retest2 | 2026-03-20 11:15:00 | 1256.80 | 2026-03-23 09:15:00 | 1221.50 | STOP_HIT | 1.00 | -2.81% |
| BUY | retest2 | 2026-03-20 12:30:00 | 1255.90 | 2026-03-23 09:15:00 | 1221.50 | STOP_HIT | 1.00 | -2.74% |
| BUY | retest2 | 2026-03-20 13:30:00 | 1260.80 | 2026-03-23 09:15:00 | 1221.50 | STOP_HIT | 1.00 | -3.12% |
| SELL | retest2 | 2026-04-06 14:15:00 | 1250.00 | 2026-04-09 12:15:00 | 1275.40 | STOP_HIT | 1.00 | -2.03% |
