# AXISBANK (AXISBANK)

## Backtest Summary

- **Window:** 2024-04-05 09:15:00 → 2026-05-08 15:15:00 (3605 bars)
- **Last close:** 1270.00
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
| ALERT2 | 5 |
| ALERT2_SKIP | 0 |
| ALERT3 | 21 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 10 |
| PARTIAL | 0 |
| TARGET_HIT | 1 |
| STOP_HIT | 9 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 10 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 1 / 9
- **Target hits / Stop hits / Partials:** 1 / 9 / 0
- **Avg / median % per leg:** 0.03% / -0.62%
- **Sum % (uncompounded):** 0.26%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 10 | 1 | 10.0% | 1 | 9 | 0 | 0.03% | 0.3% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 10 | 1 | 10.0% | 1 | 9 | 0 | 0.03% | 0.3% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 10 | 1 | 10.0% | 1 | 9 | 0 | 0.03% | 0.3% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-07-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-22 13:15:00 | 1101.00 | 1169.99 | 1170.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-22 14:15:00 | 1097.50 | 1169.27 | 1169.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-11 10:15:00 | 1074.20 | 1073.36 | 1097.23 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-11 11:00:00 | 1074.20 | 1073.36 | 1097.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 09:15:00 | 1102.30 | 1074.25 | 1096.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-12 09:45:00 | 1103.00 | 1074.25 | 1096.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 10:15:00 | 1105.80 | 1074.57 | 1097.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-12 11:00:00 | 1105.80 | 1074.57 | 1097.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 2 — BUY (started 2025-09-29 11:15:00)

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
| Stop hit — per-position SL triggered | 2025-12-22 10:15:00 | 1228.00 | 1253.31 | 1230.39 | SL hit (close<static) qty=1.00 sl=1228.10 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-22 10:15:00 | 1228.00 | 1253.31 | 1230.39 | SL hit (close<static) qty=1.00 sl=1228.10 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-22 10:15:00 | 1228.00 | 1253.31 | 1230.39 | SL hit (close<static) qty=1.00 sl=1228.10 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-22 10:15:00 | 1228.00 | 1253.31 | 1230.39 | SL hit (close<static) qty=1.00 sl=1228.10 alert=retest2 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-22 14:30:00 | 1233.50 | 1252.43 | 1230.40 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-22 15:15:00 | 1233.80 | 1252.43 | 1230.40 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-23 14:15:00 | 1225.80 | 1250.93 | 1230.40 | SL hit (close<static) qty=1.00 sl=1229.30 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-23 14:15:00 | 1225.80 | 1250.93 | 1230.40 | SL hit (close<static) qty=1.00 sl=1229.30 alert=retest2 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-24 11:15:00 | 1234.60 | 1250.31 | 1230.39 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-24 12:15:00 | 1228.20 | 1249.91 | 1230.39 | SL hit (close<static) qty=1.00 sl=1229.30 alert=retest2 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-30 09:15:00 | 1235.20 | 1246.37 | 1230.13 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 15:15:00 | 1253.90 | 1272.88 | 1254.55 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-27 09:15:00 | 1305.30 | 1272.88 | 1254.55 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2026-01-28 09:15:00 | 1358.72 | 1276.54 | 1257.14 | Target hit (10%) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-12 09:15:00 | 1245.80 | 1335.13 | 1316.12 | SL hit (close<static) qty=1.00 sl=1248.00 alert=retest2 |

### Cycle 3 — SELL (started 2026-03-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 12:15:00 | 1221.40 | 1299.88 | 1300.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 13:15:00 | 1210.80 | 1299.00 | 1299.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-07 14:15:00 | 1250.90 | 1249.64 | 1270.19 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-07 14:45:00 | 1252.10 | 1249.64 | 1270.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 09:15:00 | 1316.00 | 1250.28 | 1270.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-08 10:00:00 | 1316.00 | 1250.28 | 1270.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 4 — BUY (started 2026-04-17 14:15:00)

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
