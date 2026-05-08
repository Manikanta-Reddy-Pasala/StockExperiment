# DRREDDY (DRREDDY)

## Backtest Summary

- **Window:** 2025-05-09 09:15:00 → 2026-05-08 15:15:00 (1731 bars)
- **Last close:** 1294.50
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 9 |
| ALERT1 | 7 |
| ALERT2 | 7 |
| ALERT2_SKIP | 5 |
| ALERT3 | 8 |
| PENDING | 15 |
| PENDING_CANCEL | 2 |
| ENTRY1 | 3 |
| ENTRY2 | 10 |
| PARTIAL | 0 |
| TARGET_HIT | 0 |
| STOP_HIT | 12 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 12 (incl. partial bookings)
- **Trades open at end:** 1
- **Winners / losers:** 0 / 12
- **Target hits / Stop hits / Partials:** 0 / 12 / 0
- **Avg / median % per leg:** -2.34% / -2.22%
- **Sum % (uncompounded):** -28.13%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 4 | 0 | 0.0% | 0 | 4 | 0 | -2.71% | -10.9% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 4 | 0 | 0.0% | 0 | 4 | 0 | -2.71% | -10.9% |
| SELL (all) | 8 | 0 | 0.0% | 0 | 8 | 0 | -2.16% | -17.3% |
| SELL @ 2nd Alert (retest1) | 2 | 0 | 0.0% | 0 | 2 | 0 | -1.86% | -3.7% |
| SELL @ 3rd Alert (retest2) | 6 | 0 | 0.0% | 0 | 6 | 0 | -2.26% | -13.6% |
| retest1 (combined) | 2 | 0 | 0.0% | 0 | 2 | 0 | -1.86% | -3.7% |
| retest2 (combined) | 10 | 0 | 0.0% | 0 | 10 | 0 | -2.44% | -24.4% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-09-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-11 10:15:00 | 1294.50 | 1263.24 | 1263.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-11 12:15:00 | 1299.50 | 1263.91 | 1263.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-25 09:15:00 | 1282.80 | 1285.73 | 1276.41 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-25 11:15:00 | 1277.00 | 1285.63 | 1276.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 11:15:00 | 1277.00 | 1285.63 | 1276.45 | EMA400 retest candle locked (from upside) |

### Cycle 2 — SELL (started 2025-10-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-08 09:15:00 | 1244.00 | 1269.37 | 1269.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-08 10:15:00 | 1239.10 | 1269.07 | 1269.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-10 11:15:00 | 1266.70 | 1266.19 | 1267.78 | EMA200 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2025-10-13 09:15:00 | 1251.50 | 1265.90 | 1267.60 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-13 10:15:00 | 1250.10 | 1265.74 | 1267.51 | SELL ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2025-10-14 09:15:00 | 1249.50 | 1265.17 | 1267.17 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-14 10:15:00 | 1240.80 | 1264.93 | 1267.04 | SELL ENTRY1 attempt 2/4 (retest1 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 09:15:00 | 1268.60 | 1259.65 | 1263.92 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-10-20 09:15:00 | 1268.60 | 1259.65 | 1263.92 | SL hit (close>ema400) qty=1.00 sl=1263.92 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-10-20 09:15:00 | 1268.60 | 1259.65 | 1263.92 | SL hit (close>ema400) qty=1.00 sl=1263.92 alert=retest1 |

### Cycle 3 — BUY (started 2025-10-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-29 12:15:00 | 1286.90 | 1267.51 | 1267.45 | EMA200 above EMA400 |

### Cycle 4 — SELL (started 2025-10-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-29 13:15:00 | 1252.00 | 1267.36 | 1267.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-30 09:15:00 | 1194.10 | 1266.42 | 1266.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-14 09:15:00 | 1239.70 | 1238.18 | 1249.84 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-17 09:15:00 | 1244.80 | 1238.69 | 1249.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 09:15:00 | 1244.80 | 1238.69 | 1249.70 | EMA400 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2025-11-18 09:15:00 | 1237.60 | 1239.06 | 1249.51 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-18 10:15:00 | 1235.00 | 1239.02 | 1249.43 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-11-19 09:15:00 | 1239.50 | 1239.18 | 1249.21 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-19 10:15:00 | 1238.30 | 1239.17 | 1249.15 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-11-24 09:15:00 | 1257.90 | 1240.61 | 1248.96 | SL hit (close>static) qty=1.00 sl=1253.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-24 09:15:00 | 1257.90 | 1240.61 | 1248.96 | SL hit (close>static) qty=1.00 sl=1253.00 alert=retest2 |
| Cross detected — sustain check pending | 2025-11-24 13:15:00 | 1233.30 | 1240.82 | 1248.90 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-24 14:15:00 | 1226.20 | 1240.67 | 1248.79 | SELL ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-11-25 10:15:00 | 1237.70 | 1240.50 | 1248.58 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-11-25 11:15:00 | 1249.70 | 1240.60 | 1248.59 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-11-25 14:15:00 | 1236.00 | 1240.62 | 1248.48 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-25 15:15:00 | 1236.00 | 1240.57 | 1248.42 | SELL ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 13:15:00 | 1249.60 | 1240.80 | 1248.34 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-11-28 13:15:00 | 1255.10 | 1241.79 | 1248.34 | SL hit (close>static) qty=1.00 sl=1253.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-28 13:15:00 | 1255.10 | 1241.79 | 1248.34 | SL hit (close>static) qty=1.00 sl=1253.00 alert=retest2 |

### Cycle 5 — BUY (started 2025-12-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-11 14:15:00 | 1273.30 | 1253.46 | 1253.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-11 15:15:00 | 1274.90 | 1253.67 | 1253.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-24 09:15:00 | 1259.50 | 1263.96 | 1259.38 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-24 09:15:00 | 1259.50 | 1263.96 | 1259.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 09:15:00 | 1259.50 | 1263.96 | 1259.38 | EMA400 retest candle locked (from upside) |

### Cycle 6 — SELL (started 2026-01-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-09 10:15:00 | 1210.40 | 1257.13 | 1257.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-12 09:15:00 | 1204.20 | 1254.40 | 1255.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-23 09:15:00 | 1243.50 | 1225.87 | 1239.27 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-23 09:15:00 | 1243.50 | 1225.87 | 1239.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 09:15:00 | 1243.50 | 1225.87 | 1239.27 | EMA400 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2026-01-28 10:15:00 | 1224.00 | 1227.53 | 1239.17 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-01-28 11:15:00 | 1226.40 | 1227.52 | 1239.10 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2026-01-28 12:15:00 | 1223.70 | 1227.48 | 1239.03 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-28 13:15:00 | 1222.00 | 1227.43 | 1238.94 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2026-02-01 12:15:00 | 1221.60 | 1225.10 | 1236.61 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-01 13:15:00 | 1199.00 | 1224.84 | 1236.42 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2026-02-05 12:15:00 | 1248.00 | 1224.13 | 1234.46 | SL hit (close>static) qty=1.00 sl=1247.20 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-05 12:15:00 | 1248.00 | 1224.13 | 1234.46 | SL hit (close>static) qty=1.00 sl=1247.20 alert=retest2 |

### Cycle 7 — BUY (started 2026-02-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-17 14:15:00 | 1286.00 | 1241.91 | 1241.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-19 10:15:00 | 1288.00 | 1245.82 | 1243.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-13 12:15:00 | 1283.30 | 1283.55 | 1268.04 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-16 10:15:00 | 1274.60 | 1283.70 | 1268.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 10:15:00 | 1274.60 | 1283.70 | 1268.50 | EMA400 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2026-03-17 13:15:00 | 1284.00 | 1282.93 | 1268.85 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-17 14:15:00 | 1284.80 | 1282.95 | 1268.93 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2026-03-18 09:15:00 | 1292.50 | 1283.04 | 1269.11 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-18 10:15:00 | 1298.40 | 1283.19 | 1269.25 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2026-03-20 09:15:00 | 1298.00 | 1283.18 | 1270.13 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-20 10:15:00 | 1294.60 | 1283.29 | 1270.25 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2026-03-23 14:15:00 | 1256.30 | 1283.29 | 1270.96 | SL hit (close<static) qty=1.00 sl=1265.10 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-23 14:15:00 | 1256.30 | 1283.29 | 1270.96 | SL hit (close<static) qty=1.00 sl=1265.10 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-23 14:15:00 | 1256.30 | 1283.29 | 1270.96 | SL hit (close<static) qty=1.00 sl=1265.10 alert=retest2 |
| Cross detected — sustain check pending | 2026-03-25 09:15:00 | 1291.50 | 1281.29 | 1270.48 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-25 10:15:00 | 1291.90 | 1281.39 | 1270.58 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-30 09:15:00 | 1260.40 | 1282.82 | 1272.01 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-03-30 09:15:00 | 1260.40 | 1282.82 | 1272.01 | SL hit (close<static) qty=1.00 sl=1265.10 alert=retest2 |

### Cycle 8 — SELL (started 2026-04-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-08 09:15:00 | 1187.90 | 1263.24 | 1263.27 | EMA200 below EMA400 |

### Cycle 9 — BUY (started 2026-04-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-28 09:15:00 | 1340.40 | 1258.95 | 1258.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-28 10:15:00 | 1359.00 | 1259.94 | 1259.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-05 12:15:00 | 1273.80 | 1274.75 | 1267.46 | EMA200 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2026-05-06 09:15:00 | 1283.00 | 1274.75 | 1267.60 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-06 10:15:00 | 1283.20 | 1274.84 | 1267.68 | BUY ENTRY1 attempt 1/4 (retest1 break sustained 60m) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2025-10-13 10:15:00 | 1250.10 | 2025-10-20 09:15:00 | 1268.60 | STOP_HIT | 1.00 | -1.48% |
| SELL | retest1 | 2025-10-14 10:15:00 | 1240.80 | 2025-10-20 09:15:00 | 1268.60 | STOP_HIT | 1.00 | -2.24% |
| SELL | retest2 | 2025-11-18 10:15:00 | 1235.00 | 2025-11-24 09:15:00 | 1257.90 | STOP_HIT | 1.00 | -1.85% |
| SELL | retest2 | 2025-11-19 10:15:00 | 1238.30 | 2025-11-24 09:15:00 | 1257.90 | STOP_HIT | 1.00 | -1.58% |
| SELL | retest2 | 2025-11-24 14:15:00 | 1226.20 | 2025-11-28 13:15:00 | 1255.10 | STOP_HIT | 1.00 | -2.36% |
| SELL | retest2 | 2025-11-25 15:15:00 | 1236.00 | 2025-11-28 13:15:00 | 1255.10 | STOP_HIT | 1.00 | -1.55% |
| SELL | retest2 | 2026-01-28 13:15:00 | 1222.00 | 2026-02-05 12:15:00 | 1248.00 | STOP_HIT | 1.00 | -2.13% |
| SELL | retest2 | 2026-02-01 13:15:00 | 1199.00 | 2026-02-05 12:15:00 | 1248.00 | STOP_HIT | 1.00 | -4.09% |
| BUY | retest2 | 2026-03-17 14:15:00 | 1284.80 | 2026-03-23 14:15:00 | 1256.30 | STOP_HIT | 1.00 | -2.22% |
| BUY | retest2 | 2026-03-18 10:15:00 | 1298.40 | 2026-03-23 14:15:00 | 1256.30 | STOP_HIT | 1.00 | -3.24% |
| BUY | retest2 | 2026-03-20 10:15:00 | 1294.60 | 2026-03-23 14:15:00 | 1256.30 | STOP_HIT | 1.00 | -2.96% |
| BUY | retest2 | 2026-03-25 10:15:00 | 1291.90 | 2026-03-30 09:15:00 | 1260.40 | STOP_HIT | 1.00 | -2.44% |
