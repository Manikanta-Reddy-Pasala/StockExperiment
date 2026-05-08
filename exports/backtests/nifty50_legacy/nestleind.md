# NESTLEIND (NESTLEIND)

## Backtest Summary

- **Window:** 2023-06-08 09:15:00 → 2026-05-08 15:15:00 (4996 bars)
- **Last close:** 1482.40
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 30%)
- **Partial:** 50% qty booked @ 15% (ENTRY1) / 15% (ENTRY2), trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 11 |
| ALERT1 | 11 |
| ALERT2 | 10 |
| ALERT2_SKIP | 6 |
| ALERT3 | 17 |
| PENDING | 64 |
| PENDING_CANCEL | 17 |
| ENTRY1 | 8 |
| ENTRY2 | 39 |
| PARTIAL | 0 |
| TARGET_HIT | 0 |
| STOP_HIT | 47 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 47 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 5 / 42
- **Target hits / Stop hits / Partials:** 0 / 47 / 0
- **Avg / median % per leg:** -1.58% / -1.96%
- **Sum % (uncompounded):** -74.10%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 16 | 5 | 31.2% | 0 | 16 | 0 | -0.70% | -11.1% |
| BUY @ 2nd Alert (retest1) | 5 | 3 | 60.0% | 0 | 5 | 0 | 0.58% | 2.9% |
| BUY @ 3rd Alert (retest2) | 11 | 2 | 18.2% | 0 | 11 | 0 | -1.28% | -14.0% |
| SELL (all) | 31 | 0 | 0.0% | 0 | 31 | 0 | -2.03% | -63.0% |
| SELL @ 2nd Alert (retest1) | 3 | 0 | 0.0% | 0 | 3 | 0 | -3.03% | -9.1% |
| SELL @ 3rd Alert (retest2) | 28 | 0 | 0.0% | 0 | 28 | 0 | -1.92% | -53.9% |
| retest1 (combined) | 8 | 3 | 37.5% | 0 | 8 | 0 | -0.77% | -6.2% |
| retest2 (combined) | 39 | 2 | 5.1% | 0 | 39 | 0 | -1.74% | -67.9% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-09-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-26 15:15:00 | 1151.50 | 1116.90 | 1116.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-11 12:15:00 | 1155.00 | 1128.30 | 1123.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-28 13:15:00 | 1195.36 | 1198.22 | 1176.99 | EMA200 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2023-11-28 14:15:00 | 1211.23 | 1198.34 | 1177.16 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-28 15:15:00 | 1210.93 | 1198.47 | 1177.33 | BUY ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2023-11-29 14:15:00 | 1203.93 | 1198.90 | 1178.18 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-29 15:15:00 | 1203.90 | 1198.95 | 1178.30 | BUY ENTRY1 attempt 2/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2023-11-30 12:15:00 | 1202.80 | 1199.05 | 1178.76 | ENTRY1 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2023-11-30 13:15:00 | 1200.25 | 1199.06 | 1178.87 | ENTRY1 sustain failed after 60m |
| Cross detected — sustain check pending | 2023-11-30 14:15:00 | 1213.50 | 1199.21 | 1179.04 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-30 15:15:00 | 1211.81 | 1199.33 | 1179.21 | BUY ENTRY1 attempt 3/4 (retest1 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-18 09:15:00 | 1258.80 | 1278.70 | 1250.18 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-01-23 09:15:00 | 1240.07 | 1275.55 | 1250.48 | SL hit (close<ema400) qty=1.00 sl=1250.48 alert=retest1 |
| Stop hit — per-position SL triggered | 2024-01-23 09:15:00 | 1240.07 | 1275.55 | 1250.48 | SL hit (close<ema400) qty=1.00 sl=1250.48 alert=retest1 |
| Stop hit — per-position SL triggered | 2024-01-23 09:15:00 | 1240.07 | 1275.55 | 1250.48 | SL hit (close<ema400) qty=1.00 sl=1250.48 alert=retest1 |
| Cross detected — sustain check pending | 2024-02-22 12:15:00 | 1275.00 | 1250.52 | 1246.04 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-22 13:15:00 | 1275.95 | 1250.77 | 1246.19 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2024-03-06 09:15:00 | 1275.03 | 1268.33 | 1257.43 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2024-03-06 10:15:00 | 1260.43 | 1268.25 | 1257.45 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2024-03-07 12:15:00 | 1279.93 | 1268.20 | 1257.90 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-07 13:15:00 | 1276.93 | 1268.29 | 1257.99 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2024-03-19 10:15:00 | 1248.62 | 1278.03 | 1265.59 | SL hit (close<static) qty=1.00 sl=1249.55 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-03-19 10:15:00 | 1248.62 | 1278.03 | 1265.59 | SL hit (close<static) qty=1.00 sl=1249.55 alert=retest2 |
| Cross detected — sustain check pending | 2024-03-20 14:15:00 | 1276.30 | 1276.16 | 1265.29 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-20 15:15:00 | 1276.12 | 1276.16 | 1265.34 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2024-03-21 14:15:00 | 1276.78 | 1275.98 | 1265.57 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2024-03-21 15:15:00 | 1273.00 | 1275.95 | 1265.61 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2024-03-22 09:15:00 | 1280.68 | 1276.00 | 1265.68 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-22 10:15:00 | 1275.57 | 1276.00 | 1265.73 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-04 09:15:00 | 1266.15 | 1282.31 | 1271.45 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-04-08 13:15:00 | 1249.35 | 1279.74 | 1271.06 | SL hit (close<static) qty=1.00 sl=1249.55 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-04-08 13:15:00 | 1249.35 | 1279.74 | 1271.06 | SL hit (close<static) qty=1.00 sl=1249.55 alert=retest2 |
| Cross detected — sustain check pending | 2024-04-15 12:15:00 | 1283.75 | 1275.92 | 1270.07 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-15 13:15:00 | 1283.50 | 1276.00 | 1270.14 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2024-04-16 09:15:00 | 1282.60 | 1276.07 | 1270.27 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-16 10:15:00 | 1284.75 | 1276.16 | 1270.34 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2024-04-18 09:15:00 | 1263.10 | 1276.06 | 1270.46 | SL hit (close<static) qty=1.00 sl=1265.50 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-04-18 09:15:00 | 1263.10 | 1276.06 | 1270.46 | SL hit (close<static) qty=1.00 sl=1265.50 alert=retest2 |

### Cycle 2 — SELL (started 2024-04-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-23 15:15:00 | 1249.38 | 1265.48 | 1265.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-26 14:15:00 | 1240.80 | 1264.36 | 1264.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-30 12:15:00 | 1265.00 | 1263.52 | 1264.43 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-30 12:15:00 | 1265.00 | 1263.52 | 1264.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-30 12:15:00 | 1265.00 | 1263.52 | 1264.43 | EMA400 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2024-04-30 14:15:00 | 1253.70 | 1263.40 | 1264.36 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-30 15:15:00 | 1253.70 | 1263.30 | 1264.31 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2024-05-02 14:15:00 | 1255.50 | 1263.25 | 1264.26 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-02 15:15:00 | 1255.50 | 1263.17 | 1264.21 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2024-05-08 14:15:00 | 1268.05 | 1258.84 | 1261.71 | SL hit (close>static) qty=1.00 sl=1267.50 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-05-08 14:15:00 | 1268.05 | 1258.84 | 1261.71 | SL hit (close>static) qty=1.00 sl=1267.50 alert=retest2 |
| Cross detected — sustain check pending | 2024-05-09 09:15:00 | 1257.57 | 1258.94 | 1261.73 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-09 10:15:00 | 1258.57 | 1258.93 | 1261.71 | SELL ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2024-05-09 12:15:00 | 1254.47 | 1258.91 | 1261.67 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-09 13:15:00 | 1256.03 | 1258.88 | 1261.65 | SELL ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-10 09:15:00 | 1270.53 | 1258.96 | 1261.64 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-05-10 09:15:00 | 1270.53 | 1258.96 | 1261.64 | SL hit (close>static) qty=1.00 sl=1267.50 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-05-10 09:15:00 | 1270.53 | 1258.96 | 1261.64 | SL hit (close>static) qty=1.00 sl=1267.50 alert=retest2 |
| Cross detected — sustain check pending | 2024-05-14 09:15:00 | 1242.97 | 1259.14 | 1261.56 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-14 10:15:00 | 1242.32 | 1258.97 | 1261.47 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2024-06-05 12:15:00 | 1251.80 | 1233.63 | 1245.09 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-05 13:15:00 | 1250.72 | 1233.80 | 1245.12 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2024-06-06 09:15:00 | 1233.00 | 1234.22 | 1245.16 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-06 10:15:00 | 1235.43 | 1234.23 | 1245.12 | SELL ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2024-06-10 11:15:00 | 1277.22 | 1236.27 | 1245.37 | SL hit (close>static) qty=1.00 sl=1274.50 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-06-10 11:15:00 | 1277.22 | 1236.27 | 1245.37 | SL hit (close>static) qty=1.00 sl=1274.50 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-06-10 11:15:00 | 1277.22 | 1236.27 | 1245.37 | SL hit (close>static) qty=1.00 sl=1274.50 alert=retest2 |
| Cross detected — sustain check pending | 2024-06-21 10:15:00 | 1252.00 | 1251.01 | 1251.62 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2024-06-21 11:15:00 | 1257.07 | 1251.07 | 1251.65 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2024-06-21 14:15:00 | 1246.32 | 1251.14 | 1251.67 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-21 15:15:00 | 1245.70 | 1251.08 | 1251.64 | SELL ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-24 09:15:00 | 1258.47 | 1251.16 | 1251.67 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-06-25 12:15:00 | 1255.03 | 1252.19 | 1252.18 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 3 — BUY (started 2024-06-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-25 12:15:00 | 1255.03 | 1252.19 | 1252.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-25 14:15:00 | 1260.43 | 1252.31 | 1252.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-24 09:15:00 | 1264.50 | 1282.87 | 1271.79 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-24 09:15:00 | 1264.50 | 1282.87 | 1271.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-24 09:15:00 | 1264.50 | 1282.87 | 1271.79 | EMA400 retest candle locked (from upside) |

### Cycle 4 — SELL (started 2024-08-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-07 10:15:00 | 1257.65 | 1263.71 | 1263.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-08 09:15:00 | 1251.50 | 1263.41 | 1263.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-16 14:15:00 | 1263.75 | 1256.84 | 1259.94 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-16 14:15:00 | 1263.75 | 1256.84 | 1259.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-16 14:15:00 | 1263.75 | 1256.84 | 1259.94 | EMA400 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2024-08-19 14:15:00 | 1250.72 | 1256.92 | 1259.88 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2024-08-19 15:15:00 | 1252.53 | 1256.88 | 1259.84 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2024-08-20 13:15:00 | 1252.00 | 1256.86 | 1259.76 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2024-08-20 14:15:00 | 1257.80 | 1256.86 | 1259.75 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2024-08-28 10:15:00 | 1247.55 | 1260.04 | 1261.04 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-28 11:15:00 | 1251.30 | 1259.96 | 1260.99 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2024-08-28 13:15:00 | 1250.00 | 1259.79 | 1260.89 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-28 14:15:00 | 1245.57 | 1259.65 | 1260.82 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2024-08-29 13:15:00 | 1250.43 | 1259.17 | 1260.54 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-29 14:15:00 | 1252.47 | 1259.10 | 1260.50 | SELL ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2024-08-30 14:15:00 | 1248.57 | 1258.66 | 1260.22 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2024-08-30 15:15:00 | 1253.00 | 1258.60 | 1260.19 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2024-09-02 11:15:00 | 1252.45 | 1258.44 | 1260.08 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2024-09-02 12:15:00 | 1255.43 | 1258.41 | 1260.06 | ENTRY2 sustain failed after 60m |
| Stop hit — per-position SL triggered | 2024-09-03 09:15:00 | 1264.50 | 1258.36 | 1260.00 | SL hit (close>static) qty=1.00 sl=1264.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-09-03 09:15:00 | 1264.50 | 1258.36 | 1260.00 | SL hit (close>static) qty=1.00 sl=1264.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-09-03 09:15:00 | 1264.50 | 1258.36 | 1260.00 | SL hit (close>static) qty=1.00 sl=1264.00 alert=retest2 |
| Cross detected — sustain check pending | 2024-09-05 11:15:00 | 1252.25 | 1259.30 | 1260.38 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2024-09-05 12:15:00 | 1253.70 | 1259.25 | 1260.34 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2024-09-05 13:15:00 | 1250.00 | 1259.16 | 1260.29 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2024-09-05 14:15:00 | 1252.53 | 1259.09 | 1260.25 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2024-09-05 15:15:00 | 1252.45 | 1259.02 | 1260.21 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-06 09:15:00 | 1245.10 | 1258.89 | 1260.14 | SELL ENTRY2 attempt 4/4 (retest2 break sustained 1080m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-09 15:15:00 | 1258.50 | 1258.24 | 1259.72 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-09-10 10:15:00 | 1268.03 | 1258.38 | 1259.78 | SL hit (close>static) qty=1.00 sl=1264.00 alert=retest2 |
| Cross detected — sustain check pending | 2024-09-12 11:15:00 | 1253.00 | 1259.39 | 1260.21 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-12 12:15:00 | 1249.57 | 1259.29 | 1260.16 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2024-09-12 14:15:00 | 1262.35 | 1259.28 | 1260.14 | SL hit (close>static) qty=1.00 sl=1259.95 alert=retest2 |

### Cycle 5 — BUY (started 2024-09-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-17 09:15:00 | 1280.75 | 1260.98 | 1260.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-17 11:15:00 | 1289.22 | 1261.50 | 1261.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-04 13:15:00 | 1307.25 | 1308.17 | 1289.23 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-07 14:15:00 | 1287.40 | 1307.42 | 1289.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-07 14:15:00 | 1287.40 | 1307.42 | 1289.59 | EMA400 retest candle locked (from upside) |

### Cycle 6 — SELL (started 2024-10-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-18 10:15:00 | 1171.00 | 1277.63 | 1277.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-23 13:15:00 | 1167.45 | 1256.52 | 1266.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-03 10:15:00 | 1113.50 | 1106.49 | 1137.76 | EMA200 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2025-01-06 09:15:00 | 1101.50 | 1106.86 | 1137.02 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-06 10:15:00 | 1095.38 | 1106.75 | 1136.81 | SELL ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2025-01-07 10:15:00 | 1103.05 | 1106.07 | 1135.43 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-07 11:15:00 | 1102.90 | 1106.04 | 1135.27 | SELL ENTRY1 attempt 2/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2025-01-08 09:15:00 | 1101.97 | 1105.96 | 1134.51 | ENTRY1 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-01-08 10:15:00 | 1104.75 | 1105.95 | 1134.36 | ENTRY1 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-01-08 11:15:00 | 1102.62 | 1105.91 | 1134.20 | ENTRY1 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-01-08 12:15:00 | 1104.43 | 1105.90 | 1134.05 | ENTRY1 sustain failed after 60m |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-09 10:15:00 | 1128.22 | 1106.31 | 1133.56 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-01-10 10:15:00 | 1134.55 | 1107.91 | 1133.44 | SL hit (close>ema400) qty=1.00 sl=1133.44 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-01-10 10:15:00 | 1134.55 | 1107.91 | 1133.44 | SL hit (close>ema400) qty=1.00 sl=1133.44 alert=retest1 |
| Cross detected — sustain check pending | 2025-01-14 09:15:00 | 1105.85 | 1109.52 | 1132.67 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-14 10:15:00 | 1107.53 | 1109.50 | 1132.55 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-01-14 12:15:00 | 1110.82 | 1109.55 | 1132.34 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-14 13:15:00 | 1107.53 | 1109.53 | 1132.22 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-01-21 14:15:00 | 1097.35 | 1106.98 | 1127.05 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-21 15:15:00 | 1100.00 | 1106.91 | 1126.91 | SELL ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-01-27 10:15:00 | 1106.28 | 1106.22 | 1124.37 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-27 11:15:00 | 1104.07 | 1106.20 | 1124.27 | SELL ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-31 09:15:00 | 1124.45 | 1102.49 | 1120.04 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-01-31 10:15:00 | 1137.50 | 1102.83 | 1120.13 | SL hit (close>static) qty=1.00 sl=1135.82 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-01-31 10:15:00 | 1137.50 | 1102.83 | 1120.13 | SL hit (close>static) qty=1.00 sl=1135.82 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-01-31 10:15:00 | 1137.50 | 1102.83 | 1120.13 | SL hit (close>static) qty=1.00 sl=1135.82 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-01-31 10:15:00 | 1137.50 | 1102.83 | 1120.13 | SL hit (close>static) qty=1.00 sl=1135.82 alert=retest2 |
| Cross detected — sustain check pending | 2025-02-10 12:15:00 | 1108.00 | 1113.27 | 1122.76 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-10 13:15:00 | 1104.07 | 1113.18 | 1122.66 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-02-11 10:15:00 | 1106.62 | 1113.04 | 1122.40 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-11 11:15:00 | 1104.43 | 1112.95 | 1122.31 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-02-17 15:15:00 | 1108.25 | 1108.59 | 1118.56 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-18 09:15:00 | 1104.07 | 1108.55 | 1118.48 | SELL ENTRY2 attempt 3/4 (retest2 break sustained 1080m) |
| Cross detected — sustain check pending | 2025-02-19 09:15:00 | 1108.00 | 1108.49 | 1118.11 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-02-19 10:15:00 | 1108.93 | 1108.49 | 1118.07 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-02-19 11:15:00 | 1106.12 | 1108.47 | 1118.01 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-19 12:15:00 | 1104.00 | 1108.43 | 1117.94 | SELL ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-24 11:15:00 | 1116.50 | 1107.74 | 1116.66 | EMA400 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2025-02-24 13:15:00 | 1110.05 | 1107.83 | 1116.62 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-24 14:15:00 | 1110.03 | 1107.85 | 1116.59 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-02-25 10:15:00 | 1123.60 | 1108.13 | 1116.60 | SL hit (close>static) qty=1.00 sl=1118.25 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-02-27 14:15:00 | 1139.50 | 1110.42 | 1117.31 | SL hit (close>static) qty=1.00 sl=1134.90 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-02-27 14:15:00 | 1139.50 | 1110.42 | 1117.31 | SL hit (close>static) qty=1.00 sl=1134.90 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-02-27 14:15:00 | 1139.50 | 1110.42 | 1117.31 | SL hit (close>static) qty=1.00 sl=1134.90 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-02-27 14:15:00 | 1139.50 | 1110.42 | 1117.31 | SL hit (close>static) qty=1.00 sl=1134.90 alert=retest2 |
| Cross detected — sustain check pending | 2025-02-28 12:15:00 | 1110.53 | 1111.09 | 1117.49 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-28 13:15:00 | 1102.47 | 1111.01 | 1117.41 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-03-07 10:15:00 | 1118.57 | 1105.23 | 1113.29 | SL hit (close>static) qty=1.00 sl=1118.25 alert=retest2 |
| Cross detected — sustain check pending | 2025-03-07 15:15:00 | 1112.50 | 1105.80 | 1113.38 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-03-10 09:15:00 | 1131.90 | 1106.06 | 1113.48 | ENTRY2 sustain failed after 3960m |
| Cross detected — sustain check pending | 2025-03-12 09:15:00 | 1111.28 | 1108.82 | 1114.41 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-12 10:15:00 | 1109.03 | 1108.82 | 1114.38 | SELL ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-03-21 09:15:00 | 1121.95 | 1105.15 | 1111.29 | SL hit (close>static) qty=1.00 sl=1118.25 alert=retest2 |
| Cross detected — sustain check pending | 2025-04-01 12:15:00 | 1110.40 | 1113.63 | 1114.80 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-01 13:15:00 | 1110.60 | 1113.60 | 1114.78 | SELL ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-01 14:15:00 | 1117.25 | 1113.64 | 1114.79 | EMA400 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2025-04-02 09:15:00 | 1089.38 | 1113.42 | 1114.67 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-02 10:15:00 | 1097.47 | 1113.26 | 1114.59 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-04-03 13:15:00 | 1119.00 | 1112.77 | 1114.26 | SL hit (close>static) qty=1.00 sl=1118.25 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-04-03 13:15:00 | 1119.00 | 1112.77 | 1114.26 | SL hit (close>static) qty=1.00 sl=1118.40 alert=retest2 |
| Cross detected — sustain check pending | 2025-04-07 09:15:00 | 1101.90 | 1113.92 | 1114.78 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-07 10:15:00 | 1100.80 | 1113.79 | 1114.71 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-04-07 12:15:00 | 1108.40 | 1113.73 | 1114.68 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-04-07 13:15:00 | 1118.53 | 1113.78 | 1114.70 | ENTRY2 sustain failed after 60m |
| Stop hit — per-position SL triggered | 2025-04-07 13:15:00 | 1118.53 | 1113.78 | 1114.70 | SL hit (close>static) qty=1.00 sl=1118.40 alert=retest2 |

### Cycle 7 — BUY (started 2025-04-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-09 09:15:00 | 1151.40 | 1115.71 | 1115.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-09 10:15:00 | 1169.47 | 1116.24 | 1115.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-02 12:15:00 | 1165.00 | 1165.30 | 1145.99 | EMA200 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2025-05-05 10:15:00 | 1173.70 | 1165.55 | 1146.59 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-05 11:15:00 | 1174.05 | 1165.64 | 1146.73 | BUY ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2025-05-06 09:15:00 | 1180.00 | 1165.85 | 1147.31 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-06 10:15:00 | 1181.25 | 1166.01 | 1147.48 | BUY ENTRY1 attempt 2/4 (retest1 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-09 09:15:00 | 1149.05 | 1166.50 | 1149.51 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-05-09 09:15:00 | 1149.05 | 1166.50 | 1149.51 | SL hit (close<ema400) qty=1.00 sl=1149.51 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-05-09 09:15:00 | 1149.05 | 1166.50 | 1149.51 | SL hit (close<ema400) qty=1.00 sl=1149.51 alert=retest1 |
| Cross detected — sustain check pending | 2025-05-09 14:15:00 | 1163.05 | 1166.04 | 1149.70 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-05-09 15:15:00 | 1160.50 | 1165.98 | 1149.75 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-05-12 09:15:00 | 1182.40 | 1166.15 | 1149.91 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-12 10:15:00 | 1179.50 | 1166.28 | 1150.06 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-06-20 09:15:00 | 1169.50 | 1190.61 | 1179.34 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-20 10:15:00 | 1170.90 | 1190.42 | 1179.29 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-07-24 12:15:00 | 1170.85 | 1209.30 | 1197.81 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-24 13:15:00 | 1169.35 | 1208.91 | 1197.67 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-07-25 09:15:00 | 1143.00 | 1207.31 | 1197.03 | SL hit (close<static) qty=1.00 sl=1145.85 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-25 09:15:00 | 1143.00 | 1207.31 | 1197.03 | SL hit (close<static) qty=1.00 sl=1145.85 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-25 09:15:00 | 1143.00 | 1207.31 | 1197.03 | SL hit (close<static) qty=1.00 sl=1145.85 alert=retest2 |

### Cycle 8 — SELL (started 2025-07-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-30 14:15:00 | 1115.85 | 1187.84 | 1187.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-07 10:15:00 | 1111.75 | 1169.13 | 1177.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-18 09:15:00 | 1161.60 | 1146.26 | 1163.57 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-18 09:15:00 | 1161.60 | 1146.26 | 1163.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 09:15:00 | 1161.60 | 1146.26 | 1163.57 | EMA400 retest candle locked (from downside) |

### Cycle 9 — BUY (started 2025-09-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-10 09:15:00 | 1206.00 | 1171.03 | 1170.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-10 11:15:00 | 1216.10 | 1171.86 | 1171.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-22 14:15:00 | 1186.60 | 1187.17 | 1180.41 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-23 09:15:00 | 1171.90 | 1187.04 | 1180.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 09:15:00 | 1171.90 | 1187.04 | 1180.41 | EMA400 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2025-10-10 09:15:00 | 1191.40 | 1178.03 | 1177.04 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-10 10:15:00 | 1191.80 | 1178.17 | 1177.12 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-10-15 09:15:00 | 1197.90 | 1179.74 | 1178.06 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-15 10:15:00 | 1209.90 | 1180.04 | 1178.22 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2026-03-11 14:15:00 | 1233.70 | 1278.41 | 1278.50 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-11 14:15:00 | 1233.70 | 1278.41 | 1278.50 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 10 — SELL (started 2026-03-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-11 14:15:00 | 1233.70 | 1278.41 | 1278.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-12 09:15:00 | 1221.20 | 1277.40 | 1277.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-07 15:15:00 | 1222.80 | 1222.46 | 1243.07 | EMA200 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2026-04-08 11:15:00 | 1210.10 | 1222.40 | 1242.73 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-08 12:15:00 | 1216.70 | 1222.35 | 1242.60 | SELL ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2026-04-09 13:15:00 | 1218.00 | 1221.90 | 1241.58 | ENTRY1 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-04-09 14:15:00 | 1229.50 | 1221.98 | 1241.52 | ENTRY1 sustain failed after 60m |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-10 13:15:00 | 1240.90 | 1222.73 | 1241.32 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-04-10 14:15:00 | 1248.70 | 1222.98 | 1241.36 | SL hit (close>ema400) qty=1.00 sl=1241.36 alert=retest1 |
| Cross detected — sustain check pending | 2026-04-13 09:15:00 | 1223.70 | 1223.25 | 1241.31 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-13 10:15:00 | 1228.30 | 1223.30 | 1241.25 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2026-04-15 09:15:00 | 1244.80 | 1223.66 | 1240.90 | SL hit (close>static) qty=1.00 sl=1241.80 alert=retest2 |

### Cycle 11 — BUY (started 2026-04-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-23 11:15:00 | 1410.50 | 1255.11 | 1254.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-23 13:15:00 | 1412.80 | 1258.18 | 1256.01 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-11-28 15:15:00 | 1210.93 | 2024-01-23 09:15:00 | 1240.07 | STOP_HIT | 1.00 | 2.41% |
| BUY | retest1 | 2023-11-29 15:15:00 | 1203.90 | 2024-01-23 09:15:00 | 1240.07 | STOP_HIT | 1.00 | 3.00% |
| BUY | retest1 | 2023-11-30 15:15:00 | 1211.81 | 2024-01-23 09:15:00 | 1240.07 | STOP_HIT | 1.00 | 2.33% |
| BUY | retest2 | 2024-02-22 13:15:00 | 1275.95 | 2024-03-19 10:15:00 | 1248.62 | STOP_HIT | 1.00 | -2.14% |
| BUY | retest2 | 2024-03-07 13:15:00 | 1276.93 | 2024-03-19 10:15:00 | 1248.62 | STOP_HIT | 1.00 | -2.22% |
| BUY | retest2 | 2024-03-20 15:15:00 | 1276.12 | 2024-04-08 13:15:00 | 1249.35 | STOP_HIT | 1.00 | -2.10% |
| BUY | retest2 | 2024-03-22 10:15:00 | 1275.57 | 2024-04-08 13:15:00 | 1249.35 | STOP_HIT | 1.00 | -2.06% |
| BUY | retest2 | 2024-04-15 13:15:00 | 1283.50 | 2024-04-18 09:15:00 | 1263.10 | STOP_HIT | 1.00 | -1.59% |
| BUY | retest2 | 2024-04-16 10:15:00 | 1284.75 | 2024-04-18 09:15:00 | 1263.10 | STOP_HIT | 1.00 | -1.69% |
| SELL | retest2 | 2024-04-30 15:15:00 | 1253.70 | 2024-05-08 14:15:00 | 1268.05 | STOP_HIT | 1.00 | -1.14% |
| SELL | retest2 | 2024-05-02 15:15:00 | 1255.50 | 2024-05-08 14:15:00 | 1268.05 | STOP_HIT | 1.00 | -1.00% |
| SELL | retest2 | 2024-05-09 10:15:00 | 1258.57 | 2024-05-10 09:15:00 | 1270.53 | STOP_HIT | 1.00 | -0.95% |
| SELL | retest2 | 2024-05-09 13:15:00 | 1256.03 | 2024-05-10 09:15:00 | 1270.53 | STOP_HIT | 1.00 | -1.15% |
| SELL | retest2 | 2024-05-14 10:15:00 | 1242.32 | 2024-06-10 11:15:00 | 1277.22 | STOP_HIT | 1.00 | -2.81% |
| SELL | retest2 | 2024-06-05 13:15:00 | 1250.72 | 2024-06-10 11:15:00 | 1277.22 | STOP_HIT | 1.00 | -2.12% |
| SELL | retest2 | 2024-06-06 10:15:00 | 1235.43 | 2024-06-10 11:15:00 | 1277.22 | STOP_HIT | 1.00 | -3.38% |
| SELL | retest2 | 2024-06-21 15:15:00 | 1245.70 | 2024-06-25 12:15:00 | 1255.03 | STOP_HIT | 1.00 | -0.75% |
| SELL | retest2 | 2024-08-28 11:15:00 | 1251.30 | 2024-09-03 09:15:00 | 1264.50 | STOP_HIT | 1.00 | -1.05% |
| SELL | retest2 | 2024-08-28 14:15:00 | 1245.57 | 2024-09-03 09:15:00 | 1264.50 | STOP_HIT | 1.00 | -1.52% |
| SELL | retest2 | 2024-08-29 14:15:00 | 1252.47 | 2024-09-03 09:15:00 | 1264.50 | STOP_HIT | 1.00 | -0.96% |
| SELL | retest2 | 2024-09-06 09:15:00 | 1245.10 | 2024-09-10 10:15:00 | 1268.03 | STOP_HIT | 1.00 | -1.84% |
| SELL | retest2 | 2024-09-12 12:15:00 | 1249.57 | 2024-09-12 14:15:00 | 1262.35 | STOP_HIT | 1.00 | -1.02% |
| SELL | retest1 | 2025-01-06 10:15:00 | 1095.38 | 2025-01-10 10:15:00 | 1134.55 | STOP_HIT | 1.00 | -3.58% |
| SELL | retest1 | 2025-01-07 11:15:00 | 1102.90 | 2025-01-10 10:15:00 | 1134.55 | STOP_HIT | 1.00 | -2.87% |
| SELL | retest2 | 2025-01-14 10:15:00 | 1107.53 | 2025-01-31 10:15:00 | 1137.50 | STOP_HIT | 1.00 | -2.71% |
| SELL | retest2 | 2025-01-14 13:15:00 | 1107.53 | 2025-01-31 10:15:00 | 1137.50 | STOP_HIT | 1.00 | -2.71% |
| SELL | retest2 | 2025-01-21 15:15:00 | 1100.00 | 2025-01-31 10:15:00 | 1137.50 | STOP_HIT | 1.00 | -3.41% |
| SELL | retest2 | 2025-01-27 11:15:00 | 1104.07 | 2025-01-31 10:15:00 | 1137.50 | STOP_HIT | 1.00 | -3.03% |
| SELL | retest2 | 2025-02-10 13:15:00 | 1104.07 | 2025-02-25 10:15:00 | 1123.60 | STOP_HIT | 1.00 | -1.77% |
| SELL | retest2 | 2025-02-11 11:15:00 | 1104.43 | 2025-02-27 14:15:00 | 1139.50 | STOP_HIT | 1.00 | -3.18% |
| SELL | retest2 | 2025-02-18 09:15:00 | 1104.07 | 2025-02-27 14:15:00 | 1139.50 | STOP_HIT | 1.00 | -3.21% |
| SELL | retest2 | 2025-02-19 12:15:00 | 1104.00 | 2025-02-27 14:15:00 | 1139.50 | STOP_HIT | 1.00 | -3.22% |
| SELL | retest2 | 2025-02-24 14:15:00 | 1110.03 | 2025-02-27 14:15:00 | 1139.50 | STOP_HIT | 1.00 | -2.66% |
| SELL | retest2 | 2025-02-28 13:15:00 | 1102.47 | 2025-03-07 10:15:00 | 1118.57 | STOP_HIT | 1.00 | -1.46% |
| SELL | retest2 | 2025-03-12 10:15:00 | 1109.03 | 2025-03-21 09:15:00 | 1121.95 | STOP_HIT | 1.00 | -1.17% |
| SELL | retest2 | 2025-04-01 13:15:00 | 1110.60 | 2025-04-03 13:15:00 | 1119.00 | STOP_HIT | 1.00 | -0.76% |
| SELL | retest2 | 2025-04-02 10:15:00 | 1097.47 | 2025-04-03 13:15:00 | 1119.00 | STOP_HIT | 1.00 | -1.96% |
| SELL | retest2 | 2025-04-07 10:15:00 | 1100.80 | 2025-04-07 13:15:00 | 1118.53 | STOP_HIT | 1.00 | -1.61% |
| BUY | retest1 | 2025-05-05 11:15:00 | 1174.05 | 2025-05-09 09:15:00 | 1149.05 | STOP_HIT | 1.00 | -2.13% |
| BUY | retest1 | 2025-05-06 10:15:00 | 1181.25 | 2025-05-09 09:15:00 | 1149.05 | STOP_HIT | 1.00 | -2.73% |
| BUY | retest2 | 2025-05-12 10:15:00 | 1179.50 | 2025-07-25 09:15:00 | 1143.00 | STOP_HIT | 1.00 | -3.09% |
| BUY | retest2 | 2025-06-20 10:15:00 | 1170.90 | 2025-07-25 09:15:00 | 1143.00 | STOP_HIT | 1.00 | -2.38% |
| BUY | retest2 | 2025-07-24 13:15:00 | 1169.35 | 2025-07-25 09:15:00 | 1143.00 | STOP_HIT | 1.00 | -2.25% |
| BUY | retest2 | 2025-10-10 10:15:00 | 1191.80 | 2026-03-11 14:15:00 | 1233.70 | STOP_HIT | 1.00 | 3.52% |
| BUY | retest2 | 2025-10-15 10:15:00 | 1209.90 | 2026-03-11 14:15:00 | 1233.70 | STOP_HIT | 1.00 | 1.97% |
| SELL | retest1 | 2026-04-08 12:15:00 | 1216.70 | 2026-04-10 14:15:00 | 1248.70 | STOP_HIT | 1.00 | -2.63% |
| SELL | retest2 | 2026-04-13 10:15:00 | 1228.30 | 2026-04-15 09:15:00 | 1244.80 | STOP_HIT | 1.00 | -1.34% |
