# NESTLEIND (NESTLEIND.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-06-06 09:15:00 → 2026-05-06 15:15:00 (4996 bars)
- **Last close:** 1486.10
- **Strategy:** EMA 200/400 1H crossover (v2 BTC rules)
- **Target:** entry × (1 ± 30%)
- **Partial:** 50% qty booked @ 15%, trail SL → entry
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 3 attempts each at retest1 and retest2

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 10 |
| ALERT1 | 10 |
| ALERT2 | 9 |
| ALERT2_SKIP | 7 |
| ALERT3 | 15 |
| PENDING | 47 |
| PENDING_CANCEL | 13 |
| ENTRY1 | 3 |
| ENTRY2 | 31 |
| PARTIAL | 1 |
| TARGET_HIT | 0 |
| STOP_HIT | 34 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 35 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 4 / 31
- **Target hits / Stop hits / Partials:** 0 / 34 / 1
- **Avg / median % per leg:** -0.75% / -1.48%
- **Sum % (uncompounded):** -26.35%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 4 | 4 | 100.0% | 0 | 3 | 1 | 7.62% | 30.5% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 4 | 4 | 100.0% | 0 | 3 | 1 | 7.62% | 30.5% |
| SELL (all) | 31 | 0 | 0.0% | 0 | 31 | 0 | -1.83% | -56.8% |
| SELL @ 2nd Alert (retest1) | 3 | 0 | 0.0% | 0 | 3 | 0 | -2.76% | -8.3% |
| SELL @ 3rd Alert (retest2) | 28 | 0 | 0.0% | 0 | 28 | 0 | -1.73% | -48.5% |
| retest1 (combined) | 3 | 0 | 0.0% | 0 | 3 | 0 | -2.76% | -8.3% |
| retest2 (combined) | 32 | 4 | 12.5% | 0 | 31 | 1 | -0.56% | -18.1% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-09-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-25 14:15:00 | 1131.78 | 1114.67 | 1114.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-26 09:15:00 | 1136.95 | 1115.06 | 1114.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-03 09:15:00 | 1118.29 | 1119.91 | 1117.49 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-03 13:15:00 | 1115.20 | 1119.87 | 1117.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-03 13:15:00 | 1115.20 | 1119.87 | 1117.51 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2023-10-04 09:15:00 | 1133.18 | 1119.91 | 1117.57 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-04 10:15:00 | 1136.00 | 1120.07 | 1117.66 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Partial book — 50% qty @ 15%, trail SL → entry | 2023-12-28 11:15:00 | 1306.40 | 1238.66 | 1213.05 | Partial book 0.50 @ 15%; trail SL->entry alert=retest2 |
| Stop hit — per-position SL triggered | 2024-04-23 15:15:00 | 1249.38 | 1265.48 | 1265.48 | Force close (CROSSOVER_FLIP) qty=0.50 alert=retest2 |

### Cycle 2 — SELL (started 2024-04-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-23 15:15:00 | 1249.38 | 1265.48 | 1265.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-26 14:15:00 | 1240.80 | 1264.36 | 1264.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-30 12:15:00 | 1265.00 | 1263.52 | 1264.43 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-30 12:15:00 | 1265.00 | 1263.52 | 1264.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-30 12:15:00 | 1265.00 | 1263.52 | 1264.43 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2024-04-30 14:15:00 | 1253.70 | 1263.40 | 1264.36 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-30 15:15:00 | 1253.70 | 1263.30 | 1264.30 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2024-05-02 09:15:00 | 1267.50 | 1263.32 | 1264.31 | SL hit qty=1.00 sl=1267.50 alert=retest2 |
| Cross detected — sustain check pending | 2024-05-02 14:15:00 | 1255.50 | 1263.25 | 1264.25 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-02 15:15:00 | 1255.50 | 1263.17 | 1264.20 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2024-05-08 14:15:00 | 1267.50 | 1258.84 | 1261.71 | SL hit qty=1.00 sl=1267.50 alert=retest2 |
| Cross detected — sustain check pending | 2024-05-09 09:15:00 | 1257.57 | 1258.94 | 1261.72 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-09 10:15:00 | 1258.57 | 1258.93 | 1261.71 | SELL ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2024-05-09 12:15:00 | 1254.47 | 1258.91 | 1261.67 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-09 13:15:00 | 1256.03 | 1258.88 | 1261.64 | SELL ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-10 09:15:00 | 1270.53 | 1258.96 | 1261.64 | EMA400 retest candle locked |
| Stop hit — per-position SL triggered | 2024-05-10 09:15:00 | 1267.50 | 1258.96 | 1261.64 | SL hit qty=1.00 sl=1267.50 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-05-10 09:15:00 | 1267.50 | 1258.96 | 1261.64 | SL hit qty=1.00 sl=1267.50 alert=retest2 |
| Cross detected — sustain check pending | 2024-05-14 09:15:00 | 1242.97 | 1259.14 | 1261.56 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-14 10:15:00 | 1242.32 | 1258.97 | 1261.46 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2024-06-05 09:15:00 | 1274.50 | 1232.81 | 1244.86 | SL hit qty=1.00 sl=1274.50 alert=retest2 |
| Cross detected — sustain check pending | 2024-06-05 12:15:00 | 1251.80 | 1233.63 | 1245.09 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-05 13:15:00 | 1250.72 | 1233.80 | 1245.12 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2024-06-06 09:15:00 | 1233.00 | 1234.22 | 1245.16 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-06 10:15:00 | 1235.43 | 1234.23 | 1245.11 | SELL ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2024-06-10 11:15:00 | 1274.50 | 1236.27 | 1245.37 | SL hit qty=1.00 sl=1274.50 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-06-10 11:15:00 | 1274.50 | 1236.27 | 1245.37 | SL hit qty=1.00 sl=1274.50 alert=retest2 |
| Cross detected — sustain check pending | 2024-06-21 10:15:00 | 1252.00 | 1251.01 | 1251.62 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2024-06-21 11:15:00 | 1257.07 | 1251.07 | 1251.64 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2024-06-21 14:15:00 | 1246.32 | 1251.14 | 1251.67 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-21 15:15:00 | 1245.70 | 1251.08 | 1251.64 | SELL ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-24 09:15:00 | 1258.47 | 1251.16 | 1251.67 | EMA400 retest candle locked |
| Stop hit — per-position SL triggered | 2024-06-25 12:15:00 | 1255.03 | 1252.19 | 1252.18 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 3 — BUY (started 2024-06-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-25 12:15:00 | 1255.03 | 1252.19 | 1252.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-25 14:15:00 | 1260.43 | 1252.31 | 1252.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-24 09:15:00 | 1264.50 | 1282.87 | 1271.79 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-24 09:15:00 | 1264.50 | 1282.87 | 1271.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-24 09:15:00 | 1264.50 | 1282.87 | 1271.79 | EMA400 retest candle locked |

### Cycle 4 — SELL (started 2024-08-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-07 10:15:00 | 1257.65 | 1263.71 | 1263.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-08 09:15:00 | 1251.50 | 1263.41 | 1263.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-16 14:15:00 | 1263.75 | 1256.84 | 1259.94 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-16 14:15:00 | 1263.75 | 1256.84 | 1259.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-16 14:15:00 | 1263.75 | 1256.84 | 1259.94 | EMA400 retest candle locked |
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
| Stop hit — per-position SL triggered | 2024-09-03 09:15:00 | 1264.00 | 1258.36 | 1260.00 | SL hit qty=1.00 sl=1264.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-09-03 09:15:00 | 1264.00 | 1258.36 | 1260.00 | SL hit qty=1.00 sl=1264.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-09-03 09:15:00 | 1264.00 | 1258.36 | 1260.00 | SL hit qty=1.00 sl=1264.00 alert=retest2 |
| Cross detected — sustain check pending | 2024-09-05 11:15:00 | 1252.25 | 1259.30 | 1260.37 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2024-09-05 12:15:00 | 1253.70 | 1259.25 | 1260.34 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2024-09-05 13:15:00 | 1250.00 | 1259.16 | 1260.29 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2024-09-05 14:15:00 | 1252.53 | 1259.09 | 1260.25 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2024-09-05 15:15:00 | 1252.45 | 1259.02 | 1260.21 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-06 09:15:00 | 1245.10 | 1258.89 | 1260.14 | SELL ENTRY2 attempt 4/4 (retest2 break sustained 1080m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-09 15:15:00 | 1258.50 | 1258.24 | 1259.72 | EMA400 retest candle locked |
| Stop hit — per-position SL triggered | 2024-09-10 09:15:00 | 1264.00 | 1258.29 | 1259.74 | SL hit qty=1.00 sl=1264.00 alert=retest2 |
| Cross detected — sustain check pending | 2024-09-12 11:15:00 | 1253.00 | 1259.39 | 1260.21 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-12 12:15:00 | 1249.57 | 1259.29 | 1260.16 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2024-09-12 14:15:00 | 1259.95 | 1259.28 | 1260.14 | SL hit qty=1.00 sl=1259.95 alert=retest2 |

### Cycle 5 — BUY (started 2024-09-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-17 09:15:00 | 1280.75 | 1260.98 | 1260.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-17 11:15:00 | 1289.22 | 1261.50 | 1261.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-04 13:15:00 | 1307.25 | 1308.17 | 1289.23 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-07 14:15:00 | 1287.40 | 1307.42 | 1289.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-07 14:15:00 | 1287.40 | 1307.42 | 1289.59 | EMA400 retest candle locked |

### Cycle 6 — SELL (started 2024-10-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-18 10:15:00 | 1171.00 | 1277.63 | 1277.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-23 13:15:00 | 1167.45 | 1256.52 | 1266.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-03 10:15:00 | 1113.50 | 1106.49 | 1137.76 | EMA200 retest candle locked |
| Cross detected — sustain check pending | 2025-01-06 09:15:00 | 1101.50 | 1106.86 | 1137.02 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-06 10:15:00 | 1095.38 | 1106.75 | 1136.81 | SELL ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2025-01-07 10:15:00 | 1103.05 | 1106.07 | 1135.43 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-07 11:15:00 | 1102.90 | 1106.04 | 1135.27 | SELL ENTRY1 attempt 2/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2025-01-08 09:15:00 | 1101.97 | 1105.96 | 1134.51 | ENTRY1 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-01-08 10:15:00 | 1104.75 | 1105.95 | 1134.36 | ENTRY1 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-01-08 11:15:00 | 1102.62 | 1105.91 | 1134.20 | ENTRY1 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-01-08 12:15:00 | 1104.43 | 1105.90 | 1134.05 | ENTRY1 sustain failed after 60m |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-09 10:15:00 | 1128.22 | 1106.31 | 1133.56 | EMA400 retest candle locked |
| Stop hit — per-position SL triggered | 2025-01-09 10:15:00 | 1133.56 | 1106.31 | 1133.56 | SL hit qty=1.00 sl=1133.56 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-01-09 10:15:00 | 1133.56 | 1106.31 | 1133.56 | SL hit qty=1.00 sl=1133.56 alert=retest1 |
| Cross detected — sustain check pending | 2025-01-14 09:15:00 | 1105.85 | 1109.52 | 1132.67 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-14 10:15:00 | 1107.53 | 1109.50 | 1132.55 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-01-14 12:15:00 | 1110.82 | 1109.55 | 1132.34 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-14 13:15:00 | 1107.53 | 1109.53 | 1132.22 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-01-21 14:15:00 | 1097.35 | 1106.98 | 1127.05 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-21 15:15:00 | 1100.00 | 1106.91 | 1126.91 | SELL ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-01-27 10:15:00 | 1106.28 | 1106.22 | 1124.37 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-27 11:15:00 | 1104.07 | 1106.20 | 1124.27 | SELL ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-31 09:15:00 | 1124.45 | 1102.49 | 1120.04 | EMA400 retest candle locked |
| Stop hit — per-position SL triggered | 2025-01-31 10:15:00 | 1135.82 | 1102.83 | 1120.13 | SL hit qty=1.00 sl=1135.82 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-01-31 10:15:00 | 1135.82 | 1102.83 | 1120.13 | SL hit qty=1.00 sl=1135.82 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-01-31 10:15:00 | 1135.82 | 1102.83 | 1120.13 | SL hit qty=1.00 sl=1135.82 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-01-31 10:15:00 | 1135.82 | 1102.83 | 1120.13 | SL hit qty=1.00 sl=1135.82 alert=retest2 |
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
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-24 11:15:00 | 1116.50 | 1107.74 | 1116.66 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-02-24 13:15:00 | 1110.05 | 1107.83 | 1116.62 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-24 14:15:00 | 1110.03 | 1107.85 | 1116.59 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-02-25 09:15:00 | 1118.25 | 1107.97 | 1116.56 | SL hit qty=1.00 sl=1118.25 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-02-27 11:15:00 | 1134.90 | 1109.66 | 1117.04 | SL hit qty=1.00 sl=1134.90 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-02-27 11:15:00 | 1134.90 | 1109.66 | 1117.04 | SL hit qty=1.00 sl=1134.90 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-02-27 11:15:00 | 1134.90 | 1109.66 | 1117.04 | SL hit qty=1.00 sl=1134.90 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-02-27 11:15:00 | 1134.90 | 1109.66 | 1117.04 | SL hit qty=1.00 sl=1134.90 alert=retest2 |
| Cross detected — sustain check pending | 2025-02-28 12:15:00 | 1110.53 | 1111.09 | 1117.49 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-28 13:15:00 | 1102.47 | 1111.01 | 1117.41 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-03-07 10:15:00 | 1118.25 | 1105.23 | 1113.29 | SL hit qty=1.00 sl=1118.25 alert=retest2 |
| Cross detected — sustain check pending | 2025-03-07 15:15:00 | 1112.50 | 1105.80 | 1113.38 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-03-10 09:15:00 | 1131.90 | 1106.06 | 1113.48 | ENTRY2 sustain failed after 3960m |
| Cross detected — sustain check pending | 2025-03-12 09:15:00 | 1111.28 | 1108.82 | 1114.41 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-12 10:15:00 | 1109.03 | 1108.82 | 1114.38 | SELL ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-03-21 09:15:00 | 1118.25 | 1105.15 | 1111.29 | SL hit qty=1.00 sl=1118.25 alert=retest2 |
| Cross detected — sustain check pending | 2025-04-01 12:15:00 | 1110.40 | 1113.63 | 1114.80 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-01 13:15:00 | 1110.60 | 1113.60 | 1114.78 | SELL ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-01 14:15:00 | 1117.25 | 1113.64 | 1114.79 | EMA400 retest candle locked |
| Stop hit — per-position SL triggered | 2025-04-01 14:15:00 | 1118.25 | 1113.64 | 1114.79 | SL hit qty=1.00 sl=1118.25 alert=retest2 |
| Cross detected — sustain check pending | 2025-04-02 09:15:00 | 1089.38 | 1113.42 | 1114.67 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-02 10:15:00 | 1097.47 | 1113.26 | 1114.59 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-04-03 13:15:00 | 1118.40 | 1112.77 | 1114.26 | SL hit qty=1.00 sl=1118.40 alert=retest2 |
| Cross detected — sustain check pending | 2025-04-07 09:15:00 | 1101.90 | 1113.92 | 1114.78 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-07 10:15:00 | 1100.80 | 1113.79 | 1114.71 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-04-07 12:15:00 | 1108.40 | 1113.73 | 1114.68 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-04-07 13:15:00 | 1118.53 | 1113.78 | 1114.70 | ENTRY2 sustain failed after 60m |
| Stop hit — per-position SL triggered | 2025-04-07 13:15:00 | 1118.40 | 1113.78 | 1114.70 | SL hit qty=1.00 sl=1118.40 alert=retest2 |
| CROSSOVER_SKIP | 2025-04-09 09:15:00 | 1151.40 | 1115.71 | 1115.63 | HTF filter: close below htf_sma |

### Cycle 7 — SELL (started 2025-07-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-30 14:15:00 | 1115.85 | 1187.84 | 1187.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-07 10:15:00 | 1111.75 | 1169.13 | 1177.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-18 09:15:00 | 1161.60 | 1146.26 | 1163.57 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-18 09:15:00 | 1161.60 | 1146.26 | 1163.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 09:15:00 | 1161.60 | 1146.26 | 1163.57 | EMA400 retest candle locked |

### Cycle 8 — BUY (started 2025-09-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-10 09:15:00 | 1206.00 | 1171.03 | 1170.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-10 11:15:00 | 1216.10 | 1171.86 | 1171.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-22 14:15:00 | 1186.60 | 1187.17 | 1180.41 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-23 09:15:00 | 1171.90 | 1187.04 | 1180.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 09:15:00 | 1171.90 | 1187.04 | 1180.41 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-10-10 09:15:00 | 1191.40 | 1178.03 | 1177.04 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-10 10:15:00 | 1191.80 | 1178.17 | 1177.12 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-10-15 09:15:00 | 1197.90 | 1179.74 | 1178.06 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-15 10:15:00 | 1209.90 | 1180.04 | 1178.22 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2026-03-11 14:15:00 | 1233.70 | 1278.41 | 1278.50 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-11 14:15:00 | 1233.70 | 1278.41 | 1278.50 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 9 — SELL (started 2026-03-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-11 14:15:00 | 1233.70 | 1278.41 | 1278.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-12 09:15:00 | 1221.20 | 1277.40 | 1277.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-07 15:15:00 | 1222.80 | 1222.46 | 1243.07 | EMA200 retest candle locked |
| Cross detected — sustain check pending | 2026-04-08 11:15:00 | 1210.10 | 1222.40 | 1242.73 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-08 12:15:00 | 1216.70 | 1222.35 | 1242.60 | SELL ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2026-04-09 13:15:00 | 1218.00 | 1221.90 | 1241.58 | ENTRY1 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-04-09 14:15:00 | 1229.50 | 1221.98 | 1241.52 | ENTRY1 sustain failed after 60m |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-10 13:15:00 | 1240.90 | 1222.73 | 1241.32 | EMA400 retest candle locked |
| Stop hit — per-position SL triggered | 2026-04-10 13:15:00 | 1241.32 | 1222.73 | 1241.32 | SL hit qty=1.00 sl=1241.32 alert=retest1 |
| Cross detected — sustain check pending | 2026-04-13 09:15:00 | 1223.70 | 1223.25 | 1241.31 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-13 10:15:00 | 1228.30 | 1223.30 | 1241.25 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2026-04-15 09:15:00 | 1241.80 | 1223.66 | 1240.90 | SL hit qty=1.00 sl=1241.80 alert=retest2 |

### Cycle 10 — BUY (started 2026-04-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-23 11:15:00 | 1410.50 | 1255.11 | 1254.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-23 13:15:00 | 1412.80 | 1258.18 | 1256.01 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2023-10-04 10:15:00 | 1136.00 | 2023-12-28 11:15:00 | 1306.40 | PARTIAL | 0.50 | 15.00% |
| BUY | retest2 | 2023-10-04 10:15:00 | 1136.00 | 2024-04-23 15:15:00 | 1249.38 | STOP_HIT | 0.50 | 9.98% |
| SELL | retest2 | 2024-04-30 15:15:00 | 1253.70 | 2024-05-02 09:15:00 | 1267.50 | STOP_HIT | 1.00 | -1.10% |
| SELL | retest2 | 2024-05-02 15:15:00 | 1255.50 | 2024-05-08 14:15:00 | 1267.50 | STOP_HIT | 1.00 | -0.96% |
| SELL | retest2 | 2024-05-09 10:15:00 | 1258.57 | 2024-05-10 09:15:00 | 1267.50 | STOP_HIT | 1.00 | -0.71% |
| SELL | retest2 | 2024-05-09 13:15:00 | 1256.03 | 2024-05-10 09:15:00 | 1267.50 | STOP_HIT | 1.00 | -0.91% |
| SELL | retest2 | 2024-05-14 10:15:00 | 1242.32 | 2024-06-05 09:15:00 | 1274.50 | STOP_HIT | 1.00 | -2.59% |
| SELL | retest2 | 2024-06-05 13:15:00 | 1250.72 | 2024-06-10 11:15:00 | 1274.50 | STOP_HIT | 1.00 | -1.90% |
| SELL | retest2 | 2024-06-06 10:15:00 | 1235.43 | 2024-06-10 11:15:00 | 1274.50 | STOP_HIT | 1.00 | -3.16% |
| SELL | retest2 | 2024-06-21 15:15:00 | 1245.70 | 2024-06-25 12:15:00 | 1255.03 | STOP_HIT | 1.00 | -0.75% |
| SELL | retest2 | 2024-08-28 11:15:00 | 1251.30 | 2024-09-03 09:15:00 | 1264.00 | STOP_HIT | 1.00 | -1.01% |
| SELL | retest2 | 2024-08-28 14:15:00 | 1245.57 | 2024-09-03 09:15:00 | 1264.00 | STOP_HIT | 1.00 | -1.48% |
| SELL | retest2 | 2024-08-29 14:15:00 | 1252.47 | 2024-09-03 09:15:00 | 1264.00 | STOP_HIT | 1.00 | -0.92% |
| SELL | retest2 | 2024-09-06 09:15:00 | 1245.10 | 2024-09-10 09:15:00 | 1264.00 | STOP_HIT | 1.00 | -1.52% |
| SELL | retest2 | 2024-09-12 12:15:00 | 1249.57 | 2024-09-12 14:15:00 | 1259.95 | STOP_HIT | 1.00 | -0.83% |
| SELL | retest1 | 2025-01-06 10:15:00 | 1095.38 | 2025-01-09 10:15:00 | 1133.56 | STOP_HIT | 1.00 | -3.49% |
| SELL | retest1 | 2025-01-07 11:15:00 | 1102.90 | 2025-01-09 10:15:00 | 1133.56 | STOP_HIT | 1.00 | -2.78% |
| SELL | retest2 | 2025-01-14 10:15:00 | 1107.53 | 2025-01-31 10:15:00 | 1135.82 | STOP_HIT | 1.00 | -2.56% |
| SELL | retest2 | 2025-01-14 13:15:00 | 1107.53 | 2025-01-31 10:15:00 | 1135.82 | STOP_HIT | 1.00 | -2.56% |
| SELL | retest2 | 2025-01-21 15:15:00 | 1100.00 | 2025-01-31 10:15:00 | 1135.82 | STOP_HIT | 1.00 | -3.26% |
| SELL | retest2 | 2025-01-27 11:15:00 | 1104.07 | 2025-01-31 10:15:00 | 1135.82 | STOP_HIT | 1.00 | -2.88% |
| SELL | retest2 | 2025-02-10 13:15:00 | 1104.07 | 2025-02-25 09:15:00 | 1118.25 | STOP_HIT | 1.00 | -1.28% |
| SELL | retest2 | 2025-02-11 11:15:00 | 1104.43 | 2025-02-27 11:15:00 | 1134.90 | STOP_HIT | 1.00 | -2.76% |
| SELL | retest2 | 2025-02-18 09:15:00 | 1104.07 | 2025-02-27 11:15:00 | 1134.90 | STOP_HIT | 1.00 | -2.79% |
| SELL | retest2 | 2025-02-19 12:15:00 | 1104.00 | 2025-02-27 11:15:00 | 1134.90 | STOP_HIT | 1.00 | -2.80% |
| SELL | retest2 | 2025-02-24 14:15:00 | 1110.03 | 2025-02-27 11:15:00 | 1134.90 | STOP_HIT | 1.00 | -2.24% |
| SELL | retest2 | 2025-02-28 13:15:00 | 1102.47 | 2025-03-07 10:15:00 | 1118.25 | STOP_HIT | 1.00 | -1.43% |
| SELL | retest2 | 2025-03-12 10:15:00 | 1109.03 | 2025-03-21 09:15:00 | 1118.25 | STOP_HIT | 1.00 | -0.83% |
| SELL | retest2 | 2025-04-01 13:15:00 | 1110.60 | 2025-04-01 14:15:00 | 1118.25 | STOP_HIT | 1.00 | -0.69% |
| SELL | retest2 | 2025-04-02 10:15:00 | 1097.47 | 2025-04-03 13:15:00 | 1118.40 | STOP_HIT | 1.00 | -1.91% |
| SELL | retest2 | 2025-04-07 10:15:00 | 1100.80 | 2025-04-07 13:15:00 | 1118.40 | STOP_HIT | 1.00 | -1.60% |
| BUY | retest2 | 2025-10-10 10:15:00 | 1191.80 | 2026-03-11 14:15:00 | 1233.70 | STOP_HIT | 1.00 | 3.52% |
| BUY | retest2 | 2025-10-15 10:15:00 | 1209.90 | 2026-03-11 14:15:00 | 1233.70 | STOP_HIT | 1.00 | 1.97% |
| SELL | retest1 | 2026-04-08 12:15:00 | 1216.70 | 2026-04-10 13:15:00 | 1241.32 | STOP_HIT | 1.00 | -2.02% |
| SELL | retest2 | 2026-04-13 10:15:00 | 1228.30 | 2026-04-15 09:15:00 | 1241.80 | STOP_HIT | 1.00 | -1.10% |
