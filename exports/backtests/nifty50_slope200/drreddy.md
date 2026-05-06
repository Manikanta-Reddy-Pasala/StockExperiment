# DRREDDY (DRREDDY.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-06-06 09:15:00 → 2026-05-06 15:30:00 (4997 bars)
- **Last close:** 1311.00
- **Strategy:** EMA 200/400 1H crossover (v2 BTC rules)
- **Target:** entry × (1 ± 30%)
- **Partial:** 50% qty booked @ 15%, trail SL → entry
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 3 attempts each at retest1 and retest2

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 10 |
| ALERT1 | 6 |
| ALERT2 | 6 |
| ALERT2_SKIP | 5 |
| ALERT3 | 6 |
| PENDING | 22 |
| PENDING_CANCEL | 4 |
| ENTRY1 | 1 |
| ENTRY2 | 17 |
| PARTIAL | 1 |
| TARGET_HIT | 0 |
| STOP_HIT | 17 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 18 (incl. partial bookings)
- **Trades open at end:** 1
- **Winners / losers:** 7 / 11
- **Target hits / Stop hits / Partials:** 0 / 17 / 1
- **Avg / median % per leg:** 1.52% / -0.93%
- **Sum % (uncompounded):** 27.40%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 18 | 7 | 38.9% | 0 | 17 | 1 | 1.52% | 27.4% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 18 | 7 | 38.9% | 0 | 17 | 1 | 1.52% | 27.4% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 18 | 7 | 38.9% | 0 | 17 | 1 | 1.52% | 27.4% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-11-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-21 12:15:00 | 1130.80 | 1098.12 | 1098.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-22 09:15:00 | 1141.00 | 1099.50 | 1098.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-11 09:15:00 | 1093.71 | 1126.37 | 1114.99 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-11 09:15:00 | 1093.71 | 1126.37 | 1114.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-11 09:15:00 | 1093.71 | 1126.37 | 1114.99 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2023-12-12 09:15:00 | 1124.00 | 1124.37 | 1114.36 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-12 10:15:00 | 1110.02 | 1124.23 | 1114.33 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Partial book — 50% qty @ 15%, trail SL → entry | 2024-02-12 13:15:00 | 1276.52 | 1184.69 | 1161.05 | Partial book 0.50 @ 15%; trail SL->entry alert=retest2 |
| CROSSOVER_SKIP | 2024-05-14 14:15:00 | 1173.98 | 1221.62 | 1221.83 | HTF filter: close above htf_sma |
| Stop hit — per-position SL triggered | 2024-06-28 14:15:00 | 1279.00 | 1206.13 | 1205.92 | Force close (CROSSOVER_FLIP) qty=0.50 alert=retest2 |

### Cycle 2 — BUY (started 2024-06-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-28 14:15:00 | 1279.00 | 1206.13 | 1205.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-01 09:15:00 | 1284.52 | 1207.64 | 1206.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-03 14:15:00 | 1370.01 | 1371.98 | 1336.84 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-05 14:15:00 | 1338.94 | 1370.01 | 1338.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-05 14:15:00 | 1338.94 | 1370.01 | 1338.20 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2024-09-26 14:15:00 | 1348.08 | 1341.60 | 1333.39 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-26 15:15:00 | 1348.88 | 1341.67 | 1333.47 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2024-09-27 09:15:00 | 1337.21 | 1341.66 | 1333.50 | SL hit qty=1.00 sl=1337.21 alert=retest2 |
| Cross detected — sustain check pending | 2024-09-27 12:15:00 | 1350.00 | 1341.78 | 1333.68 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2024-09-27 13:15:00 | 1344.90 | 1341.81 | 1333.74 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2024-09-27 14:15:00 | 1350.65 | 1341.90 | 1333.82 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-27 15:15:00 | 1349.98 | 1341.98 | 1333.90 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2024-09-30 11:15:00 | 1353.56 | 1342.17 | 1334.12 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-30 12:15:00 | 1355.62 | 1342.31 | 1334.23 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2024-10-01 10:15:00 | 1337.21 | 1342.55 | 1334.55 | SL hit qty=1.00 sl=1337.21 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-10-01 10:15:00 | 1337.21 | 1342.55 | 1334.55 | SL hit qty=1.00 sl=1337.21 alert=retest2 |
| Cross detected — sustain check pending | 2024-10-01 14:15:00 | 1350.64 | 1342.67 | 1334.77 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-01 15:15:00 | 1349.82 | 1342.74 | 1334.84 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-04 09:15:00 | 1338.26 | 1343.28 | 1335.43 | EMA400 retest candle locked |
| Stop hit — per-position SL triggered | 2024-10-04 09:15:00 | 1337.21 | 1343.28 | 1335.43 | SL hit qty=1.00 sl=1337.21 alert=retest2 |
| Cross detected — sustain check pending | 2024-10-16 12:15:00 | 1346.06 | 1336.45 | 1333.53 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2024-10-16 13:15:00 | 1344.02 | 1336.53 | 1333.59 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2024-10-17 09:15:00 | 1349.41 | 1336.75 | 1333.74 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2024-10-17 10:15:00 | 1334.40 | 1336.73 | 1333.75 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2024-10-18 09:15:00 | 1353.27 | 1336.98 | 1333.96 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-18 10:15:00 | 1348.73 | 1337.09 | 1334.03 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2024-10-21 13:15:00 | 1345.84 | 1337.92 | 1334.61 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2024-10-21 14:15:00 | 1339.40 | 1337.94 | 1334.63 | ENTRY2 sustain failed after 60m |
| Stop hit — per-position SL triggered | 2024-10-22 15:15:00 | 1329.11 | 1337.94 | 1334.76 | SL hit qty=1.00 sl=1329.11 alert=retest2 |
| CROSSOVER_SKIP | 2024-10-29 09:15:00 | 1285.95 | 1331.77 | 1331.93 | HTF filter: close above htf_sma |
| Cross detected — sustain check pending | 2024-12-20 09:15:00 | 1365.20 | 1254.75 | 1270.28 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-20 10:15:00 | 1354.85 | 1255.74 | 1270.70 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2024-12-23 10:15:00 | 1349.50 | 1262.03 | 1273.39 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-23 11:15:00 | 1345.95 | 1262.86 | 1273.75 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2024-12-24 09:15:00 | 1345.85 | 1266.64 | 1275.39 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-24 10:15:00 | 1352.70 | 1267.50 | 1275.78 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2024-12-27 14:15:00 | 1389.95 | 1283.66 | 1283.48 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-12-27 14:15:00 | 1389.95 | 1283.66 | 1283.48 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-12-27 14:15:00 | 1389.95 | 1283.66 | 1283.48 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 3 — BUY (started 2024-12-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-27 14:15:00 | 1389.95 | 1283.66 | 1283.48 | EMA200 above EMA400 |
| CROSSOVER_SKIP | 2025-01-28 11:15:00 | 1179.15 | 1300.70 | 1300.75 | slope filter: EMA200 not falling 2.00% over 1400 bars |
| CROSSOVER_SKIP | 2025-05-20 12:15:00 | 1231.00 | 1182.99 | 1182.81 | HTF filter: close below htf_sma |

### Cycle 4 — SELL (started 2025-08-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-07 12:15:00 | 1188.80 | 1257.77 | 1257.92 | EMA200 below EMA400 |

### Cycle 5 — BUY (started 2025-09-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-01 09:15:00 | 1272.80 | 1255.59 | 1255.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-01 10:15:00 | 1274.40 | 1255.78 | 1255.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-02 14:15:00 | 1256.50 | 1257.38 | 1256.49 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-02 14:15:00 | 1256.50 | 1257.38 | 1256.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 14:15:00 | 1256.50 | 1257.38 | 1256.49 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-09-04 11:15:00 | 1266.90 | 1257.81 | 1256.76 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-04 12:15:00 | 1269.10 | 1257.92 | 1256.82 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-09-04 14:15:00 | 1254.20 | 1257.88 | 1256.81 | SL hit qty=1.00 sl=1254.20 alert=retest2 |
| Cross detected — sustain check pending | 2025-09-05 14:15:00 | 1267.90 | 1258.24 | 1257.03 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-05 15:15:00 | 1268.10 | 1258.33 | 1257.08 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-09-08 11:15:00 | 1254.20 | 1258.39 | 1257.13 | SL hit qty=1.00 sl=1254.20 alert=retest2 |
| Cross detected — sustain check pending | 2025-09-09 09:15:00 | 1273.10 | 1258.40 | 1257.16 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-09 10:15:00 | 1276.80 | 1258.58 | 1257.26 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-09-26 11:15:00 | 1254.20 | 1284.58 | 1273.92 | SL hit qty=1.00 sl=1254.20 alert=retest2 |
| CROSSOVER_SKIP | 2025-10-13 09:15:00 | 1251.50 | 1265.93 | 1265.97 | HTF filter: close above htf_sma |
| Cross detected — sustain check pending | 2025-10-20 09:15:00 | 1268.60 | 1259.74 | 1262.60 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-20 10:15:00 | 1277.30 | 1259.91 | 1262.67 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-10-27 13:15:00 | 1288.80 | 1265.10 | 1265.09 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 6 — BUY (started 2025-10-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-27 13:15:00 | 1288.80 | 1265.10 | 1265.09 | EMA200 above EMA400 |

### Cycle 7 — SELL (started 2025-10-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-30 11:15:00 | 1199.10 | 1265.01 | 1265.10 | EMA200 below EMA400 |

### Cycle 8 — BUY (started 2025-12-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-08 15:15:00 | 1266.50 | 1252.56 | 1252.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-11 12:15:00 | 1275.20 | 1253.08 | 1252.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-24 09:15:00 | 1259.30 | 1263.99 | 1259.10 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-24 09:15:00 | 1259.30 | 1263.99 | 1259.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 09:15:00 | 1259.30 | 1263.99 | 1259.10 | EMA400 retest candle locked |
| CROSSOVER_SKIP | 2026-01-09 11:15:00 | 1210.20 | 1256.69 | 1256.85 | slope filter: EMA200 not falling 2.00% over 1400 bars |

### Cycle 9 — BUY (started 2026-02-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-17 14:15:00 | 1286.00 | 1242.42 | 1242.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-19 10:15:00 | 1287.90 | 1246.26 | 1244.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-16 09:15:00 | 1283.50 | 1284.04 | 1268.83 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-16 10:15:00 | 1274.60 | 1283.95 | 1268.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 10:15:00 | 1274.60 | 1283.95 | 1268.86 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2026-03-17 13:15:00 | 1284.30 | 1283.15 | 1269.19 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-17 14:15:00 | 1284.90 | 1283.17 | 1269.26 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2026-03-20 09:15:00 | 1298.00 | 1283.42 | 1270.47 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-20 10:15:00 | 1294.10 | 1283.52 | 1270.59 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2026-03-23 09:15:00 | 1265.10 | 1284.11 | 1271.27 | SL hit qty=1.00 sl=1265.10 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-23 09:15:00 | 1265.10 | 1284.11 | 1271.27 | SL hit qty=1.00 sl=1265.10 alert=retest2 |
| Cross detected — sustain check pending | 2026-03-25 09:15:00 | 1291.50 | 1281.50 | 1270.79 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-25 10:15:00 | 1292.50 | 1281.61 | 1270.90 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2026-03-30 09:15:00 | 1265.10 | 1283.01 | 1272.31 | SL hit qty=1.00 sl=1265.10 alert=retest2 |
| CROSSOVER_SKIP | 2026-04-08 09:15:00 | 1187.30 | 1263.39 | 1263.53 | slope filter: EMA200 not falling 2.00% over 1400 bars |
| Cross detected — sustain check pending | 2026-04-23 09:15:00 | 1306.00 | 1243.52 | 1251.74 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-23 10:15:00 | 1299.30 | 1244.08 | 1251.98 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2026-04-28 09:15:00 | 1340.40 | 1259.31 | 1259.21 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 10 — BUY (started 2026-04-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-28 09:15:00 | 1340.40 | 1259.31 | 1259.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-28 10:15:00 | 1359.20 | 1260.31 | 1259.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-05 12:15:00 | 1273.80 | 1275.07 | 1267.89 | EMA200 retest candle locked |
| Cross detected — sustain check pending | 2026-05-06 09:15:00 | 1283.20 | 1275.06 | 1268.02 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-06 10:15:00 | 1283.20 | 1275.14 | 1268.10 | BUY ENTRY1 attempt 1/4 (retest1 break sustained 60m) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2023-12-12 10:15:00 | 1110.02 | 2024-02-12 13:15:00 | 1276.52 | PARTIAL | 0.50 | 15.00% |
| BUY | retest2 | 2023-12-12 10:15:00 | 1110.02 | 2024-06-28 14:15:00 | 1279.00 | STOP_HIT | 0.50 | 15.22% |
| BUY | retest2 | 2024-09-26 15:15:00 | 1348.88 | 2024-09-27 09:15:00 | 1337.21 | STOP_HIT | 1.00 | -0.87% |
| BUY | retest2 | 2024-09-27 15:15:00 | 1349.98 | 2024-10-01 10:15:00 | 1337.21 | STOP_HIT | 1.00 | -0.95% |
| BUY | retest2 | 2024-09-30 12:15:00 | 1355.62 | 2024-10-01 10:15:00 | 1337.21 | STOP_HIT | 1.00 | -1.36% |
| BUY | retest2 | 2024-10-01 15:15:00 | 1349.82 | 2024-10-04 09:15:00 | 1337.21 | STOP_HIT | 1.00 | -0.93% |
| BUY | retest2 | 2024-10-18 10:15:00 | 1348.73 | 2024-10-22 15:15:00 | 1329.11 | STOP_HIT | 1.00 | -1.45% |
| BUY | retest2 | 2024-12-20 10:15:00 | 1354.85 | 2024-12-27 14:15:00 | 1389.95 | STOP_HIT | 1.00 | 2.59% |
| BUY | retest2 | 2024-12-23 11:15:00 | 1345.95 | 2024-12-27 14:15:00 | 1389.95 | STOP_HIT | 1.00 | 3.27% |
| BUY | retest2 | 2024-12-24 10:15:00 | 1352.70 | 2024-12-27 14:15:00 | 1389.95 | STOP_HIT | 1.00 | 2.75% |
| BUY | retest2 | 2025-09-04 12:15:00 | 1269.10 | 2025-09-04 14:15:00 | 1254.20 | STOP_HIT | 1.00 | -1.17% |
| BUY | retest2 | 2025-09-05 15:15:00 | 1268.10 | 2025-09-08 11:15:00 | 1254.20 | STOP_HIT | 1.00 | -1.10% |
| BUY | retest2 | 2025-09-09 10:15:00 | 1276.80 | 2025-09-26 11:15:00 | 1254.20 | STOP_HIT | 1.00 | -1.77% |
| BUY | retest2 | 2025-10-20 10:15:00 | 1277.30 | 2025-10-27 13:15:00 | 1288.80 | STOP_HIT | 1.00 | 0.90% |
| BUY | retest2 | 2026-03-17 14:15:00 | 1284.90 | 2026-03-23 09:15:00 | 1265.10 | STOP_HIT | 1.00 | -1.54% |
| BUY | retest2 | 2026-03-20 10:15:00 | 1294.10 | 2026-03-23 09:15:00 | 1265.10 | STOP_HIT | 1.00 | -2.24% |
| BUY | retest2 | 2026-03-25 10:15:00 | 1292.50 | 2026-03-30 09:15:00 | 1265.10 | STOP_HIT | 1.00 | -2.12% |
| BUY | retest2 | 2026-04-23 10:15:00 | 1299.30 | 2026-04-28 09:15:00 | 1340.40 | STOP_HIT | 1.00 | 3.16% |
