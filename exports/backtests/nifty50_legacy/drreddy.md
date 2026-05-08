# DRREDDY (DRREDDY)

## Backtest Summary

- **Window:** 2023-06-08 09:15:00 → 2026-05-08 15:30:00 (4997 bars)
- **Last close:** 1293.90
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 30%)
- **Partial:** 50% qty booked @ 15% (ENTRY1) / 15% (ENTRY2), trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 18 |
| ALERT1 | 13 |
| ALERT2 | 13 |
| ALERT2_SKIP | 10 |
| ALERT3 | 14 |
| PENDING | 37 |
| PENDING_CANCEL | 8 |
| ENTRY1 | 4 |
| ENTRY2 | 25 |
| PARTIAL | 1 |
| TARGET_HIT | 0 |
| STOP_HIT | 28 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 29 (incl. partial bookings)
- **Trades open at end:** 1
- **Winners / losers:** 2 / 27
- **Target hits / Stop hits / Partials:** 0 / 28 / 1
- **Avg / median % per leg:** -1.06% / -1.53%
- **Sum % (uncompounded):** -30.67%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 18 | 2 | 11.1% | 0 | 17 | 1 | -0.33% | -6.0% |
| BUY @ 2nd Alert (retest1) | 2 | 0 | 0.0% | 0 | 2 | 0 | -4.03% | -8.1% |
| BUY @ 3rd Alert (retest2) | 16 | 2 | 12.5% | 0 | 15 | 1 | 0.13% | 2.1% |
| SELL (all) | 11 | 0 | 0.0% | 0 | 11 | 0 | -2.24% | -24.7% |
| SELL @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -1.90% | -1.9% |
| SELL @ 3rd Alert (retest2) | 10 | 0 | 0.0% | 0 | 10 | 0 | -2.28% | -22.8% |
| retest1 (combined) | 3 | 0 | 0.0% | 0 | 3 | 0 | -3.32% | -10.0% |
| retest2 (combined) | 26 | 2 | 7.7% | 0 | 25 | 1 | -0.80% | -20.7% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-11-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-02 14:15:00 | 1065.99 | 1100.11 | 1100.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-03 09:15:00 | 1063.85 | 1099.43 | 1099.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-08 09:15:00 | 1097.78 | 1094.09 | 1096.98 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-08 09:15:00 | 1097.78 | 1094.09 | 1096.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-08 09:15:00 | 1097.78 | 1094.09 | 1096.98 | EMA400 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2023-11-10 09:15:00 | 1081.99 | 1094.00 | 1096.74 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2023-11-10 10:15:00 | 1086.46 | 1093.93 | 1096.69 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2023-11-13 11:15:00 | 1079.22 | 1093.23 | 1096.22 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-13 12:15:00 | 1080.61 | 1093.10 | 1096.14 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2023-11-16 12:15:00 | 1101.94 | 1092.49 | 1095.62 | SL hit (close>static) qty=1.00 sl=1099.50 alert=retest2 |

### Cycle 2 — BUY (started 2023-11-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-21 13:15:00 | 1131.45 | 1098.45 | 1098.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-22 09:15:00 | 1141.00 | 1099.50 | 1098.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-11 09:15:00 | 1093.71 | 1126.38 | 1115.13 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-11 09:15:00 | 1093.71 | 1126.38 | 1115.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-11 09:15:00 | 1093.71 | 1126.38 | 1115.13 | EMA400 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2023-12-12 09:15:00 | 1124.00 | 1124.37 | 1114.50 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-12 10:15:00 | 1110.02 | 1124.23 | 1114.47 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-12 13:15:00 | 1276.52 | 1184.69 | 1161.08 | Partial book 0.50 @ 15%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-03-01 14:15:00 | 1237.99 | 1241.43 | 1205.09 | SL hit (close<ema200) qty=0.50 sl=1241.43 alert=retest2 |

### Cycle 3 — SELL (started 2024-05-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-14 14:15:00 | 1173.98 | 1221.62 | 1221.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-16 09:15:00 | 1165.23 | 1218.03 | 1219.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-29 12:15:00 | 1199.84 | 1198.51 | 1208.08 | EMA200 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2024-05-30 09:15:00 | 1183.20 | 1198.37 | 1207.82 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-30 10:15:00 | 1181.99 | 1198.20 | 1207.69 | SELL ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-07 11:15:00 | 1204.40 | 1187.26 | 1199.75 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-06-07 11:15:00 | 1204.40 | 1187.26 | 1199.75 | SL hit (close>ema400) qty=1.00 sl=1199.75 alert=retest1 |
| Cross detected — sustain check pending | 2024-06-19 09:15:00 | 1186.00 | 1195.98 | 1202.06 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-19 10:15:00 | 1191.40 | 1195.94 | 1202.01 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2024-06-21 09:15:00 | 1208.20 | 1195.51 | 1201.40 | SL hit (close>static) qty=1.00 sl=1205.53 alert=retest2 |

### Cycle 4 — BUY (started 2024-06-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-28 14:15:00 | 1279.00 | 1206.13 | 1205.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-01 09:15:00 | 1284.52 | 1207.64 | 1206.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-03 14:15:00 | 1370.01 | 1371.98 | 1336.84 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-05 14:15:00 | 1338.94 | 1370.01 | 1338.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-05 14:15:00 | 1338.94 | 1370.01 | 1338.20 | EMA400 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2024-09-26 14:15:00 | 1348.08 | 1341.60 | 1333.39 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-26 15:15:00 | 1348.88 | 1341.67 | 1333.47 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2024-09-27 12:15:00 | 1350.00 | 1341.78 | 1333.68 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2024-09-27 13:15:00 | 1344.90 | 1341.81 | 1333.74 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2024-09-27 14:15:00 | 1350.65 | 1341.90 | 1333.82 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-27 15:15:00 | 1349.98 | 1341.98 | 1333.90 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2024-09-30 11:15:00 | 1353.56 | 1342.17 | 1334.12 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-30 12:15:00 | 1355.62 | 1342.31 | 1334.23 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2024-10-01 14:15:00 | 1350.64 | 1342.67 | 1334.77 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-01 15:15:00 | 1349.82 | 1342.74 | 1334.84 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-04 09:15:00 | 1338.26 | 1343.28 | 1335.43 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-10-04 10:15:00 | 1334.82 | 1343.19 | 1335.43 | SL hit (close<static) qty=1.00 sl=1337.21 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-10-04 10:15:00 | 1334.82 | 1343.19 | 1335.43 | SL hit (close<static) qty=1.00 sl=1337.21 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-10-04 10:15:00 | 1334.82 | 1343.19 | 1335.43 | SL hit (close<static) qty=1.00 sl=1337.21 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-10-04 10:15:00 | 1334.82 | 1343.19 | 1335.43 | SL hit (close<static) qty=1.00 sl=1337.21 alert=retest2 |
| Cross detected — sustain check pending | 2024-10-16 12:15:00 | 1346.06 | 1336.45 | 1333.53 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2024-10-16 13:15:00 | 1344.02 | 1336.53 | 1333.59 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2024-10-17 09:15:00 | 1349.41 | 1336.75 | 1333.74 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2024-10-17 10:15:00 | 1334.40 | 1336.73 | 1333.75 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2024-10-18 09:15:00 | 1353.27 | 1336.98 | 1333.96 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-18 10:15:00 | 1348.73 | 1337.09 | 1334.03 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2024-10-21 13:15:00 | 1345.84 | 1337.92 | 1334.61 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2024-10-21 14:15:00 | 1339.40 | 1337.94 | 1334.63 | ENTRY2 sustain failed after 60m |
| Stop hit — per-position SL triggered | 2024-10-23 13:15:00 | 1328.96 | 1337.59 | 1334.67 | SL hit (close<static) qty=1.00 sl=1329.11 alert=retest2 |

### Cycle 5 — SELL (started 2024-10-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-29 09:15:00 | 1285.95 | 1331.77 | 1331.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-29 10:15:00 | 1272.70 | 1331.18 | 1331.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-06 10:15:00 | 1320.50 | 1312.03 | 1321.10 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-06 10:15:00 | 1320.50 | 1312.03 | 1321.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-06 10:15:00 | 1320.50 | 1312.03 | 1321.10 | EMA400 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2024-11-07 09:15:00 | 1287.10 | 1311.33 | 1320.48 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-07 10:15:00 | 1286.25 | 1311.08 | 1320.31 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2024-12-19 12:15:00 | 1327.35 | 1251.42 | 1268.95 | SL hit (close>static) qty=1.00 sl=1321.75 alert=retest2 |

### Cycle 6 — BUY (started 2024-12-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-27 14:15:00 | 1389.95 | 1283.66 | 1283.48 | EMA200 above EMA400 |

### Cycle 7 — SELL (started 2025-01-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-28 11:15:00 | 1179.15 | 1300.70 | 1300.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-19 09:15:00 | 1172.25 | 1240.79 | 1262.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-18 11:15:00 | 1164.55 | 1162.86 | 1203.10 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-21 12:15:00 | 1196.10 | 1165.59 | 1200.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-21 12:15:00 | 1196.10 | 1165.59 | 1200.31 | EMA400 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2025-03-25 09:15:00 | 1175.35 | 1169.16 | 1200.29 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-25 10:15:00 | 1175.75 | 1169.23 | 1200.17 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-04-24 09:15:00 | 1202.40 | 1153.87 | 1174.95 | SL hit (close>static) qty=1.00 sl=1200.85 alert=retest2 |
| Cross detected — sustain check pending | 2025-04-24 12:15:00 | 1190.70 | 1155.11 | 1175.26 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-24 13:15:00 | 1189.00 | 1155.45 | 1175.33 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-04-25 09:15:00 | 1180.10 | 1156.56 | 1175.60 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-25 10:15:00 | 1166.60 | 1156.66 | 1175.55 | SELL ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-04-28 09:15:00 | 1202.20 | 1157.85 | 1175.60 | SL hit (close>static) qty=1.00 sl=1200.85 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-04-28 09:15:00 | 1202.20 | 1157.85 | 1175.60 | SL hit (close>static) qty=1.00 sl=1200.85 alert=retest2 |
| Cross detected — sustain check pending | 2025-04-29 09:15:00 | 1179.80 | 1160.37 | 1176.27 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-29 10:15:00 | 1181.40 | 1160.58 | 1176.30 | SELL ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-29 11:15:00 | 1172.00 | 1160.69 | 1176.27 | EMA400 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2025-05-05 10:15:00 | 1165.90 | 1164.40 | 1176.74 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-05 11:15:00 | 1168.00 | 1164.44 | 1176.70 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-05-06 09:15:00 | 1159.00 | 1164.59 | 1176.47 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-06 10:15:00 | 1163.60 | 1164.58 | 1176.41 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-05-12 09:15:00 | 1182.60 | 1162.47 | 1173.80 | SL hit (close>static) qty=1.00 sl=1182.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-05-12 09:15:00 | 1182.60 | 1162.47 | 1173.80 | SL hit (close>static) qty=1.00 sl=1182.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-05-13 09:15:00 | 1236.90 | 1164.87 | 1174.62 | SL hit (close>static) qty=1.00 sl=1200.85 alert=retest2 |

### Cycle 8 — BUY (started 2025-05-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-20 12:15:00 | 1231.00 | 1182.99 | 1182.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-21 09:15:00 | 1240.90 | 1184.75 | 1183.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-30 09:15:00 | 1278.40 | 1294.47 | 1259.45 | EMA200 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2025-07-04 14:15:00 | 1307.30 | 1291.18 | 1262.96 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-04 15:15:00 | 1311.70 | 1291.39 | 1263.20 | BUY ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2025-07-07 10:15:00 | 1307.40 | 1291.62 | 1263.60 | ENTRY1 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-07-07 11:15:00 | 1300.40 | 1291.71 | 1263.78 | ENTRY1 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-07-07 12:15:00 | 1306.70 | 1291.86 | 1264.00 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-07 13:15:00 | 1307.80 | 1292.02 | 1264.21 | BUY ENTRY1 attempt 2/4 (retest1 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 09:15:00 | 1256.90 | 1290.07 | 1265.49 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-07-10 09:15:00 | 1256.90 | 1290.07 | 1265.49 | SL hit (close<ema400) qty=1.00 sl=1265.49 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-07-10 09:15:00 | 1256.90 | 1290.07 | 1265.49 | SL hit (close<ema400) qty=1.00 sl=1265.49 alert=retest1 |
| Cross detected — sustain check pending | 2025-07-10 11:15:00 | 1265.70 | 1289.50 | 1265.45 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-10 12:15:00 | 1269.10 | 1289.30 | 1265.46 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-07-11 09:15:00 | 1266.50 | 1288.35 | 1265.46 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-07-11 10:15:00 | 1258.30 | 1288.05 | 1265.42 | ENTRY2 sustain failed after 60m |
| Stop hit — per-position SL triggered | 2025-07-14 13:15:00 | 1250.10 | 1285.00 | 1264.96 | SL hit (close<static) qty=1.00 sl=1253.00 alert=retest2 |
| Cross detected — sustain check pending | 2025-07-17 11:15:00 | 1265.40 | 1280.06 | 1264.18 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-17 12:15:00 | 1270.40 | 1279.96 | 1264.21 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-07-18 10:15:00 | 1251.90 | 1279.07 | 1264.15 | SL hit (close<static) qty=1.00 sl=1253.00 alert=retest2 |
| Cross detected — sustain check pending | 2025-07-24 09:15:00 | 1276.90 | 1272.09 | 1262.33 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-24 10:15:00 | 1282.20 | 1272.19 | 1262.43 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-08-01 09:15:00 | 1243.40 | 1275.04 | 1265.81 | SL hit (close<static) qty=1.00 sl=1253.00 alert=retest2 |

### Cycle 9 — SELL (started 2025-08-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-07 12:15:00 | 1188.80 | 1257.77 | 1257.92 | EMA200 below EMA400 |

### Cycle 10 — BUY (started 2025-09-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-01 09:15:00 | 1272.80 | 1255.59 | 1255.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-01 10:15:00 | 1274.40 | 1255.78 | 1255.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-02 14:15:00 | 1256.50 | 1257.38 | 1256.49 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-02 14:15:00 | 1256.50 | 1257.38 | 1256.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 14:15:00 | 1256.50 | 1257.38 | 1256.49 | EMA400 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2025-09-04 11:15:00 | 1266.90 | 1257.81 | 1256.76 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-04 12:15:00 | 1269.10 | 1257.92 | 1256.82 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-09-04 14:15:00 | 1252.80 | 1257.88 | 1256.81 | SL hit (close<static) qty=1.00 sl=1254.20 alert=retest2 |
| Cross detected — sustain check pending | 2025-09-05 14:15:00 | 1267.90 | 1258.24 | 1257.03 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-05 15:15:00 | 1268.10 | 1258.33 | 1257.08 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-09-08 14:15:00 | 1250.80 | 1258.32 | 1257.11 | SL hit (close<static) qty=1.00 sl=1254.20 alert=retest2 |
| Cross detected — sustain check pending | 2025-09-09 09:15:00 | 1273.10 | 1258.40 | 1257.16 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-09 10:15:00 | 1276.80 | 1258.58 | 1257.26 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-09-26 14:15:00 | 1252.90 | 1283.75 | 1273.66 | SL hit (close<static) qty=1.00 sl=1254.20 alert=retest2 |

### Cycle 11 — SELL (started 2025-10-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-13 09:15:00 | 1251.50 | 1265.93 | 1265.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-13 10:15:00 | 1250.00 | 1265.77 | 1265.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-20 09:15:00 | 1268.60 | 1259.74 | 1262.60 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-20 09:15:00 | 1268.60 | 1259.74 | 1262.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 09:15:00 | 1268.60 | 1259.74 | 1262.60 | EMA400 retest candle locked (from downside) |

### Cycle 12 — BUY (started 2025-10-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-27 13:15:00 | 1288.80 | 1265.10 | 1265.09 | EMA200 above EMA400 |

### Cycle 13 — SELL (started 2025-10-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-30 11:15:00 | 1199.10 | 1265.01 | 1265.10 | EMA200 below EMA400 |

### Cycle 14 — BUY (started 2025-12-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-08 15:15:00 | 1266.50 | 1252.56 | 1252.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-11 12:15:00 | 1275.20 | 1253.08 | 1252.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-24 09:15:00 | 1259.30 | 1263.99 | 1259.10 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-24 09:15:00 | 1259.30 | 1263.99 | 1259.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 09:15:00 | 1259.30 | 1263.99 | 1259.10 | EMA400 retest candle locked (from upside) |

### Cycle 15 — SELL (started 2026-01-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-09 11:15:00 | 1210.20 | 1256.69 | 1256.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-12 09:15:00 | 1204.20 | 1254.43 | 1255.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-23 09:15:00 | 1243.50 | 1225.78 | 1239.08 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-23 09:15:00 | 1243.50 | 1225.78 | 1239.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 09:15:00 | 1243.50 | 1225.78 | 1239.08 | EMA400 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2026-01-28 10:15:00 | 1224.00 | 1227.47 | 1239.00 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-01-28 11:15:00 | 1226.40 | 1227.46 | 1238.94 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2026-01-28 12:15:00 | 1223.30 | 1227.42 | 1238.86 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-28 13:15:00 | 1222.30 | 1227.37 | 1238.78 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2026-02-05 12:15:00 | 1248.00 | 1224.93 | 1235.17 | SL hit (close>static) qty=1.00 sl=1247.00 alert=retest2 |

### Cycle 16 — BUY (started 2026-02-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-17 14:15:00 | 1286.00 | 1242.42 | 1242.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-19 10:15:00 | 1287.90 | 1246.26 | 1244.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-16 09:15:00 | 1283.50 | 1284.04 | 1268.83 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-16 10:15:00 | 1274.60 | 1283.95 | 1268.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 10:15:00 | 1274.60 | 1283.95 | 1268.86 | EMA400 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2026-03-17 13:15:00 | 1284.30 | 1283.15 | 1269.19 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-17 14:15:00 | 1284.90 | 1283.17 | 1269.26 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2026-03-20 09:15:00 | 1298.00 | 1283.42 | 1270.47 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-20 10:15:00 | 1294.10 | 1283.52 | 1270.59 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2026-03-23 14:15:00 | 1255.90 | 1283.50 | 1271.28 | SL hit (close<static) qty=1.00 sl=1265.10 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-23 14:15:00 | 1255.90 | 1283.50 | 1271.28 | SL hit (close<static) qty=1.00 sl=1265.10 alert=retest2 |
| Cross detected — sustain check pending | 2026-03-25 09:15:00 | 1291.50 | 1281.50 | 1270.79 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-25 10:15:00 | 1292.50 | 1281.61 | 1270.90 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2026-03-30 09:15:00 | 1260.40 | 1283.01 | 1272.31 | SL hit (close<static) qty=1.00 sl=1265.10 alert=retest2 |

### Cycle 17 — SELL (started 2026-04-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-08 09:15:00 | 1187.30 | 1263.39 | 1263.53 | EMA200 below EMA400 |

### Cycle 18 — BUY (started 2026-04-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-28 09:15:00 | 1340.40 | 1259.31 | 1259.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-28 10:15:00 | 1359.20 | 1260.31 | 1259.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-05 12:15:00 | 1273.80 | 1275.07 | 1267.89 | EMA200 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2026-05-06 09:15:00 | 1283.20 | 1275.06 | 1268.02 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-06 10:15:00 | 1283.20 | 1275.14 | 1268.10 | BUY ENTRY1 attempt 1/4 (retest1 break sustained 60m) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2023-11-13 12:15:00 | 1080.61 | 2023-11-16 12:15:00 | 1101.94 | STOP_HIT | 1.00 | -1.97% |
| BUY | retest2 | 2023-12-12 10:15:00 | 1110.02 | 2024-02-12 13:15:00 | 1276.52 | PARTIAL | 0.50 | 15.00% |
| BUY | retest2 | 2023-12-12 10:15:00 | 1110.02 | 2024-03-01 14:15:00 | 1237.99 | STOP_HIT | 0.50 | 11.53% |
| SELL | retest1 | 2024-05-30 10:15:00 | 1181.99 | 2024-06-07 11:15:00 | 1204.40 | STOP_HIT | 1.00 | -1.90% |
| SELL | retest2 | 2024-06-19 10:15:00 | 1191.40 | 2024-06-21 09:15:00 | 1208.20 | STOP_HIT | 1.00 | -1.41% |
| BUY | retest2 | 2024-09-26 15:15:00 | 1348.88 | 2024-10-04 10:15:00 | 1334.82 | STOP_HIT | 1.00 | -1.04% |
| BUY | retest2 | 2024-09-27 15:15:00 | 1349.98 | 2024-10-04 10:15:00 | 1334.82 | STOP_HIT | 1.00 | -1.12% |
| BUY | retest2 | 2024-09-30 12:15:00 | 1355.62 | 2024-10-04 10:15:00 | 1334.82 | STOP_HIT | 1.00 | -1.53% |
| BUY | retest2 | 2024-10-01 15:15:00 | 1349.82 | 2024-10-04 10:15:00 | 1334.82 | STOP_HIT | 1.00 | -1.11% |
| BUY | retest2 | 2024-10-18 10:15:00 | 1348.73 | 2024-10-23 13:15:00 | 1328.96 | STOP_HIT | 1.00 | -1.47% |
| SELL | retest2 | 2024-11-07 10:15:00 | 1286.25 | 2024-12-19 12:15:00 | 1327.35 | STOP_HIT | 1.00 | -3.20% |
| SELL | retest2 | 2025-03-25 10:15:00 | 1175.75 | 2025-04-24 09:15:00 | 1202.40 | STOP_HIT | 1.00 | -2.27% |
| SELL | retest2 | 2025-04-24 13:15:00 | 1189.00 | 2025-04-28 09:15:00 | 1202.20 | STOP_HIT | 1.00 | -1.11% |
| SELL | retest2 | 2025-04-25 10:15:00 | 1166.60 | 2025-04-28 09:15:00 | 1202.20 | STOP_HIT | 1.00 | -3.05% |
| SELL | retest2 | 2025-04-29 10:15:00 | 1181.40 | 2025-05-12 09:15:00 | 1182.60 | STOP_HIT | 1.00 | -0.10% |
| SELL | retest2 | 2025-05-05 11:15:00 | 1168.00 | 2025-05-12 09:15:00 | 1182.60 | STOP_HIT | 1.00 | -1.25% |
| SELL | retest2 | 2025-05-06 10:15:00 | 1163.60 | 2025-05-13 09:15:00 | 1236.90 | STOP_HIT | 1.00 | -6.30% |
| BUY | retest1 | 2025-07-04 15:15:00 | 1311.70 | 2025-07-10 09:15:00 | 1256.90 | STOP_HIT | 1.00 | -4.18% |
| BUY | retest1 | 2025-07-07 13:15:00 | 1307.80 | 2025-07-10 09:15:00 | 1256.90 | STOP_HIT | 1.00 | -3.89% |
| BUY | retest2 | 2025-07-10 12:15:00 | 1269.10 | 2025-07-14 13:15:00 | 1250.10 | STOP_HIT | 1.00 | -1.50% |
| BUY | retest2 | 2025-07-17 12:15:00 | 1270.40 | 2025-07-18 10:15:00 | 1251.90 | STOP_HIT | 1.00 | -1.46% |
| BUY | retest2 | 2025-07-24 10:15:00 | 1282.20 | 2025-08-01 09:15:00 | 1243.40 | STOP_HIT | 1.00 | -3.03% |
| BUY | retest2 | 2025-09-04 12:15:00 | 1269.10 | 2025-09-04 14:15:00 | 1252.80 | STOP_HIT | 1.00 | -1.28% |
| BUY | retest2 | 2025-09-05 15:15:00 | 1268.10 | 2025-09-08 14:15:00 | 1250.80 | STOP_HIT | 1.00 | -1.36% |
| BUY | retest2 | 2025-09-09 10:15:00 | 1276.80 | 2025-09-26 14:15:00 | 1252.90 | STOP_HIT | 1.00 | -1.87% |
| SELL | retest2 | 2026-01-28 13:15:00 | 1222.30 | 2026-02-05 12:15:00 | 1248.00 | STOP_HIT | 1.00 | -2.10% |
| BUY | retest2 | 2026-03-17 14:15:00 | 1284.90 | 2026-03-23 14:15:00 | 1255.90 | STOP_HIT | 1.00 | -2.26% |
| BUY | retest2 | 2026-03-20 10:15:00 | 1294.10 | 2026-03-23 14:15:00 | 1255.90 | STOP_HIT | 1.00 | -2.95% |
| BUY | retest2 | 2026-03-25 10:15:00 | 1292.50 | 2026-03-30 09:15:00 | 1260.40 | STOP_HIT | 1.00 | -2.48% |
