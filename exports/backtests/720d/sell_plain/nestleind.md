# NESTLEIND (NESTLEIND)

## Backtest Summary

- **Source:** Fyers history API (1H bars)
- **Window:** 2024-05-18 09:15:00 → 2026-05-07 15:15:00 (3402 bars)
- **Last close:** 1475.50
- **Strategy:** EMA 200/400 1H crossover (v2 BTC rules)
- **Target:** entry × (1 ± 30%)
- **Partial:** 50% qty booked @ 15%, trail SL → entry
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 3 attempts each at retest1 and retest2

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 4 |
| ALERT1 | 4 |
| ALERT2 | 4 |
| ALERT2_SKIP | 3 |
| ALERT3 | 6 |
| PENDING | 36 |
| PENDING_CANCEL | 20 |
| ENTRY1 | 2 |
| ENTRY2 | 14 |
| PARTIAL | 0 |
| TARGET_HIT | 0 |
| STOP_HIT | 16 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 16 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 1 / 15
- **Target hits / Stop hits / Partials:** 0 / 16 / 0
- **Avg / median % per leg:** -2.37% / -3.29%
- **Sum % (uncompounded):** -37.99%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 16 | 1 | 6.2% | 0 | 16 | 0 | -2.37% | -38.0% |
| SELL @ 2nd Alert (retest1) | 2 | 0 | 0.0% | 0 | 2 | 0 | -3.20% | -6.4% |
| SELL @ 3rd Alert (retest2) | 14 | 1 | 7.1% | 0 | 14 | 0 | -2.26% | -31.6% |
| retest1 (combined) | 2 | 0 | 0.0% | 0 | 2 | 0 | -3.20% | -6.4% |
| retest2 (combined) | 14 | 1 | 7.1% | 0 | 14 | 0 | -2.26% | -31.6% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-08-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-09 15:15:00 | 1252.65 | 1261.39 | 1261.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-12 09:15:00 | 1237.47 | 1261.15 | 1261.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-16 14:15:00 | 1263.65 | 1256.76 | 1258.94 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-16 14:15:00 | 1263.65 | 1256.76 | 1258.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-16 14:15:00 | 1263.65 | 1256.76 | 1258.94 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2024-08-19 14:15:00 | 1250.97 | 1256.85 | 1258.91 | ENTRY2 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2024-08-19 15:15:00 | 1252.50 | 1256.81 | 1258.88 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2024-08-20 13:15:00 | 1252.00 | 1256.79 | 1258.82 | ENTRY2 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2024-08-20 14:15:00 | 1257.80 | 1256.80 | 1258.82 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2024-08-28 10:15:00 | 1247.50 | 1260.08 | 1260.31 | ENTRY2 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2024-08-28 12:15:00 | 1253.25 | 1259.92 | 1260.23 | ENTRY2 sustain failed after 120m |
| Cross detected — sustain check pending | 2024-08-28 13:15:00 | 1249.80 | 1259.82 | 1260.17 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-28 15:15:00 | 1247.33 | 1259.55 | 1260.04 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 120m) |
| Cross detected — sustain check pending | 2024-08-29 13:15:00 | 1250.68 | 1259.21 | 1259.85 | ENTRY2 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2024-08-29 15:15:00 | 1252.50 | 1259.07 | 1259.77 | ENTRY2 sustain failed after 120m |
| Cross detected — sustain check pending | 2024-08-30 09:15:00 | 1250.88 | 1258.99 | 1259.73 | ENTRY2 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2024-08-30 11:15:00 | 1254.28 | 1258.87 | 1259.66 | ENTRY2 sustain failed after 120m |
| Cross detected — sustain check pending | 2024-08-30 14:15:00 | 1248.60 | 1258.69 | 1259.56 | ENTRY2 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2024-08-30 15:15:00 | 1253.00 | 1258.63 | 1259.53 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2024-09-02 11:15:00 | 1252.45 | 1258.46 | 1259.43 | ENTRY2 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2024-09-02 12:15:00 | 1255.13 | 1258.43 | 1259.41 | ENTRY2 sustain failed after 60m |
| Stop hit — per-position SL triggered | 2024-09-03 09:15:00 | 1264.50 | 1258.38 | 1259.36 | SL hit (close>static) qty=1.00 sl=1264.00 alert=retest2 |
| Cross detected — sustain check pending | 2024-09-05 13:15:00 | 1250.00 | 1259.17 | 1259.70 | ENTRY2 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2024-09-05 14:15:00 | 1252.53 | 1259.10 | 1259.67 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2024-09-06 09:15:00 | 1245.10 | 1258.92 | 1259.57 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-06 11:15:00 | 1251.60 | 1258.73 | 1259.47 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 120m) |
| Cross detected — sustain check pending | 2024-09-09 09:15:00 | 1247.50 | 1258.38 | 1259.27 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-09 11:15:00 | 1252.03 | 1258.26 | 1259.20 | SELL ENTRY2 attempt 3/4 (retest2 break sustained 120m) |
| Stop hit — per-position SL triggered | 2024-09-10 10:15:00 | 1268.22 | 1258.42 | 1259.26 | SL hit (close>static) qty=1.00 sl=1264.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-09-10 10:15:00 | 1268.22 | 1258.42 | 1259.26 | SL hit (close>static) qty=1.00 sl=1264.00 alert=retest2 |
| Cross detected — sustain check pending | 2024-09-12 12:15:00 | 1249.58 | 1259.37 | 1259.70 | ENTRY2 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2024-09-12 13:15:00 | 1254.97 | 1259.33 | 1259.67 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2024-10-09 09:15:00 | 1235.88 | 1305.16 | 1288.95 | ENTRY2 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2024-10-09 10:15:00 | 1260.65 | 1304.72 | 1288.81 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2024-10-11 09:15:00 | 1250.50 | 1299.14 | 1286.94 | ENTRY2 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2024-10-11 10:15:00 | 1252.50 | 1298.67 | 1286.77 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2024-10-14 10:15:00 | 1250.43 | 1295.67 | 1285.65 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-14 12:15:00 | 1249.00 | 1294.76 | 1285.29 | SELL ENTRY2 attempt 4/4 (retest2 break sustained 120m) |
| Stop hit — per-position SL triggered | 2024-10-18 11:15:00 | 1173.15 | 1276.59 | 1276.91 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 2 — SELL (started 2024-10-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-18 11:15:00 | 1173.15 | 1276.59 | 1276.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-23 13:15:00 | 1167.50 | 1256.53 | 1266.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-03 10:15:00 | 1113.50 | 1106.45 | 1137.51 | EMA200 retest candle locked |
| Cross detected — sustain check pending | 2025-01-06 09:15:00 | 1101.50 | 1106.81 | 1136.77 | ENTRY1 cross detected — sustain check pending (75m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-06 11:15:00 | 1095.25 | 1106.59 | 1136.36 | SELL ENTRY1 attempt 1/4 (retest1 break sustained 120m) |
| Cross detected — sustain check pending | 2025-01-07 10:15:00 | 1103.05 | 1106.06 | 1135.22 | ENTRY1 cross detected — sustain check pending (75m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-07 12:15:00 | 1102.88 | 1106.00 | 1134.90 | SELL ENTRY1 attempt 2/4 (retest1 break sustained 120m) |
| Cross detected — sustain check pending | 2025-01-08 09:15:00 | 1101.97 | 1105.93 | 1134.29 | ENTRY1 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2025-01-08 10:15:00 | 1104.75 | 1105.92 | 1134.14 | ENTRY1 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-01-08 11:15:00 | 1103.03 | 1105.89 | 1133.99 | ENTRY1 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2025-01-08 12:15:00 | 1104.43 | 1105.88 | 1133.84 | ENTRY1 sustain failed after 60m |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-09 10:15:00 | 1128.22 | 1106.27 | 1133.35 | EMA400 retest candle locked |
| Stop hit — per-position SL triggered | 2025-01-10 10:15:00 | 1134.25 | 1107.88 | 1133.24 | SL hit (close>ema400) qty=1.00 sl=1133.24 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-01-10 10:15:00 | 1134.25 | 1107.88 | 1133.24 | SL hit (close>ema400) qty=1.00 sl=1133.24 alert=retest1 |
| Cross detected — sustain check pending | 2025-01-14 09:15:00 | 1106.20 | 1109.49 | 1132.47 | ENTRY2 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2025-01-14 11:15:00 | 1113.40 | 1109.51 | 1132.26 | ENTRY2 sustain failed after 120m |
| Cross detected — sustain check pending | 2025-01-14 12:15:00 | 1110.80 | 1109.52 | 1132.15 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-14 14:15:00 | 1108.53 | 1109.49 | 1131.91 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 120m) |
| Cross detected — sustain check pending | 2025-01-15 09:15:00 | 1092.38 | 1109.35 | 1131.62 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-15 11:15:00 | 1090.38 | 1109.00 | 1131.22 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 120m) |
| Cross detected — sustain check pending | 2025-01-21 14:15:00 | 1097.35 | 1106.99 | 1126.90 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-22 09:15:00 | 1098.03 | 1106.83 | 1126.63 | SELL ENTRY2 attempt 3/4 (retest2 break sustained 1140m) |
| Cross detected — sustain check pending | 2025-01-27 10:15:00 | 1106.28 | 1106.23 | 1124.24 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-27 12:15:00 | 1096.18 | 1106.11 | 1124.00 | SELL ENTRY2 attempt 4/4 (retest2 break sustained 120m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-31 09:15:00 | 1124.45 | 1102.55 | 1119.96 | EMA400 retest candle locked |
| Stop hit — per-position SL triggered | 2025-01-31 10:15:00 | 1136.97 | 1102.90 | 1120.05 | SL hit (close>static) qty=1.00 sl=1135.90 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-01-31 10:15:00 | 1136.97 | 1102.90 | 1120.05 | SL hit (close>static) qty=1.00 sl=1135.90 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-01-31 10:15:00 | 1136.97 | 1102.90 | 1120.05 | SL hit (close>static) qty=1.00 sl=1135.90 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-01-31 10:15:00 | 1136.97 | 1102.90 | 1120.05 | SL hit (close>static) qty=1.00 sl=1135.90 alert=retest2 |
| Cross detected — sustain check pending | 2025-02-10 13:15:00 | 1104.30 | 1115.75 | 1123.74 | ENTRY2 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2025-02-10 14:15:00 | 1110.50 | 1115.70 | 1123.67 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-02-11 11:15:00 | 1104.43 | 1115.39 | 1123.36 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-11 13:15:00 | 1092.20 | 1114.94 | 1123.05 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 120m) |
| Cross detected — sustain check pending | 2025-02-17 13:15:00 | 1101.78 | 1110.40 | 1119.56 | ENTRY2 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2025-02-17 14:15:00 | 1109.20 | 1110.39 | 1119.50 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-02-18 10:15:00 | 1100.50 | 1110.20 | 1119.28 | ENTRY2 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2025-02-18 11:15:00 | 1106.55 | 1110.17 | 1119.21 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-02-19 12:15:00 | 1103.95 | 1109.97 | 1118.76 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-19 14:15:00 | 1103.18 | 1109.86 | 1118.61 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 120m) |
| Stop hit — per-position SL triggered | 2025-02-27 14:15:00 | 1139.50 | 1111.50 | 1118.01 | SL hit (close>static) qty=1.00 sl=1134.90 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-02-27 14:15:00 | 1139.50 | 1111.50 | 1118.01 | SL hit (close>static) qty=1.00 sl=1134.90 alert=retest2 |
| Cross detected — sustain check pending | 2025-02-28 13:15:00 | 1102.40 | 1112.03 | 1118.08 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-28 15:15:00 | 1098.15 | 1111.71 | 1117.87 | SELL ENTRY2 attempt 3/4 (retest2 break sustained 120m) |
| Stop hit — per-position SL triggered | 2025-03-10 10:15:00 | 1134.97 | 1107.07 | 1114.15 | SL hit (close>static) qty=1.00 sl=1134.90 alert=retest2 |
| Cross detected — sustain check pending | 2025-03-12 11:15:00 | 1100.22 | 1109.35 | 1114.83 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-12 13:15:00 | 1100.35 | 1109.17 | 1114.69 | SELL ENTRY2 attempt 4/4 (retest2 break sustained 120m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-20 14:15:00 | 1109.90 | 1105.33 | 1111.66 | EMA400 retest candle locked |
| Stop hit — per-position SL triggered | 2025-03-24 09:15:00 | 1141.72 | 1107.52 | 1112.50 | SL hit (close>static) qty=1.00 sl=1134.90 alert=retest2 |
| Cross detected — sustain check pending | 2025-04-02 09:15:00 | 1089.38 | 1113.62 | 1114.97 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-02 11:15:00 | 1101.85 | 1113.34 | 1114.82 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 120m) |
| Stop hit — per-position SL triggered | 2025-04-03 12:15:00 | 1116.70 | 1112.86 | 1114.51 | SL hit (close>static) qty=1.00 sl=1114.38 alert=retest2 |
| Cross detected — sustain check pending | 2025-04-07 09:15:00 | 1101.75 | 1114.00 | 1115.01 | ENTRY2 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2025-04-07 11:15:00 | 1113.72 | 1113.87 | 1114.94 | ENTRY2 sustain failed after 120m |
| Cross detected — sustain check pending | 2025-04-07 12:15:00 | 1108.47 | 1113.81 | 1114.91 | ENTRY2 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2025-04-07 13:15:00 | 1118.70 | 1113.86 | 1114.92 | ENTRY2 sustain failed after 60m |

### Cycle 3 — SELL (started 2025-07-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-30 14:15:00 | 1115.85 | 1187.85 | 1187.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-07 10:15:00 | 1111.90 | 1169.11 | 1177.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-18 09:15:00 | 1161.20 | 1146.25 | 1163.55 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-18 09:15:00 | 1161.20 | 1146.25 | 1163.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 09:15:00 | 1161.20 | 1146.25 | 1163.55 | EMA400 retest candle locked |

### Cycle 4 — SELL (started 2026-03-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-11 13:15:00 | 1234.50 | 1278.90 | 1279.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-12 09:15:00 | 1221.30 | 1277.41 | 1278.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 09:15:00 | 1230.70 | 1222.56 | 1243.19 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-08 09:15:00 | 1230.70 | 1222.56 | 1243.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 09:15:00 | 1230.70 | 1222.56 | 1243.19 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2026-04-08 11:15:00 | 1210.30 | 1222.43 | 1242.92 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-08 13:15:00 | 1211.50 | 1222.26 | 1242.63 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 120m) |
| Cross detected — sustain check pending | 2026-04-09 13:15:00 | 1218.00 | 1221.93 | 1241.76 | ENTRY2 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2026-04-09 14:15:00 | 1229.50 | 1222.00 | 1241.69 | ENTRY2 sustain failed after 60m |
| Stop hit — per-position SL triggered | 2026-04-15 13:15:00 | 1253.60 | 1224.69 | 1241.22 | SL hit (close>static) qty=1.00 sl=1250.60 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2024-08-28 15:15:00 | 1247.33 | 2024-09-03 09:15:00 | 1264.50 | STOP_HIT | 1.00 | -1.38% |
| SELL | retest2 | 2024-09-06 11:15:00 | 1251.60 | 2024-09-10 10:15:00 | 1268.22 | STOP_HIT | 1.00 | -1.33% |
| SELL | retest2 | 2024-09-09 11:15:00 | 1252.03 | 2024-09-10 10:15:00 | 1268.22 | STOP_HIT | 1.00 | -1.29% |
| SELL | retest2 | 2024-10-14 12:15:00 | 1249.00 | 2024-10-18 11:15:00 | 1173.15 | STOP_HIT | 1.00 | 6.07% |
| SELL | retest1 | 2025-01-06 11:15:00 | 1095.25 | 2025-01-10 10:15:00 | 1134.25 | STOP_HIT | 1.00 | -3.56% |
| SELL | retest1 | 2025-01-07 12:15:00 | 1102.88 | 2025-01-10 10:15:00 | 1134.25 | STOP_HIT | 1.00 | -2.84% |
| SELL | retest2 | 2025-01-14 14:15:00 | 1108.53 | 2025-01-31 10:15:00 | 1136.97 | STOP_HIT | 1.00 | -2.57% |
| SELL | retest2 | 2025-01-15 11:15:00 | 1090.38 | 2025-01-31 10:15:00 | 1136.97 | STOP_HIT | 1.00 | -4.27% |
| SELL | retest2 | 2025-01-22 09:15:00 | 1098.03 | 2025-01-31 10:15:00 | 1136.97 | STOP_HIT | 1.00 | -3.55% |
| SELL | retest2 | 2025-01-27 12:15:00 | 1096.18 | 2025-01-31 10:15:00 | 1136.97 | STOP_HIT | 1.00 | -3.72% |
| SELL | retest2 | 2025-02-11 13:15:00 | 1092.20 | 2025-02-27 14:15:00 | 1139.50 | STOP_HIT | 1.00 | -4.33% |
| SELL | retest2 | 2025-02-19 14:15:00 | 1103.18 | 2025-02-27 14:15:00 | 1139.50 | STOP_HIT | 1.00 | -3.29% |
| SELL | retest2 | 2025-02-28 15:15:00 | 1098.15 | 2025-03-10 10:15:00 | 1134.97 | STOP_HIT | 1.00 | -3.35% |
| SELL | retest2 | 2025-03-12 13:15:00 | 1100.35 | 2025-03-24 09:15:00 | 1141.72 | STOP_HIT | 1.00 | -3.76% |
| SELL | retest2 | 2025-04-02 11:15:00 | 1101.85 | 2025-04-03 12:15:00 | 1116.70 | STOP_HIT | 1.00 | -1.35% |
| SELL | retest2 | 2026-04-08 13:15:00 | 1211.50 | 2026-04-15 13:15:00 | 1253.60 | STOP_HIT | 1.00 | -3.48% |
