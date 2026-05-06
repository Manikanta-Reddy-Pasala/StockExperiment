# AXISBANK (AXISBANK.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-06-06 09:15:00 → 2026-05-06 15:15:00 (4996 bars)
- **Last close:** 1294.20
- **Strategy:** EMA 200/400 1H crossover (v2 BTC rules)
- **Target:** entry × (1 ± 30%)
- **Partial:** 50% qty booked @ 15%, trail SL → entry
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 3 attempts each at retest1 and retest2

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 5 |
| ALERT1 | 5 |
| ALERT2 | 5 |
| ALERT2_SKIP | 5 |
| ALERT3 | 8 |
| PENDING | 25 |
| PENDING_CANCEL | 6 |
| ENTRY1 | 0 |
| ENTRY2 | 19 |
| PARTIAL | 1 |
| TARGET_HIT | 0 |
| STOP_HIT | 14 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 15 (incl. partial bookings)
- **Trades open at end:** 5
- **Winners / losers:** 11 / 4
- **Target hits / Stop hits / Partials:** 0 / 14 / 1
- **Avg / median % per leg:** 4.20% / 1.50%
- **Sum % (uncompounded):** 63.00%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 15 | 11 | 73.3% | 0 | 14 | 1 | 4.20% | 63.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 15 | 11 | 73.3% | 0 | 14 | 1 | 4.20% | 63.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 15 | 11 | 73.3% | 0 | 14 | 1 | 4.20% | 63.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-08-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-29 13:15:00 | 980.45 | 960.45 | 960.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-30 09:15:00 | 994.50 | 961.18 | 960.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-06 13:15:00 | 966.65 | 967.72 | 964.43 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-06 14:15:00 | 963.10 | 967.67 | 964.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-06 14:15:00 | 963.10 | 967.67 | 964.42 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2023-09-07 10:15:00 | 971.20 | 967.67 | 964.47 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2023-09-07 11:15:00 | 967.05 | 967.67 | 964.48 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2023-09-07 13:15:00 | 971.25 | 967.69 | 964.52 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-07 14:15:00 | 977.45 | 967.79 | 964.59 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2023-10-23 14:15:00 | 962.00 | 1000.50 | 992.47 | SL hit qty=1.00 sl=962.00 alert=retest2 |
| Cross detected — sustain check pending | 2023-10-26 12:15:00 | 968.95 | 996.20 | 990.72 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2023-10-26 13:15:00 | 968.05 | 995.92 | 990.61 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2023-10-26 14:15:00 | 972.10 | 995.68 | 990.51 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-26 15:15:00 | 972.05 | 995.45 | 990.42 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Partial book — 50% qty @ 15%, trail SL → entry | 2023-12-04 09:15:00 | 1117.86 | 1017.54 | 1005.97 | Partial book 0.50 @ 15%; trail SL->entry alert=retest2 |
| Stop hit — per-position SL triggered | 2024-04-29 10:15:00 | 1142.50 | 1070.75 | 1070.57 | Force close (CROSSOVER_FLIP) qty=0.50 alert=retest2 |

### Cycle 2 — BUY (started 2024-04-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-29 10:15:00 | 1142.50 | 1070.75 | 1070.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-29 11:15:00 | 1150.95 | 1071.55 | 1070.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-04 11:15:00 | 1124.95 | 1140.91 | 1117.52 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-04 11:15:00 | 1124.95 | 1140.91 | 1117.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 11:15:00 | 1124.95 | 1140.91 | 1117.52 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2024-06-05 13:15:00 | 1171.60 | 1141.28 | 1118.73 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-05 14:15:00 | 1185.00 | 1141.71 | 1119.06 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2024-06-06 13:15:00 | 1169.85 | 1143.80 | 1120.79 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-06 14:15:00 | 1170.95 | 1144.07 | 1121.04 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2024-07-25 14:15:00 | 1175.10 | 1254.39 | 1217.31 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-25 15:15:00 | 1176.00 | 1253.61 | 1217.10 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2024-07-26 12:15:00 | 1173.65 | 1250.32 | 1216.16 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-26 13:15:00 | 1178.15 | 1249.60 | 1215.97 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-26 14:15:00 | 1177.30 | 1248.88 | 1215.78 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2024-07-29 10:15:00 | 1184.55 | 1246.82 | 1215.23 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2024-07-29 11:15:00 | 1177.70 | 1246.13 | 1215.05 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2024-08-27 12:15:00 | 1182.05 | 1181.74 | 1187.00 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-27 13:15:00 | 1183.80 | 1181.76 | 1186.98 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2024-08-28 09:15:00 | 1175.00 | 1181.65 | 1186.85 | SL hit qty=1.00 sl=1175.00 alert=retest2 |
| Cross detected — sustain check pending | 2024-09-02 09:15:00 | 1181.70 | 1180.41 | 1185.67 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-02 10:15:00 | 1183.00 | 1180.44 | 1185.66 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2024-09-03 11:15:00 | 1181.85 | 1180.77 | 1185.62 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-03 12:15:00 | 1182.65 | 1180.79 | 1185.61 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2024-09-04 10:15:00 | 1175.00 | 1180.96 | 1185.58 | SL hit qty=1.00 sl=1175.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-09-04 10:15:00 | 1175.00 | 1180.96 | 1185.58 | SL hit qty=1.00 sl=1175.00 alert=retest2 |
| Cross detected — sustain check pending | 2024-09-05 13:15:00 | 1181.30 | 1180.67 | 1185.20 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-05 14:15:00 | 1180.70 | 1180.67 | 1185.18 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| ALERT3_SKIP | 2024-09-05 15:15:00 | 1180.55 | 1180.66 | 1185.15 | max_alert3_locks_per_cycle=2 reached — end cycle |

### Cycle 3 — BUY (started 2024-09-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-17 13:15:00 | 1235.45 | 1187.97 | 1187.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-18 11:15:00 | 1242.40 | 1190.30 | 1188.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-03 09:15:00 | 1201.15 | 1219.05 | 1206.16 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-03 09:15:00 | 1201.15 | 1219.05 | 1206.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-03 09:15:00 | 1201.15 | 1219.05 | 1206.16 | EMA400 retest candle locked |
| CROSSOVER_SKIP | 2025-03-28 10:15:00 | 1098.45 | 1045.79 | 1045.61 | HTF filter: close below htf_sma |
| Cross detected — sustain check pending | 2025-04-21 10:15:00 | 1237.00 | 1079.62 | 1064.90 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-21 11:15:00 | 1233.10 | 1081.15 | 1065.74 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-04-23 10:15:00 | 1200.75 | 1097.97 | 1075.40 | SL hit qty=1.00 sl=1200.75 alert=retest2 |
| Cross detected — sustain check pending | 2025-05-26 09:15:00 | 1221.90 | 1171.64 | 1136.95 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-05-26 10:15:00 | 1215.70 | 1172.08 | 1137.34 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-06-09 09:15:00 | 1224.10 | 1180.35 | 1152.58 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-06-09 10:15:00 | 1218.00 | 1180.72 | 1152.91 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-06-09 11:15:00 | 1222.60 | 1181.14 | 1153.26 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-09 12:15:00 | 1221.20 | 1181.54 | 1153.59 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-06-13 09:15:00 | 1200.75 | 1191.06 | 1161.95 | SL hit qty=1.00 sl=1200.75 alert=retest2 |
| Cross detected — sustain check pending | 2025-06-17 09:15:00 | 1221.60 | 1193.62 | 1165.24 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-17 10:15:00 | 1223.80 | 1193.92 | 1165.53 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-06-18 09:15:00 | 1219.50 | 1195.30 | 1167.07 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-18 10:15:00 | 1220.90 | 1195.55 | 1167.33 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-06-30 13:15:00 | 1200.75 | 1206.31 | 1180.71 | SL hit qty=1.00 sl=1200.75 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-30 13:15:00 | 1200.75 | 1206.31 | 1180.71 | SL hit qty=1.00 sl=1200.75 alert=retest2 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 09:15:00 | 1174.60 | 1205.86 | 1180.86 | EMA400 retest candle locked |

### Cycle 4 — BUY (started 2025-09-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-29 11:15:00 | 1129.80 | 1110.78 | 1110.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-29 12:15:00 | 1133.70 | 1111.00 | 1110.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-16 09:15:00 | 1237.00 | 1261.01 | 1230.43 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-16 11:15:00 | 1236.80 | 1260.51 | 1230.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 11:15:00 | 1236.80 | 1260.51 | 1230.48 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-12-30 09:15:00 | 1240.40 | 1246.31 | 1230.21 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-30 10:15:00 | 1240.60 | 1246.25 | 1230.26 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2026-03-13 09:15:00 | 1228.60 | 1327.79 | 1312.24 | SL hit qty=1.00 sl=1228.60 alert=retest2 |
| Cross detected — sustain check pending | 2026-03-18 09:15:00 | 1242.50 | 1305.87 | 1302.22 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-18 10:15:00 | 1247.40 | 1305.29 | 1301.95 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2026-03-19 09:15:00 | 1228.60 | 1301.86 | 1300.31 | SL hit qty=1.00 sl=1228.60 alert=retest2 |
| Cross detected — sustain check pending | 2026-04-06 13:15:00 | 1238.70 | 1250.41 | 1270.83 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-06 14:15:00 | 1246.90 | 1250.38 | 1270.71 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2026-04-07 09:15:00 | 1228.60 | 1250.09 | 1270.36 | SL hit qty=1.00 sl=1228.60 alert=retest2 |
| Cross detected — sustain check pending | 2026-04-07 13:15:00 | 1240.60 | 1249.38 | 1269.60 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-07 14:15:00 | 1250.90 | 1249.40 | 1269.51 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-07 15:15:00 | 1250.10 | 1249.41 | 1269.41 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2026-04-08 09:15:00 | 1316.00 | 1250.07 | 1269.65 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-08 10:15:00 | 1326.70 | 1250.83 | 1269.93 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2026-04-17 13:15:00 | 1356.50 | 1285.46 | 1285.23 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-17 13:15:00 | 1356.50 | 1285.46 | 1285.23 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 5 — BUY (started 2026-04-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-17 13:15:00 | 1356.50 | 1285.46 | 1285.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-17 14:15:00 | 1357.90 | 1286.18 | 1285.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-28 10:15:00 | 1301.30 | 1310.05 | 1299.05 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-28 11:15:00 | 1296.40 | 1309.92 | 1299.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 11:15:00 | 1296.40 | 1309.92 | 1299.04 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2026-04-29 12:15:00 | 1304.40 | 1308.58 | 1298.78 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-04-29 13:15:00 | 1296.60 | 1308.46 | 1298.77 | ENTRY2 sustain failed after 60m |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2023-09-07 14:15:00 | 977.45 | 2023-10-23 14:15:00 | 962.00 | STOP_HIT | 1.00 | -1.58% |
| BUY | retest2 | 2023-10-26 15:15:00 | 972.05 | 2023-12-04 09:15:00 | 1117.86 | PARTIAL | 0.50 | 15.00% |
| BUY | retest2 | 2023-10-26 15:15:00 | 972.05 | 2024-04-29 10:15:00 | 1142.50 | STOP_HIT | 0.50 | 17.54% |
| BUY | retest2 | 2024-06-05 14:15:00 | 1185.00 | 2024-08-28 09:15:00 | 1175.00 | STOP_HIT | 1.00 | -0.84% |
| BUY | retest2 | 2024-06-06 14:15:00 | 1170.95 | 2024-09-04 10:15:00 | 1175.00 | STOP_HIT | 1.00 | 0.35% |
| BUY | retest2 | 2024-07-25 15:15:00 | 1176.00 | 2024-09-04 10:15:00 | 1175.00 | STOP_HIT | 1.00 | -0.09% |
| BUY | retest2 | 2024-07-26 13:15:00 | 1178.15 | 2025-04-23 10:15:00 | 1200.75 | STOP_HIT | 1.00 | 1.92% |
| BUY | retest2 | 2024-08-27 13:15:00 | 1183.80 | 2025-06-13 09:15:00 | 1200.75 | STOP_HIT | 1.00 | 1.43% |
| BUY | retest2 | 2024-09-02 10:15:00 | 1183.00 | 2025-06-30 13:15:00 | 1200.75 | STOP_HIT | 1.00 | 1.50% |
| BUY | retest2 | 2024-09-03 12:15:00 | 1182.65 | 2025-06-30 13:15:00 | 1200.75 | STOP_HIT | 1.00 | 1.53% |
| BUY | retest2 | 2024-09-05 14:15:00 | 1180.70 | 2026-03-13 09:15:00 | 1228.60 | STOP_HIT | 1.00 | 4.06% |
| BUY | retest2 | 2025-04-21 11:15:00 | 1233.10 | 2026-03-19 09:15:00 | 1228.60 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest2 | 2025-06-09 12:15:00 | 1221.20 | 2026-04-07 09:15:00 | 1228.60 | STOP_HIT | 1.00 | 0.61% |
| BUY | retest2 | 2025-06-17 10:15:00 | 1223.80 | 2026-04-17 13:15:00 | 1356.50 | STOP_HIT | 1.00 | 10.84% |
| BUY | retest2 | 2025-06-18 10:15:00 | 1220.90 | 2026-04-17 13:15:00 | 1356.50 | STOP_HIT | 1.00 | 11.11% |
