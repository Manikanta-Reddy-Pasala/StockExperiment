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
| CROSSOVER | 6 |
| ALERT1 | 6 |
| ALERT2 | 5 |
| ALERT2_SKIP | 4 |
| ALERT3 | 8 |
| PENDING | 24 |
| PENDING_CANCEL | 4 |
| ENTRY1 | 2 |
| ENTRY2 | 18 |
| PARTIAL | 4 |
| TARGET_HIT | 0 |
| STOP_HIT | 20 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 24 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 10 / 14
- **Target hits / Stop hits / Partials:** 0 / 20 / 4
- **Avg / median % per leg:** 3.96% / -0.28%
- **Sum % (uncompounded):** 94.99%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 24 | 10 | 41.7% | 0 | 20 | 4 | 3.96% | 95.0% |
| BUY @ 2nd Alert (retest1) | 2 | 0 | 0.0% | 0 | 2 | 0 | -2.39% | -4.8% |
| BUY @ 3rd Alert (retest2) | 22 | 10 | 45.5% | 0 | 18 | 4 | 4.53% | 99.8% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 2 | 0 | 0.0% | 0 | 2 | 0 | -2.39% | -4.8% |
| retest2 (combined) | 22 | 10 | 45.5% | 0 | 18 | 4 | 4.53% | 99.8% |

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
| CROSSOVER_SKIP | 2024-04-23 15:15:00 | 1249.38 | 1265.48 | 1265.48 | HTF filter: close above htf_sma |
| Stop hit — per-position SL triggered | 2024-06-25 12:15:00 | 1255.03 | 1252.19 | 1252.18 | Force close (CROSSOVER_FLIP) qty=0.50 alert=retest2 |

### Cycle 2 — BUY (started 2024-06-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-25 12:15:00 | 1255.03 | 1252.19 | 1252.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-25 14:15:00 | 1260.43 | 1252.31 | 1252.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-24 09:15:00 | 1264.50 | 1282.87 | 1271.79 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-24 09:15:00 | 1264.50 | 1282.87 | 1271.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-24 09:15:00 | 1264.50 | 1282.87 | 1271.79 | EMA400 retest candle locked |
| CROSSOVER_SKIP | 2024-08-07 10:15:00 | 1257.65 | 1263.71 | 1263.73 | HTF filter: close above htf_sma |

### Cycle 3 — BUY (started 2024-09-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-17 09:15:00 | 1280.75 | 1260.98 | 1260.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-17 11:15:00 | 1289.22 | 1261.50 | 1261.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-04 13:15:00 | 1307.25 | 1308.17 | 1289.23 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-07 14:15:00 | 1287.40 | 1307.42 | 1289.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-07 14:15:00 | 1287.40 | 1307.42 | 1289.59 | EMA400 retest candle locked |
| CROSSOVER_SKIP | 2024-10-18 10:15:00 | 1171.00 | 1277.63 | 1277.63 | slope filter: EMA200 not falling 0.50% over 350 bars |

### Cycle 4 — BUY (started 2025-04-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-09 09:15:00 | 1151.40 | 1115.71 | 1115.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-09 10:15:00 | 1169.47 | 1116.24 | 1115.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-02 12:15:00 | 1165.00 | 1165.30 | 1145.99 | EMA200 retest candle locked |
| Cross detected — sustain check pending | 2025-05-05 10:15:00 | 1173.70 | 1165.55 | 1146.59 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-05 11:15:00 | 1174.05 | 1165.64 | 1146.73 | BUY ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2025-05-06 09:15:00 | 1180.00 | 1165.85 | 1147.31 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-06 10:15:00 | 1181.25 | 1166.01 | 1147.48 | BUY ENTRY1 attempt 2/4 (retest1 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-09 09:15:00 | 1149.05 | 1166.50 | 1149.51 | EMA400 retest candle locked |
| Stop hit — per-position SL triggered | 2025-05-09 09:15:00 | 1149.51 | 1166.50 | 1149.51 | SL hit qty=1.00 sl=1149.51 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-05-09 09:15:00 | 1149.51 | 1166.50 | 1149.51 | SL hit qty=1.00 sl=1149.51 alert=retest1 |
| Cross detected — sustain check pending | 2025-05-09 14:15:00 | 1163.05 | 1166.04 | 1149.70 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-05-09 15:15:00 | 1160.50 | 1165.98 | 1149.75 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-05-12 09:15:00 | 1182.40 | 1166.15 | 1149.91 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-12 10:15:00 | 1179.50 | 1166.28 | 1150.06 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-06-20 09:15:00 | 1169.50 | 1190.61 | 1179.34 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-20 10:15:00 | 1170.90 | 1190.42 | 1179.29 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-07-24 12:15:00 | 1170.85 | 1209.30 | 1197.81 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-24 13:15:00 | 1169.35 | 1208.91 | 1197.67 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-07-25 09:15:00 | 1145.85 | 1207.31 | 1197.03 | SL hit qty=1.00 sl=1145.85 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-25 09:15:00 | 1145.85 | 1207.31 | 1197.03 | SL hit qty=1.00 sl=1145.85 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-25 09:15:00 | 1145.85 | 1207.31 | 1197.03 | SL hit qty=1.00 sl=1145.85 alert=retest2 |
| CROSSOVER_SKIP | 2025-07-30 14:15:00 | 1115.85 | 1187.84 | 1187.97 | slope filter: EMA200 not falling 0.50% over 350 bars |
| Cross detected — sustain check pending | 2025-08-18 10:15:00 | 1166.10 | 1146.46 | 1163.59 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-08-18 11:15:00 | 1160.70 | 1146.60 | 1163.57 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-08-18 12:15:00 | 1168.20 | 1146.82 | 1163.60 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-08-18 13:15:00 | 1153.10 | 1146.88 | 1163.54 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-08-19 15:15:00 | 1165.00 | 1146.97 | 1162.85 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-20 09:15:00 | 1171.60 | 1147.21 | 1162.89 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 1080m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 10:15:00 | 1163.20 | 1151.65 | 1164.07 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-08-26 11:15:00 | 1176.10 | 1152.99 | 1163.86 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-26 12:15:00 | 1172.60 | 1153.18 | 1163.91 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-08-26 15:15:00 | 1160.00 | 1153.56 | 1163.94 | SL hit qty=1.00 sl=1160.00 alert=retest2 |
| Cross detected — sustain check pending | 2025-09-01 12:15:00 | 1171.90 | 1154.77 | 1163.67 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-01 13:15:00 | 1172.10 | 1154.94 | 1163.72 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-09-10 09:15:00 | 1206.00 | 1171.03 | 1170.88 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-10 09:15:00 | 1206.00 | 1171.03 | 1170.88 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 5 — BUY (started 2025-09-10 09:15:00)

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
| CROSSOVER_SKIP | 2026-03-11 14:15:00 | 1233.70 | 1278.41 | 1278.50 | HTF filter: close above htf_sma |
| Cross detected — sustain check pending | 2026-03-20 11:15:00 | 1193.50 | 1251.22 | 1263.50 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-20 12:15:00 | 1198.00 | 1250.69 | 1263.17 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2026-03-23 10:15:00 | 1170.70 | 1247.59 | 1261.29 | SL hit qty=1.00 sl=1170.70 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-23 10:15:00 | 1170.70 | 1247.59 | 1261.29 | SL hit qty=1.00 sl=1170.70 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-23 10:15:00 | 1170.70 | 1247.59 | 1261.29 | SL hit qty=1.00 sl=1170.70 alert=retest2 |
| Cross detected — sustain check pending | 2026-03-24 13:15:00 | 1192.40 | 1240.80 | 1257.14 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-03-24 14:15:00 | 1183.00 | 1240.23 | 1256.77 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2026-03-25 09:15:00 | 1198.80 | 1239.29 | 1256.14 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-25 10:15:00 | 1204.90 | 1238.95 | 1255.88 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 11:15:00 | 1204.80 | 1238.61 | 1255.63 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2026-03-25 12:15:00 | 1209.00 | 1238.31 | 1255.39 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-25 13:15:00 | 1214.30 | 1238.07 | 1255.19 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2026-03-27 09:15:00 | 1201.50 | 1237.05 | 1254.42 | SL hit qty=1.00 sl=1201.50 alert=retest2 |
| Cross detected — sustain check pending | 2026-03-27 10:15:00 | 1211.50 | 1236.80 | 1254.20 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-27 11:15:00 | 1211.30 | 1236.54 | 1253.99 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2026-03-27 14:15:00 | 1201.50 | 1235.51 | 1253.21 | SL hit qty=1.00 sl=1201.50 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-30 14:15:00 | 1170.70 | 1231.95 | 1250.79 | SL hit qty=1.00 sl=1170.70 alert=retest2 |
| Cross detected — sustain check pending | 2026-04-06 13:15:00 | 1207.20 | 1223.11 | 1244.34 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-06 14:15:00 | 1217.00 | 1223.05 | 1244.20 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2026-04-07 09:15:00 | 1201.50 | 1222.84 | 1243.89 | SL hit qty=1.00 sl=1201.50 alert=retest2 |
| Cross detected — sustain check pending | 2026-04-07 11:15:00 | 1212.60 | 1222.56 | 1243.53 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-07 12:15:00 | 1217.60 | 1222.51 | 1243.40 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-07 13:15:00 | 1217.70 | 1222.46 | 1243.28 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2026-04-07 14:15:00 | 1221.90 | 1222.45 | 1243.17 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-07 15:15:00 | 1222.80 | 1222.46 | 1243.07 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2026-04-08 11:15:00 | 1214.50 | 1222.40 | 1242.73 | SL hit qty=1.00 sl=1214.50 alert=retest2 |
| Cross detected — sustain check pending | 2026-04-09 11:15:00 | 1220.00 | 1221.95 | 1241.80 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-09 12:15:00 | 1221.30 | 1221.94 | 1241.70 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2026-04-09 14:15:00 | 1229.50 | 1221.98 | 1241.52 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-09 15:15:00 | 1229.80 | 1222.05 | 1241.46 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Partial book — 50% qty @ 15%, trail SL → entry | 2026-04-22 09:15:00 | 1400.24 | 1240.91 | 1247.49 | Partial book 0.50 @ 15%; trail SL->entry alert=retest2 |
| Partial book — 50% qty @ 15%, trail SL → entry | 2026-04-22 09:15:00 | 1404.50 | 1240.91 | 1247.49 | Partial book 0.50 @ 15%; trail SL->entry alert=retest2 |
| Partial book — 50% qty @ 15%, trail SL → entry | 2026-04-22 09:15:00 | 1414.27 | 1240.91 | 1247.49 | Partial book 0.50 @ 15%; trail SL->entry alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-23 11:15:00 | 1410.50 | 1255.11 | 1254.46 | Force close (CROSSOVER_FLIP) qty=0.50 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-23 11:15:00 | 1410.50 | 1255.11 | 1254.46 | Force close (CROSSOVER_FLIP) qty=0.50 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-23 11:15:00 | 1410.50 | 1255.11 | 1254.46 | Force close (CROSSOVER_FLIP) qty=0.50 alert=retest2 |

### Cycle 6 — BUY (started 2026-04-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-23 11:15:00 | 1410.50 | 1255.11 | 1254.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-23 13:15:00 | 1412.80 | 1258.18 | 1256.01 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2023-10-04 10:15:00 | 1136.00 | 2023-12-28 11:15:00 | 1306.40 | PARTIAL | 0.50 | 15.00% |
| BUY | retest2 | 2023-10-04 10:15:00 | 1136.00 | 2024-06-25 12:15:00 | 1255.03 | STOP_HIT | 0.50 | 10.48% |
| BUY | retest1 | 2025-05-05 11:15:00 | 1174.05 | 2025-05-09 09:15:00 | 1149.51 | STOP_HIT | 1.00 | -2.09% |
| BUY | retest1 | 2025-05-06 10:15:00 | 1181.25 | 2025-05-09 09:15:00 | 1149.51 | STOP_HIT | 1.00 | -2.69% |
| BUY | retest2 | 2025-05-12 10:15:00 | 1179.50 | 2025-07-25 09:15:00 | 1145.85 | STOP_HIT | 1.00 | -2.85% |
| BUY | retest2 | 2025-06-20 10:15:00 | 1170.90 | 2025-07-25 09:15:00 | 1145.85 | STOP_HIT | 1.00 | -2.14% |
| BUY | retest2 | 2025-07-24 13:15:00 | 1169.35 | 2025-07-25 09:15:00 | 1145.85 | STOP_HIT | 1.00 | -2.01% |
| BUY | retest2 | 2025-08-20 09:15:00 | 1171.60 | 2025-08-26 15:15:00 | 1160.00 | STOP_HIT | 1.00 | -0.99% |
| BUY | retest2 | 2025-08-26 12:15:00 | 1172.60 | 2025-09-10 09:15:00 | 1206.00 | STOP_HIT | 1.00 | 2.85% |
| BUY | retest2 | 2025-09-01 13:15:00 | 1172.10 | 2025-09-10 09:15:00 | 1206.00 | STOP_HIT | 1.00 | 2.89% |
| BUY | retest2 | 2025-10-10 10:15:00 | 1191.80 | 2026-03-23 10:15:00 | 1170.70 | STOP_HIT | 1.00 | -1.77% |
| BUY | retest2 | 2025-10-15 10:15:00 | 1209.90 | 2026-03-23 10:15:00 | 1170.70 | STOP_HIT | 1.00 | -3.24% |
| BUY | retest2 | 2026-03-20 12:15:00 | 1198.00 | 2026-03-23 10:15:00 | 1170.70 | STOP_HIT | 1.00 | -2.28% |
| BUY | retest2 | 2026-03-25 10:15:00 | 1204.90 | 2026-03-27 09:15:00 | 1201.50 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest2 | 2026-03-25 13:15:00 | 1214.30 | 2026-03-27 14:15:00 | 1201.50 | STOP_HIT | 1.00 | -1.05% |
| BUY | retest2 | 2026-03-27 11:15:00 | 1211.30 | 2026-03-30 14:15:00 | 1170.70 | STOP_HIT | 1.00 | -3.35% |
| BUY | retest2 | 2026-04-06 14:15:00 | 1217.00 | 2026-04-07 09:15:00 | 1201.50 | STOP_HIT | 1.00 | -1.27% |
| BUY | retest2 | 2026-04-07 12:15:00 | 1217.60 | 2026-04-08 11:15:00 | 1214.50 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest2 | 2026-04-07 15:15:00 | 1222.80 | 2026-04-22 09:15:00 | 1400.24 | PARTIAL | 0.50 | 14.51% |
| BUY | retest2 | 2026-04-09 12:15:00 | 1221.30 | 2026-04-22 09:15:00 | 1404.50 | PARTIAL | 0.50 | 15.00% |
| BUY | retest2 | 2026-04-09 15:15:00 | 1229.80 | 2026-04-22 09:15:00 | 1414.27 | PARTIAL | 0.50 | 15.00% |
| BUY | retest2 | 2026-04-07 15:15:00 | 1222.80 | 2026-04-23 11:15:00 | 1410.50 | STOP_HIT | 0.50 | 15.35% |
| BUY | retest2 | 2026-04-09 12:15:00 | 1221.30 | 2026-04-23 11:15:00 | 1410.50 | STOP_HIT | 0.50 | 15.49% |
| BUY | retest2 | 2026-04-09 15:15:00 | 1229.80 | 2026-04-23 11:15:00 | 1410.50 | STOP_HIT | 0.50 | 14.69% |
