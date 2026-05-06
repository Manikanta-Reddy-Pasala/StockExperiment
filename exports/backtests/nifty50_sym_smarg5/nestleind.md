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
| ALERT3 | 7 |
| PENDING | 23 |
| PENDING_CANCEL | 4 |
| ENTRY1 | 2 |
| ENTRY2 | 17 |
| PARTIAL | 1 |
| TARGET_HIT | 0 |
| STOP_HIT | 14 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 15 (incl. partial bookings)
- **Trades open at end:** 5
- **Winners / losers:** 2 / 13
- **Target hits / Stop hits / Partials:** 0 / 14 / 1
- **Avg / median % per leg:** -0.35% / -2.56%
- **Sum % (uncompounded):** -5.28%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 9 | 2 | 22.2% | 0 | 8 | 1 | 1.36% | 12.2% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 9 | 2 | 22.2% | 0 | 8 | 1 | 1.36% | 12.2% |
| SELL (all) | 6 | 0 | 0.0% | 0 | 6 | 0 | -2.92% | -17.5% |
| SELL @ 2nd Alert (retest1) | 2 | 0 | 0.0% | 0 | 2 | 0 | -3.13% | -6.3% |
| SELL @ 3rd Alert (retest2) | 4 | 0 | 0.0% | 0 | 4 | 0 | -2.81% | -11.2% |
| retest1 (combined) | 2 | 0 | 0.0% | 0 | 2 | 0 | -3.13% | -6.3% |
| retest2 (combined) | 13 | 2 | 15.4% | 0 | 12 | 1 | 0.08% | 1.0% |

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

### Cycle 4 — SELL (started 2024-10-18 10:15:00)

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
| ALERT3_SKIP | 2025-02-24 11:15:00 | 1116.50 | 1107.74 | 1116.66 | max_alert3_locks_per_cycle=2 reached — end cycle |
| CROSSOVER_SKIP | 2025-04-09 09:15:00 | 1151.40 | 1115.71 | 1115.63 | HTF filter: close below htf_sma |
| CROSSOVER_SKIP | 2025-07-30 14:15:00 | 1115.85 | 1187.84 | 1187.97 | HTF filter: close above htf_sma |

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
| ALERT3_SKIP | 2026-04-07 13:15:00 | 1217.70 | 1222.46 | 1243.28 | max_alert3_locks_per_cycle=2 reached — end cycle |

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
| SELL | retest1 | 2025-01-06 10:15:00 | 1095.38 | 2025-01-09 10:15:00 | 1133.56 | STOP_HIT | 1.00 | -3.49% |
| SELL | retest1 | 2025-01-07 11:15:00 | 1102.90 | 2025-01-09 10:15:00 | 1133.56 | STOP_HIT | 1.00 | -2.78% |
| SELL | retest2 | 2025-01-14 10:15:00 | 1107.53 | 2025-01-31 10:15:00 | 1135.82 | STOP_HIT | 1.00 | -2.56% |
| SELL | retest2 | 2025-01-14 13:15:00 | 1107.53 | 2025-01-31 10:15:00 | 1135.82 | STOP_HIT | 1.00 | -2.56% |
| SELL | retest2 | 2025-01-21 15:15:00 | 1100.00 | 2025-01-31 10:15:00 | 1135.82 | STOP_HIT | 1.00 | -3.26% |
| SELL | retest2 | 2025-01-27 11:15:00 | 1104.07 | 2025-01-31 10:15:00 | 1135.82 | STOP_HIT | 1.00 | -2.88% |
| BUY | retest2 | 2025-10-10 10:15:00 | 1191.80 | 2026-03-23 10:15:00 | 1170.70 | STOP_HIT | 1.00 | -1.77% |
| BUY | retest2 | 2025-10-15 10:15:00 | 1209.90 | 2026-03-23 10:15:00 | 1170.70 | STOP_HIT | 1.00 | -3.24% |
| BUY | retest2 | 2026-03-20 12:15:00 | 1198.00 | 2026-03-23 10:15:00 | 1170.70 | STOP_HIT | 1.00 | -2.28% |
| BUY | retest2 | 2026-03-25 10:15:00 | 1204.90 | 2026-03-27 09:15:00 | 1201.50 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest2 | 2026-03-25 13:15:00 | 1214.30 | 2026-03-27 14:15:00 | 1201.50 | STOP_HIT | 1.00 | -1.05% |
| BUY | retest2 | 2026-03-27 11:15:00 | 1211.30 | 2026-03-30 14:15:00 | 1170.70 | STOP_HIT | 1.00 | -3.35% |
| BUY | retest2 | 2026-04-06 14:15:00 | 1217.00 | 2026-04-07 09:15:00 | 1201.50 | STOP_HIT | 1.00 | -1.27% |
