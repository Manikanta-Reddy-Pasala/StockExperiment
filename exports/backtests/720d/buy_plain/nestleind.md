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
| ALERT2 | 3 |
| ALERT2_SKIP | 2 |
| ALERT3 | 6 |
| PENDING | 22 |
| PENDING_CANCEL | 2 |
| ENTRY1 | 2 |
| ENTRY2 | 18 |
| PARTIAL | 4 |
| TARGET_HIT | 0 |
| STOP_HIT | 20 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 24 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 9 / 15
- **Target hits / Stop hits / Partials:** 0 / 20 / 4
- **Avg / median % per leg:** 3.78% / -1.20%
- **Sum % (uncompounded):** 90.84%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 24 | 9 | 37.5% | 0 | 20 | 4 | 3.78% | 90.8% |
| BUY @ 2nd Alert (retest1) | 2 | 0 | 0.0% | 0 | 2 | 0 | -2.43% | -4.9% |
| BUY @ 3rd Alert (retest2) | 22 | 9 | 40.9% | 0 | 18 | 4 | 4.35% | 95.7% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 2 | 0 | 0.0% | 0 | 2 | 0 | -2.43% | -4.9% |
| retest2 (combined) | 22 | 9 | 40.9% | 0 | 18 | 4 | 4.35% | 95.7% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-09-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-13 13:15:00 | 1270.65 | 1260.11 | 1260.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-16 14:15:00 | 1274.50 | 1260.71 | 1260.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-04 13:15:00 | 1307.10 | 1308.16 | 1288.93 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-07 14:15:00 | 1287.50 | 1307.48 | 1289.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-07 14:15:00 | 1287.50 | 1307.48 | 1289.34 | EMA400 retest candle locked |

### Cycle 2 — BUY (started 2025-04-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-09 10:15:00 | 1169.47 | 1116.34 | 1116.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-09 11:15:00 | 1176.85 | 1116.95 | 1116.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-02 12:15:00 | 1165.00 | 1165.27 | 1146.09 | EMA200 retest candle locked |
| Cross detected — sustain check pending | 2025-05-05 10:15:00 | 1174.20 | 1165.53 | 1146.70 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-05 11:15:00 | 1174.00 | 1165.62 | 1146.84 | BUY ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2025-05-06 09:15:00 | 1180.00 | 1165.85 | 1147.42 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-06 10:15:00 | 1181.25 | 1166.00 | 1147.59 | BUY ENTRY1 attempt 2/4 (retest1 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-09 09:15:00 | 1149.05 | 1166.46 | 1149.59 | EMA400 retest candle locked |
| Stop hit — per-position SL triggered | 2025-05-09 09:15:00 | 1149.05 | 1166.46 | 1149.59 | SL hit (close<ema400) qty=1.00 sl=1149.59 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-05-09 09:15:00 | 1149.05 | 1166.46 | 1149.59 | SL hit (close<ema400) qty=1.00 sl=1149.59 alert=retest1 |
| Cross detected — sustain check pending | 2025-05-09 14:15:00 | 1163.05 | 1166.00 | 1149.77 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-05-09 15:15:00 | 1160.50 | 1165.94 | 1149.83 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-05-12 09:15:00 | 1182.40 | 1166.11 | 1149.99 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-12 10:15:00 | 1179.50 | 1166.24 | 1150.14 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-06-19 11:15:00 | 1165.95 | 1191.84 | 1179.66 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-19 12:15:00 | 1164.25 | 1191.56 | 1179.59 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-06-20 09:15:00 | 1169.50 | 1190.42 | 1179.25 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-20 10:15:00 | 1170.90 | 1190.23 | 1179.21 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-07-24 12:15:00 | 1170.85 | 1209.28 | 1197.78 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-24 13:15:00 | 1169.35 | 1208.88 | 1197.64 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 14:15:00 | 1163.80 | 1208.43 | 1197.47 | EMA400 retest candle locked |
| Stop hit — per-position SL triggered | 2025-07-25 09:15:00 | 1142.95 | 1207.29 | 1197.00 | SL hit (close<static) qty=1.00 sl=1145.85 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-25 09:15:00 | 1142.95 | 1207.29 | 1197.00 | SL hit (close<static) qty=1.00 sl=1145.85 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-25 09:15:00 | 1142.95 | 1207.29 | 1197.00 | SL hit (close<static) qty=1.00 sl=1145.85 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-25 09:15:00 | 1142.95 | 1207.29 | 1197.00 | SL hit (close<static) qty=1.00 sl=1145.85 alert=retest2 |
| Cross detected — sustain check pending | 2025-08-20 09:15:00 | 1171.60 | 1147.22 | 1162.88 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-20 10:15:00 | 1173.70 | 1147.48 | 1162.94 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-08-22 11:15:00 | 1170.20 | 1151.83 | 1164.08 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-08-22 12:15:00 | 1165.50 | 1151.97 | 1164.09 | ENTRY2 sustain failed after 60m |
| Stop hit — per-position SL triggered | 2025-08-22 14:15:00 | 1160.40 | 1152.20 | 1164.09 | SL hit (close<static) qty=1.00 sl=1161.50 alert=retest2 |
| Cross detected — sustain check pending | 2025-08-26 10:15:00 | 1170.20 | 1152.74 | 1163.78 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-26 11:15:00 | 1176.50 | 1152.98 | 1163.85 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-08-26 15:15:00 | 1159.20 | 1153.53 | 1163.91 | SL hit (close<static) qty=1.00 sl=1161.50 alert=retest2 |
| Cross detected — sustain check pending | 2025-09-01 12:15:00 | 1171.90 | 1154.75 | 1163.65 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-01 13:15:00 | 1172.10 | 1154.92 | 1163.69 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-09-10 09:15:00 | 1206.00 | 1171.03 | 1170.87 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 3 — BUY (started 2025-09-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-10 09:15:00 | 1206.00 | 1171.03 | 1170.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-10 11:15:00 | 1216.20 | 1171.86 | 1171.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-22 14:15:00 | 1186.50 | 1187.13 | 1180.37 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-23 09:15:00 | 1171.90 | 1187.00 | 1180.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 09:15:00 | 1171.90 | 1187.00 | 1180.38 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-10-10 12:15:00 | 1195.00 | 1178.43 | 1177.25 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-10 13:15:00 | 1195.90 | 1178.60 | 1177.34 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-10-15 09:15:00 | 1197.90 | 1179.64 | 1177.99 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-15 10:15:00 | 1209.90 | 1179.94 | 1178.15 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2026-03-20 12:15:00 | 1198.30 | 1250.71 | 1263.40 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-20 13:15:00 | 1199.00 | 1250.20 | 1263.08 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2026-03-23 13:15:00 | 1169.70 | 1245.45 | 1260.23 | SL hit (close<static) qty=1.00 sl=1170.70 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-23 13:15:00 | 1169.70 | 1245.45 | 1260.23 | SL hit (close<static) qty=1.00 sl=1170.70 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-23 13:15:00 | 1169.70 | 1245.45 | 1260.23 | SL hit (close<static) qty=1.00 sl=1170.70 alert=retest2 |
| Cross detected — sustain check pending | 2026-03-25 09:15:00 | 1198.80 | 1239.35 | 1256.37 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-25 10:15:00 | 1204.90 | 1239.01 | 1256.12 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 11:15:00 | 1204.80 | 1238.67 | 1255.86 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2026-03-25 12:15:00 | 1209.00 | 1238.37 | 1255.63 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-25 13:15:00 | 1214.30 | 1238.13 | 1255.42 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2026-03-27 10:15:00 | 1211.50 | 1236.85 | 1254.43 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-27 11:15:00 | 1211.30 | 1236.60 | 1254.22 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2026-03-27 14:15:00 | 1190.50 | 1235.56 | 1253.43 | SL hit (close<static) qty=1.00 sl=1201.50 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-27 14:15:00 | 1190.50 | 1235.56 | 1253.43 | SL hit (close<static) qty=1.00 sl=1201.50 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-02 09:15:00 | 1164.20 | 1227.26 | 1247.75 | SL hit (close<static) qty=1.00 sl=1170.70 alert=retest2 |
| Cross detected — sustain check pending | 2026-04-06 13:15:00 | 1207.20 | 1223.19 | 1244.56 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-06 14:15:00 | 1217.00 | 1223.13 | 1244.42 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2026-04-07 11:15:00 | 1212.30 | 1222.61 | 1243.73 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-07 12:15:00 | 1217.60 | 1222.56 | 1243.60 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-07 13:15:00 | 1217.70 | 1222.51 | 1243.47 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2026-04-07 14:15:00 | 1221.90 | 1222.51 | 1243.37 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-07 15:15:00 | 1220.00 | 1222.48 | 1243.25 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2026-04-08 11:15:00 | 1210.30 | 1222.43 | 1242.92 | SL hit (close<static) qty=1.00 sl=1214.50 alert=retest2 |
| Cross detected — sustain check pending | 2026-04-09 11:15:00 | 1220.10 | 1221.97 | 1241.98 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-09 12:15:00 | 1221.30 | 1221.97 | 1241.87 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2026-04-09 14:15:00 | 1229.50 | 1222.00 | 1241.69 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-09 15:15:00 | 1230.00 | 1222.08 | 1241.64 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Partial book — 50% qty @ 15%, trail SL → entry | 2026-04-22 09:15:00 | 1399.55 | 1242.93 | 1248.47 | Partial book 0.50 @ 15%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty @ 15%, trail SL → entry | 2026-04-22 09:15:00 | 1400.24 | 1242.93 | 1248.47 | Partial book 0.50 @ 15%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty @ 15%, trail SL → entry | 2026-04-22 09:15:00 | 1404.49 | 1242.93 | 1248.47 | Partial book 0.50 @ 15%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty @ 15%, trail SL → entry | 2026-04-22 09:15:00 | 1414.50 | 1242.93 | 1248.47 | Partial book 0.50 @ 15%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-23 09:15:00 | 1408.00 | 1253.92 | 1253.87 | Force close (CROSSOVER_FLIP) qty=0.50 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-23 09:15:00 | 1408.00 | 1253.92 | 1253.87 | Force close (CROSSOVER_FLIP) qty=0.50 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-23 09:15:00 | 1408.00 | 1253.92 | 1253.87 | Force close (CROSSOVER_FLIP) qty=0.50 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-23 09:15:00 | 1408.00 | 1253.92 | 1253.87 | Force close (CROSSOVER_FLIP) qty=0.50 alert=retest2 |

### Cycle 4 — BUY (started 2026-04-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-23 09:15:00 | 1408.00 | 1253.92 | 1253.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-24 09:15:00 | 1416.30 | 1264.47 | 1259.24 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-05-05 11:15:00 | 1174.00 | 2025-05-09 09:15:00 | 1149.05 | STOP_HIT | 1.00 | -2.13% |
| BUY | retest1 | 2025-05-06 10:15:00 | 1181.25 | 2025-05-09 09:15:00 | 1149.05 | STOP_HIT | 1.00 | -2.73% |
| BUY | retest2 | 2025-05-12 10:15:00 | 1179.50 | 2025-07-25 09:15:00 | 1142.95 | STOP_HIT | 1.00 | -3.10% |
| BUY | retest2 | 2025-06-19 12:15:00 | 1164.25 | 2025-07-25 09:15:00 | 1142.95 | STOP_HIT | 1.00 | -1.83% |
| BUY | retest2 | 2025-06-20 10:15:00 | 1170.90 | 2025-07-25 09:15:00 | 1142.95 | STOP_HIT | 1.00 | -2.39% |
| BUY | retest2 | 2025-07-24 13:15:00 | 1169.35 | 2025-07-25 09:15:00 | 1142.95 | STOP_HIT | 1.00 | -2.26% |
| BUY | retest2 | 2025-08-20 10:15:00 | 1173.70 | 2025-08-22 14:15:00 | 1160.40 | STOP_HIT | 1.00 | -1.13% |
| BUY | retest2 | 2025-08-26 11:15:00 | 1176.50 | 2025-08-26 15:15:00 | 1159.20 | STOP_HIT | 1.00 | -1.47% |
| BUY | retest2 | 2025-09-01 13:15:00 | 1172.10 | 2025-09-10 09:15:00 | 1206.00 | STOP_HIT | 1.00 | 2.89% |
| BUY | retest2 | 2025-10-10 13:15:00 | 1195.90 | 2026-03-23 13:15:00 | 1169.70 | STOP_HIT | 1.00 | -2.19% |
| BUY | retest2 | 2025-10-15 10:15:00 | 1209.90 | 2026-03-23 13:15:00 | 1169.70 | STOP_HIT | 1.00 | -3.32% |
| BUY | retest2 | 2026-03-20 13:15:00 | 1199.00 | 2026-03-23 13:15:00 | 1169.70 | STOP_HIT | 1.00 | -2.44% |
| BUY | retest2 | 2026-03-25 10:15:00 | 1204.90 | 2026-03-27 14:15:00 | 1190.50 | STOP_HIT | 1.00 | -1.20% |
| BUY | retest2 | 2026-03-25 13:15:00 | 1214.30 | 2026-03-27 14:15:00 | 1190.50 | STOP_HIT | 1.00 | -1.96% |
| BUY | retest2 | 2026-03-27 11:15:00 | 1211.30 | 2026-04-02 09:15:00 | 1164.20 | STOP_HIT | 1.00 | -3.89% |
| BUY | retest2 | 2026-04-06 14:15:00 | 1217.00 | 2026-04-08 11:15:00 | 1210.30 | STOP_HIT | 1.00 | -0.55% |
| BUY | retest2 | 2026-04-07 12:15:00 | 1217.60 | 2026-04-22 09:15:00 | 1399.55 | PARTIAL | 0.50 | 14.94% |
| BUY | retest2 | 2026-04-07 15:15:00 | 1220.00 | 2026-04-22 09:15:00 | 1400.24 | PARTIAL | 0.50 | 14.77% |
| BUY | retest2 | 2026-04-09 12:15:00 | 1221.30 | 2026-04-22 09:15:00 | 1404.49 | PARTIAL | 0.50 | 15.00% |
| BUY | retest2 | 2026-04-09 15:15:00 | 1230.00 | 2026-04-22 09:15:00 | 1414.50 | PARTIAL | 0.50 | 15.00% |
| BUY | retest2 | 2026-04-07 12:15:00 | 1217.60 | 2026-04-23 09:15:00 | 1408.00 | STOP_HIT | 0.50 | 15.64% |
| BUY | retest2 | 2026-04-07 15:15:00 | 1220.00 | 2026-04-23 09:15:00 | 1408.00 | STOP_HIT | 0.50 | 15.41% |
| BUY | retest2 | 2026-04-09 12:15:00 | 1221.30 | 2026-04-23 09:15:00 | 1408.00 | STOP_HIT | 0.50 | 15.29% |
| BUY | retest2 | 2026-04-09 15:15:00 | 1230.00 | 2026-04-23 09:15:00 | 1408.00 | STOP_HIT | 0.50 | 14.47% |
