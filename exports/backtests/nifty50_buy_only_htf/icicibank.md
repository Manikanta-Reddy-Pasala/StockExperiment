# ICICIBANK (ICICIBANK.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-06-06 09:15:00 → 2026-05-06 15:30:00 (4997 bars)
- **Last close:** 1279.50
- **Strategy:** EMA 200/400 1H crossover (v2 BTC rules)
- **Target:** entry × (1 ± 30%)
- **Partial:** 50% qty booked @ 15%, trail SL → entry
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 3 attempts each at retest1 and retest2

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 8 |
| ALERT1 | 8 |
| ALERT2 | 8 |
| ALERT2_SKIP | 6 |
| ALERT3 | 10 |
| PENDING | 32 |
| PENDING_CANCEL | 9 |
| ENTRY1 | 2 |
| ENTRY2 | 21 |
| PARTIAL | 4 |
| TARGET_HIT | 2 |
| STOP_HIT | 20 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 26 (incl. partial bookings)
- **Trades open at end:** 1
- **Winners / losers:** 10 / 16
- **Target hits / Stop hits / Partials:** 2 / 20 / 4
- **Avg / median % per leg:** 4.24% / -1.23%
- **Sum % (uncompounded):** 110.18%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 10 | 8 | 80.0% | 2 | 4 | 4 | 13.14% | 131.4% |
| BUY @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -1.76% | -1.8% |
| BUY @ 3rd Alert (retest2) | 9 | 8 | 88.9% | 2 | 3 | 4 | 14.79% | 133.1% |
| SELL (all) | 16 | 2 | 12.5% | 0 | 16 | 0 | -1.32% | -21.2% |
| SELL @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -2.36% | -2.4% |
| SELL @ 3rd Alert (retest2) | 15 | 2 | 13.3% | 0 | 15 | 0 | -1.26% | -18.8% |
| retest1 (combined) | 2 | 0 | 0.0% | 0 | 2 | 0 | -2.06% | -4.1% |
| retest2 (combined) | 24 | 10 | 41.7% | 2 | 18 | 4 | 4.76% | 114.3% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-10-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-04 10:15:00 | 926.20 | 962.09 | 962.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-25 11:15:00 | 921.10 | 950.49 | 955.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-06 09:15:00 | 942.25 | 937.35 | 946.48 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-06 15:15:00 | 947.50 | 937.72 | 946.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-06 15:15:00 | 947.50 | 937.72 | 946.39 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2023-11-07 09:15:00 | 935.00 | 937.69 | 946.33 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-07 10:15:00 | 936.00 | 937.67 | 946.28 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2023-11-07 15:15:00 | 947.70 | 937.82 | 946.15 | SL hit qty=1.00 sl=947.70 alert=retest2 |
| Cross detected — sustain check pending | 2023-11-08 09:15:00 | 939.10 | 937.83 | 946.11 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-08 10:15:00 | 936.20 | 937.82 | 946.06 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2023-12-01 13:15:00 | 947.70 | 933.09 | 939.70 | SL hit qty=1.00 sl=947.70 alert=retest2 |

### Cycle 2 — BUY (started 2023-12-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-06 15:15:00 | 1002.70 | 945.74 | 945.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-08 09:15:00 | 1004.05 | 949.75 | 947.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-02 10:15:00 | 986.30 | 988.42 | 973.82 | EMA200 retest candle locked |
| Cross detected — sustain check pending | 2024-01-05 09:15:00 | 996.00 | 987.82 | 974.88 | ENTRY1 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2024-01-05 10:15:00 | 988.80 | 987.82 | 974.95 | ENTRY1 sustain failed after 60m |
| Cross detected — sustain check pending | 2024-01-05 12:15:00 | 994.80 | 987.92 | 975.13 | ENTRY1 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2024-01-05 13:15:00 | 986.60 | 987.91 | 975.18 | ENTRY1 sustain failed after 60m |
| Cross detected — sustain check pending | 2024-01-05 14:15:00 | 994.10 | 987.97 | 975.28 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-05 15:15:00 | 993.70 | 988.03 | 975.37 | BUY ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2024-01-08 10:15:00 | 993.65 | 988.13 | 975.55 | ENTRY1 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2024-01-08 11:15:00 | 990.75 | 988.16 | 975.62 | ENTRY1 sustain failed after 60m |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-09 15:15:00 | 977.75 | 988.04 | 976.24 | EMA400 retest candle locked |
| Stop hit — per-position SL triggered | 2024-01-09 15:15:00 | 976.24 | 988.04 | 976.24 | SL hit qty=1.00 sl=976.24 alert=retest1 |
| Cross detected — sustain check pending | 2024-01-10 09:15:00 | 987.55 | 988.04 | 976.30 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-10 10:15:00 | 990.10 | 988.06 | 976.36 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2024-01-17 13:15:00 | 986.10 | 991.67 | 980.34 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2024-01-17 14:15:00 | 981.40 | 991.56 | 980.34 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2024-01-18 09:15:00 | 983.70 | 991.37 | 980.35 | ENTRY2 cross detected — sustain check pending (15m) |
| Stop hit — per-position SL triggered | 2024-01-18 09:15:00 | 975.55 | 991.37 | 980.35 | SL hit qty=1.00 sl=975.55 alert=retest2 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-18 10:15:00 | 989.30 | 991.35 | 980.40 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2024-01-18 13:15:00 | 984.30 | 991.19 | 980.48 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-18 14:15:00 | 986.15 | 991.14 | 980.51 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Partial book — 50% qty @ 15%, trail SL → entry | 2024-04-29 10:15:00 | 1134.07 | 1084.72 | 1067.06 | Partial book 0.50 @ 15%; trail SL->entry alert=retest2 |
| Partial book — 50% qty @ 15%, trail SL → entry | 2024-04-29 11:15:00 | 1137.69 | 1085.29 | 1067.43 | Partial book 0.50 @ 15%; trail SL->entry alert=retest2 |
| Target hit — 30% from entry | 2024-09-18 11:15:00 | 1282.00 | 1224.19 | 1205.14 | Target hit (30%) qty=0.50 alert=retest2 |
| Target hit — 30% from entry | 2024-09-18 12:15:00 | 1286.09 | 1224.85 | 1205.57 | Target hit (30%) qty=0.50 alert=retest2 |

### Cycle 3 — SELL (started 2025-01-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-15 09:15:00 | 1237.50 | 1279.88 | 1279.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-17 09:15:00 | 1226.00 | 1274.94 | 1277.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-28 09:15:00 | 1256.25 | 1252.02 | 1263.98 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-28 11:15:00 | 1262.85 | 1252.15 | 1263.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-28 11:15:00 | 1262.85 | 1252.15 | 1263.93 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-01-28 14:15:00 | 1245.90 | 1252.14 | 1263.74 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-28 15:15:00 | 1246.75 | 1252.09 | 1263.66 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-01-31 09:15:00 | 1243.15 | 1251.88 | 1262.71 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-31 10:15:00 | 1241.50 | 1251.78 | 1262.61 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-02-03 09:15:00 | 1239.45 | 1251.49 | 1262.14 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-03 10:15:00 | 1247.10 | 1251.45 | 1262.07 | SELL ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-02-04 09:15:00 | 1264.00 | 1251.85 | 1261.96 | SL hit qty=1.00 sl=1264.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-02-04 09:15:00 | 1264.00 | 1251.85 | 1261.96 | SL hit qty=1.00 sl=1264.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-02-04 09:15:00 | 1264.00 | 1251.85 | 1261.96 | SL hit qty=1.00 sl=1264.00 alert=retest2 |
| Cross detected — sustain check pending | 2025-02-07 13:15:00 | 1250.80 | 1255.34 | 1262.64 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-02-07 14:15:00 | 1257.80 | 1255.36 | 1262.61 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-02-11 12:15:00 | 1249.60 | 1255.44 | 1262.23 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-02-11 13:15:00 | 1252.35 | 1255.41 | 1262.18 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-02-12 09:15:00 | 1249.70 | 1255.29 | 1262.02 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-12 10:15:00 | 1245.70 | 1255.20 | 1261.94 | SELL ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-13 09:15:00 | 1262.95 | 1255.13 | 1261.71 | EMA400 retest candle locked |
| Stop hit — per-position SL triggered | 2025-02-13 09:15:00 | 1264.00 | 1255.13 | 1261.71 | SL hit qty=1.00 sl=1264.00 alert=retest2 |
| Cross detected — sustain check pending | 2025-02-13 13:15:00 | 1249.15 | 1255.14 | 1261.58 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-13 14:15:00 | 1248.85 | 1255.08 | 1261.52 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-02-14 11:15:00 | 1246.00 | 1254.92 | 1261.31 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-02-14 12:15:00 | 1252.20 | 1254.89 | 1261.26 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-02-17 09:15:00 | 1236.00 | 1254.77 | 1261.08 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-17 10:15:00 | 1237.70 | 1254.60 | 1260.96 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-02-18 09:15:00 | 1244.80 | 1254.09 | 1260.51 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-18 10:15:00 | 1238.00 | 1253.93 | 1260.40 | SELL ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-02-20 11:15:00 | 1247.65 | 1253.52 | 1259.71 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-02-20 12:15:00 | 1250.55 | 1253.49 | 1259.67 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-02-20 13:15:00 | 1249.30 | 1253.45 | 1259.62 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-02-20 14:15:00 | 1250.65 | 1253.42 | 1259.57 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-02-20 15:15:00 | 1249.95 | 1253.38 | 1259.52 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-21 09:15:00 | 1227.00 | 1253.12 | 1259.36 | SELL ENTRY2 attempt 4/4 (retest2 break sustained 1080m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-11 13:15:00 | 1247.15 | 1234.34 | 1246.01 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-03-12 11:15:00 | 1240.25 | 1234.84 | 1245.98 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-12 12:15:00 | 1239.45 | 1234.89 | 1245.94 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-03-13 09:15:00 | 1247.95 | 1235.35 | 1245.96 | SL hit qty=1.00 sl=1247.95 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-03-17 09:15:00 | 1266.00 | 1236.59 | 1246.23 | SL hit qty=1.00 sl=1266.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-03-17 09:15:00 | 1266.00 | 1236.59 | 1246.23 | SL hit qty=1.00 sl=1266.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-03-17 09:15:00 | 1266.00 | 1236.59 | 1246.23 | SL hit qty=1.00 sl=1266.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-03-17 09:15:00 | 1266.00 | 1236.59 | 1246.23 | SL hit qty=1.00 sl=1266.00 alert=retest2 |

### Cycle 4 — BUY (started 2025-03-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-21 11:15:00 | 1341.20 | 1254.86 | 1254.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-24 09:15:00 | 1358.20 | 1259.24 | 1256.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-07 09:15:00 | 1287.85 | 1295.44 | 1278.62 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-07 10:15:00 | 1276.10 | 1295.25 | 1278.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-07 10:15:00 | 1276.10 | 1295.25 | 1278.61 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-04-08 09:15:00 | 1295.80 | 1294.52 | 1278.73 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-08 10:15:00 | 1296.40 | 1294.54 | 1278.82 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-04-09 11:15:00 | 1291.10 | 1294.69 | 1279.52 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-09 12:15:00 | 1296.75 | 1294.71 | 1279.60 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Partial book — 50% qty @ 15%, trail SL → entry | 2025-07-25 09:15:00 | 1490.86 | 1439.83 | 1423.74 | Partial book 0.50 @ 15%; trail SL->entry alert=retest2 |
| Partial book — 50% qty @ 15%, trail SL → entry | 2025-07-25 09:15:00 | 1491.26 | 1439.83 | 1423.74 | Partial book 0.50 @ 15%; trail SL->entry alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-03 12:15:00 | 1391.20 | 1429.48 | 1429.51 | Force close (CROSSOVER_FLIP) qty=0.50 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-03 12:15:00 | 1391.20 | 1429.48 | 1429.51 | Force close (CROSSOVER_FLIP) qty=0.50 alert=retest2 |

### Cycle 5 — SELL (started 2025-09-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-03 12:15:00 | 1391.20 | 1429.48 | 1429.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-24 09:15:00 | 1381.20 | 1416.12 | 1421.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-15 09:15:00 | 1389.80 | 1388.78 | 1401.94 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-16 09:15:00 | 1402.60 | 1389.30 | 1401.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 09:15:00 | 1402.60 | 1389.30 | 1401.75 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-10-20 14:15:00 | 1390.00 | 1393.73 | 1402.96 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-20 15:15:00 | 1393.00 | 1393.73 | 1402.91 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2026-01-06 09:15:00 | 1410.00 | 1362.22 | 1370.27 | SL hit qty=1.00 sl=1410.00 alert=retest2 |

### Cycle 6 — BUY (started 2026-01-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-12 13:15:00 | 1416.90 | 1377.35 | 1377.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-13 09:15:00 | 1422.70 | 1378.48 | 1377.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-19 09:15:00 | 1362.30 | 1386.10 | 1381.95 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-19 09:15:00 | 1362.30 | 1386.10 | 1381.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 09:15:00 | 1362.30 | 1386.10 | 1381.95 | EMA400 retest candle locked |

### Cycle 7 — SELL (started 2026-01-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-23 13:15:00 | 1347.60 | 1378.54 | 1378.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-23 14:15:00 | 1344.20 | 1378.19 | 1378.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-29 13:15:00 | 1378.00 | 1375.34 | 1376.86 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-29 13:15:00 | 1378.00 | 1375.34 | 1376.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 13:15:00 | 1378.00 | 1375.34 | 1376.86 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2026-01-30 10:15:00 | 1364.90 | 1375.41 | 1376.86 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-30 11:15:00 | 1360.30 | 1375.26 | 1376.78 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2026-02-03 09:15:00 | 1378.50 | 1372.87 | 1375.46 | SL hit qty=1.00 sl=1378.50 alert=retest2 |
| CROSSOVER_SKIP | 2026-02-05 14:15:00 | 1398.30 | 1377.89 | 1377.84 | HTF filter: close below htf_sma |
| Cross detected — sustain check pending | 2026-03-02 09:15:00 | 1367.10 | 1392.98 | 1387.88 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-02 10:15:00 | 1365.40 | 1392.70 | 1387.77 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2026-03-02 15:15:00 | 1378.50 | 1391.63 | 1387.35 | SL hit qty=1.00 sl=1378.50 alert=retest2 |
| Cross detected — sustain check pending | 2026-03-04 09:15:00 | 1350.50 | 1391.22 | 1387.16 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-04 10:15:00 | 1352.30 | 1390.83 | 1386.99 | SELL ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2026-03-06 14:15:00 | 1312.00 | 1383.42 | 1383.48 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 8 — SELL (started 2026-03-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-06 14:15:00 | 1312.00 | 1383.42 | 1383.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-09 09:15:00 | 1271.90 | 1381.64 | 1382.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 09:15:00 | 1302.40 | 1281.44 | 1317.95 | EMA200 retest candle locked |
| Cross detected — sustain check pending | 2026-04-09 11:15:00 | 1290.30 | 1283.33 | 1317.32 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-09 12:15:00 | 1286.30 | 1283.36 | 1317.16 | SELL ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-10 09:15:00 | 1321.00 | 1283.72 | 1316.67 | EMA400 retest candle locked |
| Stop hit — per-position SL triggered | 2026-04-10 09:15:00 | 1316.67 | 1283.72 | 1316.67 | SL hit qty=1.00 sl=1316.67 alert=retest1 |
| Cross detected — sustain check pending | 2026-04-29 13:15:00 | 1284.10 | 1313.14 | 1323.67 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-29 14:15:00 | 1280.10 | 1312.81 | 1323.45 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2023-11-07 10:15:00 | 936.00 | 2023-11-07 15:15:00 | 947.70 | STOP_HIT | 1.00 | -1.25% |
| SELL | retest2 | 2023-11-08 10:15:00 | 936.20 | 2023-12-01 13:15:00 | 947.70 | STOP_HIT | 1.00 | -1.23% |
| BUY | retest1 | 2024-01-05 15:15:00 | 993.70 | 2024-01-09 15:15:00 | 976.24 | STOP_HIT | 1.00 | -1.76% |
| BUY | retest2 | 2024-01-10 10:15:00 | 990.10 | 2024-01-18 09:15:00 | 975.55 | STOP_HIT | 1.00 | -1.47% |
| BUY | retest2 | 2024-01-18 10:15:00 | 989.30 | 2024-04-29 10:15:00 | 1134.07 | PARTIAL | 0.50 | 14.63% |
| BUY | retest2 | 2024-01-18 14:15:00 | 986.15 | 2024-04-29 11:15:00 | 1137.69 | PARTIAL | 0.50 | 15.37% |
| BUY | retest2 | 2024-01-18 10:15:00 | 989.30 | 2024-09-18 11:15:00 | 1282.00 | TARGET_HIT | 0.50 | 29.59% |
| BUY | retest2 | 2024-01-18 14:15:00 | 986.15 | 2024-09-18 12:15:00 | 1286.09 | TARGET_HIT | 0.50 | 30.42% |
| SELL | retest2 | 2025-01-28 15:15:00 | 1246.75 | 2025-02-04 09:15:00 | 1264.00 | STOP_HIT | 1.00 | -1.38% |
| SELL | retest2 | 2025-01-31 10:15:00 | 1241.50 | 2025-02-04 09:15:00 | 1264.00 | STOP_HIT | 1.00 | -1.81% |
| SELL | retest2 | 2025-02-03 10:15:00 | 1247.10 | 2025-02-04 09:15:00 | 1264.00 | STOP_HIT | 1.00 | -1.36% |
| SELL | retest2 | 2025-02-12 10:15:00 | 1245.70 | 2025-02-13 09:15:00 | 1264.00 | STOP_HIT | 1.00 | -1.47% |
| SELL | retest2 | 2025-02-13 14:15:00 | 1248.85 | 2025-03-13 09:15:00 | 1247.95 | STOP_HIT | 1.00 | 0.07% |
| SELL | retest2 | 2025-02-17 10:15:00 | 1237.70 | 2025-03-17 09:15:00 | 1266.00 | STOP_HIT | 1.00 | -2.29% |
| SELL | retest2 | 2025-02-18 10:15:00 | 1238.00 | 2025-03-17 09:15:00 | 1266.00 | STOP_HIT | 1.00 | -2.26% |
| SELL | retest2 | 2025-02-21 09:15:00 | 1227.00 | 2025-03-17 09:15:00 | 1266.00 | STOP_HIT | 1.00 | -3.18% |
| SELL | retest2 | 2025-03-12 12:15:00 | 1239.45 | 2025-03-17 09:15:00 | 1266.00 | STOP_HIT | 1.00 | -2.14% |
| BUY | retest2 | 2025-04-08 10:15:00 | 1296.40 | 2025-07-25 09:15:00 | 1490.86 | PARTIAL | 0.50 | 15.00% |
| BUY | retest2 | 2025-04-09 12:15:00 | 1296.75 | 2025-07-25 09:15:00 | 1491.26 | PARTIAL | 0.50 | 15.00% |
| BUY | retest2 | 2025-04-08 10:15:00 | 1296.40 | 2025-09-03 12:15:00 | 1391.20 | STOP_HIT | 0.50 | 7.31% |
| BUY | retest2 | 2025-04-09 12:15:00 | 1296.75 | 2025-09-03 12:15:00 | 1391.20 | STOP_HIT | 0.50 | 7.28% |
| SELL | retest2 | 2025-10-20 15:15:00 | 1393.00 | 2026-01-06 09:15:00 | 1410.00 | STOP_HIT | 1.00 | -1.22% |
| SELL | retest2 | 2026-01-30 11:15:00 | 1360.30 | 2026-02-03 09:15:00 | 1378.50 | STOP_HIT | 1.00 | -1.34% |
| SELL | retest2 | 2026-03-02 10:15:00 | 1365.40 | 2026-03-02 15:15:00 | 1378.50 | STOP_HIT | 1.00 | -0.96% |
| SELL | retest2 | 2026-03-04 10:15:00 | 1352.30 | 2026-03-06 14:15:00 | 1312.00 | STOP_HIT | 1.00 | 2.98% |
| SELL | retest1 | 2026-04-09 12:15:00 | 1286.30 | 2026-04-10 09:15:00 | 1316.67 | STOP_HIT | 1.00 | -2.36% |
