# ICICIBANK (ICICIBANK)

## Backtest Summary

- **Window:** 2023-06-08 09:15:00 → 2026-05-08 15:30:00 (4997 bars)
- **Last close:** 1264.80
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 30%)
- **Partial:** 50% qty booked @ 15% (ENTRY1) / 15% (ENTRY2), trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 9 |
| ALERT1 | 9 |
| ALERT2 | 9 |
| ALERT2_SKIP | 7 |
| ALERT3 | 11 |
| PENDING | 36 |
| PENDING_CANCEL | 12 |
| ENTRY1 | 2 |
| ENTRY2 | 22 |
| PARTIAL | 5 |
| TARGET_HIT | 0 |
| STOP_HIT | 23 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 28 (incl. partial bookings)
- **Trades open at end:** 1
- **Winners / losers:** 10 / 18
- **Target hits / Stop hits / Partials:** 0 / 23 / 5
- **Avg / median % per leg:** 3.54% / -1.21%
- **Sum % (uncompounded):** 99.21%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 14 | 10 | 71.4% | 0 | 9 | 5 | 9.34% | 130.7% |
| BUY @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -1.43% | -1.4% |
| BUY @ 3rd Alert (retest2) | 13 | 10 | 76.9% | 0 | 8 | 5 | 10.17% | 132.2% |
| SELL (all) | 14 | 0 | 0.0% | 0 | 14 | 0 | -2.25% | -31.5% |
| SELL @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -2.70% | -2.7% |
| SELL @ 3rd Alert (retest2) | 13 | 0 | 0.0% | 0 | 13 | 0 | -2.22% | -28.8% |
| retest1 (combined) | 2 | 0 | 0.0% | 0 | 2 | 0 | -2.06% | -4.1% |
| retest2 (combined) | 26 | 10 | 38.5% | 0 | 21 | 5 | 3.97% | 103.3% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-10-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-03 15:15:00 | 940.00 | 962.84 | 962.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-04 09:15:00 | 925.90 | 962.47 | 962.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-12 09:15:00 | 959.00 | 956.29 | 959.19 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-12 09:15:00 | 959.00 | 956.29 | 959.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-12 09:15:00 | 959.00 | 956.29 | 959.19 | EMA400 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2023-10-13 09:15:00 | 946.60 | 956.09 | 958.99 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-13 10:15:00 | 947.30 | 956.01 | 958.93 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2023-10-16 09:15:00 | 948.75 | 955.70 | 958.69 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2023-10-16 10:15:00 | 951.15 | 955.65 | 958.65 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2023-10-18 09:15:00 | 947.80 | 955.39 | 958.33 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-18 10:15:00 | 942.80 | 955.26 | 958.25 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2023-12-04 09:15:00 | 971.20 | 933.73 | 940.00 | SL hit (close>static) qty=1.00 sl=959.35 alert=retest2 |
| Stop hit — per-position SL triggered | 2023-12-04 09:15:00 | 971.20 | 933.73 | 940.00 | SL hit (close>static) qty=1.00 sl=959.35 alert=retest2 |

### Cycle 2 — BUY (started 2023-12-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-06 15:15:00 | 1002.70 | 945.74 | 945.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-08 09:15:00 | 1004.05 | 949.75 | 947.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-02 10:15:00 | 986.30 | 988.42 | 973.85 | EMA200 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2024-01-05 09:15:00 | 996.00 | 987.82 | 974.91 | ENTRY1 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2024-01-05 10:15:00 | 988.80 | 987.83 | 974.98 | ENTRY1 sustain failed after 60m |
| Cross detected — sustain check pending | 2024-01-05 12:15:00 | 994.80 | 987.92 | 975.16 | ENTRY1 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2024-01-05 13:15:00 | 986.60 | 987.91 | 975.21 | ENTRY1 sustain failed after 60m |
| Cross detected — sustain check pending | 2024-01-05 14:15:00 | 994.10 | 987.97 | 975.31 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-05 15:15:00 | 993.70 | 988.03 | 975.40 | BUY ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2024-01-08 10:15:00 | 993.65 | 988.13 | 975.58 | ENTRY1 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2024-01-08 11:15:00 | 990.75 | 988.16 | 975.65 | ENTRY1 sustain failed after 60m |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-09 15:15:00 | 977.75 | 988.04 | 976.27 | EMA400 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2024-01-10 09:15:00 | 987.55 | 988.04 | 976.32 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-10 10:15:00 | 990.10 | 988.06 | 976.39 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2024-01-17 13:15:00 | 986.10 | 991.67 | 980.36 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2024-01-17 14:15:00 | 981.40 | 991.56 | 980.36 | ENTRY2 sustain failed after 60m |
| Stop hit — per-position SL triggered | 2024-01-17 15:15:00 | 979.50 | 991.44 | 980.36 | SL hit (close<ema400) qty=1.00 sl=980.36 alert=retest1 |
| Cross detected — sustain check pending | 2024-01-18 09:15:00 | 983.70 | 991.37 | 980.38 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-18 10:15:00 | 989.30 | 991.35 | 980.42 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2024-01-18 13:15:00 | 984.30 | 991.19 | 980.50 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-18 14:15:00 | 986.15 | 991.14 | 980.53 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-29 10:15:00 | 1134.07 | 1084.72 | 1067.06 | Partial book 0.50 @ 15%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-29 11:15:00 | 1138.61 | 1085.29 | 1067.43 | Partial book 0.50 @ 15%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-29 11:15:00 | 1137.69 | 1085.29 | 1067.43 | Partial book 0.50 @ 15%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-05-22 10:15:00 | 1108.45 | 1112.67 | 1092.37 | SL hit (close<ema200) qty=0.50 sl=1112.67 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-05-22 10:15:00 | 1108.45 | 1112.67 | 1092.37 | SL hit (close<ema200) qty=0.50 sl=1112.67 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-05-22 10:15:00 | 1108.45 | 1112.67 | 1092.37 | SL hit (close<ema200) qty=0.50 sl=1112.67 alert=retest2 |

### Cycle 3 — SELL (started 2025-01-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-15 09:15:00 | 1237.50 | 1279.88 | 1279.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-17 09:15:00 | 1226.00 | 1274.94 | 1277.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-28 09:15:00 | 1256.25 | 1252.02 | 1263.98 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-28 11:15:00 | 1262.85 | 1252.15 | 1263.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-28 11:15:00 | 1262.85 | 1252.15 | 1263.93 | EMA400 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2025-01-28 14:15:00 | 1245.90 | 1252.14 | 1263.74 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-28 15:15:00 | 1246.75 | 1252.09 | 1263.66 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-01-31 09:15:00 | 1243.15 | 1251.88 | 1262.71 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-31 10:15:00 | 1241.50 | 1251.78 | 1262.61 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-02-03 09:15:00 | 1239.45 | 1251.49 | 1262.14 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-03 10:15:00 | 1247.10 | 1251.45 | 1262.07 | SELL ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-02-04 09:15:00 | 1277.10 | 1251.85 | 1261.96 | SL hit (close>static) qty=1.00 sl=1264.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-02-04 09:15:00 | 1277.10 | 1251.85 | 1261.96 | SL hit (close>static) qty=1.00 sl=1264.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-02-04 09:15:00 | 1277.10 | 1251.85 | 1261.96 | SL hit (close>static) qty=1.00 sl=1264.00 alert=retest2 |
| Cross detected — sustain check pending | 2025-02-07 13:15:00 | 1250.80 | 1255.34 | 1262.64 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-02-07 14:15:00 | 1257.80 | 1255.36 | 1262.61 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-02-11 12:15:00 | 1249.60 | 1255.44 | 1262.23 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-02-11 13:15:00 | 1252.35 | 1255.41 | 1262.18 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-02-12 09:15:00 | 1249.70 | 1255.29 | 1262.02 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-12 10:15:00 | 1245.70 | 1255.20 | 1261.94 | SELL ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-13 09:15:00 | 1262.95 | 1255.13 | 1261.71 | EMA400 retest candle locked (from downside) |
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
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-11 13:15:00 | 1247.15 | 1234.34 | 1246.01 | EMA400 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2025-03-12 11:15:00 | 1240.25 | 1234.84 | 1245.98 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-12 12:15:00 | 1239.45 | 1234.89 | 1245.94 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-03-13 09:15:00 | 1253.65 | 1235.35 | 1245.96 | SL hit (close>static) qty=1.00 sl=1247.95 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-03-17 09:15:00 | 1266.65 | 1236.59 | 1246.23 | SL hit (close>static) qty=1.00 sl=1264.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-03-17 09:15:00 | 1266.65 | 1236.59 | 1246.23 | SL hit (close>static) qty=1.00 sl=1266.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-03-17 09:15:00 | 1266.65 | 1236.59 | 1246.23 | SL hit (close>static) qty=1.00 sl=1266.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-03-17 09:15:00 | 1266.65 | 1236.59 | 1246.23 | SL hit (close>static) qty=1.00 sl=1266.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-03-17 09:15:00 | 1266.65 | 1236.59 | 1246.23 | SL hit (close>static) qty=1.00 sl=1266.00 alert=retest2 |

### Cycle 4 — BUY (started 2025-03-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-21 11:15:00 | 1341.20 | 1254.86 | 1254.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-24 09:15:00 | 1358.20 | 1259.24 | 1256.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-07 09:15:00 | 1287.85 | 1295.44 | 1278.62 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-07 10:15:00 | 1276.10 | 1295.25 | 1278.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-07 10:15:00 | 1276.10 | 1295.25 | 1278.61 | EMA400 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2025-04-08 09:15:00 | 1295.80 | 1294.52 | 1278.73 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-08 10:15:00 | 1296.40 | 1294.54 | 1278.82 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-04-09 11:15:00 | 1291.10 | 1294.69 | 1279.52 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-09 12:15:00 | 1296.75 | 1294.71 | 1279.60 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-25 09:15:00 | 1490.86 | 1439.83 | 1423.74 | Partial book 0.50 @ 15%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-25 09:15:00 | 1491.26 | 1439.83 | 1423.74 | Partial book 0.50 @ 15%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-05 09:15:00 | 1452.90 | 1454.66 | 1435.64 | SL hit (close<ema200) qty=0.50 sl=1454.66 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-05 09:15:00 | 1452.90 | 1454.66 | 1435.64 | SL hit (close<ema200) qty=0.50 sl=1454.66 alert=retest2 |

### Cycle 5 — SELL (started 2025-09-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-03 12:15:00 | 1391.20 | 1429.48 | 1429.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-24 09:15:00 | 1381.20 | 1416.12 | 1421.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-15 09:15:00 | 1389.80 | 1388.78 | 1401.94 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-16 09:15:00 | 1402.60 | 1389.30 | 1401.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 09:15:00 | 1402.60 | 1389.30 | 1401.75 | EMA400 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2025-10-20 14:15:00 | 1390.00 | 1393.73 | 1402.96 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-20 15:15:00 | 1393.00 | 1393.73 | 1402.91 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2026-01-06 12:15:00 | 1410.30 | 1363.62 | 1370.86 | SL hit (close>static) qty=1.00 sl=1410.00 alert=retest2 |

### Cycle 6 — BUY (started 2026-01-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-12 13:15:00 | 1416.90 | 1377.35 | 1377.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-13 09:15:00 | 1422.70 | 1378.48 | 1377.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-19 09:15:00 | 1362.30 | 1386.10 | 1381.95 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-19 09:15:00 | 1362.30 | 1386.10 | 1381.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 09:15:00 | 1362.30 | 1386.10 | 1381.95 | EMA400 retest candle locked (from upside) |

### Cycle 7 — SELL (started 2026-01-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-23 13:15:00 | 1347.60 | 1378.54 | 1378.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-23 14:15:00 | 1344.20 | 1378.19 | 1378.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-29 13:15:00 | 1378.00 | 1375.34 | 1376.86 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-29 13:15:00 | 1378.00 | 1375.34 | 1376.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 13:15:00 | 1378.00 | 1375.34 | 1376.86 | EMA400 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2026-01-30 10:15:00 | 1364.90 | 1375.41 | 1376.86 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-30 11:15:00 | 1360.30 | 1375.26 | 1376.78 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2026-02-03 09:15:00 | 1390.10 | 1372.87 | 1375.46 | SL hit (close>static) qty=1.00 sl=1378.50 alert=retest2 |

### Cycle 8 — BUY (started 2026-02-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-05 14:15:00 | 1398.30 | 1377.89 | 1377.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-06 09:15:00 | 1403.60 | 1378.29 | 1378.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-19 14:15:00 | 1387.30 | 1391.98 | 1386.14 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-19 14:15:00 | 1387.30 | 1391.98 | 1386.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 14:15:00 | 1387.30 | 1391.98 | 1386.14 | EMA400 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2026-02-20 09:15:00 | 1398.20 | 1392.01 | 1386.21 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-20 10:15:00 | 1398.70 | 1392.08 | 1386.27 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2026-02-20 13:15:00 | 1397.50 | 1392.20 | 1386.42 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-02-20 14:15:00 | 1394.20 | 1392.22 | 1386.46 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2026-02-23 09:15:00 | 1406.70 | 1392.39 | 1386.60 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-23 10:15:00 | 1401.70 | 1392.48 | 1386.67 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2026-02-24 14:15:00 | 1384.80 | 1392.71 | 1387.11 | SL hit (close<static) qty=1.00 sl=1385.70 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-24 14:15:00 | 1384.80 | 1392.71 | 1387.11 | SL hit (close<static) qty=1.00 sl=1385.70 alert=retest2 |
| Cross detected — sustain check pending | 2026-02-25 09:15:00 | 1399.30 | 1392.70 | 1387.16 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-02-25 10:15:00 | 1394.90 | 1392.72 | 1387.20 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2026-02-25 11:15:00 | 1399.60 | 1392.79 | 1387.26 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-25 12:15:00 | 1400.00 | 1392.86 | 1387.32 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2026-02-27 12:15:00 | 1383.80 | 1393.60 | 1388.09 | SL hit (close<static) qty=1.00 sl=1385.70 alert=retest2 |

### Cycle 9 — SELL (started 2026-03-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-06 14:15:00 | 1312.00 | 1383.42 | 1383.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-09 09:15:00 | 1271.90 | 1381.64 | 1382.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 09:15:00 | 1302.40 | 1281.44 | 1317.95 | EMA200 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2026-04-09 11:15:00 | 1290.30 | 1283.33 | 1317.32 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-09 12:15:00 | 1286.30 | 1283.36 | 1317.16 | SELL ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-10 09:15:00 | 1321.00 | 1283.72 | 1316.67 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-04-10 09:15:00 | 1321.00 | 1283.72 | 1316.67 | SL hit (close>ema400) qty=1.00 sl=1316.67 alert=retest1 |
| Cross detected — sustain check pending | 2026-04-29 13:15:00 | 1284.10 | 1313.14 | 1323.67 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-29 14:15:00 | 1280.10 | 1312.81 | 1323.45 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2023-10-13 10:15:00 | 947.30 | 2023-12-04 09:15:00 | 971.20 | STOP_HIT | 1.00 | -2.52% |
| SELL | retest2 | 2023-10-18 10:15:00 | 942.80 | 2023-12-04 09:15:00 | 971.20 | STOP_HIT | 1.00 | -3.01% |
| BUY | retest1 | 2024-01-05 15:15:00 | 993.70 | 2024-01-17 15:15:00 | 979.50 | STOP_HIT | 1.00 | -1.43% |
| BUY | retest2 | 2024-01-10 10:15:00 | 990.10 | 2024-04-29 10:15:00 | 1134.07 | PARTIAL | 0.50 | 14.54% |
| BUY | retest2 | 2024-01-18 10:15:00 | 989.30 | 2024-04-29 11:15:00 | 1138.61 | PARTIAL | 0.50 | 15.09% |
| BUY | retest2 | 2024-01-18 14:15:00 | 986.15 | 2024-04-29 11:15:00 | 1137.69 | PARTIAL | 0.50 | 15.37% |
| BUY | retest2 | 2024-01-10 10:15:00 | 990.10 | 2024-05-22 10:15:00 | 1108.45 | STOP_HIT | 0.50 | 11.95% |
| BUY | retest2 | 2024-01-18 10:15:00 | 989.30 | 2024-05-22 10:15:00 | 1108.45 | STOP_HIT | 0.50 | 12.04% |
| BUY | retest2 | 2024-01-18 14:15:00 | 986.15 | 2024-05-22 10:15:00 | 1108.45 | STOP_HIT | 0.50 | 12.40% |
| SELL | retest2 | 2025-01-28 15:15:00 | 1246.75 | 2025-02-04 09:15:00 | 1277.10 | STOP_HIT | 1.00 | -2.43% |
| SELL | retest2 | 2025-01-31 10:15:00 | 1241.50 | 2025-02-04 09:15:00 | 1277.10 | STOP_HIT | 1.00 | -2.87% |
| SELL | retest2 | 2025-02-03 10:15:00 | 1247.10 | 2025-02-04 09:15:00 | 1277.10 | STOP_HIT | 1.00 | -2.41% |
| SELL | retest2 | 2025-02-12 10:15:00 | 1245.70 | 2025-03-13 09:15:00 | 1253.65 | STOP_HIT | 1.00 | -0.64% |
| SELL | retest2 | 2025-02-13 14:15:00 | 1248.85 | 2025-03-17 09:15:00 | 1266.65 | STOP_HIT | 1.00 | -1.43% |
| SELL | retest2 | 2025-02-17 10:15:00 | 1237.70 | 2025-03-17 09:15:00 | 1266.65 | STOP_HIT | 1.00 | -2.34% |
| SELL | retest2 | 2025-02-18 10:15:00 | 1238.00 | 2025-03-17 09:15:00 | 1266.65 | STOP_HIT | 1.00 | -2.31% |
| SELL | retest2 | 2025-02-21 09:15:00 | 1227.00 | 2025-03-17 09:15:00 | 1266.65 | STOP_HIT | 1.00 | -3.23% |
| SELL | retest2 | 2025-03-12 12:15:00 | 1239.45 | 2025-03-17 09:15:00 | 1266.65 | STOP_HIT | 1.00 | -2.19% |
| BUY | retest2 | 2025-04-08 10:15:00 | 1296.40 | 2025-07-25 09:15:00 | 1490.86 | PARTIAL | 0.50 | 15.00% |
| BUY | retest2 | 2025-04-09 12:15:00 | 1296.75 | 2025-07-25 09:15:00 | 1491.26 | PARTIAL | 0.50 | 15.00% |
| BUY | retest2 | 2025-04-08 10:15:00 | 1296.40 | 2025-08-05 09:15:00 | 1452.90 | STOP_HIT | 0.50 | 12.07% |
| BUY | retest2 | 2025-04-09 12:15:00 | 1296.75 | 2025-08-05 09:15:00 | 1452.90 | STOP_HIT | 0.50 | 12.04% |
| SELL | retest2 | 2025-10-20 15:15:00 | 1393.00 | 2026-01-06 12:15:00 | 1410.30 | STOP_HIT | 1.00 | -1.24% |
| SELL | retest2 | 2026-01-30 11:15:00 | 1360.30 | 2026-02-03 09:15:00 | 1390.10 | STOP_HIT | 1.00 | -2.19% |
| BUY | retest2 | 2026-02-20 10:15:00 | 1398.70 | 2026-02-24 14:15:00 | 1384.80 | STOP_HIT | 1.00 | -0.99% |
| BUY | retest2 | 2026-02-23 10:15:00 | 1401.70 | 2026-02-24 14:15:00 | 1384.80 | STOP_HIT | 1.00 | -1.21% |
| BUY | retest2 | 2026-02-25 12:15:00 | 1400.00 | 2026-02-27 12:15:00 | 1383.80 | STOP_HIT | 1.00 | -1.16% |
| SELL | retest1 | 2026-04-09 12:15:00 | 1286.30 | 2026-04-10 09:15:00 | 1321.00 | STOP_HIT | 1.00 | -2.70% |
