# TATACONSUM (TATACONSUM)

## Backtest Summary

- **Source:** Fyers history API (1H bars)
- **Window:** 2024-05-18 09:15:00 → 2026-05-07 15:15:00 (3402 bars)
- **Last close:** 1156.40
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
| ALERT2_SKIP | 1 |
| ALERT3 | 5 |
| PENDING | 39 |
| PENDING_CANCEL | 15 |
| ENTRY1 | 5 |
| ENTRY2 | 19 |
| PARTIAL | 0 |
| TARGET_HIT | 0 |
| STOP_HIT | 24 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 24 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 2 / 22
- **Target hits / Stop hits / Partials:** 0 / 24 / 0
- **Avg / median % per leg:** -1.88% / -1.55%
- **Sum % (uncompounded):** -45.10%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 24 | 2 | 8.3% | 0 | 24 | 0 | -1.88% | -45.1% |
| BUY @ 2nd Alert (retest1) | 5 | 1 | 20.0% | 0 | 5 | 0 | -1.16% | -5.8% |
| BUY @ 3rd Alert (retest2) | 19 | 1 | 5.3% | 0 | 19 | 0 | -2.07% | -39.3% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 5 | 1 | 20.0% | 0 | 5 | 0 | -1.16% | -5.8% |
| retest2 (combined) | 19 | 1 | 5.3% | 0 | 19 | 0 | -2.07% | -39.3% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-02-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-06 11:15:00 | 1016.80 | 980.56 | 980.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-06 12:15:00 | 1020.65 | 980.96 | 980.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-21 09:15:00 | 1000.50 | 1002.45 | 993.51 | EMA200 retest candle locked |
| Cross detected — sustain check pending | 2025-02-21 10:15:00 | 1009.60 | 1002.52 | 993.59 | ENTRY1 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-02-21 11:15:00 | 1008.00 | 1002.57 | 993.66 | ENTRY1 sustain failed after 60m |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-21 13:15:00 | 997.00 | 1002.46 | 993.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 13:15:00 | 997.00 | 1002.46 | 993.69 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-02-21 14:15:00 | 1005.45 | 1002.49 | 993.75 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-21 15:15:00 | 1003.60 | 1002.50 | 993.80 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-02-28 11:15:00 | 982.25 | 1002.68 | 994.89 | SL hit (close<static) qty=1.00 sl=991.45 alert=retest2 |
| Cross detected — sustain check pending | 2025-03-28 09:15:00 | 1009.10 | 971.33 | 977.56 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-28 10:15:00 | 1004.45 | 971.66 | 977.69 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-04-01 13:15:00 | 990.60 | 974.40 | 978.80 | SL hit (close<static) qty=1.00 sl=991.45 alert=retest2 |
| Cross detected — sustain check pending | 2025-04-02 09:15:00 | 1059.05 | 975.59 | 979.33 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-02 10:15:00 | 1069.00 | 976.52 | 979.78 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-04-03 11:15:00 | 1069.40 | 983.33 | 983.13 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 2 — BUY (started 2025-04-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-03 11:15:00 | 1069.40 | 983.33 | 983.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-03 12:15:00 | 1071.00 | 984.20 | 983.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-09 09:15:00 | 1093.70 | 1101.70 | 1061.27 | EMA200 retest candle locked |
| Cross detected — sustain check pending | 2025-05-09 11:15:00 | 1109.30 | 1101.71 | 1061.68 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-09 12:15:00 | 1112.60 | 1101.81 | 1061.93 | BUY ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2025-05-29 11:15:00 | 1107.00 | 1123.01 | 1090.49 | ENTRY1 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-05-29 12:15:00 | 1104.80 | 1122.83 | 1090.57 | ENTRY1 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-05-29 14:15:00 | 1110.20 | 1122.52 | 1090.73 | ENTRY1 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-05-29 15:15:00 | 1105.60 | 1122.35 | 1090.81 | ENTRY1 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-05-30 09:15:00 | 1111.00 | 1122.24 | 1090.91 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-30 10:15:00 | 1108.40 | 1122.10 | 1090.99 | BUY ENTRY1 attempt 2/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2025-05-30 12:15:00 | 1107.50 | 1121.78 | 1091.14 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-30 13:15:00 | 1111.70 | 1121.68 | 1091.25 | BUY ENTRY1 attempt 3/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2025-05-30 15:15:00 | 1109.00 | 1121.39 | 1091.40 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-02 09:15:00 | 1107.50 | 1121.25 | 1091.48 | BUY ENTRY1 attempt 4/4 (retest1 break sustained 3960m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 10:15:00 | 1091.20 | 1118.66 | 1097.46 | EMA400 retest candle locked |
| Stop hit — per-position SL triggered | 2025-06-12 10:15:00 | 1091.20 | 1118.66 | 1097.46 | SL hit (close<ema400) qty=1.00 sl=1097.46 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-06-12 10:15:00 | 1091.20 | 1118.66 | 1097.46 | SL hit (close<ema400) qty=1.00 sl=1097.46 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-06-12 10:15:00 | 1091.20 | 1118.66 | 1097.46 | SL hit (close<ema400) qty=1.00 sl=1097.46 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-06-12 10:15:00 | 1091.20 | 1118.66 | 1097.46 | SL hit (close<ema400) qty=1.00 sl=1097.46 alert=retest1 |
| Cross detected — sustain check pending | 2025-06-23 10:15:00 | 1101.50 | 1103.95 | 1093.75 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-23 11:15:00 | 1101.60 | 1103.93 | 1093.78 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-07-01 10:15:00 | 1086.80 | 1108.53 | 1098.24 | SL hit (close<static) qty=1.00 sl=1090.60 alert=retest2 |
| Cross detected — sustain check pending | 2025-07-03 09:15:00 | 1101.80 | 1106.73 | 1097.96 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-07-03 10:15:00 | 1096.00 | 1106.63 | 1097.95 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-07-07 10:15:00 | 1100.20 | 1104.52 | 1097.44 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-07 11:15:00 | 1101.80 | 1104.49 | 1097.46 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-07-07 15:15:00 | 1106.00 | 1104.42 | 1097.57 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-08 09:15:00 | 1105.50 | 1104.43 | 1097.61 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 1080m) |
| Cross detected — sustain check pending | 2025-07-08 14:15:00 | 1100.70 | 1104.15 | 1097.63 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-07-08 15:15:00 | 1097.30 | 1104.09 | 1097.63 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-07-09 10:15:00 | 1100.70 | 1104.01 | 1097.66 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-09 11:15:00 | 1105.80 | 1104.03 | 1097.70 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 13:15:00 | 1096.50 | 1103.91 | 1097.71 | EMA400 retest candle locked |
| Stop hit — per-position SL triggered | 2025-07-10 14:15:00 | 1089.30 | 1103.24 | 1097.61 | SL hit (close<static) qty=1.00 sl=1090.60 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-10 14:15:00 | 1089.30 | 1103.24 | 1097.61 | SL hit (close<static) qty=1.00 sl=1090.60 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-10 14:15:00 | 1089.30 | 1103.24 | 1097.61 | SL hit (close<static) qty=1.00 sl=1090.60 alert=retest2 |
| Cross detected — sustain check pending | 2025-07-17 13:15:00 | 1105.50 | 1097.13 | 1095.15 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-17 14:15:00 | 1105.20 | 1097.21 | 1095.20 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-07-21 09:15:00 | 1082.80 | 1097.22 | 1095.30 | SL hit (close<static) qty=1.00 sl=1095.20 alert=retest2 |
| Cross detected — sustain check pending | 2025-08-20 14:15:00 | 1104.70 | 1073.10 | 1079.90 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-20 15:15:00 | 1105.30 | 1073.42 | 1080.02 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-08-21 09:15:00 | 1089.80 | 1073.59 | 1080.07 | SL hit (close<static) qty=1.00 sl=1095.20 alert=retest2 |
| Cross detected — sustain check pending | 2025-09-02 15:15:00 | 1105.50 | 1075.93 | 1079.90 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-09-03 09:15:00 | 1097.40 | 1076.15 | 1079.99 | ENTRY2 sustain failed after 1080m |
| Cross detected — sustain check pending | 2025-09-03 13:15:00 | 1101.50 | 1076.97 | 1080.32 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-03 14:15:00 | 1105.20 | 1077.25 | 1080.45 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-09-04 09:15:00 | 1091.00 | 1077.66 | 1080.62 | SL hit (close<static) qty=1.00 sl=1095.20 alert=retest2 |
| Cross detected — sustain check pending | 2025-09-10 14:15:00 | 1102.60 | 1078.11 | 1080.37 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-09-10 15:15:00 | 1099.00 | 1078.32 | 1080.47 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-09-11 09:15:00 | 1101.30 | 1078.55 | 1080.57 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-09-11 10:15:00 | 1097.60 | 1078.74 | 1080.65 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-09-11 11:15:00 | 1102.30 | 1078.97 | 1080.76 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-09-11 12:15:00 | 1099.80 | 1079.18 | 1080.86 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-09-11 13:15:00 | 1102.10 | 1079.41 | 1080.96 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-11 14:15:00 | 1106.10 | 1079.67 | 1081.09 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-09-15 10:15:00 | 1092.60 | 1081.67 | 1082.04 | SL hit (close<static) qty=1.00 sl=1095.20 alert=retest2 |

### Cycle 3 — BUY (started 2025-09-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-15 14:15:00 | 1102.90 | 1082.45 | 1082.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-17 09:15:00 | 1120.20 | 1083.89 | 1083.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-14 10:15:00 | 1114.90 | 1114.92 | 1104.03 | EMA200 retest candle locked |
| Cross detected — sustain check pending | 2025-10-15 13:15:00 | 1119.00 | 1115.10 | 1104.65 | ENTRY1 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-10-15 14:15:00 | 1115.50 | 1115.10 | 1104.71 | ENTRY1 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-10-16 11:15:00 | 1122.40 | 1115.13 | 1104.93 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-16 12:15:00 | 1133.10 | 1115.31 | 1105.07 | BUY ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 09:15:00 | 1149.30 | 1162.92 | 1147.12 | EMA400 retest candle locked |
| Stop hit — per-position SL triggered | 2025-12-03 10:15:00 | 1144.20 | 1162.73 | 1147.10 | SL hit (close<ema400) qty=1.00 sl=1147.10 alert=retest1 |
| Cross detected — sustain check pending | 2025-12-05 14:15:00 | 1164.90 | 1160.17 | 1147.10 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-12-05 15:15:00 | 1161.90 | 1160.19 | 1147.17 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-12-15 10:15:00 | 1165.80 | 1156.15 | 1147.17 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-12-15 11:15:00 | 1158.20 | 1156.17 | 1147.22 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-12-16 09:15:00 | 1168.30 | 1156.39 | 1147.55 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-16 10:15:00 | 1174.40 | 1156.57 | 1147.68 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2026-01-21 15:15:00 | 1165.00 | 1179.82 | 1169.02 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-22 09:15:00 | 1183.90 | 1179.86 | 1169.10 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 1080m) |
| Cross detected — sustain check pending | 2026-01-27 11:15:00 | 1168.90 | 1178.31 | 1169.13 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-27 12:15:00 | 1175.70 | 1178.29 | 1169.16 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2026-01-28 09:15:00 | 1129.00 | 1177.86 | 1169.13 | SL hit (close<static) qty=1.00 sl=1144.30 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-28 09:15:00 | 1129.00 | 1177.86 | 1169.13 | SL hit (close<static) qty=1.00 sl=1144.30 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-28 09:15:00 | 1129.00 | 1177.86 | 1169.13 | SL hit (close<static) qty=1.00 sl=1144.30 alert=retest2 |
| Cross detected — sustain check pending | 2026-02-04 09:15:00 | 1167.20 | 1160.70 | 1161.20 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-04 10:15:00 | 1168.20 | 1160.78 | 1161.24 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 12:15:00 | 1154.80 | 1160.77 | 1161.23 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2026-02-09 12:15:00 | 1169.90 | 1159.73 | 1160.63 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-02-09 13:15:00 | 1164.50 | 1159.78 | 1160.65 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2026-02-09 14:15:00 | 1167.40 | 1159.86 | 1160.68 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-02-09 15:15:00 | 1163.20 | 1159.89 | 1160.69 | ENTRY2 sustain failed after 60m |
| Stop hit — per-position SL triggered | 2026-02-13 10:15:00 | 1142.00 | 1158.18 | 1159.72 | SL hit (close<static) qty=1.00 sl=1144.30 alert=retest2 |
| Cross detected — sustain check pending | 2026-02-18 14:15:00 | 1169.70 | 1155.31 | 1157.97 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-18 15:15:00 | 1167.00 | 1155.42 | 1158.02 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2026-02-23 09:15:00 | 1170.20 | 1156.12 | 1158.19 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-02-23 10:15:00 | 1165.90 | 1156.22 | 1158.23 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2026-02-23 11:15:00 | 1176.60 | 1156.42 | 1158.32 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-23 12:15:00 | 1172.30 | 1156.58 | 1158.39 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2026-02-25 13:15:00 | 1169.60 | 1158.74 | 1159.38 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-25 14:15:00 | 1172.70 | 1158.88 | 1159.45 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2026-02-27 09:15:00 | 1142.70 | 1159.11 | 1159.54 | SL hit (close<static) qty=1.00 sl=1153.70 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-27 09:15:00 | 1142.70 | 1159.11 | 1159.54 | SL hit (close<static) qty=1.00 sl=1153.70 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-27 09:15:00 | 1142.70 | 1159.11 | 1159.54 | SL hit (close<static) qty=1.00 sl=1153.70 alert=retest2 |
| Cross detected — sustain check pending | 2026-04-22 10:15:00 | 1175.90 | 1090.76 | 1105.04 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-22 11:15:00 | 1183.60 | 1091.69 | 1105.43 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2026-04-28 11:15:00 | 1153.20 | 1111.15 | 1114.10 | SL hit (close<static) qty=1.00 sl=1153.70 alert=retest2 |

### Cycle 4 — BUY (started 2026-04-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-30 11:15:00 | 1148.60 | 1117.10 | 1116.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-04 09:15:00 | 1163.00 | 1118.63 | 1117.75 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-02-21 15:15:00 | 1003.60 | 2025-02-28 11:15:00 | 982.25 | STOP_HIT | 1.00 | -2.13% |
| BUY | retest2 | 2025-03-28 10:15:00 | 1004.45 | 2025-04-01 13:15:00 | 990.60 | STOP_HIT | 1.00 | -1.38% |
| BUY | retest2 | 2025-04-02 10:15:00 | 1069.00 | 2025-04-03 11:15:00 | 1069.40 | STOP_HIT | 1.00 | 0.04% |
| BUY | retest1 | 2025-05-09 12:15:00 | 1112.60 | 2025-06-12 10:15:00 | 1091.20 | STOP_HIT | 1.00 | -1.92% |
| BUY | retest1 | 2025-05-30 10:15:00 | 1108.40 | 2025-06-12 10:15:00 | 1091.20 | STOP_HIT | 1.00 | -1.55% |
| BUY | retest1 | 2025-05-30 13:15:00 | 1111.70 | 2025-06-12 10:15:00 | 1091.20 | STOP_HIT | 1.00 | -1.84% |
| BUY | retest1 | 2025-06-02 09:15:00 | 1107.50 | 2025-06-12 10:15:00 | 1091.20 | STOP_HIT | 1.00 | -1.47% |
| BUY | retest2 | 2025-06-23 11:15:00 | 1101.60 | 2025-07-01 10:15:00 | 1086.80 | STOP_HIT | 1.00 | -1.34% |
| BUY | retest2 | 2025-07-07 11:15:00 | 1101.80 | 2025-07-10 14:15:00 | 1089.30 | STOP_HIT | 1.00 | -1.13% |
| BUY | retest2 | 2025-07-08 09:15:00 | 1105.50 | 2025-07-10 14:15:00 | 1089.30 | STOP_HIT | 1.00 | -1.47% |
| BUY | retest2 | 2025-07-09 11:15:00 | 1105.80 | 2025-07-10 14:15:00 | 1089.30 | STOP_HIT | 1.00 | -1.49% |
| BUY | retest2 | 2025-07-17 14:15:00 | 1105.20 | 2025-07-21 09:15:00 | 1082.80 | STOP_HIT | 1.00 | -2.03% |
| BUY | retest2 | 2025-08-20 15:15:00 | 1105.30 | 2025-08-21 09:15:00 | 1089.80 | STOP_HIT | 1.00 | -1.40% |
| BUY | retest2 | 2025-09-03 14:15:00 | 1105.20 | 2025-09-04 09:15:00 | 1091.00 | STOP_HIT | 1.00 | -1.28% |
| BUY | retest2 | 2025-09-11 14:15:00 | 1106.10 | 2025-09-15 10:15:00 | 1092.60 | STOP_HIT | 1.00 | -1.22% |
| BUY | retest1 | 2025-10-16 12:15:00 | 1133.10 | 2025-12-03 10:15:00 | 1144.20 | STOP_HIT | 1.00 | 0.98% |
| BUY | retest2 | 2025-12-16 10:15:00 | 1174.40 | 2026-01-28 09:15:00 | 1129.00 | STOP_HIT | 1.00 | -3.87% |
| BUY | retest2 | 2026-01-22 09:15:00 | 1183.90 | 2026-01-28 09:15:00 | 1129.00 | STOP_HIT | 1.00 | -4.64% |
| BUY | retest2 | 2026-01-27 12:15:00 | 1175.70 | 2026-01-28 09:15:00 | 1129.00 | STOP_HIT | 1.00 | -3.97% |
| BUY | retest2 | 2026-02-04 10:15:00 | 1168.20 | 2026-02-13 10:15:00 | 1142.00 | STOP_HIT | 1.00 | -2.24% |
| BUY | retest2 | 2026-02-18 15:15:00 | 1167.00 | 2026-02-27 09:15:00 | 1142.70 | STOP_HIT | 1.00 | -2.08% |
| BUY | retest2 | 2026-02-23 12:15:00 | 1172.30 | 2026-02-27 09:15:00 | 1142.70 | STOP_HIT | 1.00 | -2.52% |
| BUY | retest2 | 2026-02-25 14:15:00 | 1172.70 | 2026-02-27 09:15:00 | 1142.70 | STOP_HIT | 1.00 | -2.56% |
| BUY | retest2 | 2026-04-22 11:15:00 | 1183.60 | 2026-04-28 11:15:00 | 1153.20 | STOP_HIT | 1.00 | -2.57% |
