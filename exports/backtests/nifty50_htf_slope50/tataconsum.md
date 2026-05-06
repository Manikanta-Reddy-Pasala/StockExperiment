# TATACONSUM (TATACONSUM.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-06-06 09:15:00 → 2026-05-06 15:30:00 (4997 bars)
- **Last close:** 1152.20
- **Strategy:** EMA 200/400 1H crossover (v2 BTC rules)
- **Target:** entry × (1 ± 30%)
- **Partial:** 50% qty booked @ 15%, trail SL → entry
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 3 attempts each at retest1 and retest2

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 5 |
| ALERT1 | 4 |
| ALERT2 | 4 |
| ALERT2_SKIP | 1 |
| ALERT3 | 6 |
| PENDING | 33 |
| PENDING_CANCEL | 10 |
| ENTRY1 | 7 |
| ENTRY2 | 16 |
| PARTIAL | 0 |
| TARGET_HIT | 0 |
| STOP_HIT | 23 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 23 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 1 / 22
- **Target hits / Stop hits / Partials:** 0 / 23 / 0
- **Avg / median % per leg:** -1.46% / -1.32%
- **Sum % (uncompounded):** -33.59%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 23 | 1 | 4.3% | 0 | 23 | 0 | -1.46% | -33.6% |
| BUY @ 2nd Alert (retest1) | 7 | 1 | 14.3% | 0 | 7 | 0 | -1.28% | -8.9% |
| BUY @ 3rd Alert (retest2) | 16 | 0 | 0.0% | 0 | 16 | 0 | -1.54% | -24.7% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 7 | 1 | 14.3% | 0 | 7 | 0 | -1.28% | -8.9% |
| retest2 (combined) | 16 | 0 | 0.0% | 0 | 16 | 0 | -1.54% | -24.7% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-07-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-08 11:15:00 | 1146.50 | 1111.08 | 1110.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-16 10:15:00 | 1170.50 | 1122.91 | 1117.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-12 09:15:00 | 1166.95 | 1176.42 | 1155.36 | EMA200 retest candle locked |
| Cross detected — sustain check pending | 2024-08-16 12:15:00 | 1187.85 | 1175.97 | 1157.50 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-16 13:15:00 | 1188.90 | 1176.09 | 1157.66 | BUY ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2024-08-22 09:15:00 | 1204.00 | 1176.24 | 1159.82 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-22 10:15:00 | 1200.20 | 1176.48 | 1160.03 | BUY ENTRY1 attempt 2/4 (retest1 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-06 09:15:00 | 1176.45 | 1189.53 | 1173.17 | EMA400 retest candle locked |
| Stop hit — per-position SL triggered | 2024-09-06 09:15:00 | 1173.17 | 1189.53 | 1173.17 | SL hit qty=1.00 sl=1173.17 alert=retest1 |
| Stop hit — per-position SL triggered | 2024-09-06 09:15:00 | 1173.17 | 1189.53 | 1173.17 | SL hit qty=1.00 sl=1173.17 alert=retest1 |
| Cross detected — sustain check pending | 2024-09-09 14:15:00 | 1194.40 | 1188.81 | 1173.74 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-09 15:15:00 | 1192.05 | 1188.84 | 1173.83 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2024-09-26 09:15:00 | 1201.85 | 1201.57 | 1186.80 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-26 10:15:00 | 1204.95 | 1201.60 | 1186.89 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2024-10-03 09:15:00 | 1171.55 | 1201.39 | 1188.66 | SL hit qty=1.00 sl=1171.55 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-10-03 09:15:00 | 1171.55 | 1201.39 | 1188.66 | SL hit qty=1.00 sl=1171.55 alert=retest2 |
| CROSSOVER_SKIP | 2024-10-10 12:15:00 | 1122.00 | 1177.80 | 1178.07 | slope filter: EMA200 not falling 0.50% over 350 bars |
| CROSSOVER_SKIP | 2025-02-10 09:15:00 | 1031.50 | 980.77 | 980.65 | HTF filter: close below htf_sma |
| CROSSOVER_SKIP | 2025-03-07 14:15:00 | 962.05 | 987.27 | 987.36 | slope filter: EMA200 not falling 0.50% over 350 bars |

### Cycle 2 — BUY (started 2025-04-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-03 10:15:00 | 1066.90 | 982.09 | 982.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-03 12:15:00 | 1071.00 | 983.83 | 982.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-09 09:15:00 | 1093.70 | 1101.67 | 1061.03 | EMA200 retest candle locked |
| Cross detected — sustain check pending | 2025-05-09 11:15:00 | 1109.30 | 1101.68 | 1061.44 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-09 12:15:00 | 1112.60 | 1101.79 | 1061.69 | BUY ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2025-05-15 10:15:00 | 1120.20 | 1107.58 | 1069.69 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-15 11:15:00 | 1129.20 | 1107.79 | 1069.98 | BUY ENTRY1 attempt 2/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2025-05-29 11:15:00 | 1106.80 | 1123.00 | 1090.34 | ENTRY1 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-05-29 12:15:00 | 1104.80 | 1122.82 | 1090.42 | ENTRY1 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-05-29 14:15:00 | 1110.20 | 1122.51 | 1090.58 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-29 15:15:00 | 1109.80 | 1122.39 | 1090.68 | BUY ENTRY1 attempt 3/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2025-05-30 12:15:00 | 1107.50 | 1121.81 | 1091.02 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-30 13:15:00 | 1111.70 | 1121.71 | 1091.12 | BUY ENTRY1 attempt 4/4 (retest1 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 10:15:00 | 1091.60 | 1118.63 | 1097.34 | EMA400 retest candle locked |
| Stop hit — per-position SL triggered | 2025-06-12 10:15:00 | 1097.34 | 1118.63 | 1097.34 | SL hit qty=1.00 sl=1097.34 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-06-12 10:15:00 | 1097.34 | 1118.63 | 1097.34 | SL hit qty=1.00 sl=1097.34 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-06-12 10:15:00 | 1097.34 | 1118.63 | 1097.34 | SL hit qty=1.00 sl=1097.34 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-06-12 10:15:00 | 1097.34 | 1118.63 | 1097.34 | SL hit qty=1.00 sl=1097.34 alert=retest1 |
| Cross detected — sustain check pending | 2025-06-23 10:15:00 | 1101.30 | 1103.92 | 1093.64 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-23 11:15:00 | 1101.60 | 1103.89 | 1093.68 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-06-30 13:15:00 | 1090.90 | 1109.02 | 1098.20 | SL hit qty=1.00 sl=1090.90 alert=retest2 |
| Cross detected — sustain check pending | 2025-07-03 09:15:00 | 1102.20 | 1106.64 | 1097.84 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-07-03 10:15:00 | 1096.00 | 1106.53 | 1097.83 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-07-07 10:15:00 | 1100.20 | 1104.45 | 1097.34 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-07 11:15:00 | 1101.80 | 1104.42 | 1097.36 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-07-07 15:15:00 | 1106.00 | 1104.36 | 1097.46 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-08 09:15:00 | 1105.50 | 1104.37 | 1097.50 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 1080m) |
| Cross detected — sustain check pending | 2025-07-08 14:15:00 | 1100.70 | 1104.09 | 1097.53 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-07-08 15:15:00 | 1097.30 | 1104.02 | 1097.53 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-07-09 10:15:00 | 1100.70 | 1103.95 | 1097.56 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-09 11:15:00 | 1105.80 | 1103.97 | 1097.60 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 13:15:00 | 1096.50 | 1103.85 | 1097.61 | EMA400 retest candle locked |
| Stop hit — per-position SL triggered | 2025-07-10 14:15:00 | 1090.90 | 1103.20 | 1097.52 | SL hit qty=1.00 sl=1090.90 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-10 14:15:00 | 1090.90 | 1103.20 | 1097.52 | SL hit qty=1.00 sl=1090.90 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-10 14:15:00 | 1090.90 | 1103.20 | 1097.52 | SL hit qty=1.00 sl=1090.90 alert=retest2 |
| Cross detected — sustain check pending | 2025-07-17 13:15:00 | 1105.50 | 1097.13 | 1095.09 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-17 14:15:00 | 1105.30 | 1097.21 | 1095.15 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-07-18 12:15:00 | 1095.20 | 1097.38 | 1095.28 | SL hit qty=1.00 sl=1095.20 alert=retest2 |
| CROSSOVER_SKIP | 2025-07-24 09:15:00 | 1070.40 | 1093.37 | 1093.44 | HTF filter: close above htf_sma |
| Cross detected — sustain check pending | 2025-08-20 14:15:00 | 1104.70 | 1073.05 | 1079.85 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-20 15:15:00 | 1105.30 | 1073.37 | 1079.97 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-08-21 09:15:00 | 1095.20 | 1073.53 | 1080.02 | SL hit qty=1.00 sl=1095.20 alert=retest2 |
| Cross detected — sustain check pending | 2025-09-02 15:15:00 | 1105.50 | 1075.99 | 1079.91 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-09-03 09:15:00 | 1097.50 | 1076.21 | 1080.00 | ENTRY2 sustain failed after 1080m |
| Cross detected — sustain check pending | 2025-09-03 13:15:00 | 1101.50 | 1077.02 | 1080.33 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-03 14:15:00 | 1105.10 | 1077.30 | 1080.46 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-09-04 09:15:00 | 1095.20 | 1077.71 | 1080.63 | SL hit qty=1.00 sl=1095.20 alert=retest2 |
| Cross detected — sustain check pending | 2025-09-10 14:15:00 | 1102.60 | 1078.10 | 1080.36 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-09-10 15:15:00 | 1100.00 | 1078.32 | 1080.45 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-09-11 09:15:00 | 1101.30 | 1078.55 | 1080.56 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-09-11 10:15:00 | 1097.60 | 1078.74 | 1080.64 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-09-11 11:15:00 | 1102.30 | 1078.97 | 1080.75 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-09-11 12:15:00 | 1099.30 | 1079.18 | 1080.84 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-09-11 13:15:00 | 1102.10 | 1079.40 | 1080.95 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-11 14:15:00 | 1106.10 | 1079.67 | 1081.07 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-09-12 12:15:00 | 1095.20 | 1080.80 | 1081.61 | SL hit qty=1.00 sl=1095.20 alert=retest2 |

### Cycle 3 — BUY (started 2025-09-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-15 14:15:00 | 1102.70 | 1082.44 | 1082.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-17 09:15:00 | 1120.20 | 1083.85 | 1083.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-13 12:15:00 | 1114.80 | 1114.88 | 1103.73 | EMA200 retest candle locked |
| Cross detected — sustain check pending | 2025-10-16 11:15:00 | 1122.40 | 1115.19 | 1104.95 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-16 12:15:00 | 1133.10 | 1115.37 | 1105.09 | BUY ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 09:15:00 | 1149.20 | 1163.06 | 1147.22 | EMA400 retest candle locked |
| Stop hit — per-position SL triggered | 2025-12-03 09:15:00 | 1147.22 | 1163.06 | 1147.22 | SL hit qty=1.00 sl=1147.22 alert=retest1 |
| Cross detected — sustain check pending | 2025-12-05 14:15:00 | 1164.90 | 1160.32 | 1147.20 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-05 15:15:00 | 1162.90 | 1160.35 | 1147.28 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-12-08 15:15:00 | 1144.50 | 1159.76 | 1147.43 | SL hit qty=1.00 sl=1144.50 alert=retest2 |
| Cross detected — sustain check pending | 2025-12-15 10:15:00 | 1165.80 | 1156.24 | 1147.25 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-12-15 11:15:00 | 1158.20 | 1156.26 | 1147.30 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-12-15 12:15:00 | 1163.00 | 1156.33 | 1147.38 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-12-15 13:15:00 | 1158.00 | 1156.34 | 1147.43 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-12-16 09:15:00 | 1168.30 | 1156.49 | 1147.64 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-16 10:15:00 | 1174.40 | 1156.67 | 1147.77 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2026-01-21 14:15:00 | 1163.40 | 1180.08 | 1169.14 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-21 15:15:00 | 1163.10 | 1179.91 | 1169.11 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2026-01-27 09:15:00 | 1162.00 | 1178.67 | 1169.26 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-01-27 10:15:00 | 1159.00 | 1178.47 | 1169.21 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2026-01-27 11:15:00 | 1168.90 | 1178.37 | 1169.21 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-27 12:15:00 | 1175.70 | 1178.35 | 1169.24 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 13:15:00 | 1169.60 | 1178.26 | 1169.24 | EMA400 retest candle locked |
| Stop hit — per-position SL triggered | 2026-01-28 09:15:00 | 1144.50 | 1177.90 | 1169.20 | SL hit qty=1.00 sl=1144.50 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-28 09:15:00 | 1144.50 | 1177.90 | 1169.20 | SL hit qty=1.00 sl=1144.50 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-28 09:15:00 | 1144.50 | 1177.90 | 1169.20 | SL hit qty=1.00 sl=1144.50 alert=retest2 |
| CROSSOVER_SKIP | 2026-02-09 10:15:00 | 1160.40 | 1161.93 | 1161.93 | HTF filter: close above htf_sma |

### Cycle 4 — BUY (started 2026-02-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-09 12:15:00 | 1169.90 | 1161.97 | 1161.95 | EMA200 above EMA400 |
| CROSSOVER_SKIP | 2026-02-10 12:15:00 | 1153.60 | 1161.91 | 1161.92 | HTF filter: close above htf_sma |
| First Alert — break + close above crossover candle high | 2026-02-23 09:15:00 | 1170.20 | 1157.32 | 1159.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-26 11:15:00 | 1159.00 | 1160.26 | 1160.53 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-26 11:15:00 | 1159.00 | 1160.26 | 1160.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 11:15:00 | 1159.00 | 1160.26 | 1160.53 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2026-04-22 10:15:00 | 1175.90 | 1089.57 | 1105.00 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-22 11:15:00 | 1183.60 | 1090.51 | 1105.39 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2026-04-24 14:15:00 | 1172.00 | 1104.19 | 1111.32 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-24 15:15:00 | 1174.00 | 1104.89 | 1111.63 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2026-04-27 15:15:00 | 1158.50 | 1108.98 | 1113.48 | SL hit qty=1.00 sl=1158.50 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-27 15:15:00 | 1158.50 | 1108.98 | 1113.48 | SL hit qty=1.00 sl=1158.50 alert=retest2 |

### Cycle 5 — BUY (started 2026-05-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-04 09:15:00 | 1163.00 | 1117.93 | 1117.74 | EMA200 above EMA400 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-08-16 13:15:00 | 1188.90 | 2024-09-06 09:15:00 | 1173.17 | STOP_HIT | 1.00 | -1.32% |
| BUY | retest1 | 2024-08-22 10:15:00 | 1200.20 | 2024-09-06 09:15:00 | 1173.17 | STOP_HIT | 1.00 | -2.25% |
| BUY | retest2 | 2024-09-09 15:15:00 | 1192.05 | 2024-10-03 09:15:00 | 1171.55 | STOP_HIT | 1.00 | -1.72% |
| BUY | retest2 | 2024-09-26 10:15:00 | 1204.95 | 2024-10-03 09:15:00 | 1171.55 | STOP_HIT | 1.00 | -2.77% |
| BUY | retest1 | 2025-05-09 12:15:00 | 1112.60 | 2025-06-12 10:15:00 | 1097.34 | STOP_HIT | 1.00 | -1.37% |
| BUY | retest1 | 2025-05-15 11:15:00 | 1129.20 | 2025-06-12 10:15:00 | 1097.34 | STOP_HIT | 1.00 | -2.82% |
| BUY | retest1 | 2025-05-29 15:15:00 | 1109.80 | 2025-06-12 10:15:00 | 1097.34 | STOP_HIT | 1.00 | -1.12% |
| BUY | retest1 | 2025-05-30 13:15:00 | 1111.70 | 2025-06-12 10:15:00 | 1097.34 | STOP_HIT | 1.00 | -1.29% |
| BUY | retest2 | 2025-06-23 11:15:00 | 1101.60 | 2025-06-30 13:15:00 | 1090.90 | STOP_HIT | 1.00 | -0.97% |
| BUY | retest2 | 2025-07-07 11:15:00 | 1101.80 | 2025-07-10 14:15:00 | 1090.90 | STOP_HIT | 1.00 | -0.99% |
| BUY | retest2 | 2025-07-08 09:15:00 | 1105.50 | 2025-07-10 14:15:00 | 1090.90 | STOP_HIT | 1.00 | -1.32% |
| BUY | retest2 | 2025-07-09 11:15:00 | 1105.80 | 2025-07-10 14:15:00 | 1090.90 | STOP_HIT | 1.00 | -1.35% |
| BUY | retest2 | 2025-07-17 14:15:00 | 1105.30 | 2025-07-18 12:15:00 | 1095.20 | STOP_HIT | 1.00 | -0.91% |
| BUY | retest2 | 2025-08-20 15:15:00 | 1105.30 | 2025-08-21 09:15:00 | 1095.20 | STOP_HIT | 1.00 | -0.91% |
| BUY | retest2 | 2025-09-03 14:15:00 | 1105.10 | 2025-09-04 09:15:00 | 1095.20 | STOP_HIT | 1.00 | -0.90% |
| BUY | retest2 | 2025-09-11 14:15:00 | 1106.10 | 2025-09-12 12:15:00 | 1095.20 | STOP_HIT | 1.00 | -0.99% |
| BUY | retest1 | 2025-10-16 12:15:00 | 1133.10 | 2025-12-03 09:15:00 | 1147.22 | STOP_HIT | 1.00 | 1.25% |
| BUY | retest2 | 2025-12-05 15:15:00 | 1162.90 | 2025-12-08 15:15:00 | 1144.50 | STOP_HIT | 1.00 | -1.58% |
| BUY | retest2 | 2025-12-16 10:15:00 | 1174.40 | 2026-01-28 09:15:00 | 1144.50 | STOP_HIT | 1.00 | -2.55% |
| BUY | retest2 | 2026-01-21 15:15:00 | 1163.10 | 2026-01-28 09:15:00 | 1144.50 | STOP_HIT | 1.00 | -1.60% |
| BUY | retest2 | 2026-01-27 12:15:00 | 1175.70 | 2026-01-28 09:15:00 | 1144.50 | STOP_HIT | 1.00 | -2.65% |
| BUY | retest2 | 2026-04-22 11:15:00 | 1183.60 | 2026-04-27 15:15:00 | 1158.50 | STOP_HIT | 1.00 | -2.12% |
| BUY | retest2 | 2026-04-24 15:15:00 | 1174.00 | 2026-04-27 15:15:00 | 1158.50 | STOP_HIT | 1.00 | -1.32% |
