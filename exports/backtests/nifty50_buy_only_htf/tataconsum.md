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
| CROSSOVER | 11 |
| ALERT1 | 9 |
| ALERT2 | 8 |
| ALERT2_SKIP | 4 |
| ALERT3 | 12 |
| PENDING | 50 |
| PENDING_CANCEL | 15 |
| ENTRY1 | 9 |
| ENTRY2 | 26 |
| PARTIAL | 0 |
| TARGET_HIT | 0 |
| STOP_HIT | 35 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 35 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 1 / 34
- **Target hits / Stop hits / Partials:** 0 / 35 / 0
- **Avg / median % per leg:** -1.58% / -1.48%
- **Sum % (uncompounded):** -55.17%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 18 | 1 | 5.6% | 0 | 18 | 0 | -1.52% | -27.4% |
| BUY @ 2nd Alert (retest1) | 7 | 1 | 14.3% | 0 | 7 | 0 | -1.28% | -8.9% |
| BUY @ 3rd Alert (retest2) | 11 | 0 | 0.0% | 0 | 11 | 0 | -1.67% | -18.4% |
| SELL (all) | 17 | 0 | 0.0% | 0 | 17 | 0 | -1.64% | -27.8% |
| SELL @ 2nd Alert (retest1) | 2 | 0 | 0.0% | 0 | 2 | 0 | -1.67% | -3.3% |
| SELL @ 3rd Alert (retest2) | 15 | 0 | 0.0% | 0 | 15 | 0 | -1.63% | -24.5% |
| retest1 (combined) | 9 | 1 | 11.1% | 0 | 9 | 0 | -1.36% | -12.3% |
| retest2 (combined) | 26 | 0 | 0.0% | 0 | 26 | 0 | -1.65% | -42.9% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-05-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-07 13:15:00 | 1100.00 | 1123.60 | 1123.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-09 10:15:00 | 1091.95 | 1121.97 | 1122.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-22 11:15:00 | 1113.45 | 1107.55 | 1114.29 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-22 11:15:00 | 1113.45 | 1107.55 | 1114.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-22 11:15:00 | 1113.45 | 1107.55 | 1114.29 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2024-05-24 09:15:00 | 1103.05 | 1108.37 | 1114.32 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-24 10:15:00 | 1102.50 | 1108.31 | 1114.26 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2024-05-24 12:15:00 | 1101.55 | 1108.21 | 1114.15 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-24 13:15:00 | 1101.00 | 1108.14 | 1114.08 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2024-06-05 09:15:00 | 1118.30 | 1095.58 | 1105.72 | SL hit qty=1.00 sl=1118.30 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-06-05 09:15:00 | 1118.30 | 1095.58 | 1105.72 | SL hit qty=1.00 sl=1118.30 alert=retest2 |
| Cross detected — sustain check pending | 2024-06-20 09:15:00 | 1103.15 | 1110.95 | 1112.05 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-20 10:15:00 | 1098.20 | 1110.82 | 1111.98 | SELL ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2024-06-21 09:15:00 | 1094.30 | 1110.18 | 1111.62 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-21 10:15:00 | 1096.80 | 1110.04 | 1111.54 | SELL ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-02 15:15:00 | 1110.00 | 1102.64 | 1107.00 | EMA400 retest candle locked |
| Stop hit — per-position SL triggered | 2024-07-03 09:15:00 | 1118.30 | 1102.88 | 1107.09 | SL hit qty=1.00 sl=1118.30 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-07-03 09:15:00 | 1118.30 | 1102.88 | 1107.09 | SL hit qty=1.00 sl=1118.30 alert=retest2 |

### Cycle 2 — BUY (started 2024-07-08 11:15:00)

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

### Cycle 3 — SELL (started 2024-10-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-10 12:15:00 | 1122.00 | 1177.80 | 1178.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-10 14:15:00 | 1113.00 | 1176.57 | 1177.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-03 10:15:00 | 934.30 | 932.90 | 975.78 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-09 10:15:00 | 981.30 | 937.84 | 972.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-09 10:15:00 | 981.30 | 937.84 | 972.77 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-01-13 09:15:00 | 961.90 | 941.88 | 972.66 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-01-13 10:15:00 | 966.25 | 942.12 | 972.63 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-01-15 09:15:00 | 946.10 | 944.90 | 972.15 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-15 10:15:00 | 953.00 | 944.98 | 972.05 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-01-20 13:15:00 | 962.25 | 945.69 | 969.34 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-20 14:15:00 | 960.05 | 945.83 | 969.30 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-01-22 09:15:00 | 961.55 | 947.88 | 969.31 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-01-22 10:15:00 | 966.30 | 948.06 | 969.30 | ENTRY2 sustain failed after 60m |
| Stop hit — per-position SL triggered | 2025-01-23 09:15:00 | 981.60 | 949.31 | 969.30 | SL hit qty=1.00 sl=981.60 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-01-23 09:15:00 | 981.60 | 949.31 | 969.30 | SL hit qty=1.00 sl=981.60 alert=retest2 |
| Cross detected — sustain check pending | 2025-01-28 13:15:00 | 961.85 | 956.34 | 970.72 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-28 14:15:00 | 961.60 | 956.40 | 970.68 | SELL ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-01-30 11:15:00 | 958.20 | 956.65 | 970.04 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-01-30 12:15:00 | 966.10 | 956.75 | 970.02 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-01-30 13:15:00 | 961.05 | 956.79 | 969.98 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-01-30 14:15:00 | 968.45 | 956.90 | 969.97 | ENTRY2 sustain failed after 60m |
| Stop hit — per-position SL triggered | 2025-01-31 09:15:00 | 981.60 | 957.53 | 970.15 | SL hit qty=1.00 sl=981.60 alert=retest2 |
| CROSSOVER_SKIP | 2025-02-10 09:15:00 | 1031.50 | 980.77 | 980.65 | HTF filter: close below htf_sma |
| Cross detected — sustain check pending | 2025-03-03 10:15:00 | 957.25 | 998.80 | 992.70 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-03 11:15:00 | 954.55 | 998.36 | 992.51 | SELL ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-03-07 14:15:00 | 962.05 | 987.27 | 987.36 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 4 — SELL (started 2025-03-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-07 14:15:00 | 962.05 | 987.27 | 987.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-10 14:15:00 | 957.75 | 985.61 | 986.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-24 11:15:00 | 973.70 | 971.20 | 977.88 | EMA200 retest candle locked |
| Cross detected — sustain check pending | 2025-03-25 10:15:00 | 963.75 | 971.18 | 977.67 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-25 11:15:00 | 959.30 | 971.06 | 977.58 | SELL ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2025-03-26 10:15:00 | 958.05 | 970.81 | 977.26 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-26 11:15:00 | 962.00 | 970.72 | 977.18 | SELL ENTRY1 attempt 2/4 (retest1 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 15:15:00 | 980.40 | 970.49 | 976.72 | EMA400 retest candle locked |
| Stop hit — per-position SL triggered | 2025-03-27 15:15:00 | 976.72 | 970.49 | 976.72 | SL hit qty=1.00 sl=976.72 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-03-27 15:15:00 | 976.72 | 970.49 | 976.72 | SL hit qty=1.00 sl=976.72 alert=retest1 |

### Cycle 5 — BUY (started 2025-04-03 10:15:00)

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

### Cycle 6 — SELL (started 2025-07-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-24 09:15:00 | 1070.40 | 1093.37 | 1093.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-25 09:15:00 | 1061.20 | 1092.08 | 1092.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-18 09:15:00 | 1081.30 | 1070.88 | 1079.52 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-18 09:15:00 | 1081.30 | 1070.88 | 1079.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 09:15:00 | 1081.30 | 1070.88 | 1079.52 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-08-18 11:15:00 | 1073.00 | 1070.99 | 1079.48 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-08-18 12:15:00 | 1077.70 | 1071.05 | 1079.47 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-08-18 14:15:00 | 1071.30 | 1071.10 | 1079.41 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-18 15:15:00 | 1071.30 | 1071.10 | 1079.37 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-08-19 13:15:00 | 1086.40 | 1071.42 | 1079.33 | SL hit qty=1.00 sl=1086.40 alert=retest2 |
| Cross detected — sustain check pending | 2025-08-28 09:15:00 | 1070.30 | 1076.38 | 1080.71 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-28 10:15:00 | 1071.20 | 1076.33 | 1080.67 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-09-02 09:15:00 | 1086.40 | 1074.85 | 1079.47 | SL hit qty=1.00 sl=1086.40 alert=retest2 |
| Cross detected — sustain check pending | 2025-09-04 14:15:00 | 1070.80 | 1077.62 | 1080.51 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-04 15:15:00 | 1071.10 | 1077.56 | 1080.47 | SELL ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-09-05 14:15:00 | 1071.70 | 1077.21 | 1080.21 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-05 15:15:00 | 1072.90 | 1077.17 | 1080.17 | SELL ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 11:15:00 | 1078.90 | 1077.19 | 1080.14 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-09-08 14:15:00 | 1073.20 | 1077.19 | 1080.09 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-08 15:15:00 | 1074.20 | 1077.16 | 1080.06 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-09-09 12:15:00 | 1075.00 | 1077.02 | 1079.93 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-09-09 13:15:00 | 1080.10 | 1077.05 | 1079.93 | ENTRY2 sustain failed after 60m |
| Stop hit — per-position SL triggered | 2025-09-09 14:15:00 | 1081.30 | 1077.10 | 1079.95 | SL hit qty=1.00 sl=1081.30 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-10 09:15:00 | 1086.40 | 1077.25 | 1079.99 | SL hit qty=1.00 sl=1086.40 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-10 09:15:00 | 1086.40 | 1077.25 | 1079.99 | SL hit qty=1.00 sl=1086.40 alert=retest2 |

### Cycle 7 — BUY (started 2025-09-15 14:15:00)

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

### Cycle 8 — SELL (started 2026-02-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-09 10:15:00 | 1160.40 | 1161.93 | 1161.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-09 11:15:00 | 1157.40 | 1161.89 | 1161.91 | Break + close below crossover candle low |

### Cycle 9 — BUY (started 2026-02-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-09 12:15:00 | 1169.90 | 1161.97 | 1161.95 | EMA200 above EMA400 |

### Cycle 10 — SELL (started 2026-02-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-10 12:15:00 | 1153.60 | 1161.91 | 1161.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-10 13:15:00 | 1152.00 | 1161.81 | 1161.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-18 10:15:00 | 1157.90 | 1156.44 | 1158.94 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-18 10:15:00 | 1157.90 | 1156.44 | 1158.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 10:15:00 | 1157.90 | 1156.44 | 1158.94 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2026-02-27 09:15:00 | 1142.70 | 1160.07 | 1160.43 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-27 10:15:00 | 1137.90 | 1159.85 | 1160.31 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2026-04-22 09:15:00 | 1159.60 | 1088.70 | 1104.64 | SL hit qty=1.00 sl=1159.60 alert=retest2 |
| Cross detected — sustain check pending | 2026-04-28 13:15:00 | 1148.10 | 1111.14 | 1114.47 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-04-28 14:15:00 | 1148.50 | 1111.51 | 1114.64 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2026-04-28 15:15:00 | 1144.00 | 1111.83 | 1114.78 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-04-29 09:15:00 | 1156.70 | 1112.28 | 1114.99 | ENTRY2 sustain failed after 1080m |
| Cross detected — sustain check pending | 2026-04-30 10:15:00 | 1145.40 | 1116.04 | 1116.80 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-04-30 11:15:00 | 1148.60 | 1116.36 | 1116.96 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2026-04-30 12:15:00 | 1141.70 | 1116.62 | 1117.09 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-30 13:15:00 | 1146.00 | 1116.91 | 1117.23 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2026-05-04 09:15:00 | 1163.00 | 1117.93 | 1117.74 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 11 — BUY (started 2026-05-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-04 09:15:00 | 1163.00 | 1117.93 | 1117.74 | EMA200 above EMA400 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2024-05-24 10:15:00 | 1102.50 | 2024-06-05 09:15:00 | 1118.30 | STOP_HIT | 1.00 | -1.43% |
| SELL | retest2 | 2024-05-24 13:15:00 | 1101.00 | 2024-06-05 09:15:00 | 1118.30 | STOP_HIT | 1.00 | -1.57% |
| SELL | retest2 | 2024-06-20 10:15:00 | 1098.20 | 2024-07-03 09:15:00 | 1118.30 | STOP_HIT | 1.00 | -1.83% |
| SELL | retest2 | 2024-06-21 10:15:00 | 1096.80 | 2024-07-03 09:15:00 | 1118.30 | STOP_HIT | 1.00 | -1.96% |
| BUY | retest1 | 2024-08-16 13:15:00 | 1188.90 | 2024-09-06 09:15:00 | 1173.17 | STOP_HIT | 1.00 | -1.32% |
| BUY | retest1 | 2024-08-22 10:15:00 | 1200.20 | 2024-09-06 09:15:00 | 1173.17 | STOP_HIT | 1.00 | -2.25% |
| BUY | retest2 | 2024-09-09 15:15:00 | 1192.05 | 2024-10-03 09:15:00 | 1171.55 | STOP_HIT | 1.00 | -1.72% |
| BUY | retest2 | 2024-09-26 10:15:00 | 1204.95 | 2024-10-03 09:15:00 | 1171.55 | STOP_HIT | 1.00 | -2.77% |
| SELL | retest2 | 2025-01-15 10:15:00 | 953.00 | 2025-01-23 09:15:00 | 981.60 | STOP_HIT | 1.00 | -3.00% |
| SELL | retest2 | 2025-01-20 14:15:00 | 960.05 | 2025-01-23 09:15:00 | 981.60 | STOP_HIT | 1.00 | -2.24% |
| SELL | retest2 | 2025-01-28 14:15:00 | 961.60 | 2025-01-31 09:15:00 | 981.60 | STOP_HIT | 1.00 | -2.08% |
| SELL | retest2 | 2025-03-03 11:15:00 | 954.55 | 2025-03-07 14:15:00 | 962.05 | STOP_HIT | 1.00 | -0.79% |
| SELL | retest1 | 2025-03-25 11:15:00 | 959.30 | 2025-03-27 15:15:00 | 976.72 | STOP_HIT | 1.00 | -1.82% |
| SELL | retest1 | 2025-03-26 11:15:00 | 962.00 | 2025-03-27 15:15:00 | 976.72 | STOP_HIT | 1.00 | -1.53% |
| BUY | retest1 | 2025-05-09 12:15:00 | 1112.60 | 2025-06-12 10:15:00 | 1097.34 | STOP_HIT | 1.00 | -1.37% |
| BUY | retest1 | 2025-05-15 11:15:00 | 1129.20 | 2025-06-12 10:15:00 | 1097.34 | STOP_HIT | 1.00 | -2.82% |
| BUY | retest1 | 2025-05-29 15:15:00 | 1109.80 | 2025-06-12 10:15:00 | 1097.34 | STOP_HIT | 1.00 | -1.12% |
| BUY | retest1 | 2025-05-30 13:15:00 | 1111.70 | 2025-06-12 10:15:00 | 1097.34 | STOP_HIT | 1.00 | -1.29% |
| BUY | retest2 | 2025-06-23 11:15:00 | 1101.60 | 2025-06-30 13:15:00 | 1090.90 | STOP_HIT | 1.00 | -0.97% |
| BUY | retest2 | 2025-07-07 11:15:00 | 1101.80 | 2025-07-10 14:15:00 | 1090.90 | STOP_HIT | 1.00 | -0.99% |
| BUY | retest2 | 2025-07-08 09:15:00 | 1105.50 | 2025-07-10 14:15:00 | 1090.90 | STOP_HIT | 1.00 | -1.32% |
| BUY | retest2 | 2025-07-09 11:15:00 | 1105.80 | 2025-07-10 14:15:00 | 1090.90 | STOP_HIT | 1.00 | -1.35% |
| BUY | retest2 | 2025-07-17 14:15:00 | 1105.30 | 2025-07-18 12:15:00 | 1095.20 | STOP_HIT | 1.00 | -0.91% |
| SELL | retest2 | 2025-08-18 15:15:00 | 1071.30 | 2025-08-19 13:15:00 | 1086.40 | STOP_HIT | 1.00 | -1.41% |
| SELL | retest2 | 2025-08-28 10:15:00 | 1071.20 | 2025-09-02 09:15:00 | 1086.40 | STOP_HIT | 1.00 | -1.42% |
| SELL | retest2 | 2025-09-04 15:15:00 | 1071.10 | 2025-09-09 14:15:00 | 1081.30 | STOP_HIT | 1.00 | -0.95% |
| SELL | retest2 | 2025-09-05 15:15:00 | 1072.90 | 2025-09-10 09:15:00 | 1086.40 | STOP_HIT | 1.00 | -1.26% |
| SELL | retest2 | 2025-09-08 15:15:00 | 1074.20 | 2025-09-10 09:15:00 | 1086.40 | STOP_HIT | 1.00 | -1.14% |
| BUY | retest1 | 2025-10-16 12:15:00 | 1133.10 | 2025-12-03 09:15:00 | 1147.22 | STOP_HIT | 1.00 | 1.25% |
| BUY | retest2 | 2025-12-05 15:15:00 | 1162.90 | 2025-12-08 15:15:00 | 1144.50 | STOP_HIT | 1.00 | -1.58% |
| BUY | retest2 | 2025-12-16 10:15:00 | 1174.40 | 2026-01-28 09:15:00 | 1144.50 | STOP_HIT | 1.00 | -2.55% |
| BUY | retest2 | 2026-01-21 15:15:00 | 1163.10 | 2026-01-28 09:15:00 | 1144.50 | STOP_HIT | 1.00 | -1.60% |
| BUY | retest2 | 2026-01-27 12:15:00 | 1175.70 | 2026-01-28 09:15:00 | 1144.50 | STOP_HIT | 1.00 | -2.65% |
| SELL | retest2 | 2026-02-27 10:15:00 | 1137.90 | 2026-04-22 09:15:00 | 1159.60 | STOP_HIT | 1.00 | -1.91% |
| SELL | retest2 | 2026-04-30 13:15:00 | 1146.00 | 2026-05-04 09:15:00 | 1163.00 | STOP_HIT | 1.00 | -1.48% |
