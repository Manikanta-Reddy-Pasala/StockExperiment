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
| CROSSOVER | 11 |
| ALERT1 | 11 |
| ALERT2 | 11 |
| ALERT2_SKIP | 10 |
| ALERT3 | 13 |
| PENDING | 28 |
| PENDING_CANCEL | 9 |
| ENTRY1 | 2 |
| ENTRY2 | 17 |
| PARTIAL | 6 |
| TARGET_HIT | 0 |
| STOP_HIT | 19 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 25 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 11 / 14
- **Target hits / Stop hits / Partials:** 0 / 19 / 6
- **Avg / median % per leg:** 4.21% / -0.51%
- **Sum % (uncompounded):** 105.36%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 12 | 3 | 25.0% | 0 | 10 | 2 | 2.22% | 26.7% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 12 | 3 | 25.0% | 0 | 10 | 2 | 2.22% | 26.7% |
| SELL (all) | 13 | 8 | 61.5% | 0 | 9 | 4 | 6.05% | 78.7% |
| SELL @ 2nd Alert (retest1) | 2 | 0 | 0.0% | 0 | 2 | 0 | -1.04% | -2.1% |
| SELL @ 3rd Alert (retest2) | 11 | 8 | 72.7% | 0 | 7 | 4 | 7.34% | 80.8% |
| retest1 (combined) | 2 | 0 | 0.0% | 0 | 2 | 0 | -1.04% | -2.1% |
| retest2 (combined) | 23 | 11 | 47.8% | 0 | 17 | 6 | 4.67% | 107.4% |

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
| Stop hit — per-position SL triggered | 2024-03-21 13:15:00 | 1031.90 | 1075.66 | 1075.85 | Force close (CROSSOVER_FLIP) qty=0.50 alert=retest2 |

### Cycle 2 — SELL (started 2024-03-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-21 13:15:00 | 1031.90 | 1075.66 | 1075.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-18 13:15:00 | 1026.10 | 1065.63 | 1068.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-24 09:15:00 | 1064.60 | 1060.46 | 1065.79 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-24 09:15:00 | 1064.60 | 1060.46 | 1065.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-24 09:15:00 | 1064.60 | 1060.46 | 1065.79 | EMA400 retest candle locked |

### Cycle 3 — BUY (started 2024-04-29 10:15:00)

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
| Stop hit — per-position SL triggered | 2024-08-13 13:15:00 | 1162.95 | 1194.48 | 1194.50 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-08-13 13:15:00 | 1162.95 | 1194.48 | 1194.50 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-08-13 13:15:00 | 1162.95 | 1194.48 | 1194.50 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-08-13 13:15:00 | 1162.95 | 1194.48 | 1194.50 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 4 — SELL (started 2024-08-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-13 13:15:00 | 1162.95 | 1194.48 | 1194.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-13 14:15:00 | 1158.95 | 1194.12 | 1194.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-27 12:15:00 | 1182.05 | 1181.74 | 1187.00 | EMA200 retest candle locked |
| Cross detected — sustain check pending | 2024-08-28 09:15:00 | 1174.50 | 1181.65 | 1186.85 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-28 10:15:00 | 1172.00 | 1181.55 | 1186.77 | SELL ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2024-08-29 14:15:00 | 1174.40 | 1180.70 | 1186.06 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-29 15:15:00 | 1175.00 | 1180.64 | 1186.00 | SELL ENTRY1 attempt 2/4 (retest1 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-02 10:15:00 | 1183.00 | 1180.44 | 1185.66 | EMA400 retest candle locked |
| Stop hit — per-position SL triggered | 2024-09-02 10:15:00 | 1185.66 | 1180.44 | 1185.66 | SL hit qty=1.00 sl=1185.66 alert=retest1 |
| Stop hit — per-position SL triggered | 2024-09-02 10:15:00 | 1185.66 | 1180.44 | 1185.66 | SL hit qty=1.00 sl=1185.66 alert=retest1 |
| Cross detected — sustain check pending | 2024-09-03 10:15:00 | 1178.35 | 1180.76 | 1185.64 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2024-09-03 11:15:00 | 1181.85 | 1180.77 | 1185.62 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2024-09-04 10:15:00 | 1175.35 | 1180.96 | 1185.58 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-04 11:15:00 | 1176.55 | 1180.92 | 1185.53 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2024-09-06 09:15:00 | 1166.35 | 1180.52 | 1185.06 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-06 10:15:00 | 1170.05 | 1180.42 | 1184.99 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2024-09-10 09:15:00 | 1186.35 | 1178.29 | 1183.59 | SL hit qty=1.00 sl=1186.35 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-09-10 09:15:00 | 1186.35 | 1178.29 | 1183.59 | SL hit qty=1.00 sl=1186.35 alert=retest2 |

### Cycle 5 — BUY (started 2024-09-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-17 13:15:00 | 1235.45 | 1187.97 | 1187.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-18 11:15:00 | 1242.40 | 1190.30 | 1188.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-03 09:15:00 | 1201.15 | 1219.05 | 1206.16 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-03 09:15:00 | 1201.15 | 1219.05 | 1206.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-03 09:15:00 | 1201.15 | 1219.05 | 1206.16 | EMA400 retest candle locked |

### Cycle 6 — SELL (started 2024-10-15 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-15 15:15:00 | 1158.00 | 1196.57 | 1196.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-16 11:15:00 | 1150.60 | 1195.33 | 1196.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-18 11:15:00 | 1194.20 | 1189.49 | 1193.03 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-18 11:15:00 | 1194.20 | 1189.49 | 1193.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 11:15:00 | 1194.20 | 1189.49 | 1193.03 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2024-10-22 12:15:00 | 1181.00 | 1189.91 | 1193.00 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2024-10-22 13:15:00 | 1186.70 | 1189.88 | 1192.97 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2024-10-22 14:15:00 | 1175.45 | 1189.74 | 1192.88 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-22 15:15:00 | 1175.75 | 1189.60 | 1192.79 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2024-10-25 09:15:00 | 1198.95 | 1186.55 | 1190.98 | SL hit qty=1.00 sl=1198.95 alert=retest2 |
| Cross detected — sustain check pending | 2024-10-28 10:15:00 | 1177.50 | 1186.06 | 1190.55 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2024-10-28 11:15:00 | 1184.65 | 1186.04 | 1190.52 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2024-10-28 12:15:00 | 1179.35 | 1185.98 | 1190.47 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-28 13:15:00 | 1173.40 | 1185.85 | 1190.38 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2024-10-30 09:15:00 | 1173.05 | 1185.02 | 1189.73 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-30 10:15:00 | 1174.90 | 1184.92 | 1189.66 | SELL ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2024-12-09 09:15:00 | 1171.75 | 1158.08 | 1167.77 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-09 10:15:00 | 1175.55 | 1158.25 | 1167.81 | SELL ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-09 11:15:00 | 1172.75 | 1158.40 | 1167.83 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2024-12-09 13:15:00 | 1165.40 | 1158.62 | 1167.85 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-09 14:15:00 | 1162.65 | 1158.66 | 1167.82 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Partial book — 50% qty @ 15%, trail SL → entry | 2025-01-17 09:15:00 | 997.39 | 1083.67 | 1113.00 | Partial book 0.50 @ 15%; trail SL->entry alert=retest2 |
| Partial book — 50% qty @ 15%, trail SL → entry | 2025-01-17 09:15:00 | 998.67 | 1083.67 | 1113.00 | Partial book 0.50 @ 15%; trail SL->entry alert=retest2 |
| Partial book — 50% qty @ 15%, trail SL → entry | 2025-01-17 09:15:00 | 999.22 | 1083.67 | 1113.00 | Partial book 0.50 @ 15%; trail SL->entry alert=retest2 |
| Partial book — 50% qty @ 15%, trail SL → entry | 2025-01-17 09:15:00 | 988.25 | 1083.67 | 1113.00 | Partial book 0.50 @ 15%; trail SL->entry alert=retest2 |
| Stop hit — per-position SL triggered | 2025-03-28 10:15:00 | 1098.45 | 1045.79 | 1045.61 | Force close (CROSSOVER_FLIP) qty=0.50 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-03-28 10:15:00 | 1098.45 | 1045.79 | 1045.61 | Force close (CROSSOVER_FLIP) qty=0.50 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-03-28 10:15:00 | 1098.45 | 1045.79 | 1045.61 | Force close (CROSSOVER_FLIP) qty=0.50 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-03-28 10:15:00 | 1098.45 | 1045.79 | 1045.61 | Force close (CROSSOVER_FLIP) qty=0.50 alert=retest2 |

### Cycle 7 — BUY (started 2025-03-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-28 10:15:00 | 1098.45 | 1045.79 | 1045.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-28 14:15:00 | 1103.55 | 1047.95 | 1046.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-07 09:15:00 | 1046.55 | 1057.97 | 1052.26 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-07 09:15:00 | 1046.55 | 1057.97 | 1052.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-07 09:15:00 | 1046.55 | 1057.97 | 1052.26 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-04-08 10:15:00 | 1072.15 | 1057.34 | 1052.16 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-08 11:15:00 | 1079.40 | 1057.56 | 1052.30 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-04-11 12:15:00 | 1070.50 | 1059.12 | 1053.50 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-04-11 13:15:00 | 1066.50 | 1059.20 | 1053.56 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-04-11 14:15:00 | 1069.75 | 1059.30 | 1053.64 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-04-11 15:15:00 | 1068.95 | 1059.40 | 1053.72 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-04-15 09:15:00 | 1100.60 | 1059.81 | 1053.95 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-15 10:15:00 | 1106.60 | 1060.27 | 1054.21 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Partial book — 50% qty @ 15%, trail SL → entry | 2025-06-27 09:15:00 | 1241.31 | 1204.97 | 1178.60 | Partial book 0.50 @ 15%; trail SL->entry alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-21 09:15:00 | 1079.40 | 1178.19 | 1174.06 | SL hit qty=0.50 sl=1079.40 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-22 13:15:00 | 1101.00 | 1170.02 | 1170.08 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 8 — SELL (started 2025-07-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-22 13:15:00 | 1101.00 | 1170.02 | 1170.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-22 14:15:00 | 1097.60 | 1169.30 | 1169.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-11 10:15:00 | 1074.20 | 1073.30 | 1097.22 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-12 09:15:00 | 1102.30 | 1074.20 | 1096.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 09:15:00 | 1102.30 | 1074.20 | 1096.96 | EMA400 retest candle locked |

### Cycle 9 — BUY (started 2025-09-29 11:15:00)

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

### Cycle 10 — SELL (started 2026-03-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 13:15:00 | 1210.80 | 1298.57 | 1298.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 14:15:00 | 1204.20 | 1297.63 | 1298.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-07 14:15:00 | 1250.90 | 1249.40 | 1269.51 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-08 09:15:00 | 1316.00 | 1250.07 | 1269.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 09:15:00 | 1316.00 | 1250.07 | 1269.65 | EMA400 retest candle locked |

### Cycle 11 — BUY (started 2026-04-17 13:15:00)

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
| BUY | retest2 | 2023-10-26 15:15:00 | 972.05 | 2024-03-21 13:15:00 | 1031.90 | STOP_HIT | 0.50 | 6.16% |
| BUY | retest2 | 2024-06-05 14:15:00 | 1185.00 | 2024-08-13 13:15:00 | 1162.95 | STOP_HIT | 1.00 | -1.86% |
| BUY | retest2 | 2024-06-06 14:15:00 | 1170.95 | 2024-08-13 13:15:00 | 1162.95 | STOP_HIT | 1.00 | -0.68% |
| BUY | retest2 | 2024-07-25 15:15:00 | 1176.00 | 2024-08-13 13:15:00 | 1162.95 | STOP_HIT | 1.00 | -1.11% |
| BUY | retest2 | 2024-07-26 13:15:00 | 1178.15 | 2024-08-13 13:15:00 | 1162.95 | STOP_HIT | 1.00 | -1.29% |
| SELL | retest1 | 2024-08-28 10:15:00 | 1172.00 | 2024-09-02 10:15:00 | 1185.66 | STOP_HIT | 1.00 | -1.17% |
| SELL | retest1 | 2024-08-29 15:15:00 | 1175.00 | 2024-09-02 10:15:00 | 1185.66 | STOP_HIT | 1.00 | -0.91% |
| SELL | retest2 | 2024-09-04 11:15:00 | 1176.55 | 2024-09-10 09:15:00 | 1186.35 | STOP_HIT | 1.00 | -0.83% |
| SELL | retest2 | 2024-09-06 10:15:00 | 1170.05 | 2024-09-10 09:15:00 | 1186.35 | STOP_HIT | 1.00 | -1.39% |
| SELL | retest2 | 2024-10-22 15:15:00 | 1175.75 | 2024-10-25 09:15:00 | 1198.95 | STOP_HIT | 1.00 | -1.97% |
| SELL | retest2 | 2024-10-28 13:15:00 | 1173.40 | 2025-01-17 09:15:00 | 997.39 | PARTIAL | 0.50 | 15.00% |
| SELL | retest2 | 2024-10-30 10:15:00 | 1174.90 | 2025-01-17 09:15:00 | 998.67 | PARTIAL | 0.50 | 15.00% |
| SELL | retest2 | 2024-12-09 10:15:00 | 1175.55 | 2025-01-17 09:15:00 | 999.22 | PARTIAL | 0.50 | 15.00% |
| SELL | retest2 | 2024-12-09 14:15:00 | 1162.65 | 2025-01-17 09:15:00 | 988.25 | PARTIAL | 0.50 | 15.00% |
| SELL | retest2 | 2024-10-28 13:15:00 | 1173.40 | 2025-03-28 10:15:00 | 1098.45 | STOP_HIT | 0.50 | 6.39% |
| SELL | retest2 | 2024-10-30 10:15:00 | 1174.90 | 2025-03-28 10:15:00 | 1098.45 | STOP_HIT | 0.50 | 6.51% |
| SELL | retest2 | 2024-12-09 10:15:00 | 1175.55 | 2025-03-28 10:15:00 | 1098.45 | STOP_HIT | 0.50 | 6.56% |
| SELL | retest2 | 2024-12-09 14:15:00 | 1162.65 | 2025-03-28 10:15:00 | 1098.45 | STOP_HIT | 0.50 | 5.52% |
| BUY | retest2 | 2025-04-08 11:15:00 | 1079.40 | 2025-06-27 09:15:00 | 1241.31 | PARTIAL | 0.50 | 15.00% |
| BUY | retest2 | 2025-04-08 11:15:00 | 1079.40 | 2025-07-21 09:15:00 | 1079.40 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest2 | 2025-04-15 10:15:00 | 1106.60 | 2025-07-22 13:15:00 | 1101.00 | STOP_HIT | 1.00 | -0.51% |
| BUY | retest2 | 2025-12-30 10:15:00 | 1240.60 | 2026-03-13 09:15:00 | 1228.60 | STOP_HIT | 1.00 | -0.97% |
| BUY | retest2 | 2026-03-18 10:15:00 | 1247.40 | 2026-03-19 09:15:00 | 1228.60 | STOP_HIT | 1.00 | -1.51% |
