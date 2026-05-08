# AXISBANK (AXISBANK)

## Backtest Summary

- **Window:** 2023-06-08 09:15:00 → 2026-05-08 15:30:00 (4997 bars)
- **Last close:** 1268.30
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 30%)
- **Partial:** 50% qty booked @ 15% (ENTRY1) / 15% (ENTRY2), trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 11 |
| ALERT1 | 11 |
| ALERT2 | 11 |
| ALERT2_SKIP | 10 |
| ALERT3 | 13 |
| PENDING | 28 |
| PENDING_CANCEL | 8 |
| ENTRY1 | 2 |
| ENTRY2 | 18 |
| PARTIAL | 8 |
| TARGET_HIT | 0 |
| STOP_HIT | 20 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 28 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 16 / 12
- **Target hits / Stop hits / Partials:** 0 / 20 / 8
- **Avg / median % per leg:** 7.07% / 11.76%
- **Sum % (uncompounded):** 198.00%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 14 | 6 | 42.9% | 0 | 11 | 3 | 4.67% | 65.3% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 14 | 6 | 42.9% | 0 | 11 | 3 | 4.67% | 65.3% |
| SELL (all) | 14 | 10 | 71.4% | 0 | 9 | 5 | 9.48% | 132.7% |
| SELL @ 2nd Alert (retest1) | 2 | 0 | 0.0% | 0 | 2 | 0 | -1.50% | -3.0% |
| SELL @ 3rd Alert (retest2) | 12 | 10 | 83.3% | 0 | 7 | 5 | 11.31% | 135.7% |
| retest1 (combined) | 2 | 0 | 0.0% | 0 | 2 | 0 | -1.50% | -3.0% |
| retest2 (combined) | 26 | 16 | 61.5% | 0 | 18 | 8 | 7.73% | 201.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-08-31 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-31 14:15:00 | 974.20 | 964.08 | 964.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-01 12:15:00 | 984.30 | 964.61 | 964.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-06 13:15:00 | 966.65 | 967.90 | 966.09 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-06 13:15:00 | 966.65 | 967.90 | 966.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-06 13:15:00 | 966.65 | 967.90 | 966.09 | EMA400 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2023-09-07 14:15:00 | 977.45 | 967.96 | 966.18 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-07 15:15:00 | 978.20 | 968.06 | 966.24 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2023-10-23 14:15:00 | 962.10 | 1000.52 | 993.02 | SL hit (close<static) qty=1.00 sl=965.85 alert=retest2 |
| Cross detected — sustain check pending | 2023-10-27 09:15:00 | 980.25 | 995.31 | 990.89 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-27 10:15:00 | 984.65 | 995.21 | 990.86 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2023-11-01 14:15:00 | 972.35 | 992.60 | 990.01 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2023-11-01 15:15:00 | 971.40 | 992.39 | 989.91 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2023-11-02 09:15:00 | 980.75 | 992.27 | 989.87 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-02 10:15:00 | 980.80 | 992.16 | 989.82 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-04 14:15:00 | 1127.92 | 1022.20 | 1008.83 | Partial book 0.50 @ 15%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-05 09:15:00 | 1132.35 | 1024.39 | 1010.06 | Partial book 0.50 @ 15%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-01-02 12:15:00 | 1083.65 | 1085.98 | 1059.01 | SL hit (close<ema200) qty=0.50 sl=1085.98 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-01-02 12:15:00 | 1083.65 | 1085.98 | 1059.01 | SL hit (close<ema200) qty=0.50 sl=1085.98 alert=retest2 |

### Cycle 2 — SELL (started 2024-03-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-21 13:15:00 | 1031.90 | 1075.66 | 1075.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-18 13:15:00 | 1026.10 | 1065.63 | 1068.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-24 09:15:00 | 1064.60 | 1060.46 | 1065.80 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-24 09:15:00 | 1064.60 | 1060.46 | 1065.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-24 09:15:00 | 1064.60 | 1060.46 | 1065.80 | EMA400 retest candle locked (from downside) |

### Cycle 3 — BUY (started 2024-04-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-29 10:15:00 | 1142.50 | 1070.75 | 1070.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-29 11:15:00 | 1150.95 | 1071.55 | 1070.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-04 11:15:00 | 1124.95 | 1140.91 | 1117.52 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-04 11:15:00 | 1124.95 | 1140.91 | 1117.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 11:15:00 | 1124.95 | 1140.91 | 1117.52 | EMA400 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2024-06-05 13:15:00 | 1171.60 | 1141.28 | 1118.73 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-05 14:15:00 | 1185.00 | 1141.71 | 1119.06 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2024-06-06 13:15:00 | 1169.85 | 1143.80 | 1120.79 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-06 14:15:00 | 1170.95 | 1144.07 | 1121.04 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2024-07-25 14:15:00 | 1175.10 | 1254.39 | 1217.31 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-25 15:15:00 | 1176.00 | 1253.61 | 1217.10 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2024-07-26 12:15:00 | 1173.65 | 1250.32 | 1216.16 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-26 13:15:00 | 1178.15 | 1249.60 | 1215.97 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-26 14:15:00 | 1177.30 | 1248.88 | 1215.78 | EMA400 retest candle locked (from upside) |
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
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-27 12:15:00 | 1182.05 | 1181.74 | 1187.00 | EMA200 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2024-08-28 09:15:00 | 1174.50 | 1181.65 | 1186.85 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-28 10:15:00 | 1172.00 | 1181.55 | 1186.77 | SELL ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2024-08-29 14:15:00 | 1174.40 | 1180.70 | 1186.06 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-29 15:15:00 | 1175.00 | 1180.64 | 1186.00 | SELL ENTRY1 attempt 2/4 (retest1 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-02 10:15:00 | 1183.00 | 1180.44 | 1185.66 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-09-02 13:15:00 | 1191.10 | 1180.57 | 1185.65 | SL hit (close>ema400) qty=1.00 sl=1185.65 alert=retest1 |
| Stop hit — per-position SL triggered | 2024-09-02 13:15:00 | 1191.10 | 1180.57 | 1185.65 | SL hit (close>ema400) qty=1.00 sl=1185.65 alert=retest1 |
| Cross detected — sustain check pending | 2024-09-03 10:15:00 | 1178.35 | 1180.76 | 1185.64 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2024-09-03 11:15:00 | 1181.85 | 1180.77 | 1185.62 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2024-09-04 10:15:00 | 1175.35 | 1180.96 | 1185.58 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-04 11:15:00 | 1176.55 | 1180.92 | 1185.53 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2024-09-06 09:15:00 | 1166.35 | 1180.52 | 1185.06 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-06 10:15:00 | 1170.05 | 1180.42 | 1184.99 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2024-09-10 11:15:00 | 1187.75 | 1178.46 | 1183.63 | SL hit (close>static) qty=1.00 sl=1186.35 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-09-10 11:15:00 | 1187.75 | 1178.46 | 1183.63 | SL hit (close>static) qty=1.00 sl=1186.35 alert=retest2 |

### Cycle 5 — BUY (started 2024-09-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-17 13:15:00 | 1235.45 | 1187.97 | 1187.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-18 11:15:00 | 1242.40 | 1190.30 | 1188.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-03 09:15:00 | 1201.15 | 1219.05 | 1206.16 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-03 09:15:00 | 1201.15 | 1219.05 | 1206.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-03 09:15:00 | 1201.15 | 1219.05 | 1206.16 | EMA400 retest candle locked (from upside) |

### Cycle 6 — SELL (started 2024-10-15 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-15 15:15:00 | 1158.00 | 1196.57 | 1196.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-16 11:15:00 | 1150.60 | 1195.33 | 1196.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-18 11:15:00 | 1194.20 | 1189.49 | 1193.03 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-18 11:15:00 | 1194.20 | 1189.49 | 1193.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 11:15:00 | 1194.20 | 1189.49 | 1193.03 | EMA400 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2024-10-22 12:15:00 | 1181.00 | 1189.91 | 1193.00 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2024-10-22 13:15:00 | 1186.70 | 1189.88 | 1192.97 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2024-10-22 14:15:00 | 1175.45 | 1189.74 | 1192.88 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-22 15:15:00 | 1175.75 | 1189.60 | 1192.79 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2024-10-28 10:15:00 | 1177.50 | 1186.06 | 1190.55 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2024-10-28 11:15:00 | 1184.65 | 1186.04 | 1190.52 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2024-10-28 12:15:00 | 1179.35 | 1185.98 | 1190.47 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-28 13:15:00 | 1173.40 | 1185.85 | 1190.38 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2024-10-30 09:15:00 | 1173.05 | 1185.02 | 1189.73 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-30 10:15:00 | 1174.90 | 1184.92 | 1189.66 | SELL ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2024-12-09 09:15:00 | 1171.75 | 1158.08 | 1167.77 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-09 10:15:00 | 1175.55 | 1158.25 | 1167.81 | SELL ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-09 11:15:00 | 1172.75 | 1158.40 | 1167.83 | EMA400 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2024-12-09 13:15:00 | 1165.40 | 1158.62 | 1167.85 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-09 14:15:00 | 1162.65 | 1158.66 | 1167.82 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-17 09:15:00 | 999.39 | 1083.67 | 1113.00 | Partial book 0.50 @ 15%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-17 09:15:00 | 997.39 | 1083.67 | 1113.00 | Partial book 0.50 @ 15%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-17 09:15:00 | 998.67 | 1083.67 | 1113.00 | Partial book 0.50 @ 15%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-17 09:15:00 | 999.22 | 1083.67 | 1113.00 | Partial book 0.50 @ 15%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-17 09:15:00 | 988.25 | 1083.67 | 1113.00 | Partial book 0.50 @ 15%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-02-07 12:15:00 | 1024.35 | 1021.00 | 1060.11 | SL hit (close>ema200) qty=0.50 sl=1021.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-02-07 12:15:00 | 1024.35 | 1021.00 | 1060.11 | SL hit (close>ema200) qty=0.50 sl=1021.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-02-07 12:15:00 | 1024.35 | 1021.00 | 1060.11 | SL hit (close>ema200) qty=0.50 sl=1021.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-02-07 12:15:00 | 1024.35 | 1021.00 | 1060.11 | SL hit (close>ema200) qty=0.50 sl=1021.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-02-07 12:15:00 | 1024.35 | 1021.00 | 1060.11 | SL hit (close>ema200) qty=0.50 sl=1021.00 alert=retest2 |

### Cycle 7 — BUY (started 2025-03-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-28 10:15:00 | 1098.45 | 1045.79 | 1045.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-28 14:15:00 | 1103.55 | 1047.95 | 1046.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-07 09:15:00 | 1046.55 | 1057.97 | 1052.26 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-07 09:15:00 | 1046.55 | 1057.97 | 1052.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-07 09:15:00 | 1046.55 | 1057.97 | 1052.26 | EMA400 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2025-04-08 10:15:00 | 1072.15 | 1057.34 | 1052.16 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-08 11:15:00 | 1079.40 | 1057.56 | 1052.30 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-04-11 12:15:00 | 1070.50 | 1059.12 | 1053.50 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-04-11 13:15:00 | 1066.50 | 1059.20 | 1053.56 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-04-11 14:15:00 | 1069.75 | 1059.30 | 1053.64 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-04-11 15:15:00 | 1068.95 | 1059.40 | 1053.72 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-04-15 09:15:00 | 1100.60 | 1059.81 | 1053.95 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-15 10:15:00 | 1106.60 | 1060.27 | 1054.21 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-27 09:15:00 | 1241.31 | 1204.97 | 1178.60 | Partial book 0.50 @ 15%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-30 12:15:00 | 1206.30 | 1206.39 | 1180.62 | SL hit (close<ema200) qty=0.50 sl=1206.39 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-22 13:15:00 | 1101.00 | 1170.02 | 1170.08 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 8 — SELL (started 2025-07-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-22 13:15:00 | 1101.00 | 1170.02 | 1170.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-22 14:15:00 | 1097.60 | 1169.30 | 1169.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-11 10:15:00 | 1074.20 | 1073.30 | 1097.22 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-12 09:15:00 | 1102.30 | 1074.20 | 1096.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 09:15:00 | 1102.30 | 1074.20 | 1096.96 | EMA400 retest candle locked (from downside) |

### Cycle 9 — BUY (started 2025-09-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-29 11:15:00 | 1129.80 | 1110.78 | 1110.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-29 12:15:00 | 1133.70 | 1111.00 | 1110.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-16 09:15:00 | 1237.00 | 1261.01 | 1230.43 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-16 11:15:00 | 1236.80 | 1260.51 | 1230.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 11:15:00 | 1236.80 | 1260.51 | 1230.48 | EMA400 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2025-12-30 09:15:00 | 1240.40 | 1246.31 | 1230.21 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-30 10:15:00 | 1240.60 | 1246.25 | 1230.26 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2026-03-13 09:15:00 | 1215.00 | 1327.79 | 1312.24 | SL hit (close<static) qty=1.00 sl=1228.60 alert=retest2 |
| Cross detected — sustain check pending | 2026-03-18 09:15:00 | 1242.50 | 1305.87 | 1302.22 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-18 10:15:00 | 1247.40 | 1305.29 | 1301.95 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2026-03-19 09:15:00 | 1212.20 | 1301.86 | 1300.31 | SL hit (close<static) qty=1.00 sl=1228.60 alert=retest2 |

### Cycle 10 — SELL (started 2026-03-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 13:15:00 | 1210.80 | 1298.57 | 1298.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 14:15:00 | 1204.20 | 1297.63 | 1298.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-07 14:15:00 | 1250.90 | 1249.40 | 1269.51 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-08 09:15:00 | 1316.00 | 1250.07 | 1269.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 09:15:00 | 1316.00 | 1250.07 | 1269.65 | EMA400 retest candle locked (from downside) |

### Cycle 11 — BUY (started 2026-04-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-17 13:15:00 | 1356.50 | 1285.46 | 1285.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-17 14:15:00 | 1357.90 | 1286.18 | 1285.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-28 10:15:00 | 1301.30 | 1310.05 | 1299.05 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-28 11:15:00 | 1296.40 | 1309.92 | 1299.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 11:15:00 | 1296.40 | 1309.92 | 1299.04 | EMA400 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2026-04-29 12:15:00 | 1304.40 | 1308.58 | 1298.78 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-04-29 13:15:00 | 1296.60 | 1308.46 | 1298.77 | ENTRY2 sustain failed after 60m |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2023-09-07 15:15:00 | 978.20 | 2023-10-23 14:15:00 | 962.10 | STOP_HIT | 1.00 | -1.65% |
| BUY | retest2 | 2023-10-27 10:15:00 | 984.65 | 2023-12-04 14:15:00 | 1127.92 | PARTIAL | 0.50 | 14.55% |
| BUY | retest2 | 2023-11-02 10:15:00 | 980.80 | 2023-12-05 09:15:00 | 1132.35 | PARTIAL | 0.50 | 15.45% |
| BUY | retest2 | 2023-10-27 10:15:00 | 984.65 | 2024-01-02 12:15:00 | 1083.65 | STOP_HIT | 0.50 | 10.05% |
| BUY | retest2 | 2023-11-02 10:15:00 | 980.80 | 2024-01-02 12:15:00 | 1083.65 | STOP_HIT | 0.50 | 10.49% |
| BUY | retest2 | 2024-06-05 14:15:00 | 1185.00 | 2024-08-13 13:15:00 | 1162.95 | STOP_HIT | 1.00 | -1.86% |
| BUY | retest2 | 2024-06-06 14:15:00 | 1170.95 | 2024-08-13 13:15:00 | 1162.95 | STOP_HIT | 1.00 | -0.68% |
| BUY | retest2 | 2024-07-25 15:15:00 | 1176.00 | 2024-08-13 13:15:00 | 1162.95 | STOP_HIT | 1.00 | -1.11% |
| BUY | retest2 | 2024-07-26 13:15:00 | 1178.15 | 2024-08-13 13:15:00 | 1162.95 | STOP_HIT | 1.00 | -1.29% |
| SELL | retest1 | 2024-08-28 10:15:00 | 1172.00 | 2024-09-02 13:15:00 | 1191.10 | STOP_HIT | 1.00 | -1.63% |
| SELL | retest1 | 2024-08-29 15:15:00 | 1175.00 | 2024-09-02 13:15:00 | 1191.10 | STOP_HIT | 1.00 | -1.37% |
| SELL | retest2 | 2024-09-04 11:15:00 | 1176.55 | 2024-09-10 11:15:00 | 1187.75 | STOP_HIT | 1.00 | -0.95% |
| SELL | retest2 | 2024-09-06 10:15:00 | 1170.05 | 2024-09-10 11:15:00 | 1187.75 | STOP_HIT | 1.00 | -1.51% |
| SELL | retest2 | 2024-10-22 15:15:00 | 1175.75 | 2025-01-17 09:15:00 | 999.39 | PARTIAL | 0.50 | 15.00% |
| SELL | retest2 | 2024-10-28 13:15:00 | 1173.40 | 2025-01-17 09:15:00 | 997.39 | PARTIAL | 0.50 | 15.00% |
| SELL | retest2 | 2024-10-30 10:15:00 | 1174.90 | 2025-01-17 09:15:00 | 998.67 | PARTIAL | 0.50 | 15.00% |
| SELL | retest2 | 2024-12-09 10:15:00 | 1175.55 | 2025-01-17 09:15:00 | 999.22 | PARTIAL | 0.50 | 15.00% |
| SELL | retest2 | 2024-12-09 14:15:00 | 1162.65 | 2025-01-17 09:15:00 | 988.25 | PARTIAL | 0.50 | 15.00% |
| SELL | retest2 | 2024-10-22 15:15:00 | 1175.75 | 2025-02-07 12:15:00 | 1024.35 | STOP_HIT | 0.50 | 12.88% |
| SELL | retest2 | 2024-10-28 13:15:00 | 1173.40 | 2025-02-07 12:15:00 | 1024.35 | STOP_HIT | 0.50 | 12.70% |
| SELL | retest2 | 2024-10-30 10:15:00 | 1174.90 | 2025-02-07 12:15:00 | 1024.35 | STOP_HIT | 0.50 | 12.81% |
| SELL | retest2 | 2024-12-09 10:15:00 | 1175.55 | 2025-02-07 12:15:00 | 1024.35 | STOP_HIT | 0.50 | 12.86% |
| SELL | retest2 | 2024-12-09 14:15:00 | 1162.65 | 2025-02-07 12:15:00 | 1024.35 | STOP_HIT | 0.50 | 11.90% |
| BUY | retest2 | 2025-04-08 11:15:00 | 1079.40 | 2025-06-27 09:15:00 | 1241.31 | PARTIAL | 0.50 | 15.00% |
| BUY | retest2 | 2025-04-08 11:15:00 | 1079.40 | 2025-06-30 12:15:00 | 1206.30 | STOP_HIT | 0.50 | 11.76% |
| BUY | retest2 | 2025-04-15 10:15:00 | 1106.60 | 2025-07-22 13:15:00 | 1101.00 | STOP_HIT | 1.00 | -0.51% |
| BUY | retest2 | 2025-12-30 10:15:00 | 1240.60 | 2026-03-13 09:15:00 | 1215.00 | STOP_HIT | 1.00 | -2.06% |
| BUY | retest2 | 2026-03-18 10:15:00 | 1247.40 | 2026-03-19 09:15:00 | 1212.20 | STOP_HIT | 1.00 | -2.82% |
