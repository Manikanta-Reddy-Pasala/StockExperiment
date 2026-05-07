# AXISBANK (AXISBANK)

## Backtest Summary

- **Source:** Fyers history API (1H bars)
- **Window:** 2024-05-18 09:15:00 → 2026-05-07 15:15:00 (3402 bars)
- **Last close:** 1296.00
- **Strategy:** EMA 200/400 1H crossover (v2 BTC rules)
- **Target:** entry × (1 ± 30%)
- **Partial:** 50% qty booked @ 15%, trail SL → entry
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 3 attempts each at retest1 and retest2

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 3 |
| ALERT1 | 3 |
| ALERT2 | 3 |
| ALERT2_SKIP | 3 |
| ALERT3 | 5 |
| PENDING | 17 |
| PENDING_CANCEL | 4 |
| ENTRY1 | 0 |
| ENTRY2 | 13 |
| PARTIAL | 5 |
| TARGET_HIT | 0 |
| STOP_HIT | 10 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 15 (incl. partial bookings)
- **Trades open at end:** 3
- **Winners / losers:** 11 / 4
- **Target hits / Stop hits / Partials:** 0 / 10 / 5
- **Avg / median % per leg:** 8.71% / 12.96%
- **Sum % (uncompounded):** 130.64%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 15 | 11 | 73.3% | 0 | 10 | 5 | 8.71% | 130.6% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 15 | 11 | 73.3% | 0 | 10 | 5 | 8.71% | 130.6% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 15 | 11 | 73.3% | 0 | 10 | 5 | 8.71% | 130.6% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-10-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-14 15:15:00 | 1168.75 | 1199.37 | 1199.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-15 11:15:00 | 1162.20 | 1198.32 | 1198.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-18 11:15:00 | 1194.45 | 1189.54 | 1194.21 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-18 11:15:00 | 1194.45 | 1189.54 | 1194.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 11:15:00 | 1194.45 | 1189.54 | 1194.21 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2024-10-22 12:15:00 | 1181.00 | 1189.96 | 1194.10 | ENTRY2 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2024-10-22 13:15:00 | 1187.25 | 1189.93 | 1194.06 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2024-10-22 14:15:00 | 1175.45 | 1189.79 | 1193.97 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-23 09:15:00 | 1175.70 | 1189.52 | 1193.79 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 1140m) |
| Cross detected — sustain check pending | 2024-10-28 10:15:00 | 1177.50 | 1186.10 | 1191.52 | ENTRY2 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2024-10-28 11:15:00 | 1184.40 | 1186.09 | 1191.49 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2024-10-28 12:15:00 | 1179.35 | 1186.02 | 1191.43 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-28 14:15:00 | 1172.50 | 1185.76 | 1191.24 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 120m) |
| Cross detected — sustain check pending | 2024-10-30 09:15:00 | 1173.00 | 1185.03 | 1190.63 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-30 11:15:00 | 1170.45 | 1184.78 | 1190.45 | SELL ENTRY2 attempt 3/4 (retest2 break sustained 120m) |
| Cross detected — sustain check pending | 2024-12-09 09:15:00 | 1171.75 | 1158.03 | 1168.06 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-09 11:15:00 | 1172.75 | 1158.35 | 1168.12 | SELL ENTRY2 attempt 4/4 (retest2 break sustained 120m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-09 12:15:00 | 1173.65 | 1158.50 | 1168.15 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2024-12-09 13:15:00 | 1165.45 | 1158.57 | 1168.14 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-09 15:15:00 | 1167.90 | 1158.70 | 1168.11 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 120m) |
| Partial book — 50% qty @ 15%, trail SL → entry | 2025-01-17 09:15:00 | 999.35 | 1083.67 | 1113.13 | Partial book 0.50 @ 15%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty @ 15%, trail SL → entry | 2025-01-17 09:15:00 | 996.62 | 1083.67 | 1113.13 | Partial book 0.50 @ 15%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty @ 15%, trail SL → entry | 2025-01-17 09:15:00 | 994.88 | 1083.67 | 1113.13 | Partial book 0.50 @ 15%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty @ 15%, trail SL → entry | 2025-01-17 09:15:00 | 996.84 | 1083.67 | 1113.13 | Partial book 0.50 @ 15%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty @ 15%, trail SL → entry | 2025-01-17 09:15:00 | 992.72 | 1083.67 | 1113.13 | Partial book 0.50 @ 15%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-02-06 14:15:00 | 1020.55 | 1019.55 | 1058.99 | SL hit (close>ema200) qty=0.50 sl=1019.55 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-02-06 14:15:00 | 1020.55 | 1019.55 | 1058.99 | SL hit (close>ema200) qty=0.50 sl=1019.55 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-02-06 14:15:00 | 1020.55 | 1019.55 | 1058.99 | SL hit (close>ema200) qty=0.50 sl=1019.55 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-02-06 14:15:00 | 1020.55 | 1019.55 | 1058.99 | SL hit (close>ema200) qty=0.50 sl=1019.55 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-02-06 14:15:00 | 1020.55 | 1019.55 | 1058.99 | SL hit (close>ema200) qty=0.50 sl=1019.55 alert=retest2 |
| Cross detected — sustain check pending | 2025-04-25 09:15:00 | 1162.50 | 1110.88 | 1083.12 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-25 11:15:00 | 1154.50 | 1111.71 | 1083.81 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 120m) |
| Stop hit — per-position SL triggered | 2025-04-28 09:15:00 | 1182.60 | 1114.48 | 1085.91 | SL hit (close>static) qty=1.00 sl=1174.20 alert=retest2 |
| Cross detected — sustain check pending | 2025-05-05 11:15:00 | 1169.80 | 1133.54 | 1100.15 | ENTRY2 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2025-05-05 13:15:00 | 1172.70 | 1134.29 | 1100.86 | ENTRY2 sustain failed after 120m |
| Cross detected — sustain check pending | 2025-05-06 10:15:00 | 1165.20 | 1135.72 | 1102.25 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-06 12:15:00 | 1165.10 | 1136.30 | 1102.87 | SELL ENTRY2 attempt 3/4 (retest2 break sustained 120m) |
| Stop hit — per-position SL triggered | 2025-05-08 12:15:00 | 1177.30 | 1140.05 | 1107.06 | SL hit (close>static) qty=1.00 sl=1174.20 alert=retest2 |
| Cross detected — sustain check pending | 2025-05-09 09:15:00 | 1159.60 | 1141.18 | 1108.29 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-09 11:15:00 | 1160.70 | 1141.55 | 1108.80 | SELL ENTRY2 attempt 4/4 (retest2 break sustained 120m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-09 12:15:00 | 1156.50 | 1141.70 | 1109.04 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-05-09 13:15:00 | 1153.90 | 1141.82 | 1109.26 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-09 15:15:00 | 1153.00 | 1142.01 | 1109.68 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 120m) |
| Stop hit — per-position SL triggered | 2025-05-12 09:15:00 | 1194.70 | 1142.54 | 1110.11 | SL hit (close>static) qty=1.00 sl=1174.20 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-05-12 09:15:00 | 1194.70 | 1142.54 | 1110.11 | SL hit (close>static) qty=1.00 sl=1161.00 alert=retest2 |
| Cross detected — sustain check pending | 2025-06-06 09:15:00 | 1153.50 | 1178.95 | 1150.76 | ENTRY2 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2025-06-06 10:15:00 | 1192.40 | 1179.09 | 1150.97 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-07-18 09:15:00 | 1115.10 | 1183.57 | 1176.49 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-18 11:15:00 | 1113.40 | 1182.15 | 1175.85 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 120m) |
| Stop hit — per-position SL triggered | 2025-07-22 13:15:00 | 1101.00 | 1169.99 | 1170.01 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 2 — SELL (started 2025-07-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-22 13:15:00 | 1101.00 | 1169.99 | 1170.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-22 14:15:00 | 1097.50 | 1169.27 | 1169.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-11 10:15:00 | 1074.20 | 1073.36 | 1097.24 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-12 09:15:00 | 1102.30 | 1074.25 | 1096.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 09:15:00 | 1102.30 | 1074.25 | 1096.98 | EMA400 retest candle locked |

### Cycle 3 — SELL (started 2026-03-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 12:15:00 | 1221.40 | 1299.88 | 1300.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 13:15:00 | 1210.80 | 1299.00 | 1299.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-07 14:15:00 | 1250.90 | 1249.64 | 1270.19 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-08 09:15:00 | 1316.00 | 1250.28 | 1270.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 09:15:00 | 1316.00 | 1250.28 | 1270.30 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2026-04-28 11:15:00 | 1296.00 | 1311.73 | 1300.55 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-28 13:15:00 | 1291.00 | 1311.33 | 1300.45 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 120m) |
| Cross detected — sustain check pending | 2026-04-29 13:15:00 | 1296.60 | 1310.11 | 1300.21 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-29 15:15:00 | 1290.60 | 1309.78 | 1300.14 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 120m) |
| Cross detected — sustain check pending | 2026-05-07 09:15:00 | 1298.50 | 1300.02 | 1296.19 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-07 11:15:00 | 1287.60 | 1299.86 | 1296.15 | SELL ENTRY2 attempt 3/4 (retest2 break sustained 120m) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2024-10-23 09:15:00 | 1175.70 | 2025-01-17 09:15:00 | 999.35 | PARTIAL | 0.50 | 15.00% |
| SELL | retest2 | 2024-10-28 14:15:00 | 1172.50 | 2025-01-17 09:15:00 | 996.62 | PARTIAL | 0.50 | 15.00% |
| SELL | retest2 | 2024-10-30 11:15:00 | 1170.45 | 2025-01-17 09:15:00 | 994.88 | PARTIAL | 0.50 | 15.00% |
| SELL | retest2 | 2024-12-09 11:15:00 | 1172.75 | 2025-01-17 09:15:00 | 996.84 | PARTIAL | 0.50 | 15.00% |
| SELL | retest2 | 2024-12-09 15:15:00 | 1167.90 | 2025-01-17 09:15:00 | 992.72 | PARTIAL | 0.50 | 15.00% |
| SELL | retest2 | 2024-10-23 09:15:00 | 1175.70 | 2025-02-06 14:15:00 | 1020.55 | STOP_HIT | 0.50 | 13.20% |
| SELL | retest2 | 2024-10-28 14:15:00 | 1172.50 | 2025-02-06 14:15:00 | 1020.55 | STOP_HIT | 0.50 | 12.96% |
| SELL | retest2 | 2024-10-30 11:15:00 | 1170.45 | 2025-02-06 14:15:00 | 1020.55 | STOP_HIT | 0.50 | 12.81% |
| SELL | retest2 | 2024-12-09 11:15:00 | 1172.75 | 2025-02-06 14:15:00 | 1020.55 | STOP_HIT | 0.50 | 12.98% |
| SELL | retest2 | 2024-12-09 15:15:00 | 1167.90 | 2025-02-06 14:15:00 | 1020.55 | STOP_HIT | 0.50 | 12.62% |
| SELL | retest2 | 2025-04-25 11:15:00 | 1154.50 | 2025-04-28 09:15:00 | 1182.60 | STOP_HIT | 1.00 | -2.43% |
| SELL | retest2 | 2025-05-06 12:15:00 | 1165.10 | 2025-05-08 12:15:00 | 1177.30 | STOP_HIT | 1.00 | -1.05% |
| SELL | retest2 | 2025-05-09 11:15:00 | 1160.70 | 2025-05-12 09:15:00 | 1194.70 | STOP_HIT | 1.00 | -2.93% |
| SELL | retest2 | 2025-05-09 15:15:00 | 1153.00 | 2025-05-12 09:15:00 | 1194.70 | STOP_HIT | 1.00 | -3.62% |
| SELL | retest2 | 2025-07-18 11:15:00 | 1113.40 | 2025-07-22 13:15:00 | 1101.00 | STOP_HIT | 1.00 | 1.11% |
