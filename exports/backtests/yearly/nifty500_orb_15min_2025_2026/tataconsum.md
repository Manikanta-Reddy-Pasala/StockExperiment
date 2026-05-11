# Tata Consumer Products Ltd. (TATACONSUM)

## Backtest Summary

- **Window:** 2025-09-08 09:15:00 → 2026-05-08 15:25:00 (12238 bars)
- **Last close:** 1176.60
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 0 |
| ALERT1 | 0 |
| ALERT2 | 0 |
| ALERT2_SKIP | 0 |
| ALERT3 | 0 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 52 |
| ENTRY2 | 0 |
| PARTIAL | 22 |
| TARGET_HIT | 11 |
| STOP_HIT | 41 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 74 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 33 / 41
- **Target hits / Stop hits / Partials:** 11 / 41 / 22
- **Avg / median % per leg:** 0.11% / 0.00%
- **Sum % (uncompounded):** 8.25%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 41 | 20 | 48.8% | 7 | 21 | 13 | 0.17% | 6.8% |
| BUY @ 2nd Alert (retest1) | 41 | 20 | 48.8% | 7 | 21 | 13 | 0.17% | 6.8% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 33 | 13 | 39.4% | 4 | 20 | 9 | 0.04% | 1.5% |
| SELL @ 2nd Alert (retest1) | 33 | 13 | 39.4% | 4 | 20 | 9 | 0.04% | 1.5% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 74 | 33 | 44.6% | 11 | 41 | 22 | 0.11% | 8.2% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-09-10 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-10 10:30:00 | 1089.20 | 1085.83 | 0.00 | ORB-long ORB[1082.10,1088.90] vol=1.6x ATR=2.42 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-10 11:20:00 | 1092.83 | 1086.80 | 0.00 | T1 1.5R @ 1092.83 |
| Target hit | 2025-09-10 15:20:00 | 1101.70 | 1092.03 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 2 — BUY (started 2025-09-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-11 10:15:00 | 1107.00 | 1101.95 | 0.00 | ORB-long ORB[1097.20,1104.80] vol=2.1x ATR=2.56 |
| Stop hit — per-position SL triggered | 2025-09-11 10:35:00 | 1104.44 | 1102.37 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2025-09-15 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-15 11:00:00 | 1094.40 | 1097.47 | 0.00 | ORB-short ORB[1096.80,1104.40] vol=3.0x ATR=1.38 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-15 11:10:00 | 1092.33 | 1097.20 | 0.00 | T1 1.5R @ 1092.33 |
| Stop hit — per-position SL triggered | 2025-09-15 11:30:00 | 1094.40 | 1095.72 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2025-09-16 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-16 10:35:00 | 1097.00 | 1099.20 | 0.00 | ORB-short ORB[1097.60,1105.40] vol=2.6x ATR=2.11 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-16 13:30:00 | 1093.84 | 1097.43 | 0.00 | T1 1.5R @ 1093.84 |
| Target hit | 2025-09-16 15:20:00 | 1092.30 | 1094.69 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 5 — BUY (started 2025-09-22 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-22 11:05:00 | 1134.10 | 1129.28 | 0.00 | ORB-long ORB[1119.90,1127.90] vol=3.0x ATR=2.38 |
| Stop hit — per-position SL triggered | 2025-09-22 11:15:00 | 1131.72 | 1129.56 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2025-09-24 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-24 09:30:00 | 1123.40 | 1124.43 | 0.00 | ORB-short ORB[1123.80,1129.60] vol=3.1x ATR=2.14 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-24 09:40:00 | 1120.18 | 1124.05 | 0.00 | T1 1.5R @ 1120.18 |
| Stop hit — per-position SL triggered | 2025-09-24 09:55:00 | 1123.40 | 1123.89 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2025-09-25 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-25 10:40:00 | 1133.60 | 1141.13 | 0.00 | ORB-short ORB[1138.00,1146.50] vol=1.8x ATR=2.81 |
| Stop hit — per-position SL triggered | 2025-09-25 11:00:00 | 1136.41 | 1140.41 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2025-10-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-07 11:15:00 | 1130.20 | 1136.02 | 0.00 | ORB-short ORB[1135.70,1144.90] vol=2.0x ATR=2.22 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-07 11:20:00 | 1126.88 | 1135.34 | 0.00 | T1 1.5R @ 1126.88 |
| Stop hit — per-position SL triggered | 2025-10-07 11:55:00 | 1130.20 | 1133.43 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2025-10-10 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-10 11:00:00 | 1122.10 | 1119.24 | 0.00 | ORB-long ORB[1108.40,1119.00] vol=6.2x ATR=2.38 |
| Stop hit — per-position SL triggered | 2025-10-10 11:05:00 | 1119.72 | 1119.48 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2025-10-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-14 11:15:00 | 1114.10 | 1116.73 | 0.00 | ORB-short ORB[1115.00,1122.70] vol=3.0x ATR=2.52 |
| Stop hit — per-position SL triggered | 2025-10-14 13:25:00 | 1116.62 | 1115.46 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2025-10-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-17 10:15:00 | 1157.40 | 1152.31 | 0.00 | ORB-long ORB[1142.50,1152.40] vol=1.6x ATR=3.43 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-17 11:10:00 | 1162.55 | 1156.58 | 0.00 | T1 1.5R @ 1162.55 |
| Target hit | 2025-10-17 15:20:00 | 1166.50 | 1163.55 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 12 — SELL (started 2025-11-12 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-12 10:00:00 | 1155.00 | 1158.97 | 0.00 | ORB-short ORB[1156.30,1164.10] vol=2.0x ATR=2.27 |
| Stop hit — per-position SL triggered | 2025-11-12 10:25:00 | 1157.27 | 1158.22 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2025-11-14 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-14 10:50:00 | 1160.50 | 1157.91 | 0.00 | ORB-long ORB[1146.80,1158.80] vol=4.8x ATR=2.85 |
| Stop hit — per-position SL triggered | 2025-11-14 11:20:00 | 1157.65 | 1158.42 | 0.00 | SL hit |

### Cycle 14 — SELL (started 2025-11-17 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-17 10:45:00 | 1159.50 | 1166.32 | 0.00 | ORB-short ORB[1162.00,1172.00] vol=1.6x ATR=2.55 |
| Stop hit — per-position SL triggered | 2025-11-17 11:10:00 | 1162.05 | 1165.70 | 0.00 | SL hit |

### Cycle 15 — SELL (started 2025-11-18 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-18 10:35:00 | 1159.90 | 1166.79 | 0.00 | ORB-short ORB[1173.20,1178.70] vol=4.2x ATR=2.56 |
| Stop hit — per-position SL triggered | 2025-11-18 10:45:00 | 1162.46 | 1166.03 | 0.00 | SL hit |

### Cycle 16 — BUY (started 2025-11-20 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-20 10:25:00 | 1176.30 | 1173.32 | 0.00 | ORB-long ORB[1163.00,1169.70] vol=1.9x ATR=2.43 |
| Stop hit — per-position SL triggered | 2025-11-20 11:15:00 | 1173.87 | 1174.30 | 0.00 | SL hit |

### Cycle 17 — SELL (started 2025-12-08 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-08 11:05:00 | 1159.90 | 1162.38 | 0.00 | ORB-short ORB[1160.20,1171.00] vol=2.6x ATR=2.40 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-08 11:25:00 | 1156.30 | 1161.77 | 0.00 | T1 1.5R @ 1156.30 |
| Target hit | 2025-12-08 15:20:00 | 1144.10 | 1152.57 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 18 — BUY (started 2025-12-11 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-11 10:55:00 | 1147.00 | 1142.97 | 0.00 | ORB-long ORB[1136.10,1146.00] vol=2.0x ATR=2.32 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-11 11:15:00 | 1150.48 | 1144.14 | 0.00 | T1 1.5R @ 1150.48 |
| Target hit | 2025-12-11 13:55:00 | 1148.30 | 1148.39 | 0.00 | Trail-exit close<VWAP |

### Cycle 19 — SELL (started 2025-12-12 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-12 10:35:00 | 1141.80 | 1143.85 | 0.00 | ORB-short ORB[1142.10,1147.30] vol=4.2x ATR=2.98 |
| Stop hit — per-position SL triggered | 2025-12-12 12:15:00 | 1144.78 | 1143.19 | 0.00 | SL hit |

### Cycle 20 — BUY (started 2025-12-15 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-15 10:40:00 | 1160.60 | 1149.72 | 0.00 | ORB-long ORB[1140.10,1149.00] vol=2.2x ATR=2.47 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-15 10:45:00 | 1164.31 | 1150.87 | 0.00 | T1 1.5R @ 1164.31 |
| Stop hit — per-position SL triggered | 2025-12-15 11:05:00 | 1160.60 | 1155.22 | 0.00 | SL hit |

### Cycle 21 — BUY (started 2025-12-17 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-17 10:45:00 | 1180.70 | 1178.17 | 0.00 | ORB-long ORB[1170.00,1177.20] vol=7.4x ATR=2.37 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-17 11:10:00 | 1184.26 | 1179.39 | 0.00 | T1 1.5R @ 1184.26 |
| Stop hit — per-position SL triggered | 2025-12-17 12:00:00 | 1180.70 | 1180.37 | 0.00 | SL hit |

### Cycle 22 — BUY (started 2025-12-23 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-23 10:55:00 | 1183.60 | 1177.91 | 0.00 | ORB-long ORB[1174.50,1178.80] vol=1.8x ATR=2.14 |
| Stop hit — per-position SL triggered | 2025-12-23 11:40:00 | 1181.46 | 1179.89 | 0.00 | SL hit |

### Cycle 23 — SELL (started 2025-12-24 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-24 10:40:00 | 1180.60 | 1182.76 | 0.00 | ORB-short ORB[1182.00,1189.50] vol=4.6x ATR=1.98 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-24 10:55:00 | 1177.63 | 1182.28 | 0.00 | T1 1.5R @ 1177.63 |
| Target hit | 2025-12-24 14:55:00 | 1179.50 | 1178.26 | 0.00 | Trail-exit close>VWAP |

### Cycle 24 — BUY (started 2025-12-29 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-29 11:10:00 | 1178.60 | 1176.72 | 0.00 | ORB-long ORB[1170.50,1177.20] vol=2.4x ATR=2.39 |
| Stop hit — per-position SL triggered | 2025-12-29 11:15:00 | 1176.21 | 1176.80 | 0.00 | SL hit |

### Cycle 25 — SELL (started 2026-01-01 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-01 10:20:00 | 1179.50 | 1183.07 | 0.00 | ORB-short ORB[1187.50,1195.50] vol=5.7x ATR=2.28 |
| Stop hit — per-position SL triggered | 2026-01-01 10:25:00 | 1181.78 | 1183.01 | 0.00 | SL hit |

### Cycle 26 — BUY (started 2026-01-05 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-05 10:25:00 | 1180.90 | 1175.30 | 0.00 | ORB-long ORB[1165.00,1177.10] vol=1.6x ATR=2.80 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-05 11:35:00 | 1185.10 | 1177.64 | 0.00 | T1 1.5R @ 1185.10 |
| Stop hit — per-position SL triggered | 2026-01-05 14:35:00 | 1180.90 | 1181.30 | 0.00 | SL hit |

### Cycle 27 — SELL (started 2026-01-08 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-08 11:10:00 | 1198.90 | 1201.95 | 0.00 | ORB-short ORB[1201.00,1212.00] vol=1.6x ATR=2.53 |
| Stop hit — per-position SL triggered | 2026-01-08 11:15:00 | 1201.43 | 1201.79 | 0.00 | SL hit |

### Cycle 28 — BUY (started 2026-01-12 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-12 10:40:00 | 1182.50 | 1182.15 | 0.00 | ORB-long ORB[1173.50,1182.10] vol=1.7x ATR=2.96 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-12 10:45:00 | 1186.94 | 1182.65 | 0.00 | T1 1.5R @ 1186.94 |
| Stop hit — per-position SL triggered | 2026-01-12 11:05:00 | 1182.50 | 1182.94 | 0.00 | SL hit |

### Cycle 29 — BUY (started 2026-01-16 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-16 10:05:00 | 1187.50 | 1181.62 | 0.00 | ORB-long ORB[1166.30,1181.40] vol=1.6x ATR=3.05 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-16 10:10:00 | 1192.07 | 1182.31 | 0.00 | T1 1.5R @ 1192.07 |
| Target hit | 2026-01-16 11:55:00 | 1190.10 | 1190.18 | 0.00 | Trail-exit close<VWAP |

### Cycle 30 — BUY (started 2026-01-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-22 11:15:00 | 1180.40 | 1178.09 | 0.00 | ORB-long ORB[1163.70,1172.50] vol=1.7x ATR=3.14 |
| Stop hit — per-position SL triggered | 2026-01-22 11:50:00 | 1177.26 | 1178.37 | 0.00 | SL hit |

### Cycle 31 — SELL (started 2026-01-23 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-23 11:00:00 | 1173.90 | 1182.20 | 0.00 | ORB-short ORB[1174.10,1183.00] vol=3.1x ATR=3.28 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-23 12:10:00 | 1168.98 | 1179.43 | 0.00 | T1 1.5R @ 1168.98 |
| Stop hit — per-position SL triggered | 2026-01-23 12:25:00 | 1173.90 | 1178.92 | 0.00 | SL hit |

### Cycle 32 — SELL (started 2026-01-29 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-29 11:05:00 | 1101.50 | 1114.03 | 0.00 | ORB-short ORB[1115.70,1131.40] vol=2.7x ATR=3.01 |
| Stop hit — per-position SL triggered | 2026-01-29 11:15:00 | 1104.51 | 1112.61 | 0.00 | SL hit |

### Cycle 33 — BUY (started 2026-01-30 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-30 11:10:00 | 1133.00 | 1122.41 | 0.00 | ORB-long ORB[1100.10,1114.30] vol=2.1x ATR=3.13 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-30 11:15:00 | 1137.70 | 1123.43 | 0.00 | T1 1.5R @ 1137.70 |
| Stop hit — per-position SL triggered | 2026-01-30 11:20:00 | 1133.00 | 1123.76 | 0.00 | SL hit |

### Cycle 34 — BUY (started 2026-02-01 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-01 10:45:00 | 1147.60 | 1136.07 | 0.00 | ORB-long ORB[1125.90,1139.00] vol=2.1x ATR=4.00 |
| Stop hit — per-position SL triggered | 2026-02-01 11:00:00 | 1143.60 | 1137.22 | 0.00 | SL hit |

### Cycle 35 — BUY (started 2026-02-02 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-02 09:30:00 | 1104.80 | 1094.08 | 0.00 | ORB-long ORB[1084.00,1094.00] vol=2.2x ATR=8.63 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-02 11:35:00 | 1117.75 | 1104.74 | 0.00 | T1 1.5R @ 1117.75 |
| Target hit | 2026-02-02 15:20:00 | 1127.10 | 1114.60 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 36 — BUY (started 2026-02-04 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-04 11:00:00 | 1169.40 | 1162.18 | 0.00 | ORB-long ORB[1145.90,1158.00] vol=2.0x ATR=2.36 |
| Stop hit — per-position SL triggered | 2026-02-04 11:25:00 | 1167.04 | 1162.64 | 0.00 | SL hit |

### Cycle 37 — SELL (started 2026-02-05 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-05 11:10:00 | 1148.50 | 1154.03 | 0.00 | ORB-short ORB[1149.00,1163.60] vol=3.4x ATR=2.78 |
| Stop hit — per-position SL triggered | 2026-02-05 13:50:00 | 1151.28 | 1151.31 | 0.00 | SL hit |

### Cycle 38 — SELL (started 2026-02-10 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-10 10:55:00 | 1159.60 | 1162.06 | 0.00 | ORB-short ORB[1161.10,1167.80] vol=3.5x ATR=1.98 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-10 11:10:00 | 1156.63 | 1160.43 | 0.00 | T1 1.5R @ 1156.63 |
| Stop hit — per-position SL triggered | 2026-02-10 12:00:00 | 1159.60 | 1159.60 | 0.00 | SL hit |

### Cycle 39 — SELL (started 2026-02-13 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-13 10:30:00 | 1139.80 | 1144.02 | 0.00 | ORB-short ORB[1144.30,1154.00] vol=1.5x ATR=2.66 |
| Stop hit — per-position SL triggered | 2026-02-13 10:55:00 | 1142.46 | 1143.25 | 0.00 | SL hit |

### Cycle 40 — BUY (started 2026-02-18 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-18 11:00:00 | 1158.50 | 1152.53 | 0.00 | ORB-long ORB[1145.80,1154.50] vol=2.7x ATR=2.26 |
| Stop hit — per-position SL triggered | 2026-02-18 11:25:00 | 1156.24 | 1153.43 | 0.00 | SL hit |

### Cycle 41 — SELL (started 2026-02-25 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-25 11:00:00 | 1173.00 | 1176.29 | 0.00 | ORB-short ORB[1175.70,1185.00] vol=1.6x ATR=2.03 |
| Stop hit — per-position SL triggered | 2026-02-25 11:05:00 | 1175.03 | 1176.25 | 0.00 | SL hit |

### Cycle 42 — SELL (started 2026-02-27 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-27 10:20:00 | 1135.30 | 1145.62 | 0.00 | ORB-short ORB[1146.50,1159.50] vol=1.7x ATR=2.80 |
| Stop hit — per-position SL triggered | 2026-02-27 10:30:00 | 1138.10 | 1144.22 | 0.00 | SL hit |

### Cycle 43 — BUY (started 2026-03-13 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-13 10:50:00 | 1075.00 | 1065.52 | 0.00 | ORB-long ORB[1048.00,1060.80] vol=2.0x ATR=3.66 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-13 11:30:00 | 1080.49 | 1069.79 | 0.00 | T1 1.5R @ 1080.49 |
| Target hit | 2026-03-13 13:15:00 | 1078.20 | 1078.32 | 0.00 | Trail-exit close<VWAP |

### Cycle 44 — SELL (started 2026-03-16 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-16 10:50:00 | 1072.30 | 1080.61 | 0.00 | ORB-short ORB[1078.10,1093.00] vol=2.0x ATR=3.60 |
| Stop hit — per-position SL triggered | 2026-03-16 11:00:00 | 1075.90 | 1080.34 | 0.00 | SL hit |

### Cycle 45 — BUY (started 2026-03-17 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-17 10:25:00 | 1100.00 | 1098.08 | 0.00 | ORB-long ORB[1088.00,1096.20] vol=1.8x ATR=3.18 |
| Stop hit — per-position SL triggered | 2026-03-17 10:30:00 | 1096.82 | 1097.60 | 0.00 | SL hit |

### Cycle 46 — SELL (started 2026-03-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-24 11:15:00 | 1029.90 | 1030.97 | 0.00 | ORB-short ORB[1030.90,1044.60] vol=1.6x ATR=3.25 |
| Stop hit — per-position SL triggered | 2026-03-24 11:40:00 | 1033.15 | 1030.91 | 0.00 | SL hit |

### Cycle 47 — BUY (started 2026-04-02 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-02 11:05:00 | 1024.40 | 1019.24 | 0.00 | ORB-long ORB[1007.20,1022.00] vol=2.0x ATR=3.17 |
| Stop hit — per-position SL triggered | 2026-04-02 11:30:00 | 1021.23 | 1019.73 | 0.00 | SL hit |

### Cycle 48 — BUY (started 2026-04-10 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-10 11:00:00 | 1090.10 | 1087.07 | 0.00 | ORB-long ORB[1080.60,1088.50] vol=3.0x ATR=2.93 |
| Stop hit — per-position SL triggered | 2026-04-10 12:55:00 | 1087.17 | 1088.95 | 0.00 | SL hit |

### Cycle 49 — BUY (started 2026-04-20 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-20 10:00:00 | 1121.40 | 1116.05 | 0.00 | ORB-long ORB[1106.70,1119.90] vol=1.5x ATR=3.43 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-20 11:40:00 | 1126.54 | 1120.05 | 0.00 | T1 1.5R @ 1126.54 |
| Target hit | 2026-04-20 13:55:00 | 1122.10 | 1122.43 | 0.00 | Trail-exit close<VWAP |

### Cycle 50 — BUY (started 2026-04-22 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-22 10:40:00 | 1178.10 | 1161.43 | 0.00 | ORB-long ORB[1139.80,1156.00] vol=2.0x ATR=4.35 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-22 12:10:00 | 1184.62 | 1172.03 | 0.00 | T1 1.5R @ 1184.62 |
| Stop hit — per-position SL triggered | 2026-04-22 14:15:00 | 1178.10 | 1177.09 | 0.00 | SL hit |

### Cycle 51 — BUY (started 2026-04-24 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-24 09:40:00 | 1192.00 | 1190.21 | 0.00 | ORB-long ORB[1174.00,1190.00] vol=2.4x ATR=4.95 |
| Stop hit — per-position SL triggered | 2026-04-24 09:50:00 | 1187.05 | 1190.00 | 0.00 | SL hit |

### Cycle 52 — SELL (started 2026-05-06 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-06 10:05:00 | 1157.60 | 1159.96 | 0.00 | ORB-short ORB[1159.00,1166.90] vol=2.2x ATR=3.13 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-06 10:35:00 | 1152.91 | 1159.17 | 0.00 | T1 1.5R @ 1152.91 |
| Target hit | 2026-05-06 14:05:00 | 1152.20 | 1149.81 | 0.00 | Trail-exit close>VWAP |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-09-10 10:30:00 | 1089.20 | 2025-09-10 11:20:00 | 1092.83 | PARTIAL | 0.50 | 0.33% |
| BUY | retest1 | 2025-09-10 10:30:00 | 1089.20 | 2025-09-10 15:20:00 | 1101.70 | TARGET_HIT | 0.50 | 1.15% |
| BUY | retest1 | 2025-09-11 10:15:00 | 1107.00 | 2025-09-11 10:35:00 | 1104.44 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2025-09-15 11:00:00 | 1094.40 | 2025-09-15 11:10:00 | 1092.33 | PARTIAL | 0.50 | 0.19% |
| SELL | retest1 | 2025-09-15 11:00:00 | 1094.40 | 2025-09-15 11:30:00 | 1094.40 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-09-16 10:35:00 | 1097.00 | 2025-09-16 13:30:00 | 1093.84 | PARTIAL | 0.50 | 0.29% |
| SELL | retest1 | 2025-09-16 10:35:00 | 1097.00 | 2025-09-16 15:20:00 | 1092.30 | TARGET_HIT | 0.50 | 0.43% |
| BUY | retest1 | 2025-09-22 11:05:00 | 1134.10 | 2025-09-22 11:15:00 | 1131.72 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2025-09-24 09:30:00 | 1123.40 | 2025-09-24 09:40:00 | 1120.18 | PARTIAL | 0.50 | 0.29% |
| SELL | retest1 | 2025-09-24 09:30:00 | 1123.40 | 2025-09-24 09:55:00 | 1123.40 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-09-25 10:40:00 | 1133.60 | 2025-09-25 11:00:00 | 1136.41 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2025-10-07 11:15:00 | 1130.20 | 2025-10-07 11:20:00 | 1126.88 | PARTIAL | 0.50 | 0.29% |
| SELL | retest1 | 2025-10-07 11:15:00 | 1130.20 | 2025-10-07 11:55:00 | 1130.20 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-10-10 11:00:00 | 1122.10 | 2025-10-10 11:05:00 | 1119.72 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2025-10-14 11:15:00 | 1114.10 | 2025-10-14 13:25:00 | 1116.62 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2025-10-17 10:15:00 | 1157.40 | 2025-10-17 11:10:00 | 1162.55 | PARTIAL | 0.50 | 0.44% |
| BUY | retest1 | 2025-10-17 10:15:00 | 1157.40 | 2025-10-17 15:20:00 | 1166.50 | TARGET_HIT | 0.50 | 0.79% |
| SELL | retest1 | 2025-11-12 10:00:00 | 1155.00 | 2025-11-12 10:25:00 | 1157.27 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2025-11-14 10:50:00 | 1160.50 | 2025-11-14 11:20:00 | 1157.65 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2025-11-17 10:45:00 | 1159.50 | 2025-11-17 11:10:00 | 1162.05 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2025-11-18 10:35:00 | 1159.90 | 2025-11-18 10:45:00 | 1162.46 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2025-11-20 10:25:00 | 1176.30 | 2025-11-20 11:15:00 | 1173.87 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2025-12-08 11:05:00 | 1159.90 | 2025-12-08 11:25:00 | 1156.30 | PARTIAL | 0.50 | 0.31% |
| SELL | retest1 | 2025-12-08 11:05:00 | 1159.90 | 2025-12-08 15:20:00 | 1144.10 | TARGET_HIT | 0.50 | 1.36% |
| BUY | retest1 | 2025-12-11 10:55:00 | 1147.00 | 2025-12-11 11:15:00 | 1150.48 | PARTIAL | 0.50 | 0.30% |
| BUY | retest1 | 2025-12-11 10:55:00 | 1147.00 | 2025-12-11 13:55:00 | 1148.30 | TARGET_HIT | 0.50 | 0.11% |
| SELL | retest1 | 2025-12-12 10:35:00 | 1141.80 | 2025-12-12 12:15:00 | 1144.78 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2025-12-15 10:40:00 | 1160.60 | 2025-12-15 10:45:00 | 1164.31 | PARTIAL | 0.50 | 0.32% |
| BUY | retest1 | 2025-12-15 10:40:00 | 1160.60 | 2025-12-15 11:05:00 | 1160.60 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-12-17 10:45:00 | 1180.70 | 2025-12-17 11:10:00 | 1184.26 | PARTIAL | 0.50 | 0.30% |
| BUY | retest1 | 2025-12-17 10:45:00 | 1180.70 | 2025-12-17 12:00:00 | 1180.70 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-12-23 10:55:00 | 1183.60 | 2025-12-23 11:40:00 | 1181.46 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest1 | 2025-12-24 10:40:00 | 1180.60 | 2025-12-24 10:55:00 | 1177.63 | PARTIAL | 0.50 | 0.25% |
| SELL | retest1 | 2025-12-24 10:40:00 | 1180.60 | 2025-12-24 14:55:00 | 1179.50 | TARGET_HIT | 0.50 | 0.09% |
| BUY | retest1 | 2025-12-29 11:10:00 | 1178.60 | 2025-12-29 11:15:00 | 1176.21 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2026-01-01 10:20:00 | 1179.50 | 2026-01-01 10:25:00 | 1181.78 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest1 | 2026-01-05 10:25:00 | 1180.90 | 2026-01-05 11:35:00 | 1185.10 | PARTIAL | 0.50 | 0.36% |
| BUY | retest1 | 2026-01-05 10:25:00 | 1180.90 | 2026-01-05 14:35:00 | 1180.90 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-01-08 11:10:00 | 1198.90 | 2026-01-08 11:15:00 | 1201.43 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2026-01-12 10:40:00 | 1182.50 | 2026-01-12 10:45:00 | 1186.94 | PARTIAL | 0.50 | 0.38% |
| BUY | retest1 | 2026-01-12 10:40:00 | 1182.50 | 2026-01-12 11:05:00 | 1182.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-01-16 10:05:00 | 1187.50 | 2026-01-16 10:10:00 | 1192.07 | PARTIAL | 0.50 | 0.38% |
| BUY | retest1 | 2026-01-16 10:05:00 | 1187.50 | 2026-01-16 11:55:00 | 1190.10 | TARGET_HIT | 0.50 | 0.22% |
| BUY | retest1 | 2026-01-22 11:15:00 | 1180.40 | 2026-01-22 11:50:00 | 1177.26 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2026-01-23 11:00:00 | 1173.90 | 2026-01-23 12:10:00 | 1168.98 | PARTIAL | 0.50 | 0.42% |
| SELL | retest1 | 2026-01-23 11:00:00 | 1173.90 | 2026-01-23 12:25:00 | 1173.90 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-01-29 11:05:00 | 1101.50 | 2026-01-29 11:15:00 | 1104.51 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2026-01-30 11:10:00 | 1133.00 | 2026-01-30 11:15:00 | 1137.70 | PARTIAL | 0.50 | 0.41% |
| BUY | retest1 | 2026-01-30 11:10:00 | 1133.00 | 2026-01-30 11:20:00 | 1133.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-01 10:45:00 | 1147.60 | 2026-02-01 11:00:00 | 1143.60 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2026-02-02 09:30:00 | 1104.80 | 2026-02-02 11:35:00 | 1117.75 | PARTIAL | 0.50 | 1.17% |
| BUY | retest1 | 2026-02-02 09:30:00 | 1104.80 | 2026-02-02 15:20:00 | 1127.10 | TARGET_HIT | 0.50 | 2.02% |
| BUY | retest1 | 2026-02-04 11:00:00 | 1169.40 | 2026-02-04 11:25:00 | 1167.04 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2026-02-05 11:10:00 | 1148.50 | 2026-02-05 13:50:00 | 1151.28 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2026-02-10 10:55:00 | 1159.60 | 2026-02-10 11:10:00 | 1156.63 | PARTIAL | 0.50 | 0.26% |
| SELL | retest1 | 2026-02-10 10:55:00 | 1159.60 | 2026-02-10 12:00:00 | 1159.60 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-02-13 10:30:00 | 1139.80 | 2026-02-13 10:55:00 | 1142.46 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2026-02-18 11:00:00 | 1158.50 | 2026-02-18 11:25:00 | 1156.24 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2026-02-25 11:00:00 | 1173.00 | 2026-02-25 11:05:00 | 1175.03 | STOP_HIT | 1.00 | -0.17% |
| SELL | retest1 | 2026-02-27 10:20:00 | 1135.30 | 2026-02-27 10:30:00 | 1138.10 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2026-03-13 10:50:00 | 1075.00 | 2026-03-13 11:30:00 | 1080.49 | PARTIAL | 0.50 | 0.51% |
| BUY | retest1 | 2026-03-13 10:50:00 | 1075.00 | 2026-03-13 13:15:00 | 1078.20 | TARGET_HIT | 0.50 | 0.30% |
| SELL | retest1 | 2026-03-16 10:50:00 | 1072.30 | 2026-03-16 11:00:00 | 1075.90 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2026-03-17 10:25:00 | 1100.00 | 2026-03-17 10:30:00 | 1096.82 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2026-03-24 11:15:00 | 1029.90 | 2026-03-24 11:40:00 | 1033.15 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2026-04-02 11:05:00 | 1024.40 | 2026-04-02 11:30:00 | 1021.23 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2026-04-10 11:00:00 | 1090.10 | 2026-04-10 12:55:00 | 1087.17 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2026-04-20 10:00:00 | 1121.40 | 2026-04-20 11:40:00 | 1126.54 | PARTIAL | 0.50 | 0.46% |
| BUY | retest1 | 2026-04-20 10:00:00 | 1121.40 | 2026-04-20 13:55:00 | 1122.10 | TARGET_HIT | 0.50 | 0.06% |
| BUY | retest1 | 2026-04-22 10:40:00 | 1178.10 | 2026-04-22 12:10:00 | 1184.62 | PARTIAL | 0.50 | 0.55% |
| BUY | retest1 | 2026-04-22 10:40:00 | 1178.10 | 2026-04-22 14:15:00 | 1178.10 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-24 09:40:00 | 1192.00 | 2026-04-24 09:50:00 | 1187.05 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest1 | 2026-05-06 10:05:00 | 1157.60 | 2026-05-06 10:35:00 | 1152.91 | PARTIAL | 0.50 | 0.41% |
| SELL | retest1 | 2026-05-06 10:05:00 | 1157.60 | 2026-05-06 14:05:00 | 1152.20 | TARGET_HIT | 0.50 | 0.47% |
