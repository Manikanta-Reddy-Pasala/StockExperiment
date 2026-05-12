# Carborundum Universal Ltd. (CARBORUNIV)

## Backtest Summary

- **Window:** 2022-04-08 09:15:00 → 2026-05-08 15:15:00 (7047 bars)
- **Last close:** 1020.20
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 14 |
| ALERT1 | 10 |
| ALERT2 | 9 |
| ALERT2_SKIP | 6 |
| ALERT3 | 30 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 23 |
| PARTIAL | 9 |
| TARGET_HIT | 7 |
| STOP_HIT | 16 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 32 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 18 / 14
- **Target hits / Stop hits / Partials:** 7 / 16 / 9
- **Avg / median % per leg:** 2.98% / 4.84%
- **Sum % (uncompounded):** 95.38%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 9 | 0 | 0.0% | 0 | 9 | 0 | -1.68% | -15.2% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 9 | 0 | 0.0% | 0 | 9 | 0 | -1.68% | -15.2% |
| SELL (all) | 23 | 18 | 78.3% | 7 | 7 | 9 | 4.81% | 110.5% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 23 | 18 | 78.3% | 7 | 7 | 9 | 4.81% | 110.5% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 32 | 18 | 56.2% | 7 | 16 | 9 | 2.98% | 95.4% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-08-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-18 13:15:00 | 1090.30 | 1162.25 | 1162.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-18 15:15:00 | 1088.05 | 1160.80 | 1161.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-04 10:15:00 | 1149.00 | 1146.78 | 1153.15 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-09-04 10:30:00 | 1148.15 | 1146.78 | 1153.15 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-04 12:15:00 | 1160.90 | 1146.88 | 1153.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-04 12:45:00 | 1160.80 | 1146.88 | 1153.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-04 13:15:00 | 1182.00 | 1147.22 | 1153.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-04 13:30:00 | 1180.00 | 1147.22 | 1153.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 2 — BUY (started 2023-09-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-11 10:15:00 | 1205.00 | 1158.59 | 1158.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-11 11:15:00 | 1210.00 | 1159.10 | 1158.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-20 10:15:00 | 1160.50 | 1169.49 | 1164.58 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-20 10:15:00 | 1160.50 | 1169.49 | 1164.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-20 10:15:00 | 1160.50 | 1169.49 | 1164.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-20 11:00:00 | 1160.50 | 1169.49 | 1164.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-20 11:15:00 | 1150.30 | 1169.30 | 1164.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-20 12:00:00 | 1150.30 | 1169.30 | 1164.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-20 13:15:00 | 1158.75 | 1169.14 | 1164.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-20 14:00:00 | 1158.75 | 1169.14 | 1164.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-20 14:15:00 | 1160.85 | 1169.05 | 1164.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-20 15:15:00 | 1155.20 | 1169.05 | 1164.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-20 15:15:00 | 1155.20 | 1168.92 | 1164.41 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-21 10:45:00 | 1168.55 | 1168.79 | 1164.40 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-22 09:15:00 | 1173.50 | 1168.61 | 1164.41 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-10-09 09:15:00 | 1144.00 | 1174.92 | 1169.57 | SL hit (close<static) qty=1.00 sl=1153.75 alert=retest2 |

### Cycle 3 — SELL (started 2023-10-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-23 09:15:00 | 1099.50 | 1165.54 | 1165.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-25 11:15:00 | 1093.70 | 1160.15 | 1162.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-17 09:15:00 | 1127.35 | 1106.70 | 1127.82 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-17 09:15:00 | 1127.35 | 1106.70 | 1127.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-17 09:15:00 | 1127.35 | 1106.70 | 1127.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-17 10:00:00 | 1127.35 | 1106.70 | 1127.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-17 10:15:00 | 1123.00 | 1106.86 | 1127.79 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-17 11:30:00 | 1118.30 | 1106.92 | 1127.72 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-11-22 12:15:00 | 1163.30 | 1106.09 | 1125.11 | SL hit (close>static) qty=1.00 sl=1133.50 alert=retest2 |

### Cycle 4 — BUY (started 2023-12-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-07 12:15:00 | 1202.00 | 1137.39 | 1137.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-07 13:15:00 | 1211.55 | 1138.12 | 1137.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-20 10:15:00 | 1156.95 | 1163.58 | 1152.57 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-12-20 11:00:00 | 1156.95 | 1163.58 | 1152.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-20 11:15:00 | 1148.00 | 1163.43 | 1152.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-20 12:00:00 | 1148.00 | 1163.43 | 1152.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-20 12:15:00 | 1139.50 | 1163.19 | 1152.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-20 13:00:00 | 1139.50 | 1163.19 | 1152.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 5 — SELL (started 2023-12-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-29 15:15:00 | 1115.50 | 1143.79 | 1143.86 | EMA200 below EMA400 |

### Cycle 6 — BUY (started 2024-01-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-15 10:15:00 | 1173.50 | 1142.66 | 1142.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-15 11:15:00 | 1189.40 | 1143.12 | 1142.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-17 11:15:00 | 1141.10 | 1146.38 | 1144.60 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-17 11:15:00 | 1141.10 | 1146.38 | 1144.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-17 11:15:00 | 1141.10 | 1146.38 | 1144.60 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-20 10:15:00 | 1148.95 | 1144.84 | 1143.96 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-01-20 11:15:00 | 1134.40 | 1144.69 | 1143.89 | SL hit (close<static) qty=1.00 sl=1138.55 alert=retest2 |

### Cycle 7 — SELL (started 2024-01-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-23 13:15:00 | 1120.50 | 1142.99 | 1143.06 | EMA200 below EMA400 |

### Cycle 8 — BUY (started 2024-01-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-24 09:15:00 | 1195.00 | 1143.14 | 1143.13 | EMA200 above EMA400 |

### Cycle 9 — SELL (started 2024-02-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-01 11:15:00 | 1131.20 | 1143.37 | 1143.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-02 10:15:00 | 1126.55 | 1142.70 | 1143.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-07 09:15:00 | 1169.55 | 1137.31 | 1140.18 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-07 09:15:00 | 1169.55 | 1137.31 | 1140.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-07 09:15:00 | 1169.55 | 1137.31 | 1140.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-07 10:00:00 | 1169.55 | 1137.31 | 1140.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-07 10:15:00 | 1159.50 | 1137.53 | 1140.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-07 10:45:00 | 1164.95 | 1137.53 | 1140.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-07 14:15:00 | 1139.15 | 1137.88 | 1140.40 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-07 15:15:00 | 1129.90 | 1137.88 | 1140.40 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-02-08 12:15:00 | 1146.50 | 1137.85 | 1140.32 | SL hit (close>static) qty=1.00 sl=1146.00 alert=retest2 |

### Cycle 10 — BUY (started 2024-03-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-27 11:15:00 | 1249.15 | 1119.16 | 1118.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-28 09:15:00 | 1273.05 | 1125.66 | 1122.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-04 14:15:00 | 1516.95 | 1539.96 | 1429.56 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-04 15:00:00 | 1516.95 | 1539.96 | 1429.56 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 12:15:00 | 1665.65 | 1687.90 | 1614.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-23 12:30:00 | 1642.95 | 1687.90 | 1614.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-05 09:15:00 | 1631.90 | 1700.06 | 1641.05 | EMA400 retest candle locked (from upside) |

### Cycle 11 — SELL (started 2024-08-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-28 09:15:00 | 1538.85 | 1608.17 | 1608.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-28 10:15:00 | 1529.90 | 1607.40 | 1608.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-15 09:15:00 | 1522.50 | 1493.37 | 1526.03 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-15 09:15:00 | 1522.50 | 1493.37 | 1526.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-15 09:15:00 | 1522.50 | 1493.37 | 1526.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-15 10:00:00 | 1522.50 | 1493.37 | 1526.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-15 10:15:00 | 1533.00 | 1493.77 | 1526.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-15 11:00:00 | 1533.00 | 1493.77 | 1526.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-15 11:15:00 | 1531.85 | 1494.15 | 1526.09 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-15 12:30:00 | 1526.05 | 1494.40 | 1526.06 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-21 09:15:00 | 1449.75 | 1491.37 | 1520.78 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2024-10-25 15:15:00 | 1373.44 | 1468.70 | 1504.03 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 12 — BUY (started 2025-09-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-17 12:15:00 | 1002.10 | 949.82 | 949.65 | EMA200 above EMA400 |

### Cycle 13 — SELL (started 2025-09-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-30 10:15:00 | 925.00 | 950.64 | 950.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-30 11:15:00 | 916.25 | 950.30 | 950.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-10 14:15:00 | 939.30 | 938.10 | 943.60 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-10 14:15:00 | 939.30 | 938.10 | 943.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 14:15:00 | 939.30 | 938.10 | 943.60 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-13 09:45:00 | 924.20 | 937.93 | 943.46 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-13 13:00:00 | 925.75 | 937.60 | 943.21 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-13 15:00:00 | 925.15 | 937.37 | 943.04 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-14 10:15:00 | 925.45 | 937.15 | 942.87 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 14:15:00 | 932.85 | 923.02 | 932.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-29 14:45:00 | 931.95 | 923.02 | 932.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 15:15:00 | 931.00 | 923.10 | 932.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-30 09:15:00 | 927.00 | 923.10 | 932.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 09:15:00 | 920.10 | 923.07 | 932.85 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-31 09:45:00 | 911.00 | 923.06 | 932.51 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-04 11:30:00 | 912.55 | 920.77 | 930.59 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-06 11:15:00 | 879.46 | 918.79 | 929.25 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-06 11:15:00 | 879.18 | 918.79 | 929.25 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-07 09:15:00 | 877.99 | 917.21 | 928.19 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-07 09:15:00 | 878.89 | 917.21 | 928.19 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-18 09:15:00 | 865.45 | 907.69 | 920.63 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-18 09:15:00 | 866.92 | 907.69 | 920.63 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-11-19 13:15:00 | 833.18 | 901.82 | 916.93 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 14 — BUY (started 2026-04-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-15 11:15:00 | 892.50 | 818.14 | 818.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-17 09:15:00 | 907.70 | 826.18 | 822.24 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2023-09-21 10:45:00 | 1168.55 | 2023-10-09 09:15:00 | 1144.00 | STOP_HIT | 1.00 | -2.10% |
| BUY | retest2 | 2023-09-22 09:15:00 | 1173.50 | 2023-10-09 09:15:00 | 1144.00 | STOP_HIT | 1.00 | -2.51% |
| BUY | retest2 | 2023-10-11 10:45:00 | 1165.35 | 2023-10-12 14:15:00 | 1155.10 | STOP_HIT | 1.00 | -0.88% |
| BUY | retest2 | 2023-10-11 11:15:00 | 1170.00 | 2023-10-13 15:15:00 | 1163.00 | STOP_HIT | 1.00 | -0.60% |
| BUY | retest2 | 2023-10-12 09:30:00 | 1176.00 | 2023-10-16 11:15:00 | 1159.90 | STOP_HIT | 1.00 | -1.37% |
| BUY | retest2 | 2023-10-13 09:45:00 | 1175.45 | 2023-10-18 13:15:00 | 1160.00 | STOP_HIT | 1.00 | -1.31% |
| BUY | retest2 | 2023-10-16 09:30:00 | 1174.40 | 2023-10-18 14:15:00 | 1144.00 | STOP_HIT | 1.00 | -2.59% |
| BUY | retest2 | 2023-10-17 14:45:00 | 1173.70 | 2023-10-18 14:15:00 | 1144.00 | STOP_HIT | 1.00 | -2.53% |
| SELL | retest2 | 2023-11-17 11:30:00 | 1118.30 | 2023-11-22 12:15:00 | 1163.30 | STOP_HIT | 1.00 | -4.02% |
| BUY | retest2 | 2024-01-20 10:15:00 | 1148.95 | 2024-01-20 11:15:00 | 1134.40 | STOP_HIT | 1.00 | -1.27% |
| SELL | retest2 | 2024-02-07 15:15:00 | 1129.90 | 2024-02-08 12:15:00 | 1146.50 | STOP_HIT | 1.00 | -1.47% |
| SELL | retest2 | 2024-02-09 09:15:00 | 1136.85 | 2024-02-12 10:15:00 | 1151.20 | STOP_HIT | 1.00 | -1.26% |
| SELL | retest2 | 2024-02-09 14:45:00 | 1135.20 | 2024-02-12 10:15:00 | 1151.20 | STOP_HIT | 1.00 | -1.41% |
| SELL | retest2 | 2024-02-13 09:45:00 | 1137.85 | 2024-02-21 14:15:00 | 1080.96 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-02-13 09:45:00 | 1137.85 | 2024-03-15 12:15:00 | 1095.70 | STOP_HIT | 0.50 | 3.70% |
| SELL | retest2 | 2024-03-18 12:45:00 | 1089.95 | 2024-03-20 10:15:00 | 1115.20 | STOP_HIT | 1.00 | -2.32% |
| SELL | retest2 | 2024-10-15 12:30:00 | 1526.05 | 2024-10-21 09:15:00 | 1449.75 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-15 12:30:00 | 1526.05 | 2024-10-25 15:15:00 | 1373.44 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-10-13 09:45:00 | 924.20 | 2025-11-06 11:15:00 | 879.46 | PARTIAL | 0.50 | 4.84% |
| SELL | retest2 | 2025-10-13 13:00:00 | 925.75 | 2025-11-06 11:15:00 | 879.18 | PARTIAL | 0.50 | 5.03% |
| SELL | retest2 | 2025-10-13 15:00:00 | 925.15 | 2025-11-07 09:15:00 | 877.99 | PARTIAL | 0.50 | 5.10% |
| SELL | retest2 | 2025-10-14 10:15:00 | 925.45 | 2025-11-07 09:15:00 | 878.89 | PARTIAL | 0.50 | 5.03% |
| SELL | retest2 | 2025-10-31 09:45:00 | 911.00 | 2025-11-18 09:15:00 | 865.45 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-04 11:30:00 | 912.55 | 2025-11-18 09:15:00 | 866.92 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-13 09:45:00 | 924.20 | 2025-11-19 13:15:00 | 833.18 | TARGET_HIT | 0.50 | 9.85% |
| SELL | retest2 | 2025-10-13 13:00:00 | 925.75 | 2025-11-20 10:15:00 | 831.78 | TARGET_HIT | 0.50 | 10.15% |
| SELL | retest2 | 2025-10-13 15:00:00 | 925.15 | 2025-11-20 10:15:00 | 832.63 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-10-14 10:15:00 | 925.45 | 2025-11-20 10:15:00 | 832.91 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-10-31 09:45:00 | 911.00 | 2025-11-21 14:15:00 | 819.90 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-11-04 11:30:00 | 912.55 | 2025-11-21 14:15:00 | 821.29 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-11-26 12:00:00 | 909.10 | 2025-11-27 10:15:00 | 863.64 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-26 12:00:00 | 909.10 | 2025-12-01 15:15:00 | 888.00 | STOP_HIT | 0.50 | 2.32% |
