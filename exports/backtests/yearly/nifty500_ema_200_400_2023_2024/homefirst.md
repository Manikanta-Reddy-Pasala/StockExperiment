# Home First Finance Company India Ltd. (HOMEFIRST)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 1200.60
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 10 |
| ALERT1 | 10 |
| ALERT2 | 9 |
| ALERT2_SKIP | 7 |
| ALERT3 | 37 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 4 |
| ENTRY2 | 34 |
| PARTIAL | 10 |
| TARGET_HIT | 10 |
| STOP_HIT | 28 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 48 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 21 / 27
- **Target hits / Stop hits / Partials:** 10 / 28 / 10
- **Avg / median % per leg:** 0.77% / -0.55%
- **Sum % (uncompounded):** 37.16%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 19 | 5 | 26.3% | 3 | 15 | 1 | -1.09% | -20.7% |
| BUY @ 2nd Alert (retest1) | 5 | 1 | 20.0% | 0 | 4 | 1 | -3.04% | -15.2% |
| BUY @ 3rd Alert (retest2) | 14 | 4 | 28.6% | 3 | 11 | 0 | -0.39% | -5.5% |
| SELL (all) | 29 | 16 | 55.2% | 7 | 13 | 9 | 2.00% | 57.9% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 29 | 16 | 55.2% | 7 | 13 | 9 | 2.00% | 57.9% |
| retest1 (combined) | 5 | 1 | 20.0% | 0 | 4 | 1 | -3.04% | -15.2% |
| retest2 (combined) | 43 | 20 | 46.5% | 10 | 24 | 9 | 1.22% | 52.3% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-02-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-27 14:15:00 | 902.15 | 951.70 | 951.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-28 10:15:00 | 892.55 | 950.08 | 950.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-26 09:15:00 | 896.70 | 885.33 | 909.03 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-27 13:15:00 | 906.50 | 886.80 | 908.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-27 13:15:00 | 906.50 | 886.80 | 908.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-27 13:45:00 | 905.55 | 886.80 | 908.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-27 14:15:00 | 919.00 | 887.12 | 908.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-27 15:00:00 | 919.00 | 887.12 | 908.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-27 15:15:00 | 916.00 | 887.41 | 908.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-28 09:15:00 | 916.45 | 887.41 | 908.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-01 09:15:00 | 916.95 | 888.14 | 908.15 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-10 14:45:00 | 903.05 | 910.37 | 916.01 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-15 09:15:00 | 857.90 | 909.21 | 915.17 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-04-30 11:15:00 | 912.40 | 893.06 | 903.92 | SL hit (close>ema200) qty=0.50 sl=893.06 alert=retest2 |

### Cycle 2 — BUY (started 2024-06-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-14 14:15:00 | 1071.00 | 878.14 | 877.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-14 15:15:00 | 1080.25 | 880.16 | 878.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-25 09:15:00 | 1028.85 | 1040.84 | 992.83 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-25 11:00:00 | 1046.00 | 1040.89 | 993.10 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-26 09:15:00 | 1127.85 | 1040.62 | 994.14 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-26 09:15:00 | 1098.30 | 1041.21 | 994.67 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-26 15:00:00 | 1060.65 | 1042.86 | 996.66 | BUY ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-31 10:15:00 | 1040.20 | 1045.95 | 1002.04 | SL hit (close<ema200) qty=0.50 sl=1045.95 alert=retest1 |

### Cycle 3 — BUY (started 2024-07-31 11:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-31 11:45:00 | 1040.55 | 1045.92 | 1002.25 | BUY ENTRY1 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-02 09:15:00 | 1028.90 | 1044.73 | 1004.19 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-08-05 11:15:00 | 1004.70 | 1042.67 | 1004.92 | SL hit (close<ema400) qty=1.00 sl=1004.92 alert=retest1 |

### Cycle 4 — SELL (started 2024-11-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-14 13:15:00 | 1048.40 | 1139.51 | 1139.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-14 15:15:00 | 1041.80 | 1137.67 | 1138.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-25 09:15:00 | 1151.70 | 1120.96 | 1129.42 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-25 09:15:00 | 1151.70 | 1120.96 | 1129.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-25 09:15:00 | 1151.70 | 1120.96 | 1129.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-25 10:00:00 | 1151.70 | 1120.96 | 1129.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-25 10:15:00 | 1123.30 | 1120.98 | 1129.39 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-25 11:30:00 | 1117.35 | 1120.94 | 1129.32 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-26 09:15:00 | 1113.20 | 1121.49 | 1129.44 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-29 13:15:00 | 1061.48 | 1115.19 | 1125.17 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-29 13:15:00 | 1057.54 | 1115.19 | 1125.17 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2024-12-02 09:15:00 | 1005.61 | 1112.88 | 1123.86 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 5 — BUY (started 2025-04-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-15 09:15:00 | 1127.50 | 1007.78 | 1007.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-15 10:15:00 | 1138.50 | 1009.08 | 1007.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-22 10:15:00 | 1155.20 | 1157.11 | 1115.15 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-22 10:30:00 | 1156.30 | 1157.11 | 1115.15 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 09:15:00 | 1303.70 | 1363.51 | 1305.54 | EMA400 retest candle locked (from upside) |

### Cycle 6 — SELL (started 2025-09-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-02 12:15:00 | 1255.20 | 1277.76 | 1277.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-02 13:15:00 | 1247.60 | 1277.46 | 1277.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-08 09:15:00 | 1273.80 | 1272.75 | 1275.09 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-08 09:15:00 | 1273.80 | 1272.75 | 1275.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 09:15:00 | 1273.80 | 1272.75 | 1275.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-08 09:45:00 | 1272.30 | 1272.75 | 1275.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 10:15:00 | 1273.40 | 1272.75 | 1275.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-08 10:30:00 | 1274.70 | 1272.75 | 1275.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 11:15:00 | 1274.20 | 1272.77 | 1275.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-08 11:30:00 | 1276.10 | 1272.77 | 1275.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 12:15:00 | 1280.10 | 1272.84 | 1275.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-08 12:30:00 | 1280.80 | 1272.84 | 1275.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 13:15:00 | 1262.80 | 1272.74 | 1275.04 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-08 14:45:00 | 1261.90 | 1272.67 | 1274.99 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-09 09:45:00 | 1261.80 | 1272.50 | 1274.88 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-09 11:15:00 | 1255.60 | 1272.41 | 1274.83 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-10 15:15:00 | 1283.00 | 1272.08 | 1274.50 | SL hit (close>static) qty=1.00 sl=1281.60 alert=retest2 |

### Cycle 7 — BUY (started 2025-09-15 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-15 15:15:00 | 1295.10 | 1276.74 | 1276.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-16 09:15:00 | 1307.60 | 1277.05 | 1276.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-17 10:15:00 | 1274.30 | 1278.27 | 1277.50 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-17 10:15:00 | 1274.30 | 1278.27 | 1277.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 10:15:00 | 1274.30 | 1278.27 | 1277.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-17 11:00:00 | 1274.30 | 1278.27 | 1277.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 11:15:00 | 1264.80 | 1278.13 | 1277.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-17 11:45:00 | 1264.90 | 1278.13 | 1277.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 12:15:00 | 1261.50 | 1277.97 | 1277.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-17 13:00:00 | 1261.50 | 1277.97 | 1277.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 15:15:00 | 1277.00 | 1277.83 | 1277.30 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-18 09:15:00 | 1305.40 | 1277.83 | 1277.30 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-18 14:45:00 | 1284.00 | 1278.31 | 1277.56 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-18 15:15:00 | 1283.00 | 1278.31 | 1277.56 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-19 09:15:00 | 1269.00 | 1278.26 | 1277.54 | SL hit (close<static) qty=1.00 sl=1273.90 alert=retest2 |

### Cycle 8 — SELL (started 2025-09-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-22 13:15:00 | 1264.70 | 1276.80 | 1276.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-22 14:15:00 | 1264.60 | 1276.68 | 1276.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-24 11:15:00 | 1277.50 | 1275.60 | 1276.21 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-24 11:15:00 | 1277.50 | 1275.60 | 1276.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 11:15:00 | 1277.50 | 1275.60 | 1276.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-24 11:45:00 | 1279.20 | 1275.60 | 1276.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 12:15:00 | 1281.90 | 1275.66 | 1276.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-24 13:00:00 | 1281.90 | 1275.66 | 1276.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 13:15:00 | 1283.10 | 1275.74 | 1276.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-24 13:45:00 | 1279.50 | 1275.74 | 1276.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 14:15:00 | 1277.50 | 1275.75 | 1276.27 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-25 09:15:00 | 1259.40 | 1275.82 | 1276.30 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-26 09:15:00 | 1196.43 | 1272.96 | 1274.83 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-15 10:15:00 | 1260.00 | 1241.84 | 1255.20 | SL hit (close>ema200) qty=0.50 sl=1241.84 alert=retest2 |

### Cycle 9 — BUY (started 2026-02-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-12 15:15:00 | 1187.60 | 1144.97 | 1144.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-13 11:15:00 | 1190.00 | 1146.06 | 1145.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-19 15:15:00 | 1153.90 | 1154.32 | 1150.05 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-19 15:15:00 | 1153.90 | 1154.32 | 1150.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 15:15:00 | 1153.90 | 1154.32 | 1150.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-20 09:15:00 | 1140.00 | 1154.32 | 1150.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 09:15:00 | 1151.40 | 1154.29 | 1150.06 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-20 10:30:00 | 1156.80 | 1154.24 | 1150.05 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-20 12:30:00 | 1157.30 | 1154.26 | 1150.10 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-24 09:30:00 | 1159.00 | 1155.90 | 1151.17 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-27 13:15:00 | 1088.20 | 1163.30 | 1155.76 | SL hit (close<static) qty=1.00 sl=1135.30 alert=retest2 |

### Cycle 10 — SELL (started 2026-03-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-05 12:15:00 | 1043.00 | 1148.52 | 1148.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-05 13:15:00 | 1038.50 | 1147.43 | 1148.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 09:15:00 | 1060.60 | 1011.42 | 1059.22 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-08 09:15:00 | 1060.60 | 1011.42 | 1059.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 09:15:00 | 1060.60 | 1011.42 | 1059.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-08 09:45:00 | 1063.20 | 1011.42 | 1059.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 10:15:00 | 1060.00 | 1011.90 | 1059.23 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-09 09:30:00 | 1050.40 | 1014.66 | 1059.23 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-09 11:00:00 | 1054.55 | 1015.06 | 1059.21 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-10 09:15:00 | 1102.00 | 1018.13 | 1059.46 | SL hit (close>static) qty=1.00 sl=1065.90 alert=retest2 |

### Cycle 11 — BUY (started 2026-04-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-29 09:15:00 | 1129.65 | 1084.81 | 1084.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-29 12:15:00 | 1151.60 | 1086.53 | 1085.59 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2024-04-10 14:45:00 | 903.05 | 2024-04-15 09:15:00 | 857.90 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-04-10 14:45:00 | 903.05 | 2024-04-30 11:15:00 | 912.40 | STOP_HIT | 0.50 | -1.04% |
| SELL | retest2 | 2024-04-30 12:15:00 | 901.30 | 2024-05-06 10:15:00 | 908.20 | STOP_HIT | 1.00 | -0.77% |
| SELL | retest2 | 2024-04-30 14:45:00 | 906.45 | 2024-05-06 10:15:00 | 908.20 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest2 | 2024-05-02 09:15:00 | 899.35 | 2024-05-09 12:15:00 | 861.13 | PARTIAL | 0.50 | 4.25% |
| SELL | retest2 | 2024-05-03 11:30:00 | 895.25 | 2024-05-09 13:15:00 | 856.23 | PARTIAL | 0.50 | 4.36% |
| SELL | retest2 | 2024-05-03 15:15:00 | 889.00 | 2024-05-09 13:15:00 | 854.38 | PARTIAL | 0.50 | 3.89% |
| SELL | retest2 | 2024-05-06 14:00:00 | 898.55 | 2024-05-09 13:15:00 | 853.62 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-05-07 09:15:00 | 893.10 | 2024-05-09 14:15:00 | 848.44 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-05-02 09:15:00 | 899.35 | 2024-05-13 09:15:00 | 811.17 | TARGET_HIT | 0.50 | 9.80% |
| SELL | retest2 | 2024-05-03 11:30:00 | 895.25 | 2024-05-13 09:15:00 | 815.81 | TARGET_HIT | 0.50 | 8.87% |
| SELL | retest2 | 2024-05-03 15:15:00 | 889.00 | 2024-05-13 09:15:00 | 809.42 | TARGET_HIT | 0.50 | 8.95% |
| SELL | retest2 | 2024-05-06 14:00:00 | 898.55 | 2024-05-13 09:15:00 | 808.69 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-05-07 09:15:00 | 893.10 | 2024-05-13 09:15:00 | 803.79 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-06-07 09:15:00 | 848.90 | 2024-06-11 09:15:00 | 928.15 | STOP_HIT | 1.00 | -9.34% |
| SELL | retest2 | 2024-06-07 09:45:00 | 853.90 | 2024-06-11 09:15:00 | 928.15 | STOP_HIT | 1.00 | -8.70% |
| SELL | retest2 | 2024-06-07 12:15:00 | 854.35 | 2024-06-11 09:15:00 | 928.15 | STOP_HIT | 1.00 | -8.64% |
| SELL | retest2 | 2024-06-07 14:30:00 | 854.65 | 2024-06-11 09:15:00 | 928.15 | STOP_HIT | 1.00 | -8.60% |
| BUY | retest1 | 2024-07-25 11:00:00 | 1046.00 | 2024-07-26 09:15:00 | 1098.30 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2024-07-25 11:00:00 | 1046.00 | 2024-07-31 10:15:00 | 1040.20 | STOP_HIT | 0.50 | -0.55% |
| BUY | retest1 | 2024-07-26 09:15:00 | 1127.85 | 2024-08-05 11:15:00 | 1004.70 | STOP_HIT | 1.00 | -10.92% |
| BUY | retest1 | 2024-07-26 15:00:00 | 1060.65 | 2024-08-05 11:15:00 | 1004.70 | STOP_HIT | 1.00 | -5.28% |
| BUY | retest1 | 2024-07-31 11:45:00 | 1040.55 | 2024-08-05 11:15:00 | 1004.70 | STOP_HIT | 1.00 | -3.45% |
| BUY | retest2 | 2024-08-06 09:15:00 | 1034.45 | 2024-08-12 11:15:00 | 1002.80 | STOP_HIT | 1.00 | -3.06% |
| BUY | retest2 | 2024-08-06 10:00:00 | 1036.50 | 2024-08-12 11:15:00 | 1002.80 | STOP_HIT | 1.00 | -3.25% |
| BUY | retest2 | 2024-08-06 13:15:00 | 1039.80 | 2024-08-12 11:15:00 | 1002.80 | STOP_HIT | 1.00 | -3.56% |
| BUY | retest2 | 2024-08-09 09:15:00 | 1045.95 | 2024-08-12 11:15:00 | 1002.80 | STOP_HIT | 1.00 | -4.13% |
| BUY | retest2 | 2024-08-19 09:15:00 | 1044.65 | 2024-08-30 09:15:00 | 1149.12 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-08-20 09:15:00 | 1034.65 | 2024-08-30 09:15:00 | 1138.12 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-08-20 15:15:00 | 1045.00 | 2024-08-30 09:15:00 | 1149.50 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-11-13 13:45:00 | 1032.65 | 2024-11-14 13:15:00 | 1048.40 | STOP_HIT | 1.00 | 1.53% |
| SELL | retest2 | 2024-11-25 11:30:00 | 1117.35 | 2024-11-29 13:15:00 | 1061.48 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-11-26 09:15:00 | 1113.20 | 2024-11-29 13:15:00 | 1057.54 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-11-25 11:30:00 | 1117.35 | 2024-12-02 09:15:00 | 1005.61 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-11-26 09:15:00 | 1113.20 | 2024-12-02 09:15:00 | 1001.88 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-09-08 14:45:00 | 1261.90 | 2025-09-10 15:15:00 | 1283.00 | STOP_HIT | 1.00 | -1.67% |
| SELL | retest2 | 2025-09-09 09:45:00 | 1261.80 | 2025-09-10 15:15:00 | 1283.00 | STOP_HIT | 1.00 | -1.68% |
| SELL | retest2 | 2025-09-09 11:15:00 | 1255.60 | 2025-09-10 15:15:00 | 1283.00 | STOP_HIT | 1.00 | -2.18% |
| BUY | retest2 | 2025-09-18 09:15:00 | 1305.40 | 2025-09-19 09:15:00 | 1269.00 | STOP_HIT | 1.00 | -2.79% |
| BUY | retest2 | 2025-09-18 14:45:00 | 1284.00 | 2025-09-19 09:15:00 | 1269.00 | STOP_HIT | 1.00 | -1.17% |
| BUY | retest2 | 2025-09-18 15:15:00 | 1283.00 | 2025-09-19 09:15:00 | 1269.00 | STOP_HIT | 1.00 | -1.09% |
| SELL | retest2 | 2025-09-25 09:15:00 | 1259.40 | 2025-09-26 09:15:00 | 1196.43 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-25 09:15:00 | 1259.40 | 2025-10-15 10:15:00 | 1260.00 | STOP_HIT | 0.50 | -0.05% |
| BUY | retest2 | 2026-02-20 10:30:00 | 1156.80 | 2026-02-27 13:15:00 | 1088.20 | STOP_HIT | 1.00 | -5.93% |
| BUY | retest2 | 2026-02-20 12:30:00 | 1157.30 | 2026-02-27 13:15:00 | 1088.20 | STOP_HIT | 1.00 | -5.97% |
| BUY | retest2 | 2026-02-24 09:30:00 | 1159.00 | 2026-02-27 13:15:00 | 1088.20 | STOP_HIT | 1.00 | -6.11% |
| SELL | retest2 | 2026-04-09 09:30:00 | 1050.40 | 2026-04-10 09:15:00 | 1102.00 | STOP_HIT | 1.00 | -4.91% |
| SELL | retest2 | 2026-04-09 11:00:00 | 1054.55 | 2026-04-10 09:15:00 | 1102.00 | STOP_HIT | 1.00 | -4.50% |
