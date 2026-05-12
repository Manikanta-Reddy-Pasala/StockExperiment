# Great Eastern Shipping Co. Ltd. (GESHIP)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 1589.10
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 12 |
| ALERT1 | 9 |
| ALERT2 | 9 |
| ALERT2_SKIP | 4 |
| ALERT3 | 60 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 44 |
| PARTIAL | 5 |
| TARGET_HIT | 13 |
| STOP_HIT | 31 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 49 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 23 / 26
- **Target hits / Stop hits / Partials:** 13 / 31 / 5
- **Avg / median % per leg:** 1.89% / -1.13%
- **Sum % (uncompounded):** 92.64%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 28 | 12 | 42.9% | 12 | 16 | 0 | 2.55% | 71.3% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 28 | 12 | 42.9% | 12 | 16 | 0 | 2.55% | 71.3% |
| SELL (all) | 21 | 11 | 52.4% | 1 | 15 | 5 | 1.02% | 21.3% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 21 | 11 | 52.4% | 1 | 15 | 5 | 1.02% | 21.3% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 49 | 23 | 46.9% | 13 | 31 | 5 | 1.89% | 92.6% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-11-15 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-15 13:15:00 | 792.30 | 802.09 | 802.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-15 15:15:00 | 790.70 | 801.88 | 802.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-22 15:15:00 | 801.00 | 797.71 | 799.68 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-22 15:15:00 | 801.00 | 797.71 | 799.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-22 15:15:00 | 801.00 | 797.71 | 799.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-23 09:15:00 | 809.90 | 797.71 | 799.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-23 09:15:00 | 810.75 | 797.84 | 799.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-23 09:45:00 | 807.75 | 797.84 | 799.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 2 — BUY (started 2023-11-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-28 09:15:00 | 822.30 | 801.58 | 801.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-30 10:15:00 | 843.50 | 803.80 | 802.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-18 09:15:00 | 944.45 | 946.08 | 907.24 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-01-18 09:45:00 | 938.00 | 946.08 | 907.24 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-09 09:15:00 | 960.20 | 974.16 | 940.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-09 09:45:00 | 940.00 | 974.16 | 940.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-12 09:15:00 | 936.70 | 972.96 | 940.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-12 10:00:00 | 936.70 | 972.96 | 940.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-12 10:15:00 | 944.30 | 972.67 | 940.84 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-13 10:15:00 | 946.40 | 970.28 | 940.57 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-13 13:00:00 | 952.85 | 969.51 | 940.62 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-14 10:45:00 | 946.25 | 968.10 | 940.62 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-15 09:15:00 | 948.65 | 966.66 | 940.57 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-19 09:15:00 | 930.50 | 965.19 | 941.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-19 10:00:00 | 930.50 | 965.19 | 941.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-19 10:15:00 | 938.70 | 964.93 | 941.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-19 10:30:00 | 933.05 | 964.93 | 941.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-19 11:15:00 | 932.10 | 964.60 | 941.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-19 12:00:00 | 932.10 | 964.60 | 941.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-02-19 13:15:00 | 925.50 | 963.88 | 941.51 | SL hit (close<static) qty=1.00 sl=927.05 alert=retest2 |

### Cycle 3 — SELL (started 2024-09-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-26 15:15:00 | 1206.50 | 1271.59 | 1271.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-30 09:15:00 | 1190.45 | 1267.00 | 1269.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-08 12:15:00 | 1251.50 | 1250.02 | 1259.63 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-08 13:00:00 | 1251.50 | 1250.02 | 1259.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 14:15:00 | 1272.65 | 1250.30 | 1259.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-08 15:00:00 | 1272.65 | 1250.30 | 1259.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 15:15:00 | 1275.00 | 1250.55 | 1259.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-09 09:15:00 | 1283.20 | 1250.55 | 1259.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-11 10:15:00 | 1267.45 | 1255.31 | 1261.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-11 11:00:00 | 1267.45 | 1255.31 | 1261.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-11 11:15:00 | 1266.00 | 1255.41 | 1261.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-11 12:15:00 | 1270.25 | 1255.41 | 1261.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-11 12:15:00 | 1271.55 | 1255.58 | 1261.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-11 12:30:00 | 1275.45 | 1255.58 | 1261.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 4 — BUY (started 2024-10-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-17 15:15:00 | 1313.00 | 1266.68 | 1266.60 | EMA200 above EMA400 |

### Cycle 5 — SELL (started 2024-10-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-22 09:15:00 | 1226.10 | 1266.18 | 1266.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-23 09:15:00 | 1196.85 | 1263.46 | 1264.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-30 10:15:00 | 1261.65 | 1253.19 | 1259.03 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-30 10:15:00 | 1261.65 | 1253.19 | 1259.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-30 10:15:00 | 1261.65 | 1253.19 | 1259.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-30 11:00:00 | 1261.65 | 1253.19 | 1259.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-30 11:15:00 | 1275.70 | 1253.42 | 1259.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-30 11:30:00 | 1275.65 | 1253.42 | 1259.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 10:15:00 | 1280.95 | 1257.47 | 1260.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-04 10:45:00 | 1279.25 | 1257.47 | 1260.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 11:15:00 | 1281.55 | 1257.70 | 1260.91 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-06 10:00:00 | 1276.70 | 1261.07 | 1262.46 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-07 09:15:00 | 1308.05 | 1261.45 | 1262.60 | SL hit (close>static) qty=1.00 sl=1285.00 alert=retest2 |

### Cycle 6 — BUY (started 2024-11-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-07 15:15:00 | 1304.00 | 1264.02 | 1263.87 | EMA200 above EMA400 |

### Cycle 7 — SELL (started 2024-11-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-08 09:15:00 | 1208.40 | 1263.46 | 1263.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-11 13:15:00 | 1183.15 | 1257.16 | 1260.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-31 10:15:00 | 976.40 | 965.73 | 1022.14 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-31 11:00:00 | 976.40 | 965.73 | 1022.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-19 11:15:00 | 934.95 | 894.94 | 932.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-19 12:00:00 | 934.95 | 894.94 | 932.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-19 12:15:00 | 928.85 | 895.28 | 932.75 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-19 13:30:00 | 926.50 | 895.65 | 932.75 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-20 09:15:00 | 937.85 | 896.76 | 932.76 | SL hit (close>static) qty=1.00 sl=936.50 alert=retest2 |

### Cycle 8 — BUY (started 2025-05-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-30 14:15:00 | 931.00 | 917.74 | 917.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-02 09:15:00 | 965.55 | 918.35 | 918.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-19 10:15:00 | 953.10 | 957.65 | 942.26 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-19 10:30:00 | 953.35 | 957.65 | 942.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 09:15:00 | 942.25 | 957.41 | 942.59 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-20 11:00:00 | 961.55 | 957.45 | 942.68 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-25 14:15:00 | 958.65 | 986.55 | 972.50 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-28 09:15:00 | 962.90 | 985.99 | 972.36 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-28 14:15:00 | 934.50 | 983.95 | 971.72 | SL hit (close<static) qty=1.00 sl=935.70 alert=retest2 |

### Cycle 9 — SELL (started 2025-08-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-07 12:15:00 | 923.00 | 962.31 | 962.44 | EMA200 below EMA400 |

### Cycle 10 — BUY (started 2025-08-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-20 14:15:00 | 983.35 | 961.80 | 961.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-21 09:15:00 | 985.00 | 962.24 | 962.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-25 09:15:00 | 961.95 | 963.66 | 962.77 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-25 09:15:00 | 961.95 | 963.66 | 962.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 09:15:00 | 961.95 | 963.66 | 962.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-25 09:45:00 | 959.60 | 963.66 | 962.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 10:15:00 | 961.90 | 963.64 | 962.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-25 11:15:00 | 961.30 | 963.64 | 962.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 11:15:00 | 961.50 | 963.62 | 962.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-25 12:00:00 | 961.50 | 963.62 | 962.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 14:15:00 | 955.40 | 963.48 | 962.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-25 15:00:00 | 955.40 | 963.48 | 962.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 11 — SELL (started 2025-08-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-28 10:15:00 | 942.05 | 961.86 | 961.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-28 11:15:00 | 938.20 | 961.62 | 961.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-02 09:15:00 | 961.35 | 957.52 | 959.61 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-02 09:15:00 | 961.35 | 957.52 | 959.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 09:15:00 | 961.35 | 957.52 | 959.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-02 09:45:00 | 959.30 | 957.52 | 959.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 10:15:00 | 971.00 | 957.65 | 959.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-02 10:30:00 | 970.75 | 957.65 | 959.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 14:15:00 | 964.00 | 958.00 | 959.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-02 15:00:00 | 964.00 | 958.00 | 959.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 15:15:00 | 958.00 | 958.00 | 959.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-03 09:45:00 | 963.40 | 958.08 | 959.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 10:15:00 | 962.85 | 958.13 | 959.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-03 10:30:00 | 963.50 | 958.13 | 959.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 14:15:00 | 958.95 | 958.26 | 959.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-03 14:30:00 | 960.85 | 958.26 | 959.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 15:15:00 | 959.00 | 958.27 | 959.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-04 09:15:00 | 963.50 | 958.27 | 959.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 09:15:00 | 968.00 | 958.37 | 959.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-04 09:30:00 | 971.50 | 958.37 | 959.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 10:15:00 | 969.95 | 958.48 | 959.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-04 10:45:00 | 970.95 | 958.48 | 959.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 10:15:00 | 964.00 | 959.95 | 960.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-08 11:00:00 | 964.00 | 959.95 | 960.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 11:15:00 | 963.05 | 959.99 | 960.64 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-08 14:00:00 | 953.85 | 959.97 | 960.62 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-09 09:15:00 | 979.00 | 960.01 | 960.64 | SL hit (close>static) qty=1.00 sl=965.50 alert=retest2 |

### Cycle 12 — BUY (started 2025-09-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-10 09:15:00 | 989.45 | 961.49 | 961.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-15 09:15:00 | 998.90 | 966.84 | 964.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-26 11:15:00 | 997.45 | 1000.47 | 984.56 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-26 12:00:00 | 997.45 | 1000.47 | 984.56 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 09:15:00 | 1002.95 | 1000.17 | 984.80 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-30 09:45:00 | 1013.05 | 999.72 | 985.11 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-03 09:15:00 | 1032.40 | 998.68 | 985.49 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2025-10-31 11:15:00 | 1114.36 | 1035.46 | 1015.45 | Target hit (10%) qty=1.00 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-02-13 10:15:00 | 946.40 | 2024-02-19 13:15:00 | 925.50 | STOP_HIT | 1.00 | -2.21% |
| BUY | retest2 | 2024-02-13 13:00:00 | 952.85 | 2024-02-19 13:15:00 | 925.50 | STOP_HIT | 1.00 | -2.87% |
| BUY | retest2 | 2024-02-14 10:45:00 | 946.25 | 2024-02-19 13:15:00 | 925.50 | STOP_HIT | 1.00 | -2.19% |
| BUY | retest2 | 2024-02-15 09:15:00 | 948.65 | 2024-02-19 13:15:00 | 925.50 | STOP_HIT | 1.00 | -2.44% |
| BUY | retest2 | 2024-02-21 09:15:00 | 941.00 | 2024-02-21 12:15:00 | 926.95 | STOP_HIT | 1.00 | -1.49% |
| BUY | retest2 | 2024-02-22 13:15:00 | 939.50 | 2024-02-22 14:15:00 | 923.00 | STOP_HIT | 1.00 | -1.76% |
| BUY | retest2 | 2024-02-23 09:30:00 | 937.10 | 2024-03-05 09:15:00 | 1030.81 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-03-14 09:30:00 | 937.00 | 2024-04-02 09:15:00 | 1030.70 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-03-14 13:15:00 | 958.80 | 2024-04-04 09:15:00 | 1054.68 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-03-14 14:45:00 | 961.25 | 2024-04-04 09:15:00 | 1057.38 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-03-15 10:30:00 | 961.40 | 2024-04-04 09:15:00 | 1057.54 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-03-15 12:00:00 | 962.55 | 2024-04-04 09:15:00 | 1058.81 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-05-13 11:30:00 | 1018.15 | 2024-05-14 09:15:00 | 1119.97 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-05-14 09:15:00 | 1087.00 | 2024-05-28 12:15:00 | 993.50 | STOP_HIT | 1.00 | -8.60% |
| BUY | retest2 | 2024-05-24 15:15:00 | 1019.00 | 2024-05-28 12:15:00 | 993.50 | STOP_HIT | 1.00 | -2.50% |
| BUY | retest2 | 2024-05-28 09:15:00 | 1015.50 | 2024-05-28 12:15:00 | 993.50 | STOP_HIT | 1.00 | -2.17% |
| BUY | retest2 | 2024-05-29 09:15:00 | 1052.00 | 2024-06-04 10:15:00 | 990.45 | STOP_HIT | 1.00 | -5.85% |
| BUY | retest2 | 2024-06-04 09:30:00 | 1022.85 | 2024-06-04 10:15:00 | 990.45 | STOP_HIT | 1.00 | -3.17% |
| BUY | retest2 | 2024-06-04 13:15:00 | 1010.85 | 2024-06-04 14:15:00 | 991.10 | STOP_HIT | 1.00 | -1.95% |
| BUY | retest2 | 2024-06-05 09:15:00 | 1014.15 | 2024-06-07 13:15:00 | 1115.57 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-06-05 11:15:00 | 1037.95 | 2024-06-07 13:15:00 | 1141.75 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-06-05 12:00:00 | 1037.10 | 2024-06-07 13:15:00 | 1140.81 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-11-06 10:00:00 | 1276.70 | 2024-11-07 09:15:00 | 1308.05 | STOP_HIT | 1.00 | -2.46% |
| SELL | retest2 | 2025-03-19 13:30:00 | 926.50 | 2025-03-20 09:15:00 | 937.85 | STOP_HIT | 1.00 | -1.23% |
| SELL | retest2 | 2025-03-25 12:45:00 | 926.30 | 2025-03-28 09:15:00 | 944.60 | STOP_HIT | 1.00 | -1.98% |
| SELL | retest2 | 2025-03-26 09:30:00 | 921.35 | 2025-03-28 09:15:00 | 944.60 | STOP_HIT | 1.00 | -2.52% |
| SELL | retest2 | 2025-04-01 10:45:00 | 925.50 | 2025-04-03 14:15:00 | 937.85 | STOP_HIT | 1.00 | -1.33% |
| SELL | retest2 | 2025-04-04 10:00:00 | 916.65 | 2025-04-07 09:15:00 | 824.99 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-04-22 12:45:00 | 918.90 | 2025-04-25 11:15:00 | 876.33 | PARTIAL | 0.50 | 4.63% |
| SELL | retest2 | 2025-04-22 12:45:00 | 918.90 | 2025-04-28 13:15:00 | 902.20 | STOP_HIT | 0.50 | 1.82% |
| SELL | retest2 | 2025-04-23 10:00:00 | 918.60 | 2025-04-30 14:15:00 | 872.95 | PARTIAL | 0.50 | 4.97% |
| SELL | retest2 | 2025-04-23 10:30:00 | 917.60 | 2025-04-30 14:15:00 | 872.67 | PARTIAL | 0.50 | 4.90% |
| SELL | retest2 | 2025-04-24 11:15:00 | 922.45 | 2025-04-30 14:15:00 | 871.72 | PARTIAL | 0.50 | 5.50% |
| SELL | retest2 | 2025-04-23 10:00:00 | 918.60 | 2025-05-05 11:15:00 | 912.90 | STOP_HIT | 0.50 | 0.62% |
| SELL | retest2 | 2025-04-23 10:30:00 | 917.60 | 2025-05-05 11:15:00 | 912.90 | STOP_HIT | 0.50 | 0.51% |
| SELL | retest2 | 2025-04-24 11:15:00 | 922.45 | 2025-05-05 11:15:00 | 912.90 | STOP_HIT | 0.50 | 1.04% |
| SELL | retest2 | 2025-05-05 14:00:00 | 921.75 | 2025-05-05 14:15:00 | 931.40 | STOP_HIT | 1.00 | -1.05% |
| SELL | retest2 | 2025-05-06 09:15:00 | 912.50 | 2025-05-08 13:15:00 | 866.88 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-05-06 09:15:00 | 912.50 | 2025-05-12 10:15:00 | 900.90 | STOP_HIT | 0.50 | 1.27% |
| SELL | retest2 | 2025-05-14 13:30:00 | 923.70 | 2025-05-16 09:15:00 | 950.45 | STOP_HIT | 1.00 | -2.90% |
| SELL | retest2 | 2025-05-21 11:30:00 | 918.00 | 2025-05-23 10:15:00 | 928.40 | STOP_HIT | 1.00 | -1.13% |
| SELL | retest2 | 2025-05-22 11:00:00 | 913.00 | 2025-05-23 10:15:00 | 928.40 | STOP_HIT | 1.00 | -1.69% |
| BUY | retest2 | 2025-06-20 11:00:00 | 961.55 | 2025-07-28 14:15:00 | 934.50 | STOP_HIT | 1.00 | -2.81% |
| BUY | retest2 | 2025-07-25 14:15:00 | 958.65 | 2025-07-28 14:15:00 | 934.50 | STOP_HIT | 1.00 | -2.52% |
| BUY | retest2 | 2025-07-28 09:15:00 | 962.90 | 2025-07-28 14:15:00 | 934.50 | STOP_HIT | 1.00 | -2.95% |
| BUY | retest2 | 2025-07-29 14:45:00 | 959.90 | 2025-08-01 09:15:00 | 929.05 | STOP_HIT | 1.00 | -3.21% |
| SELL | retest2 | 2025-09-08 14:00:00 | 953.85 | 2025-09-09 09:15:00 | 979.00 | STOP_HIT | 1.00 | -2.64% |
| BUY | retest2 | 2025-09-30 09:45:00 | 1013.05 | 2025-10-31 11:15:00 | 1114.36 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-10-03 09:15:00 | 1032.40 | 2025-10-31 11:15:00 | 1135.64 | TARGET_HIT | 1.00 | 10.00% |
