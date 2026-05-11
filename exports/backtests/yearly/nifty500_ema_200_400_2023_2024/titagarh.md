# Titagarh Rail Systems Ltd. (TITAGARH)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 840.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 10 |
| ALERT1 | 9 |
| ALERT2 | 9 |
| ALERT2_SKIP | 4 |
| ALERT3 | 48 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 37 |
| PARTIAL | 10 |
| TARGET_HIT | 10 |
| STOP_HIT | 27 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 47 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 20 / 27
- **Target hits / Stop hits / Partials:** 10 / 27 / 10
- **Avg / median % per leg:** 1.25% / -0.34%
- **Sum % (uncompounded):** 58.79%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 17 | 2 | 11.8% | 2 | 15 | 0 | -0.14% | -2.4% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 17 | 2 | 11.8% | 2 | 15 | 0 | -0.14% | -2.4% |
| SELL (all) | 30 | 18 | 60.0% | 8 | 12 | 10 | 2.04% | 61.2% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 30 | 18 | 60.0% | 8 | 12 | 10 | 2.04% | 61.2% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 47 | 20 | 42.6% | 10 | 27 | 10 | 1.25% | 58.8% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-03-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-05 11:15:00 | 912.80 | 986.78 | 986.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-06 09:15:00 | 887.60 | 983.12 | 985.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-27 10:15:00 | 927.50 | 916.84 | 943.75 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-27 10:15:00 | 927.50 | 916.84 | 943.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-27 10:15:00 | 927.50 | 916.84 | 943.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-12 09:15:00 | 944.85 | 932.99 | 945.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-12 09:15:00 | 938.30 | 933.04 | 945.22 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-12 13:45:00 | 932.70 | 933.25 | 945.08 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-12 15:00:00 | 932.55 | 933.24 | 945.02 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-15 09:15:00 | 886.07 | 933.02 | 944.79 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-15 09:15:00 | 885.92 | 933.02 | 944.79 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-04-16 09:15:00 | 934.80 | 931.92 | 943.83 | SL hit (close>ema200) qty=0.50 sl=931.92 alert=retest2 |

### Cycle 2 — BUY (started 2024-04-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-25 11:15:00 | 1044.45 | 953.11 | 953.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-26 09:15:00 | 1051.50 | 957.65 | 955.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-04 11:15:00 | 1194.00 | 1195.85 | 1109.03 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-04 12:00:00 | 1194.00 | 1195.85 | 1109.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 09:15:00 | 1044.90 | 1195.22 | 1110.86 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-10 09:15:00 | 1273.00 | 1189.46 | 1115.77 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2024-06-12 11:15:00 | 1400.30 | 1213.31 | 1134.16 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 3 — SELL (started 2024-09-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-09 13:15:00 | 1374.45 | 1457.71 | 1458.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-11 11:15:00 | 1370.00 | 1449.42 | 1453.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-30 12:15:00 | 1207.80 | 1193.65 | 1264.10 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-30 12:45:00 | 1207.20 | 1193.65 | 1264.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 09:15:00 | 1234.70 | 1164.18 | 1213.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-28 10:00:00 | 1234.70 | 1164.18 | 1213.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 10:15:00 | 1231.65 | 1164.85 | 1213.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-28 10:30:00 | 1236.30 | 1164.85 | 1213.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 14:15:00 | 1229.85 | 1167.19 | 1213.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-28 15:00:00 | 1229.85 | 1167.19 | 1213.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-04 09:15:00 | 1213.85 | 1173.30 | 1211.92 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-05 09:30:00 | 1186.10 | 1175.22 | 1211.58 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-05 14:45:00 | 1187.60 | 1175.91 | 1211.02 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-06 11:45:00 | 1189.95 | 1176.33 | 1210.54 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-06 12:30:00 | 1186.50 | 1176.45 | 1210.43 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-09 09:15:00 | 1200.20 | 1177.02 | 1210.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-09 09:30:00 | 1210.00 | 1177.02 | 1210.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-09 10:15:00 | 1209.55 | 1177.35 | 1210.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-09 11:15:00 | 1214.05 | 1177.35 | 1210.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-09 11:15:00 | 1214.25 | 1177.72 | 1210.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-09 11:45:00 | 1212.00 | 1177.72 | 1210.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-09 12:15:00 | 1215.00 | 1178.09 | 1210.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-09 12:45:00 | 1216.05 | 1178.09 | 1210.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-09 15:15:00 | 1214.20 | 1179.15 | 1210.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-10 09:15:00 | 1237.50 | 1179.15 | 1210.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-12-11 09:15:00 | 1330.65 | 1184.03 | 1211.42 | SL hit (close>static) qty=1.00 sl=1243.00 alert=retest2 |

### Cycle 4 — BUY (started 2024-12-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-20 13:15:00 | 1272.05 | 1233.37 | 1233.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-20 14:15:00 | 1318.10 | 1234.22 | 1233.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-24 09:15:00 | 1234.00 | 1236.63 | 1234.99 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-24 09:15:00 | 1234.00 | 1236.63 | 1234.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-24 09:15:00 | 1234.00 | 1236.63 | 1234.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-24 09:30:00 | 1231.15 | 1236.63 | 1234.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-24 10:15:00 | 1232.10 | 1236.58 | 1234.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-24 11:00:00 | 1232.10 | 1236.58 | 1234.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-24 11:15:00 | 1228.20 | 1236.50 | 1234.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-24 12:00:00 | 1228.20 | 1236.50 | 1234.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-24 12:15:00 | 1225.30 | 1236.39 | 1234.89 | EMA400 retest candle locked (from upside) |

### Cycle 5 — SELL (started 2024-12-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-26 13:15:00 | 1182.50 | 1233.41 | 1233.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-26 14:15:00 | 1176.60 | 1232.84 | 1233.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-01 09:15:00 | 1095.65 | 1054.75 | 1113.66 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-01 10:00:00 | 1095.65 | 1054.75 | 1113.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-14 13:15:00 | 795.90 | 757.59 | 797.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-14 13:45:00 | 796.05 | 757.59 | 797.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-14 14:15:00 | 805.35 | 758.06 | 797.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-14 14:45:00 | 804.45 | 758.06 | 797.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-14 15:15:00 | 810.50 | 758.58 | 797.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-15 09:15:00 | 811.00 | 758.58 | 797.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 6 — BUY (started 2025-05-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-27 14:15:00 | 938.80 | 825.84 | 825.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-06 13:15:00 | 942.75 | 863.95 | 846.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-16 09:15:00 | 881.60 | 884.78 | 861.52 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-16 09:45:00 | 873.90 | 884.78 | 861.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 09:15:00 | 862.95 | 884.52 | 863.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-19 10:00:00 | 862.95 | 884.52 | 863.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 10:15:00 | 855.70 | 884.23 | 863.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-19 10:30:00 | 853.00 | 884.23 | 863.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 12:15:00 | 865.00 | 881.22 | 863.04 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-20 15:15:00 | 872.00 | 880.82 | 863.02 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-06-27 09:15:00 | 959.20 | 890.14 | 870.38 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 7 — SELL (started 2025-08-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-08 09:15:00 | 834.75 | 890.39 | 890.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-08 11:15:00 | 826.55 | 889.22 | 889.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-21 13:15:00 | 858.90 | 857.61 | 871.35 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-21 14:00:00 | 858.90 | 857.61 | 871.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 09:15:00 | 853.00 | 857.53 | 871.11 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-22 10:45:00 | 849.35 | 857.41 | 870.98 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-26 09:30:00 | 849.50 | 857.83 | 870.34 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-28 09:15:00 | 848.25 | 858.39 | 870.26 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-28 11:15:00 | 849.35 | 858.26 | 870.08 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 10:15:00 | 874.00 | 852.54 | 863.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-09 10:30:00 | 876.35 | 852.54 | 863.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 11:15:00 | 879.10 | 852.80 | 863.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-09 11:30:00 | 878.00 | 852.80 | 863.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-09-09 15:15:00 | 886.45 | 853.93 | 864.30 | SL hit (close>static) qty=1.00 sl=885.50 alert=retest2 |

### Cycle 8 — BUY (started 2025-09-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-16 13:15:00 | 934.65 | 873.48 | 873.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-16 14:15:00 | 937.70 | 874.12 | 873.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-26 11:15:00 | 888.85 | 895.49 | 886.07 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-26 11:15:00 | 888.85 | 895.49 | 886.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 11:15:00 | 888.85 | 895.49 | 886.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-26 11:30:00 | 887.00 | 895.49 | 886.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 13:15:00 | 882.05 | 895.27 | 886.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-26 14:00:00 | 882.05 | 895.27 | 886.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 14:15:00 | 880.20 | 895.12 | 886.02 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-29 09:15:00 | 893.70 | 894.98 | 886.00 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-30 09:45:00 | 884.50 | 894.46 | 886.09 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-30 11:15:00 | 872.50 | 894.09 | 885.99 | SL hit (close<static) qty=1.00 sl=876.50 alert=retest2 |

### Cycle 9 — SELL (started 2025-11-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-10 11:15:00 | 846.65 | 885.87 | 885.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-10 12:15:00 | 846.00 | 885.48 | 885.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-12 09:15:00 | 886.95 | 882.66 | 884.31 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-12 09:15:00 | 886.95 | 882.66 | 884.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 09:15:00 | 886.95 | 882.66 | 884.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-12 10:00:00 | 886.95 | 882.66 | 884.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 10:15:00 | 887.35 | 882.71 | 884.33 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-12 13:45:00 | 881.40 | 882.78 | 884.34 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-17 09:15:00 | 893.80 | 880.75 | 883.16 | SL hit (close>static) qty=1.00 sl=891.90 alert=retest2 |

### Cycle 10 — BUY (started 2026-05-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-05 12:15:00 | 863.85 | 725.43 | 724.75 | EMA200 above EMA400 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2024-04-12 13:45:00 | 932.70 | 2024-04-15 09:15:00 | 886.07 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-04-12 15:00:00 | 932.55 | 2024-04-15 09:15:00 | 885.92 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-04-12 13:45:00 | 932.70 | 2024-04-16 09:15:00 | 934.80 | STOP_HIT | 0.50 | -0.23% |
| SELL | retest2 | 2024-04-12 15:00:00 | 932.55 | 2024-04-16 09:15:00 | 934.80 | STOP_HIT | 0.50 | -0.24% |
| BUY | retest2 | 2024-06-10 09:15:00 | 1273.00 | 2024-06-12 11:15:00 | 1400.30 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-12-05 09:30:00 | 1186.10 | 2024-12-11 09:15:00 | 1330.65 | STOP_HIT | 1.00 | -12.19% |
| SELL | retest2 | 2024-12-05 14:45:00 | 1187.60 | 2024-12-11 09:15:00 | 1330.65 | STOP_HIT | 1.00 | -12.05% |
| SELL | retest2 | 2024-12-06 11:45:00 | 1189.95 | 2024-12-11 09:15:00 | 1330.65 | STOP_HIT | 1.00 | -11.82% |
| SELL | retest2 | 2024-12-06 12:30:00 | 1186.50 | 2024-12-11 09:15:00 | 1330.65 | STOP_HIT | 1.00 | -12.15% |
| BUY | retest2 | 2025-06-20 15:15:00 | 872.00 | 2025-06-27 09:15:00 | 959.20 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-07-29 12:00:00 | 872.20 | 2025-08-01 09:15:00 | 854.95 | STOP_HIT | 1.00 | -1.98% |
| BUY | retest2 | 2025-07-29 13:45:00 | 867.50 | 2025-08-01 09:15:00 | 854.95 | STOP_HIT | 1.00 | -1.45% |
| BUY | retest2 | 2025-07-29 14:15:00 | 867.50 | 2025-08-01 09:15:00 | 854.95 | STOP_HIT | 1.00 | -1.45% |
| SELL | retest2 | 2025-08-22 10:45:00 | 849.35 | 2025-09-09 15:15:00 | 886.45 | STOP_HIT | 1.00 | -4.37% |
| SELL | retest2 | 2025-08-26 09:30:00 | 849.50 | 2025-09-09 15:15:00 | 886.45 | STOP_HIT | 1.00 | -4.35% |
| SELL | retest2 | 2025-08-28 09:15:00 | 848.25 | 2025-09-09 15:15:00 | 886.45 | STOP_HIT | 1.00 | -4.50% |
| SELL | retest2 | 2025-08-28 11:15:00 | 849.35 | 2025-09-09 15:15:00 | 886.45 | STOP_HIT | 1.00 | -4.37% |
| BUY | retest2 | 2025-09-29 09:15:00 | 893.70 | 2025-09-30 11:15:00 | 872.50 | STOP_HIT | 1.00 | -2.37% |
| BUY | retest2 | 2025-09-30 09:45:00 | 884.50 | 2025-09-30 11:15:00 | 872.50 | STOP_HIT | 1.00 | -1.36% |
| BUY | retest2 | 2025-10-01 15:15:00 | 886.00 | 2025-10-06 09:15:00 | 882.95 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest2 | 2025-10-03 11:00:00 | 883.00 | 2025-10-14 14:15:00 | 882.50 | STOP_HIT | 1.00 | -0.06% |
| BUY | retest2 | 2025-10-03 15:15:00 | 890.95 | 2025-10-14 14:15:00 | 882.50 | STOP_HIT | 1.00 | -0.95% |
| BUY | retest2 | 2025-10-06 14:30:00 | 892.50 | 2025-10-16 12:15:00 | 882.10 | STOP_HIT | 1.00 | -1.17% |
| BUY | retest2 | 2025-10-13 12:30:00 | 890.55 | 2025-10-17 12:15:00 | 868.15 | STOP_HIT | 1.00 | -2.52% |
| BUY | retest2 | 2025-10-15 09:15:00 | 890.35 | 2025-10-17 12:15:00 | 868.15 | STOP_HIT | 1.00 | -2.49% |
| BUY | retest2 | 2025-10-23 09:30:00 | 889.00 | 2025-10-23 14:15:00 | 878.75 | STOP_HIT | 1.00 | -1.15% |
| BUY | retest2 | 2025-10-23 10:45:00 | 887.00 | 2025-10-23 14:15:00 | 878.75 | STOP_HIT | 1.00 | -0.93% |
| BUY | retest2 | 2025-10-27 11:45:00 | 888.55 | 2025-11-04 13:15:00 | 878.55 | STOP_HIT | 1.00 | -1.13% |
| BUY | retest2 | 2025-11-03 09:15:00 | 906.00 | 2025-11-04 13:15:00 | 878.55 | STOP_HIT | 1.00 | -3.03% |
| SELL | retest2 | 2025-11-12 13:45:00 | 881.40 | 2025-11-17 09:15:00 | 893.80 | STOP_HIT | 1.00 | -1.41% |
| SELL | retest2 | 2025-11-17 09:30:00 | 882.00 | 2025-11-25 09:15:00 | 839.09 | PARTIAL | 0.50 | 4.87% |
| SELL | retest2 | 2025-11-17 11:30:00 | 883.25 | 2025-11-25 09:15:00 | 838.94 | PARTIAL | 0.50 | 5.02% |
| SELL | retest2 | 2025-11-17 12:00:00 | 883.10 | 2025-11-25 14:15:00 | 837.90 | PARTIAL | 0.50 | 5.12% |
| SELL | retest2 | 2025-11-17 14:15:00 | 880.30 | 2025-11-25 14:15:00 | 836.28 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-17 15:15:00 | 880.00 | 2025-11-25 15:15:00 | 836.00 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-17 09:30:00 | 882.00 | 2025-12-03 13:15:00 | 793.80 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-11-17 11:30:00 | 883.25 | 2025-12-03 13:15:00 | 794.93 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-11-17 12:00:00 | 883.10 | 2025-12-03 13:15:00 | 794.79 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-11-17 14:15:00 | 880.30 | 2025-12-03 13:15:00 | 792.27 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-11-17 15:15:00 | 880.00 | 2025-12-03 14:15:00 | 792.00 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-12-30 15:00:00 | 878.20 | 2025-12-31 09:15:00 | 888.45 | STOP_HIT | 1.00 | -1.17% |
| SELL | retest2 | 2026-01-05 14:00:00 | 880.70 | 2026-01-08 10:15:00 | 836.66 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-05 14:00:00 | 880.70 | 2026-01-12 09:15:00 | 792.63 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-02-01 12:15:00 | 789.75 | 2026-02-16 09:15:00 | 750.26 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-01 13:00:00 | 798.75 | 2026-02-16 09:15:00 | 758.81 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-01 12:15:00 | 789.75 | 2026-02-25 15:15:00 | 718.88 | TARGET_HIT | 0.50 | 8.97% |
| SELL | retest2 | 2026-02-01 13:00:00 | 798.75 | 2026-02-26 14:15:00 | 710.77 | TARGET_HIT | 0.50 | 11.01% |
