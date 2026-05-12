# NATCO Pharma Ltd. (NATCOPHARM)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 1174.90
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 12 |
| ALERT1 | 10 |
| ALERT2 | 10 |
| ALERT2_SKIP | 4 |
| ALERT3 | 31 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 12 |
| PARTIAL | 2 |
| TARGET_HIT | 4 |
| STOP_HIT | 8 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 14 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 7 / 7
- **Target hits / Stop hits / Partials:** 4 / 8 / 2
- **Avg / median % per leg:** 2.67% / 2.15%
- **Sum % (uncompounded):** 37.37%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 5 | 3 | 60.0% | 3 | 2 | 0 | 4.79% | 24.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 5 | 3 | 60.0% | 3 | 2 | 0 | 4.79% | 24.0% |
| SELL (all) | 9 | 4 | 44.4% | 1 | 6 | 2 | 1.49% | 13.4% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 9 | 4 | 44.4% | 1 | 6 | 2 | 1.49% | 13.4% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 14 | 7 | 50.0% | 4 | 8 | 2 | 2.67% | 37.4% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-11-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-03 12:15:00 | 747.75 | 824.45 | 824.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-03 13:15:00 | 743.95 | 823.64 | 824.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-22 09:15:00 | 806.10 | 794.10 | 805.70 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-22 09:15:00 | 806.10 | 794.10 | 805.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-22 09:15:00 | 806.10 | 794.10 | 805.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-22 10:00:00 | 806.10 | 794.10 | 805.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-22 10:15:00 | 806.00 | 794.21 | 805.70 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-22 11:15:00 | 799.85 | 794.21 | 805.70 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-12 10:15:00 | 759.86 | 788.37 | 798.09 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2023-12-20 09:15:00 | 782.65 | 782.59 | 793.01 | SL hit (close>ema200) qty=0.50 sl=782.59 alert=retest2 |

### Cycle 2 — BUY (started 2024-01-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-05 10:15:00 | 838.40 | 798.44 | 798.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-05 11:15:00 | 843.95 | 798.89 | 798.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-17 14:15:00 | 818.40 | 818.71 | 810.18 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-01-17 14:45:00 | 817.70 | 818.71 | 810.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-18 09:15:00 | 819.60 | 818.72 | 810.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-18 09:45:00 | 811.05 | 818.72 | 810.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-12 10:15:00 | 832.25 | 845.50 | 830.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-12 11:00:00 | 832.25 | 845.50 | 830.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-12 12:15:00 | 830.85 | 845.22 | 830.77 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-12 13:30:00 | 834.90 | 845.14 | 830.80 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-12 15:15:00 | 835.00 | 845.02 | 830.82 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-13 09:45:00 | 835.35 | 844.83 | 830.86 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Target hit | 2024-02-16 09:15:00 | 918.39 | 851.92 | 835.91 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 3 — SELL (started 2024-10-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-28 15:15:00 | 1320.90 | 1399.66 | 1399.67 | EMA200 below EMA400 |

### Cycle 4 — BUY (started 2024-11-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-06 14:15:00 | 1446.95 | 1398.99 | 1398.94 | EMA200 above EMA400 |

### Cycle 5 — SELL (started 2024-11-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-18 14:15:00 | 1356.95 | 1399.30 | 1399.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-18 15:15:00 | 1353.00 | 1398.84 | 1399.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-02 09:15:00 | 1438.00 | 1383.49 | 1390.41 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-02 09:15:00 | 1438.00 | 1383.49 | 1390.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-02 09:15:00 | 1438.00 | 1383.49 | 1390.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-02 10:00:00 | 1438.00 | 1383.49 | 1390.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-02 10:15:00 | 1429.90 | 1383.95 | 1390.61 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-02 13:45:00 | 1420.00 | 1385.28 | 1391.18 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-03 11:45:00 | 1421.10 | 1387.34 | 1392.07 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-04 09:15:00 | 1459.00 | 1389.72 | 1393.16 | SL hit (close>static) qty=1.00 sl=1455.00 alert=retest2 |

### Cycle 6 — BUY (started 2024-12-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-06 11:15:00 | 1442.50 | 1396.78 | 1396.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-06 12:15:00 | 1457.70 | 1397.39 | 1396.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-20 14:15:00 | 1412.00 | 1427.10 | 1414.81 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-20 14:15:00 | 1412.00 | 1427.10 | 1414.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 14:15:00 | 1412.00 | 1427.10 | 1414.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-20 15:00:00 | 1412.00 | 1427.10 | 1414.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 15:15:00 | 1426.00 | 1427.09 | 1414.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-23 09:30:00 | 1405.00 | 1426.88 | 1414.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-23 10:15:00 | 1401.85 | 1426.64 | 1414.76 | EMA400 retest candle locked (from upside) |

### Cycle 7 — SELL (started 2025-01-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-06 10:15:00 | 1321.35 | 1406.10 | 1406.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-06 13:15:00 | 1313.75 | 1403.62 | 1404.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-05 09:15:00 | 1265.65 | 1254.98 | 1306.37 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-05 10:00:00 | 1265.65 | 1254.98 | 1306.37 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 09:15:00 | 1293.70 | 1256.46 | 1305.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-06 09:30:00 | 1301.20 | 1256.46 | 1305.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 11:15:00 | 1317.05 | 1257.53 | 1305.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-06 11:45:00 | 1317.05 | 1257.53 | 1305.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 12:15:00 | 1325.00 | 1258.20 | 1305.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-06 12:45:00 | 1324.25 | 1258.20 | 1305.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 10:15:00 | 1313.50 | 1261.27 | 1305.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-07 11:00:00 | 1313.50 | 1261.27 | 1305.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 11:15:00 | 1312.55 | 1261.78 | 1305.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-07 12:15:00 | 1323.70 | 1261.78 | 1305.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 15:15:00 | 1317.95 | 1263.97 | 1306.15 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-10 09:15:00 | 1302.00 | 1263.97 | 1306.15 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-12 09:15:00 | 1236.90 | 1263.72 | 1303.01 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-02-13 09:15:00 | 1171.80 | 1258.08 | 1298.81 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 8 — BUY (started 2025-07-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-03 13:15:00 | 970.55 | 887.30 | 887.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-03 14:15:00 | 973.25 | 888.16 | 887.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-28 12:15:00 | 962.35 | 967.41 | 938.46 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-28 12:45:00 | 962.65 | 967.41 | 938.46 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 15:15:00 | 939.00 | 966.64 | 938.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-29 09:15:00 | 946.95 | 966.64 | 938.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 09:15:00 | 946.00 | 966.43 | 938.55 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-29 12:00:00 | 951.00 | 966.03 | 938.62 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-01 14:15:00 | 928.20 | 963.85 | 940.60 | SL hit (close<static) qty=1.00 sl=935.00 alert=retest2 |

### Cycle 9 — SELL (started 2025-08-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-21 11:15:00 | 888.70 | 927.61 | 927.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-21 12:15:00 | 885.00 | 927.19 | 927.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-18 10:15:00 | 892.65 | 877.56 | 895.15 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-18 10:15:00 | 892.65 | 877.56 | 895.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 10:15:00 | 892.65 | 877.56 | 895.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-18 10:45:00 | 890.80 | 877.56 | 895.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 09:15:00 | 832.60 | 823.12 | 839.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-19 09:30:00 | 842.50 | 823.12 | 839.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 09:15:00 | 862.00 | 824.03 | 839.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-20 10:00:00 | 862.00 | 824.03 | 839.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 10:15:00 | 883.25 | 824.62 | 840.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-20 11:00:00 | 883.25 | 824.62 | 840.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 15:15:00 | 840.80 | 830.94 | 841.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-25 09:15:00 | 830.50 | 830.94 | 841.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 09:15:00 | 836.00 | 830.99 | 841.93 | EMA400 retest candle locked (from downside) |

### Cycle 10 — BUY (started 2025-12-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-02 13:15:00 | 925.00 | 851.13 | 850.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-02 14:15:00 | 932.80 | 851.94 | 851.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-10 14:15:00 | 868.15 | 873.33 | 863.42 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-10 15:00:00 | 868.15 | 873.33 | 863.42 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 14:15:00 | 891.20 | 904.29 | 889.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-09 14:30:00 | 891.75 | 904.29 | 889.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 15:15:00 | 888.35 | 904.13 | 889.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-12 09:15:00 | 864.00 | 904.13 | 889.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 09:15:00 | 865.60 | 903.75 | 889.31 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-14 14:45:00 | 891.30 | 898.38 | 887.79 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-16 14:15:00 | 858.90 | 896.64 | 887.27 | SL hit (close<static) qty=1.00 sl=862.10 alert=retest2 |

### Cycle 11 — SELL (started 2026-01-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-23 14:15:00 | 830.75 | 879.29 | 879.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-27 09:15:00 | 818.45 | 878.22 | 878.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-11 10:15:00 | 855.75 | 852.30 | 862.60 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-11 10:30:00 | 858.80 | 852.30 | 862.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 14:15:00 | 840.00 | 852.24 | 862.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-12 14:30:00 | 859.05 | 852.24 | 862.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 09:15:00 | 917.00 | 850.99 | 860.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-16 10:00:00 | 917.00 | 850.99 | 860.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 10:15:00 | 927.30 | 851.75 | 861.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-16 11:00:00 | 927.30 | 851.75 | 861.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 12 — BUY (started 2026-02-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-24 15:15:00 | 943.55 | 869.18 | 868.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-25 09:15:00 | 976.85 | 870.25 | 869.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-16 12:15:00 | 939.80 | 941.05 | 912.38 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-16 13:00:00 | 939.80 | 941.05 | 912.38 | Sideways (15m bar) within 4 candles — skip ENTRY1 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2023-11-22 11:15:00 | 799.85 | 2023-12-12 10:15:00 | 759.86 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-11-22 11:15:00 | 799.85 | 2023-12-20 09:15:00 | 782.65 | STOP_HIT | 0.50 | 2.15% |
| SELL | retest2 | 2023-12-26 13:00:00 | 804.20 | 2023-12-27 09:15:00 | 813.35 | STOP_HIT | 1.00 | -1.14% |
| SELL | retest2 | 2023-12-27 12:30:00 | 801.55 | 2023-12-29 13:15:00 | 810.80 | STOP_HIT | 1.00 | -1.15% |
| SELL | retest2 | 2023-12-27 13:00:00 | 802.45 | 2023-12-29 13:15:00 | 810.80 | STOP_HIT | 1.00 | -1.04% |
| BUY | retest2 | 2024-02-12 13:30:00 | 834.90 | 2024-02-16 09:15:00 | 918.39 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-02-12 15:15:00 | 835.00 | 2024-02-16 09:15:00 | 918.50 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-02-13 09:45:00 | 835.35 | 2024-02-16 09:15:00 | 918.89 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-12-02 13:45:00 | 1420.00 | 2024-12-04 09:15:00 | 1459.00 | STOP_HIT | 1.00 | -2.75% |
| SELL | retest2 | 2024-12-03 11:45:00 | 1421.10 | 2024-12-04 09:15:00 | 1459.00 | STOP_HIT | 1.00 | -2.67% |
| SELL | retest2 | 2025-02-10 09:15:00 | 1302.00 | 2025-02-12 09:15:00 | 1236.90 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-10 09:15:00 | 1302.00 | 2025-02-13 09:15:00 | 1171.80 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2025-07-29 12:00:00 | 951.00 | 2025-08-01 14:15:00 | 928.20 | STOP_HIT | 1.00 | -2.40% |
| BUY | retest2 | 2026-01-14 14:45:00 | 891.30 | 2026-01-16 14:15:00 | 858.90 | STOP_HIT | 1.00 | -3.64% |
