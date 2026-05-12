# Amara Raja Energy & Mobility Ltd. (ARE&M)

## Backtest Summary

- **Window:** 2025-03-13 09:15:00 → 2026-05-08 15:15:00 (1976 bars)
- **Last close:** 890.85
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 67 |
| ALERT1 | 42 |
| ALERT2 | 42 |
| ALERT2_SKIP | 26 |
| ALERT3 | 116 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 68 |
| PARTIAL | 8 |
| TARGET_HIT | 3 |
| STOP_HIT | 65 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 76 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 36 / 40
- **Target hits / Stop hits / Partials:** 3 / 65 / 8
- **Avg / median % per leg:** 1.08% / -0.20%
- **Sum % (uncompounded):** 82.01%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 30 | 11 | 36.7% | 3 | 27 | 0 | 0.50% | 14.9% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 30 | 11 | 36.7% | 3 | 27 | 0 | 0.50% | 14.9% |
| SELL (all) | 46 | 25 | 54.3% | 0 | 38 | 8 | 1.46% | 67.2% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 46 | 25 | 54.3% | 0 | 38 | 8 | 1.46% | 67.2% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 76 | 36 | 47.4% | 3 | 65 | 8 | 1.08% | 82.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 09:15:00 | 987.75 | 961.46 | 959.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 10:15:00 | 996.00 | 968.37 | 963.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-14 12:15:00 | 1009.50 | 1011.98 | 1000.90 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-14 12:45:00 | 1008.60 | 1011.98 | 1000.90 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 14:15:00 | 1035.00 | 1037.22 | 1031.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-19 14:45:00 | 1034.20 | 1037.22 | 1031.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 09:15:00 | 1023.15 | 1033.91 | 1030.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 09:30:00 | 1024.40 | 1033.91 | 1030.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 10:15:00 | 1025.00 | 1032.13 | 1030.17 | EMA400 retest candle locked (from upside) |

### Cycle 2 — SELL (started 2025-05-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-20 12:15:00 | 1021.75 | 1028.73 | 1028.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-20 13:15:00 | 1013.00 | 1025.58 | 1027.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-21 14:15:00 | 1015.20 | 1014.40 | 1019.32 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-21 15:00:00 | 1015.20 | 1014.40 | 1019.32 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 09:15:00 | 1016.95 | 1014.68 | 1018.59 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-22 13:45:00 | 1008.85 | 1013.43 | 1016.80 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-23 11:15:00 | 1027.05 | 1017.92 | 1017.73 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 3 — BUY (started 2025-05-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-23 11:15:00 | 1027.05 | 1017.92 | 1017.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-26 09:15:00 | 1048.95 | 1026.97 | 1022.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-27 09:15:00 | 1039.25 | 1042.18 | 1034.15 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-27 10:00:00 | 1039.25 | 1042.18 | 1034.15 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 13:15:00 | 1039.00 | 1043.83 | 1037.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-27 13:30:00 | 1037.30 | 1043.83 | 1037.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 14:15:00 | 1047.00 | 1044.47 | 1038.56 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-28 10:45:00 | 1049.65 | 1045.88 | 1040.76 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-28 11:30:00 | 1056.50 | 1048.41 | 1042.37 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-30 12:15:00 | 1037.90 | 1052.43 | 1053.51 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 4 — SELL (started 2025-05-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-30 12:15:00 | 1037.90 | 1052.43 | 1053.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-30 14:15:00 | 1032.80 | 1045.84 | 1050.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-04 11:15:00 | 1009.20 | 1008.48 | 1017.05 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-04 11:30:00 | 1010.80 | 1008.48 | 1017.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 09:15:00 | 1001.50 | 1001.67 | 1006.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-06 09:45:00 | 1005.40 | 1001.67 | 1006.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-09 09:15:00 | 1005.60 | 1001.85 | 1004.02 | EMA400 retest candle locked (from downside) |

### Cycle 5 — BUY (started 2025-06-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-09 12:15:00 | 1013.00 | 1005.23 | 1005.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-12 10:15:00 | 1022.70 | 1016.03 | 1013.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-12 12:15:00 | 1014.20 | 1015.68 | 1013.63 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-12 12:15:00 | 1014.20 | 1015.68 | 1013.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 12:15:00 | 1014.20 | 1015.68 | 1013.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-12 13:15:00 | 1011.90 | 1015.68 | 1013.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 13:15:00 | 1003.00 | 1013.14 | 1012.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-12 14:00:00 | 1003.00 | 1013.14 | 1012.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 6 — SELL (started 2025-06-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-12 14:15:00 | 1007.50 | 1012.01 | 1012.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-13 10:15:00 | 998.30 | 1006.90 | 1009.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-13 15:15:00 | 1006.80 | 1004.17 | 1006.89 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-13 15:15:00 | 1006.80 | 1004.17 | 1006.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 15:15:00 | 1006.80 | 1004.17 | 1006.89 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-16 09:15:00 | 993.00 | 1004.17 | 1006.89 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-16 11:15:00 | 1000.50 | 1002.29 | 1005.46 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-16 11:45:00 | 1001.00 | 1002.15 | 1005.11 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-16 12:15:00 | 1001.20 | 1002.15 | 1005.11 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 14:15:00 | 1004.60 | 1002.31 | 1004.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 15:00:00 | 1004.60 | 1002.31 | 1004.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 15:15:00 | 1004.40 | 1002.73 | 1004.42 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-17 09:15:00 | 998.00 | 1002.73 | 1004.42 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-27 10:15:00 | 977.20 | 973.51 | 973.37 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 7 — BUY (started 2025-06-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-27 10:15:00 | 977.20 | 973.51 | 973.37 | EMA200 above EMA400 |

### Cycle 8 — SELL (started 2025-06-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-27 13:15:00 | 970.20 | 973.31 | 973.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-27 14:15:00 | 961.20 | 970.89 | 972.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-30 11:15:00 | 974.90 | 970.14 | 971.29 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-30 11:15:00 | 974.90 | 970.14 | 971.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 11:15:00 | 974.90 | 970.14 | 971.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-30 12:00:00 | 974.90 | 970.14 | 971.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 12:15:00 | 967.20 | 969.55 | 970.92 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-30 13:15:00 | 966.00 | 969.55 | 970.92 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-30 13:45:00 | 966.40 | 969.40 | 970.72 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-30 15:00:00 | 963.40 | 968.20 | 970.06 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-01 09:45:00 | 965.80 | 966.45 | 968.90 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 09:15:00 | 960.60 | 962.47 | 965.29 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-02 14:15:00 | 959.90 | 961.98 | 964.13 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-03 11:30:00 | 959.05 | 958.90 | 961.52 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-03 12:00:00 | 958.65 | 958.90 | 961.52 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-03 14:30:00 | 960.00 | 959.76 | 961.34 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-04 09:15:00 | 975.70 | 962.84 | 962.46 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 9 — BUY (started 2025-07-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-04 09:15:00 | 975.70 | 962.84 | 962.46 | EMA200 above EMA400 |

### Cycle 10 — SELL (started 2025-07-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-08 09:15:00 | 960.20 | 963.74 | 964.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-08 10:15:00 | 957.05 | 962.40 | 963.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-08 14:15:00 | 966.50 | 961.46 | 962.49 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-08 14:15:00 | 966.50 | 961.46 | 962.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 14:15:00 | 966.50 | 961.46 | 962.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-08 15:00:00 | 966.50 | 961.46 | 962.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 15:15:00 | 964.00 | 961.96 | 962.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-09 09:15:00 | 973.85 | 961.96 | 962.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 11 — BUY (started 2025-07-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-09 09:15:00 | 973.05 | 964.18 | 963.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-09 13:15:00 | 977.65 | 971.37 | 967.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-11 09:15:00 | 977.85 | 981.14 | 976.84 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-11 10:00:00 | 977.85 | 981.14 | 976.84 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 10:15:00 | 981.30 | 981.17 | 977.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-11 10:30:00 | 977.25 | 981.17 | 977.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 12:15:00 | 975.80 | 980.01 | 977.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-11 13:15:00 | 972.30 | 980.01 | 977.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 13:15:00 | 978.45 | 979.70 | 977.49 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-11 15:00:00 | 978.80 | 979.52 | 977.61 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-14 09:45:00 | 980.35 | 979.75 | 978.05 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-16 13:15:00 | 982.00 | 982.93 | 983.02 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 12 — SELL (started 2025-07-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-16 13:15:00 | 982.00 | 982.93 | 983.02 | EMA200 below EMA400 |

### Cycle 13 — BUY (started 2025-07-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-16 14:15:00 | 987.60 | 983.86 | 983.44 | EMA200 above EMA400 |

### Cycle 14 — SELL (started 2025-07-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-18 09:15:00 | 978.00 | 983.10 | 983.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-18 12:15:00 | 977.45 | 980.73 | 982.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-21 09:15:00 | 985.50 | 980.20 | 981.37 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-21 09:15:00 | 985.50 | 980.20 | 981.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 09:15:00 | 985.50 | 980.20 | 981.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-21 10:00:00 | 985.50 | 980.20 | 981.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 15 — BUY (started 2025-07-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-21 10:15:00 | 990.10 | 982.18 | 982.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-21 11:15:00 | 1008.00 | 987.35 | 984.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-22 11:15:00 | 998.00 | 999.65 | 993.88 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-22 11:30:00 | 997.30 | 999.65 | 993.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 13:15:00 | 1000.50 | 999.06 | 994.56 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-23 10:30:00 | 1002.20 | 999.91 | 996.39 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-25 11:15:00 | 995.10 | 1002.86 | 1003.51 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 16 — SELL (started 2025-07-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-25 11:15:00 | 995.10 | 1002.86 | 1003.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-25 13:15:00 | 993.45 | 999.84 | 1001.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-29 10:15:00 | 981.75 | 980.68 | 987.97 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-29 11:00:00 | 981.75 | 980.68 | 987.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 11:15:00 | 983.55 | 981.26 | 987.57 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-30 15:15:00 | 978.70 | 984.88 | 987.05 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-06 14:15:00 | 929.76 | 939.42 | 948.21 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-07 13:15:00 | 934.15 | 931.13 | 939.34 | SL hit (close>ema200) qty=0.50 sl=931.13 alert=retest2 |

### Cycle 17 — BUY (started 2025-08-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-11 11:15:00 | 949.00 | 939.25 | 938.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-11 12:15:00 | 950.70 | 941.54 | 939.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-13 09:15:00 | 954.00 | 954.09 | 949.61 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-13 09:15:00 | 954.00 | 954.09 | 949.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 09:15:00 | 954.00 | 954.09 | 949.61 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-13 13:45:00 | 959.20 | 954.95 | 951.43 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-13 15:00:00 | 958.95 | 955.75 | 952.12 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-18 09:15:00 | 947.65 | 951.69 | 951.70 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 18 — SELL (started 2025-08-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-18 09:15:00 | 947.65 | 951.69 | 951.70 | EMA200 below EMA400 |

### Cycle 19 — BUY (started 2025-08-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-18 10:15:00 | 957.65 | 952.89 | 952.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-18 11:15:00 | 960.10 | 954.33 | 952.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-18 14:15:00 | 955.80 | 956.46 | 954.45 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-18 15:00:00 | 955.80 | 956.46 | 954.45 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 15:15:00 | 955.00 | 956.17 | 954.50 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-19 09:15:00 | 961.45 | 956.17 | 954.50 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-26 09:15:00 | 976.50 | 985.84 | 986.73 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 20 — SELL (started 2025-08-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-26 09:15:00 | 976.50 | 985.84 | 986.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-26 14:15:00 | 971.75 | 980.95 | 983.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-29 09:15:00 | 970.85 | 970.55 | 975.39 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-29 09:45:00 | 972.10 | 970.55 | 975.39 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 10:15:00 | 982.20 | 972.88 | 976.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-29 11:00:00 | 982.20 | 972.88 | 976.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 11:15:00 | 988.20 | 975.94 | 977.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-29 11:30:00 | 987.00 | 975.94 | 977.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 21 — BUY (started 2025-08-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-29 13:15:00 | 990.45 | 980.29 | 978.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-01 09:15:00 | 1010.95 | 989.23 | 983.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-02 13:15:00 | 1017.05 | 1018.27 | 1006.79 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-02 14:00:00 | 1017.05 | 1018.27 | 1006.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 09:15:00 | 1009.50 | 1016.59 | 1008.90 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-03 11:45:00 | 1020.00 | 1015.03 | 1009.41 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-03 14:30:00 | 1020.80 | 1017.03 | 1011.78 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-04 14:15:00 | 1000.50 | 1012.43 | 1012.23 | SL hit (close<static) qty=1.00 sl=1006.20 alert=retest2 |

### Cycle 22 — SELL (started 2025-09-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-04 15:15:00 | 1009.00 | 1011.74 | 1011.94 | EMA200 below EMA400 |

### Cycle 23 — BUY (started 2025-09-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-05 09:15:00 | 1021.50 | 1013.69 | 1012.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-08 12:15:00 | 1036.00 | 1024.38 | 1019.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-09 09:15:00 | 1026.75 | 1028.37 | 1023.14 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-09 10:00:00 | 1026.75 | 1028.37 | 1023.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 14:15:00 | 1023.15 | 1026.81 | 1024.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-09 15:00:00 | 1023.15 | 1026.81 | 1024.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 15:15:00 | 1022.50 | 1025.95 | 1024.22 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-10 09:15:00 | 1028.05 | 1025.95 | 1024.22 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-15 15:15:00 | 1034.00 | 1036.40 | 1036.65 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 24 — SELL (started 2025-09-15 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-15 15:15:00 | 1034.00 | 1036.40 | 1036.65 | EMA200 below EMA400 |

### Cycle 25 — BUY (started 2025-09-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-16 09:15:00 | 1048.00 | 1038.72 | 1037.68 | EMA200 above EMA400 |

### Cycle 26 — SELL (started 2025-09-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-17 10:15:00 | 1034.40 | 1037.34 | 1037.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-17 15:15:00 | 1029.00 | 1034.61 | 1036.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-18 12:15:00 | 1035.00 | 1032.17 | 1034.17 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-18 12:15:00 | 1035.00 | 1032.17 | 1034.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 12:15:00 | 1035.00 | 1032.17 | 1034.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-18 13:00:00 | 1035.00 | 1032.17 | 1034.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 13:15:00 | 1032.80 | 1032.30 | 1034.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-18 13:45:00 | 1035.05 | 1032.30 | 1034.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 09:15:00 | 1029.70 | 1031.65 | 1033.30 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-19 13:45:00 | 1027.35 | 1031.93 | 1033.09 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-19 15:00:00 | 1021.10 | 1029.76 | 1032.00 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-26 09:15:00 | 975.98 | 985.62 | 995.21 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-26 09:15:00 | 970.04 | 985.62 | 995.21 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-26 15:15:00 | 979.00 | 978.60 | 986.69 | SL hit (close>ema200) qty=0.50 sl=978.60 alert=retest2 |

### Cycle 27 — BUY (started 2025-09-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-30 15:15:00 | 991.05 | 987.50 | 987.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-01 14:15:00 | 994.20 | 989.63 | 988.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-06 10:15:00 | 993.00 | 995.64 | 993.26 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-06 10:15:00 | 993.00 | 995.64 | 993.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 10:15:00 | 993.00 | 995.64 | 993.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-06 11:00:00 | 993.00 | 995.64 | 993.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 11:15:00 | 993.20 | 995.15 | 993.25 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-06 12:30:00 | 996.15 | 997.49 | 994.49 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-09 11:15:00 | 994.20 | 1000.67 | 1001.29 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 28 — SELL (started 2025-10-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-09 11:15:00 | 994.20 | 1000.67 | 1001.29 | EMA200 below EMA400 |

### Cycle 29 — BUY (started 2025-10-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-10 14:15:00 | 1005.95 | 1001.06 | 1000.65 | EMA200 above EMA400 |

### Cycle 30 — SELL (started 2025-10-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-13 09:15:00 | 993.30 | 1000.35 | 1000.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-13 10:15:00 | 992.15 | 998.71 | 999.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-13 14:15:00 | 998.00 | 995.93 | 997.80 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-13 14:15:00 | 998.00 | 995.93 | 997.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 14:15:00 | 998.00 | 995.93 | 997.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-13 15:00:00 | 998.00 | 995.93 | 997.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 15:15:00 | 998.15 | 996.37 | 997.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-14 09:15:00 | 1000.50 | 996.37 | 997.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 09:15:00 | 991.95 | 995.49 | 997.29 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-14 10:30:00 | 991.25 | 993.71 | 996.32 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-15 10:15:00 | 1000.60 | 997.26 | 996.88 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 31 — BUY (started 2025-10-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-15 10:15:00 | 1000.60 | 997.26 | 996.88 | EMA200 above EMA400 |

### Cycle 32 — SELL (started 2025-10-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-15 14:15:00 | 994.60 | 996.42 | 996.61 | EMA200 below EMA400 |

### Cycle 33 — BUY (started 2025-10-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-16 15:15:00 | 997.60 | 996.48 | 996.40 | EMA200 above EMA400 |

### Cycle 34 — SELL (started 2025-10-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-17 09:15:00 | 993.80 | 995.94 | 996.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-17 10:15:00 | 991.40 | 995.03 | 995.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-17 12:15:00 | 995.25 | 994.56 | 995.36 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-17 12:15:00 | 995.25 | 994.56 | 995.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 12:15:00 | 995.25 | 994.56 | 995.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-17 12:45:00 | 994.95 | 994.56 | 995.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 13:15:00 | 995.95 | 994.84 | 995.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-17 14:00:00 | 995.95 | 994.84 | 995.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 14:15:00 | 995.00 | 994.87 | 995.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-17 15:00:00 | 995.00 | 994.87 | 995.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 15:15:00 | 993.20 | 994.54 | 995.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-20 09:15:00 | 992.55 | 994.54 | 995.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 09:15:00 | 988.60 | 993.35 | 994.58 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-20 10:15:00 | 986.60 | 993.35 | 994.58 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-20 13:15:00 | 987.05 | 991.14 | 993.16 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-20 14:45:00 | 987.15 | 989.74 | 992.14 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-23 10:15:00 | 996.85 | 993.79 | 993.37 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 35 — BUY (started 2025-10-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-23 10:15:00 | 996.85 | 993.79 | 993.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-23 15:15:00 | 1000.00 | 996.11 | 994.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-24 09:15:00 | 994.70 | 995.83 | 994.73 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-24 09:15:00 | 994.70 | 995.83 | 994.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 09:15:00 | 994.70 | 995.83 | 994.73 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-24 12:45:00 | 997.95 | 995.74 | 994.91 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-24 13:15:00 | 997.20 | 995.74 | 994.91 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-24 14:00:00 | 997.50 | 996.09 | 995.15 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-27 09:15:00 | 1000.05 | 996.31 | 995.41 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 09:15:00 | 1000.60 | 997.16 | 995.88 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-28 09:15:00 | 1009.00 | 999.32 | 997.59 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-28 12:30:00 | 1005.90 | 1001.48 | 999.23 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-29 09:15:00 | 1007.35 | 1002.70 | 1000.45 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-29 10:00:00 | 1008.65 | 1003.89 | 1001.19 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 09:15:00 | 1009.80 | 1012.39 | 1009.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-31 10:00:00 | 1009.80 | 1012.39 | 1009.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 10:15:00 | 1002.45 | 1010.40 | 1008.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-31 11:00:00 | 1002.45 | 1010.40 | 1008.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 11:15:00 | 1003.55 | 1009.03 | 1008.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-31 11:30:00 | 1001.15 | 1009.03 | 1008.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 14:15:00 | 1002.80 | 1009.22 | 1008.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-31 15:00:00 | 1002.80 | 1009.22 | 1008.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 15:15:00 | 1006.00 | 1008.57 | 1008.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-03 09:15:00 | 1004.60 | 1008.57 | 1008.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 10:15:00 | 1007.20 | 1008.38 | 1008.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-03 10:45:00 | 1007.60 | 1008.38 | 1008.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-11-03 11:15:00 | 1003.00 | 1007.31 | 1007.89 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 36 — SELL (started 2025-11-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-03 11:15:00 | 1003.00 | 1007.31 | 1007.89 | EMA200 below EMA400 |

### Cycle 37 — BUY (started 2025-11-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-03 15:15:00 | 1012.00 | 1008.42 | 1008.17 | EMA200 above EMA400 |

### Cycle 38 — SELL (started 2025-11-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-04 09:15:00 | 1001.20 | 1006.98 | 1007.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-04 10:15:00 | 1000.40 | 1005.66 | 1006.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-06 12:15:00 | 997.80 | 991.89 | 997.10 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-06 12:15:00 | 997.80 | 991.89 | 997.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 12:15:00 | 997.80 | 991.89 | 997.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-06 13:00:00 | 997.80 | 991.89 | 997.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 13:15:00 | 992.20 | 991.95 | 996.65 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-06 14:15:00 | 989.90 | 991.95 | 996.65 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-06 15:00:00 | 986.20 | 990.80 | 995.70 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-20 10:15:00 | 965.40 | 960.41 | 960.14 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 39 — BUY (started 2025-11-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-20 10:15:00 | 965.40 | 960.41 | 960.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-20 13:15:00 | 968.30 | 963.12 | 961.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-21 09:15:00 | 956.80 | 963.43 | 962.22 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-21 09:15:00 | 956.80 | 963.43 | 962.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 09:15:00 | 956.80 | 963.43 | 962.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-21 10:00:00 | 956.80 | 963.43 | 962.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 10:15:00 | 955.80 | 961.91 | 961.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-21 10:30:00 | 954.80 | 961.91 | 961.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 40 — SELL (started 2025-11-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-21 11:15:00 | 953.50 | 960.23 | 960.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-21 14:15:00 | 952.40 | 957.48 | 959.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-24 14:15:00 | 954.90 | 950.04 | 953.80 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-24 14:15:00 | 954.90 | 950.04 | 953.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 14:15:00 | 954.90 | 950.04 | 953.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-24 15:00:00 | 954.90 | 950.04 | 953.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 15:15:00 | 956.70 | 951.37 | 954.07 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-25 09:15:00 | 944.00 | 951.37 | 954.07 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-27 10:00:00 | 952.30 | 949.81 | 949.96 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-27 10:15:00 | 952.00 | 950.25 | 950.15 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 41 — BUY (started 2025-11-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-27 10:15:00 | 952.00 | 950.25 | 950.15 | EMA200 above EMA400 |

### Cycle 42 — SELL (started 2025-11-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-27 12:15:00 | 943.30 | 949.09 | 949.65 | EMA200 below EMA400 |

### Cycle 43 — BUY (started 2025-11-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-28 15:15:00 | 952.00 | 949.14 | 948.76 | EMA200 above EMA400 |

### Cycle 44 — SELL (started 2025-12-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-03 12:15:00 | 946.95 | 949.90 | 950.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-04 10:15:00 | 945.00 | 947.50 | 948.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-05 14:15:00 | 946.00 | 941.71 | 943.76 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-05 14:15:00 | 946.00 | 941.71 | 943.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 14:15:00 | 946.00 | 941.71 | 943.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-05 15:00:00 | 946.00 | 941.71 | 943.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 15:15:00 | 945.95 | 942.56 | 943.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-08 09:15:00 | 940.65 | 942.56 | 943.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 12:15:00 | 934.60 | 931.22 | 934.96 | EMA400 retest candle locked (from downside) |

### Cycle 45 — BUY (started 2025-12-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-10 09:15:00 | 942.80 | 936.67 | 936.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-10 12:15:00 | 950.40 | 940.03 | 938.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-11 09:15:00 | 935.00 | 941.68 | 939.89 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-11 09:15:00 | 935.00 | 941.68 | 939.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 09:15:00 | 935.00 | 941.68 | 939.89 | EMA400 retest candle locked (from upside) |

### Cycle 46 — SELL (started 2025-12-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-11 12:15:00 | 934.00 | 938.30 | 938.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-11 15:15:00 | 930.05 | 935.78 | 937.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-15 11:15:00 | 935.00 | 931.00 | 933.20 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-15 11:15:00 | 935.00 | 931.00 | 933.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 11:15:00 | 935.00 | 931.00 | 933.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-15 11:45:00 | 935.15 | 931.00 | 933.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 12:15:00 | 935.35 | 931.87 | 933.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-15 12:45:00 | 935.40 | 931.87 | 933.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 10:15:00 | 926.75 | 932.14 | 933.18 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-16 11:15:00 | 926.00 | 932.14 | 933.18 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-16 11:45:00 | 926.00 | 930.98 | 932.56 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-16 12:15:00 | 926.00 | 930.98 | 932.56 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-18 09:15:00 | 920.45 | 925.32 | 927.28 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 14:15:00 | 925.40 | 921.63 | 923.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-18 15:00:00 | 925.40 | 921.63 | 923.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 15:15:00 | 925.50 | 922.41 | 924.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 09:15:00 | 927.00 | 922.41 | 924.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 10:15:00 | 924.30 | 923.53 | 924.36 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-19 11:15:00 | 923.05 | 923.53 | 924.36 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-19 12:00:00 | 922.00 | 923.22 | 924.14 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-22 09:15:00 | 930.80 | 925.44 | 924.80 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 47 — BUY (started 2025-12-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-22 09:15:00 | 930.80 | 925.44 | 924.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-23 11:15:00 | 934.40 | 929.55 | 927.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-23 14:15:00 | 930.80 | 930.89 | 928.75 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-23 15:00:00 | 930.80 | 930.89 | 928.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 10:15:00 | 931.50 | 931.39 | 929.55 | EMA400 retest candle locked (from upside) |

### Cycle 48 — SELL (started 2025-12-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-24 14:15:00 | 925.75 | 928.81 | 928.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-24 15:15:00 | 924.40 | 927.93 | 928.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-31 09:15:00 | 902.80 | 900.77 | 907.42 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-31 10:15:00 | 906.65 | 901.94 | 907.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 10:15:00 | 906.65 | 901.94 | 907.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 11:00:00 | 906.65 | 901.94 | 907.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 12:15:00 | 909.00 | 904.05 | 907.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 12:30:00 | 909.00 | 904.05 | 907.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 13:15:00 | 909.20 | 905.08 | 907.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 14:00:00 | 909.20 | 905.08 | 907.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 14:15:00 | 909.30 | 905.93 | 907.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 15:00:00 | 909.30 | 905.93 | 907.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 11:15:00 | 906.55 | 906.68 | 907.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-01 12:00:00 | 906.55 | 906.68 | 907.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 12:15:00 | 906.10 | 906.56 | 907.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-01 12:45:00 | 907.15 | 906.56 | 907.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 49 — BUY (started 2026-01-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-01 13:15:00 | 928.00 | 910.85 | 909.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-02 10:15:00 | 932.75 | 921.13 | 915.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-05 13:15:00 | 926.85 | 932.69 | 926.23 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-05 14:00:00 | 926.85 | 932.69 | 926.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 14:15:00 | 924.30 | 931.01 | 926.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-05 15:00:00 | 924.30 | 931.01 | 926.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 15:15:00 | 930.80 | 930.97 | 926.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-06 09:15:00 | 924.20 | 930.97 | 926.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 09:15:00 | 927.00 | 930.17 | 926.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-06 09:30:00 | 925.90 | 930.17 | 926.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 10:15:00 | 925.30 | 929.20 | 926.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-06 10:45:00 | 926.15 | 929.20 | 926.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 11:15:00 | 921.65 | 927.69 | 925.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-06 12:00:00 | 921.65 | 927.69 | 925.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 12:15:00 | 924.75 | 927.10 | 925.87 | EMA400 retest candle locked (from upside) |

### Cycle 50 — SELL (started 2026-01-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-06 14:15:00 | 919.50 | 924.83 | 925.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-07 10:15:00 | 915.95 | 921.77 | 923.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-07 14:15:00 | 919.95 | 919.19 | 921.45 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-07 14:45:00 | 919.75 | 919.19 | 921.45 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 15:15:00 | 921.95 | 919.74 | 921.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-08 09:15:00 | 920.80 | 919.74 | 921.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 09:15:00 | 926.80 | 921.15 | 921.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-08 09:45:00 | 927.15 | 921.15 | 921.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 10:15:00 | 921.00 | 921.12 | 921.89 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-08 11:15:00 | 919.65 | 921.12 | 921.89 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-08 11:45:00 | 919.25 | 920.50 | 921.54 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-08 14:45:00 | 919.25 | 918.59 | 920.35 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-14 10:15:00 | 873.67 | 881.16 | 889.36 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-14 10:15:00 | 873.29 | 881.16 | 889.36 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-14 10:15:00 | 873.29 | 881.16 | 889.36 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-16 09:15:00 | 879.35 | 876.06 | 882.45 | SL hit (close>ema200) qty=0.50 sl=876.06 alert=retest2 |

### Cycle 51 — BUY (started 2026-01-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-30 11:15:00 | 844.60 | 835.19 | 834.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-01 09:15:00 | 851.45 | 840.41 | 837.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-01 11:15:00 | 837.90 | 841.28 | 838.26 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-01 11:15:00 | 837.90 | 841.28 | 838.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 11:15:00 | 837.90 | 841.28 | 838.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 12:00:00 | 837.90 | 841.28 | 838.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 12:15:00 | 837.65 | 840.55 | 838.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 12:30:00 | 841.10 | 840.55 | 838.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 13:15:00 | 828.65 | 838.17 | 837.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 14:00:00 | 828.65 | 838.17 | 837.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 52 — SELL (started 2026-02-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-01 14:15:00 | 829.00 | 836.34 | 836.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-01 15:15:00 | 826.95 | 834.46 | 835.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-03 09:15:00 | 832.70 | 822.48 | 826.65 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-03 09:15:00 | 832.70 | 822.48 | 826.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 09:15:00 | 832.70 | 822.48 | 826.65 | EMA400 retest candle locked (from downside) |

### Cycle 53 — BUY (started 2026-02-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 11:15:00 | 847.65 | 830.79 | 829.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-04 09:15:00 | 857.65 | 840.23 | 835.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-06 09:15:00 | 862.70 | 871.56 | 863.71 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-06 09:15:00 | 862.70 | 871.56 | 863.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 09:15:00 | 862.70 | 871.56 | 863.71 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-09 09:15:00 | 877.05 | 866.07 | 863.75 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-12 11:15:00 | 867.90 | 889.84 | 892.38 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 54 — SELL (started 2026-02-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-12 11:15:00 | 867.90 | 889.84 | 892.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-12 12:15:00 | 864.85 | 884.84 | 889.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-17 09:15:00 | 843.50 | 839.84 | 849.50 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-17 10:00:00 | 843.50 | 839.84 | 849.50 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 09:15:00 | 839.80 | 841.84 | 846.01 | EMA400 retest candle locked (from downside) |

### Cycle 55 — BUY (started 2026-02-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-18 15:15:00 | 854.95 | 848.65 | 847.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-20 11:15:00 | 857.25 | 850.95 | 849.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-20 15:15:00 | 853.00 | 854.01 | 851.86 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-20 15:15:00 | 853.00 | 854.01 | 851.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 15:15:00 | 853.00 | 854.01 | 851.86 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-23 09:15:00 | 861.00 | 854.01 | 851.86 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-24 10:15:00 | 856.45 | 858.98 | 856.80 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-24 13:00:00 | 856.55 | 858.29 | 857.01 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-24 13:45:00 | 856.95 | 858.08 | 857.03 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 14:15:00 | 859.00 | 858.27 | 857.21 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-02-25 12:15:00 | 849.20 | 855.80 | 856.40 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 56 — SELL (started 2026-02-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-25 12:15:00 | 849.20 | 855.80 | 856.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-27 09:15:00 | 844.75 | 852.13 | 853.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-27 12:15:00 | 853.15 | 851.16 | 852.69 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-27 12:15:00 | 853.15 | 851.16 | 852.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 12:15:00 | 853.15 | 851.16 | 852.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-27 13:00:00 | 853.15 | 851.16 | 852.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 13:15:00 | 855.85 | 852.10 | 852.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-27 14:00:00 | 855.85 | 852.10 | 852.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 14:15:00 | 842.80 | 850.24 | 852.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-27 14:45:00 | 858.50 | 850.24 | 852.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 14:15:00 | 807.80 | 806.48 | 810.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-06 14:30:00 | 811.00 | 806.48 | 810.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 15:15:00 | 790.00 | 788.78 | 793.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-11 09:15:00 | 800.45 | 788.78 | 793.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 09:15:00 | 802.00 | 791.42 | 794.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-11 09:45:00 | 802.50 | 791.42 | 794.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 10:15:00 | 799.35 | 793.01 | 794.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-11 10:30:00 | 799.00 | 793.01 | 794.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 13:15:00 | 793.35 | 794.23 | 795.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-11 13:30:00 | 795.75 | 794.23 | 795.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 14:15:00 | 792.95 | 793.98 | 794.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-11 14:30:00 | 793.70 | 793.98 | 794.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 10:15:00 | 802.15 | 794.31 | 794.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-12 11:00:00 | 802.15 | 794.31 | 794.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 11:15:00 | 794.25 | 794.30 | 794.71 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-12 14:45:00 | 790.90 | 793.49 | 794.27 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-13 09:30:00 | 784.70 | 790.19 | 792.59 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-18 09:15:00 | 785.25 | 778.50 | 778.41 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 57 — BUY (started 2026-03-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 09:15:00 | 785.25 | 778.50 | 778.41 | EMA200 above EMA400 |

### Cycle 58 — SELL (started 2026-03-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 12:15:00 | 775.60 | 779.20 | 779.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 13:15:00 | 773.25 | 778.01 | 779.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-25 09:15:00 | 743.75 | 734.47 | 742.39 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-25 09:15:00 | 743.75 | 734.47 | 742.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 09:15:00 | 743.75 | 734.47 | 742.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-25 10:00:00 | 743.75 | 734.47 | 742.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 10:15:00 | 740.60 | 735.69 | 742.23 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-25 11:30:00 | 733.25 | 735.55 | 741.57 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-25 12:15:00 | 734.15 | 735.55 | 741.57 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-30 09:15:00 | 696.59 | 703.88 | 717.54 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-30 09:15:00 | 697.44 | 703.88 | 717.54 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-01 09:15:00 | 708.70 | 688.98 | 700.90 | SL hit (close>ema200) qty=0.50 sl=688.98 alert=retest2 |

### Cycle 59 — BUY (started 2026-04-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 14:15:00 | 712.05 | 706.29 | 706.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-01 15:15:00 | 717.20 | 708.47 | 707.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-02 09:15:00 | 698.00 | 706.38 | 706.32 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-02 09:15:00 | 698.00 | 706.38 | 706.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 09:15:00 | 698.00 | 706.38 | 706.32 | EMA400 retest candle locked (from upside) |

### Cycle 60 — SELL (started 2026-04-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-02 10:15:00 | 702.75 | 705.65 | 706.00 | EMA200 below EMA400 |

### Cycle 61 — BUY (started 2026-04-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-02 11:15:00 | 710.10 | 706.54 | 706.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-02 12:15:00 | 716.85 | 708.60 | 707.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-06 09:15:00 | 710.00 | 715.62 | 711.76 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-06 09:15:00 | 710.00 | 715.62 | 711.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-06 09:15:00 | 710.00 | 715.62 | 711.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-06 10:00:00 | 710.00 | 715.62 | 711.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-06 10:15:00 | 707.75 | 714.04 | 711.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-06 11:00:00 | 707.75 | 714.04 | 711.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-07 09:15:00 | 717.75 | 717.47 | 714.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-07 09:45:00 | 717.05 | 717.47 | 714.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 15:15:00 | 740.00 | 743.42 | 738.68 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-10 09:15:00 | 748.00 | 743.42 | 738.68 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-10 15:00:00 | 742.55 | 745.32 | 742.10 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-13 09:15:00 | 735.00 | 742.80 | 741.49 | SL hit (close<static) qty=1.00 sl=738.00 alert=retest2 |

### Cycle 62 — SELL (started 2026-04-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-30 10:15:00 | 879.30 | 886.75 | 887.01 | EMA200 below EMA400 |

### Cycle 63 — BUY (started 2026-05-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-04 15:15:00 | 889.00 | 883.48 | 883.36 | EMA200 above EMA400 |

### Cycle 64 — SELL (started 2026-05-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-05 10:15:00 | 881.55 | 883.02 | 883.17 | EMA200 below EMA400 |

### Cycle 65 — BUY (started 2026-05-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-05 15:15:00 | 887.90 | 883.57 | 883.26 | EMA200 above EMA400 |

### Cycle 66 — SELL (started 2026-05-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-06 12:15:00 | 874.00 | 881.45 | 882.39 | EMA200 below EMA400 |

### Cycle 67 — BUY (started 2026-05-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-07 09:15:00 | 890.25 | 882.73 | 882.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-07 10:15:00 | 901.00 | 886.39 | 884.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-07 14:15:00 | 894.25 | 894.64 | 889.48 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-05-07 15:00:00 | 894.25 | 894.64 | 889.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 10:15:00 | 895.05 | 895.60 | 891.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-08 11:00:00 | 895.05 | 895.60 | 891.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 11:15:00 | 888.30 | 894.14 | 891.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-08 11:45:00 | 887.10 | 894.14 | 891.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 12:15:00 | 886.85 | 892.68 | 890.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-08 12:45:00 | 886.75 | 892.68 | 890.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 15:15:00 | 890.85 | 891.15 | 890.35 | EMA400 retest candle locked (from upside) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-05-22 13:45:00 | 1008.85 | 2025-05-23 11:15:00 | 1027.05 | STOP_HIT | 1.00 | -1.80% |
| BUY | retest2 | 2025-05-28 10:45:00 | 1049.65 | 2025-05-30 12:15:00 | 1037.90 | STOP_HIT | 1.00 | -1.12% |
| BUY | retest2 | 2025-05-28 11:30:00 | 1056.50 | 2025-05-30 12:15:00 | 1037.90 | STOP_HIT | 1.00 | -1.76% |
| SELL | retest2 | 2025-06-16 09:15:00 | 993.00 | 2025-06-27 10:15:00 | 977.20 | STOP_HIT | 1.00 | 1.59% |
| SELL | retest2 | 2025-06-16 11:15:00 | 1000.50 | 2025-06-27 10:15:00 | 977.20 | STOP_HIT | 1.00 | 2.33% |
| SELL | retest2 | 2025-06-16 11:45:00 | 1001.00 | 2025-06-27 10:15:00 | 977.20 | STOP_HIT | 1.00 | 2.38% |
| SELL | retest2 | 2025-06-16 12:15:00 | 1001.20 | 2025-06-27 10:15:00 | 977.20 | STOP_HIT | 1.00 | 2.40% |
| SELL | retest2 | 2025-06-17 09:15:00 | 998.00 | 2025-06-27 10:15:00 | 977.20 | STOP_HIT | 1.00 | 2.08% |
| SELL | retest2 | 2025-06-30 13:15:00 | 966.00 | 2025-07-04 09:15:00 | 975.70 | STOP_HIT | 1.00 | -1.00% |
| SELL | retest2 | 2025-06-30 13:45:00 | 966.40 | 2025-07-04 09:15:00 | 975.70 | STOP_HIT | 1.00 | -0.96% |
| SELL | retest2 | 2025-06-30 15:00:00 | 963.40 | 2025-07-04 09:15:00 | 975.70 | STOP_HIT | 1.00 | -1.28% |
| SELL | retest2 | 2025-07-01 09:45:00 | 965.80 | 2025-07-04 09:15:00 | 975.70 | STOP_HIT | 1.00 | -1.03% |
| SELL | retest2 | 2025-07-02 14:15:00 | 959.90 | 2025-07-04 09:15:00 | 975.70 | STOP_HIT | 1.00 | -1.65% |
| SELL | retest2 | 2025-07-03 11:30:00 | 959.05 | 2025-07-04 09:15:00 | 975.70 | STOP_HIT | 1.00 | -1.74% |
| SELL | retest2 | 2025-07-03 12:00:00 | 958.65 | 2025-07-04 09:15:00 | 975.70 | STOP_HIT | 1.00 | -1.78% |
| SELL | retest2 | 2025-07-03 14:30:00 | 960.00 | 2025-07-04 09:15:00 | 975.70 | STOP_HIT | 1.00 | -1.64% |
| BUY | retest2 | 2025-07-11 15:00:00 | 978.80 | 2025-07-16 13:15:00 | 982.00 | STOP_HIT | 1.00 | 0.33% |
| BUY | retest2 | 2025-07-14 09:45:00 | 980.35 | 2025-07-16 13:15:00 | 982.00 | STOP_HIT | 1.00 | 0.17% |
| BUY | retest2 | 2025-07-23 10:30:00 | 1002.20 | 2025-07-25 11:15:00 | 995.10 | STOP_HIT | 1.00 | -0.71% |
| SELL | retest2 | 2025-07-30 15:15:00 | 978.70 | 2025-08-06 14:15:00 | 929.76 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-30 15:15:00 | 978.70 | 2025-08-07 13:15:00 | 934.15 | STOP_HIT | 0.50 | 4.55% |
| BUY | retest2 | 2025-08-13 13:45:00 | 959.20 | 2025-08-18 09:15:00 | 947.65 | STOP_HIT | 1.00 | -1.20% |
| BUY | retest2 | 2025-08-13 15:00:00 | 958.95 | 2025-08-18 09:15:00 | 947.65 | STOP_HIT | 1.00 | -1.18% |
| BUY | retest2 | 2025-08-19 09:15:00 | 961.45 | 2025-08-26 09:15:00 | 976.50 | STOP_HIT | 1.00 | 1.57% |
| BUY | retest2 | 2025-09-03 11:45:00 | 1020.00 | 2025-09-04 14:15:00 | 1000.50 | STOP_HIT | 1.00 | -1.91% |
| BUY | retest2 | 2025-09-03 14:30:00 | 1020.80 | 2025-09-04 14:15:00 | 1000.50 | STOP_HIT | 1.00 | -1.99% |
| BUY | retest2 | 2025-09-10 09:15:00 | 1028.05 | 2025-09-15 15:15:00 | 1034.00 | STOP_HIT | 1.00 | 0.58% |
| SELL | retest2 | 2025-09-19 13:45:00 | 1027.35 | 2025-09-26 09:15:00 | 975.98 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-19 15:00:00 | 1021.10 | 2025-09-26 09:15:00 | 970.04 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-19 13:45:00 | 1027.35 | 2025-09-26 15:15:00 | 979.00 | STOP_HIT | 0.50 | 4.71% |
| SELL | retest2 | 2025-09-19 15:00:00 | 1021.10 | 2025-09-26 15:15:00 | 979.00 | STOP_HIT | 0.50 | 4.12% |
| BUY | retest2 | 2025-10-06 12:30:00 | 996.15 | 2025-10-09 11:15:00 | 994.20 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest2 | 2025-10-14 10:30:00 | 991.25 | 2025-10-15 10:15:00 | 1000.60 | STOP_HIT | 1.00 | -0.94% |
| SELL | retest2 | 2025-10-20 10:15:00 | 986.60 | 2025-10-23 10:15:00 | 996.85 | STOP_HIT | 1.00 | -1.04% |
| SELL | retest2 | 2025-10-20 13:15:00 | 987.05 | 2025-10-23 10:15:00 | 996.85 | STOP_HIT | 1.00 | -0.99% |
| SELL | retest2 | 2025-10-20 14:45:00 | 987.15 | 2025-10-23 10:15:00 | 996.85 | STOP_HIT | 1.00 | -0.98% |
| BUY | retest2 | 2025-10-24 12:45:00 | 997.95 | 2025-11-03 11:15:00 | 1003.00 | STOP_HIT | 1.00 | 0.51% |
| BUY | retest2 | 2025-10-24 13:15:00 | 997.20 | 2025-11-03 11:15:00 | 1003.00 | STOP_HIT | 1.00 | 0.58% |
| BUY | retest2 | 2025-10-24 14:00:00 | 997.50 | 2025-11-03 11:15:00 | 1003.00 | STOP_HIT | 1.00 | 0.55% |
| BUY | retest2 | 2025-10-27 09:15:00 | 1000.05 | 2025-11-03 11:15:00 | 1003.00 | STOP_HIT | 1.00 | 0.29% |
| BUY | retest2 | 2025-10-28 09:15:00 | 1009.00 | 2025-11-03 11:15:00 | 1003.00 | STOP_HIT | 1.00 | -0.59% |
| BUY | retest2 | 2025-10-28 12:30:00 | 1005.90 | 2025-11-03 11:15:00 | 1003.00 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest2 | 2025-10-29 09:15:00 | 1007.35 | 2025-11-03 11:15:00 | 1003.00 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest2 | 2025-10-29 10:00:00 | 1008.65 | 2025-11-03 11:15:00 | 1003.00 | STOP_HIT | 1.00 | -0.56% |
| SELL | retest2 | 2025-11-06 14:15:00 | 989.90 | 2025-11-20 10:15:00 | 965.40 | STOP_HIT | 1.00 | 2.47% |
| SELL | retest2 | 2025-11-06 15:00:00 | 986.20 | 2025-11-20 10:15:00 | 965.40 | STOP_HIT | 1.00 | 2.11% |
| SELL | retest2 | 2025-11-25 09:15:00 | 944.00 | 2025-11-27 10:15:00 | 952.00 | STOP_HIT | 1.00 | -0.85% |
| SELL | retest2 | 2025-11-27 10:00:00 | 952.30 | 2025-11-27 10:15:00 | 952.00 | STOP_HIT | 1.00 | 0.03% |
| SELL | retest2 | 2025-12-16 11:15:00 | 926.00 | 2025-12-22 09:15:00 | 930.80 | STOP_HIT | 1.00 | -0.52% |
| SELL | retest2 | 2025-12-16 11:45:00 | 926.00 | 2025-12-22 09:15:00 | 930.80 | STOP_HIT | 1.00 | -0.52% |
| SELL | retest2 | 2025-12-16 12:15:00 | 926.00 | 2025-12-22 09:15:00 | 930.80 | STOP_HIT | 1.00 | -0.52% |
| SELL | retest2 | 2025-12-18 09:15:00 | 920.45 | 2025-12-22 09:15:00 | 930.80 | STOP_HIT | 1.00 | -1.12% |
| SELL | retest2 | 2025-12-19 11:15:00 | 923.05 | 2025-12-22 09:15:00 | 930.80 | STOP_HIT | 1.00 | -0.84% |
| SELL | retest2 | 2025-12-19 12:00:00 | 922.00 | 2025-12-22 09:15:00 | 930.80 | STOP_HIT | 1.00 | -0.95% |
| SELL | retest2 | 2026-01-08 11:15:00 | 919.65 | 2026-01-14 10:15:00 | 873.67 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-08 11:45:00 | 919.25 | 2026-01-14 10:15:00 | 873.29 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-08 14:45:00 | 919.25 | 2026-01-14 10:15:00 | 873.29 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-08 11:15:00 | 919.65 | 2026-01-16 09:15:00 | 879.35 | STOP_HIT | 0.50 | 4.38% |
| SELL | retest2 | 2026-01-08 11:45:00 | 919.25 | 2026-01-16 09:15:00 | 879.35 | STOP_HIT | 0.50 | 4.34% |
| SELL | retest2 | 2026-01-08 14:45:00 | 919.25 | 2026-01-16 09:15:00 | 879.35 | STOP_HIT | 0.50 | 4.34% |
| BUY | retest2 | 2026-02-09 09:15:00 | 877.05 | 2026-02-12 11:15:00 | 867.90 | STOP_HIT | 1.00 | -1.04% |
| BUY | retest2 | 2026-02-23 09:15:00 | 861.00 | 2026-02-25 12:15:00 | 849.20 | STOP_HIT | 1.00 | -1.37% |
| BUY | retest2 | 2026-02-24 10:15:00 | 856.45 | 2026-02-25 12:15:00 | 849.20 | STOP_HIT | 1.00 | -0.85% |
| BUY | retest2 | 2026-02-24 13:00:00 | 856.55 | 2026-02-25 12:15:00 | 849.20 | STOP_HIT | 1.00 | -0.86% |
| BUY | retest2 | 2026-02-24 13:45:00 | 856.95 | 2026-02-25 12:15:00 | 849.20 | STOP_HIT | 1.00 | -0.90% |
| SELL | retest2 | 2026-03-12 14:45:00 | 790.90 | 2026-03-18 09:15:00 | 785.25 | STOP_HIT | 1.00 | 0.71% |
| SELL | retest2 | 2026-03-13 09:30:00 | 784.70 | 2026-03-18 09:15:00 | 785.25 | STOP_HIT | 1.00 | -0.07% |
| SELL | retest2 | 2026-03-25 11:30:00 | 733.25 | 2026-03-30 09:15:00 | 696.59 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-25 12:15:00 | 734.15 | 2026-03-30 09:15:00 | 697.44 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-25 11:30:00 | 733.25 | 2026-04-01 09:15:00 | 708.70 | STOP_HIT | 0.50 | 3.35% |
| SELL | retest2 | 2026-03-25 12:15:00 | 734.15 | 2026-04-01 09:15:00 | 708.70 | STOP_HIT | 0.50 | 3.47% |
| BUY | retest2 | 2026-04-10 09:15:00 | 748.00 | 2026-04-13 09:15:00 | 735.00 | STOP_HIT | 1.00 | -1.74% |
| BUY | retest2 | 2026-04-10 15:00:00 | 742.55 | 2026-04-13 09:15:00 | 735.00 | STOP_HIT | 1.00 | -1.02% |
| BUY | retest2 | 2026-04-13 10:45:00 | 747.85 | 2026-04-22 09:15:00 | 816.15 | TARGET_HIT | 1.00 | 9.13% |
| BUY | retest2 | 2026-04-13 13:30:00 | 741.95 | 2026-04-22 10:15:00 | 822.64 | TARGET_HIT | 1.00 | 10.87% |
| BUY | retest2 | 2026-04-15 09:15:00 | 755.75 | 2026-04-22 10:15:00 | 831.33 | TARGET_HIT | 1.00 | 10.00% |
