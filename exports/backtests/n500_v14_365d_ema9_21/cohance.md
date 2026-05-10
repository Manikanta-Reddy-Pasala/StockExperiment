# Cohance Lifesciences Ltd. (COHANCE)

## Backtest Summary

- **Window:** 2025-04-11 09:15:00 → 2026-05-08 15:15:00 (1850 bars)
- **Last close:** 487.90
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 77 |
| ALERT1 | 48 |
| ALERT2 | 47 |
| ALERT2_SKIP | 31 |
| ALERT3 | 98 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 1 |
| ENTRY2 | 42 |
| PARTIAL | 13 |
| TARGET_HIT | 8 |
| STOP_HIT | 35 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 56 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 34 / 22
- **Target hits / Stop hits / Partials:** 8 / 35 / 13
- **Avg / median % per leg:** 2.29% / 2.80%
- **Sum % (uncompounded):** 128.30%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 15 | 3 | 20.0% | 3 | 12 | 0 | 0.26% | 3.8% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 15 | 3 | 20.0% | 3 | 12 | 0 | 0.26% | 3.8% |
| SELL (all) | 41 | 31 | 75.6% | 5 | 23 | 13 | 3.04% | 124.5% |
| SELL @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -3.15% | -3.2% |
| SELL @ 3rd Alert (retest2) | 40 | 31 | 77.5% | 5 | 22 | 13 | 3.19% | 127.6% |
| retest1 (combined) | 1 | 0 | 0.0% | 0 | 1 | 0 | -3.15% | -3.2% |
| retest2 (combined) | 55 | 34 | 61.8% | 8 | 34 | 13 | 2.39% | 131.5% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-13 15:15:00 | 1082.00 | 1077.26 | 1076.86 | EMA200 above EMA400 |

### Cycle 2 — SELL (started 2025-05-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-14 11:15:00 | 1071.40 | 1076.82 | 1076.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-14 13:15:00 | 1067.00 | 1073.68 | 1075.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-14 14:15:00 | 1081.10 | 1075.17 | 1075.88 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-14 14:15:00 | 1081.10 | 1075.17 | 1075.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-14 14:15:00 | 1081.10 | 1075.17 | 1075.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-14 15:00:00 | 1081.10 | 1075.17 | 1075.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-14 15:15:00 | 1070.00 | 1074.13 | 1075.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-15 09:15:00 | 1079.00 | 1074.13 | 1075.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-15 09:15:00 | 1070.70 | 1073.45 | 1074.92 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-15 11:00:00 | 1065.30 | 1071.82 | 1074.05 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-16 12:30:00 | 1067.40 | 1062.02 | 1066.04 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-16 14:15:00 | 1085.10 | 1069.72 | 1069.03 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-05-16 14:15:00 | 1085.10 | 1069.72 | 1069.03 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 3 — BUY (started 2025-05-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-16 14:15:00 | 1085.10 | 1069.72 | 1069.03 | EMA200 above EMA400 |

### Cycle 4 — SELL (started 2025-05-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-19 09:15:00 | 1044.00 | 1065.90 | 1067.50 | EMA200 below EMA400 |

### Cycle 5 — BUY (started 2025-05-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-20 10:15:00 | 1073.00 | 1065.68 | 1064.79 | EMA200 above EMA400 |

### Cycle 6 — SELL (started 2025-05-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-20 13:15:00 | 1052.40 | 1062.17 | 1063.36 | EMA200 below EMA400 |

### Cycle 7 — BUY (started 2025-05-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-21 13:15:00 | 1075.60 | 1064.67 | 1063.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-22 14:15:00 | 1081.10 | 1070.03 | 1067.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-23 13:15:00 | 1075.50 | 1078.66 | 1073.78 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-23 14:00:00 | 1075.50 | 1078.66 | 1073.78 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 14:15:00 | 1088.80 | 1080.69 | 1075.15 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-23 15:15:00 | 1092.60 | 1080.69 | 1075.15 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-26 09:30:00 | 1090.50 | 1085.16 | 1078.26 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-26 13:00:00 | 1091.70 | 1088.96 | 1081.96 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-27 09:30:00 | 1092.60 | 1092.23 | 1085.83 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 10:15:00 | 1089.30 | 1091.64 | 1086.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-27 10:45:00 | 1089.90 | 1091.64 | 1086.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 14:15:00 | 1092.70 | 1091.95 | 1088.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-27 14:45:00 | 1092.00 | 1091.95 | 1088.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 15:15:00 | 1089.00 | 1091.36 | 1088.19 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-28 09:15:00 | 1097.40 | 1091.36 | 1088.19 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-28 10:00:00 | 1095.00 | 1092.09 | 1088.80 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-29 09:15:00 | 1067.20 | 1096.91 | 1094.53 | SL hit (close<static) qty=1.00 sl=1074.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-05-29 09:15:00 | 1067.20 | 1096.91 | 1094.53 | SL hit (close<static) qty=1.00 sl=1074.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-05-29 09:15:00 | 1067.20 | 1096.91 | 1094.53 | SL hit (close<static) qty=1.00 sl=1074.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-05-29 09:15:00 | 1067.20 | 1096.91 | 1094.53 | SL hit (close<static) qty=1.00 sl=1074.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-05-29 09:15:00 | 1067.20 | 1096.91 | 1094.53 | SL hit (close<static) qty=1.00 sl=1088.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-05-29 09:15:00 | 1067.20 | 1096.91 | 1094.53 | SL hit (close<static) qty=1.00 sl=1088.00 alert=retest2 |

### Cycle 8 — SELL (started 2025-05-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-29 10:15:00 | 1062.40 | 1090.01 | 1091.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-30 10:15:00 | 1056.50 | 1069.57 | 1078.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-03 15:15:00 | 1029.00 | 1024.50 | 1036.84 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-04 09:30:00 | 1009.00 | 1022.08 | 1034.62 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 09:15:00 | 1040.80 | 1026.73 | 1031.07 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-06-05 09:15:00 | 1040.80 | 1026.73 | 1031.07 | SL hit (close>ema400) qty=1.00 sl=1031.07 alert=retest1 |
| ALERT3_SIDEWAYS | 2025-06-05 09:45:00 | 1045.40 | 1026.73 | 1031.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 10:15:00 | 1045.00 | 1030.38 | 1032.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-05 10:45:00 | 1047.40 | 1030.38 | 1032.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 9 — BUY (started 2025-06-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-05 12:15:00 | 1048.70 | 1036.39 | 1034.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-05 14:15:00 | 1064.00 | 1043.64 | 1038.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-06 11:15:00 | 1052.50 | 1052.73 | 1045.33 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-06 11:30:00 | 1052.00 | 1052.73 | 1045.33 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 14:15:00 | 1056.00 | 1053.84 | 1047.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-06 15:00:00 | 1056.00 | 1053.84 | 1047.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-09 09:15:00 | 1035.30 | 1050.00 | 1047.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-09 10:00:00 | 1035.30 | 1050.00 | 1047.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-09 10:15:00 | 1027.20 | 1045.44 | 1045.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-09 11:00:00 | 1027.20 | 1045.44 | 1045.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 10 — SELL (started 2025-06-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-09 11:15:00 | 1026.10 | 1041.57 | 1043.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-10 09:15:00 | 1019.20 | 1030.24 | 1036.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-12 09:15:00 | 1032.00 | 1010.04 | 1015.58 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-12 09:15:00 | 1032.00 | 1010.04 | 1015.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 09:15:00 | 1032.00 | 1010.04 | 1015.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-12 09:45:00 | 1032.20 | 1010.04 | 1015.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 10:15:00 | 1022.80 | 1012.60 | 1016.24 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-12 11:45:00 | 1015.90 | 1013.82 | 1016.46 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-18 11:15:00 | 1010.60 | 998.74 | 997.57 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 11 — BUY (started 2025-06-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-18 11:15:00 | 1010.60 | 998.74 | 997.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-18 12:15:00 | 1015.30 | 1002.05 | 999.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-19 10:15:00 | 1005.70 | 1007.68 | 1003.55 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-19 10:15:00 | 1005.70 | 1007.68 | 1003.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 10:15:00 | 1005.70 | 1007.68 | 1003.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-19 11:00:00 | 1005.70 | 1007.68 | 1003.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 11:15:00 | 1003.10 | 1006.76 | 1003.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-19 12:00:00 | 1003.10 | 1006.76 | 1003.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 12:15:00 | 1001.20 | 1005.65 | 1003.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-19 12:45:00 | 1000.60 | 1005.65 | 1003.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 13:15:00 | 996.60 | 1003.84 | 1002.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-19 13:45:00 | 992.50 | 1003.84 | 1002.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 12 — SELL (started 2025-06-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-19 14:15:00 | 992.00 | 1001.47 | 1001.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-19 15:15:00 | 982.40 | 997.66 | 999.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-23 09:15:00 | 997.80 | 988.42 | 992.54 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-23 09:15:00 | 997.80 | 988.42 | 992.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 09:15:00 | 997.80 | 988.42 | 992.54 | EMA400 retest candle locked (from downside) |

### Cycle 13 — BUY (started 2025-06-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-25 10:15:00 | 1005.00 | 991.78 | 990.49 | EMA200 above EMA400 |

### Cycle 14 — SELL (started 2025-06-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-25 14:15:00 | 981.20 | 988.63 | 989.56 | EMA200 below EMA400 |

### Cycle 15 — BUY (started 2025-06-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-26 11:15:00 | 995.70 | 989.60 | 989.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-26 12:15:00 | 999.70 | 991.62 | 990.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-27 09:15:00 | 996.20 | 996.67 | 993.69 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-27 09:15:00 | 996.20 | 996.67 | 993.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 09:15:00 | 996.20 | 996.67 | 993.69 | EMA400 retest candle locked (from upside) |

### Cycle 16 — SELL (started 2025-06-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-27 11:15:00 | 977.00 | 990.63 | 991.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-27 12:15:00 | 970.30 | 986.56 | 989.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-30 12:15:00 | 966.20 | 962.76 | 972.89 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-30 12:15:00 | 966.20 | 962.76 | 972.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 12:15:00 | 966.20 | 962.76 | 972.89 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-01 09:15:00 | 948.45 | 965.44 | 971.69 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-02 10:00:00 | 954.35 | 960.27 | 965.09 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-02 11:30:00 | 958.00 | 958.80 | 963.53 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-03 09:15:00 | 990.95 | 968.95 | 966.87 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-03 09:15:00 | 990.95 | 968.95 | 966.87 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-03 09:15:00 | 990.95 | 968.95 | 966.87 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 17 — BUY (started 2025-07-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-03 09:15:00 | 990.95 | 968.95 | 966.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-03 10:15:00 | 997.00 | 974.56 | 969.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-07 10:15:00 | 1020.15 | 1021.41 | 1006.65 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-07 11:00:00 | 1020.15 | 1021.41 | 1006.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 11:15:00 | 1005.85 | 1018.30 | 1006.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-07 12:00:00 | 1005.85 | 1018.30 | 1006.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 12:15:00 | 1014.35 | 1017.51 | 1007.28 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-07 14:45:00 | 1021.75 | 1018.26 | 1009.42 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-08 10:15:00 | 998.10 | 1012.36 | 1008.83 | SL hit (close<static) qty=1.00 sl=1005.00 alert=retest2 |

### Cycle 18 — SELL (started 2025-07-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-08 12:15:00 | 993.45 | 1005.58 | 1006.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-09 14:15:00 | 986.65 | 995.50 | 999.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-10 09:15:00 | 996.50 | 995.30 | 998.80 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-10 09:15:00 | 996.50 | 995.30 | 998.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 09:15:00 | 996.50 | 995.30 | 998.80 | EMA400 retest candle locked (from downside) |

### Cycle 19 — BUY (started 2025-07-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-11 09:15:00 | 1019.95 | 1001.16 | 1000.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-14 14:15:00 | 1030.95 | 1016.84 | 1011.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-18 13:15:00 | 1086.15 | 1087.42 | 1075.73 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-18 14:00:00 | 1086.15 | 1087.42 | 1075.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 15:15:00 | 1074.10 | 1084.45 | 1076.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-21 09:15:00 | 1067.20 | 1084.45 | 1076.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 09:15:00 | 1075.15 | 1082.59 | 1076.28 | EMA400 retest candle locked (from upside) |

### Cycle 20 — SELL (started 2025-07-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-21 15:15:00 | 1070.00 | 1073.57 | 1073.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-22 09:15:00 | 1067.60 | 1072.37 | 1073.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-29 14:15:00 | 1008.40 | 998.27 | 1008.13 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-29 14:15:00 | 1008.40 | 998.27 | 1008.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 14:15:00 | 1008.40 | 998.27 | 1008.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 15:00:00 | 1008.40 | 998.27 | 1008.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 15:15:00 | 1002.00 | 999.02 | 1007.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-30 09:15:00 | 1010.60 | 999.02 | 1007.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 09:15:00 | 1008.00 | 1000.82 | 1007.62 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-30 10:30:00 | 1004.75 | 1001.26 | 1007.20 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-30 11:45:00 | 1002.50 | 1001.65 | 1006.84 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-01 12:15:00 | 954.51 | 973.96 | 985.61 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-01 14:15:00 | 981.00 | 974.22 | 983.63 | SL hit (close>ema200) qty=0.50 sl=974.22 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-05 11:15:00 | 952.38 | 961.42 | 969.62 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-05 14:15:00 | 961.55 | 959.80 | 966.69 | SL hit (close>ema200) qty=0.50 sl=959.80 alert=retest2 |

### Cycle 21 — BUY (started 2025-08-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-12 10:15:00 | 949.40 | 935.93 | 934.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-12 11:15:00 | 950.80 | 938.90 | 936.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-14 09:15:00 | 941.20 | 969.82 | 960.25 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-14 09:15:00 | 941.20 | 969.82 | 960.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 09:15:00 | 941.20 | 969.82 | 960.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-14 09:45:00 | 943.30 | 969.82 | 960.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 10:15:00 | 919.30 | 959.71 | 956.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-14 11:00:00 | 919.30 | 959.71 | 956.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 22 — SELL (started 2025-08-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-14 11:15:00 | 913.85 | 950.54 | 952.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-14 13:15:00 | 909.75 | 937.02 | 945.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-18 15:15:00 | 907.00 | 904.71 | 919.41 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-19 09:15:00 | 901.30 | 904.71 | 919.41 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 11:15:00 | 916.50 | 907.72 | 917.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-19 12:00:00 | 916.50 | 907.72 | 917.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 12:15:00 | 914.60 | 909.10 | 916.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-19 13:15:00 | 917.45 | 909.10 | 916.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 13:15:00 | 917.45 | 910.77 | 916.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-19 14:00:00 | 917.45 | 910.77 | 916.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 14:15:00 | 920.30 | 912.67 | 917.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-19 15:00:00 | 920.30 | 912.67 | 917.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 15:15:00 | 921.05 | 914.35 | 917.63 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-20 09:15:00 | 908.50 | 914.35 | 917.63 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-25 10:00:00 | 909.70 | 893.26 | 897.13 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-28 10:15:00 | 863.07 | 880.07 | 887.88 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-28 10:15:00 | 864.22 | 880.07 | 887.88 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-29 10:15:00 | 876.05 | 873.20 | 879.70 | SL hit (close>ema200) qty=0.50 sl=873.20 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-29 10:15:00 | 876.05 | 873.20 | 879.70 | SL hit (close>ema200) qty=0.50 sl=873.20 alert=retest2 |

### Cycle 23 — BUY (started 2025-09-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-01 10:15:00 | 895.55 | 882.55 | 881.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-01 15:15:00 | 904.90 | 892.23 | 887.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-03 12:15:00 | 907.50 | 910.10 | 902.75 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-03 13:00:00 | 907.50 | 910.10 | 902.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 14:15:00 | 904.15 | 908.51 | 903.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-03 15:00:00 | 904.15 | 908.51 | 903.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 15:15:00 | 902.90 | 907.39 | 903.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 09:15:00 | 897.05 | 907.39 | 903.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 09:15:00 | 905.75 | 907.06 | 903.47 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-05 09:45:00 | 915.15 | 904.58 | 903.28 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-05 13:45:00 | 918.00 | 908.10 | 905.37 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2025-09-08 12:15:00 | 1006.67 | 949.36 | 928.33 | Target hit (10%) qty=1.00 alert=retest2 |
| Target hit | 2025-09-08 14:15:00 | 1009.80 | 969.07 | 941.22 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 24 — SELL (started 2025-09-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-12 15:15:00 | 976.00 | 986.15 | 986.85 | EMA200 below EMA400 |

### Cycle 25 — BUY (started 2025-09-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-16 12:15:00 | 991.80 | 985.26 | 984.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-16 13:15:00 | 992.85 | 986.77 | 985.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-16 15:15:00 | 985.50 | 987.19 | 985.92 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-16 15:15:00 | 985.50 | 987.19 | 985.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 15:15:00 | 985.50 | 987.19 | 985.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-17 09:15:00 | 990.65 | 987.19 | 985.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 09:15:00 | 992.00 | 988.15 | 986.47 | EMA400 retest candle locked (from upside) |

### Cycle 26 — SELL (started 2025-09-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-17 13:15:00 | 972.55 | 984.49 | 985.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-17 14:15:00 | 970.80 | 981.75 | 983.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-19 13:15:00 | 917.20 | 916.27 | 934.13 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-19 13:30:00 | 917.50 | 916.27 | 934.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 15:15:00 | 900.00 | 894.47 | 903.18 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-24 09:15:00 | 887.60 | 894.47 | 903.18 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-24 10:45:00 | 892.00 | 893.44 | 901.19 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-24 15:15:00 | 906.00 | 893.04 | 897.50 | SL hit (close>static) qty=1.00 sl=905.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-24 15:15:00 | 906.00 | 893.04 | 897.50 | SL hit (close>static) qty=1.00 sl=905.00 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-25 09:15:00 | 892.30 | 893.04 | 897.50 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-25 13:15:00 | 895.15 | 892.70 | 895.83 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 13:15:00 | 895.00 | 893.16 | 895.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-25 14:15:00 | 895.00 | 893.16 | 895.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 14:15:00 | 898.35 | 894.20 | 895.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-25 14:45:00 | 904.00 | 894.20 | 895.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 15:15:00 | 892.00 | 893.76 | 895.63 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-26 09:15:00 | 880.85 | 893.76 | 895.63 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-03 14:15:00 | 875.25 | 875.06 | 875.05 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-03 14:15:00 | 875.25 | 875.06 | 875.05 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-03 14:15:00 | 875.25 | 875.06 | 875.05 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 27 — BUY (started 2025-10-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-03 14:15:00 | 875.25 | 875.06 | 875.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-03 15:15:00 | 879.00 | 875.85 | 875.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-06 09:15:00 | 873.50 | 875.38 | 875.23 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-06 09:15:00 | 873.50 | 875.38 | 875.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 09:15:00 | 873.50 | 875.38 | 875.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-06 09:30:00 | 872.15 | 875.38 | 875.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 28 — SELL (started 2025-10-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-06 10:15:00 | 873.20 | 874.94 | 875.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-06 11:15:00 | 868.90 | 873.73 | 874.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-07 10:15:00 | 874.15 | 870.91 | 872.38 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-07 10:15:00 | 874.15 | 870.91 | 872.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 10:15:00 | 874.15 | 870.91 | 872.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-07 11:00:00 | 874.15 | 870.91 | 872.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 11:15:00 | 874.45 | 871.62 | 872.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-07 11:45:00 | 875.05 | 871.62 | 872.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 12:15:00 | 874.65 | 872.23 | 872.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-07 12:30:00 | 875.30 | 872.23 | 872.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 29 — BUY (started 2025-10-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-07 14:15:00 | 875.20 | 873.31 | 873.19 | EMA200 above EMA400 |

### Cycle 30 — SELL (started 2025-10-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-07 15:15:00 | 871.95 | 873.04 | 873.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-08 09:15:00 | 866.20 | 871.67 | 872.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-09 09:15:00 | 870.55 | 865.13 | 867.84 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-09 09:15:00 | 870.55 | 865.13 | 867.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 09:15:00 | 870.55 | 865.13 | 867.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-09 09:30:00 | 866.65 | 865.13 | 867.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 10:15:00 | 877.00 | 867.51 | 868.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-09 11:00:00 | 877.00 | 867.51 | 868.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 31 — BUY (started 2025-10-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-09 11:15:00 | 882.20 | 870.44 | 869.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-10 11:15:00 | 885.65 | 879.23 | 875.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-13 09:15:00 | 890.50 | 892.92 | 884.91 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-13 09:45:00 | 890.15 | 892.92 | 884.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 10:15:00 | 889.55 | 892.25 | 885.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-13 10:30:00 | 889.85 | 892.25 | 885.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 11:15:00 | 890.00 | 891.80 | 885.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-13 11:45:00 | 887.55 | 891.80 | 885.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 15:15:00 | 886.65 | 889.31 | 886.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-14 09:15:00 | 882.40 | 889.31 | 886.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 09:15:00 | 874.00 | 886.25 | 885.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-14 10:00:00 | 874.00 | 886.25 | 885.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 32 — SELL (started 2025-10-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-14 10:15:00 | 873.10 | 883.62 | 884.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-14 13:15:00 | 866.80 | 878.97 | 881.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-15 12:15:00 | 871.30 | 870.49 | 875.40 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-15 13:00:00 | 871.30 | 870.49 | 875.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 14:15:00 | 866.60 | 870.20 | 874.45 | EMA400 retest candle locked (from downside) |

### Cycle 33 — BUY (started 2025-10-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-16 10:15:00 | 892.30 | 876.69 | 876.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-16 11:15:00 | 896.00 | 880.56 | 878.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-17 09:15:00 | 889.00 | 889.93 | 884.54 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-17 12:15:00 | 881.90 | 887.72 | 884.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 12:15:00 | 881.90 | 887.72 | 884.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-17 13:00:00 | 881.90 | 887.72 | 884.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 13:15:00 | 888.20 | 887.81 | 885.15 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-17 14:30:00 | 889.95 | 888.38 | 885.65 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-17 15:00:00 | 890.65 | 888.38 | 885.65 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-20 09:15:00 | 880.75 | 887.26 | 885.64 | SL hit (close<static) qty=1.00 sl=881.10 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-20 09:15:00 | 880.75 | 887.26 | 885.64 | SL hit (close<static) qty=1.00 sl=881.10 alert=retest2 |

### Cycle 34 — SELL (started 2025-10-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-20 11:15:00 | 878.45 | 884.27 | 884.48 | EMA200 below EMA400 |

### Cycle 35 — BUY (started 2025-10-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-21 13:15:00 | 903.00 | 886.41 | 885.21 | EMA200 above EMA400 |

### Cycle 36 — SELL (started 2025-10-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-24 10:15:00 | 882.45 | 885.84 | 886.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-24 11:15:00 | 880.90 | 884.85 | 885.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-28 10:15:00 | 860.35 | 859.27 | 867.78 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-28 11:00:00 | 860.35 | 859.27 | 867.78 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 13:15:00 | 859.90 | 860.90 | 866.54 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-28 14:45:00 | 859.30 | 860.23 | 865.72 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-10-29 09:15:00 | 773.37 | 848.90 | 859.61 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 37 — BUY (started 2025-11-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-27 13:15:00 | 575.15 | 570.20 | 570.03 | EMA200 above EMA400 |

### Cycle 38 — SELL (started 2025-11-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-28 09:15:00 | 564.75 | 569.88 | 570.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-28 10:15:00 | 563.45 | 568.59 | 569.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-01 09:15:00 | 566.10 | 565.86 | 567.38 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-01 09:15:00 | 566.10 | 565.86 | 567.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 09:15:00 | 566.10 | 565.86 | 567.38 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-01 10:45:00 | 562.50 | 565.36 | 567.02 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-01 11:30:00 | 561.80 | 564.71 | 566.57 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-05 09:15:00 | 534.38 | 546.57 | 548.72 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-05 09:15:00 | 533.71 | 546.57 | 548.72 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-09 12:15:00 | 532.10 | 530.78 | 535.27 | SL hit (close>ema200) qty=0.50 sl=530.78 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-09 12:15:00 | 532.10 | 530.78 | 535.27 | SL hit (close>ema200) qty=0.50 sl=530.78 alert=retest2 |

### Cycle 39 — BUY (started 2025-12-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-12 10:15:00 | 535.00 | 532.37 | 532.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-12 14:15:00 | 536.80 | 533.36 | 532.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-15 12:15:00 | 532.90 | 534.44 | 533.57 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-15 12:15:00 | 532.90 | 534.44 | 533.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 12:15:00 | 532.90 | 534.44 | 533.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-15 13:00:00 | 532.90 | 534.44 | 533.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 13:15:00 | 534.05 | 534.36 | 533.61 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-16 09:15:00 | 535.00 | 533.31 | 533.24 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-17 09:15:00 | 526.60 | 535.02 | 534.95 | SL hit (close<static) qty=1.00 sl=532.65 alert=retest2 |

### Cycle 40 — SELL (started 2025-12-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-17 10:15:00 | 522.95 | 532.61 | 533.86 | EMA200 below EMA400 |

### Cycle 41 — BUY (started 2025-12-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-19 10:15:00 | 535.70 | 530.59 | 529.94 | EMA200 above EMA400 |

### Cycle 42 — SELL (started 2025-12-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-19 12:15:00 | 525.95 | 529.32 | 529.46 | EMA200 below EMA400 |

### Cycle 43 — BUY (started 2025-12-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-19 14:15:00 | 538.50 | 531.32 | 530.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-22 11:15:00 | 540.80 | 534.98 | 532.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-23 09:15:00 | 537.50 | 537.90 | 535.23 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-23 09:15:00 | 537.50 | 537.90 | 535.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 09:15:00 | 537.50 | 537.90 | 535.23 | EMA400 retest candle locked (from upside) |

### Cycle 44 — SELL (started 2025-12-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-26 11:15:00 | 535.10 | 535.95 | 535.98 | EMA200 below EMA400 |

### Cycle 45 — BUY (started 2025-12-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-26 14:15:00 | 537.00 | 536.04 | 536.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-26 15:15:00 | 538.00 | 536.43 | 536.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-29 09:15:00 | 535.00 | 536.14 | 536.08 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-29 09:15:00 | 535.00 | 536.14 | 536.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 09:15:00 | 535.00 | 536.14 | 536.08 | EMA400 retest candle locked (from upside) |

### Cycle 46 — SELL (started 2025-12-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-29 10:15:00 | 528.70 | 534.66 | 535.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-30 09:15:00 | 524.95 | 531.16 | 533.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-31 10:15:00 | 528.75 | 527.94 | 529.88 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-31 10:15:00 | 528.75 | 527.94 | 529.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 10:15:00 | 528.75 | 527.94 | 529.88 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-31 15:15:00 | 527.50 | 528.64 | 529.64 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-01 09:30:00 | 527.10 | 528.03 | 529.18 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-06 15:15:00 | 501.12 | 508.25 | 513.45 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-06 15:15:00 | 500.75 | 508.25 | 513.45 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-07 09:15:00 | 512.35 | 509.07 | 513.35 | SL hit (close>ema200) qty=0.50 sl=509.07 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-07 09:15:00 | 512.35 | 509.07 | 513.35 | SL hit (close>ema200) qty=0.50 sl=509.07 alert=retest2 |

### Cycle 47 — BUY (started 2026-01-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-30 13:15:00 | 383.85 | 380.28 | 379.83 | EMA200 above EMA400 |

### Cycle 48 — SELL (started 2026-02-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-01 12:15:00 | 371.85 | 378.33 | 379.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-02 09:15:00 | 366.40 | 372.85 | 375.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-02 14:15:00 | 371.40 | 367.58 | 371.55 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-02 14:15:00 | 371.40 | 367.58 | 371.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 14:15:00 | 371.40 | 367.58 | 371.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-02 15:00:00 | 371.40 | 367.58 | 371.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 15:15:00 | 373.45 | 368.76 | 371.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-03 09:15:00 | 384.80 | 368.76 | 371.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 09:15:00 | 384.50 | 371.91 | 372.88 | EMA400 retest candle locked (from downside) |

### Cycle 49 — BUY (started 2026-02-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 10:15:00 | 386.25 | 374.77 | 374.10 | EMA200 above EMA400 |

### Cycle 50 — SELL (started 2026-02-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-04 12:15:00 | 367.80 | 374.82 | 375.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-04 14:15:00 | 366.35 | 372.32 | 374.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-09 09:15:00 | 353.65 | 345.39 | 352.20 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-09 09:15:00 | 353.65 | 345.39 | 352.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 09:15:00 | 353.65 | 345.39 | 352.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-09 10:00:00 | 353.65 | 345.39 | 352.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 10:15:00 | 357.85 | 347.88 | 352.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-09 11:00:00 | 357.85 | 347.88 | 352.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 12:15:00 | 350.75 | 348.31 | 352.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-09 12:45:00 | 353.20 | 348.31 | 352.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 09:15:00 | 352.55 | 349.60 | 351.51 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-11 09:30:00 | 348.05 | 351.44 | 351.74 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-11 14:15:00 | 356.10 | 352.49 | 352.10 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 51 — BUY (started 2026-02-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-11 14:15:00 | 356.10 | 352.49 | 352.10 | EMA200 above EMA400 |

### Cycle 52 — SELL (started 2026-02-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-12 12:15:00 | 346.40 | 351.04 | 351.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-13 09:15:00 | 324.30 | 345.75 | 349.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-17 09:15:00 | 311.85 | 311.50 | 321.75 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-17 09:45:00 | 313.80 | 311.50 | 321.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 09:15:00 | 325.70 | 312.55 | 316.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-18 09:30:00 | 328.95 | 312.55 | 316.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 10:15:00 | 326.65 | 315.37 | 317.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-18 10:45:00 | 336.55 | 315.37 | 317.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 53 — BUY (started 2026-02-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-18 12:15:00 | 324.00 | 319.28 | 319.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-18 13:15:00 | 329.35 | 321.29 | 320.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-19 09:15:00 | 322.55 | 322.57 | 321.09 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-19 09:15:00 | 322.55 | 322.57 | 321.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 09:15:00 | 322.55 | 322.57 | 321.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 09:45:00 | 321.85 | 322.57 | 321.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 10:15:00 | 319.00 | 321.86 | 320.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 10:45:00 | 319.80 | 321.86 | 320.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 11:15:00 | 317.20 | 320.92 | 320.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 12:00:00 | 317.20 | 320.92 | 320.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 54 — SELL (started 2026-02-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-19 12:15:00 | 314.65 | 319.67 | 320.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-20 09:15:00 | 311.90 | 315.88 | 317.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-23 09:15:00 | 315.70 | 310.30 | 313.32 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-23 09:15:00 | 315.70 | 310.30 | 313.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 09:15:00 | 315.70 | 310.30 | 313.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-23 09:30:00 | 322.15 | 310.30 | 313.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 10:15:00 | 313.20 | 310.88 | 313.31 | EMA400 retest candle locked (from downside) |

### Cycle 55 — BUY (started 2026-02-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-23 13:15:00 | 317.50 | 315.07 | 314.87 | EMA200 above EMA400 |

### Cycle 56 — SELL (started 2026-02-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-23 14:15:00 | 313.00 | 314.65 | 314.70 | EMA200 below EMA400 |

### Cycle 57 — BUY (started 2026-02-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-23 15:15:00 | 315.10 | 314.74 | 314.74 | EMA200 above EMA400 |

### Cycle 58 — SELL (started 2026-02-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-24 09:15:00 | 310.10 | 313.81 | 314.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-24 12:15:00 | 304.50 | 310.22 | 312.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-24 14:15:00 | 310.00 | 309.49 | 311.65 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-24 15:00:00 | 310.00 | 309.49 | 311.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 09:15:00 | 307.85 | 309.27 | 311.19 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-26 12:30:00 | 306.40 | 310.14 | 310.53 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-27 09:15:00 | 304.90 | 309.14 | 309.92 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-27 12:15:00 | 305.70 | 307.47 | 308.83 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-02 10:30:00 | 304.75 | 307.42 | 308.29 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 09:15:00 | 289.40 | 297.81 | 301.53 | EMA400 retest candle locked (from downside) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-05 09:15:00 | 291.08 | 297.81 | 301.53 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-05 09:15:00 | 289.65 | 297.81 | 301.53 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-05 09:15:00 | 290.41 | 297.81 | 301.53 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-05 09:15:00 | 289.51 | 297.81 | 301.53 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-06 09:15:00 | 285.60 | 292.53 | 296.75 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2026-03-09 09:15:00 | 275.76 | 280.58 | 287.13 | Target hit (10%) qty=0.50 alert=retest2 |
| Target hit | 2026-03-09 09:15:00 | 274.41 | 280.58 | 287.13 | Target hit (10%) qty=0.50 alert=retest2 |
| Target hit | 2026-03-09 09:15:00 | 275.13 | 280.58 | 287.13 | Target hit (10%) qty=0.50 alert=retest2 |
| Target hit | 2026-03-09 09:15:00 | 274.28 | 280.58 | 287.13 | Target hit (10%) qty=0.50 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-09 09:15:00 | 271.32 | 280.58 | 287.13 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-09 14:15:00 | 282.75 | 279.63 | 283.95 | SL hit (close>ema200) qty=0.50 sl=279.63 alert=retest2 |

### Cycle 59 — BUY (started 2026-03-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-10 13:15:00 | 293.25 | 285.84 | 285.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-10 14:15:00 | 295.20 | 287.71 | 286.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-11 14:15:00 | 291.80 | 297.17 | 293.04 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-11 14:15:00 | 291.80 | 297.17 | 293.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 14:15:00 | 291.80 | 297.17 | 293.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-11 15:00:00 | 291.80 | 297.17 | 293.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 15:15:00 | 291.50 | 296.03 | 292.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-12 09:15:00 | 286.45 | 296.03 | 292.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 09:15:00 | 286.65 | 294.16 | 292.33 | EMA400 retest candle locked (from upside) |

### Cycle 60 — SELL (started 2026-03-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-12 12:15:00 | 288.15 | 291.24 | 291.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-13 09:15:00 | 284.80 | 289.29 | 290.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-13 11:15:00 | 289.25 | 288.65 | 289.81 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-13 11:15:00 | 289.25 | 288.65 | 289.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-13 11:15:00 | 289.25 | 288.65 | 289.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-13 11:30:00 | 289.10 | 288.65 | 289.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-13 12:15:00 | 290.85 | 289.09 | 289.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-13 13:00:00 | 290.85 | 289.09 | 289.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 61 — BUY (started 2026-03-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-13 13:15:00 | 296.60 | 290.59 | 290.51 | EMA200 above EMA400 |

### Cycle 62 — SELL (started 2026-03-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-16 13:15:00 | 289.00 | 291.06 | 291.19 | EMA200 below EMA400 |

### Cycle 63 — BUY (started 2026-03-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-16 14:15:00 | 292.95 | 291.43 | 291.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-17 09:15:00 | 300.05 | 293.56 | 292.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-18 12:15:00 | 301.85 | 302.27 | 298.86 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-18 13:00:00 | 301.85 | 302.27 | 298.86 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 09:15:00 | 301.75 | 303.75 | 300.81 | EMA400 retest candle locked (from upside) |

### Cycle 64 — SELL (started 2026-03-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-20 10:15:00 | 297.35 | 299.98 | 300.03 | EMA200 below EMA400 |

### Cycle 65 — BUY (started 2026-03-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-23 14:15:00 | 300.00 | 299.57 | 299.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-23 15:15:00 | 301.00 | 299.86 | 299.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-24 09:15:00 | 298.95 | 299.68 | 299.62 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-24 09:15:00 | 298.95 | 299.68 | 299.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 09:15:00 | 298.95 | 299.68 | 299.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-24 09:45:00 | 299.00 | 299.68 | 299.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 66 — SELL (started 2026-03-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-24 10:15:00 | 295.50 | 298.84 | 299.25 | EMA200 below EMA400 |

### Cycle 67 — BUY (started 2026-03-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 09:15:00 | 306.90 | 299.88 | 299.53 | EMA200 above EMA400 |

### Cycle 68 — SELL (started 2026-03-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 14:15:00 | 300.50 | 301.43 | 301.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 09:15:00 | 299.55 | 300.87 | 301.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-30 11:15:00 | 302.25 | 300.64 | 300.98 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-30 11:15:00 | 302.25 | 300.64 | 300.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-30 11:15:00 | 302.25 | 300.64 | 300.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-30 12:00:00 | 302.25 | 300.64 | 300.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-30 12:15:00 | 302.55 | 301.02 | 301.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-30 12:30:00 | 303.25 | 301.02 | 301.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 69 — BUY (started 2026-03-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-30 13:15:00 | 303.70 | 301.56 | 301.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-01 09:15:00 | 312.60 | 304.02 | 302.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-02 09:15:00 | 302.50 | 308.23 | 306.17 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-02 09:15:00 | 302.50 | 308.23 | 306.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 09:15:00 | 302.50 | 308.23 | 306.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-02 10:00:00 | 302.50 | 308.23 | 306.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 10:15:00 | 303.35 | 307.26 | 305.91 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-02 11:30:00 | 304.85 | 307.03 | 305.93 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-06 09:15:00 | 293.40 | 304.56 | 305.24 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 70 — SELL (started 2026-04-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-06 09:15:00 | 293.40 | 304.56 | 305.24 | EMA200 below EMA400 |

### Cycle 71 — BUY (started 2026-04-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-08 10:15:00 | 307.90 | 303.41 | 303.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-09 09:15:00 | 310.35 | 307.61 | 305.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-09 13:15:00 | 309.45 | 309.87 | 307.59 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-09 13:15:00 | 309.45 | 309.87 | 307.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 13:15:00 | 309.45 | 309.87 | 307.59 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-09 15:00:00 | 314.00 | 310.69 | 308.17 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2026-04-10 09:15:00 | 345.40 | 321.48 | 313.61 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 72 — SELL (started 2026-04-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-21 10:15:00 | 361.40 | 368.23 | 368.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-21 13:15:00 | 360.30 | 364.67 | 366.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-22 09:15:00 | 368.00 | 364.05 | 365.63 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-22 09:15:00 | 368.00 | 364.05 | 365.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-22 09:15:00 | 368.00 | 364.05 | 365.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-22 10:00:00 | 368.00 | 364.05 | 365.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-22 10:15:00 | 367.45 | 364.73 | 365.80 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-22 11:15:00 | 364.90 | 364.73 | 365.80 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-23 11:15:00 | 374.10 | 367.11 | 366.31 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 73 — BUY (started 2026-04-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-23 11:15:00 | 374.10 | 367.11 | 366.31 | EMA200 above EMA400 |

### Cycle 74 — SELL (started 2026-04-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-23 15:15:00 | 364.95 | 365.94 | 365.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-24 11:15:00 | 360.00 | 364.03 | 365.04 | Break + close below crossover candle low |

### Cycle 75 — BUY (started 2026-04-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 09:15:00 | 432.10 | 375.13 | 369.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-28 09:15:00 | 506.10 | 434.95 | 406.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-29 10:15:00 | 472.30 | 475.04 | 448.63 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-29 10:45:00 | 472.00 | 475.04 | 448.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 09:15:00 | 447.25 | 466.86 | 455.94 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-30 12:15:00 | 470.50 | 463.08 | 455.94 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-05-05 11:15:00 | 460.80 | 466.84 | 466.94 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 76 — SELL (started 2026-05-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-05 11:15:00 | 460.80 | 466.84 | 466.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-05-05 15:15:00 | 454.00 | 460.12 | 463.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-05-06 09:15:00 | 483.75 | 464.85 | 465.23 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-06 09:15:00 | 483.75 | 464.85 | 465.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 09:15:00 | 483.75 | 464.85 | 465.23 | EMA400 retest candle locked (from downside) |

### Cycle 77 — BUY (started 2026-05-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-06 10:15:00 | 476.50 | 467.18 | 466.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-06 12:15:00 | 484.10 | 472.87 | 469.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-08 14:15:00 | 487.45 | 488.44 | 483.70 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-05-08 15:00:00 | 487.45 | 488.44 | 483.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 15:15:00 | 487.90 | 488.33 | 484.08 | EMA400 retest candle locked (from upside) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-05-15 11:00:00 | 1065.30 | 2025-05-16 14:15:00 | 1085.10 | STOP_HIT | 1.00 | -1.86% |
| SELL | retest2 | 2025-05-16 12:30:00 | 1067.40 | 2025-05-16 14:15:00 | 1085.10 | STOP_HIT | 1.00 | -1.66% |
| BUY | retest2 | 2025-05-23 15:15:00 | 1092.60 | 2025-05-29 09:15:00 | 1067.20 | STOP_HIT | 1.00 | -2.32% |
| BUY | retest2 | 2025-05-26 09:30:00 | 1090.50 | 2025-05-29 09:15:00 | 1067.20 | STOP_HIT | 1.00 | -2.14% |
| BUY | retest2 | 2025-05-26 13:00:00 | 1091.70 | 2025-05-29 09:15:00 | 1067.20 | STOP_HIT | 1.00 | -2.24% |
| BUY | retest2 | 2025-05-27 09:30:00 | 1092.60 | 2025-05-29 09:15:00 | 1067.20 | STOP_HIT | 1.00 | -2.32% |
| BUY | retest2 | 2025-05-28 09:15:00 | 1097.40 | 2025-05-29 09:15:00 | 1067.20 | STOP_HIT | 1.00 | -2.75% |
| BUY | retest2 | 2025-05-28 10:00:00 | 1095.00 | 2025-05-29 09:15:00 | 1067.20 | STOP_HIT | 1.00 | -2.54% |
| SELL | retest1 | 2025-06-04 09:30:00 | 1009.00 | 2025-06-05 09:15:00 | 1040.80 | STOP_HIT | 1.00 | -3.15% |
| SELL | retest2 | 2025-06-12 11:45:00 | 1015.90 | 2025-06-18 11:15:00 | 1010.60 | STOP_HIT | 1.00 | 0.52% |
| SELL | retest2 | 2025-07-01 09:15:00 | 948.45 | 2025-07-03 09:15:00 | 990.95 | STOP_HIT | 1.00 | -4.48% |
| SELL | retest2 | 2025-07-02 10:00:00 | 954.35 | 2025-07-03 09:15:00 | 990.95 | STOP_HIT | 1.00 | -3.84% |
| SELL | retest2 | 2025-07-02 11:30:00 | 958.00 | 2025-07-03 09:15:00 | 990.95 | STOP_HIT | 1.00 | -3.44% |
| BUY | retest2 | 2025-07-07 14:45:00 | 1021.75 | 2025-07-08 10:15:00 | 998.10 | STOP_HIT | 1.00 | -2.31% |
| SELL | retest2 | 2025-07-30 10:30:00 | 1004.75 | 2025-08-01 12:15:00 | 954.51 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-30 10:30:00 | 1004.75 | 2025-08-01 14:15:00 | 981.00 | STOP_HIT | 0.50 | 2.36% |
| SELL | retest2 | 2025-07-30 11:45:00 | 1002.50 | 2025-08-05 11:15:00 | 952.38 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-30 11:45:00 | 1002.50 | 2025-08-05 14:15:00 | 961.55 | STOP_HIT | 0.50 | 4.08% |
| SELL | retest2 | 2025-08-20 09:15:00 | 908.50 | 2025-08-28 10:15:00 | 863.07 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-25 10:00:00 | 909.70 | 2025-08-28 10:15:00 | 864.22 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-20 09:15:00 | 908.50 | 2025-08-29 10:15:00 | 876.05 | STOP_HIT | 0.50 | 3.57% |
| SELL | retest2 | 2025-08-25 10:00:00 | 909.70 | 2025-08-29 10:15:00 | 876.05 | STOP_HIT | 0.50 | 3.70% |
| BUY | retest2 | 2025-09-05 09:45:00 | 915.15 | 2025-09-08 12:15:00 | 1006.67 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-09-05 13:45:00 | 918.00 | 2025-09-08 14:15:00 | 1009.80 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-09-24 09:15:00 | 887.60 | 2025-09-24 15:15:00 | 906.00 | STOP_HIT | 1.00 | -2.07% |
| SELL | retest2 | 2025-09-24 10:45:00 | 892.00 | 2025-09-24 15:15:00 | 906.00 | STOP_HIT | 1.00 | -1.57% |
| SELL | retest2 | 2025-09-25 09:15:00 | 892.30 | 2025-10-03 14:15:00 | 875.25 | STOP_HIT | 1.00 | 1.91% |
| SELL | retest2 | 2025-09-25 13:15:00 | 895.15 | 2025-10-03 14:15:00 | 875.25 | STOP_HIT | 1.00 | 2.22% |
| SELL | retest2 | 2025-09-26 09:15:00 | 880.85 | 2025-10-03 14:15:00 | 875.25 | STOP_HIT | 1.00 | 0.64% |
| BUY | retest2 | 2025-10-17 14:30:00 | 889.95 | 2025-10-20 09:15:00 | 880.75 | STOP_HIT | 1.00 | -1.03% |
| BUY | retest2 | 2025-10-17 15:00:00 | 890.65 | 2025-10-20 09:15:00 | 880.75 | STOP_HIT | 1.00 | -1.11% |
| SELL | retest2 | 2025-10-28 14:45:00 | 859.30 | 2025-10-29 09:15:00 | 773.37 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-12-01 10:45:00 | 562.50 | 2025-12-05 09:15:00 | 534.38 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-01 11:30:00 | 561.80 | 2025-12-05 09:15:00 | 533.71 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-01 10:45:00 | 562.50 | 2025-12-09 12:15:00 | 532.10 | STOP_HIT | 0.50 | 5.40% |
| SELL | retest2 | 2025-12-01 11:30:00 | 561.80 | 2025-12-09 12:15:00 | 532.10 | STOP_HIT | 0.50 | 5.29% |
| BUY | retest2 | 2025-12-16 09:15:00 | 535.00 | 2025-12-17 09:15:00 | 526.60 | STOP_HIT | 1.00 | -1.57% |
| SELL | retest2 | 2025-12-31 15:15:00 | 527.50 | 2026-01-06 15:15:00 | 501.12 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-01 09:30:00 | 527.10 | 2026-01-06 15:15:00 | 500.75 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-31 15:15:00 | 527.50 | 2026-01-07 09:15:00 | 512.35 | STOP_HIT | 0.50 | 2.87% |
| SELL | retest2 | 2026-01-01 09:30:00 | 527.10 | 2026-01-07 09:15:00 | 512.35 | STOP_HIT | 0.50 | 2.80% |
| SELL | retest2 | 2026-02-11 09:30:00 | 348.05 | 2026-02-11 14:15:00 | 356.10 | STOP_HIT | 1.00 | -2.31% |
| SELL | retest2 | 2026-02-26 12:30:00 | 306.40 | 2026-03-05 09:15:00 | 291.08 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-27 09:15:00 | 304.90 | 2026-03-05 09:15:00 | 289.65 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-27 12:15:00 | 305.70 | 2026-03-05 09:15:00 | 290.41 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-02 10:30:00 | 304.75 | 2026-03-05 09:15:00 | 289.51 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-26 12:30:00 | 306.40 | 2026-03-09 09:15:00 | 275.76 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-02-27 09:15:00 | 304.90 | 2026-03-09 09:15:00 | 274.41 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-02-27 12:15:00 | 305.70 | 2026-03-09 09:15:00 | 275.13 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-03-02 10:30:00 | 304.75 | 2026-03-09 09:15:00 | 274.28 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-03-06 09:15:00 | 285.60 | 2026-03-09 09:15:00 | 271.32 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-06 09:15:00 | 285.60 | 2026-03-09 14:15:00 | 282.75 | STOP_HIT | 0.50 | 1.00% |
| BUY | retest2 | 2026-04-02 11:30:00 | 304.85 | 2026-04-06 09:15:00 | 293.40 | STOP_HIT | 1.00 | -3.76% |
| BUY | retest2 | 2026-04-09 15:00:00 | 314.00 | 2026-04-10 09:15:00 | 345.40 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2026-04-22 11:15:00 | 364.90 | 2026-04-23 11:15:00 | 374.10 | STOP_HIT | 1.00 | -2.52% |
| BUY | retest2 | 2026-04-30 12:15:00 | 470.50 | 2026-05-05 11:15:00 | 460.80 | STOP_HIT | 1.00 | -2.06% |
