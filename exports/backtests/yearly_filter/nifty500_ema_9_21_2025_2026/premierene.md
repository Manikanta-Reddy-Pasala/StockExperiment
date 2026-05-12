# Premier Energies Ltd. (PREMIERENE)

## Backtest Summary

- **Window:** 2025-03-13 09:15:00 → 2026-05-11 15:15:00 (1983 bars)
- **Last close:** 1002.50
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 74 |
| ALERT1 | 52 |
| ALERT2 | 50 |
| ALERT2_SKIP | 27 |
| ALERT3 | 140 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 1 |
| ENTRY2 | 71 |
| PARTIAL | 8 |
| TARGET_HIT | 9 |
| STOP_HIT | 63 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 80 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 25 / 55
- **Target hits / Stop hits / Partials:** 9 / 63 / 8
- **Avg / median % per leg:** 0.32% / -1.38%
- **Sum % (uncompounded):** 25.49%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 27 | 2 | 7.4% | 2 | 25 | 0 | -1.34% | -36.0% |
| BUY @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -2.65% | -2.7% |
| BUY @ 3rd Alert (retest2) | 26 | 2 | 7.7% | 2 | 24 | 0 | -1.28% | -33.4% |
| SELL (all) | 53 | 23 | 43.4% | 7 | 38 | 8 | 1.16% | 61.5% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 53 | 23 | 43.4% | 7 | 38 | 8 | 1.16% | 61.5% |
| retest1 (combined) | 1 | 0 | 0.0% | 0 | 1 | 0 | -2.65% | -2.7% |
| retest2 (combined) | 79 | 25 | 31.6% | 9 | 62 | 8 | 0.36% | 28.1% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 10:15:00 | 984.25 | 959.86 | 957.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-13 09:15:00 | 1005.95 | 984.26 | 972.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-14 15:15:00 | 1015.00 | 1015.80 | 1004.18 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-15 09:15:00 | 1010.50 | 1015.80 | 1004.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 15:15:00 | 1089.80 | 1099.94 | 1081.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 09:15:00 | 1086.00 | 1099.94 | 1081.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 09:15:00 | 1096.50 | 1099.26 | 1082.76 | EMA400 retest candle locked (from upside) |

### Cycle 2 — SELL (started 2025-05-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-20 15:15:00 | 1058.00 | 1073.15 | 1074.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-21 11:15:00 | 1051.50 | 1066.62 | 1071.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-21 14:15:00 | 1089.20 | 1068.04 | 1070.50 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-21 14:15:00 | 1089.20 | 1068.04 | 1070.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 14:15:00 | 1089.20 | 1068.04 | 1070.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-21 14:45:00 | 1086.00 | 1068.04 | 1070.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 15:15:00 | 1077.00 | 1069.84 | 1071.09 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-22 09:15:00 | 1070.35 | 1069.84 | 1071.09 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-22 10:15:00 | 1073.95 | 1071.92 | 1071.90 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 3 — BUY (started 2025-05-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-22 10:15:00 | 1073.95 | 1071.92 | 1071.90 | EMA200 above EMA400 |

### Cycle 4 — SELL (started 2025-05-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-23 09:15:00 | 1031.15 | 1067.01 | 1070.13 | EMA200 below EMA400 |

### Cycle 5 — BUY (started 2025-05-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-29 10:15:00 | 1079.50 | 1054.78 | 1052.11 | EMA200 above EMA400 |

### Cycle 6 — SELL (started 2025-06-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-02 12:15:00 | 1057.90 | 1059.85 | 1059.91 | EMA200 below EMA400 |

### Cycle 7 — BUY (started 2025-06-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-02 13:15:00 | 1070.00 | 1061.88 | 1060.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-02 14:15:00 | 1077.70 | 1065.04 | 1062.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-03 12:15:00 | 1063.30 | 1068.79 | 1065.78 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-03 12:15:00 | 1063.30 | 1068.79 | 1065.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 12:15:00 | 1063.30 | 1068.79 | 1065.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-03 13:00:00 | 1063.30 | 1068.79 | 1065.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 13:15:00 | 1095.00 | 1074.03 | 1068.43 | EMA400 retest candle locked (from upside) |

### Cycle 8 — SELL (started 2025-06-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-04 15:15:00 | 1055.20 | 1066.76 | 1068.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-05 13:15:00 | 1051.50 | 1058.23 | 1062.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-06 09:15:00 | 1058.00 | 1055.11 | 1060.09 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-06 09:15:00 | 1058.00 | 1055.11 | 1060.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 09:15:00 | 1058.00 | 1055.11 | 1060.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-06 10:15:00 | 1064.80 | 1055.11 | 1060.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 10:15:00 | 1058.40 | 1055.77 | 1059.93 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-06 12:30:00 | 1054.50 | 1055.83 | 1059.27 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-09 09:15:00 | 1071.70 | 1058.06 | 1059.05 | SL hit (close>static) qty=1.00 sl=1066.00 alert=retest2 |

### Cycle 9 — BUY (started 2025-06-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-09 10:15:00 | 1066.70 | 1059.79 | 1059.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-09 15:15:00 | 1080.00 | 1066.64 | 1063.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-10 14:15:00 | 1087.00 | 1088.55 | 1077.92 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-10 14:30:00 | 1087.40 | 1088.55 | 1077.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 09:15:00 | 1065.80 | 1083.03 | 1077.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-11 09:30:00 | 1065.50 | 1083.03 | 1077.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 10:15:00 | 1069.20 | 1080.27 | 1076.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-11 10:30:00 | 1068.30 | 1080.27 | 1076.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 11:15:00 | 1075.00 | 1079.21 | 1076.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-11 11:30:00 | 1072.00 | 1079.21 | 1076.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 12:15:00 | 1081.60 | 1079.69 | 1076.81 | EMA400 retest candle locked (from upside) |

### Cycle 10 — SELL (started 2025-06-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-11 14:15:00 | 1059.00 | 1072.72 | 1073.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-12 09:15:00 | 1026.40 | 1061.02 | 1068.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-16 10:15:00 | 1017.40 | 1014.28 | 1027.97 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-16 11:00:00 | 1017.40 | 1014.28 | 1027.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 12:15:00 | 1027.10 | 1018.43 | 1027.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 12:45:00 | 1028.50 | 1018.43 | 1027.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 13:15:00 | 1028.30 | 1020.41 | 1027.64 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-17 09:15:00 | 1019.40 | 1024.04 | 1028.16 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-19 12:15:00 | 968.43 | 984.87 | 997.84 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-20 09:15:00 | 1007.50 | 982.62 | 991.89 | SL hit (close>ema200) qty=0.50 sl=982.62 alert=retest2 |

### Cycle 11 — BUY (started 2025-06-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-20 12:15:00 | 1018.20 | 1000.92 | 998.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-24 09:15:00 | 1034.00 | 1021.11 | 1014.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-24 13:15:00 | 1023.90 | 1026.70 | 1019.61 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-24 13:15:00 | 1023.90 | 1026.70 | 1019.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 13:15:00 | 1023.90 | 1026.70 | 1019.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-24 14:00:00 | 1023.90 | 1026.70 | 1019.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 14:15:00 | 1022.90 | 1025.94 | 1019.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-24 15:00:00 | 1022.90 | 1025.94 | 1019.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 15:15:00 | 1021.90 | 1025.13 | 1020.09 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-25 09:15:00 | 1036.10 | 1025.13 | 1020.09 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-25 10:15:00 | 1017.10 | 1023.33 | 1020.13 | SL hit (close<static) qty=1.00 sl=1020.00 alert=retest2 |

### Cycle 12 — SELL (started 2025-06-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-25 13:15:00 | 1010.00 | 1017.36 | 1017.93 | EMA200 below EMA400 |

### Cycle 13 — BUY (started 2025-06-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-26 09:15:00 | 1021.50 | 1018.82 | 1018.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-26 11:15:00 | 1034.10 | 1023.74 | 1020.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-27 12:15:00 | 1037.10 | 1040.61 | 1033.57 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-27 13:00:00 | 1037.10 | 1040.61 | 1033.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 12:15:00 | 1058.10 | 1061.70 | 1053.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-01 13:00:00 | 1058.10 | 1061.70 | 1053.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 13:15:00 | 1063.50 | 1062.06 | 1054.36 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-04 09:15:00 | 1067.10 | 1055.89 | 1055.23 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-04 10:15:00 | 1069.00 | 1057.65 | 1056.09 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-04 12:15:00 | 1048.80 | 1054.85 | 1055.13 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 14 — SELL (started 2025-07-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-04 12:15:00 | 1048.80 | 1054.85 | 1055.13 | EMA200 below EMA400 |

### Cycle 15 — BUY (started 2025-07-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-04 14:15:00 | 1059.80 | 1055.30 | 1055.25 | EMA200 above EMA400 |

### Cycle 16 — SELL (started 2025-07-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-07 09:15:00 | 1048.60 | 1054.07 | 1054.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-07 10:15:00 | 1045.30 | 1052.32 | 1053.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-08 09:15:00 | 1048.80 | 1044.77 | 1048.39 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-08 09:15:00 | 1048.80 | 1044.77 | 1048.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 09:15:00 | 1048.80 | 1044.77 | 1048.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-08 09:30:00 | 1054.00 | 1044.77 | 1048.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 10:15:00 | 1044.80 | 1044.77 | 1048.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-08 11:15:00 | 1051.50 | 1044.77 | 1048.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 11:15:00 | 1045.90 | 1045.00 | 1047.87 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-08 12:15:00 | 1042.10 | 1045.00 | 1047.87 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-08 15:15:00 | 1056.60 | 1050.57 | 1049.80 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 17 — BUY (started 2025-07-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-08 15:15:00 | 1056.60 | 1050.57 | 1049.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-09 09:15:00 | 1072.10 | 1054.87 | 1051.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-11 09:15:00 | 1092.30 | 1100.62 | 1086.19 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-11 10:00:00 | 1092.30 | 1100.62 | 1086.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 10:15:00 | 1080.50 | 1096.59 | 1085.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-11 10:45:00 | 1080.00 | 1096.59 | 1085.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 11:15:00 | 1073.20 | 1091.92 | 1084.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-11 12:00:00 | 1073.20 | 1091.92 | 1084.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 12:15:00 | 1100.20 | 1104.58 | 1100.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-15 13:00:00 | 1100.20 | 1104.58 | 1100.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 13:15:00 | 1081.70 | 1100.00 | 1098.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-15 14:00:00 | 1081.70 | 1100.00 | 1098.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 18 — SELL (started 2025-07-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-15 14:15:00 | 1076.40 | 1095.28 | 1096.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-16 13:15:00 | 1067.90 | 1079.40 | 1086.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-17 10:15:00 | 1080.40 | 1078.70 | 1083.98 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-17 10:15:00 | 1080.40 | 1078.70 | 1083.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 10:15:00 | 1080.40 | 1078.70 | 1083.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-17 10:45:00 | 1078.10 | 1078.70 | 1083.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 11:15:00 | 1087.90 | 1080.54 | 1084.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-17 12:00:00 | 1087.90 | 1080.54 | 1084.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 12:15:00 | 1082.00 | 1080.83 | 1084.12 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-17 14:15:00 | 1080.10 | 1081.26 | 1084.02 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-18 09:30:00 | 1078.20 | 1079.77 | 1082.64 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-18 10:00:00 | 1074.90 | 1079.77 | 1082.64 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-23 12:15:00 | 1073.70 | 1057.85 | 1057.49 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 19 — BUY (started 2025-07-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-23 12:15:00 | 1073.70 | 1057.85 | 1057.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-23 13:15:00 | 1089.90 | 1064.26 | 1060.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-25 10:15:00 | 1085.70 | 1091.52 | 1082.09 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-25 11:00:00 | 1085.70 | 1091.52 | 1082.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 11:15:00 | 1085.60 | 1090.33 | 1082.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-25 12:00:00 | 1085.60 | 1090.33 | 1082.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 12:15:00 | 1083.40 | 1088.95 | 1082.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-25 13:15:00 | 1083.50 | 1088.95 | 1082.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 13:15:00 | 1090.60 | 1089.28 | 1083.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-25 13:30:00 | 1087.00 | 1089.28 | 1083.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 15:15:00 | 1084.70 | 1089.68 | 1084.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-28 09:15:00 | 1069.50 | 1089.68 | 1084.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 09:15:00 | 1070.10 | 1085.76 | 1083.24 | EMA400 retest candle locked (from upside) |

### Cycle 20 — SELL (started 2025-07-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-28 10:15:00 | 1058.90 | 1080.39 | 1081.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-28 13:15:00 | 1048.00 | 1067.19 | 1074.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-29 10:15:00 | 1061.30 | 1061.16 | 1068.61 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-29 10:15:00 | 1061.30 | 1061.16 | 1068.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 10:15:00 | 1061.30 | 1061.16 | 1068.61 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-29 12:00:00 | 1052.20 | 1059.37 | 1067.12 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-29 14:45:00 | 1055.70 | 1055.81 | 1063.41 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-30 09:15:00 | 1081.80 | 1060.33 | 1064.11 | SL hit (close>static) qty=1.00 sl=1076.00 alert=retest2 |

### Cycle 21 — BUY (started 2025-07-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-30 11:15:00 | 1083.00 | 1068.21 | 1067.24 | EMA200 above EMA400 |

### Cycle 22 — SELL (started 2025-07-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-31 09:15:00 | 1042.80 | 1062.57 | 1065.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-01 11:15:00 | 1022.50 | 1037.24 | 1047.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-04 11:15:00 | 1024.70 | 1022.16 | 1032.95 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-04 11:45:00 | 1024.90 | 1022.16 | 1032.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 09:15:00 | 1013.30 | 1020.63 | 1028.22 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-05 10:15:00 | 1011.00 | 1020.63 | 1028.22 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-11 15:15:00 | 1003.30 | 994.24 | 993.71 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 23 — BUY (started 2025-08-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-11 15:15:00 | 1003.30 | 994.24 | 993.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-12 09:15:00 | 1014.30 | 998.25 | 995.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-13 09:15:00 | 994.50 | 1007.72 | 1003.36 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-13 09:15:00 | 994.50 | 1007.72 | 1003.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 09:15:00 | 994.50 | 1007.72 | 1003.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-13 10:00:00 | 994.50 | 1007.72 | 1003.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 10:15:00 | 999.50 | 1006.07 | 1003.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-13 10:30:00 | 993.00 | 1006.07 | 1003.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 11:15:00 | 999.00 | 1004.66 | 1002.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-13 11:45:00 | 997.30 | 1004.66 | 1002.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 12:15:00 | 1002.20 | 1004.17 | 1002.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-13 12:45:00 | 1004.00 | 1004.17 | 1002.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 13:15:00 | 1001.20 | 1003.57 | 1002.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-13 13:30:00 | 999.50 | 1003.57 | 1002.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 14:15:00 | 999.00 | 1002.66 | 1002.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-13 15:00:00 | 999.00 | 1002.66 | 1002.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 15:15:00 | 1001.00 | 1002.33 | 1002.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-14 09:15:00 | 992.40 | 1002.33 | 1002.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 24 — SELL (started 2025-08-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-14 09:15:00 | 994.60 | 1000.78 | 1001.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-14 14:15:00 | 984.50 | 994.06 | 997.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-18 09:15:00 | 1025.00 | 998.54 | 998.86 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-18 09:15:00 | 1025.00 | 998.54 | 998.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 09:15:00 | 1025.00 | 998.54 | 998.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-18 10:00:00 | 1025.00 | 998.54 | 998.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 25 — BUY (started 2025-08-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-18 10:15:00 | 1020.10 | 1002.85 | 1000.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-19 12:15:00 | 1026.60 | 1012.68 | 1006.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-21 10:15:00 | 1033.10 | 1034.53 | 1026.03 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-21 11:00:00 | 1033.10 | 1034.53 | 1026.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 13:15:00 | 1023.00 | 1032.74 | 1027.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-21 14:00:00 | 1023.00 | 1032.74 | 1027.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 14:15:00 | 1023.30 | 1030.85 | 1026.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-21 15:00:00 | 1023.30 | 1030.85 | 1026.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 15:15:00 | 1026.00 | 1029.88 | 1026.87 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-22 09:15:00 | 1028.60 | 1029.88 | 1026.87 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-22 11:15:00 | 1018.20 | 1025.41 | 1025.40 | SL hit (close<static) qty=1.00 sl=1020.70 alert=retest2 |

### Cycle 26 — SELL (started 2025-08-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-22 12:15:00 | 1014.60 | 1023.25 | 1024.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-22 13:15:00 | 1012.60 | 1021.12 | 1023.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-25 14:15:00 | 1003.80 | 1002.31 | 1010.30 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-25 15:00:00 | 1003.80 | 1002.31 | 1010.30 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 10:15:00 | 1003.00 | 1001.36 | 1007.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-26 11:00:00 | 1003.00 | 1001.36 | 1007.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 11:15:00 | 1006.70 | 1002.43 | 1007.73 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-26 13:00:00 | 999.00 | 1001.75 | 1006.94 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-28 09:15:00 | 1017.90 | 1006.39 | 1007.49 | SL hit (close>static) qty=1.00 sl=1017.80 alert=retest2 |

### Cycle 27 — BUY (started 2025-08-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-28 11:15:00 | 1013.00 | 1008.66 | 1008.38 | EMA200 above EMA400 |

### Cycle 28 — SELL (started 2025-08-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-28 14:15:00 | 1003.30 | 1008.35 | 1008.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-29 13:15:00 | 999.10 | 1004.73 | 1006.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-01 09:15:00 | 1019.90 | 1004.30 | 1005.50 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-01 09:15:00 | 1019.90 | 1004.30 | 1005.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 09:15:00 | 1019.90 | 1004.30 | 1005.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 09:45:00 | 1013.00 | 1004.30 | 1005.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 29 — BUY (started 2025-09-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-01 10:15:00 | 1016.40 | 1006.72 | 1006.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-01 12:15:00 | 1031.20 | 1013.42 | 1009.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-02 12:15:00 | 1033.15 | 1033.72 | 1024.01 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-02 13:00:00 | 1033.15 | 1033.72 | 1024.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 13:15:00 | 1025.20 | 1032.02 | 1024.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-02 14:00:00 | 1025.20 | 1032.02 | 1024.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 14:15:00 | 1020.00 | 1029.61 | 1023.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-02 15:00:00 | 1020.00 | 1029.61 | 1023.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 15:15:00 | 1021.00 | 1027.89 | 1023.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-03 09:15:00 | 1030.50 | 1027.89 | 1023.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 10:15:00 | 1026.85 | 1027.79 | 1024.22 | EMA400 retest candle locked (from upside) |

### Cycle 30 — SELL (started 2025-09-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-04 11:15:00 | 1020.70 | 1023.22 | 1023.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-04 12:15:00 | 1014.55 | 1021.48 | 1022.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-08 09:15:00 | 1011.65 | 1005.84 | 1010.82 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-08 09:15:00 | 1011.65 | 1005.84 | 1010.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 09:15:00 | 1011.65 | 1005.84 | 1010.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-08 09:30:00 | 1015.95 | 1005.84 | 1010.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 10:15:00 | 1000.00 | 1004.67 | 1009.84 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-08 14:30:00 | 998.00 | 1000.93 | 1006.28 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-09 09:45:00 | 996.05 | 999.20 | 1004.52 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-09 12:00:00 | 997.50 | 999.03 | 1003.53 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-09 12:30:00 | 998.00 | 998.53 | 1002.90 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 09:15:00 | 1014.65 | 1001.08 | 1002.56 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-09-10 09:15:00 | 1014.65 | 1001.08 | 1002.56 | SL hit (close>static) qty=1.00 sl=1013.90 alert=retest2 |

### Cycle 31 — BUY (started 2025-09-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-10 12:15:00 | 1011.05 | 1004.60 | 1003.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-10 13:15:00 | 1017.75 | 1007.23 | 1005.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-12 12:15:00 | 1050.20 | 1051.83 | 1039.26 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-12 13:00:00 | 1050.20 | 1051.83 | 1039.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 14:15:00 | 1052.60 | 1051.08 | 1041.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-12 14:30:00 | 1042.00 | 1051.08 | 1041.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 09:15:00 | 1075.80 | 1058.53 | 1051.17 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-16 14:45:00 | 1079.80 | 1068.21 | 1059.22 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-17 09:15:00 | 1084.30 | 1070.27 | 1060.98 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-17 10:30:00 | 1084.45 | 1073.74 | 1064.23 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-17 12:45:00 | 1082.20 | 1075.19 | 1066.60 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 09:15:00 | 1069.90 | 1074.58 | 1069.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-18 10:00:00 | 1069.90 | 1074.58 | 1069.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 10:15:00 | 1066.10 | 1072.89 | 1068.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-18 11:00:00 | 1066.10 | 1072.89 | 1068.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 11:15:00 | 1055.95 | 1069.50 | 1067.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-18 11:45:00 | 1054.80 | 1069.50 | 1067.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-09-18 13:15:00 | 1053.80 | 1064.23 | 1065.44 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 32 — SELL (started 2025-09-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-18 13:15:00 | 1053.80 | 1064.23 | 1065.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-19 09:15:00 | 1051.45 | 1058.46 | 1062.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-24 09:15:00 | 1030.50 | 1020.20 | 1029.69 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-24 09:15:00 | 1030.50 | 1020.20 | 1029.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 09:15:00 | 1030.50 | 1020.20 | 1029.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-24 10:00:00 | 1030.50 | 1020.20 | 1029.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 10:15:00 | 1035.00 | 1023.16 | 1030.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-24 11:00:00 | 1035.00 | 1023.16 | 1030.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 11:15:00 | 1064.35 | 1031.40 | 1033.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-24 12:00:00 | 1064.35 | 1031.40 | 1033.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 33 — BUY (started 2025-09-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-24 13:15:00 | 1042.00 | 1035.66 | 1035.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-25 14:15:00 | 1049.00 | 1041.05 | 1038.90 | Break + close above crossover candle high |

### Cycle 34 — SELL (started 2025-09-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-26 09:15:00 | 1016.50 | 1037.51 | 1037.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-26 11:15:00 | 1012.30 | 1029.18 | 1033.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-26 14:15:00 | 1027.85 | 1023.51 | 1029.50 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-26 14:15:00 | 1027.85 | 1023.51 | 1029.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 14:15:00 | 1027.85 | 1023.51 | 1029.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-26 15:00:00 | 1027.85 | 1023.51 | 1029.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 15:15:00 | 1022.20 | 1023.24 | 1028.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-29 09:15:00 | 1019.70 | 1023.24 | 1028.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 09:15:00 | 1019.80 | 1022.56 | 1028.02 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-29 11:30:00 | 1011.50 | 1019.30 | 1025.52 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-29 12:00:00 | 1011.55 | 1019.30 | 1025.52 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-29 12:30:00 | 1010.00 | 1018.64 | 1024.66 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-30 11:15:00 | 1011.80 | 1017.87 | 1022.07 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 11:15:00 | 1018.40 | 1017.98 | 1021.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-30 12:00:00 | 1018.40 | 1017.98 | 1021.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 12:15:00 | 1015.00 | 1017.38 | 1021.13 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-30 13:30:00 | 1012.85 | 1016.36 | 1020.32 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-30 14:15:00 | 1023.50 | 1017.79 | 1020.61 | SL hit (close>static) qty=1.00 sl=1022.00 alert=retest2 |

### Cycle 35 — BUY (started 2025-10-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-03 09:15:00 | 1032.30 | 1021.30 | 1020.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-03 10:15:00 | 1044.70 | 1025.98 | 1022.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-06 09:15:00 | 1020.00 | 1030.27 | 1027.08 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-06 09:15:00 | 1020.00 | 1030.27 | 1027.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 09:15:00 | 1020.00 | 1030.27 | 1027.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-06 09:30:00 | 1020.00 | 1030.27 | 1027.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 10:15:00 | 1020.00 | 1028.22 | 1026.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-06 10:30:00 | 1018.30 | 1028.22 | 1026.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 36 — SELL (started 2025-10-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-06 12:15:00 | 1018.50 | 1024.69 | 1025.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-07 10:15:00 | 1017.00 | 1020.96 | 1022.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-08 14:15:00 | 1018.40 | 1016.67 | 1018.65 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-08 14:15:00 | 1018.40 | 1016.67 | 1018.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 14:15:00 | 1018.40 | 1016.67 | 1018.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-08 15:00:00 | 1018.40 | 1016.67 | 1018.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 15:15:00 | 1018.80 | 1017.09 | 1018.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-09 09:15:00 | 1020.50 | 1017.09 | 1018.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 09:15:00 | 1017.10 | 1017.10 | 1018.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-09 09:30:00 | 1019.00 | 1017.10 | 1018.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 10:15:00 | 1016.30 | 1016.94 | 1018.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-09 10:30:00 | 1017.10 | 1016.94 | 1018.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 11:15:00 | 1017.70 | 1017.09 | 1018.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-09 12:00:00 | 1017.70 | 1017.09 | 1018.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 12:15:00 | 1017.80 | 1017.23 | 1018.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-09 12:45:00 | 1019.70 | 1017.23 | 1018.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 13:15:00 | 1013.90 | 1016.56 | 1017.83 | EMA400 retest candle locked (from downside) |

### Cycle 37 — BUY (started 2025-10-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-09 14:15:00 | 1027.90 | 1018.83 | 1018.74 | EMA200 above EMA400 |

### Cycle 38 — SELL (started 2025-10-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-13 12:15:00 | 1020.50 | 1020.73 | 1020.73 | EMA200 below EMA400 |

### Cycle 39 — BUY (started 2025-10-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-13 14:15:00 | 1030.10 | 1022.53 | 1021.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-14 09:15:00 | 1047.40 | 1029.02 | 1024.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-15 14:15:00 | 1059.50 | 1061.64 | 1049.65 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-15 15:00:00 | 1059.50 | 1061.64 | 1049.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 11:15:00 | 1051.00 | 1058.90 | 1052.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-16 12:00:00 | 1051.00 | 1058.90 | 1052.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 12:15:00 | 1053.20 | 1057.76 | 1052.23 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-16 14:45:00 | 1066.80 | 1059.00 | 1053.71 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-17 11:00:00 | 1058.40 | 1060.75 | 1055.99 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-17 14:15:00 | 1069.70 | 1057.39 | 1055.50 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-20 11:15:00 | 1058.00 | 1058.24 | 1056.74 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 11:15:00 | 1056.50 | 1057.90 | 1056.72 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-10-20 13:15:00 | 1038.80 | 1053.77 | 1055.03 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 40 — SELL (started 2025-10-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-20 13:15:00 | 1038.80 | 1053.77 | 1055.03 | EMA200 below EMA400 |

### Cycle 41 — BUY (started 2025-10-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-23 09:15:00 | 1064.30 | 1056.72 | 1055.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-23 10:15:00 | 1077.00 | 1060.78 | 1057.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-23 13:15:00 | 1066.50 | 1066.65 | 1061.68 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-23 14:00:00 | 1066.50 | 1066.65 | 1061.68 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 14:15:00 | 1055.10 | 1064.34 | 1061.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-23 15:00:00 | 1055.10 | 1064.34 | 1061.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 15:15:00 | 1069.00 | 1065.27 | 1061.80 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-27 09:15:00 | 1100.40 | 1066.99 | 1064.41 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-28 12:45:00 | 1078.20 | 1084.45 | 1080.67 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-29 09:30:00 | 1087.30 | 1085.95 | 1082.70 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-31 11:15:00 | 1077.00 | 1086.84 | 1088.00 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 42 — SELL (started 2025-10-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-31 11:15:00 | 1077.00 | 1086.84 | 1088.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-03 10:15:00 | 1068.20 | 1079.57 | 1083.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-07 11:15:00 | 1011.90 | 1009.60 | 1027.47 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-07 12:00:00 | 1011.90 | 1009.60 | 1027.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 14:15:00 | 1023.70 | 1012.79 | 1024.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-07 15:00:00 | 1023.70 | 1012.79 | 1024.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 15:15:00 | 1031.80 | 1016.59 | 1025.17 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-10 09:15:00 | 1014.50 | 1016.59 | 1025.17 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-10 15:00:00 | 1016.80 | 1012.02 | 1018.71 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-11 15:15:00 | 1019.00 | 1014.64 | 1016.81 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-12 09:15:00 | 1041.90 | 1020.79 | 1019.27 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 43 — BUY (started 2025-11-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-12 09:15:00 | 1041.90 | 1020.79 | 1019.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-12 11:15:00 | 1052.00 | 1030.91 | 1024.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-13 10:15:00 | 1049.70 | 1049.96 | 1038.59 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-13 10:45:00 | 1047.80 | 1049.96 | 1038.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 09:15:00 | 1032.50 | 1047.81 | 1042.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-14 10:00:00 | 1032.50 | 1047.81 | 1042.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 10:15:00 | 1017.00 | 1041.65 | 1040.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-14 10:45:00 | 1017.00 | 1041.65 | 1040.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 44 — SELL (started 2025-11-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-14 11:15:00 | 1009.10 | 1035.14 | 1037.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-14 13:15:00 | 1004.10 | 1025.16 | 1032.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-17 14:15:00 | 1011.40 | 1009.81 | 1018.11 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-17 14:15:00 | 1011.40 | 1009.81 | 1018.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 14:15:00 | 1011.40 | 1009.81 | 1018.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-17 15:00:00 | 1011.40 | 1009.81 | 1018.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 09:15:00 | 997.60 | 994.13 | 999.35 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-20 12:00:00 | 995.00 | 994.70 | 998.73 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-20 15:15:00 | 994.90 | 995.87 | 998.28 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-27 12:15:00 | 985.30 | 981.71 | 981.49 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 45 — BUY (started 2025-11-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-27 12:15:00 | 985.30 | 981.71 | 981.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-27 14:15:00 | 987.60 | 983.50 | 982.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-28 09:15:00 | 980.60 | 983.56 | 982.64 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-28 09:15:00 | 980.60 | 983.56 | 982.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 09:15:00 | 980.60 | 983.56 | 982.64 | EMA400 retest candle locked (from upside) |

### Cycle 46 — SELL (started 2025-11-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-28 11:15:00 | 978.50 | 981.64 | 981.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-28 14:15:00 | 975.60 | 979.46 | 980.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-01 15:15:00 | 972.80 | 971.69 | 975.09 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-02 09:15:00 | 972.50 | 971.69 | 975.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 09:15:00 | 968.35 | 971.03 | 974.48 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-02 13:45:00 | 964.70 | 968.93 | 972.38 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-02 15:15:00 | 963.00 | 968.84 | 972.02 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-03 10:15:00 | 965.00 | 967.47 | 970.79 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-03 12:15:00 | 964.50 | 966.22 | 969.57 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 12:15:00 | 964.30 | 965.83 | 969.09 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-04 09:45:00 | 962.00 | 966.38 | 968.47 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-05 09:15:00 | 916.47 | 945.90 | 956.54 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-05 09:15:00 | 916.75 | 945.90 | 956.54 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-05 10:15:00 | 914.85 | 939.40 | 952.62 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-05 10:15:00 | 916.27 | 939.40 | 952.62 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-05 10:15:00 | 913.90 | 939.40 | 952.62 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-12-08 13:15:00 | 868.23 | 888.41 | 911.72 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 47 — BUY (started 2025-12-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-11 15:15:00 | 914.45 | 875.31 | 872.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-12 09:15:00 | 918.80 | 884.01 | 876.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-15 12:15:00 | 908.90 | 909.38 | 898.83 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-15 13:00:00 | 908.90 | 909.38 | 898.83 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 14:15:00 | 909.75 | 918.49 | 911.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 15:00:00 | 909.75 | 918.49 | 911.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 15:15:00 | 906.40 | 916.07 | 911.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-17 09:15:00 | 903.00 | 916.07 | 911.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 10:15:00 | 901.70 | 912.38 | 910.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-17 11:00:00 | 901.70 | 912.38 | 910.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 11:15:00 | 901.60 | 910.23 | 909.50 | EMA400 retest candle locked (from upside) |

### Cycle 48 — SELL (started 2025-12-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-17 12:15:00 | 904.05 | 908.99 | 909.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-18 10:15:00 | 900.50 | 905.01 | 906.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-19 09:15:00 | 900.00 | 898.41 | 902.16 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-19 09:15:00 | 900.00 | 898.41 | 902.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 09:15:00 | 900.00 | 898.41 | 902.16 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-19 10:15:00 | 895.80 | 898.41 | 902.16 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-19 10:45:00 | 891.40 | 896.53 | 900.96 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-24 10:15:00 | 900.00 | 888.99 | 888.68 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 49 — BUY (started 2025-12-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-24 10:15:00 | 900.00 | 888.99 | 888.68 | EMA200 above EMA400 |

### Cycle 50 — SELL (started 2025-12-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-24 14:15:00 | 885.90 | 888.22 | 888.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-26 12:15:00 | 881.00 | 885.57 | 887.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-30 14:15:00 | 852.30 | 851.89 | 861.21 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-30 15:00:00 | 852.30 | 851.89 | 861.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 09:15:00 | 855.70 | 852.19 | 859.70 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-31 11:00:00 | 852.55 | 852.26 | 859.05 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-31 13:15:00 | 852.45 | 853.06 | 858.24 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-05 09:15:00 | 809.92 | 837.32 | 844.78 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-05 09:15:00 | 809.83 | 837.32 | 844.78 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-01-06 09:15:00 | 767.29 | 797.55 | 817.76 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 51 — BUY (started 2026-01-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-13 10:15:00 | 741.50 | 738.93 | 738.89 | EMA200 above EMA400 |

### Cycle 52 — SELL (started 2026-01-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-14 12:15:00 | 731.15 | 738.34 | 739.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-14 13:15:00 | 725.65 | 735.80 | 738.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-16 09:15:00 | 749.50 | 735.98 | 737.31 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-16 09:15:00 | 749.50 | 735.98 | 737.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 09:15:00 | 749.50 | 735.98 | 737.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-16 10:00:00 | 749.50 | 735.98 | 737.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 53 — BUY (started 2026-01-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-16 10:15:00 | 749.75 | 738.74 | 738.44 | EMA200 above EMA400 |

### Cycle 54 — SELL (started 2026-01-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-19 12:15:00 | 732.55 | 739.09 | 739.83 | EMA200 below EMA400 |

### Cycle 55 — BUY (started 2026-01-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-20 11:15:00 | 742.00 | 740.20 | 739.96 | EMA200 above EMA400 |

### Cycle 56 — SELL (started 2026-01-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-20 12:15:00 | 738.15 | 739.79 | 739.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-20 13:15:00 | 726.60 | 737.15 | 738.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-22 09:15:00 | 728.60 | 721.10 | 727.26 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-22 09:15:00 | 728.60 | 721.10 | 727.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 09:15:00 | 728.60 | 721.10 | 727.26 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-22 10:30:00 | 724.50 | 720.94 | 726.63 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-22 13:15:00 | 722.80 | 722.78 | 726.50 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-22 15:15:00 | 747.00 | 730.92 | 729.52 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 57 — BUY (started 2026-01-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-22 15:15:00 | 747.00 | 730.92 | 729.52 | EMA200 above EMA400 |

### Cycle 58 — SELL (started 2026-01-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-23 09:15:00 | 709.05 | 726.55 | 727.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-23 14:15:00 | 681.95 | 708.76 | 717.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-27 09:15:00 | 716.70 | 705.74 | 714.53 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-27 09:15:00 | 716.70 | 705.74 | 714.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 09:15:00 | 716.70 | 705.74 | 714.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-27 09:30:00 | 707.90 | 705.74 | 714.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 10:15:00 | 712.40 | 707.08 | 714.34 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-28 10:15:00 | 703.20 | 708.61 | 712.27 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-28 14:00:00 | 705.75 | 707.72 | 710.63 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-28 15:15:00 | 719.95 | 711.25 | 711.80 | SL hit (close>static) qty=1.00 sl=716.85 alert=retest2 |

### Cycle 59 — BUY (started 2026-01-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-29 14:15:00 | 713.30 | 711.56 | 711.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-30 09:15:00 | 725.80 | 714.78 | 712.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-30 14:15:00 | 717.95 | 718.90 | 716.01 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-30 15:00:00 | 717.95 | 718.90 | 716.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 11:15:00 | 711.45 | 719.48 | 717.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 12:00:00 | 711.45 | 719.48 | 717.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 12:15:00 | 703.75 | 716.33 | 716.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 12:30:00 | 699.45 | 716.33 | 716.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 60 — SELL (started 2026-02-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-01 13:15:00 | 698.80 | 712.82 | 714.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-01 14:15:00 | 690.00 | 708.26 | 712.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-02 09:15:00 | 712.90 | 706.86 | 710.90 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-02 09:15:00 | 712.90 | 706.86 | 710.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 09:15:00 | 712.90 | 706.86 | 710.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-02 09:45:00 | 715.45 | 706.86 | 710.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 10:15:00 | 710.05 | 707.50 | 710.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-02 10:45:00 | 710.65 | 707.50 | 710.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 11:15:00 | 717.05 | 709.41 | 711.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-02 11:30:00 | 716.05 | 709.41 | 711.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 12:15:00 | 722.20 | 711.97 | 712.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-02 13:00:00 | 722.20 | 711.97 | 712.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 61 — BUY (started 2026-02-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-02 13:15:00 | 722.00 | 713.97 | 713.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-02 14:15:00 | 734.70 | 718.12 | 715.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-04 09:15:00 | 774.85 | 778.64 | 757.22 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-04 09:30:00 | 775.90 | 778.64 | 757.22 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 09:15:00 | 779.50 | 781.82 | 769.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-05 09:45:00 | 770.90 | 781.82 | 769.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 13:15:00 | 772.55 | 778.33 | 771.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-05 14:00:00 | 772.55 | 778.33 | 771.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 14:15:00 | 791.00 | 780.86 | 773.63 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-05 15:15:00 | 793.70 | 780.86 | 773.63 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-06 10:00:00 | 791.50 | 785.04 | 776.91 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-06 11:00:00 | 791.50 | 786.33 | 778.24 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-06 14:30:00 | 791.60 | 787.85 | 781.44 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 09:15:00 | 789.00 | 810.83 | 804.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-11 09:30:00 | 787.55 | 810.83 | 804.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 10:15:00 | 786.30 | 805.92 | 802.71 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-02-11 12:15:00 | 782.75 | 797.72 | 799.32 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 62 — SELL (started 2026-02-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-11 12:15:00 | 782.75 | 797.72 | 799.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-12 09:15:00 | 771.40 | 786.43 | 792.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-16 09:15:00 | 762.50 | 761.30 | 769.87 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-17 09:15:00 | 765.70 | 763.03 | 766.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 09:15:00 | 765.70 | 763.03 | 766.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 09:45:00 | 765.60 | 763.03 | 766.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 09:15:00 | 754.30 | 759.74 | 763.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-18 09:30:00 | 759.90 | 759.74 | 763.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 10:15:00 | 759.30 | 759.65 | 762.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-18 11:00:00 | 759.30 | 759.65 | 762.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 11:15:00 | 760.75 | 759.87 | 762.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-18 11:45:00 | 761.25 | 759.87 | 762.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 13:15:00 | 764.00 | 760.73 | 762.51 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-19 09:15:00 | 758.65 | 761.84 | 762.74 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-19 13:00:00 | 759.65 | 756.89 | 759.65 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-20 09:15:00 | 771.00 | 758.08 | 759.19 | SL hit (close>static) qty=1.00 sl=768.40 alert=retest2 |

### Cycle 63 — BUY (started 2026-02-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-20 10:15:00 | 769.15 | 760.30 | 760.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-20 13:15:00 | 774.20 | 765.89 | 762.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-20 15:15:00 | 765.25 | 766.42 | 763.73 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-23 09:15:00 | 771.00 | 766.42 | 763.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 10:15:00 | 762.60 | 765.56 | 763.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-23 11:00:00 | 762.60 | 765.56 | 763.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 11:15:00 | 766.00 | 765.65 | 764.00 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-23 13:00:00 | 767.70 | 766.06 | 764.34 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-23 13:45:00 | 769.60 | 767.02 | 764.93 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-24 13:00:00 | 769.20 | 768.68 | 767.07 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-24 14:30:00 | 768.00 | 768.91 | 767.40 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-25 09:15:00 | 735.20 | 764.15 | 765.62 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 64 — SELL (started 2026-02-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-25 09:15:00 | 735.20 | 764.15 | 765.62 | EMA200 below EMA400 |

### Cycle 65 — BUY (started 2026-03-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-06 09:15:00 | 725.75 | 714.99 | 713.90 | EMA200 above EMA400 |

### Cycle 66 — SELL (started 2026-03-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-09 11:15:00 | 706.55 | 715.19 | 715.65 | EMA200 below EMA400 |

### Cycle 67 — BUY (started 2026-03-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-09 15:15:00 | 719.05 | 715.93 | 715.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-10 09:15:00 | 728.65 | 718.47 | 716.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-13 09:15:00 | 776.75 | 782.18 | 769.63 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-13 10:45:00 | 796.65 | 785.33 | 772.20 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-13 13:15:00 | 776.70 | 785.11 | 775.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-13 14:00:00 | 776.70 | 785.11 | 775.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-13 14:15:00 | 782.40 | 784.57 | 776.21 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-03-16 10:15:00 | 775.50 | 781.88 | 776.96 | SL hit (close<ema400) qty=1.00 sl=776.96 alert=retest1 |

### Cycle 68 — SELL (started 2026-03-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-30 12:15:00 | 884.35 | 892.31 | 892.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 13:15:00 | 882.00 | 890.24 | 891.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-30 14:15:00 | 890.95 | 890.39 | 891.36 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-30 14:15:00 | 890.95 | 890.39 | 891.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-30 14:15:00 | 890.95 | 890.39 | 891.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-30 15:00:00 | 890.95 | 890.39 | 891.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 69 — BUY (started 2026-03-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-30 15:15:00 | 898.75 | 892.06 | 892.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-01 09:15:00 | 906.10 | 894.87 | 893.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-02 09:15:00 | 911.70 | 918.81 | 908.72 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-02 09:15:00 | 911.70 | 918.81 | 908.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 09:15:00 | 911.70 | 918.81 | 908.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-02 10:00:00 | 911.70 | 918.81 | 908.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 10:15:00 | 901.15 | 915.28 | 908.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-02 10:45:00 | 901.00 | 915.28 | 908.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 11:15:00 | 901.80 | 912.58 | 907.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-02 12:15:00 | 902.30 | 912.58 | 907.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 12:15:00 | 904.00 | 910.86 | 907.15 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-02 13:30:00 | 909.25 | 911.36 | 907.71 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2026-04-15 10:15:00 | 1000.18 | 983.17 | 974.17 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 70 — SELL (started 2026-04-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-21 11:15:00 | 1000.00 | 1008.53 | 1008.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-21 13:15:00 | 996.00 | 1004.51 | 1006.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-23 09:15:00 | 1006.00 | 996.16 | 999.09 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-23 09:15:00 | 1006.00 | 996.16 | 999.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-23 09:15:00 | 1006.00 | 996.16 | 999.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-23 10:00:00 | 1006.00 | 996.16 | 999.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-23 10:15:00 | 1001.25 | 997.18 | 999.29 | EMA400 retest candle locked (from downside) |

### Cycle 71 — BUY (started 2026-04-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-23 13:15:00 | 1005.65 | 1001.43 | 1000.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-24 09:15:00 | 1014.45 | 1003.89 | 1002.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-24 14:15:00 | 1005.80 | 1008.10 | 1005.22 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-24 14:15:00 | 1005.80 | 1008.10 | 1005.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 14:15:00 | 1005.80 | 1008.10 | 1005.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-24 15:00:00 | 1005.80 | 1008.10 | 1005.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 15:15:00 | 1012.00 | 1008.88 | 1005.83 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-27 09:15:00 | 1022.20 | 1008.88 | 1005.83 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-30 09:15:00 | 1003.75 | 1028.73 | 1026.99 | SL hit (close<static) qty=1.00 sl=1004.00 alert=retest2 |

### Cycle 72 — SELL (started 2026-04-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-30 10:15:00 | 1000.60 | 1023.11 | 1024.59 | EMA200 below EMA400 |

### Cycle 73 — BUY (started 2026-05-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-04 11:15:00 | 1033.70 | 1022.52 | 1021.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-05 14:15:00 | 1044.20 | 1035.77 | 1030.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-06 10:15:00 | 1035.00 | 1037.58 | 1033.11 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-05-06 11:00:00 | 1035.00 | 1037.58 | 1033.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 11:15:00 | 1033.00 | 1036.67 | 1033.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-06 11:30:00 | 1033.60 | 1036.67 | 1033.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 12:15:00 | 1032.70 | 1035.87 | 1033.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-06 13:00:00 | 1032.70 | 1035.87 | 1033.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 13:15:00 | 1029.60 | 1034.62 | 1032.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-06 13:30:00 | 1029.50 | 1034.62 | 1032.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 14:15:00 | 1033.50 | 1034.39 | 1032.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-06 14:30:00 | 1029.80 | 1034.39 | 1032.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 15:15:00 | 1030.90 | 1033.70 | 1032.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-07 09:15:00 | 1029.20 | 1033.70 | 1032.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 74 — SELL (started 2026-05-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-07 09:15:00 | 1015.10 | 1029.98 | 1031.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-05-08 14:15:00 | 1009.90 | 1018.59 | 1022.63 | Break + close below crossover candle low |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-05-22 09:15:00 | 1070.35 | 2025-05-22 10:15:00 | 1073.95 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest2 | 2025-06-06 12:30:00 | 1054.50 | 2025-06-09 09:15:00 | 1071.70 | STOP_HIT | 1.00 | -1.63% |
| SELL | retest2 | 2025-06-17 09:15:00 | 1019.40 | 2025-06-19 12:15:00 | 968.43 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-06-17 09:15:00 | 1019.40 | 2025-06-20 09:15:00 | 1007.50 | STOP_HIT | 0.50 | 1.17% |
| SELL | retest2 | 2025-06-20 12:00:00 | 1021.20 | 2025-06-20 12:15:00 | 1018.20 | STOP_HIT | 1.00 | 0.29% |
| BUY | retest2 | 2025-06-25 09:15:00 | 1036.10 | 2025-06-25 10:15:00 | 1017.10 | STOP_HIT | 1.00 | -1.83% |
| BUY | retest2 | 2025-07-04 09:15:00 | 1067.10 | 2025-07-04 12:15:00 | 1048.80 | STOP_HIT | 1.00 | -1.71% |
| BUY | retest2 | 2025-07-04 10:15:00 | 1069.00 | 2025-07-04 12:15:00 | 1048.80 | STOP_HIT | 1.00 | -1.89% |
| SELL | retest2 | 2025-07-08 12:15:00 | 1042.10 | 2025-07-08 15:15:00 | 1056.60 | STOP_HIT | 1.00 | -1.39% |
| SELL | retest2 | 2025-07-17 14:15:00 | 1080.10 | 2025-07-23 12:15:00 | 1073.70 | STOP_HIT | 1.00 | 0.59% |
| SELL | retest2 | 2025-07-18 09:30:00 | 1078.20 | 2025-07-23 12:15:00 | 1073.70 | STOP_HIT | 1.00 | 0.42% |
| SELL | retest2 | 2025-07-18 10:00:00 | 1074.90 | 2025-07-23 12:15:00 | 1073.70 | STOP_HIT | 1.00 | 0.11% |
| SELL | retest2 | 2025-07-29 12:00:00 | 1052.20 | 2025-07-30 09:15:00 | 1081.80 | STOP_HIT | 1.00 | -2.81% |
| SELL | retest2 | 2025-07-29 14:45:00 | 1055.70 | 2025-07-30 09:15:00 | 1081.80 | STOP_HIT | 1.00 | -2.47% |
| SELL | retest2 | 2025-08-05 10:15:00 | 1011.00 | 2025-08-11 15:15:00 | 1003.30 | STOP_HIT | 1.00 | 0.76% |
| BUY | retest2 | 2025-08-22 09:15:00 | 1028.60 | 2025-08-22 11:15:00 | 1018.20 | STOP_HIT | 1.00 | -1.01% |
| SELL | retest2 | 2025-08-26 13:00:00 | 999.00 | 2025-08-28 09:15:00 | 1017.90 | STOP_HIT | 1.00 | -1.89% |
| SELL | retest2 | 2025-08-28 09:30:00 | 1000.90 | 2025-08-28 11:15:00 | 1013.00 | STOP_HIT | 1.00 | -1.21% |
| SELL | retest2 | 2025-09-08 14:30:00 | 998.00 | 2025-09-10 09:15:00 | 1014.65 | STOP_HIT | 1.00 | -1.67% |
| SELL | retest2 | 2025-09-09 09:45:00 | 996.05 | 2025-09-10 09:15:00 | 1014.65 | STOP_HIT | 1.00 | -1.87% |
| SELL | retest2 | 2025-09-09 12:00:00 | 997.50 | 2025-09-10 09:15:00 | 1014.65 | STOP_HIT | 1.00 | -1.72% |
| SELL | retest2 | 2025-09-09 12:30:00 | 998.00 | 2025-09-10 09:15:00 | 1014.65 | STOP_HIT | 1.00 | -1.67% |
| SELL | retest2 | 2025-09-10 11:15:00 | 1006.80 | 2025-09-10 12:15:00 | 1011.05 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest2 | 2025-09-16 14:45:00 | 1079.80 | 2025-09-18 13:15:00 | 1053.80 | STOP_HIT | 1.00 | -2.41% |
| BUY | retest2 | 2025-09-17 09:15:00 | 1084.30 | 2025-09-18 13:15:00 | 1053.80 | STOP_HIT | 1.00 | -2.81% |
| BUY | retest2 | 2025-09-17 10:30:00 | 1084.45 | 2025-09-18 13:15:00 | 1053.80 | STOP_HIT | 1.00 | -2.83% |
| BUY | retest2 | 2025-09-17 12:45:00 | 1082.20 | 2025-09-18 13:15:00 | 1053.80 | STOP_HIT | 1.00 | -2.62% |
| SELL | retest2 | 2025-09-29 11:30:00 | 1011.50 | 2025-09-30 14:15:00 | 1023.50 | STOP_HIT | 1.00 | -1.19% |
| SELL | retest2 | 2025-09-29 12:00:00 | 1011.55 | 2025-10-03 09:15:00 | 1032.30 | STOP_HIT | 1.00 | -2.05% |
| SELL | retest2 | 2025-09-29 12:30:00 | 1010.00 | 2025-10-03 09:15:00 | 1032.30 | STOP_HIT | 1.00 | -2.21% |
| SELL | retest2 | 2025-09-30 11:15:00 | 1011.80 | 2025-10-03 09:15:00 | 1032.30 | STOP_HIT | 1.00 | -2.03% |
| SELL | retest2 | 2025-09-30 13:30:00 | 1012.85 | 2025-10-03 09:15:00 | 1032.30 | STOP_HIT | 1.00 | -1.92% |
| BUY | retest2 | 2025-10-16 14:45:00 | 1066.80 | 2025-10-20 13:15:00 | 1038.80 | STOP_HIT | 1.00 | -2.62% |
| BUY | retest2 | 2025-10-17 11:00:00 | 1058.40 | 2025-10-20 13:15:00 | 1038.80 | STOP_HIT | 1.00 | -1.85% |
| BUY | retest2 | 2025-10-17 14:15:00 | 1069.70 | 2025-10-20 13:15:00 | 1038.80 | STOP_HIT | 1.00 | -2.89% |
| BUY | retest2 | 2025-10-20 11:15:00 | 1058.00 | 2025-10-20 13:15:00 | 1038.80 | STOP_HIT | 1.00 | -1.81% |
| BUY | retest2 | 2025-10-27 09:15:00 | 1100.40 | 2025-10-31 11:15:00 | 1077.00 | STOP_HIT | 1.00 | -2.13% |
| BUY | retest2 | 2025-10-28 12:45:00 | 1078.20 | 2025-10-31 11:15:00 | 1077.00 | STOP_HIT | 1.00 | -0.11% |
| BUY | retest2 | 2025-10-29 09:30:00 | 1087.30 | 2025-10-31 11:15:00 | 1077.00 | STOP_HIT | 1.00 | -0.95% |
| SELL | retest2 | 2025-11-10 09:15:00 | 1014.50 | 2025-11-12 09:15:00 | 1041.90 | STOP_HIT | 1.00 | -2.70% |
| SELL | retest2 | 2025-11-10 15:00:00 | 1016.80 | 2025-11-12 09:15:00 | 1041.90 | STOP_HIT | 1.00 | -2.47% |
| SELL | retest2 | 2025-11-11 15:15:00 | 1019.00 | 2025-11-12 09:15:00 | 1041.90 | STOP_HIT | 1.00 | -2.25% |
| SELL | retest2 | 2025-11-20 12:00:00 | 995.00 | 2025-11-27 12:15:00 | 985.30 | STOP_HIT | 1.00 | 0.97% |
| SELL | retest2 | 2025-11-20 15:15:00 | 994.90 | 2025-11-27 12:15:00 | 985.30 | STOP_HIT | 1.00 | 0.96% |
| SELL | retest2 | 2025-12-02 13:45:00 | 964.70 | 2025-12-05 09:15:00 | 916.47 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-02 15:15:00 | 963.00 | 2025-12-05 09:15:00 | 916.75 | PARTIAL | 0.50 | 4.80% |
| SELL | retest2 | 2025-12-03 10:15:00 | 965.00 | 2025-12-05 10:15:00 | 914.85 | PARTIAL | 0.50 | 5.20% |
| SELL | retest2 | 2025-12-03 12:15:00 | 964.50 | 2025-12-05 10:15:00 | 916.27 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-04 09:45:00 | 962.00 | 2025-12-05 10:15:00 | 913.90 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-02 13:45:00 | 964.70 | 2025-12-08 13:15:00 | 868.23 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-12-02 15:15:00 | 963.00 | 2025-12-08 13:15:00 | 866.70 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-12-03 10:15:00 | 965.00 | 2025-12-08 13:15:00 | 868.50 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-12-03 12:15:00 | 964.50 | 2025-12-08 13:15:00 | 868.05 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-12-04 09:45:00 | 962.00 | 2025-12-08 14:15:00 | 865.80 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-12-19 10:15:00 | 895.80 | 2025-12-24 10:15:00 | 900.00 | STOP_HIT | 1.00 | -0.47% |
| SELL | retest2 | 2025-12-19 10:45:00 | 891.40 | 2025-12-24 10:15:00 | 900.00 | STOP_HIT | 1.00 | -0.96% |
| SELL | retest2 | 2025-12-31 11:00:00 | 852.55 | 2026-01-05 09:15:00 | 809.92 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-31 13:15:00 | 852.45 | 2026-01-05 09:15:00 | 809.83 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-31 11:00:00 | 852.55 | 2026-01-06 09:15:00 | 767.29 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-12-31 13:15:00 | 852.45 | 2026-01-06 09:15:00 | 767.21 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-01-22 10:30:00 | 724.50 | 2026-01-22 15:15:00 | 747.00 | STOP_HIT | 1.00 | -3.11% |
| SELL | retest2 | 2026-01-22 13:15:00 | 722.80 | 2026-01-22 15:15:00 | 747.00 | STOP_HIT | 1.00 | -3.35% |
| SELL | retest2 | 2026-01-28 10:15:00 | 703.20 | 2026-01-28 15:15:00 | 719.95 | STOP_HIT | 1.00 | -2.38% |
| SELL | retest2 | 2026-01-28 14:00:00 | 705.75 | 2026-01-28 15:15:00 | 719.95 | STOP_HIT | 1.00 | -2.01% |
| SELL | retest2 | 2026-01-29 10:15:00 | 704.30 | 2026-01-29 14:15:00 | 713.30 | STOP_HIT | 1.00 | -1.28% |
| SELL | retest2 | 2026-01-29 11:00:00 | 705.10 | 2026-01-29 14:15:00 | 713.30 | STOP_HIT | 1.00 | -1.16% |
| BUY | retest2 | 2026-02-05 15:15:00 | 793.70 | 2026-02-11 12:15:00 | 782.75 | STOP_HIT | 1.00 | -1.38% |
| BUY | retest2 | 2026-02-06 10:00:00 | 791.50 | 2026-02-11 12:15:00 | 782.75 | STOP_HIT | 1.00 | -1.11% |
| BUY | retest2 | 2026-02-06 11:00:00 | 791.50 | 2026-02-11 12:15:00 | 782.75 | STOP_HIT | 1.00 | -1.11% |
| BUY | retest2 | 2026-02-06 14:30:00 | 791.60 | 2026-02-11 12:15:00 | 782.75 | STOP_HIT | 1.00 | -1.12% |
| SELL | retest2 | 2026-02-19 09:15:00 | 758.65 | 2026-02-20 09:15:00 | 771.00 | STOP_HIT | 1.00 | -1.63% |
| SELL | retest2 | 2026-02-19 13:00:00 | 759.65 | 2026-02-20 09:15:00 | 771.00 | STOP_HIT | 1.00 | -1.49% |
| BUY | retest2 | 2026-02-23 13:00:00 | 767.70 | 2026-02-25 09:15:00 | 735.20 | STOP_HIT | 1.00 | -4.23% |
| BUY | retest2 | 2026-02-23 13:45:00 | 769.60 | 2026-02-25 09:15:00 | 735.20 | STOP_HIT | 1.00 | -4.47% |
| BUY | retest2 | 2026-02-24 13:00:00 | 769.20 | 2026-02-25 09:15:00 | 735.20 | STOP_HIT | 1.00 | -4.42% |
| BUY | retest2 | 2026-02-24 14:30:00 | 768.00 | 2026-02-25 09:15:00 | 735.20 | STOP_HIT | 1.00 | -4.27% |
| BUY | retest1 | 2026-03-13 10:45:00 | 796.65 | 2026-03-16 10:15:00 | 775.50 | STOP_HIT | 1.00 | -2.65% |
| BUY | retest2 | 2026-03-16 14:15:00 | 789.40 | 2026-03-18 13:15:00 | 868.34 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-02 13:30:00 | 909.25 | 2026-04-15 10:15:00 | 1000.18 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-27 09:15:00 | 1022.20 | 2026-04-30 09:15:00 | 1003.75 | STOP_HIT | 1.00 | -1.80% |
