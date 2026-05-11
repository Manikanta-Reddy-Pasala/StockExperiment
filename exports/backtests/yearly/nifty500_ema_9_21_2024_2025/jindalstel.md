# Jindal Steel Ltd. (JINDALSTEL)

## Backtest Summary

- **Window:** 2024-03-12 15:15:00 → 2026-05-08 15:15:00 (3711 bars)
- **Last close:** 1248.50
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 158 |
| ALERT1 | 109 |
| ALERT2 | 107 |
| ALERT2_SKIP | 55 |
| ALERT3 | 305 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 7 |
| ENTRY2 | 130 |
| PARTIAL | 2 |
| TARGET_HIT | 10 |
| STOP_HIT | 127 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 139 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 24 / 115
- **Target hits / Stop hits / Partials:** 10 / 127 / 2
- **Avg / median % per leg:** -0.21% / -1.31%
- **Sum % (uncompounded):** -28.65%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 77 | 21 | 27.3% | 10 | 66 | 1 | 0.62% | 47.6% |
| BUY @ 2nd Alert (retest1) | 8 | 5 | 62.5% | 0 | 7 | 1 | 1.36% | 10.9% |
| BUY @ 3rd Alert (retest2) | 69 | 16 | 23.2% | 10 | 59 | 0 | 0.53% | 36.7% |
| SELL (all) | 62 | 3 | 4.8% | 0 | 61 | 1 | -1.23% | -76.3% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 62 | 3 | 4.8% | 0 | 61 | 1 | -1.23% | -76.3% |
| retest1 (combined) | 8 | 5 | 62.5% | 0 | 7 | 1 | 1.36% | 10.9% |
| retest2 (combined) | 131 | 19 | 14.5% | 10 | 120 | 1 | -0.30% | -39.5% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-05-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-13 13:15:00 | 936.05 | 930.93 | 930.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-14 09:15:00 | 972.30 | 941.25 | 935.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-18 12:15:00 | 1014.00 | 1014.95 | 1005.66 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-21 09:30:00 | 1023.20 | 1018.56 | 1008.14 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-21 14:15:00 | 1074.36 | 1047.47 | 1027.84 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Stop hit — per-position SL triggered | 2024-05-23 09:15:00 | 1063.10 | 1069.35 | 1053.85 | SL hit (close<ema200) qty=0.50 sl=1069.35 alert=retest1 |

### Cycle 2 — SELL (started 2024-05-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-27 15:15:00 | 1060.10 | 1064.58 | 1064.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-28 09:15:00 | 1057.95 | 1063.26 | 1064.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-30 15:15:00 | 1036.00 | 1034.15 | 1040.87 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-31 09:15:00 | 1038.85 | 1034.15 | 1040.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 09:15:00 | 1033.20 | 1033.96 | 1040.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-31 09:30:00 | 1040.40 | 1033.96 | 1040.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-03 09:15:00 | 1057.00 | 1036.75 | 1038.17 | EMA400 retest candle locked (from downside) |

### Cycle 3 — BUY (started 2024-06-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-03 10:15:00 | 1058.55 | 1041.11 | 1040.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-03 11:15:00 | 1064.55 | 1045.80 | 1042.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-04 09:15:00 | 1015.00 | 1046.61 | 1045.15 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-04 09:15:00 | 1015.00 | 1046.61 | 1045.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 09:15:00 | 1015.00 | 1046.61 | 1045.15 | EMA400 retest candle locked (from upside) |

### Cycle 4 — SELL (started 2024-06-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-04 10:15:00 | 968.10 | 1030.91 | 1038.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-04 11:15:00 | 925.50 | 1009.82 | 1027.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-05 12:15:00 | 981.55 | 971.62 | 993.17 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-06-05 13:00:00 | 981.55 | 971.62 | 993.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 13:15:00 | 994.55 | 976.20 | 993.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-05 14:00:00 | 994.55 | 976.20 | 993.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 14:15:00 | 1004.70 | 981.90 | 994.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-05 15:00:00 | 1004.70 | 981.90 | 994.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 15:15:00 | 1003.85 | 986.29 | 995.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-06 09:15:00 | 1022.55 | 986.29 | 995.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 5 — BUY (started 2024-06-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-06 11:15:00 | 1029.00 | 1004.56 | 1002.12 | EMA200 above EMA400 |

### Cycle 6 — SELL (started 2024-06-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-10 15:15:00 | 1012.00 | 1017.57 | 1017.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-11 14:15:00 | 1008.75 | 1013.47 | 1015.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-12 09:15:00 | 1023.85 | 1014.91 | 1015.61 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-12 09:15:00 | 1023.85 | 1014.91 | 1015.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-12 09:15:00 | 1023.85 | 1014.91 | 1015.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-12 09:30:00 | 1023.35 | 1014.91 | 1015.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 7 — BUY (started 2024-06-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-12 10:15:00 | 1029.30 | 1017.79 | 1016.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-12 12:15:00 | 1035.45 | 1023.28 | 1019.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-18 09:15:00 | 1038.85 | 1047.36 | 1041.52 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-18 09:15:00 | 1038.85 | 1047.36 | 1041.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-18 09:15:00 | 1038.85 | 1047.36 | 1041.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-18 10:00:00 | 1038.85 | 1047.36 | 1041.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-18 10:15:00 | 1036.60 | 1045.21 | 1041.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-18 10:30:00 | 1031.15 | 1045.21 | 1041.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 8 — SELL (started 2024-06-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-18 14:15:00 | 1032.75 | 1037.96 | 1038.47 | EMA200 below EMA400 |

### Cycle 9 — BUY (started 2024-06-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-19 13:15:00 | 1044.35 | 1038.77 | 1038.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-19 15:15:00 | 1049.30 | 1042.08 | 1039.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-20 14:15:00 | 1055.00 | 1057.33 | 1050.18 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-20 15:00:00 | 1055.00 | 1057.33 | 1050.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-21 09:15:00 | 1066.65 | 1059.30 | 1052.32 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-21 11:15:00 | 1071.40 | 1061.04 | 1053.75 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-21 12:00:00 | 1071.00 | 1063.03 | 1055.31 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-21 13:15:00 | 1073.55 | 1062.29 | 1055.68 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-21 14:00:00 | 1089.90 | 1067.81 | 1058.79 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-24 09:15:00 | 1059.60 | 1068.90 | 1061.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-24 09:30:00 | 1050.00 | 1068.90 | 1061.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-24 10:15:00 | 1070.90 | 1069.30 | 1062.60 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-24 14:00:00 | 1072.95 | 1069.89 | 1064.51 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-24 15:15:00 | 1073.25 | 1070.01 | 1065.06 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-25 10:15:00 | 1056.20 | 1066.63 | 1064.74 | SL hit (close<static) qty=1.00 sl=1058.05 alert=retest2 |

### Cycle 10 — SELL (started 2024-06-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-25 12:15:00 | 1057.55 | 1062.67 | 1063.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-26 10:15:00 | 1047.55 | 1058.02 | 1060.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-27 14:15:00 | 1049.90 | 1045.26 | 1050.37 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-27 14:15:00 | 1049.90 | 1045.26 | 1050.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 14:15:00 | 1049.90 | 1045.26 | 1050.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-27 15:00:00 | 1049.90 | 1045.26 | 1050.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 15:15:00 | 1047.20 | 1045.65 | 1050.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-28 09:15:00 | 1058.45 | 1045.65 | 1050.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-28 09:15:00 | 1050.35 | 1046.59 | 1050.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-28 09:30:00 | 1052.45 | 1046.59 | 1050.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-28 10:15:00 | 1049.60 | 1047.19 | 1050.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-28 10:30:00 | 1049.40 | 1047.19 | 1050.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-28 11:15:00 | 1048.55 | 1047.46 | 1049.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-28 11:30:00 | 1049.25 | 1047.46 | 1049.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-28 12:15:00 | 1049.95 | 1047.96 | 1049.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-28 13:00:00 | 1049.95 | 1047.96 | 1049.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-28 13:15:00 | 1050.00 | 1048.37 | 1049.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-28 13:45:00 | 1050.00 | 1048.37 | 1049.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-28 14:15:00 | 1042.95 | 1047.29 | 1049.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-28 14:45:00 | 1046.80 | 1047.29 | 1049.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-28 15:15:00 | 1044.10 | 1046.65 | 1048.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-01 09:15:00 | 1056.85 | 1046.65 | 1048.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-01 09:15:00 | 1049.75 | 1047.27 | 1048.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-01 09:30:00 | 1050.15 | 1047.27 | 1048.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-01 10:15:00 | 1058.70 | 1049.55 | 1049.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-01 11:00:00 | 1058.70 | 1049.55 | 1049.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-01 11:15:00 | 1050.90 | 1049.82 | 1049.90 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-01 12:15:00 | 1046.00 | 1049.82 | 1049.90 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-01 15:15:00 | 1065.15 | 1051.08 | 1050.23 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 11 — BUY (started 2024-07-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-01 15:15:00 | 1065.15 | 1051.08 | 1050.23 | EMA200 above EMA400 |

### Cycle 12 — SELL (started 2024-07-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-02 12:15:00 | 1041.50 | 1049.80 | 1050.05 | EMA200 below EMA400 |

### Cycle 13 — BUY (started 2024-07-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-04 09:15:00 | 1052.20 | 1048.77 | 1048.65 | EMA200 above EMA400 |

### Cycle 14 — SELL (started 2024-07-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-08 09:15:00 | 1046.95 | 1051.59 | 1051.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-08 10:15:00 | 1039.85 | 1049.24 | 1050.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-10 15:15:00 | 1011.25 | 1008.92 | 1018.62 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-07-11 09:15:00 | 1020.90 | 1008.92 | 1018.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-11 09:15:00 | 1016.75 | 1010.48 | 1018.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-11 09:45:00 | 1023.35 | 1010.48 | 1018.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-11 10:15:00 | 1019.85 | 1012.36 | 1018.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-11 11:00:00 | 1019.85 | 1012.36 | 1018.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-11 11:15:00 | 1012.25 | 1012.34 | 1018.00 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-12 12:15:00 | 1009.35 | 1013.81 | 1016.33 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-15 09:15:00 | 1008.25 | 1013.72 | 1015.50 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-15 11:30:00 | 1010.40 | 1013.34 | 1014.96 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-16 10:15:00 | 1022.75 | 1013.45 | 1013.76 | SL hit (close>static) qty=1.00 sl=1020.50 alert=retest2 |

### Cycle 15 — BUY (started 2024-07-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-16 11:15:00 | 1018.30 | 1014.42 | 1014.18 | EMA200 above EMA400 |

### Cycle 16 — SELL (started 2024-07-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-16 15:15:00 | 1008.90 | 1013.44 | 1013.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-18 09:15:00 | 1006.15 | 1011.99 | 1013.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-22 15:15:00 | 953.50 | 953.31 | 964.90 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-07-23 09:15:00 | 948.15 | 953.31 | 964.90 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 14:15:00 | 950.35 | 947.75 | 956.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-23 14:45:00 | 954.60 | 947.75 | 956.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-24 09:15:00 | 955.10 | 950.06 | 955.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-24 10:00:00 | 955.10 | 950.06 | 955.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-24 10:15:00 | 950.05 | 950.06 | 955.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-24 10:45:00 | 956.55 | 950.06 | 955.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-24 11:15:00 | 969.55 | 953.95 | 956.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-24 11:45:00 | 969.85 | 953.95 | 956.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-24 12:15:00 | 971.25 | 957.41 | 957.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-24 12:30:00 | 967.85 | 957.41 | 957.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 17 — BUY (started 2024-07-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-24 13:15:00 | 965.35 | 959.00 | 958.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-24 14:15:00 | 972.70 | 961.74 | 959.89 | Break + close above crossover candle high |

### Cycle 18 — SELL (started 2024-07-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-25 09:15:00 | 940.70 | 958.87 | 958.99 | EMA200 below EMA400 |

### Cycle 19 — BUY (started 2024-07-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-26 11:15:00 | 967.60 | 955.07 | 953.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-26 13:15:00 | 972.05 | 960.86 | 956.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-29 12:15:00 | 967.95 | 968.19 | 963.10 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-29 13:30:00 | 974.60 | 967.75 | 963.36 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-30 09:15:00 | 967.15 | 967.70 | 964.43 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-30 10:15:00 | 971.70 | 967.70 | 964.43 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-30 10:45:00 | 973.40 | 969.12 | 965.37 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-30 12:15:00 | 973.85 | 969.32 | 965.80 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-30 13:45:00 | 972.00 | 970.38 | 966.90 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-01 13:15:00 | 979.85 | 984.79 | 980.73 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-08-01 13:15:00 | 979.85 | 984.79 | 980.73 | SL hit (close<ema400) qty=1.00 sl=980.73 alert=retest1 |

### Cycle 20 — SELL (started 2024-08-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-02 10:15:00 | 963.25 | 977.19 | 978.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-02 14:15:00 | 955.95 | 967.39 | 972.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-06 09:15:00 | 929.70 | 927.81 | 943.79 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-07 09:15:00 | 940.00 | 929.52 | 936.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 09:15:00 | 940.00 | 929.52 | 936.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-07 10:00:00 | 940.00 | 929.52 | 936.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 10:15:00 | 938.95 | 931.41 | 936.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-07 10:45:00 | 944.20 | 931.41 | 936.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 11:15:00 | 938.40 | 932.81 | 936.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-07 12:00:00 | 938.40 | 932.81 | 936.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 12:15:00 | 939.35 | 934.11 | 937.21 | EMA400 retest candle locked (from downside) |

### Cycle 21 — BUY (started 2024-08-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-07 15:15:00 | 944.40 | 939.30 | 939.08 | EMA200 above EMA400 |

### Cycle 22 — SELL (started 2024-08-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-08 09:15:00 | 928.00 | 937.04 | 938.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-08 12:15:00 | 923.20 | 931.05 | 934.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-12 13:15:00 | 918.30 | 915.11 | 920.60 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-12 13:15:00 | 918.30 | 915.11 | 920.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-12 13:15:00 | 918.30 | 915.11 | 920.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-12 14:00:00 | 918.30 | 915.11 | 920.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-13 09:15:00 | 919.40 | 916.85 | 920.11 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-13 10:30:00 | 916.45 | 916.47 | 919.64 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-13 12:00:00 | 915.65 | 916.31 | 919.28 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-13 12:45:00 | 915.50 | 916.17 | 918.94 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-14 13:15:00 | 929.65 | 918.56 | 917.90 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 23 — BUY (started 2024-08-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-14 13:15:00 | 929.65 | 918.56 | 917.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-16 09:15:00 | 936.00 | 925.24 | 921.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-20 09:15:00 | 941.00 | 941.96 | 935.17 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-20 10:00:00 | 941.00 | 941.96 | 935.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-20 11:15:00 | 942.00 | 941.53 | 936.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-20 11:45:00 | 938.90 | 941.53 | 936.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-27 14:15:00 | 969.05 | 973.66 | 969.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-27 15:00:00 | 969.05 | 973.66 | 969.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-27 15:15:00 | 969.20 | 972.77 | 969.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-28 09:15:00 | 971.50 | 972.77 | 969.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-28 09:15:00 | 964.25 | 971.07 | 969.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-28 10:00:00 | 964.25 | 971.07 | 969.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-28 10:15:00 | 967.90 | 970.43 | 969.09 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-28 11:30:00 | 970.35 | 970.86 | 969.41 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-29 09:15:00 | 963.30 | 968.72 | 969.02 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 24 — SELL (started 2024-08-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-29 09:15:00 | 963.30 | 968.72 | 969.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-29 11:15:00 | 957.65 | 965.59 | 967.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-30 09:15:00 | 968.05 | 962.41 | 964.72 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-30 09:15:00 | 968.05 | 962.41 | 964.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-30 09:15:00 | 968.05 | 962.41 | 964.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-30 09:45:00 | 969.90 | 962.41 | 964.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-30 10:15:00 | 973.80 | 964.69 | 965.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-30 11:00:00 | 973.80 | 964.69 | 965.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 25 — BUY (started 2024-08-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-30 12:15:00 | 973.70 | 967.59 | 966.79 | EMA200 above EMA400 |

### Cycle 26 — SELL (started 2024-09-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-02 13:15:00 | 964.10 | 967.39 | 967.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-03 09:15:00 | 960.95 | 964.94 | 966.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-04 14:15:00 | 948.45 | 948.33 | 953.67 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-04 15:00:00 | 948.45 | 948.33 | 953.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-05 11:15:00 | 953.00 | 948.10 | 951.70 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-06 09:30:00 | 942.65 | 947.90 | 950.48 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-06 12:15:00 | 958.25 | 951.06 | 951.35 | SL hit (close>static) qty=1.00 sl=955.65 alert=retest2 |

### Cycle 27 — BUY (started 2024-09-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-06 13:15:00 | 955.55 | 951.95 | 951.73 | EMA200 above EMA400 |

### Cycle 28 — SELL (started 2024-09-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-09 09:15:00 | 943.70 | 950.90 | 951.36 | EMA200 below EMA400 |

### Cycle 29 — BUY (started 2024-09-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-10 10:15:00 | 956.00 | 951.22 | 951.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-10 12:15:00 | 970.55 | 956.36 | 953.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-11 09:15:00 | 956.05 | 959.17 | 956.10 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-11 09:15:00 | 956.05 | 959.17 | 956.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-11 09:15:00 | 956.05 | 959.17 | 956.10 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-11 10:30:00 | 962.75 | 960.48 | 956.98 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2024-09-17 12:15:00 | 1059.03 | 1045.55 | 1029.79 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 30 — SELL (started 2024-09-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-18 15:15:00 | 1024.70 | 1032.66 | 1032.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-19 09:15:00 | 1019.10 | 1029.95 | 1031.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-20 10:15:00 | 1016.05 | 1015.31 | 1021.19 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-20 10:15:00 | 1016.05 | 1015.31 | 1021.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 10:15:00 | 1016.05 | 1015.31 | 1021.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-20 11:00:00 | 1016.05 | 1015.31 | 1021.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 12:15:00 | 1014.90 | 1015.12 | 1020.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-20 12:45:00 | 1014.00 | 1015.12 | 1020.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-23 09:15:00 | 1020.40 | 1014.19 | 1017.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-23 09:45:00 | 1015.75 | 1014.19 | 1017.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-23 10:15:00 | 1011.30 | 1013.61 | 1017.23 | EMA400 retest candle locked (from downside) |

### Cycle 31 — BUY (started 2024-09-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-24 09:15:00 | 1044.00 | 1022.56 | 1020.11 | EMA200 above EMA400 |

### Cycle 32 — SELL (started 2024-09-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-25 15:15:00 | 1025.00 | 1027.91 | 1028.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-26 10:15:00 | 1017.10 | 1024.53 | 1026.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-26 12:15:00 | 1025.25 | 1024.15 | 1026.04 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-26 12:15:00 | 1025.25 | 1024.15 | 1026.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-26 12:15:00 | 1025.25 | 1024.15 | 1026.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-26 13:00:00 | 1025.25 | 1024.15 | 1026.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-26 13:15:00 | 1027.00 | 1024.72 | 1026.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-26 14:00:00 | 1027.00 | 1024.72 | 1026.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-26 14:15:00 | 1027.25 | 1025.23 | 1026.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-26 14:30:00 | 1034.45 | 1025.23 | 1026.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-26 15:15:00 | 1031.00 | 1026.38 | 1026.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-27 09:15:00 | 1038.35 | 1026.38 | 1026.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 33 — BUY (started 2024-09-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-27 09:15:00 | 1041.90 | 1029.48 | 1028.05 | EMA200 above EMA400 |

### Cycle 34 — SELL (started 2024-09-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-27 12:15:00 | 1015.00 | 1025.33 | 1026.43 | EMA200 below EMA400 |

### Cycle 35 — BUY (started 2024-09-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-30 10:15:00 | 1046.30 | 1029.59 | 1027.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-30 12:15:00 | 1052.00 | 1036.69 | 1031.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-01 09:15:00 | 1025.00 | 1035.11 | 1032.53 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-01 09:15:00 | 1025.00 | 1035.11 | 1032.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-01 09:15:00 | 1025.00 | 1035.11 | 1032.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-01 10:00:00 | 1025.00 | 1035.11 | 1032.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-01 10:15:00 | 1029.60 | 1034.01 | 1032.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-01 10:30:00 | 1022.00 | 1034.01 | 1032.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-01 12:15:00 | 1029.00 | 1033.15 | 1032.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-01 12:45:00 | 1029.80 | 1033.15 | 1032.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-01 13:15:00 | 1032.45 | 1033.01 | 1032.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-01 13:30:00 | 1027.95 | 1033.01 | 1032.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-01 14:15:00 | 1034.00 | 1033.21 | 1032.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-01 14:30:00 | 1033.15 | 1033.21 | 1032.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-03 09:15:00 | 1055.45 | 1037.70 | 1034.56 | EMA400 retest candle locked (from upside) |

### Cycle 36 — SELL (started 2024-10-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-04 13:15:00 | 1026.90 | 1040.83 | 1040.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-04 14:15:00 | 1019.85 | 1036.63 | 1039.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-09 10:15:00 | 1003.30 | 992.76 | 1002.21 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-09 10:15:00 | 1003.30 | 992.76 | 1002.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-09 10:15:00 | 1003.30 | 992.76 | 1002.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-09 10:45:00 | 1003.25 | 992.76 | 1002.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-09 11:15:00 | 1000.95 | 994.40 | 1002.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-09 12:15:00 | 1006.55 | 994.40 | 1002.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-09 12:15:00 | 1002.65 | 996.05 | 1002.14 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-09 14:15:00 | 998.10 | 997.18 | 1002.10 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-09 15:00:00 | 998.50 | 997.44 | 1001.77 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-10 09:15:00 | 1012.40 | 1000.63 | 1002.48 | SL hit (close>static) qty=1.00 sl=1009.65 alert=retest2 |

### Cycle 37 — BUY (started 2024-10-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-11 11:15:00 | 1010.50 | 1002.95 | 1001.97 | EMA200 above EMA400 |

### Cycle 38 — SELL (started 2024-10-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-14 10:15:00 | 989.10 | 1000.27 | 1001.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-14 12:15:00 | 988.05 | 996.24 | 999.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-18 09:15:00 | 942.05 | 937.41 | 950.86 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-18 10:00:00 | 942.05 | 937.41 | 950.86 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 10:15:00 | 950.20 | 939.97 | 950.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-18 11:00:00 | 950.20 | 939.97 | 950.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 11:15:00 | 966.00 | 945.17 | 952.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-18 12:00:00 | 966.00 | 945.17 | 952.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 12:15:00 | 970.00 | 950.14 | 953.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-18 13:00:00 | 970.00 | 950.14 | 953.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 39 — BUY (started 2024-10-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-18 14:15:00 | 965.25 | 956.22 | 956.12 | EMA200 above EMA400 |

### Cycle 40 — SELL (started 2024-10-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-21 13:15:00 | 948.80 | 955.68 | 956.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-21 15:15:00 | 946.00 | 952.70 | 954.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-24 09:15:00 | 926.80 | 921.27 | 928.64 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-24 09:15:00 | 926.80 | 921.27 | 928.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-24 09:15:00 | 926.80 | 921.27 | 928.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-24 10:00:00 | 926.80 | 921.27 | 928.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-24 10:15:00 | 919.50 | 920.92 | 927.81 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-25 09:15:00 | 916.15 | 924.22 | 927.15 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-28 11:15:00 | 918.00 | 907.55 | 913.13 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-28 14:00:00 | 917.60 | 911.56 | 913.75 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-28 15:15:00 | 918.00 | 913.08 | 914.24 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-29 10:15:00 | 903.40 | 912.47 | 913.80 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-29 11:15:00 | 901.80 | 912.47 | 913.80 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-29 11:45:00 | 902.10 | 910.58 | 912.82 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-29 15:15:00 | 918.70 | 911.35 | 912.24 | SL hit (close>static) qty=1.00 sl=917.50 alert=retest2 |

### Cycle 41 — BUY (started 2024-10-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-30 09:15:00 | 928.90 | 914.86 | 913.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-01 17:15:00 | 933.10 | 923.88 | 921.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-04 09:15:00 | 909.80 | 921.96 | 920.93 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-04 09:15:00 | 909.80 | 921.96 | 920.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 09:15:00 | 909.80 | 921.96 | 920.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-04 10:00:00 | 909.80 | 921.96 | 920.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 42 — SELL (started 2024-11-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-04 10:15:00 | 902.05 | 917.98 | 919.22 | EMA200 below EMA400 |

### Cycle 43 — BUY (started 2024-11-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-05 12:15:00 | 921.55 | 916.96 | 916.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-05 14:15:00 | 935.30 | 922.02 | 919.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-07 11:15:00 | 946.35 | 949.00 | 939.56 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-07 12:00:00 | 946.35 | 949.00 | 939.56 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 12:15:00 | 939.70 | 947.14 | 939.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-07 13:00:00 | 939.70 | 947.14 | 939.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 13:15:00 | 945.95 | 946.90 | 940.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-07 13:30:00 | 932.20 | 946.90 | 940.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-08 09:15:00 | 947.90 | 946.72 | 941.73 | EMA400 retest candle locked (from upside) |

### Cycle 44 — SELL (started 2024-11-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-08 12:15:00 | 929.95 | 938.93 | 939.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-08 13:15:00 | 922.40 | 935.63 | 937.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-14 09:15:00 | 875.40 | 870.76 | 885.89 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-14 10:00:00 | 875.40 | 870.76 | 885.89 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-14 12:15:00 | 885.55 | 875.93 | 884.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-14 13:00:00 | 885.55 | 875.93 | 884.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-14 13:15:00 | 885.70 | 877.89 | 884.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-14 14:00:00 | 885.70 | 877.89 | 884.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-14 14:15:00 | 877.10 | 877.73 | 884.04 | EMA400 retest candle locked (from downside) |

### Cycle 45 — BUY (started 2024-11-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-18 13:15:00 | 894.90 | 885.93 | 885.54 | EMA200 above EMA400 |

### Cycle 46 — SELL (started 2024-11-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-19 09:15:00 | 881.30 | 884.78 | 885.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-19 10:15:00 | 876.70 | 883.16 | 884.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-19 12:15:00 | 882.75 | 882.41 | 883.75 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-19 13:00:00 | 882.75 | 882.41 | 883.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 13:15:00 | 878.00 | 881.52 | 883.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-19 13:30:00 | 886.55 | 881.52 | 883.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-21 11:15:00 | 876.30 | 873.28 | 877.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-21 11:45:00 | 876.65 | 873.28 | 877.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-21 12:15:00 | 878.65 | 874.35 | 877.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-21 12:45:00 | 875.90 | 874.35 | 877.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-21 13:15:00 | 874.65 | 874.41 | 877.40 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-21 14:15:00 | 872.30 | 874.41 | 877.40 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-22 13:15:00 | 882.75 | 875.44 | 876.00 | SL hit (close>static) qty=1.00 sl=879.95 alert=retest2 |

### Cycle 47 — BUY (started 2024-11-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-22 15:15:00 | 877.90 | 876.47 | 876.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-25 09:15:00 | 884.25 | 878.02 | 877.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-25 12:15:00 | 878.80 | 880.49 | 878.71 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-25 12:15:00 | 878.80 | 880.49 | 878.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-25 12:15:00 | 878.80 | 880.49 | 878.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-25 13:00:00 | 878.80 | 880.49 | 878.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-25 13:15:00 | 885.85 | 881.56 | 879.36 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-25 14:15:00 | 890.50 | 881.56 | 879.36 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-25 14:15:00 | 877.90 | 880.83 | 879.23 | SL hit (close<static) qty=1.00 sl=878.25 alert=retest2 |

### Cycle 48 — SELL (started 2024-12-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-16 10:15:00 | 974.85 | 988.04 | 989.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-16 11:15:00 | 970.10 | 984.45 | 987.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-17 09:15:00 | 979.05 | 978.86 | 982.99 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-17 09:30:00 | 980.10 | 978.86 | 982.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-23 09:15:00 | 935.50 | 922.10 | 929.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-23 10:00:00 | 935.50 | 922.10 | 929.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-23 10:15:00 | 944.50 | 926.58 | 931.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-23 10:30:00 | 945.65 | 926.58 | 931.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 49 — BUY (started 2024-12-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-23 14:15:00 | 942.10 | 933.63 | 933.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-24 13:15:00 | 943.35 | 938.64 | 936.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-26 09:15:00 | 936.30 | 939.50 | 937.35 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-26 09:15:00 | 936.30 | 939.50 | 937.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-26 09:15:00 | 936.30 | 939.50 | 937.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-26 10:00:00 | 936.30 | 939.50 | 937.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-26 10:15:00 | 933.15 | 938.23 | 936.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-26 10:45:00 | 932.35 | 938.23 | 936.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 50 — SELL (started 2024-12-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-26 11:15:00 | 927.15 | 936.02 | 936.08 | EMA200 below EMA400 |

### Cycle 51 — BUY (started 2024-12-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-26 14:15:00 | 942.40 | 936.47 | 936.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-26 15:15:00 | 944.00 | 937.98 | 936.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-27 10:15:00 | 935.15 | 939.00 | 937.62 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-27 10:15:00 | 935.15 | 939.00 | 937.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-27 10:15:00 | 935.15 | 939.00 | 937.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-27 11:00:00 | 935.15 | 939.00 | 937.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-27 11:15:00 | 932.75 | 937.75 | 937.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-27 11:45:00 | 933.35 | 937.75 | 937.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 52 — SELL (started 2024-12-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-27 12:15:00 | 927.05 | 935.61 | 936.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-27 14:15:00 | 924.30 | 932.04 | 934.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-30 15:15:00 | 922.00 | 920.55 | 925.83 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-31 09:15:00 | 921.30 | 920.55 | 925.83 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-31 09:15:00 | 918.45 | 920.13 | 925.16 | EMA400 retest candle locked (from downside) |

### Cycle 53 — BUY (started 2025-01-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-01 10:15:00 | 932.55 | 926.94 | 926.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-01 12:15:00 | 934.95 | 929.61 | 927.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-06 09:15:00 | 941.40 | 949.83 | 944.61 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-06 09:15:00 | 941.40 | 949.83 | 944.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 09:15:00 | 941.40 | 949.83 | 944.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 10:00:00 | 941.40 | 949.83 | 944.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 10:15:00 | 929.50 | 945.77 | 943.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 11:00:00 | 929.50 | 945.77 | 943.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 11:15:00 | 934.45 | 943.50 | 942.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 11:45:00 | 928.15 | 943.50 | 942.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 13:15:00 | 938.70 | 942.01 | 941.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 13:45:00 | 937.50 | 942.01 | 941.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 54 — SELL (started 2025-01-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-06 14:15:00 | 939.10 | 941.42 | 941.66 | EMA200 below EMA400 |

### Cycle 55 — BUY (started 2025-01-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-07 09:15:00 | 948.05 | 941.94 | 941.80 | EMA200 above EMA400 |

### Cycle 56 — SELL (started 2025-01-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-08 11:15:00 | 934.50 | 941.50 | 942.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-09 09:15:00 | 926.45 | 937.34 | 939.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-10 12:15:00 | 926.95 | 925.92 | 931.14 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-10 13:00:00 | 926.95 | 925.92 | 931.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-14 09:15:00 | 906.35 | 897.52 | 910.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-14 09:45:00 | 909.50 | 897.52 | 910.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-14 10:15:00 | 919.85 | 901.98 | 910.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-14 11:00:00 | 919.85 | 901.98 | 910.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-14 11:15:00 | 915.00 | 904.59 | 911.28 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-14 12:15:00 | 913.50 | 904.59 | 911.28 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-14 14:00:00 | 911.10 | 907.13 | 911.35 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-15 11:30:00 | 913.45 | 910.18 | 911.33 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-15 12:00:00 | 912.90 | 910.18 | 911.33 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-15 13:15:00 | 903.40 | 909.17 | 910.70 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-15 14:15:00 | 899.65 | 909.17 | 910.70 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-16 09:15:00 | 925.45 | 912.42 | 911.81 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 57 — BUY (started 2025-01-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-16 09:15:00 | 925.45 | 912.42 | 911.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-20 12:15:00 | 935.00 | 926.28 | 922.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-21 09:15:00 | 926.25 | 929.69 | 925.54 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-21 09:15:00 | 926.25 | 929.69 | 925.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 09:15:00 | 926.25 | 929.69 | 925.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-21 10:00:00 | 926.25 | 929.69 | 925.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 10:15:00 | 920.00 | 927.75 | 925.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-21 10:30:00 | 918.10 | 927.75 | 925.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 11:15:00 | 925.80 | 927.36 | 925.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-21 11:30:00 | 923.40 | 927.36 | 925.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 12:15:00 | 929.45 | 927.78 | 925.50 | EMA400 retest candle locked (from upside) |

### Cycle 58 — SELL (started 2025-01-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-22 09:15:00 | 899.40 | 919.78 | 922.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-22 10:15:00 | 897.30 | 915.28 | 920.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-24 10:15:00 | 894.95 | 894.64 | 900.65 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-24 11:00:00 | 894.95 | 894.64 | 900.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-24 11:15:00 | 905.60 | 896.83 | 901.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-24 12:00:00 | 905.60 | 896.83 | 901.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-24 12:15:00 | 901.30 | 897.72 | 901.12 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-24 13:30:00 | 893.10 | 895.78 | 899.92 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-27 10:15:00 | 848.44 | 876.52 | 888.89 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-01-28 11:15:00 | 850.60 | 849.01 | 865.24 | SL hit (close>ema200) qty=0.50 sl=849.01 alert=retest2 |

### Cycle 59 — BUY (started 2025-02-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-04 11:15:00 | 812.20 | 802.54 | 801.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-05 09:15:00 | 814.65 | 805.67 | 803.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-06 10:15:00 | 814.10 | 814.57 | 810.68 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-06 10:15:00 | 814.10 | 814.57 | 810.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 10:15:00 | 814.10 | 814.57 | 810.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-06 10:45:00 | 811.80 | 814.57 | 810.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 12:15:00 | 811.05 | 813.49 | 810.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-06 12:30:00 | 809.00 | 813.49 | 810.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 13:15:00 | 807.55 | 812.30 | 810.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-06 13:45:00 | 808.35 | 812.30 | 810.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 14:15:00 | 808.45 | 811.53 | 810.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-06 14:30:00 | 809.35 | 811.53 | 810.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 15:15:00 | 809.50 | 811.13 | 810.27 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-07 09:15:00 | 821.60 | 811.13 | 810.27 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-12 09:15:00 | 804.00 | 829.04 | 830.50 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 60 — SELL (started 2025-02-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-12 09:15:00 | 804.00 | 829.04 | 830.50 | EMA200 below EMA400 |

### Cycle 61 — BUY (started 2025-02-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-13 09:15:00 | 847.70 | 831.52 | 830.25 | EMA200 above EMA400 |

### Cycle 62 — SELL (started 2025-02-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-14 12:15:00 | 818.15 | 831.52 | 833.31 | EMA200 below EMA400 |

### Cycle 63 — BUY (started 2025-02-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-17 14:15:00 | 840.00 | 833.42 | 832.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-18 13:15:00 | 846.85 | 838.99 | 836.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-21 09:15:00 | 864.90 | 870.09 | 861.21 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-21 10:00:00 | 864.90 | 870.09 | 861.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-24 09:15:00 | 875.95 | 874.77 | 868.10 | EMA400 retest candle locked (from upside) |

### Cycle 64 — SELL (started 2025-02-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-25 11:15:00 | 860.95 | 866.36 | 866.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-25 13:15:00 | 857.80 | 863.80 | 865.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-27 09:15:00 | 863.00 | 861.38 | 863.67 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-27 09:15:00 | 863.00 | 861.38 | 863.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-27 09:15:00 | 863.00 | 861.38 | 863.67 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-27 11:15:00 | 852.45 | 860.10 | 862.88 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-27 12:45:00 | 852.85 | 856.70 | 860.74 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-28 09:15:00 | 850.60 | 858.86 | 860.69 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-28 12:00:00 | 851.45 | 857.12 | 859.43 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-28 14:15:00 | 856.00 | 854.05 | 857.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-28 15:00:00 | 856.00 | 854.05 | 857.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-28 15:15:00 | 850.80 | 853.40 | 856.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-03 09:15:00 | 864.25 | 853.40 | 856.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-03 09:15:00 | 851.75 | 853.07 | 856.19 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-03 11:00:00 | 850.35 | 852.52 | 855.66 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-03 11:30:00 | 848.80 | 851.98 | 855.12 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-04 10:15:00 | 864.30 | 857.93 | 857.17 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 65 — BUY (started 2025-03-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-04 10:15:00 | 864.30 | 857.93 | 857.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-05 09:15:00 | 877.70 | 866.44 | 862.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-07 14:15:00 | 908.95 | 912.35 | 901.97 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-07 15:00:00 | 908.95 | 912.35 | 901.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 13:15:00 | 907.85 | 914.85 | 908.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-10 14:00:00 | 907.85 | 914.85 | 908.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 14:15:00 | 905.60 | 913.00 | 908.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-10 14:30:00 | 903.55 | 913.00 | 908.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 15:15:00 | 903.05 | 911.01 | 907.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-11 09:15:00 | 896.20 | 911.01 | 907.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-11 09:15:00 | 898.10 | 908.43 | 906.75 | EMA400 retest candle locked (from upside) |

### Cycle 66 — SELL (started 2025-03-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-11 12:15:00 | 900.00 | 905.22 | 905.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-12 10:15:00 | 891.65 | 901.65 | 903.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-12 13:15:00 | 900.20 | 898.00 | 901.24 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-12 14:00:00 | 900.20 | 898.00 | 901.24 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 14:15:00 | 902.50 | 898.90 | 901.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-12 15:00:00 | 902.50 | 898.90 | 901.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 15:15:00 | 902.00 | 899.52 | 901.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-13 09:15:00 | 905.80 | 899.52 | 901.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-13 10:15:00 | 902.50 | 900.61 | 901.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-13 10:30:00 | 901.75 | 900.61 | 901.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-13 11:15:00 | 898.60 | 900.21 | 901.33 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-13 13:00:00 | 893.45 | 898.85 | 900.62 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-17 11:45:00 | 892.85 | 896.63 | 898.28 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-17 12:15:00 | 893.70 | 896.63 | 898.28 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-17 13:00:00 | 893.65 | 896.03 | 897.86 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 13:15:00 | 898.75 | 896.58 | 897.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-17 13:30:00 | 899.50 | 896.58 | 897.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 14:15:00 | 896.55 | 896.57 | 897.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-17 15:15:00 | 897.70 | 896.57 | 897.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 15:15:00 | 897.70 | 896.80 | 897.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-18 09:15:00 | 900.00 | 896.80 | 897.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-03-18 09:15:00 | 910.55 | 899.55 | 898.96 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 67 — BUY (started 2025-03-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-18 09:15:00 | 910.55 | 899.55 | 898.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-18 10:15:00 | 921.70 | 903.98 | 901.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-20 09:15:00 | 918.90 | 922.70 | 917.61 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-20 09:15:00 | 918.90 | 922.70 | 917.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-20 09:15:00 | 918.90 | 922.70 | 917.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-20 09:45:00 | 921.75 | 922.70 | 917.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-20 10:15:00 | 920.95 | 922.35 | 917.91 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-20 12:00:00 | 923.25 | 922.53 | 918.40 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-21 09:30:00 | 925.80 | 923.52 | 920.72 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-21 10:15:00 | 923.40 | 923.52 | 920.72 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-21 14:15:00 | 924.15 | 921.02 | 920.26 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-21 14:15:00 | 913.50 | 919.52 | 919.65 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 68 — SELL (started 2025-03-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-21 14:15:00 | 913.50 | 919.52 | 919.65 | EMA200 below EMA400 |

### Cycle 69 — BUY (started 2025-03-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-24 09:15:00 | 922.90 | 919.81 | 919.73 | EMA200 above EMA400 |

### Cycle 70 — SELL (started 2025-03-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-25 10:15:00 | 914.10 | 919.78 | 920.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-25 11:15:00 | 910.35 | 917.90 | 919.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-27 14:15:00 | 917.50 | 905.90 | 908.17 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-27 14:15:00 | 917.50 | 905.90 | 908.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 14:15:00 | 917.50 | 905.90 | 908.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-27 15:00:00 | 917.50 | 905.90 | 908.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 15:15:00 | 913.10 | 907.34 | 908.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-28 09:15:00 | 920.50 | 907.34 | 908.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 71 — BUY (started 2025-03-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-28 10:15:00 | 915.25 | 909.92 | 909.63 | EMA200 above EMA400 |

### Cycle 72 — SELL (started 2025-04-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-01 09:15:00 | 904.70 | 910.00 | 910.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-01 11:15:00 | 897.85 | 905.87 | 908.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-02 09:15:00 | 906.50 | 903.91 | 905.98 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-02 09:15:00 | 906.50 | 903.91 | 905.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 09:15:00 | 906.50 | 903.91 | 905.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-02 10:00:00 | 906.50 | 903.91 | 905.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 10:15:00 | 907.00 | 904.53 | 906.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-02 11:00:00 | 907.00 | 904.53 | 906.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 11:15:00 | 902.05 | 904.03 | 905.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-02 11:30:00 | 907.00 | 904.03 | 905.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 13:15:00 | 904.95 | 903.55 | 905.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-02 13:30:00 | 904.85 | 903.55 | 905.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 14:15:00 | 909.35 | 904.71 | 905.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-02 15:00:00 | 909.35 | 904.71 | 905.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 15:15:00 | 907.65 | 905.30 | 905.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-03 09:15:00 | 913.20 | 905.30 | 905.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-03 11:15:00 | 905.50 | 904.41 | 905.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-03 12:00:00 | 905.50 | 904.41 | 905.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-03 12:15:00 | 908.00 | 905.13 | 905.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-03 13:00:00 | 908.00 | 905.13 | 905.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 73 — BUY (started 2025-04-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-03 13:15:00 | 909.00 | 905.90 | 905.73 | EMA200 above EMA400 |

### Cycle 74 — SELL (started 2025-04-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-03 15:15:00 | 904.00 | 905.36 | 905.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-04 09:15:00 | 864.45 | 897.18 | 901.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-08 12:15:00 | 805.50 | 801.08 | 824.91 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-08 12:45:00 | 805.00 | 801.08 | 824.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-11 09:15:00 | 806.15 | 789.99 | 800.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-11 09:45:00 | 810.10 | 789.99 | 800.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-11 10:15:00 | 809.45 | 793.88 | 801.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-11 11:00:00 | 809.45 | 793.88 | 801.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-11 15:15:00 | 807.85 | 803.80 | 804.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-15 09:15:00 | 833.45 | 803.80 | 804.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 75 — BUY (started 2025-04-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-15 09:15:00 | 838.55 | 810.75 | 807.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-15 14:15:00 | 847.40 | 830.94 | 819.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-23 09:15:00 | 901.90 | 909.58 | 897.54 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-23 10:00:00 | 901.90 | 909.58 | 897.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 10:15:00 | 901.85 | 908.04 | 897.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-23 10:45:00 | 895.75 | 908.04 | 897.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-24 12:15:00 | 906.50 | 912.12 | 906.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-24 13:00:00 | 906.50 | 912.12 | 906.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-24 13:15:00 | 904.20 | 910.54 | 906.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-24 14:00:00 | 904.20 | 910.54 | 906.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-24 14:15:00 | 908.50 | 910.13 | 906.82 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-25 09:15:00 | 913.35 | 909.70 | 906.93 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-25 09:15:00 | 896.10 | 906.98 | 905.94 | SL hit (close<static) qty=1.00 sl=903.95 alert=retest2 |

### Cycle 76 — SELL (started 2025-04-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-25 10:15:00 | 884.60 | 902.51 | 904.00 | EMA200 below EMA400 |

### Cycle 77 — BUY (started 2025-04-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-28 12:15:00 | 906.50 | 902.23 | 901.71 | EMA200 above EMA400 |

### Cycle 78 — SELL (started 2025-04-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-29 12:15:00 | 898.15 | 902.50 | 902.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-29 14:15:00 | 896.05 | 900.73 | 901.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-30 11:15:00 | 899.60 | 898.80 | 900.27 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-30 11:15:00 | 899.60 | 898.80 | 900.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-30 11:15:00 | 899.60 | 898.80 | 900.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-30 11:45:00 | 900.10 | 898.80 | 900.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-30 12:15:00 | 906.00 | 900.24 | 900.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-30 13:00:00 | 906.00 | 900.24 | 900.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-30 13:15:00 | 896.05 | 899.40 | 900.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-30 13:30:00 | 905.90 | 899.40 | 900.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-30 15:15:00 | 892.50 | 897.52 | 899.31 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-02 09:15:00 | 872.05 | 897.52 | 899.31 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-02 10:45:00 | 880.95 | 892.59 | 896.66 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-02 13:15:00 | 901.70 | 893.53 | 895.94 | SL hit (close>static) qty=1.00 sl=900.95 alert=retest2 |

### Cycle 79 — BUY (started 2025-05-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-05 10:15:00 | 904.75 | 897.26 | 897.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-05 11:15:00 | 907.75 | 899.36 | 898.02 | Break + close above crossover candle high |

### Cycle 80 — SELL (started 2025-05-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-06 09:15:00 | 884.45 | 897.96 | 898.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-06 12:15:00 | 880.15 | 891.31 | 894.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-07 11:15:00 | 879.75 | 878.25 | 885.50 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-07 11:30:00 | 879.75 | 878.25 | 885.50 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 12:15:00 | 881.65 | 878.93 | 885.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-07 12:45:00 | 882.75 | 878.93 | 885.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-12 09:15:00 | 881.40 | 859.96 | 863.91 | EMA400 retest candle locked (from downside) |

### Cycle 81 — BUY (started 2025-05-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 11:15:00 | 879.25 | 867.59 | 866.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 12:15:00 | 897.30 | 873.54 | 869.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-19 09:15:00 | 972.60 | 974.44 | 959.03 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-19 09:30:00 | 970.15 | 974.44 | 959.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 11:15:00 | 956.40 | 969.54 | 959.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-19 12:00:00 | 956.40 | 969.54 | 959.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 12:15:00 | 973.20 | 970.27 | 960.65 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-19 13:15:00 | 975.10 | 970.27 | 960.65 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-19 15:00:00 | 974.40 | 971.09 | 962.69 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-20 09:15:00 | 974.05 | 970.67 | 963.27 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-21 10:15:00 | 957.15 | 962.58 | 962.99 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 82 — SELL (started 2025-05-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-21 10:15:00 | 957.15 | 962.58 | 962.99 | EMA200 below EMA400 |

### Cycle 83 — BUY (started 2025-05-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-21 15:15:00 | 966.45 | 962.92 | 962.80 | EMA200 above EMA400 |

### Cycle 84 — SELL (started 2025-05-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-22 14:15:00 | 960.00 | 963.03 | 963.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-23 09:15:00 | 957.15 | 961.21 | 962.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-26 09:15:00 | 966.00 | 958.48 | 959.81 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-26 09:15:00 | 966.00 | 958.48 | 959.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 09:15:00 | 966.00 | 958.48 | 959.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-26 09:45:00 | 964.45 | 958.48 | 959.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 10:15:00 | 964.10 | 959.61 | 960.20 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-26 12:15:00 | 960.00 | 960.20 | 960.42 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-26 12:15:00 | 964.45 | 961.05 | 960.78 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 85 — BUY (started 2025-05-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-26 12:15:00 | 964.45 | 961.05 | 960.78 | EMA200 above EMA400 |

### Cycle 86 — SELL (started 2025-05-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-26 14:15:00 | 954.95 | 959.48 | 960.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-26 15:15:00 | 954.00 | 958.38 | 959.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-29 09:15:00 | 956.00 | 948.21 | 950.84 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-29 09:15:00 | 956.00 | 948.21 | 950.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 09:15:00 | 956.00 | 948.21 | 950.84 | EMA400 retest candle locked (from downside) |

### Cycle 87 — BUY (started 2025-05-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-29 12:15:00 | 967.45 | 954.56 | 953.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-29 14:15:00 | 970.00 | 959.18 | 955.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-30 10:15:00 | 956.25 | 961.72 | 958.07 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-30 10:15:00 | 956.25 | 961.72 | 958.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 10:15:00 | 956.25 | 961.72 | 958.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-30 11:00:00 | 956.25 | 961.72 | 958.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 11:15:00 | 959.05 | 961.19 | 958.16 | EMA400 retest candle locked (from upside) |

### Cycle 88 — SELL (started 2025-05-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-30 14:15:00 | 948.90 | 956.31 | 956.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-30 15:15:00 | 947.50 | 954.55 | 955.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-03 09:15:00 | 946.00 | 945.07 | 948.83 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-03 10:15:00 | 949.30 | 945.91 | 948.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 10:15:00 | 949.30 | 945.91 | 948.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-03 11:00:00 | 949.30 | 945.91 | 948.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 11:15:00 | 952.55 | 947.24 | 949.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-03 12:00:00 | 952.55 | 947.24 | 949.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 12:15:00 | 946.90 | 947.17 | 949.00 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-03 15:15:00 | 944.20 | 946.88 | 948.55 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-04 11:00:00 | 945.25 | 945.91 | 947.63 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-04 11:30:00 | 944.35 | 946.14 | 947.57 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-04 14:15:00 | 951.40 | 948.56 | 948.42 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 89 — BUY (started 2025-06-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-04 14:15:00 | 951.40 | 948.56 | 948.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-05 09:15:00 | 957.55 | 950.83 | 949.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-05 13:15:00 | 949.40 | 952.36 | 950.85 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-05 13:15:00 | 949.40 | 952.36 | 950.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 13:15:00 | 949.40 | 952.36 | 950.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-05 14:00:00 | 949.40 | 952.36 | 950.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 14:15:00 | 951.65 | 952.22 | 950.92 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-06 09:15:00 | 956.20 | 951.99 | 950.93 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-11 14:15:00 | 956.85 | 968.04 | 968.95 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 90 — SELL (started 2025-06-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-11 14:15:00 | 956.85 | 968.04 | 968.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-11 15:15:00 | 953.00 | 965.04 | 967.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-16 11:15:00 | 924.45 | 924.09 | 935.00 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-16 11:45:00 | 922.85 | 924.09 | 935.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 10:15:00 | 899.55 | 892.11 | 897.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 11:00:00 | 899.55 | 892.11 | 897.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 11:15:00 | 895.30 | 892.75 | 897.44 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-20 12:15:00 | 891.00 | 892.75 | 897.44 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-23 10:15:00 | 901.85 | 895.78 | 896.90 | SL hit (close>static) qty=1.00 sl=900.20 alert=retest2 |

### Cycle 91 — BUY (started 2025-06-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-23 11:15:00 | 906.90 | 898.01 | 897.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-23 12:15:00 | 913.50 | 901.11 | 899.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-24 15:15:00 | 920.40 | 922.18 | 914.41 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-25 09:15:00 | 931.70 | 922.18 | 914.41 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-25 14:15:00 | 922.40 | 925.77 | 920.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-25 15:00:00 | 922.40 | 925.77 | 920.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-25 15:15:00 | 918.65 | 924.34 | 920.01 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-06-25 15:15:00 | 918.65 | 924.34 | 920.01 | SL hit (close<ema400) qty=1.00 sl=920.01 alert=retest1 |

### Cycle 92 — SELL (started 2025-07-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-04 12:15:00 | 948.95 | 957.50 | 958.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-09 09:15:00 | 941.70 | 948.88 | 951.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-09 14:15:00 | 945.45 | 944.67 | 948.11 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-09 15:00:00 | 945.45 | 944.67 | 948.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 09:15:00 | 944.85 | 944.36 | 947.35 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-11 09:45:00 | 939.80 | 943.69 | 945.68 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-14 11:15:00 | 935.20 | 939.31 | 941.36 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-15 12:15:00 | 944.40 | 939.47 | 939.30 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 93 — BUY (started 2025-07-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-15 12:15:00 | 944.40 | 939.47 | 939.30 | EMA200 above EMA400 |

### Cycle 94 — SELL (started 2025-07-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-16 13:15:00 | 935.25 | 939.21 | 939.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-16 14:15:00 | 931.40 | 937.65 | 938.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-17 11:15:00 | 935.85 | 935.61 | 937.36 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-17 11:15:00 | 935.85 | 935.61 | 937.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 11:15:00 | 935.85 | 935.61 | 937.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-17 11:45:00 | 935.00 | 935.61 | 937.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 12:15:00 | 943.45 | 937.18 | 937.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-17 13:00:00 | 943.45 | 937.18 | 937.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 95 — BUY (started 2025-07-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-17 13:15:00 | 946.40 | 939.02 | 938.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-18 09:15:00 | 953.90 | 945.05 | 941.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-22 15:15:00 | 963.70 | 964.09 | 959.92 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-23 09:45:00 | 966.75 | 964.37 | 960.42 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-23 10:15:00 | 966.05 | 964.37 | 960.42 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 09:15:00 | 987.55 | 994.67 | 988.70 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-07-28 09:15:00 | 987.55 | 994.67 | 988.70 | SL hit (close<ema400) qty=1.00 sl=988.70 alert=retest1 |

### Cycle 96 — SELL (started 2025-07-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-28 13:15:00 | 974.95 | 984.80 | 985.45 | EMA200 below EMA400 |

### Cycle 97 — BUY (started 2025-07-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-30 10:15:00 | 989.10 | 983.97 | 983.79 | EMA200 above EMA400 |

### Cycle 98 — SELL (started 2025-07-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-30 15:15:00 | 980.70 | 983.28 | 983.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-31 09:15:00 | 970.80 | 980.78 | 982.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-04 09:15:00 | 975.20 | 959.52 | 965.41 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-04 09:15:00 | 975.20 | 959.52 | 965.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 09:15:00 | 975.20 | 959.52 | 965.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 10:00:00 | 975.20 | 959.52 | 965.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 10:15:00 | 974.20 | 962.46 | 966.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 11:00:00 | 974.20 | 962.46 | 966.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 99 — BUY (started 2025-08-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-04 12:15:00 | 985.00 | 969.21 | 968.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-05 10:15:00 | 990.15 | 979.11 | 974.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-06 10:15:00 | 990.50 | 991.18 | 984.07 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-06 10:45:00 | 989.00 | 991.18 | 984.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 11:15:00 | 990.20 | 990.99 | 984.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-06 11:30:00 | 984.85 | 990.99 | 984.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 12:15:00 | 996.30 | 992.05 | 985.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-06 12:45:00 | 992.00 | 992.05 | 985.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 14:15:00 | 984.65 | 990.59 | 986.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-06 15:00:00 | 984.65 | 990.59 | 986.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 15:15:00 | 989.05 | 990.28 | 986.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-07 09:15:00 | 995.25 | 990.28 | 986.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 09:15:00 | 992.60 | 990.74 | 986.96 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-07 15:00:00 | 1000.20 | 991.07 | 988.26 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-08 10:15:00 | 981.40 | 989.26 | 988.22 | SL hit (close<static) qty=1.00 sl=983.05 alert=retest2 |

### Cycle 100 — SELL (started 2025-08-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-08 11:15:00 | 978.30 | 987.07 | 987.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-08 14:15:00 | 975.50 | 983.17 | 985.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-11 09:15:00 | 984.90 | 982.03 | 984.35 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-11 09:15:00 | 984.90 | 982.03 | 984.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 09:15:00 | 984.90 | 982.03 | 984.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-11 09:30:00 | 987.10 | 982.03 | 984.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 10:15:00 | 984.85 | 982.59 | 984.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-11 10:30:00 | 988.15 | 982.59 | 984.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 11:15:00 | 984.90 | 983.05 | 984.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-11 12:15:00 | 988.00 | 983.05 | 984.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 12:15:00 | 988.85 | 984.21 | 984.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-11 13:00:00 | 988.85 | 984.21 | 984.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 13:15:00 | 989.40 | 985.25 | 985.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-11 14:00:00 | 989.40 | 985.25 | 985.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 101 — BUY (started 2025-08-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-11 14:15:00 | 990.75 | 986.35 | 985.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-12 10:15:00 | 994.65 | 988.42 | 986.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-12 15:15:00 | 993.25 | 993.30 | 990.31 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-13 09:15:00 | 1002.35 | 993.30 | 990.31 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-13 13:30:00 | 1004.75 | 1002.37 | 996.83 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 14:15:00 | 995.90 | 1001.08 | 996.75 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-08-13 14:15:00 | 995.90 | 1001.08 | 996.75 | SL hit (close<ema400) qty=1.00 sl=996.75 alert=retest1 |

### Cycle 102 — SELL (started 2025-08-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-14 10:15:00 | 973.70 | 990.90 | 992.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-14 11:15:00 | 970.80 | 986.88 | 990.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-18 09:15:00 | 989.75 | 981.72 | 986.07 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-18 09:15:00 | 989.75 | 981.72 | 986.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 09:15:00 | 989.75 | 981.72 | 986.07 | EMA400 retest candle locked (from downside) |

### Cycle 103 — BUY (started 2025-08-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-18 15:15:00 | 994.90 | 987.76 | 987.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-19 09:15:00 | 1003.80 | 990.97 | 988.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-21 11:15:00 | 1013.40 | 1013.85 | 1007.97 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-21 12:00:00 | 1013.40 | 1013.85 | 1007.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 12:15:00 | 1012.80 | 1013.64 | 1008.41 | EMA400 retest candle locked (from upside) |

### Cycle 104 — SELL (started 2025-08-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-22 11:15:00 | 999.65 | 1005.71 | 1006.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-22 14:15:00 | 996.45 | 1001.98 | 1004.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-25 10:15:00 | 1000.25 | 999.78 | 1002.67 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-25 11:00:00 | 1000.25 | 999.78 | 1002.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 11:15:00 | 998.95 | 999.61 | 1002.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-25 11:30:00 | 1000.00 | 999.61 | 1002.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 14:15:00 | 963.20 | 958.15 | 962.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 15:00:00 | 963.20 | 958.15 | 962.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 15:15:00 | 963.55 | 959.23 | 963.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-02 09:15:00 | 961.20 | 959.23 | 963.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 09:15:00 | 968.20 | 961.02 | 963.51 | EMA400 retest candle locked (from downside) |

### Cycle 105 — BUY (started 2025-09-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-02 11:15:00 | 980.75 | 966.77 | 965.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-03 09:15:00 | 993.90 | 976.15 | 971.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-08 15:15:00 | 1039.45 | 1041.35 | 1032.98 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-09 09:15:00 | 1034.00 | 1041.35 | 1032.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 09:15:00 | 1029.60 | 1039.00 | 1032.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-09 10:00:00 | 1029.60 | 1039.00 | 1032.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 10:15:00 | 1038.85 | 1038.97 | 1033.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-09 10:30:00 | 1028.45 | 1038.97 | 1033.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 11:15:00 | 1032.85 | 1037.75 | 1033.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-09 12:00:00 | 1032.85 | 1037.75 | 1033.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 12:15:00 | 1030.65 | 1036.33 | 1032.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-09 12:45:00 | 1030.15 | 1036.33 | 1032.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 13:15:00 | 1029.55 | 1034.97 | 1032.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-09 13:30:00 | 1028.65 | 1034.97 | 1032.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 09:15:00 | 1036.10 | 1034.72 | 1033.08 | EMA400 retest candle locked (from upside) |

### Cycle 106 — SELL (started 2025-09-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-12 11:15:00 | 1029.65 | 1034.85 | 1035.22 | EMA200 below EMA400 |

### Cycle 107 — BUY (started 2025-09-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-15 09:15:00 | 1037.90 | 1035.51 | 1035.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-15 10:15:00 | 1043.15 | 1037.04 | 1036.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-16 10:15:00 | 1040.40 | 1042.56 | 1040.06 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-16 10:15:00 | 1040.40 | 1042.56 | 1040.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 10:15:00 | 1040.40 | 1042.56 | 1040.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-16 10:45:00 | 1039.30 | 1042.56 | 1040.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 11:15:00 | 1042.65 | 1042.58 | 1040.29 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-16 12:15:00 | 1047.30 | 1042.58 | 1040.29 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-17 11:45:00 | 1046.65 | 1046.02 | 1043.88 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-17 13:15:00 | 1032.25 | 1042.78 | 1042.75 | SL hit (close<static) qty=1.00 sl=1039.00 alert=retest2 |

### Cycle 108 — SELL (started 2025-09-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-17 14:15:00 | 1033.50 | 1040.92 | 1041.91 | EMA200 below EMA400 |

### Cycle 109 — BUY (started 2025-09-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-18 13:15:00 | 1048.65 | 1041.68 | 1041.48 | EMA200 above EMA400 |

### Cycle 110 — SELL (started 2025-09-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-22 11:15:00 | 1039.10 | 1042.47 | 1042.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-22 14:15:00 | 1031.75 | 1039.68 | 1041.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-23 12:15:00 | 1055.25 | 1040.07 | 1040.42 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-23 12:15:00 | 1055.25 | 1040.07 | 1040.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 12:15:00 | 1055.25 | 1040.07 | 1040.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-23 13:00:00 | 1055.25 | 1040.07 | 1040.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 111 — BUY (started 2025-09-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-23 13:15:00 | 1063.90 | 1044.84 | 1042.56 | EMA200 above EMA400 |

### Cycle 112 — SELL (started 2025-09-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-26 11:15:00 | 1047.25 | 1049.86 | 1050.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-26 13:15:00 | 1036.15 | 1045.87 | 1048.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-29 09:15:00 | 1046.85 | 1041.54 | 1045.13 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-29 09:15:00 | 1046.85 | 1041.54 | 1045.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 09:15:00 | 1046.85 | 1041.54 | 1045.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-29 09:30:00 | 1045.00 | 1041.54 | 1045.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 10:15:00 | 1043.00 | 1041.83 | 1044.94 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-29 11:30:00 | 1041.45 | 1041.22 | 1044.38 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-29 12:00:00 | 1038.80 | 1041.22 | 1044.38 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-29 15:15:00 | 1040.00 | 1042.10 | 1044.02 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-30 09:15:00 | 1054.65 | 1044.28 | 1044.66 | SL hit (close>static) qty=1.00 sl=1049.05 alert=retest2 |

### Cycle 113 — BUY (started 2025-09-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-30 10:15:00 | 1064.50 | 1048.32 | 1046.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-30 12:15:00 | 1067.55 | 1054.92 | 1049.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-01 10:15:00 | 1058.60 | 1060.01 | 1054.85 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-01 10:30:00 | 1057.40 | 1060.01 | 1054.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 11:15:00 | 1062.40 | 1060.49 | 1055.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-01 11:30:00 | 1057.30 | 1060.49 | 1055.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 09:15:00 | 1054.50 | 1061.11 | 1058.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-03 10:15:00 | 1051.80 | 1061.11 | 1058.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 10:15:00 | 1068.50 | 1062.59 | 1058.96 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-03 11:30:00 | 1079.00 | 1066.45 | 1061.04 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-06 10:15:00 | 1047.60 | 1065.30 | 1063.87 | SL hit (close<static) qty=1.00 sl=1049.30 alert=retest2 |

### Cycle 114 — SELL (started 2025-10-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-06 11:15:00 | 1048.20 | 1061.88 | 1062.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-07 10:15:00 | 1041.80 | 1052.31 | 1056.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-09 09:15:00 | 1025.60 | 1024.68 | 1033.93 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-09 09:15:00 | 1025.60 | 1024.68 | 1033.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 09:15:00 | 1025.60 | 1024.68 | 1033.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-09 09:30:00 | 1028.40 | 1024.68 | 1033.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 10:15:00 | 1031.00 | 1025.95 | 1033.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-09 11:00:00 | 1031.00 | 1025.95 | 1033.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 11:15:00 | 1032.70 | 1027.30 | 1033.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-09 11:30:00 | 1034.80 | 1027.30 | 1033.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 12:15:00 | 1033.00 | 1028.44 | 1033.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-09 12:45:00 | 1033.80 | 1028.44 | 1033.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 13:15:00 | 1035.10 | 1029.77 | 1033.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-09 14:00:00 | 1035.10 | 1029.77 | 1033.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 14:15:00 | 1038.60 | 1031.54 | 1034.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-09 15:00:00 | 1038.60 | 1031.54 | 1034.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 09:15:00 | 1008.00 | 1010.07 | 1016.63 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-14 10:45:00 | 1005.90 | 1008.44 | 1015.29 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-16 10:15:00 | 1019.00 | 1004.33 | 1004.71 | SL hit (close>static) qty=1.00 sl=1018.40 alert=retest2 |

### Cycle 115 — BUY (started 2025-10-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-16 11:15:00 | 1017.00 | 1006.87 | 1005.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-16 13:15:00 | 1022.90 | 1012.11 | 1008.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-17 12:15:00 | 1007.80 | 1016.92 | 1013.37 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-17 12:15:00 | 1007.80 | 1016.92 | 1013.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 12:15:00 | 1007.80 | 1016.92 | 1013.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-17 13:00:00 | 1007.80 | 1016.92 | 1013.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 13:15:00 | 1009.10 | 1015.36 | 1012.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-17 13:30:00 | 1007.50 | 1015.36 | 1012.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 116 — SELL (started 2025-10-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-20 09:15:00 | 1004.40 | 1010.84 | 1011.31 | EMA200 below EMA400 |

### Cycle 117 — BUY (started 2025-10-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-23 10:15:00 | 1015.10 | 1010.34 | 1009.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-24 09:15:00 | 1022.70 | 1013.32 | 1011.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-24 12:15:00 | 1011.60 | 1013.53 | 1012.30 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-24 12:15:00 | 1011.60 | 1013.53 | 1012.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 12:15:00 | 1011.60 | 1013.53 | 1012.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-24 13:00:00 | 1011.60 | 1013.53 | 1012.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 13:15:00 | 1009.90 | 1012.80 | 1012.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-24 13:45:00 | 1006.00 | 1012.80 | 1012.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 118 — SELL (started 2025-10-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-24 15:15:00 | 1005.20 | 1010.48 | 1011.10 | EMA200 below EMA400 |

### Cycle 119 — BUY (started 2025-10-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-27 09:15:00 | 1020.10 | 1012.40 | 1011.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-27 11:15:00 | 1027.40 | 1016.27 | 1013.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-29 13:15:00 | 1070.80 | 1073.88 | 1058.43 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-29 14:00:00 | 1070.80 | 1073.88 | 1058.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 09:15:00 | 1068.90 | 1072.26 | 1061.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-30 09:30:00 | 1064.40 | 1072.26 | 1061.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 10:15:00 | 1066.10 | 1071.03 | 1061.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-30 10:45:00 | 1058.30 | 1071.03 | 1061.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 09:15:00 | 1068.90 | 1069.02 | 1064.59 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-03 09:15:00 | 1076.40 | 1068.34 | 1066.07 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-03 10:00:00 | 1076.50 | 1069.97 | 1067.02 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-03 10:45:00 | 1074.40 | 1072.00 | 1068.21 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-04 09:30:00 | 1075.90 | 1075.35 | 1072.10 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 10:15:00 | 1072.60 | 1074.80 | 1072.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-04 10:30:00 | 1070.90 | 1074.80 | 1072.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 11:15:00 | 1073.00 | 1074.44 | 1072.22 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-04 13:45:00 | 1077.20 | 1074.11 | 1072.44 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-04 14:15:00 | 1077.10 | 1074.11 | 1072.44 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-06 09:15:00 | 1067.20 | 1073.95 | 1072.92 | SL hit (close<static) qty=1.00 sl=1071.70 alert=retest2 |

### Cycle 120 — SELL (started 2025-11-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-06 10:15:00 | 1062.80 | 1071.72 | 1072.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-06 11:15:00 | 1052.40 | 1067.85 | 1070.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-07 10:15:00 | 1055.30 | 1052.94 | 1060.33 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-07 11:00:00 | 1055.30 | 1052.94 | 1060.33 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 12:15:00 | 1070.20 | 1056.02 | 1060.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-07 13:00:00 | 1070.20 | 1056.02 | 1060.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 13:15:00 | 1066.10 | 1058.03 | 1060.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-07 13:30:00 | 1073.20 | 1058.03 | 1060.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 121 — BUY (started 2025-11-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-10 09:15:00 | 1072.40 | 1063.97 | 1063.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-11 13:15:00 | 1085.90 | 1073.36 | 1069.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-12 13:15:00 | 1074.00 | 1077.09 | 1074.12 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-12 13:15:00 | 1074.00 | 1077.09 | 1074.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 13:15:00 | 1074.00 | 1077.09 | 1074.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-12 14:00:00 | 1074.00 | 1077.09 | 1074.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 14:15:00 | 1081.00 | 1077.87 | 1074.75 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-13 09:15:00 | 1084.00 | 1078.09 | 1075.14 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-13 12:45:00 | 1081.60 | 1083.27 | 1078.79 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-13 14:15:00 | 1083.10 | 1082.54 | 1078.87 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-14 13:00:00 | 1083.70 | 1086.81 | 1083.05 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 13:15:00 | 1084.40 | 1086.33 | 1083.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-14 13:45:00 | 1085.90 | 1086.33 | 1083.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 14:15:00 | 1076.00 | 1084.26 | 1082.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-14 14:45:00 | 1070.50 | 1084.26 | 1082.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 15:15:00 | 1077.10 | 1082.83 | 1082.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-17 09:15:00 | 1078.90 | 1082.83 | 1082.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 09:15:00 | 1077.90 | 1081.84 | 1081.65 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-11-17 10:15:00 | 1078.20 | 1081.12 | 1081.34 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 122 — SELL (started 2025-11-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-17 10:15:00 | 1078.20 | 1081.12 | 1081.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-17 11:15:00 | 1075.40 | 1079.97 | 1080.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-17 14:15:00 | 1080.40 | 1079.04 | 1080.06 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-17 14:15:00 | 1080.40 | 1079.04 | 1080.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 14:15:00 | 1080.40 | 1079.04 | 1080.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-17 14:30:00 | 1078.00 | 1079.04 | 1080.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 15:15:00 | 1077.40 | 1078.72 | 1079.82 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-18 09:15:00 | 1065.20 | 1078.72 | 1079.82 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-27 09:15:00 | 1040.00 | 1037.46 | 1037.30 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 123 — BUY (started 2025-11-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-27 09:15:00 | 1040.00 | 1037.46 | 1037.30 | EMA200 above EMA400 |

### Cycle 124 — SELL (started 2025-12-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-02 12:15:00 | 1036.30 | 1042.81 | 1043.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-02 13:15:00 | 1031.30 | 1040.51 | 1041.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-05 12:15:00 | 1007.10 | 1006.17 | 1014.41 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-05 13:00:00 | 1007.10 | 1006.17 | 1014.41 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 11:15:00 | 999.90 | 1004.39 | 1010.01 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-08 12:15:00 | 996.90 | 1004.39 | 1010.01 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-09 11:30:00 | 998.00 | 999.15 | 1003.87 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-10 10:15:00 | 1010.70 | 1004.72 | 1004.86 | SL hit (close>static) qty=1.00 sl=1010.60 alert=retest2 |

### Cycle 125 — BUY (started 2025-12-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-10 11:15:00 | 1009.20 | 1005.62 | 1005.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-11 13:15:00 | 1013.50 | 1007.96 | 1006.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-15 10:15:00 | 1024.20 | 1024.34 | 1018.51 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-15 10:45:00 | 1024.30 | 1024.34 | 1018.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 09:15:00 | 1015.40 | 1026.33 | 1022.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 10:00:00 | 1015.40 | 1026.33 | 1022.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 10:15:00 | 1014.00 | 1023.86 | 1021.80 | EMA400 retest candle locked (from upside) |

### Cycle 126 — SELL (started 2025-12-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-16 12:15:00 | 1013.00 | 1020.29 | 1020.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-16 13:15:00 | 1010.20 | 1018.27 | 1019.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-19 09:15:00 | 991.90 | 990.99 | 998.60 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-19 09:15:00 | 991.90 | 990.99 | 998.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 09:15:00 | 991.90 | 990.99 | 998.60 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-19 10:45:00 | 983.50 | 989.53 | 997.24 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-19 12:15:00 | 986.50 | 988.97 | 996.28 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-19 13:00:00 | 986.10 | 988.39 | 995.36 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-22 09:15:00 | 1004.00 | 992.42 | 995.04 | SL hit (close>static) qty=1.00 sl=1002.50 alert=retest2 |

### Cycle 127 — BUY (started 2025-12-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-22 10:15:00 | 1014.60 | 996.86 | 996.82 | EMA200 above EMA400 |

### Cycle 128 — SELL (started 2025-12-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-24 14:15:00 | 998.80 | 1003.76 | 1003.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-26 09:15:00 | 990.00 | 1000.07 | 1002.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-29 09:15:00 | 1003.40 | 993.77 | 996.90 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-29 09:15:00 | 1003.40 | 993.77 | 996.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 09:15:00 | 1003.40 | 993.77 | 996.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-29 09:30:00 | 1006.60 | 993.77 | 996.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 10:15:00 | 1004.70 | 995.95 | 997.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-29 10:30:00 | 1000.40 | 995.95 | 997.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 09:15:00 | 1001.90 | 995.48 | 996.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-30 10:00:00 | 1001.90 | 995.48 | 996.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 129 — BUY (started 2025-12-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-30 10:15:00 | 1006.10 | 997.61 | 997.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-30 12:15:00 | 1012.40 | 1001.96 | 999.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-06 13:15:00 | 1078.50 | 1082.06 | 1075.48 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-06 14:00:00 | 1078.50 | 1082.06 | 1075.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 14:15:00 | 1079.60 | 1081.57 | 1075.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-06 14:30:00 | 1076.30 | 1081.57 | 1075.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 09:15:00 | 1081.00 | 1081.35 | 1076.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-07 10:00:00 | 1081.00 | 1081.35 | 1076.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 10:15:00 | 1075.00 | 1080.08 | 1076.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-07 11:00:00 | 1075.00 | 1080.08 | 1076.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 11:15:00 | 1073.40 | 1078.74 | 1076.29 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-07 13:00:00 | 1077.50 | 1078.49 | 1076.40 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-07 14:45:00 | 1078.40 | 1077.20 | 1076.13 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-08 09:15:00 | 1034.40 | 1068.13 | 1072.16 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 130 — SELL (started 2026-01-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-08 09:15:00 | 1034.40 | 1068.13 | 1072.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-08 12:15:00 | 1026.90 | 1049.43 | 1061.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-12 12:15:00 | 1024.10 | 1014.03 | 1025.17 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-12 12:15:00 | 1024.10 | 1014.03 | 1025.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 12:15:00 | 1024.10 | 1014.03 | 1025.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-12 13:00:00 | 1024.10 | 1014.03 | 1025.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 13:15:00 | 1015.80 | 1014.38 | 1024.32 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 12:45:00 | 1014.30 | 1019.29 | 1023.17 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-14 10:15:00 | 1039.10 | 1020.74 | 1021.75 | SL hit (close>static) qty=1.00 sl=1025.00 alert=retest2 |

### Cycle 131 — BUY (started 2026-01-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-14 11:15:00 | 1036.00 | 1023.79 | 1023.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-16 09:15:00 | 1045.50 | 1035.99 | 1030.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-16 13:15:00 | 1041.30 | 1042.36 | 1035.50 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-16 14:00:00 | 1041.30 | 1042.36 | 1035.50 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 09:15:00 | 1043.00 | 1042.16 | 1037.11 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-19 12:00:00 | 1056.30 | 1046.19 | 1039.89 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-19 13:15:00 | 1055.00 | 1047.37 | 1041.00 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-20 11:45:00 | 1054.90 | 1052.07 | 1046.67 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-21 10:15:00 | 1039.30 | 1045.32 | 1045.34 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 132 — SELL (started 2026-01-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-21 10:15:00 | 1039.30 | 1045.32 | 1045.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-21 13:15:00 | 1030.90 | 1041.21 | 1043.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-21 15:15:00 | 1045.80 | 1041.92 | 1043.29 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-21 15:15:00 | 1045.80 | 1041.92 | 1043.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-21 15:15:00 | 1045.80 | 1041.92 | 1043.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 09:15:00 | 1060.00 | 1041.92 | 1043.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 133 — BUY (started 2026-01-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-22 09:15:00 | 1070.40 | 1047.62 | 1045.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-22 14:15:00 | 1075.20 | 1064.29 | 1055.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-23 10:15:00 | 1071.60 | 1072.94 | 1062.40 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-23 11:00:00 | 1071.60 | 1072.94 | 1062.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 11:15:00 | 1068.10 | 1071.97 | 1062.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-23 11:45:00 | 1062.30 | 1071.97 | 1062.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 14:15:00 | 1062.60 | 1070.36 | 1064.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-23 14:45:00 | 1066.00 | 1070.36 | 1064.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 15:15:00 | 1061.00 | 1068.49 | 1064.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-27 09:15:00 | 1065.00 | 1068.49 | 1064.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 09:15:00 | 1069.80 | 1068.75 | 1064.68 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-27 14:45:00 | 1080.00 | 1071.51 | 1067.54 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-01 14:15:00 | 1100.10 | 1122.73 | 1123.80 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 134 — SELL (started 2026-02-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-01 14:15:00 | 1100.10 | 1122.73 | 1123.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-02 10:15:00 | 1090.50 | 1111.27 | 1117.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-02 13:15:00 | 1108.50 | 1106.40 | 1113.54 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-02 13:45:00 | 1108.30 | 1106.40 | 1113.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 14:15:00 | 1123.10 | 1109.74 | 1114.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-02 15:00:00 | 1123.10 | 1109.74 | 1114.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 15:15:00 | 1131.50 | 1114.09 | 1115.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-03 09:15:00 | 1155.20 | 1114.09 | 1115.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 135 — BUY (started 2026-02-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 09:15:00 | 1140.80 | 1119.43 | 1118.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-04 09:15:00 | 1175.70 | 1149.37 | 1136.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-05 09:15:00 | 1159.30 | 1162.19 | 1151.08 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-05 09:15:00 | 1159.30 | 1162.19 | 1151.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 09:15:00 | 1159.30 | 1162.19 | 1151.08 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-05 11:00:00 | 1174.00 | 1164.55 | 1153.16 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-06 10:45:00 | 1169.90 | 1173.22 | 1164.28 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-13 10:15:00 | 1171.20 | 1188.54 | 1190.41 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 136 — SELL (started 2026-02-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-13 10:15:00 | 1171.20 | 1188.54 | 1190.41 | EMA200 below EMA400 |

### Cycle 137 — BUY (started 2026-02-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-16 10:15:00 | 1201.60 | 1189.11 | 1188.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-16 11:15:00 | 1203.90 | 1192.07 | 1190.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-17 09:15:00 | 1199.50 | 1202.11 | 1196.83 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-17 10:00:00 | 1199.50 | 1202.11 | 1196.83 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 10:15:00 | 1193.50 | 1200.39 | 1196.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-17 11:00:00 | 1193.50 | 1200.39 | 1196.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 11:15:00 | 1210.10 | 1202.33 | 1197.76 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-17 12:15:00 | 1212.90 | 1202.33 | 1197.76 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-17 15:15:00 | 1214.00 | 1203.69 | 1199.53 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-23 11:15:00 | 1211.20 | 1216.02 | 1216.07 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 138 — SELL (started 2026-02-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-23 11:15:00 | 1211.20 | 1216.02 | 1216.07 | EMA200 below EMA400 |

### Cycle 139 — BUY (started 2026-02-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-24 09:15:00 | 1223.10 | 1216.49 | 1216.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-24 13:15:00 | 1225.50 | 1220.07 | 1218.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-26 13:15:00 | 1260.00 | 1260.08 | 1249.11 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-26 13:45:00 | 1259.80 | 1260.08 | 1249.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 09:15:00 | 1251.00 | 1258.34 | 1251.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-27 10:00:00 | 1251.00 | 1258.34 | 1251.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 10:15:00 | 1254.40 | 1257.55 | 1251.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-27 10:30:00 | 1250.00 | 1257.55 | 1251.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 11:15:00 | 1251.40 | 1256.32 | 1251.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-27 11:30:00 | 1251.80 | 1256.32 | 1251.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 12:15:00 | 1254.20 | 1255.90 | 1251.62 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-27 13:15:00 | 1257.70 | 1255.90 | 1251.62 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-27 13:45:00 | 1259.50 | 1256.86 | 1252.44 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-27 14:15:00 | 1245.10 | 1254.51 | 1251.77 | SL hit (close<static) qty=1.00 sl=1250.00 alert=retest2 |

### Cycle 140 — SELL (started 2026-03-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-02 10:15:00 | 1241.30 | 1249.02 | 1249.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-02 14:15:00 | 1238.60 | 1244.20 | 1246.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-05 10:15:00 | 1188.30 | 1186.85 | 1206.57 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-05 10:30:00 | 1192.70 | 1186.85 | 1206.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 11:15:00 | 1186.00 | 1184.45 | 1193.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-06 11:30:00 | 1190.70 | 1184.45 | 1193.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 09:15:00 | 1180.10 | 1157.39 | 1168.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-10 10:00:00 | 1180.10 | 1157.39 | 1168.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 10:15:00 | 1186.40 | 1163.19 | 1170.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-10 11:00:00 | 1186.40 | 1163.19 | 1170.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 141 — BUY (started 2026-03-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-10 13:15:00 | 1190.00 | 1175.12 | 1174.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-11 09:15:00 | 1193.50 | 1182.48 | 1178.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-11 14:15:00 | 1185.40 | 1188.54 | 1183.59 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-11 15:15:00 | 1185.00 | 1188.54 | 1183.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 15:15:00 | 1185.00 | 1187.83 | 1183.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-12 09:15:00 | 1179.20 | 1187.83 | 1183.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 09:15:00 | 1184.40 | 1187.15 | 1183.78 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-12 10:15:00 | 1190.20 | 1187.15 | 1183.78 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-13 11:15:00 | 1160.90 | 1189.71 | 1191.26 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 142 — SELL (started 2026-03-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-13 11:15:00 | 1160.90 | 1189.71 | 1191.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-13 12:15:00 | 1154.50 | 1182.67 | 1187.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-17 09:15:00 | 1150.40 | 1141.71 | 1155.74 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-17 10:00:00 | 1150.40 | 1141.71 | 1155.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 10:15:00 | 1155.30 | 1144.43 | 1155.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 11:00:00 | 1155.30 | 1144.43 | 1155.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 11:15:00 | 1150.70 | 1145.68 | 1155.24 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-17 12:15:00 | 1145.20 | 1145.68 | 1155.24 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-17 14:15:00 | 1157.30 | 1150.44 | 1155.25 | SL hit (close>static) qty=1.00 sl=1157.10 alert=retest2 |

### Cycle 143 — BUY (started 2026-03-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 12:15:00 | 1178.50 | 1158.52 | 1157.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-18 13:15:00 | 1182.10 | 1163.24 | 1159.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 09:15:00 | 1146.80 | 1164.28 | 1161.38 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-19 09:15:00 | 1146.80 | 1164.28 | 1161.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 09:15:00 | 1146.80 | 1164.28 | 1161.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-19 09:45:00 | 1147.30 | 1164.28 | 1161.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 10:15:00 | 1145.20 | 1160.46 | 1159.91 | EMA400 retest candle locked (from upside) |

### Cycle 144 — SELL (started 2026-03-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 11:15:00 | 1150.50 | 1158.47 | 1159.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 13:15:00 | 1138.70 | 1152.33 | 1156.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-20 09:15:00 | 1178.90 | 1154.79 | 1155.90 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-20 09:15:00 | 1178.90 | 1154.79 | 1155.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 09:15:00 | 1178.90 | 1154.79 | 1155.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 10:00:00 | 1178.90 | 1154.79 | 1155.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 145 — BUY (started 2026-03-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-20 10:15:00 | 1181.40 | 1160.11 | 1158.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-20 11:15:00 | 1189.90 | 1166.07 | 1161.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-23 09:15:00 | 1128.40 | 1168.25 | 1165.65 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-23 09:15:00 | 1128.40 | 1168.25 | 1165.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-23 09:15:00 | 1128.40 | 1168.25 | 1165.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-23 10:00:00 | 1128.40 | 1168.25 | 1165.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 146 — SELL (started 2026-03-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-23 10:15:00 | 1115.40 | 1157.68 | 1161.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-23 11:15:00 | 1108.70 | 1147.88 | 1156.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 12:15:00 | 1127.40 | 1122.13 | 1134.47 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-24 12:30:00 | 1125.60 | 1122.13 | 1134.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 13:15:00 | 1127.20 | 1123.14 | 1133.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 13:30:00 | 1130.00 | 1123.14 | 1133.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 09:15:00 | 1144.00 | 1127.23 | 1132.98 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-25 11:30:00 | 1131.30 | 1132.37 | 1134.51 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-25 14:15:00 | 1143.20 | 1136.94 | 1136.26 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 147 — BUY (started 2026-03-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 14:15:00 | 1143.20 | 1136.94 | 1136.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-27 09:15:00 | 1147.20 | 1139.53 | 1137.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-27 13:15:00 | 1144.00 | 1144.07 | 1140.75 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-27 14:00:00 | 1144.00 | 1144.07 | 1140.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 14:15:00 | 1130.20 | 1141.30 | 1139.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 15:00:00 | 1130.20 | 1141.30 | 1139.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 15:15:00 | 1129.80 | 1139.00 | 1138.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-30 09:15:00 | 1120.50 | 1139.00 | 1138.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 148 — SELL (started 2026-03-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-30 09:15:00 | 1117.60 | 1134.72 | 1136.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 14:15:00 | 1110.50 | 1121.20 | 1128.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 1152.60 | 1125.69 | 1129.28 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 1152.60 | 1125.69 | 1129.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 1152.60 | 1125.69 | 1129.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 10:00:00 | 1152.60 | 1125.69 | 1129.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 10:15:00 | 1149.30 | 1130.41 | 1131.10 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-01 11:15:00 | 1146.20 | 1130.41 | 1131.10 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-01 11:15:00 | 1149.80 | 1134.29 | 1132.80 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 149 — BUY (started 2026-04-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 11:15:00 | 1149.80 | 1134.29 | 1132.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-01 12:15:00 | 1154.10 | 1138.25 | 1134.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-01 14:15:00 | 1136.50 | 1138.74 | 1135.62 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 14:15:00 | 1136.50 | 1138.74 | 1135.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 14:15:00 | 1136.50 | 1138.74 | 1135.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-01 14:45:00 | 1136.10 | 1138.74 | 1135.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 15:15:00 | 1135.00 | 1137.99 | 1135.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-02 09:15:00 | 1114.50 | 1137.99 | 1135.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 150 — SELL (started 2026-04-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-02 09:15:00 | 1105.50 | 1131.49 | 1132.83 | EMA200 below EMA400 |

### Cycle 151 — BUY (started 2026-04-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-02 15:15:00 | 1141.90 | 1133.24 | 1132.25 | EMA200 above EMA400 |

### Cycle 152 — SELL (started 2026-04-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-06 10:15:00 | 1103.40 | 1126.58 | 1129.35 | EMA200 below EMA400 |

### Cycle 153 — BUY (started 2026-04-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-07 09:15:00 | 1145.20 | 1132.69 | 1131.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-07 12:15:00 | 1149.30 | 1139.62 | 1134.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-09 12:15:00 | 1201.50 | 1203.88 | 1186.06 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-09 12:45:00 | 1202.80 | 1203.88 | 1186.06 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 1191.30 | 1206.04 | 1198.69 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 10:45:00 | 1216.90 | 1208.81 | 1200.62 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-23 10:15:00 | 1261.30 | 1277.52 | 1278.02 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 154 — SELL (started 2026-04-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-23 10:15:00 | 1261.30 | 1277.52 | 1278.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-23 13:15:00 | 1257.90 | 1269.37 | 1273.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-24 11:15:00 | 1264.50 | 1258.24 | 1265.64 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-24 11:15:00 | 1264.50 | 1258.24 | 1265.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 11:15:00 | 1264.50 | 1258.24 | 1265.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-24 12:00:00 | 1264.50 | 1258.24 | 1265.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 12:15:00 | 1270.90 | 1260.77 | 1266.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-24 13:00:00 | 1270.90 | 1260.77 | 1266.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 13:15:00 | 1263.90 | 1261.40 | 1265.92 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-24 14:45:00 | 1258.70 | 1259.92 | 1264.84 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-27 09:15:00 | 1281.50 | 1263.80 | 1265.72 | SL hit (close>static) qty=1.00 sl=1273.10 alert=retest2 |

### Cycle 155 — BUY (started 2026-04-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 11:15:00 | 1277.40 | 1268.95 | 1267.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-27 12:15:00 | 1284.50 | 1272.06 | 1269.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-28 11:15:00 | 1274.90 | 1277.80 | 1274.11 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-28 11:15:00 | 1274.90 | 1277.80 | 1274.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 11:15:00 | 1274.90 | 1277.80 | 1274.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-28 12:00:00 | 1274.90 | 1277.80 | 1274.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 12:15:00 | 1269.40 | 1276.12 | 1273.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-28 12:45:00 | 1269.20 | 1276.12 | 1273.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 13:15:00 | 1262.80 | 1273.45 | 1272.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-28 14:00:00 | 1262.80 | 1273.45 | 1272.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 156 — SELL (started 2026-04-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-28 14:15:00 | 1264.00 | 1271.56 | 1271.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-29 09:15:00 | 1239.80 | 1264.16 | 1268.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-30 14:15:00 | 1220.80 | 1220.22 | 1235.11 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-30 15:00:00 | 1220.80 | 1220.22 | 1235.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 09:15:00 | 1263.00 | 1230.32 | 1237.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-04 09:45:00 | 1266.60 | 1230.32 | 1237.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 10:15:00 | 1260.00 | 1236.26 | 1239.29 | EMA400 retest candle locked (from downside) |

### Cycle 157 — BUY (started 2026-05-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-04 12:15:00 | 1264.60 | 1245.89 | 1243.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-05 09:15:00 | 1278.40 | 1259.86 | 1251.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-05 11:15:00 | 1261.30 | 1262.44 | 1254.21 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-05-05 12:00:00 | 1261.30 | 1262.44 | 1254.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 10:15:00 | 1255.70 | 1260.97 | 1257.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-06 11:00:00 | 1255.70 | 1260.97 | 1257.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 11:15:00 | 1263.50 | 1261.48 | 1257.71 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-06 14:15:00 | 1267.00 | 1260.70 | 1257.95 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-07 09:15:00 | 1269.80 | 1261.66 | 1258.90 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-07 10:30:00 | 1269.40 | 1264.96 | 1261.00 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-07 11:45:00 | 1266.50 | 1265.73 | 1261.71 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-07 13:15:00 | 1261.60 | 1264.45 | 1261.80 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-05-08 09:15:00 | 1241.60 | 1258.39 | 1259.56 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 158 — SELL (started 2026-05-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-08 09:15:00 | 1241.60 | 1258.39 | 1259.56 | EMA200 below EMA400 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-05-21 09:30:00 | 1023.20 | 2024-05-21 14:15:00 | 1074.36 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2024-05-21 09:30:00 | 1023.20 | 2024-05-23 09:15:00 | 1063.10 | STOP_HIT | 0.50 | 3.90% |
| BUY | retest2 | 2024-05-23 12:30:00 | 1063.95 | 2024-05-27 15:15:00 | 1060.10 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest2 | 2024-05-23 13:45:00 | 1064.20 | 2024-05-27 15:15:00 | 1060.10 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest2 | 2024-05-27 10:00:00 | 1062.75 | 2024-05-27 15:15:00 | 1060.10 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest2 | 2024-06-21 11:15:00 | 1071.40 | 2024-06-25 10:15:00 | 1056.20 | STOP_HIT | 1.00 | -1.42% |
| BUY | retest2 | 2024-06-21 12:00:00 | 1071.00 | 2024-06-25 10:15:00 | 1056.20 | STOP_HIT | 1.00 | -1.38% |
| BUY | retest2 | 2024-06-21 13:15:00 | 1073.55 | 2024-06-25 12:15:00 | 1057.55 | STOP_HIT | 1.00 | -1.49% |
| BUY | retest2 | 2024-06-21 14:00:00 | 1089.90 | 2024-06-25 12:15:00 | 1057.55 | STOP_HIT | 1.00 | -2.97% |
| BUY | retest2 | 2024-06-24 14:00:00 | 1072.95 | 2024-06-25 12:15:00 | 1057.55 | STOP_HIT | 1.00 | -1.44% |
| BUY | retest2 | 2024-06-24 15:15:00 | 1073.25 | 2024-06-25 12:15:00 | 1057.55 | STOP_HIT | 1.00 | -1.46% |
| SELL | retest2 | 2024-07-01 12:15:00 | 1046.00 | 2024-07-01 15:15:00 | 1065.15 | STOP_HIT | 1.00 | -1.83% |
| SELL | retest2 | 2024-07-12 12:15:00 | 1009.35 | 2024-07-16 10:15:00 | 1022.75 | STOP_HIT | 1.00 | -1.33% |
| SELL | retest2 | 2024-07-15 09:15:00 | 1008.25 | 2024-07-16 10:15:00 | 1022.75 | STOP_HIT | 1.00 | -1.44% |
| SELL | retest2 | 2024-07-15 11:30:00 | 1010.40 | 2024-07-16 10:15:00 | 1022.75 | STOP_HIT | 1.00 | -1.22% |
| BUY | retest1 | 2024-07-29 13:30:00 | 974.60 | 2024-08-01 13:15:00 | 979.85 | STOP_HIT | 1.00 | 0.54% |
| BUY | retest2 | 2024-07-30 10:15:00 | 971.70 | 2024-08-02 10:15:00 | 963.25 | STOP_HIT | 1.00 | -0.87% |
| BUY | retest2 | 2024-07-30 10:45:00 | 973.40 | 2024-08-02 10:15:00 | 963.25 | STOP_HIT | 1.00 | -1.04% |
| BUY | retest2 | 2024-07-30 12:15:00 | 973.85 | 2024-08-02 10:15:00 | 963.25 | STOP_HIT | 1.00 | -1.09% |
| BUY | retest2 | 2024-07-30 13:45:00 | 972.00 | 2024-08-02 10:15:00 | 963.25 | STOP_HIT | 1.00 | -0.90% |
| SELL | retest2 | 2024-08-13 10:30:00 | 916.45 | 2024-08-14 13:15:00 | 929.65 | STOP_HIT | 1.00 | -1.44% |
| SELL | retest2 | 2024-08-13 12:00:00 | 915.65 | 2024-08-14 13:15:00 | 929.65 | STOP_HIT | 1.00 | -1.53% |
| SELL | retest2 | 2024-08-13 12:45:00 | 915.50 | 2024-08-14 13:15:00 | 929.65 | STOP_HIT | 1.00 | -1.55% |
| BUY | retest2 | 2024-08-28 11:30:00 | 970.35 | 2024-08-29 09:15:00 | 963.30 | STOP_HIT | 1.00 | -0.73% |
| SELL | retest2 | 2024-09-06 09:30:00 | 942.65 | 2024-09-06 12:15:00 | 958.25 | STOP_HIT | 1.00 | -1.65% |
| BUY | retest2 | 2024-09-11 10:30:00 | 962.75 | 2024-09-17 12:15:00 | 1059.03 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-10-09 14:15:00 | 998.10 | 2024-10-10 09:15:00 | 1012.40 | STOP_HIT | 1.00 | -1.43% |
| SELL | retest2 | 2024-10-09 15:00:00 | 998.50 | 2024-10-10 09:15:00 | 1012.40 | STOP_HIT | 1.00 | -1.39% |
| SELL | retest2 | 2024-10-10 12:15:00 | 999.20 | 2024-10-11 10:15:00 | 1012.70 | STOP_HIT | 1.00 | -1.35% |
| SELL | retest2 | 2024-10-10 12:45:00 | 999.35 | 2024-10-11 10:15:00 | 1012.70 | STOP_HIT | 1.00 | -1.34% |
| SELL | retest2 | 2024-10-25 09:15:00 | 916.15 | 2024-10-29 15:15:00 | 918.70 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest2 | 2024-10-28 11:15:00 | 918.00 | 2024-10-29 15:15:00 | 918.70 | STOP_HIT | 1.00 | -0.08% |
| SELL | retest2 | 2024-10-28 14:00:00 | 917.60 | 2024-10-30 09:15:00 | 928.90 | STOP_HIT | 1.00 | -1.23% |
| SELL | retest2 | 2024-10-28 15:15:00 | 918.00 | 2024-10-30 09:15:00 | 928.90 | STOP_HIT | 1.00 | -1.19% |
| SELL | retest2 | 2024-10-29 11:15:00 | 901.80 | 2024-10-30 09:15:00 | 928.90 | STOP_HIT | 1.00 | -3.01% |
| SELL | retest2 | 2024-10-29 11:45:00 | 902.10 | 2024-10-30 09:15:00 | 928.90 | STOP_HIT | 1.00 | -2.97% |
| SELL | retest2 | 2024-11-21 14:15:00 | 872.30 | 2024-11-22 13:15:00 | 882.75 | STOP_HIT | 1.00 | -1.20% |
| BUY | retest2 | 2024-11-25 14:15:00 | 890.50 | 2024-11-25 14:15:00 | 877.90 | STOP_HIT | 1.00 | -1.41% |
| BUY | retest2 | 2024-11-26 10:00:00 | 890.65 | 2024-12-10 09:15:00 | 979.72 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-11-27 10:00:00 | 888.00 | 2024-12-10 09:15:00 | 976.80 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-11-27 12:45:00 | 887.75 | 2024-12-10 09:15:00 | 976.53 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-11-28 10:15:00 | 900.75 | 2024-12-10 09:15:00 | 990.83 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-11-28 13:45:00 | 902.05 | 2024-12-10 09:15:00 | 992.25 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-11-28 14:45:00 | 897.45 | 2024-12-10 09:15:00 | 987.20 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-11-29 09:45:00 | 897.10 | 2024-12-10 09:15:00 | 986.81 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-11-29 11:15:00 | 902.75 | 2024-12-10 09:15:00 | 993.03 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-11-29 12:00:00 | 899.80 | 2024-12-10 09:15:00 | 989.78 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-01-14 12:15:00 | 913.50 | 2025-01-16 09:15:00 | 925.45 | STOP_HIT | 1.00 | -1.31% |
| SELL | retest2 | 2025-01-14 14:00:00 | 911.10 | 2025-01-16 09:15:00 | 925.45 | STOP_HIT | 1.00 | -1.58% |
| SELL | retest2 | 2025-01-15 11:30:00 | 913.45 | 2025-01-16 09:15:00 | 925.45 | STOP_HIT | 1.00 | -1.31% |
| SELL | retest2 | 2025-01-15 12:00:00 | 912.90 | 2025-01-16 09:15:00 | 925.45 | STOP_HIT | 1.00 | -1.37% |
| SELL | retest2 | 2025-01-15 14:15:00 | 899.65 | 2025-01-16 09:15:00 | 925.45 | STOP_HIT | 1.00 | -2.87% |
| SELL | retest2 | 2025-01-24 13:30:00 | 893.10 | 2025-01-27 10:15:00 | 848.44 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-24 13:30:00 | 893.10 | 2025-01-28 11:15:00 | 850.60 | STOP_HIT | 0.50 | 4.76% |
| BUY | retest2 | 2025-02-07 09:15:00 | 821.60 | 2025-02-12 09:15:00 | 804.00 | STOP_HIT | 1.00 | -2.14% |
| SELL | retest2 | 2025-02-27 11:15:00 | 852.45 | 2025-03-04 10:15:00 | 864.30 | STOP_HIT | 1.00 | -1.39% |
| SELL | retest2 | 2025-02-27 12:45:00 | 852.85 | 2025-03-04 10:15:00 | 864.30 | STOP_HIT | 1.00 | -1.34% |
| SELL | retest2 | 2025-02-28 09:15:00 | 850.60 | 2025-03-04 10:15:00 | 864.30 | STOP_HIT | 1.00 | -1.61% |
| SELL | retest2 | 2025-02-28 12:00:00 | 851.45 | 2025-03-04 10:15:00 | 864.30 | STOP_HIT | 1.00 | -1.51% |
| SELL | retest2 | 2025-03-03 11:00:00 | 850.35 | 2025-03-04 10:15:00 | 864.30 | STOP_HIT | 1.00 | -1.64% |
| SELL | retest2 | 2025-03-03 11:30:00 | 848.80 | 2025-03-04 10:15:00 | 864.30 | STOP_HIT | 1.00 | -1.83% |
| SELL | retest2 | 2025-03-13 13:00:00 | 893.45 | 2025-03-18 09:15:00 | 910.55 | STOP_HIT | 1.00 | -1.91% |
| SELL | retest2 | 2025-03-17 11:45:00 | 892.85 | 2025-03-18 09:15:00 | 910.55 | STOP_HIT | 1.00 | -1.98% |
| SELL | retest2 | 2025-03-17 12:15:00 | 893.70 | 2025-03-18 09:15:00 | 910.55 | STOP_HIT | 1.00 | -1.89% |
| SELL | retest2 | 2025-03-17 13:00:00 | 893.65 | 2025-03-18 09:15:00 | 910.55 | STOP_HIT | 1.00 | -1.89% |
| BUY | retest2 | 2025-03-20 12:00:00 | 923.25 | 2025-03-21 14:15:00 | 913.50 | STOP_HIT | 1.00 | -1.06% |
| BUY | retest2 | 2025-03-21 09:30:00 | 925.80 | 2025-03-21 14:15:00 | 913.50 | STOP_HIT | 1.00 | -1.33% |
| BUY | retest2 | 2025-03-21 10:15:00 | 923.40 | 2025-03-21 14:15:00 | 913.50 | STOP_HIT | 1.00 | -1.07% |
| BUY | retest2 | 2025-03-21 14:15:00 | 924.15 | 2025-03-21 14:15:00 | 913.50 | STOP_HIT | 1.00 | -1.15% |
| BUY | retest2 | 2025-04-25 09:15:00 | 913.35 | 2025-04-25 09:15:00 | 896.10 | STOP_HIT | 1.00 | -1.89% |
| SELL | retest2 | 2025-05-02 09:15:00 | 872.05 | 2025-05-02 13:15:00 | 901.70 | STOP_HIT | 1.00 | -3.40% |
| SELL | retest2 | 2025-05-02 10:45:00 | 880.95 | 2025-05-02 13:15:00 | 901.70 | STOP_HIT | 1.00 | -2.36% |
| BUY | retest2 | 2025-05-19 13:15:00 | 975.10 | 2025-05-21 10:15:00 | 957.15 | STOP_HIT | 1.00 | -1.84% |
| BUY | retest2 | 2025-05-19 15:00:00 | 974.40 | 2025-05-21 10:15:00 | 957.15 | STOP_HIT | 1.00 | -1.77% |
| BUY | retest2 | 2025-05-20 09:15:00 | 974.05 | 2025-05-21 10:15:00 | 957.15 | STOP_HIT | 1.00 | -1.74% |
| SELL | retest2 | 2025-05-26 12:15:00 | 960.00 | 2025-05-26 12:15:00 | 964.45 | STOP_HIT | 1.00 | -0.46% |
| SELL | retest2 | 2025-06-03 15:15:00 | 944.20 | 2025-06-04 14:15:00 | 951.40 | STOP_HIT | 1.00 | -0.76% |
| SELL | retest2 | 2025-06-04 11:00:00 | 945.25 | 2025-06-04 14:15:00 | 951.40 | STOP_HIT | 1.00 | -0.65% |
| SELL | retest2 | 2025-06-04 11:30:00 | 944.35 | 2025-06-04 14:15:00 | 951.40 | STOP_HIT | 1.00 | -0.75% |
| BUY | retest2 | 2025-06-06 09:15:00 | 956.20 | 2025-06-11 14:15:00 | 956.85 | STOP_HIT | 1.00 | 0.07% |
| SELL | retest2 | 2025-06-20 12:15:00 | 891.00 | 2025-06-23 10:15:00 | 901.85 | STOP_HIT | 1.00 | -1.22% |
| BUY | retest1 | 2025-06-25 09:15:00 | 931.70 | 2025-06-25 15:15:00 | 918.65 | STOP_HIT | 1.00 | -1.40% |
| BUY | retest2 | 2025-06-26 09:15:00 | 929.60 | 2025-07-04 12:15:00 | 948.95 | STOP_HIT | 1.00 | 2.08% |
| BUY | retest2 | 2025-06-26 11:45:00 | 926.00 | 2025-07-04 12:15:00 | 948.95 | STOP_HIT | 1.00 | 2.48% |
| SELL | retest2 | 2025-07-11 09:45:00 | 939.80 | 2025-07-15 12:15:00 | 944.40 | STOP_HIT | 1.00 | -0.49% |
| SELL | retest2 | 2025-07-14 11:15:00 | 935.20 | 2025-07-15 12:15:00 | 944.40 | STOP_HIT | 1.00 | -0.98% |
| BUY | retest1 | 2025-07-23 09:45:00 | 966.75 | 2025-07-28 09:15:00 | 987.55 | STOP_HIT | 1.00 | 2.15% |
| BUY | retest1 | 2025-07-23 10:15:00 | 966.05 | 2025-07-28 09:15:00 | 987.55 | STOP_HIT | 1.00 | 2.23% |
| BUY | retest2 | 2025-08-07 15:00:00 | 1000.20 | 2025-08-08 10:15:00 | 981.40 | STOP_HIT | 1.00 | -1.88% |
| BUY | retest1 | 2025-08-13 09:15:00 | 1002.35 | 2025-08-13 14:15:00 | 995.90 | STOP_HIT | 1.00 | -0.64% |
| BUY | retest1 | 2025-08-13 13:30:00 | 1004.75 | 2025-08-13 14:15:00 | 995.90 | STOP_HIT | 1.00 | -0.88% |
| BUY | retest2 | 2025-09-16 12:15:00 | 1047.30 | 2025-09-17 13:15:00 | 1032.25 | STOP_HIT | 1.00 | -1.44% |
| BUY | retest2 | 2025-09-17 11:45:00 | 1046.65 | 2025-09-17 13:15:00 | 1032.25 | STOP_HIT | 1.00 | -1.38% |
| SELL | retest2 | 2025-09-29 11:30:00 | 1041.45 | 2025-09-30 09:15:00 | 1054.65 | STOP_HIT | 1.00 | -1.27% |
| SELL | retest2 | 2025-09-29 12:00:00 | 1038.80 | 2025-09-30 09:15:00 | 1054.65 | STOP_HIT | 1.00 | -1.53% |
| SELL | retest2 | 2025-09-29 15:15:00 | 1040.00 | 2025-09-30 09:15:00 | 1054.65 | STOP_HIT | 1.00 | -1.41% |
| BUY | retest2 | 2025-10-03 11:30:00 | 1079.00 | 2025-10-06 10:15:00 | 1047.60 | STOP_HIT | 1.00 | -2.91% |
| SELL | retest2 | 2025-10-14 10:45:00 | 1005.90 | 2025-10-16 10:15:00 | 1019.00 | STOP_HIT | 1.00 | -1.30% |
| BUY | retest2 | 2025-11-03 09:15:00 | 1076.40 | 2025-11-06 09:15:00 | 1067.20 | STOP_HIT | 1.00 | -0.85% |
| BUY | retest2 | 2025-11-03 10:00:00 | 1076.50 | 2025-11-06 09:15:00 | 1067.20 | STOP_HIT | 1.00 | -0.86% |
| BUY | retest2 | 2025-11-03 10:45:00 | 1074.40 | 2025-11-06 10:15:00 | 1062.80 | STOP_HIT | 1.00 | -1.08% |
| BUY | retest2 | 2025-11-04 09:30:00 | 1075.90 | 2025-11-06 10:15:00 | 1062.80 | STOP_HIT | 1.00 | -1.22% |
| BUY | retest2 | 2025-11-04 13:45:00 | 1077.20 | 2025-11-06 10:15:00 | 1062.80 | STOP_HIT | 1.00 | -1.34% |
| BUY | retest2 | 2025-11-04 14:15:00 | 1077.10 | 2025-11-06 10:15:00 | 1062.80 | STOP_HIT | 1.00 | -1.33% |
| BUY | retest2 | 2025-11-13 09:15:00 | 1084.00 | 2025-11-17 10:15:00 | 1078.20 | STOP_HIT | 1.00 | -0.54% |
| BUY | retest2 | 2025-11-13 12:45:00 | 1081.60 | 2025-11-17 10:15:00 | 1078.20 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest2 | 2025-11-13 14:15:00 | 1083.10 | 2025-11-17 10:15:00 | 1078.20 | STOP_HIT | 1.00 | -0.45% |
| BUY | retest2 | 2025-11-14 13:00:00 | 1083.70 | 2025-11-17 10:15:00 | 1078.20 | STOP_HIT | 1.00 | -0.51% |
| SELL | retest2 | 2025-11-18 09:15:00 | 1065.20 | 2025-11-27 09:15:00 | 1040.00 | STOP_HIT | 1.00 | 2.37% |
| SELL | retest2 | 2025-12-08 12:15:00 | 996.90 | 2025-12-10 10:15:00 | 1010.70 | STOP_HIT | 1.00 | -1.38% |
| SELL | retest2 | 2025-12-09 11:30:00 | 998.00 | 2025-12-10 10:15:00 | 1010.70 | STOP_HIT | 1.00 | -1.27% |
| SELL | retest2 | 2025-12-19 10:45:00 | 983.50 | 2025-12-22 09:15:00 | 1004.00 | STOP_HIT | 1.00 | -2.08% |
| SELL | retest2 | 2025-12-19 12:15:00 | 986.50 | 2025-12-22 09:15:00 | 1004.00 | STOP_HIT | 1.00 | -1.77% |
| SELL | retest2 | 2025-12-19 13:00:00 | 986.10 | 2025-12-22 09:15:00 | 1004.00 | STOP_HIT | 1.00 | -1.82% |
| BUY | retest2 | 2026-01-07 13:00:00 | 1077.50 | 2026-01-08 09:15:00 | 1034.40 | STOP_HIT | 1.00 | -4.00% |
| BUY | retest2 | 2026-01-07 14:45:00 | 1078.40 | 2026-01-08 09:15:00 | 1034.40 | STOP_HIT | 1.00 | -4.08% |
| SELL | retest2 | 2026-01-13 12:45:00 | 1014.30 | 2026-01-14 10:15:00 | 1039.10 | STOP_HIT | 1.00 | -2.45% |
| BUY | retest2 | 2026-01-19 12:00:00 | 1056.30 | 2026-01-21 10:15:00 | 1039.30 | STOP_HIT | 1.00 | -1.61% |
| BUY | retest2 | 2026-01-19 13:15:00 | 1055.00 | 2026-01-21 10:15:00 | 1039.30 | STOP_HIT | 1.00 | -1.49% |
| BUY | retest2 | 2026-01-20 11:45:00 | 1054.90 | 2026-01-21 10:15:00 | 1039.30 | STOP_HIT | 1.00 | -1.48% |
| BUY | retest2 | 2026-01-27 14:45:00 | 1080.00 | 2026-02-01 14:15:00 | 1100.10 | STOP_HIT | 1.00 | 1.86% |
| BUY | retest2 | 2026-02-05 11:00:00 | 1174.00 | 2026-02-13 10:15:00 | 1171.20 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest2 | 2026-02-06 10:45:00 | 1169.90 | 2026-02-13 10:15:00 | 1171.20 | STOP_HIT | 1.00 | 0.11% |
| BUY | retest2 | 2026-02-17 12:15:00 | 1212.90 | 2026-02-23 11:15:00 | 1211.20 | STOP_HIT | 1.00 | -0.14% |
| BUY | retest2 | 2026-02-17 15:15:00 | 1214.00 | 2026-02-23 11:15:00 | 1211.20 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest2 | 2026-02-27 13:15:00 | 1257.70 | 2026-02-27 14:15:00 | 1245.10 | STOP_HIT | 1.00 | -1.00% |
| BUY | retest2 | 2026-02-27 13:45:00 | 1259.50 | 2026-02-27 14:15:00 | 1245.10 | STOP_HIT | 1.00 | -1.14% |
| BUY | retest2 | 2026-03-12 10:15:00 | 1190.20 | 2026-03-13 11:15:00 | 1160.90 | STOP_HIT | 1.00 | -2.46% |
| SELL | retest2 | 2026-03-17 12:15:00 | 1145.20 | 2026-03-17 14:15:00 | 1157.30 | STOP_HIT | 1.00 | -1.06% |
| SELL | retest2 | 2026-03-18 09:15:00 | 1143.90 | 2026-03-18 11:15:00 | 1166.80 | STOP_HIT | 1.00 | -2.00% |
| SELL | retest2 | 2026-03-18 09:45:00 | 1144.80 | 2026-03-18 11:15:00 | 1166.80 | STOP_HIT | 1.00 | -1.92% |
| SELL | retest2 | 2026-03-25 11:30:00 | 1131.30 | 2026-03-25 14:15:00 | 1143.20 | STOP_HIT | 1.00 | -1.05% |
| SELL | retest2 | 2026-04-01 11:15:00 | 1146.20 | 2026-04-01 11:15:00 | 1149.80 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest2 | 2026-04-13 10:45:00 | 1216.90 | 2026-04-23 10:15:00 | 1261.30 | STOP_HIT | 1.00 | 3.65% |
| SELL | retest2 | 2026-04-24 14:45:00 | 1258.70 | 2026-04-27 09:15:00 | 1281.50 | STOP_HIT | 1.00 | -1.81% |
| BUY | retest2 | 2026-05-06 14:15:00 | 1267.00 | 2026-05-08 09:15:00 | 1241.60 | STOP_HIT | 1.00 | -2.00% |
| BUY | retest2 | 2026-05-07 09:15:00 | 1269.80 | 2026-05-08 09:15:00 | 1241.60 | STOP_HIT | 1.00 | -2.22% |
| BUY | retest2 | 2026-05-07 10:30:00 | 1269.40 | 2026-05-08 09:15:00 | 1241.60 | STOP_HIT | 1.00 | -2.19% |
| BUY | retest2 | 2026-05-07 11:45:00 | 1266.50 | 2026-05-08 09:15:00 | 1241.60 | STOP_HIT | 1.00 | -1.97% |
